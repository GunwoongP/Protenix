# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""MetaDiffusion potentials — native Protenix TFG integration.

Implements:

- :class:`SteeringPotential` — pull a collective variable toward a target.
  Two shape modes are supported:

    * ``"harmonic"``  :  ``V = 0.5 * k * (CV - target)^2``
      (unbounded, matches Boltz-Metadiffusion ``HarmonicSteeringPotential``).

    * ``"gaussian"``  :  ``V = -A * exp(-(CV - target)^2 / (2 * sigma^2))``
      (bounded well; near the target this is ``≈ -A + A·(CV-t)^2/(2σ²)``
      so it reduces to a harmonic with effective ``k = A/σ²``).

- :class:`OptPotential` — push a CV toward higher or lower values
  monotonically. ``V = ± strength * CV`` (linear), with optional
  ``log_gradient`` rescale that normalises the gradient by
  ``max(|grad|, ε)`` per sample (same trick Boltz uses to keep the pull
  constant-magnitude across diffusion steps).

Both classes consume a CV factory entry (see ``schema.py``): a dict
carrying ``cv_function``, ``cv_kwargs``, a ``warmup``/``cutoff`` window,
and any mode-specific parameters. The CV function itself always returns
``(value, dvalue/dcoords)`` with the grad computed via autograd inside
``cv.py`` — the potentials compose that chain-rule gradient here.

Registration pattern matches the rest of ``protenix.tfg.potentials``:
``@register`` at class definition puts the class name into
``CLASS_REGISTRY`` so the TFG config builder can instantiate it from a
string.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import torch

from protenix.tfg.potentials import Potential, register
from protenix.tfg.metadiffusion.cv import get_cv
from protenix.tfg.metadiffusion.gradient_mods import (
    ScalingConfig,
    ProjectionConfig,
    apply_modifiers,
)

logger = logging.getLogger(__name__)

# Module-level verbosity switch for tests / ad-hoc debugging. Enable
# with `METADIFFUSION_DEBUG=1` in the env or directly set the logger's
# level. When True, every `_eval` call emits a one-line summary:
#   [metadiff/<class>/<cv>]  t=<progress>  CV=<val>  target=<t>  E=<mean>
#
# We read it per-call (cheap bool check) so test scripts can toggle at
# runtime. The legacy `Potential` base has no history slot; we keep the
# running trace in `_diagnostics_history` attribute if opted into.
_DEBUG_ENV = os.environ.get("METADIFFUSION_DEBUG", "").lower() in {"1", "true", "yes"}

# When the env switch is on we raise this module's logger to INFO so the
# per-eval summaries actually reach the console regardless of the root
# logger's default level (typically WARNING). Per-term `debug=True`
# alone does NOT flip this — that's a per-call flag for unit testing.
if _DEBUG_ENV:
    logger.setLevel(logging.INFO)


def _debug_enabled(params: dict[str, Any]) -> bool:
    """Union of env switch, per-term `debug=True`, and DEBUG logger level."""
    if params.get("debug", False):
        return True
    if _DEBUG_ENV:
        return True
    return logger.isEnabledFor(logging.DEBUG)


def _debug_log(msg: str, *args) -> None:
    """Emit at INFO when env switch is on, else at DEBUG (caller silent)."""
    if _DEBUG_ENV:
        logger.info(msg, *args)
    else:
        logger.debug(msg, *args)


def _record_diag(pot: "Potential", entry: dict[str, Any]) -> None:
    """Append a snapshot to the potential's in-memory trace (opt-in)."""
    hist = getattr(pot, "_diagnostics_history", None)
    if hist is None:
        return  # caller didn't opt in; no-op
    hist.append(entry)
    # Cap history to avoid unbounded growth in long runs.
    max_hist = getattr(pot, "_diagnostics_history_max", 10_000)
    if len(hist) > max_hist:
        del hist[: len(hist) - max_hist]


def enable_diagnostics(potential: "Potential", max_history: int = 10_000) -> list[dict[str, Any]]:
    """Opt a potential into recording per-``_eval`` snapshots.

    Usage in a test::

        pot = SteeringPotential()
        hist = enable_diagnostics(pot)
        ...run diffusion...
        print(len(hist), 'steps recorded')

    The same list is returned on each call (stored on the instance) so
    callers can append/inspect/clear as needed.
    """
    hist = getattr(potential, "_diagnostics_history", None)
    if hist is None:
        hist = []
        potential._diagnostics_history = hist
    potential._diagnostics_history_max = int(max_history)
    return hist


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _time_progress(params: dict[str, Any]) -> float:
    """Return a normalised progress in ``[0, 1]`` for warmup/cutoff gating.

    Metadiffusion conventions use ``progress = 0.0`` at the start of
    diffusion and ``progress = 1.0`` at the end. TFG's schedule already
    feeds normalised ``t`` into each term; the engine can stash it into
    ``params['_relaxation']`` or ``params['_t']``. Default to 0.5 when
    neither is available (term always active in that case).
    """
    for k in ("_relaxation", "_t", "t", "progress"):
        v = params.get(k, None)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().mean().item())
        return float(v)
    return 0.5


def _window_active(params: dict[str, Any]) -> bool:
    """True iff current progress is within ``[warmup, cutoff]``."""
    lo = float(params.get("warmup", 0.0))
    hi = float(params.get("cutoff", 1.0))
    t = _time_progress(params)
    return lo <= t <= hi


def _invoke_cv(
    cv_function: Callable,
    coords: torch.Tensor,
    feats: dict[str, Any],
    cv_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call a CV function returning ``(value, grad_wrt_coords)``.

    The CV functions in :mod:`protenix.tfg.metadiffusion.cv` always
    detach both outputs, so this is a shim that just forwards.

    CRITICAL: ``runner.inference.InferenceRunner.predict`` is decorated
    with ``@torch.no_grad()``, which causes *any* tensor created inside
    — even one we explicitly ``requires_grad_(True)`` — to be produced
    without a ``grad_fn``. That makes ``torch.autograd.grad`` inside our
    CV functions silently raise ``element 0 ... does not have a
    grad_fn``, and the guarded ``except RuntimeError`` returns zeros.
    End result: ALL CV-based guidance was being applied with zero
    force across the whole diffusion. Forcing an ``enable_grad``
    scope here restores the per-CV autograd path.
    """
    with torch.enable_grad():
        value, grad = cv_function(coords, feats, **cv_kwargs)
    return value, grad


def _apply_grad_mods(
    grad: Optional[torch.Tensor],
    coords: torch.Tensor,
    feats: dict[str, Any],
    params: dict[str, Any],
) -> Optional[torch.Tensor]:
    """Stack scaling + projection on a raw potential gradient.

    Reads pre-built ``ScalingConfig`` / ``ProjectionConfig`` lists from
    term-specific feats (stashed by ``schema.build_metadiffusion_features``
    under the ``metadiffusion_mods__<term>`` key). Silently no-op when
    neither is configured.
    """
    if grad is None:
        return grad
    term_name = params.get("_term_name")
    if not term_name:
        return grad
    mods_key = f"metadiffusion_mods__{term_name}"
    mods = feats.get(mods_key)
    if not mods:
        return grad
    scalers = mods.get("scaling") or []
    projectors = mods.get("projection") or []
    order = mods.get("modifier_order") or ["scaling", "projection"]
    if not scalers and not projectors:
        return grad
    # Modifier CVs (scaling + projection) also invoke autograd-backed
    # CV functions under the outer ``@torch.no_grad()`` inference decorator,
    # so — just like ``_invoke_cv`` — we need an explicit ``enable_grad``
    # scope here or the modifier grads come back zero and effectively
    # cancel the outer term's guidance (scalers) or no-op (projectors).
    with torch.enable_grad():
        return apply_modifiers(
            grad, coords, feats,
            progress=_time_progress(params),
            scalers=scalers, projectors=projectors, order=order,
        )


def _resolve_cv_spec(
    feats: dict[str, Any], params: dict[str, Any]
) -> tuple[Optional[str], dict[str, Any]]:
    """Pick the right CV name + kwargs for the calling term.

    Lookup precedence, for correct multi-term routing (Copilot #6):

        1. term-specific keys ``metadiffusion_{cv_name,cv_kwargs}__<term>``
           (set by schema.build_metadiffusion_features for every term).
        2. singleton legacy keys ``metadiffusion_{cv_name,cv_kwargs}``
           (only populated when exactly one term is configured).
        3. ``params['cv']`` — final fallback so that even without a
           feature-dict, a user-configured CV name still works.

    Returns ``(cv_name, cv_kwargs)``; ``cv_name`` is None iff no CV is
    configured (in which case the potential should skip).
    """
    term_name = params.get("_term_name")
    if term_name:
        cv_name = feats.get(f"metadiffusion_cv_name__{term_name}")
        cv_kwargs = feats.get(f"metadiffusion_cv_kwargs__{term_name}")
        if cv_name is not None:
            return cv_name, cv_kwargs or {}
    cv_name = feats.get("metadiffusion_cv_name")
    if cv_name is not None:
        return cv_name, feats.get("metadiffusion_cv_kwargs") or {}
    cv_name = params.get("cv")
    if cv_name is None or cv_name == "None":
        return None, {}
    return str(cv_name), {}


def _zero_energy(coords: torch.Tensor) -> torch.Tensor:
    return torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)


def _zero_energy_and_grad(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _zero_energy(coords), torch.zeros_like(coords)


# ═══════════════════════════════════════════════════════════════════════
# SteeringPotential
# ═══════════════════════════════════════════════════════════════════════

@register
class SteeringPotential(Potential):
    """Pull a collective variable toward a user-specified target value.

    Expected feats
    --------------
    Provided by the metadiffusion factory (see ``schema.py``):

    - ``metadiffusion_cv_name``     : str (e.g. ``"rg"``)
    - ``metadiffusion_cv_kwargs``   : dict of CV kwargs (mask tensors,
                                      reference coords, …)

    Params (resolved at ``_eval``)
    ------------------------------
    target        : float              — desired CV value.
    strength      : float, default 1.0 — ``k`` for harmonic, ``A`` for gaussian.
    sigma         : float, default 2.0 — Gaussian width (``mode="gaussian"`` only).
    mode          : ``"harmonic"`` | ``"gaussian"``, default ``"harmonic"``.
    warmup        : float in [0, 1]    — turn on after this fraction.
    cutoff        : float in [0, 1]    — turn off after this fraction.
    ensemble      : bool, default False — if True, energy is computed on
                                         the batch-averaged CV so all
                                         samples move coherently toward
                                         the same target.
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults: dict[str, Any] = {
            "target": 0.0,
            "strength": 1.0,
            "sigma": 2.0,
            "mode": "harmonic",
            "warmup": 0.0,
            "cutoff": 1.0,
            "ensemble": False,
        }
        if default_params:
            defaults.update(default_params)
        super().__init__(defaults)

    @staticmethod
    def _harmonic(cv: torch.Tensor, target: float, k: float) -> tuple[torch.Tensor, torch.Tensor]:
        delta = cv - target
        energy = 0.5 * k * delta.pow(2)
        dE_dCV = k * delta
        return energy, dE_dCV

    @staticmethod
    def _gaussian(
        cv: torch.Tensor, target: float, A: float, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = cv - target
        s2 = max(float(sigma) ** 2, 1e-8)
        kernel = torch.exp(-delta.pow(2) / (2.0 * s2))
        energy = -A * kernel
        # d/dCV [-A * exp(-(CV-t)^2 / 2σ^2)] = A * (CV-t)/σ² * kernel
        dE_dCV = A * delta / s2 * kernel
        return energy, dE_dCV

    def _eval(
        self,
        coords: torch.Tensor,
        feats: dict[str, Any],
        params: dict[str, Any],
        need_grad: bool,
    ):
        progress = _time_progress(params)
        if not _window_active(params):
            if _debug_enabled(params):
                _debug_log(
                    "[metadiff/Steering] skip (progress=%.3f outside "
                    "warmup=%.2f..cutoff=%.2f)",
                    progress, float(params.get("warmup", 0.0)),
                    float(params.get("cutoff", 1.0)),
                )
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)

        cv_name, cv_kwargs = _resolve_cv_spec(feats, params)
        if cv_name is None:
            if _debug_enabled(params):
                logger.warning("[metadiff/Steering] no CV set in feats/params; skipping")
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)
        cv_function = get_cv(cv_name)

        target = float(params["target"])
        strength = float(params["strength"])
        mode = str(params.get("mode", "harmonic")).lower()
        ensemble = bool(params.get("ensemble", False))

        cv_value, cv_grad = _invoke_cv(cv_function, coords, feats, cv_kwargs)
        # cv_value : [...]  (batch shape)
        # cv_grad  : [..., N_atom, 3]

        # Symmetric with OptPotential: scale-invariant log_gradient
        # rescaling. Critical for CVs whose raw gradient magnitude
        # depends on the value scale (e.g. ``sasa`` with values in
        # the 10⁴ Å² range). Without this, ``steer`` with ``sasa`` /
        # ``asphericity`` / ``coordination`` applies an enormous force
        # in the first diffusion step regardless of the configured
        # strength, blowing up the fold even at strength 0.05.
        if bool(params.get("log_gradient", False)):
            g_norm = cv_grad.norm(dim=-1, keepdim=True)
            max_norm = g_norm.flatten(start_dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
            cv_grad = cv_grad / max_norm.clamp_min(1e-8)

        if ensemble and cv_value.ndim >= 1:
            n = max(cv_value.numel(), 1)
            cv_for_loss = cv_value.mean()
            cv_grad = cv_grad / float(n)
        else:
            cv_for_loss = cv_value

        if mode == "gaussian":
            sigma = float(params.get("sigma", 2.0))
            energy, dE_dCV = self._gaussian(cv_for_loss, target, strength, sigma)
        else:  # harmonic (default)
            energy, dE_dCV = self._harmonic(cv_for_loss, target, strength)

        if ensemble and cv_value.ndim >= 1:
            energy = energy.expand(cv_value.shape)
            dE_dCV = dE_dCV.expand(cv_value.shape)

        grad = None
        if need_grad:
            broadcast_shape = dE_dCV.shape + (1, 1)
            grad = dE_dCV.reshape(broadcast_shape) * cv_grad

        # ─── Debug / diagnostics ───
        if _debug_enabled(params):
            with torch.no_grad():
                cv_mean = float(cv_value.float().mean().item())
                cv_min = float(cv_value.float().min().item()) if cv_value.numel() > 1 else cv_mean
                cv_max = float(cv_value.float().max().item()) if cv_value.numel() > 1 else cv_mean
                e_mean = float(energy.float().mean().item())
                g_norm = (
                    float(grad.norm(dim=-1).mean().item())
                    if grad is not None else float("nan")
                )
            _debug_log(
                "[metadiff/Steering/%s] t=%.3f  CV=%.4f (min=%.4f, max=%.4f) "
                "target=%.4f  mode=%s  k/A=%.3g  E_mean=%.4g  |grad|=%.4g",
                cv_name, progress, cv_mean, cv_min, cv_max,
                target, mode, strength, e_mean, g_norm,
            )
            _record_diag(self, {
                "term": "Steering", "cv": cv_name, "progress": progress,
                "cv_mean": cv_mean, "cv_min": cv_min, "cv_max": cv_max,
                "target": target, "mode": mode, "energy_mean": e_mean,
                "grad_norm": g_norm,
            })

        grad = _apply_grad_mods(grad, coords, feats, params)


        if need_grad:
            return energy, grad
        return energy


# ═══════════════════════════════════════════════════════════════════════
# MetadynamicsPotential — explore mode (Gaussian hills)
# ═══════════════════════════════════════════════════════════════════════

@register
class MetadynamicsPotential(Potential):
    """Enhanced sampling via Gaussian hill deposition on a CV.

        V(s) = Σ_i  h_i · exp(-(s - s_i)^2 / (2 σ^2))

    Each call to ``_eval`` increments an internal counter; every
    ``hill_interval`` calls we deposit a new hill at the current CV
    value. The accumulated bias then drives the sampler AWAY from
    previously visited CV regions, which is the metadynamics trick.

    Well-tempered variant (enabled by ``well_tempered=True``):

        h_i = h_0 · exp(-V(s_{i-1}) / (kT · (γ - 1)))

    so the bias deposition rate decays as the surface fills up and the
    free-energy estimate converges.

    Params (resolved at ``_eval``)
    ------------------------------
    hill_height   : float, default 0.5  (base Gaussian height)
    hill_sigma    : float, default 2.0  (Gaussian width in CV units)
    hill_interval : int,   default 5    (deposit every N engine calls)
    well_tempered : bool,  default False
    bias_factor   : float, default 10.0 (γ, for well-tempered)
    kT            : float, default 2.5  (thermal scale; CASP-tuned)
    max_hills     : int,   default 1000 (cap stored hills, FIFO prune)
    warmup, cutoff, ensemble : same semantics as SteeringPotential.
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults: dict[str, Any] = {
            # Mode: "hills" (metadynamics with deposited Gaussians) or
            # "repulsion" (stateless pairwise Gaussian kernel on per-
            # sample CV values; Boltz explore_type="repulsion").
            "explore_type": "hills",
            "hill_height": 0.5, "hill_sigma": 2.0, "hill_interval": 5,
            "well_tempered": False, "bias_factor": 10.0, "kT": 2.5,
            "max_hills": 1000,
            "warmup": 0.0, "cutoff": 0.75, "ensemble": False,
            # Repulsion mode knobs (Boltz-equivalent):
            "repulsion_strength": 1.0,
            "repulsion_sigma": None,   # default: fall back to hill_sigma
        }
        if default_params:
            defaults.update(default_params)
        super().__init__(defaults)
        self.hills: list[dict[str, float]] = []
        self._call_counter = 0
        self._hills_path: Optional[str] = None
        self._hills_loaded_from_disk = False
        # Index of the first NOT-YET-PERSISTED hill; save_hills only writes
        # this tail so we don't re-emit (and duplicate) hills that already
        # live on disk. Reset on every load or reset_hills.
        self._hills_persisted_count = 0

    def reset_hills(self) -> None:
        """Clear history between independent runs.

        Also resets the on-disk persistence state so that reusing the
        same ``MetadynamicsPotential`` instance for a second run
        (e.g. a test that calls ``reset_hills()`` between scenarios)
        starts with an empty ledger and re-loads any configured
        ``hills_path`` on the next ``_eval`` — without this the
        atexit hook would still point at the first run's JSON
        (Codex PR#3 round 4).
        """
        self.hills.clear()
        self._call_counter = 0
        self._hills_persisted_count = 0
        self._hills_path = None
        self._hills_loaded_from_disk = False

    # ─────────────────────────────────────────────────────────────────
    # Cross-run persistence (true "avoid previously-visited" semantics)
    # ─────────────────────────────────────────────────────────────────
    def save_hills(self, path: str) -> None:
        """Serialise the accumulated hill list to JSON.

        Hills are tiny — each entry is a dict of four floats — so JSON
        is fast, portable, and diffable. Call this at the end of a
        run, then ``load_hills(path)`` at the start of the next run to
        continue pushing the sampler away from previously-visited CV
        regions. This is the cross-run analogue of within-run
        metadynamics: across *different* Protenix invocations, each
        new batch feels the cumulative bias of every batch before it.

        Concurrency (Codex PR#2 P2 follow-up): the whole read-merge-
        replace critical section is held under a single exclusive
        ``flock`` on a sibling ``<path>.lock`` file, so two concurrent
        processes never interleave reads/writes. We *don't* deduplicate
        hills — metadynamics is explicitly additive: two visits to the
        same CV value are two separate hills that linearly stack.
        Instead we track how many of our in-memory hills were already
        persisted (``self._hills_persisted_count``) and on each save
        append only the tail, merging them with whatever appeared on
        disk from peer processes. Falls back to the previous behaviour
        silently when ``fcntl`` is unavailable.
        """
        import json, os, tempfile
        try:
            import fcntl
            have_lock = True
        except ImportError:
            have_lock = False

        lock_path = str(path) + ".lock"
        lock_fd = None
        if have_lock:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        try:
            # 1) Read whatever peer processes left on disk.
            on_disk: list[dict[str, float]] = []
            on_disk_counter = 0
            if os.path.isfile(path):
                try:
                    with open(path, "r") as rf:
                        blob = json.load(rf)
                    on_disk = list(blob.get("hills", []))
                    on_disk_counter = int(blob.get("call_counter", 0))
                except Exception:
                    on_disk = []
                    on_disk_counter = 0

            # 2) Figure out which of OUR in-memory hills are new since the
            # last ``save_hills`` — those are the tail beyond
            # ``_hills_persisted_count``. Everything before that index
            # came from the previous load() or previous save(), so it
            # (and any additions from peer processes) already lives on
            # disk.
            persisted_so_far = int(getattr(self, "_hills_persisted_count", 0))
            our_new = self.hills[persisted_so_far:]

            # 3) Union = on-disk contents (which already includes any
            # peer additions + our previously-saved hills) + our NEW
            # hills. No dedup by CV triple — metadynamics hills are
            # additive and legitimate duplicates must survive.
            merged_hills = on_disk + list(our_new)
            merged_counter = max(int(self._call_counter), on_disk_counter)

            # Re-apply the FIFO cap on disk too (Codex PR#3 round 4):
            # deposit_hill() already prunes self.hills to max_hills in
            # memory, but without this on-disk grows unbounded across
            # save/reload cycles, and ``load_hills`` would resurrect
            # pruned entries on the next run — blowing past the
            # configured cap.
            max_h_cap = getattr(self, "_max_hills_cap", None)
            if max_h_cap is None:
                max_h_cap = int((self.defaults or {}).get("max_hills", 1000))
            if len(merged_hills) > max_h_cap:
                merged_hills = merged_hills[-max_h_cap:]

            # 4) Atomic replace.
            d = os.path.dirname(os.path.abspath(path)) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".hills.", suffix=".json", dir=d)
            try:
                with os.fdopen(fd, "w") as wf:
                    json.dump({"hills": merged_hills,
                               "call_counter": merged_counter}, wf)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # 5) Now that our in-memory hills are safely on disk, update
            # the watermark. (If peer hills were prepended we don't care
            # — their contents are in merged_hills and our "new tail"
            # count was computed before that union.)
            self._hills_persisted_count = len(self.hills)

        finally:
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def _save_on_exit(self) -> None:
        """atexit hook — persist hills before the Protenix process dies."""
        if not self._hills_path or not self.hills:
            return
        try:
            self.save_hills(self._hills_path)
            logger.info(
                "[metadiff/Metad] saved %d hills to %s (exit hook)",
                len(self.hills), self._hills_path,
            )
        except Exception as e:
            logger.warning(
                "[metadiff/Metad] save-on-exit failed: %s", e,
            )

    def load_hills(self, path: str, append: bool = True) -> None:
        """Restore hills from a prior ``save_hills`` dump.

        By default the loaded hills are *appended* to any existing
        ones (so chained runs accumulate), but passing ``append=False``
        resets to exactly the saved set.
        """
        import json, os
        if not os.path.isfile(path):
            return
        with open(path) as f:
            blob = json.load(f)
        new_hills = list(blob.get("hills", []))
        if append:
            self.hills.extend(new_hills)
        else:
            self.hills = new_hills
        # Carry over the call counter so hill_interval gating keeps
        # advancing; a fresh run without this would re-deposit at
        # step 1 and clobber the imported timeline.
        self._call_counter = int(blob.get("call_counter", 0))
        # Everything currently in ``self.hills`` is already on disk, so
        # the next save_hills() starts appending from this point onward.
        self._hills_persisted_count = len(self.hills)

    def _hill_bias(
        self, cv: torch.Tensor, sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum of Gaussians: V(cv) = Σ h_i exp(-(cv - s_i)^2 / 2σ^2).

        Returns ``(V, dV/dcv)`` both with ``cv``'s shape. If no hills
        yet, returns zeros.
        """
        if not self.hills:
            return torch.zeros_like(cv), torch.zeros_like(cv)
        V = torch.zeros_like(cv)
        dV_dCV = torch.zeros_like(cv)
        s2_inv = 1.0 / (sigma * sigma + 1e-12)
        for h in self.hills:
            s_i = float(h["cv_center"])
            A = float(h["height"])
            delta = cv - s_i
            kernel = torch.exp(-0.5 * delta * delta * s2_inv)
            V = V + A * kernel
            dV_dCV = dV_dCV + (-A * delta * s2_inv) * kernel
        return V, dV_dCV

    def _well_tempered_height(
        self, h_0: float, cv_scalar: float, kT: float, bias_factor: float
    ) -> float:
        """h_i = h_0 · exp(-V(cv) / (kT · (γ-1)))."""
        if bias_factor <= 1.0:
            return h_0
        dT = kT * (bias_factor - 1.0)
        # Evaluate V(cv) with current hills as pure-Python scalars.
        import math as _math
        V_here = 0.0
        for h in self.hills:
            delta = cv_scalar - float(h["cv_center"])
            V_here += float(h["height"]) * _math.exp(
                -0.5 * delta * delta / (float(h["sigma"]) ** 2 + 1e-12)
            )
        return h_0 * _math.exp(-V_here / (dT + 1e-8))

    def deposit_hill(
        self, cv_value_scalar: float, step_idx: int,
        height: float, sigma: float, max_hills: int = 1000,
    ) -> None:
        """Add one hill, prune FIFO to ``max_hills``.

        ``max_hills`` is passed in from the resolved params at each
        ``_eval`` (Copilot #2): a runtime override via the schema or a
        param schedule now takes effect, instead of being frozen at
        constructor defaults.
        """
        self.hills.append({
            "cv_center": float(cv_value_scalar),
            "height": float(height),
            "sigma": float(sigma),
            "step": int(step_idx),
        })
        max_h = max(1, int(max_hills))
        # Remember the most recent cap so save_hills can honour it too
        # (otherwise the on-disk JSON grows past the in-memory limit
        # across save/reload cycles — see Codex PR#3 round 4 item 3).
        self._max_hills_cap = max_h
        if len(self.hills) > max_h:
            n_pruned = len(self.hills) - max_h
            del self.hills[:n_pruned]
            # Shift the persisted-watermark to match (Codex PR#2 round 2):
            # if the prune drops entries we had previously persisted,
            # ``_hills_persisted_count`` must be lowered by the same amount
            # so ``self.hills[persisted_so_far:]`` still refers to the
            # NEW (not-yet-saved) tail instead of going empty.
            self._hills_persisted_count = max(
                0, getattr(self, "_hills_persisted_count", 0) - n_pruned,
            )

    def _eval(
        self,
        coords: torch.Tensor,
        feats: dict[str, Any],
        params: dict[str, Any],
        need_grad: bool,
    ):
        # Lazy-load hills from disk on the very first eval, if a path
        # was configured in the YAML. This keeps the cross-run
        # persistence flow simple — the user drops the same
        # `hills_path` into every run's explore body and the bias
        # surface accumulates across invocations.
        if not self._hills_loaded_from_disk:
            hp = params.get("hills_path")
            if hp:
                self._hills_path = str(hp)
                try:
                    self.load_hills(self._hills_path, append=True)
                    _debug_log(
                        "[metadiff/Metad] loaded %d hills from %s",
                        len(self.hills), self._hills_path,
                    )
                except Exception as e:
                    logger.warning(
                        "[metadiff/Metad] could not load hills from %s "
                        "(%s); starting with empty history",
                        self._hills_path, e,
                    )
            self._hills_loaded_from_disk = True
            # Register a save-on-exit hook. Each run exits clean, so
            # atexit is the reliable place to persist state.
            if self._hills_path:
                import atexit
                atexit.register(self._save_on_exit)

        # CRITICAL (Copilot #2): MetadynamicsPotential is stateful.
        # TFG invokes potentials on BOTH the energy-only path (MC logp
        # estimate in `_logp_x0`) and the energy-and-grad path (actual
        # x0 refinement). If we counted calls and deposited hills on
        # every invocation, a single diffusion step would advance
        # `_call_counter` by ``1 + eps_batch`` and pollute the bias
        # surface with duplicate hills. The fix: only mutate state
        # when ``need_grad=True`` — i.e. when the engine is about to
        # actually apply our gradient to coordinates.
        if need_grad:
            self._call_counter += 1

        if not _window_active(params):
            if _debug_enabled(params):
                _debug_log(
                    "[metadiff/Metad] skip (t=%.3f outside window)",
                    _time_progress(params),
                )
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)

        cv_name, cv_kwargs = _resolve_cv_spec(feats, params)
        if cv_name is None:
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)
        cv_function = get_cv(cv_name)

        explore_type = str(params.get("explore_type", "hills")).lower()
        hill_sigma = float(params.get("hill_sigma", 2.0))
        hill_h0 = float(params.get("hill_height", 0.5))
        hill_interval = max(1, int(params.get("hill_interval", 5)))
        well_tempered = bool(params.get("well_tempered", False))
        bias_factor = float(params.get("bias_factor", 10.0))
        kT = float(params.get("kT", 2.5))

        ensemble = bool(params.get("ensemble", False))

        cv_value, cv_grad = _invoke_cv(cv_function, coords, feats, cv_kwargs)

        # ── Repulsion branch (Boltz explore_type="repulsion") ──
        # Stateless pairwise Gaussian on per-sample CV values:
        #     E_i = (1/(N-1)) · Σ_{j≠i} strength · exp(-(s_i-s_j)²/(2σ²))
        # Minimising this via OptPotential semantics pushes samples
        # with similar CV values apart — "안 나왔던 값"으로 유도.
        if explore_type == "repulsion":
            if cv_value.ndim < 1 or cv_value.shape[0] < 2:
                # Single sample — no pair to repel.
                return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)
            strength = float(params.get("repulsion_strength",
                                        params.get("strength", 1.0)))
            sigma = float(params.get("repulsion_sigma") or hill_sigma)
            s2_inv = 1.0 / (sigma * sigma + 1e-12)
            B = cv_value.shape[0]

            # [B, B] pairwise kernel on CV values; diag = 1 (self-pairs).
            # Convention: ``diff[i, j] = cv[j] - cv[i]`` (the outer-subtract
            # above evaluates to ``cv_value.unsqueeze(0)[i,j] - cv_value.unsqueeze(1)[i,j]``
            # = ``cv[j] - cv[i]``). Keep this in mind when reading the
            # gradient formula below.
            diff = cv_value.unsqueeze(0) - cv_value.unsqueeze(1)   # [B, B]  diff[i,j]=s_j-s_i
            kernel = torch.exp(-0.5 * diff * diff * s2_inv)
            eye = torch.eye(B, device=diff.device, dtype=diff.dtype)
            kernel_off = kernel * (1.0 - eye)

            # Per-sample repulsion energy (mean over pairs → count-invariant).
            energy = strength * kernel_off.sum(dim=-1) / float(B - 1)

            if need_grad:
                # dE_i/dCV_i = strength/(N-1) · Σ_{j≠i} d/ds_i exp(-(s_i-s_j)²/2σ²)
                #            = strength/(N-1) · Σ_{j≠i} kernel_ij · -(s_i-s_j)/σ²
                #            = strength/(N-1) · Σ_{j≠i} kernel_ij ·  (s_j-s_i)/σ²
                #            = strength/(N-1) · Σ_{j≠i} kernel_ij ·  diff[i,j]/σ²
                # NB: earlier versions had an extra minus which inverted the
                # repulsion into attraction (Codex review P1).
                dE_dCV = strength * (diff * s2_inv * kernel_off).sum(dim=-1) / float(B - 1)
                broadcast_shape = dE_dCV.shape + (1, 1)
                grad = dE_dCV.reshape(broadcast_shape) * cv_grad
            else:
                grad = None

            if _debug_enabled(params):
                _debug_log(
                    "[metadiff/Metad-rep/%s] t=%.3f  CVs=%s  E_mean=%.4g",
                    cv_name, _time_progress(params),
                    [f"{v:.3f}" for v in cv_value.tolist()],
                    float(energy.mean().item()),
                )

            if need_grad:
                grad = _apply_grad_mods(grad, coords, feats, params)
                return energy, grad
            return energy

        # Otherwise: hills mode (default, existing behaviour follows)

        # Deposit hill every `hill_interval` engine calls.
        # Hill center is always the batch-mean CV (hills are a
        # shared-state object; one hill per step covers the ensemble).
        # Deposition is also gated on `need_grad` so energy-only
        # evaluations don't pollute the bias surface (Copilot #2).
        if need_grad and self._call_counter % hill_interval == 0:
            cv_for_hill = float(cv_value.float().mean().item())
            if well_tempered:
                h_i = self._well_tempered_height(hill_h0, cv_for_hill, kT, bias_factor)
            else:
                h_i = hill_h0
            max_hills = int(params.get("max_hills", 1000))
            self.deposit_hill(
                cv_for_hill, self._call_counter, h_i, hill_sigma,
                max_hills=max_hills,
            )

        # Ensemble mode (Copilot #14): SteeringPotential-style semantics
        # so the API is consistent across modes.
        #   - ensemble=False (default): V / dV evaluated on per-sample CV
        #     → each sample feels an independent push away from the hill
        #       centers. Natural for free-energy estimation.
        #   - ensemble=True: V evaluated on the batch-mean CV, grad split
        #     equally over the batch (factor 1/N) so every sample gets
        #     the same coherent update. Matches SteeringPotential.
        if ensemble and cv_value.ndim >= 1:
            n = max(cv_value.numel(), 1)
            cv_for_loss = cv_value.mean()
            cv_grad = cv_grad / float(n)
        else:
            cv_for_loss = cv_value

        # Compute bias energy + dV/dCV.
        V, dV_dCV = self._hill_bias(cv_for_loss, hill_sigma)

        if ensemble and cv_value.ndim >= 1:
            V = V.expand(cv_value.shape)
            dV_dCV = dV_dCV.expand(cv_value.shape)

        grad = None
        if need_grad:
            broadcast_shape = dV_dCV.shape + (1, 1)
            grad = dV_dCV.reshape(broadcast_shape) * cv_grad

        if _debug_enabled(params):
            _debug_log(
                "[metadiff/Metad/%s] t=%.3f  CV=%.4f  hills=%d  V_mean=%.4g",
                cv_name, _time_progress(params),
                float(cv_value.float().mean().item()),
                len(self.hills),
                float(V.float().mean().item()),
            )
            _record_diag(self, {
                "term": "Metadynamics", "cv": cv_name,
                "progress": _time_progress(params),
                "cv_mean": float(cv_value.float().mean().item()),
                "hills": len(self.hills),
                "V_mean": float(V.float().mean().item()),
            })

        grad = _apply_grad_mods(grad, coords, feats, params)


        if need_grad:
            return V, grad
        return V


# ═══════════════════════════════════════════════════════════════════════
# OptPotential
# ═══════════════════════════════════════════════════════════════════════

@register
class OptPotential(Potential):
    """Minimise or maximise a collective variable monotonically.

    ``V = direction * strength * CV``  →  ``dV/dr = direction * strength * dCV/dr``

    - ``direction = +1`` : push CV lower  (``opt: {target: "min", ...}``)
    - ``direction = -1`` : push CV higher (``opt: {target: "max", ...}``)

    Optional ``log_gradient=True`` rescales the per-sample gradient to
    unit max-norm (keeps pull magnitude steady across diffusion steps,
    same trick as Boltz).
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults: dict[str, Any] = {
            "direction": 1.0,
            "strength": 1.0,
            "warmup": 0.0,
            "cutoff": 1.0,
            "log_gradient": False,
        }
        if default_params:
            defaults.update(default_params)
        super().__init__(defaults)

    def _eval(
        self,
        coords: torch.Tensor,
        feats: dict[str, Any],
        params: dict[str, Any],
        need_grad: bool,
    ):
        if not _window_active(params):
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)

        cv_name, cv_kwargs = _resolve_cv_spec(feats, params)
        if cv_name is None:
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)
        cv_function = get_cv(cv_name)

        direction = float(params.get("direction", 1.0))
        strength = float(params["strength"])

        cv_value, cv_grad = _invoke_cv(cv_function, coords, feats, cv_kwargs)
        coef = direction * strength
        energy = coef * cv_value

        if bool(params.get("log_gradient", False)):
            g_norm = cv_grad.norm(dim=-1, keepdim=True)
            max_norm = g_norm.flatten(start_dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
            cv_grad = cv_grad / max_norm.clamp_min(1e-8)

        grad = None
        if need_grad:
            dE_dCV = (
                torch.full_like(cv_value, coef) if cv_value.ndim
                else torch.tensor(coef, device=coords.device, dtype=coords.dtype)
            )
            broadcast_shape = dE_dCV.shape + (1, 1) if dE_dCV.ndim else (1, 1)
            grad = dE_dCV.reshape(broadcast_shape) * cv_grad

        # ─── Debug / diagnostics ───
        if _debug_enabled(params):
            with torch.no_grad():
                cv_mean = float(cv_value.float().mean().item())
                e_mean = float(energy.float().mean().item())
                g_norm_v = (
                    float(grad.norm(dim=-1).mean().item())
                    if grad is not None else float("nan")
                )
            _debug_log(
                "[metadiff/Opt/%s] t=%.3f  CV=%.4f  direction=%+.1f  "
                "strength=%.3g  E_mean=%.4g  |grad|=%.4g",
                cv_name, _time_progress(params), cv_mean,
                direction, strength, e_mean, g_norm_v,
            )
            _record_diag(self, {
                "term": "Opt", "cv": cv_name,
                "progress": _time_progress(params),
                "cv_mean": cv_mean, "direction": direction,
                "strength": strength, "energy_mean": e_mean,
                "grad_norm": g_norm_v,
            })

        grad = _apply_grad_mods(grad, coords, feats, params)


        if need_grad:
            return energy, grad
        return energy
