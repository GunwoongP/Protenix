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
    """
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
            "hill_height": 0.5, "hill_sigma": 2.0, "hill_interval": 5,
            "well_tempered": False, "bias_factor": 10.0, "kT": 2.5,
            "max_hills": 1000,
            "warmup": 0.0, "cutoff": 0.75, "ensemble": False,
        }
        if default_params:
            defaults.update(default_params)
        super().__init__(defaults)
        self.hills: list[dict[str, float]] = []
        self._call_counter = 0

    def reset_hills(self) -> None:
        """Clear history between independent runs."""
        self.hills.clear()
        self._call_counter = 0

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
        if len(self.hills) > max_h:
            del self.hills[: len(self.hills) - max_h]

    def _eval(
        self,
        coords: torch.Tensor,
        feats: dict[str, Any],
        params: dict[str, Any],
        need_grad: bool,
    ):
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

        hill_sigma = float(params.get("hill_sigma", 2.0))
        hill_h0 = float(params.get("hill_height", 0.5))
        hill_interval = max(1, int(params.get("hill_interval", 5)))
        well_tempered = bool(params.get("well_tempered", False))
        bias_factor = float(params.get("bias_factor", 10.0))
        kT = float(params.get("kT", 2.5))

        ensemble = bool(params.get("ensemble", False))

        cv_value, cv_grad = _invoke_cv(cv_function, coords, feats, cv_kwargs)

        # Deposit hill every `hill_interval` engine calls.
        # Hill center is always the batch-mean CV (hills are a
        # shared-state object; one hill per step covers the ensemble).
        if self._call_counter % hill_interval == 0:
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
