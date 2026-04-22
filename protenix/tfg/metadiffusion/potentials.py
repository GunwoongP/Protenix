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


def _debug_enabled(params: dict[str, Any]) -> bool:
    """Union of env switch, per-term `debug=True`, and DEBUG logger level."""
    if params.get("debug", False):
        return True
    if _DEBUG_ENV:
        return True
    return logger.isEnabledFor(logging.DEBUG)


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
                logger.debug(
                    "[metadiff/Steering] skip (progress=%.3f outside "
                    "warmup=%.2f..cutoff=%.2f)",
                    progress, float(params.get("warmup", 0.0)),
                    float(params.get("cutoff", 1.0)),
                )
            return _zero_energy_and_grad(coords) if need_grad else _zero_energy(coords)

        cv_name = feats.get("metadiffusion_cv_name", params.get("cv", None))
        cv_kwargs = feats.get("metadiffusion_cv_kwargs", {})
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
            logger.debug(
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

        if need_grad:
            return energy, grad
        return energy


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

        cv_name = feats.get("metadiffusion_cv_name", params.get("cv", None))
        cv_kwargs = feats.get("metadiffusion_cv_kwargs", {})
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
            logger.debug(
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

        if need_grad:
            return energy, grad
        return energy
