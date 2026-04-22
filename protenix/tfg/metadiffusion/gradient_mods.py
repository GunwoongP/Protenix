# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Gradient post-processing for MetaDiffusion — Boltz Phase D.

Two families of modifiers re-shape the raw ``dE/dr`` gradient that a
SteeringPotential / OptPotential / MetadynamicsPotential emits BEFORE
the TFG engine applies it to the diffusion coordinates:

- :class:`GradientScaler` — per-atom weight derived from another CV's
  gradient magnitude. Atoms that contribute more (or less, with
  ``strength < 0``) to the scaling CV get a proportionally larger pull
  from the primary potential.

- :class:`GradientProjector` — projects the raw gradient onto the CV's
  gradient direction, either to amplify (``direction='max'``) or kill
  (``direction='min'``) the component along that CV. Useful for "move
  toward a target but preserve another CV".

Both modifiers are built from :class:`ScalingConfig` / :class:`ProjectionConfig`
entries in a steer/opt/explore body, which the schema parser picks up
and attaches to the term. Multiple scalers / projectors stack in the
``modifier_order`` the user specifies (default: ``['scaling', 'projection']``
— scaling first, then projection).

This is a *simplified* port of Boltz's ``gradient_scaler.py`` (~760
lines). We keep the CASP-relevant behaviour: per-atom weight from any
of the CVs registered in :mod:`protenix.tfg.metadiffusion.cv`,
multiplicative stacking of multiple scalers, warmup/cutoff window,
and logging of per-atom weight distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from protenix.tfg.metadiffusion.cv import get_cv

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Config dataclasses (deserialised straight from the metadiffusion body)
# ───────────────────────────────────────────────────────────────────────

@dataclass
class ScalingConfig:
    """Single scaling CV entry.

    The scaling CV's per-atom gradient magnitude ``|dCV/dr_i|`` becomes
    the per-atom weight ``w_i``. Positive ``strength`` → atoms that
    contribute MORE to the scaling CV get a LARGER pull (highlight).
    Negative ``strength`` → INVERT: atoms that contribute MORE get
    LESS pull (suppress).
    """
    collective_variable: str
    strength: float = 1.0
    warmup: float = 0.0
    cutoff: float = 1.0
    # Optional per-scaler kwargs (masks, references). If not provided,
    # resolved at feature-build time from the CV spec keys.
    cv_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionConfig:
    """Projection onto a CV's gradient direction."""
    collective_variable: str
    strength: float = 1.0
    direction: str = "min"        # "min" or "max"
    zero_threshold: float = 1e-8  # atoms with |grad| below this are zeroed
    warmup: float = 0.0
    cutoff: float = 1.0
    cv_kwargs: dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────
# Core modifiers
# ───────────────────────────────────────────────────────────────────────

def _time_in_window(progress: float, lo: float, hi: float) -> bool:
    return float(lo) <= float(progress) <= float(hi)


class GradientScaler:
    """CV-based per-atom gradient re-weighting.

    Call :meth:`apply` with the raw gradient; returns a new gradient
    of the same shape with per-atom magnitudes scaled (but overall
    norm roughly preserved via mean-normalisation).
    """

    def __init__(self, configs: list[ScalingConfig]) -> None:
        self.configs = list(configs)

    def apply(
        self,
        grad: torch.Tensor,
        coords: torch.Tensor,
        feats: dict[str, Any],
        progress: float,
    ) -> torch.Tensor:
        """Scale ``grad`` by ∏ over scaling CVs, then normalise.

        Steps per scaler:
            1. Skip if outside ``[warmup, cutoff]``.
            2. Evaluate scaling CV's gradient magnitude per atom.
            3. Convert to weight w_i = ``|dCV/dr_i|`` (or its inverse
               when ``strength < 0``).
            4. Multiply into a running per-atom weight tensor.

        After all scalers:
            5. Normalise weights so ``mean(w) = 1`` (preserves total
               gradient magnitude).
            6. Apply per-atom to ``grad``.
        """
        if not self.configs:
            return grad
        device = grad.device
        N = grad.shape[-2]
        weights = torch.ones(*grad.shape[:-1], device=device, dtype=grad.dtype)

        any_applied = False
        for cfg in self.configs:
            if not _time_in_window(progress, cfg.warmup, cfg.cutoff):
                continue
            cv_fn = get_cv(cfg.collective_variable)
            try:
                _, cv_grad = cv_fn(coords, feats, **cfg.cv_kwargs)
            except Exception as e:
                logger.warning(
                    "[metadiff/scaler] CV '%s' evaluation failed "
                    "(%s). Skipping this scaler.",
                    cfg.collective_variable, type(e).__name__,
                )
                continue
            w_atom = cv_grad.norm(dim=-1)                # [..., N]
            if float(cfg.strength) < 0:
                # Invert: atoms that contribute MORE to scaling CV get
                # SMALLER weight. Add epsilon to avoid div-by-zero.
                w_atom = 1.0 / (w_atom + 1e-8)
            weights = weights * (w_atom ** abs(float(cfg.strength)))
            any_applied = True

        if not any_applied:
            return grad

        # Mean-normalise so sum of weights == N (preserves total magnitude).
        mean_w = weights.mean(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights / mean_w

        return grad * weights.unsqueeze(-1)


class GradientProjector:
    """Project raw gradient onto (or away from) a CV's gradient direction.

    - ``direction == 'max'`` amplifies the component of ``grad`` that
      aligns with ``dCV/dr`` (equivalent to blending in extra motion
      along CV's increasing direction).
    - ``direction == 'min'`` kills that component, so the primary
      potential can steer while leaving this CV unchanged.
    """

    def __init__(self, configs: list[ProjectionConfig]) -> None:
        self.configs = list(configs)

    def apply(
        self,
        grad: torch.Tensor,
        coords: torch.Tensor,
        feats: dict[str, Any],
        progress: float,
    ) -> torch.Tensor:
        if not self.configs:
            return grad
        out = grad
        for cfg in self.configs:
            if not _time_in_window(progress, cfg.warmup, cfg.cutoff):
                continue
            cv_fn = get_cv(cfg.collective_variable)
            try:
                _, cv_grad = cv_fn(coords, feats, **cfg.cv_kwargs)
            except Exception as e:
                logger.warning(
                    "[metadiff/projector] CV '%s' evaluation failed "
                    "(%s). Skipping.",
                    cfg.collective_variable, type(e).__name__,
                )
                continue

            # Flatten per-sample atomic vectors → [..., N*3] for dot product.
            grad_flat = out.reshape(*out.shape[:-2], -1)
            cv_flat = cv_grad.reshape(*cv_grad.shape[:-2], -1)
            cv_norm_sq = (cv_flat * cv_flat).sum(dim=-1, keepdim=True).clamp_min(float(cfg.zero_threshold) ** 2)
            dot = (grad_flat * cv_flat).sum(dim=-1, keepdim=True)
            # Projection of grad onto cv direction: (g·v/|v|²) · v
            projection = (dot / cv_norm_sq) * cv_flat

            if cfg.direction.lower() == "min":
                # Remove the CV-aligned component so CV stays constant.
                out_flat = grad_flat - float(cfg.strength) * projection
            elif cfg.direction.lower() == "max":
                # Amplify the CV-aligned component.
                out_flat = grad_flat + float(cfg.strength) * projection
            else:
                raise ValueError(
                    f"GradientProjector.direction must be 'min' or 'max', "
                    f"got {cfg.direction!r}."
                )
            out = out_flat.reshape_as(out)
        return out


# ───────────────────────────────────────────────────────────────────────
# Factory: parse a metadiffusion body's scaling/projection entries
# ───────────────────────────────────────────────────────────────────────

def parse_scaling(
    body: dict[str, Any], resolve_cv_kwargs: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[ScalingConfig]:
    """Return a list of :class:`ScalingConfig`. ``resolve_cv_kwargs``
    builds CV-specific kwargs (atom masks, references) from a Boltz
    sub-body in the same way the main parser does it for the primary CV.
    """
    raw = body.get("scaling")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cfg = ScalingConfig(
            collective_variable=str(entry["collective_variable"]),
            strength=float(entry.get("strength", 1.0)),
            warmup=float(entry.get("warmup", 0.0)),
            cutoff=float(entry.get("cutoff", 1.0)),
            cv_kwargs=resolve_cv_kwargs(entry),
        )
        out.append(cfg)
    return out


def parse_projection(
    body: dict[str, Any], resolve_cv_kwargs: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[ProjectionConfig]:
    """Return a list of :class:`ProjectionConfig`. A single dict body is
    tolerated as a one-element list (Boltz examples commonly do this)."""
    raw = body.get("projection")
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        out.append(ProjectionConfig(
            collective_variable=str(entry["collective_variable"]),
            strength=float(entry.get("strength", 1.0)),
            direction=str(entry.get("direction", "min")).lower(),
            zero_threshold=float(entry.get("zero_threshold", 1e-8)),
            warmup=float(entry.get("warmup", 0.0)),
            cutoff=float(entry.get("cutoff", 1.0)),
            cv_kwargs=resolve_cv_kwargs(entry),
        ))
    return out


def apply_modifiers(
    grad: torch.Tensor,
    coords: torch.Tensor,
    feats: dict[str, Any],
    progress: float,
    scalers: list[ScalingConfig],
    projectors: list[ProjectionConfig],
    order: list[str],
) -> torch.Tensor:
    """Apply scalers and projectors in the user-specified order.

    ``order`` is a list of strings, e.g. ``["scaling", "projection"]``
    (default). Unknown order entries are silently skipped.
    """
    scaler = GradientScaler(scalers) if scalers else None
    projector = GradientProjector(projectors) if projectors else None
    out = grad
    for kind in order:
        k = kind.lower()
        if k == "scaling" and scaler is not None:
            out = scaler.apply(out, coords, feats, progress)
        elif k == "projection" and projector is not None:
            out = projector.apply(out, coords, feats, progress)
    return out
