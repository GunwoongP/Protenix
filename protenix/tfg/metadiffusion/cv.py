# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Collective-variable (CV) functions for MetaDiffusion in Protenix TFG.

Each CV exposes a minimal callable contract:

    cv_fn(coords, feats, **kwargs) -> (value, grad)

- ``coords``   : ``[..., N_atom, 3]`` (matches Protenix TFG convention).
- ``feats``    : Feature dict. Most CVs need a boolean atom mask; look up
                 ``atom_selection_mask`` (set by the factory) or fall back
                 to ``atom_pad_mask`` if present.
- returns
    value      : ``[...]`` (batch shape) — CV value per sample.
    grad       : ``[..., N_atom, 3]`` — ``d(value)/d(coords)``; zeros on
                 masked atoms.

The CVs that cannot be written in closed form differentiably use
``torch.autograd.grad`` on a local clone; we always detach the returned
gradient so the caller owns a plain tensor with no leftover graph.

Ports from user's own GPU implementations where available
(``MassiveFoldClustering_Tool/massiveclustering/core/gpu/{drmsd,tmscore}.py``)
so the semantics match what the CASP dev/QA pipeline has been using.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _resolve_atom_mask(
    feats: dict[str, Any], n_atom: int, device: torch.device
) -> torch.Tensor:
    """Return a ``[N_atom]`` bool mask from feats, or all-True fallback."""
    for k in ("atom_selection_mask", "atom_pad_mask"):
        m = feats.get(k, None)
        if m is None:
            continue
        m = m.to(device=device).to(torch.bool)
        while m.dim() > 1:
            m = m[0]
        if m.shape[0] == n_atom:
            return m
    return torch.ones(n_atom, dtype=torch.bool, device=device)


def _autograd_grad(value: torch.Tensor, coords_leaf: torch.Tensor) -> torch.Tensor:
    """Run ``torch.autograd.grad`` and NaN-guard; never raise back to caller."""
    try:
        (g,) = torch.autograd.grad(value.sum(), coords_leaf, create_graph=False)
    except RuntimeError:
        g = torch.zeros_like(coords_leaf)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g.detach()


# ═══════════════════════════════════════════════════════════════════════
# radius_of_gyration
# ═══════════════════════════════════════════════════════════════════════

def radius_of_gyration_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rg = sqrt(mean(||r_i - r_com||^2)) over masked atoms.

    Differentiable w.r.t. atom coordinates. Gradient on masked-out atoms
    is zero.
    """
    n_atom = coords.shape[-2]
    device = coords.device
    mask = _resolve_atom_mask(feats, n_atom, device).float()
    n_valid = mask.sum().clamp_min(1e-8)

    # Build a requires_grad leaf so autograd can walk through.
    coords_leaf = coords.detach().clone().requires_grad_(True)
    w = mask.view(*([1] * (coords_leaf.ndim - 2)), n_atom, 1)

    com = (coords_leaf * w).sum(dim=-2, keepdim=True) / n_valid
    centered = (coords_leaf - com) * w
    rg_sq = (centered ** 2).sum(dim=(-2, -1)) / n_valid
    rg = torch.sqrt(rg_sq + 1e-8)

    grad = _autograd_grad(rg, coords_leaf) * w
    return rg.detach(), grad


# ═══════════════════════════════════════════════════════════════════════
# distance between two groups / regions
# ═══════════════════════════════════════════════════════════════════════

def distance_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Distance between two groups' centres of mass.

    ``mask1`` / ``mask2`` : ``[N_atom]`` bool. If omitted, the factory
    should pass explicit masks; otherwise this reduces to a no-op.
    """
    n_atom = coords.shape[-2]
    device = coords.device
    if mask1 is None or mask2 is None:
        return torch.zeros(coords.shape[:-2], device=device), torch.zeros_like(coords)

    m1 = mask1.to(device=device, dtype=torch.float32)
    m2 = mask2.to(device=device, dtype=torch.float32)
    coords_leaf = coords.detach().clone().requires_grad_(True)

    w1 = m1.view(*([1] * (coords_leaf.ndim - 2)), n_atom, 1)
    w2 = m2.view(*([1] * (coords_leaf.ndim - 2)), n_atom, 1)
    com1 = (coords_leaf * w1).sum(dim=-2) / m1.sum().clamp_min(1e-8)
    com2 = (coords_leaf * w2).sum(dim=-2) / m2.sum().clamp_min(1e-8)
    dist = torch.linalg.norm(com1 - com2, dim=-1)

    grad = _autograd_grad(dist, coords_leaf)
    return dist.detach(), grad


# ═══════════════════════════════════════════════════════════════════════
# dRMSD (distance-matrix RMSD)  — pure-torch port of
# massiveclustering.core.gpu.drmsd.compute_drmsd_cpu
# ═══════════════════════════════════════════════════════════════════════

def drmsd_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    reference_coords: torch.Tensor,
    reference_mask: Optional[torch.Tensor] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """dRMSD between the masked pairwise distance matrices.

        dRMSD = sqrt(mean_ij (||r_i - r_j|| - ||r_i^* - r_j^*||)^2)

    Rigid-motion invariant by construction; no alignment needed.
    Autograd-friendly (pure elementwise + pdist-equivalent ops).

    Args:
        reference_coords : ``[N_ref, 3]`` — target structure Cα (or same
                           subset as ``reference_mask``).
        reference_mask   : ``[N_atom]`` bool over the model atoms that
                           correspond 1-to-1 with ``reference_coords``.
                           Length match is asserted.
    """
    n_atom = coords.shape[-2]
    device = coords.device
    if reference_mask is None:
        reference_mask = _resolve_atom_mask(feats, n_atom, device)
    reference_mask = reference_mask.to(device=device, dtype=torch.bool)
    assert reference_mask.sum().item() == reference_coords.shape[0], (
        f"reference_coords ({reference_coords.shape[0]}) must match mask "
        f"count ({int(reference_mask.sum().item())})"
    )

    coords_leaf = coords.detach().clone().requires_grad_(True)

    # Gather masked atoms. ``mask.nonzero`` is 1-D indices into N_atom.
    idx = reference_mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx)  # [..., K, 3]
    ref = reference_coords.to(device=device, dtype=sub.dtype)

    # Pairwise distance matrices (upper-triangular is sufficient; we
    # use the full matrix for simplicity — still O(K^2)).
    d_sub = torch.linalg.norm(sub.unsqueeze(-2) - sub.unsqueeze(-3), dim=-1)
    d_ref = torch.linalg.norm(ref.unsqueeze(-2) - ref.unsqueeze(-3), dim=-1)

    K = ref.shape[0]
    # Exclude the diagonal from the average.
    off_diag = 1.0 - torch.eye(K, device=device, dtype=d_sub.dtype)
    diff_sq = ((d_sub - d_ref) ** 2) * off_diag
    denom = (K * (K - 1)) if K > 1 else 1
    drmsd = torch.sqrt(diff_sq.sum(dim=(-2, -1)) / denom + 1e-12)

    grad = _autograd_grad(drmsd, coords_leaf)
    return drmsd.detach(), grad


# ═══════════════════════════════════════════════════════════════════════
# differentiable TM-score  — adapted from
# massiveclustering.core.gpu.tmscore.compute_tmscore_batch_iterative
# with n_iter=1 (single Kabsch + TM term) for autograd stability.
# ═══════════════════════════════════════════════════════════════════════

def _compute_d0(length: int | float) -> float:
    """Official TM-score d0 normalisation (length-aware)."""
    L = max(float(length), 1.0)
    if L <= 21:
        return 0.5
    return 1.24 * ((L - 15.0) ** (1.0 / 3.0)) - 1.8


def d_tm_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    reference_coords: torch.Tensor,
    reference_mask: Optional[torch.Tensor] = None,
    d0: Optional[float] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable TM-score on masked Cα.

    We align the model's masked subset onto ``reference_coords`` with
    one shot of Kabsch (centring + SVD) and evaluate the TM sum

        TM = (1/L) * sum_i 1 / (1 + (d_i / d0)^2)

    The iterative weight-update scheme in the original CASP clustering
    impl is dropped for autograd stability (single iteration is smooth
    while iterated reweighting introduces near-discontinuities in the
    gradient).
    """
    n_atom = coords.shape[-2]
    device = coords.device
    if reference_mask is None:
        reference_mask = _resolve_atom_mask(feats, n_atom, device)
    reference_mask = reference_mask.to(device=device, dtype=torch.bool)
    assert reference_mask.sum().item() == reference_coords.shape[0], (
        f"reference_coords ({reference_coords.shape[0]}) must match mask "
        f"count ({int(reference_mask.sum().item())})"
    )

    coords_leaf = coords.detach().clone().requires_grad_(True)
    idx = reference_mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx).to(torch.float32)  # [..., K, 3]
    ref = reference_coords.to(device=device, dtype=torch.float32)

    K = ref.shape[0]
    if d0 is None:
        d0 = _compute_d0(K)
    d0_t = torch.as_tensor(d0, device=device, dtype=sub.dtype)

    # Centre + Kabsch (single iter, no weighting → stable gradient).
    sub_c = sub - sub.mean(dim=-2, keepdim=True)
    ref_c = ref - ref.mean(dim=-2, keepdim=True)
    H = sub_c.transpose(-2, -1) @ ref_c
    U, _, Vt = torch.linalg.svd(H)
    det = torch.linalg.det(Vt.transpose(-2, -1) @ U.transpose(-2, -1))
    sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    D = torch.diag_embed(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)
    )
    R = Vt.transpose(-2, -1) @ D @ U.transpose(-2, -1)
    sub_rot = sub_c @ R.transpose(-2, -1)

    dists = torch.linalg.norm(sub_rot - ref_c, dim=-1)
    tm_terms = 1.0 / (1.0 + (dists / d0_t) ** 2)
    tm = tm_terms.sum(dim=-1) / float(K)

    grad = _autograd_grad(tm, coords_leaf)
    return tm.detach(), grad


# ═══════════════════════════════════════════════════════════════════════
# Registry of built-in CV functions (extendable from factory).
# ═══════════════════════════════════════════════════════════════════════

CV_REGISTRY = {
    "rg": radius_of_gyration_cv,
    "radius_of_gyration": radius_of_gyration_cv,
    "distance": distance_cv,
    "drmsd": drmsd_cv,
    "d_tm": d_tm_cv,
    "tm": d_tm_cv,
}


def get_cv(name: str):
    """Resolve a CV function by name."""
    if name not in CV_REGISTRY:
        raise KeyError(
            f"Unknown collective_variable '{name}'. Available: "
            f"{sorted(set(CV_REGISTRY.keys()))}"
        )
    return CV_REGISTRY[name]
