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


# ═══════════════════════════════════════════════════════════════════════
# pairwise dRMSD between samples in a batch — the "diversity" CV.
# When its value is MAXIMISED (via OptPotential direction=-1), the
# sampler is pushed to generate more structurally distinct conformers.
# ═══════════════════════════════════════════════════════════════════════

def pair_drmsd_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mean pairwise dRMSD across the batch dimension.

    For a coords tensor of shape ``[B, N, 3]`` (B samples, N atoms) we
    first build each sample's Cα distance matrix, then for every pair
    (i, j) with i < j compute::

        pair(i, j) = sqrt(mean_{a, b} (D_i - D_j)^2)

    and finally the *per-sample* CV::

        CV_k = (1 / (B - 1)) * sum_{j != k} pair(k, j)

    So the returned value has shape ``[B]``, with each entry telling
    sample k how different it is from the other B-1 samples on average.

    Why per-sample matters
    ----------------------
    When used as an OptPotential target (``direction = -1`` pushes the
    CV up), each sample receives an *independent* gradient that pushes
    it AWAY from the mean of the others. That is what drives diversity.
    A scalar (batch-mean) CV would emit identical gradients to every
    sample and fail to break symmetry — which is exactly what we saw in
    our first implementation.

    Autograd handles the cross-sample coupling implicitly: because
    pair(i, j) depends on both coords[i] and coords[j], each CV_k's
    gradient lands on *every* sample coord (not just coords[k]).
    """
    if coords.ndim < 3 or coords.shape[0] < 2:
        # No pair to compute (single sample or missing batch dim).
        return (
            torch.zeros(coords.shape[:-2], device=coords.device),
            torch.zeros_like(coords),
        )

    n_atom = coords.shape[-2]
    B = coords.shape[0]
    mask = _resolve_atom_mask(feats, n_atom, coords.device)
    coords_leaf = coords.detach().clone().requires_grad_(True)

    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx)               # [B, K, 3]
    K = sub.shape[-2]
    off_diag = 1.0 - torch.eye(K, device=sub.device, dtype=sub.dtype)

    # Per-sample distance matrices: [B, K, K]
    d_mat = torch.linalg.norm(sub.unsqueeze(-2) - sub.unsqueeze(-3), dim=-1)
    denom = float(K * (K - 1)) if K > 1 else 1.0

    # Build a full pair-dRMSD matrix ``pair_mat[B, B]`` (symmetric).
    # Vectorised: diff[i, j] = d_mat[i] - d_mat[j] has shape [B, B, K, K].
    # Squared distance-matrix difference, off-diagonal only:
    diff = d_mat.unsqueeze(1) - d_mat.unsqueeze(0)        # [B, B, K, K]
    diff_sq = (diff * diff) * off_diag
    pair_mat = torch.sqrt(diff_sq.sum(dim=(-2, -1)) / denom + 1e-12)
    # ``pair_mat[i, i] = 0`` by construction; zero out diagonal and
    # average over the other ``B - 1`` columns to get per-sample CV.
    eye_B = torch.eye(B, device=pair_mat.device, dtype=pair_mat.dtype)
    pair_off = pair_mat * (1.0 - eye_B)
    cv_per_sample = pair_off.sum(dim=-1) / float(B - 1)    # [B]

    # Gradient: `autograd.grad(CV_k.sum(), coords_leaf)` gives a
    # ``[B, N, 3]`` tensor where each row is the total "push" on that
    # sample arising from its role in every pair it participates in.
    grad = _autograd_grad(cv_per_sample.sum(), coords_leaf)
    return cv_per_sample.detach(), grad


CV_REGISTRY["pair_drmsd"] = pair_drmsd_cv
CV_REGISTRY["pair_rmsd"] = pair_drmsd_cv   # alias — Boltz's pair_rmsd ≈ this


# ═══════════════════════════════════════════════════════════════════════
# inter_chain distance — Ab-Ag docking positioning control
# Port of boltz.model.potentials.collective_variables.inter_chain_cv
# ═══════════════════════════════════════════════════════════════════════

def inter_chain_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    chain1_mask: Optional[torch.Tensor] = None,
    chain2_mask: Optional[torch.Tensor] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Distance between centres of mass of two chains.

    ``CV = || COM(chain1) - COM(chain2) ||``

    This is the same math as ``distance_cv`` but semantically tied to
    chain-level selection — callers pass chain-wide masks instead of
    arbitrary atom groups. Used for Ab-Ag docking pose control: steer
    or opt the inter-chain distance to control how close the antigen
    sits to the antibody.

    Both masks are ``[N_atom]`` bool. Returns per-sample distance and
    ``[..., N_atom, 3]`` gradient.
    """
    n_atom = coords.shape[-2]
    device = coords.device
    if chain1_mask is None or chain2_mask is None:
        return torch.zeros(coords.shape[:-2], device=device), torch.zeros_like(coords)

    m1 = chain1_mask.to(device=device, dtype=torch.float32)
    m2 = chain2_mask.to(device=device, dtype=torch.float32)
    coords_leaf = coords.detach().clone().requires_grad_(True)

    w1 = m1.view(*([1] * (coords_leaf.ndim - 2)), n_atom, 1)
    w2 = m2.view(*([1] * (coords_leaf.ndim - 2)), n_atom, 1)
    com1 = (coords_leaf * w1).sum(dim=-2) / m1.sum().clamp_min(1e-8)
    com2 = (coords_leaf * w2).sum(dim=-2) / m2.sum().clamp_min(1e-8)
    dist = torch.linalg.norm(com1 - com2, dim=-1)

    grad = _autograd_grad(dist, coords_leaf)
    return dist.detach(), grad


CV_REGISTRY["inter_chain"] = inter_chain_cv


# ═══════════════════════════════════════════════════════════════════════
# native_contacts (fraction Q) — fold/interface preservation
# Port of boltz.model.potentials.collective_variables.native_contacts_cv
# Uses a soft switching function so Q stays differentiable.
# ═══════════════════════════════════════════════════════════════════════

def native_contacts_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    reference_coords: torch.Tensor,
    reference_mask: Optional[torch.Tensor] = None,
    contact_cutoff: float = 4.5,
    beta: float = 5.0,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fraction of native contacts preserved (Q).

    ``Q = (1/|C|) · Σ_{(i,j) ∈ C}  σ((cutoff - d_ij) · β)``

    where ``C`` is the set of native contacts (pairs with
    ``||r_i* - r_j*|| <= cutoff`` in the reference), and σ is the
    logistic sigmoid. ``Q = 1`` means every native contact is intact;
    ``Q = 0`` means all contacts broken.

    - Steer Q=1 (harmonic) → preserve reference fold.
    - Opt Q max → push toward native-like contact map.
    - Opt Q min → push away from reference (explore).

    Args:
        reference_coords : [K, 3] Cα coords of reference (native/FFT pose).
        reference_mask   : [N_atom] bool. True where model atoms correspond
                           1-to-1 with reference_coords. Length must match.
        contact_cutoff   : Å. Pairs within this distance in the reference
                           are treated as "native contacts".
        beta             : logistic steepness. Higher = sharper cutoff,
                           less smooth gradient. Default 5.0 matches Boltz.
    """
    n_atom = coords.shape[-2]
    device = coords.device
    if reference_mask is None:
        reference_mask = _resolve_atom_mask(feats, n_atom, device)
    reference_mask = reference_mask.to(device=device, dtype=torch.bool)
    ref = reference_coords.to(device=device, dtype=torch.float32)
    K = ref.shape[0]
    assert int(reference_mask.sum().item()) == K, (
        f"reference_coords ({K}) must match reference_mask count "
        f"({int(reference_mask.sum().item())})"
    )

    # Build the native-contact pair set from the reference.
    d_ref = torch.linalg.norm(ref.unsqueeze(-2) - ref.unsqueeze(-3), dim=-1)
    native_mask = (d_ref <= float(contact_cutoff)) & (~torch.eye(K, dtype=torch.bool, device=device))
    n_native = native_mask.sum().clamp_min(1).float()
    if n_native.item() < 1:
        # Degenerate reference — return zeros rather than NaN.
        return torch.zeros(coords.shape[:-2], device=device), torch.zeros_like(coords)

    coords_leaf = coords.detach().clone().requires_grad_(True)
    idx = reference_mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx).to(torch.float32)  # [..., K, 3]

    # Per-sample pairwise distance in the query.
    d_query = torch.linalg.norm(sub.unsqueeze(-2) - sub.unsqueeze(-3), dim=-1)

    # Soft contact indicator: σ((cutoff - d) * β). Differentiable everywhere.
    contact_score = torch.sigmoid((float(contact_cutoff) - d_query) * float(beta))
    # Only count native pairs.
    contact_score = contact_score * native_mask.to(contact_score.dtype)
    Q = contact_score.sum(dim=(-2, -1)) / n_native

    grad = _autograd_grad(Q.sum(), coords_leaf)
    return Q.detach(), grad


CV_REGISTRY["native_contacts"] = native_contacts_cv
CV_REGISTRY["Q"] = native_contacts_cv  # physics convention


# ═══════════════════════════════════════════════════════════════════════
# RMSD (Kabsch-aligned) — classic structural similarity CV
# Port of boltz.model.potentials.collective_variables.rmsd_cv
# ═══════════════════════════════════════════════════════════════════════

def rmsd_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    reference_coords: torch.Tensor,
    reference_mask: Optional[torch.Tensor] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kabsch-aligned RMSD against a reference structure.

    Unlike ``drmsd_cv`` (which uses the distance matrix), this does
    explicit Kabsch rotation to align the query onto the reference
    before computing atomic RMSD. The gradient is taken w.r.t. the
    rotated query, so the result is rigid-invariant.

    This matches Boltz's ``rmsd_cv`` semantically (single-iteration
    Kabsch, no weighted iterative refinement — stable autograd).
    """
    n_atom = coords.shape[-2]
    device = coords.device
    if reference_mask is None:
        reference_mask = _resolve_atom_mask(feats, n_atom, device)
    reference_mask = reference_mask.to(device=device, dtype=torch.bool)
    ref = reference_coords.to(device=device, dtype=torch.float32)
    K = ref.shape[0]
    assert int(reference_mask.sum().item()) == K

    coords_leaf = coords.detach().clone().requires_grad_(True)
    idx = reference_mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx).to(torch.float32)

    # Kabsch: centre both, SVD, reflection fix.
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

    diff_sq = ((sub_rot - ref_c) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(diff_sq.mean(dim=-1) + 1e-12)

    grad = _autograd_grad(rmsd.sum(), coords_leaf)
    return rmsd.detach(), grad


CV_REGISTRY["rmsd"] = rmsd_cv


# ═══════════════════════════════════════════════════════════════════════
# max_diameter — largest pairwise distance within a group
# Useful for steering overall molecular extent (complements rg).
# ═══════════════════════════════════════════════════════════════════════

def max_diameter_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft-max approximation of the maximum pairwise distance.

    Because ``torch.max`` is non-smooth, we use
    ``d_max ≈ (Σ exp(β·d_ij)) / (Σ exp(β·d_ij) / d_ij)`` for large β
    (smooth soft-max). Default β = 5 gives a tight approximation to the
    true max while keeping gradients continuous.
    """
    n_atom = coords.shape[-2]
    mask = _resolve_atom_mask(feats, n_atom, coords.device)
    coords_leaf = coords.detach().clone().requires_grad_(True)

    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx)                # [..., K, 3]
    K = sub.shape[-2]
    # Compute pairwise distances.
    d = torch.linalg.norm(sub.unsqueeze(-2) - sub.unsqueeze(-3), dim=-1)
    # Zero out self-pairs, then soft-max via LogSumExp.
    mask_pair = 1.0 - torch.eye(K, device=d.device, dtype=d.dtype)
    d_masked = d * mask_pair - 1e6 * (1.0 - mask_pair)  # push diag far below
    beta = 5.0
    # logsumexp(β·d) / β ≈ max when β large.
    lse = torch.logsumexp((beta * d_masked).reshape(*d.shape[:-2], -1), dim=-1)
    d_max = lse / beta

    grad = _autograd_grad(d_max.sum(), coords_leaf)
    return d_max.detach(), grad


CV_REGISTRY["max_diameter"] = max_diameter_cv


# ═══════════════════════════════════════════════════════════════════════
# SASA (solvent-accessible surface area) — Shrake–Rupley approximation
# A simpler alternative to Boltz's 3000-line LCPO implementation:
# per-atom approximate SASA via spherical quadrature.
# ═══════════════════════════════════════════════════════════════════════

def _sphere_points(n: int, device, dtype) -> torch.Tensor:
    """Fibonacci sphere points ``[n, 3]`` (unit sphere)."""
    gold = (1.0 + torch.sqrt(torch.tensor(5.0))) / 2.0
    i = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = 2.0 * torch.pi * (i / gold) % (2 * torch.pi)
    z = 1.0 - 2.0 * i / n
    xy = torch.sqrt((1.0 - z * z).clamp_min(0))
    return torch.stack([xy * torch.cos(phi), xy * torch.sin(phi), z], dim=-1)


def sasa_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    probe_radius: float = 1.4,
    atom_radius: float = 1.9,
    n_quad: int = 48,
    chunk_size: int = 32,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate Shrake–Rupley SASA (autograd-friendly, chunked).

    ``SASA = Σ_i (4πR²) · visible_fraction(atom i)``  where
    ``visible_fraction`` is the mean of per-quadrature-point "not
    buried by any OTHER atom" scores, made continuous by a soft
    logistic:

        buried_soft(p, j) = σ((R − ||p − r_j||) · β)
        visible(p)        = Π_{j ≠ owner(p)} (1 − buried_soft(p, j))

    The original un-chunked layout materialised a
    ``[B, N·Q, N]`` distance matrix — for N=500, Q=48 that is
    **~3 GB peak** on autograd. This chunked version iterates atoms
    ``chunk_size`` at a time so the live distance tensor is at most
    ``[B, chunk·Q, N]`` (typically **~50 MB**).

    Args:
        probe_radius / atom_radius : Å. Water + atom VDW (default 1.4/1.9).
        n_quad       : Fibonacci-sphere points per atom (default 48).
        chunk_size   : atoms per forward chunk (default 32). Larger →
                       faster but more memory; smaller → slower, safer.
    """
    n_atom = coords.shape[-2]
    device = coords.device
    mask = _resolve_atom_mask(feats, n_atom, device).float()

    coords_leaf = coords.detach().clone().requires_grad_(True)
    sub = coords_leaf                                       # [..., N, 3]
    m = mask.view(*([1] * (sub.ndim - 2)), n_atom)          # [..., N]

    R = float(atom_radius + probe_radius)
    quad = _sphere_points(n_quad, device=device, dtype=sub.dtype) * R  # [Q, 3]
    beta = 10.0
    per_atom_area = 4.0 * torch.pi * R * R

    # Chunk kernel: forward only, autograd checkpoints the intermediate
    # distance/sigmoid tensors instead of keeping them alive until
    # backward. This drops peak memory for N=677 from ~2.9 GB to a few
    # hundred MB at ~2× compute cost.
    def _chunk_forward(centres_all, quad_pts, start, end):
        n_c = end - start
        pts_c = centres_all[..., start:end, :].unsqueeze(-2) + quad_pts.view(
            *([1] * (centres_all.ndim - 1)), n_quad, 3
        )
        pts_flat = pts_c.reshape(*centres_all.shape[:-2], n_c * n_quad, 3)
        d = torch.linalg.norm(
            pts_flat.unsqueeze(-2) - centres_all.unsqueeze(-3), dim=-1
        )
        buried_soft = torch.sigmoid((R - d) * beta)
        owner_global = torch.arange(start, end, device=device).repeat_interleave(n_quad)
        j_idx = torch.arange(n_atom, device=device).view(1, n_atom)
        own_mask = (owner_global.view(n_c * n_quad, 1) != j_idx).to(d.dtype)
        not_buried = 1.0 - buried_soft * own_mask
        visibility = not_buried.prod(dim=-1)
        return visibility.reshape(*centres_all.shape[:-2], n_c, n_quad).mean(dim=-1)

    from torch.utils.checkpoint import checkpoint

    vis_chunks = []
    for start in range(0, n_atom, chunk_size):
        end = min(start + chunk_size, n_atom)
        # `use_reentrant=False` is the non-legacy API; saves memory
        # without any side-effects for this pure function.
        vis_c = checkpoint(
            _chunk_forward, sub, quad, start, end, use_reentrant=False,
        )
        vis_chunks.append(vis_c)
    vis_per_atom = torch.cat(vis_chunks, dim=-1)

    sasa_total = (vis_per_atom * m).sum(dim=-1) * per_atom_area
    grad = _autograd_grad(sasa_total.sum(), coords_leaf)
    return sasa_total.detach(), grad


CV_REGISTRY["sasa"] = sasa_cv


# ═══════════════════════════════════════════════════════════════════════
# asphericity — shape-tensor eigenvalue spread
# A = (λ1-λ2)² + (λ2-λ3)² + (λ1-λ3)²  where λᵢ are the eigenvalues
# of the 3×3 gyration tensor. 0 = sphere; positive = elongated/flat.
# ═══════════════════════════════════════════════════════════════════════

def asphericity_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_atom = coords.shape[-2]
    mask = _resolve_atom_mask(feats, n_atom, coords.device).float()
    coords_leaf = coords.detach().clone().requires_grad_(True)

    w = mask.view(*([1] * (coords_leaf.ndim - 2)), n_atom, 1)
    n = mask.sum().clamp_min(1e-8)
    com = (coords_leaf * w).sum(dim=-2, keepdim=True) / n
    centred = (coords_leaf - com) * w                              # [..., N, 3]
    # Gyration tensor S = (1/N) Σ (r - com)(r - com)^T → [..., 3, 3]
    S = centred.transpose(-2, -1) @ centred / n
    # Symmetric eigvals (ascending). SVD of symmetric matrix == eig.
    eigvals = torch.linalg.eigvalsh(S)                             # [..., 3]
    l1, l2, l3 = eigvals.unbind(dim=-1)
    A = (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l1 - l3) ** 2
    grad = _autograd_grad(A.sum(), coords_leaf)
    return A.detach(), grad


CV_REGISTRY["asphericity"] = asphericity_cv


# ═══════════════════════════════════════════════════════════════════════
# min_distance — soft-min between two atom groups
# Differentiable via LogSumExp: min ≈ -1/β · log Σ exp(-β·d_ij)
# ═══════════════════════════════════════════════════════════════════════

def min_distance_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    region1_mask: Optional[torch.Tensor] = None,
    region2_mask: Optional[torch.Tensor] = None,
    softmin_beta: float = 10.0,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if region1_mask is None or region2_mask is None:
        return torch.zeros(coords.shape[:-2], device=coords.device), torch.zeros_like(coords)
    device = coords.device
    coords_leaf = coords.detach().clone().requires_grad_(True)

    i1 = region1_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    i2 = region2_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    if i1.numel() == 0 or i2.numel() == 0:
        return torch.zeros(coords.shape[:-2], device=device), torch.zeros_like(coords)

    # Pairwise distances: [..., |i1|, |i2|]
    a = coords_leaf.index_select(-2, i1).unsqueeze(-2)     # [..., n1, 1, 3]
    b = coords_leaf.index_select(-2, i2).unsqueeze(-3)     # [..., 1, n2, 3]
    d = torch.linalg.norm(a - b, dim=-1)
    beta = float(softmin_beta)
    # softmin = -logsumexp(-β·d) / β
    d_flat = d.reshape(*d.shape[:-2], -1)
    softmin = -torch.logsumexp(-beta * d_flat, dim=-1) / beta

    grad = _autograd_grad(softmin.sum(), coords_leaf)
    return softmin.detach(), grad


CV_REGISTRY["min_distance"] = min_distance_cv


# ═══════════════════════════════════════════════════════════════════════
# angle — 3-atom bond angle (by atom index)
# θ = acos((r21·r23) / (|r21|·|r23|)) at atom2.
# ═══════════════════════════════════════════════════════════════════════

def angle_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    atom1_idx: Optional[int] = None,
    atom2_idx: Optional[int] = None,
    atom3_idx: Optional[int] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if atom1_idx is None or atom2_idx is None or atom3_idx is None:
        return torch.zeros(coords.shape[:-2], device=coords.device), torch.zeros_like(coords)
    coords_leaf = coords.detach().clone().requires_grad_(True)
    r21 = coords_leaf[..., atom1_idx, :] - coords_leaf[..., atom2_idx, :]
    r23 = coords_leaf[..., atom3_idx, :] - coords_leaf[..., atom2_idx, :]
    cos_t = (r21 * r23).sum(-1) / (
        torch.linalg.norm(r21, dim=-1).clamp_min(1e-8)
        * torch.linalg.norm(r23, dim=-1).clamp_min(1e-8)
    )
    theta = torch.arccos(cos_t.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    grad = _autograd_grad(theta.sum(), coords_leaf)
    return theta.detach(), grad


CV_REGISTRY["angle"] = angle_cv


# ═══════════════════════════════════════════════════════════════════════
# dihedral — 4-atom torsion (by atom index)
# φ = atan2( (b1×b2)·(b2×b3)|b2|, (b1×b2·b3)·|b2| )
# ═══════════════════════════════════════════════════════════════════════

def dihedral_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    atom1_idx: Optional[int] = None,
    atom2_idx: Optional[int] = None,
    atom3_idx: Optional[int] = None,
    atom4_idx: Optional[int] = None,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if None in (atom1_idx, atom2_idx, atom3_idx, atom4_idx):
        return torch.zeros(coords.shape[:-2], device=coords.device), torch.zeros_like(coords)
    coords_leaf = coords.detach().clone().requires_grad_(True)
    r1 = coords_leaf[..., atom1_idx, :]
    r2 = coords_leaf[..., atom2_idx, :]
    r3 = coords_leaf[..., atom3_idx, :]
    r4 = coords_leaf[..., atom4_idx, :]
    b1, b2, b3 = r2 - r1, r3 - r2, r4 - r3
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)
    m1 = torch.linalg.cross(n1, b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp_min(1e-8), dim=-1)
    y = (m1 * n2).sum(-1)
    x = (n1 * n2).sum(-1)
    phi = torch.atan2(y, x)
    grad = _autograd_grad(phi.sum(), coords_leaf)
    return phi.detach(), grad


CV_REGISTRY["dihedral"] = dihedral_cv


# ═══════════════════════════════════════════════════════════════════════
# hbond_count — backbone N⋯O distance-based soft hbond count
# Uses supplied donor/acceptor masks (schema builds them from atom_array).
# ═══════════════════════════════════════════════════════════════════════

def hbond_count_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    donor_mask: Optional[torch.Tensor] = None,
    acceptor_mask: Optional[torch.Tensor] = None,
    distance_cutoff: float = 3.5,
    beta: float = 5.0,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if donor_mask is None or acceptor_mask is None:
        return torch.zeros(coords.shape[:-2], device=coords.device), torch.zeros_like(coords)
    device = coords.device
    coords_leaf = coords.detach().clone().requires_grad_(True)
    di = donor_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    ai = acceptor_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    if di.numel() == 0 or ai.numel() == 0:
        return torch.zeros(coords.shape[:-2], device=device), torch.zeros_like(coords)
    a = coords_leaf.index_select(-2, di).unsqueeze(-2)
    b = coords_leaf.index_select(-2, ai).unsqueeze(-3)
    d = torch.linalg.norm(a - b, dim=-1)
    count = torch.sigmoid((float(distance_cutoff) - d) * float(beta)).sum(dim=(-2, -1))
    grad = _autograd_grad(count.sum(), coords_leaf)
    return count.detach(), grad


CV_REGISTRY["hbond_count"] = hbond_count_cv


# ═══════════════════════════════════════════════════════════════════════
# salt_bridges — positive/negative charged side-chain pair count
# ═══════════════════════════════════════════════════════════════════════

def salt_bridges_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    positive_mask: Optional[torch.Tensor] = None,
    negative_mask: Optional[torch.Tensor] = None,
    distance_cutoff: float = 4.0,
    beta: float = 5.0,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return hbond_count_cv(
        coords, feats,
        donor_mask=positive_mask, acceptor_mask=negative_mask,
        distance_cutoff=distance_cutoff, beta=beta,
    )


CV_REGISTRY["salt_bridges"] = salt_bridges_cv


# ═══════════════════════════════════════════════════════════════════════
# coordination — total soft contact count within a cutoff
# Differentiable via sigmoid switching on pairwise distance matrix.
# ═══════════════════════════════════════════════════════════════════════

def coordination_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    contact_cutoff: float = 6.0,
    beta: float = 5.0,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_atom = coords.shape[-2]
    mask = _resolve_atom_mask(feats, n_atom, coords.device)
    coords_leaf = coords.detach().clone().requires_grad_(True)
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx)
    K = sub.shape[-2]
    d = torch.linalg.norm(sub.unsqueeze(-2) - sub.unsqueeze(-3), dim=-1)
    off_diag = 1.0 - torch.eye(K, device=d.device, dtype=d.dtype)
    coord_num = (torch.sigmoid((float(contact_cutoff) - d) * float(beta)) * off_diag
                 ).sum(dim=(-2, -1)) * 0.5  # count each pair once
    grad = _autograd_grad(coord_num.sum(), coords_leaf)
    return coord_num.detach(), grad


CV_REGISTRY["coordination"] = coordination_cv


# ═══════════════════════════════════════════════════════════════════════
# contact_order — mean sequence separation of contacts / L
# ═══════════════════════════════════════════════════════════════════════

def contact_order_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    *,
    contact_cutoff: float = 8.0,
    beta: float = 5.0,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_atom = coords.shape[-2]
    mask = _resolve_atom_mask(feats, n_atom, coords.device)
    coords_leaf = coords.detach().clone().requires_grad_(True)
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    sub = coords_leaf.index_select(-2, idx)
    K = sub.shape[-2]
    if K < 2:
        return torch.zeros(coords.shape[:-2], device=coords.device), torch.zeros_like(coords)
    d = torch.linalg.norm(sub.unsqueeze(-2) - sub.unsqueeze(-3), dim=-1)
    contact = torch.sigmoid((float(contact_cutoff) - d) * float(beta))
    off_diag = 1.0 - torch.eye(K, device=d.device, dtype=d.dtype)
    # Sequence separation |i - j| between masked atoms.
    seq_sep = (idx.unsqueeze(-1) - idx.unsqueeze(-2)).abs().to(d.dtype)
    # CO = Σ contact(i,j) · |i-j| / (Σ contact(i,j) · L)
    weighted = (contact * seq_sep * off_diag).sum(dim=(-2, -1))
    total = (contact * off_diag).sum(dim=(-2, -1)).clamp_min(1e-8)
    co = weighted / (total * float(K))
    grad = _autograd_grad(co.sum(), coords_leaf)
    return co.detach(), grad


CV_REGISTRY["contact_order"] = contact_order_cv


# ═══════════════════════════════════════════════════════════════════════
# rmsf — root mean square fluctuation across batch
# Per-atom sqrt(mean_s ||r_s - <r>||²), then mean over atoms.
# Shared value across all samples (ensemble-level CV), like pair_drmsd.
# ═══════════════════════════════════════════════════════════════════════

def rmsf_cv(
    coords: torch.Tensor,
    feats: dict[str, Any],
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if coords.ndim < 3 or coords.shape[0] < 2:
        return torch.zeros(coords.shape[:-2], device=coords.device), torch.zeros_like(coords)
    n_atom = coords.shape[-2]
    mask = _resolve_atom_mask(feats, n_atom, coords.device).float()
    coords_leaf = coords.detach().clone().requires_grad_(True)

    mean_pos = coords_leaf.mean(dim=0, keepdim=True)          # [1, N, 3]
    dev_sq = ((coords_leaf - mean_pos) ** 2).sum(dim=-1)       # [B, N]
    rmsf_per_atom = torch.sqrt(dev_sq.mean(dim=0) + 1e-12)     # [N]
    rmsf_mean = (rmsf_per_atom * mask).sum() / mask.sum().clamp_min(1e-8)
    value = rmsf_mean.expand(coords_leaf.shape[0]).contiguous()

    grad = _autograd_grad(value.sum(), coords_leaf)
    return value.detach(), grad


CV_REGISTRY["rmsf"] = rmsf_cv


def get_cv(name: str):
    """Resolve a CV function by name."""
    if name not in CV_REGISTRY:
        raise KeyError(
            f"Unknown collective_variable '{name}'. Available: "
            f"{sorted(set(CV_REGISTRY.keys()))}"
        )
    return CV_REGISTRY[name]
