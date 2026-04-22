# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""MetaDiffusion YAML/JSON schema → Protenix TFG config bridge.

Parses the ``metadiffusion: [...]`` list (Boltz-compatible layout) and
produces a ``guidance.terms`` mapping that :func:`protenix.tfg.config.parse_tfg_config`
already knows how to consume.

Supported item shapes (MVP)::

    metadiffusion:
      - total_bias_clip: 4.0        # (optional) handled by engine clip
      - noise_scale: 1.0            # (optional) handled by engine noise

      - steer:
          collective_variable: rg
          target: 10.0
          strength: 4.0
          mode: harmonic            # "harmonic" (default) | "gaussian"
          sigma: 2.0                # used when mode == "gaussian"
          warmup: 0.1
          cutoff: 0.8
          ensemble: false
          groups: [A]               # (optional) chain/region for CV mask
          reference_structure: /abs/path.cif   # (optional) for CV rmsd/tm/drmsd

      - opt:
          collective_variable: pair_rmsd
          target: max               # "min" | "max"
          strength: 2.0
          warmup: 0.1
          cutoff: 0.8
          log_gradient: true

Everything else in Boltz's schema (explore/saxs/chemical_shift) raises
``NotImplementedError`` for now — intentional scope for Phase 1.

Returned shape
--------------
:func:`parse_metadiffusion_block` returns a ``terms_dict`` suitable for
dropping directly into the TFG config's ``guidance.terms`` mapping.
Keys are *unique* term names (``SteeringPotential__0``, ``OptPotential__1``,
…) because the same potential class can appear multiple times.

The ``feats`` dict enrichment (``metadiffusion_cv_name`` /
``metadiffusion_cv_kwargs``) happens separately in
:func:`build_metadiffusion_features`, which is invoked from the inference
dataloader after ``AtomArray`` / ``TokenArray`` are built (so CV kwargs
like atom masks and reference coords can be resolved against the actual
token layout).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import numpy as np
import torch


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Term-builder: metadiffusion list → guidance.terms mapping
# ───────────────────────────────────────────────────────────────────────

_GLOBAL_KEYS = {"total_bias_clip", "noise_scale", "guidance_mode", "denoise_tempering"}
_UNSUPPORTED_KEYS = {"explore", "saxs", "chemical_shift"}


def _as_term_body(mode_cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy metadiffusion item body into the shape ``_build_terms`` expects.

    Anything not recognised as a potential param is dropped here
    (logged once) — the factory (``build_metadiffusion_features``)
    picks those up separately when constructing CV kwargs.
    """
    return {k: v for k, v in mode_cfg.items() if k not in _CV_SPEC_KEYS}


# Keys that describe *CV selection*, not potential behaviour. Consumed
# by :func:`build_metadiffusion_features`, never by the potential itself.
_CV_SPEC_KEYS = {
    "collective_variable",
    "cv",
    "groups",
    "atom1", "atom2", "atom3", "atom4",
    "region1", "region2", "region3", "region4",
    "reference_structure",
    "rmsd_groups",
    "selection",
    "contact_cutoff",
    "probe_radius",
    "sasa_method",
}


def parse_metadiffusion_block(
    metadiffusion_list: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Split metadiffusion list into TFG terms + global engine options.

    Returns:
        terms_dict : ``{term_name: {interval, weight, ...params}}``.
        globals    : ``{total_bias_clip?, noise_scale?, ...}`` — to be
                     consumed by the TFG engine; unknown globals are
                     passed through.
    """
    if not metadiffusion_list:
        return {}, {}

    terms: dict[str, dict[str, Any]] = {}
    globals_: dict[str, Any] = {}

    steer_i = 0
    opt_i = 0

    for i, item in enumerate(metadiffusion_list):
        if not isinstance(item, dict):
            raise TypeError(
                f"metadiffusion[{i}]: each entry must be a mapping, got {type(item)}"
            )

        # Global-only item (single-key, key in _GLOBAL_KEYS).
        if len(item) == 1 and next(iter(item)) in _GLOBAL_KEYS:
            key, val = next(iter(item.items()))
            globals_[key] = val
            continue

        # Mode items — one of {steer, opt, explore, saxs, chemical_shift}.
        if "steer" in item:
            body = item["steer"]
            name = f"SteeringPotential__{steer_i}"
            steer_i += 1
            term_cfg: dict[str, Any] = {
                "interval": int(body.get("guidance_interval", 1)),
                "weight": float(body.get("weight", 1.0)),
            }
            term_cfg.update(_as_term_body(body))
            # The potential reads `mode`, `target`, `strength`, `sigma`,
            # `warmup`, `cutoff`, `ensemble` directly; pass through.
            terms[name] = term_cfg

        elif "opt" in item:
            body = item["opt"]
            name = f"OptPotential__{opt_i}"
            opt_i += 1
            # "target" in opt is "min" | "max" → translate to direction.
            target = body.get("target", "min")
            if isinstance(target, str):
                direction = -1.0 if target.lower() == "max" else +1.0
            else:
                # Numeric means "min |CV - target|" which is really
                # steering; recommend the user switch modes.
                raise ValueError(
                    f"metadiffusion[{i}].opt.target must be 'min' or 'max'; "
                    f"numeric target belongs under `steer:` instead."
                )
            term_cfg = {
                "interval": int(body.get("guidance_interval", 1)),
                "weight": float(body.get("weight", 1.0)),
                "direction": direction,
            }
            term_cfg.update(_as_term_body(body))
            term_cfg.pop("target", None)  # consumed above
            terms[name] = term_cfg

        elif any(k in item for k in _UNSUPPORTED_KEYS):
            mode = next(k for k in _UNSUPPORTED_KEYS if k in item)
            raise NotImplementedError(
                f"metadiffusion[{i}] mode '{mode}' is not supported yet "
                f"(Phase 1 covers steer + opt only)."
            )
        else:
            raise KeyError(
                f"metadiffusion[{i}] has no recognised mode key "
                f"(expected one of: steer, opt, explore, saxs, "
                f"chemical_shift, or a global key {sorted(_GLOBAL_KEYS)})."
            )

    return terms, globals_


# ───────────────────────────────────────────────────────────────────────
# Feature-builder: resolve CV spec → masks & reference coords
# ───────────────────────────────────────────────────────────────────────

def _chain_mask(atom_array, chain_ids: Iterable[str]) -> np.ndarray:
    """Return a ``[N_atom]`` bool mask for atoms on the given chain ids."""
    ids = set(str(c) for c in chain_ids)
    return np.isin(atom_array.chain_id.astype(str), list(ids))


def _ca_mask(atom_array) -> np.ndarray:
    """CA atoms only (standard residues)."""
    return atom_array.atom_name.astype(str) == "CA"


def _load_reference_ca(path: str, chain_filter: Iterable[str] | None = None) -> torch.Tensor:
    """Load Cα coords from a CIF / PDB reference structure.

    Returns ``[N_ref, 3]`` fp32. If ``chain_filter`` is provided, only
    atoms on those chains are included.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"reference_structure not found: {path}")

    # Lazy import so biotite is only needed when a reference is used.
    from biotite.structure.io.pdbx import CIFFile, get_structure
    from biotite.structure.io.pdb import PDBFile

    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        aa = get_structure(CIFFile.read(path), model=1)
    elif ext == ".pdb":
        aa = PDBFile.read(path).get_structure(model=1)
    else:
        raise ValueError(f"Unsupported reference format: {ext}")

    ca_mask = aa.atom_name == "CA"
    if chain_filter is not None:
        chain_filter = set(str(c) for c in chain_filter)
        ca_mask &= np.isin(aa.chain_id.astype(str), list(chain_filter))

    coords = aa.coord[ca_mask].astype(np.float32)
    return torch.from_numpy(coords)


def _query_ca_mask_for_chains(
    atom_array, chain_ids: Iterable[str] | None
) -> np.ndarray:
    """Return a ``[N_atom]`` bool Cα mask restricted to given chains.

    When ``chain_ids`` is None, includes Cα on all chains.
    """
    mask = _ca_mask(atom_array)
    if chain_ids is not None:
        mask &= _chain_mask(atom_array, chain_ids)
    return mask


def build_metadiffusion_features(
    metadiffusion_list: list[dict[str, Any]],
    atom_array,
) -> dict[str, Any]:
    """Turn ``metadiffusion: [...]`` into feats injected into TFG.

    For each mode item we compute CV-level kwargs (atom masks, reference
    coords). These are attached under term-indexed keys so the engine
    can route them to the matching potential::

        metadiffusion_cv_name    -> str (if single term; convenience)
        metadiffusion_cv_kwargs  -> dict (if single term; convenience)
        metadiffusion_terms      -> list[dict]  (canonical, supports
                                                 many terms):
            [{'term': 'SteeringPotential__0',
              'cv': 'rg',
              'cv_kwargs': {'reference_coords': tensor, ...}}, ...]
    """
    if not metadiffusion_list:
        return {}

    terms_info: list[dict[str, Any]] = []
    steer_i = 0
    opt_i = 0

    for i, item in enumerate(metadiffusion_list):
        if not isinstance(item, dict):
            continue
        if len(item) == 1 and next(iter(item)) in _GLOBAL_KEYS:
            continue

        if "steer" in item:
            body = item["steer"]; term_name = f"SteeringPotential__{steer_i}"; steer_i += 1
        elif "opt" in item:
            body = item["opt"]; term_name = f"OptPotential__{opt_i}"; opt_i += 1
        else:
            continue

        cv_name = str(body.get("collective_variable") or body.get("cv"))
        cv_kwargs: dict[str, Any] = {}

        # Atom selection mask (CV's main target atoms) from `groups`.
        groups = body.get("groups")
        if groups:
            sel_mask_np = _query_ca_mask_for_chains(atom_array, groups)
            sel_mask = torch.from_numpy(sel_mask_np.astype(bool))
            cv_kwargs["atom_selection_mask"] = sel_mask
        else:
            sel_mask_np = _query_ca_mask_for_chains(atom_array, None)
            sel_mask = torch.from_numpy(sel_mask_np.astype(bool))
            cv_kwargs["atom_selection_mask"] = sel_mask

        # Reference coordinates for CVs that need them (rmsd/tm/drmsd).
        if cv_name in {"drmsd", "d_tm", "tm", "rmsd"}:
            ref_path = body.get("reference_structure")
            if ref_path is None:
                raise KeyError(
                    f"metadiffusion[{i}].{'steer' if 'steer' in item else 'opt'}: "
                    f"CV '{cv_name}' requires 'reference_structure'."
                )
            ref_chains = groups  # reference subset follows query groups
            ref_coords = _load_reference_ca(ref_path, ref_chains)
            cv_kwargs["reference_coords"] = ref_coords
            # For drmsd/d_tm the mask must count exactly reference_coords.
            # The CV functions assert this; surface early if mismatched.
            if int(sel_mask.sum().item()) != ref_coords.shape[0]:
                logger.warning(
                    "[metadiffusion] term %s: query Cα count %d != "
                    "reference Cα count %d. CV will likely fail; check "
                    "chain_mapping / reference_structure.",
                    term_name, int(sel_mask.sum().item()), ref_coords.shape[0],
                )
            cv_kwargs["reference_mask"] = sel_mask

        terms_info.append({
            "term": term_name,
            "cv": cv_name,
            "cv_kwargs": cv_kwargs,
        })

    out: dict[str, Any] = {"metadiffusion_terms": terms_info}
    # Convenience: if only one term, expose at top-level too so
    # potentials can look it up without iterating.
    if len(terms_info) == 1:
        out["metadiffusion_cv_name"] = terms_info[0]["cv"]
        out["metadiffusion_cv_kwargs"] = terms_info[0]["cv_kwargs"]
    return out
