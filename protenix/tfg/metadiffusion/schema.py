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

In addition to ``steer`` and ``opt``, Phase 2 adds ``explore`` with
Gaussian-hill deposition (well-tempered supported). SAXS and
chemical-shift potentials remain ``NotImplementedError`` — they need
experimental data that's not in scope for CASP17.

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
# Phase 1 shipped steer + opt. Phase 2 adds explore (metadynamics hills).
# SAXS / chemical_shift still require experimental data and stay Phase 3.
_UNSUPPORTED_KEYS = {"saxs", "chemical_shift"}

# Boltz-native fields that our Phase-1 parser accepts but does not yet
# act on. Warn the user once so they know the setting is silently
# dropped rather than wondering why tuning has no effect.
_ADVANCED_IGNORED_FIELDS = {
    # gradient post-processing — bias_tempering is the only one here
    # that is still not wired in (scaling / projection / modifier_order
    # landed in Phase D and are honoured by the potentials).
    "bias_tempering",
    # CV auto-tuning
    "target_from_saxs", "auto_rg_scale", "gaussian_noise_scale",
    # Boltz-specific selection hints that our parser does not honour
    "atom1", "atom2", "atom3", "atom4",
    "rmsd_groups", "selection",
    # SASA params beyond what our simple CV honours (probe_radius is
    # accepted; sasa_method (LCPO vs Shrake–Rupley) is not).
    "sasa_method",
}

# Set of seen warnings keyed by (term, field) so we warn only once per
# (term, field) per process lifetime.
_WARNED_ONCE: set[tuple[str, str]] = set()


def _warn_unsupported_fields(
    item_idx: int, term_name: str, body: dict[str, Any]
) -> None:
    """Emit a one-time WARN per advanced/ignored Boltz field."""
    for f in _ADVANCED_IGNORED_FIELDS:
        if f in body:
            key = (term_name, f)
            if key in _WARNED_ONCE:
                continue
            _WARNED_ONCE.add(key)
            logger.warning(
                "[metadiffusion] %s (item %d) uses '%s' — accepted by "
                "the parser but NOT yet applied (deferred Phase D). Your "
                "tuning here "
                "will have no effect.",
                term_name, item_idx, f,
            )


def _as_term_body(mode_cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy metadiffusion item body into the shape ``_build_terms`` expects.

    Anything not recognised as a potential param is dropped here —
    the factory (``build_metadiffusion_features``) picks those up
    separately when constructing CV kwargs.

    ``cv`` is preserved (under the param key ``cv``) so each term
    carries its own CV name; this lets a single run stack multiple
    metadiffusion terms with different CVs without them stepping on
    each other's ``metadiffusion_cv_name`` slot in ``feats``.
    """
    out = {k: v for k, v in mode_cfg.items() if k not in _CV_SPEC_KEYS}
    cv_name = mode_cfg.get("collective_variable") or mode_cfg.get("cv")
    if cv_name is not None:
        out["cv"] = str(cv_name)
    return out


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
    explore_i = 0

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
        # Validate the body dict up-front (Copilot #12). A common user
        # error is `- steer:` with no value, which yields `body = None`
        # and later crashes with a cryptic AttributeError / TypeError
        # deep inside `_warn_unsupported_fields` or `.get()`.
        for _mk in ("steer", "opt", "explore"):
            if _mk in item and not isinstance(item[_mk], dict):
                raise TypeError(
                    f"metadiffusion[{i}].{_mk} must be a mapping, got "
                    f"{type(item[_mk]).__name__}: {item[_mk]!r}. "
                    f"Did you forget the body?"
                )

        if "steer" in item:
            body = item["steer"]
            name = f"SteeringPotential__{steer_i}"
            steer_i += 1
            _warn_unsupported_fields(i, name, body)
            term_cfg: dict[str, Any] = {
                "interval": int(body.get("guidance_interval", 1)),
                "weight": float(body.get("weight", 1.0)),
                # Injected so the potential can look up its own CV kwargs
                # in ``feats`` (see build_metadiffusion_features).
                "_term_name": name,
            }
            term_cfg.update(_as_term_body(body))
            terms[name] = term_cfg

        elif "explore" in item:
            # Metadynamics hill-deposition mode. Phase 2.
            body = item["explore"]
            etype = str(body.get("type", body.get("explore_type", "hills"))).lower()
            if etype == "repulsion":
                raise NotImplementedError(
                    f"metadiffusion[{i}].explore type 'repulsion' is not "
                    f"supported yet (Phase 2 handles 'hills' / well-tempered)."
                )
            name = f"MetadynamicsPotential__{explore_i}"
            explore_i += 1
            _warn_unsupported_fields(i, name, body)
            term_cfg = {
                "interval": int(body.get("guidance_interval", 1)),
                "weight": float(body.get("weight", 1.0)),
                "_term_name": name,
            }
            term_cfg.update(_as_term_body(body))
            term_cfg.pop("type", None)
            term_cfg.pop("explore_type", None)
            terms[name] = term_cfg

        elif "opt" in item:
            body = item["opt"]
            name = f"OptPotential__{opt_i}"
            opt_i += 1
            _warn_unsupported_fields(i, name, body)
            target = body.get("target", "min")
            if not isinstance(target, str):
                raise ValueError(
                    f"metadiffusion[{i}].opt.target must be 'min' or 'max'; "
                    f"numeric target belongs under `steer:` instead."
                )
            t_lower = target.lower()
            if t_lower not in {"min", "max"}:
                raise ValueError(
                    f"metadiffusion[{i}].opt.target: expected 'min' or 'max', "
                    f"got {target!r}. (typo guard — see issue #8.)"
                )
            direction = -1.0 if t_lower == "max" else +1.0
            term_cfg = {
                "interval": int(body.get("guidance_interval", 1)),
                "weight": float(body.get("weight", 1.0)),
                "direction": direction,
                "_term_name": name,
            }
            term_cfg.update(_as_term_body(body))
            term_cfg.pop("target", None)
            terms[name] = term_cfg

        elif any(k in item for k in _UNSUPPORTED_KEYS):
            mode = next(k for k in _UNSUPPORTED_KEYS if k in item)
            raise NotImplementedError(
                f"metadiffusion[{i}] mode '{mode}' is not supported yet "
                f"(current support: steer / opt / explore. "
                f"saxs / chemical_shift deferred — need experimental data.)"
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


# ───────────────────────────────────────────────────────────────────────
# Region spec parser — Boltz-compatible
# ───────────────────────────────────────────────────────────────────────

def _parse_region_spec(spec: str, atom_array) -> np.ndarray:
    """Resolve a Boltz-style region spec to a ``[N_atom]`` bool mask.

    Supported grammar (matches
    ``boltz.model.potentials.factory._resolve_region_to_mask``)::

        "A"              whole chain A (all atoms)
        "A:1-50"         chain A residues 1..50 (all atoms)
        "A:5"            chain A residue 5 (all atoms)
        "A::CA"          chain A CA atoms only (whole chain)
        "A:1-50:CA"      chain A residues 1..50, CA atoms only
        "A:5:CA"         chain A residue 5 CA atom

    Residue IDs match ``atom_array.res_id``. Atom names match
    ``atom_array.atom_name``. Missing pieces are treated as wildcards.
    """
    n = len(atom_array)
    mask = np.zeros(n, dtype=bool)
    parts = [p.strip() for p in str(spec).split(":")]
    if not parts or not parts[0]:
        return mask

    chain_id = parts[0]
    chain_sel = atom_array.chain_id.astype(str) == chain_id
    if not chain_sel.any():
        logger.warning(
            "[metadiffusion/region] chain '%s' not in atom_array. "
            "Empty mask.", chain_id,
        )
        return mask

    # Residue filter (parts[1] if present and non-empty).
    if len(parts) >= 2 and parts[1]:
        rs = parts[1]
        if "-" in rs:
            lo, hi = rs.split("-", 1)
            res_sel = (atom_array.res_id >= int(lo)) & (atom_array.res_id <= int(hi))
        else:
            res_sel = atom_array.res_id == int(rs)
    else:
        res_sel = np.ones(n, dtype=bool)

    # Atom-name filter (parts[2] if present and non-empty).
    if len(parts) >= 3 and parts[2]:
        atom_sel = atom_array.atom_name.astype(str) == parts[2]
    else:
        atom_sel = np.ones(n, dtype=bool)

    mask = chain_sel & res_sel & atom_sel
    if not mask.any():
        logger.warning(
            "[metadiffusion/region] spec '%s' matched 0 atoms.", spec
        )
    return mask


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
    explore_i = 0

    for i, item in enumerate(metadiffusion_list):
        if not isinstance(item, dict):
            continue
        if len(item) == 1 and next(iter(item)) in _GLOBAL_KEYS:
            continue

        if "steer" in item:
            body = item["steer"]; term_name = f"SteeringPotential__{steer_i}"; steer_i += 1
        elif "opt" in item:
            body = item["opt"]; term_name = f"OptPotential__{opt_i}"; opt_i += 1
        elif "explore" in item:
            body = item["explore"]; term_name = f"MetadynamicsPotential__{explore_i}"; explore_i += 1
        else:
            continue

        # Early validation (Copilot #7): refuse silent 'None' fallbacks.
        cv_raw = body.get("collective_variable") or body.get("cv")
        if cv_raw is None:
            raise KeyError(
                f"metadiffusion[{i}].{'steer' if 'steer' in item else ('opt' if 'opt' in item else 'explore')}: "
                f"missing 'collective_variable' (or 'cv') key. "
                f"Metadiffusion requires a named CV."
            )
        cv_name = str(cv_raw)
        cv_kwargs: dict[str, Any] = {}

        # ── 1. Atom selection: `region1..4` takes precedence over `groups` ──
        regions = [body.get(f"region{i}") for i in (1, 2, 3, 4)]
        regions = [r for r in regions if r]
        groups = body.get("groups")

        if regions:
            # Build masks per region spec. Store on cv_kwargs as
            # `region1_mask`..`region4_mask`. The first region doubles as
            # the main `atom_selection_mask` (backwards compatible with
            # single-group CVs like rg / sasa).
            region_masks = []
            for r_i, spec in enumerate(regions, start=1):
                m_np = _parse_region_spec(spec, atom_array)
                m_t = torch.from_numpy(m_np.astype(bool))
                cv_kwargs[f"region{r_i}_mask"] = m_t
                region_masks.append(m_t)
            cv_kwargs["atom_selection_mask"] = region_masks[0]
            # Convenience aliases that match the CV function kwarg names.
            # Eg. `inter_chain_cv(chain1_mask, chain2_mask)` maps naturally.
            if len(region_masks) >= 2:
                cv_kwargs.setdefault("chain1_mask", region_masks[0])
                cv_kwargs.setdefault("chain2_mask", region_masks[1])
                cv_kwargs.setdefault("mask1", region_masks[0])
                cv_kwargs.setdefault("mask2", region_masks[1])
        elif groups:
            # Chain-level atom selection (Cα only).
            sel_mask_np = _query_ca_mask_for_chains(atom_array, groups)
            sel_mask = torch.from_numpy(sel_mask_np.astype(bool))
            cv_kwargs["atom_selection_mask"] = sel_mask
            # For 2-chain CVs (`inter_chain`, `distance`) unambiguously
            # split the groups into two chain masks.
            if len(groups) >= 2 and cv_name in {"inter_chain", "distance"}:
                m1_np = _chain_mask(atom_array, [groups[0]])
                m2_np = _chain_mask(atom_array, [groups[1]])
                cv_kwargs["chain1_mask"] = torch.from_numpy(m1_np.astype(bool))
                cv_kwargs["chain2_mask"] = torch.from_numpy(m2_np.astype(bool))
                cv_kwargs["mask1"] = cv_kwargs["chain1_mask"]
                cv_kwargs["mask2"] = cv_kwargs["chain2_mask"]
        else:
            sel_mask_np = _query_ca_mask_for_chains(atom_array, None)
            sel_mask = torch.from_numpy(sel_mask_np.astype(bool))
            cv_kwargs["atom_selection_mask"] = sel_mask

        # ── 2. Reference coords (for rmsd / drmsd / d_tm / native_contacts) ──
        if cv_name in {"drmsd", "d_tm", "tm", "rmsd", "native_contacts", "Q"}:
            ref_path = body.get("reference_structure")
            if ref_path is None:
                # Derive the actual mode label so the message doesn't
                # misreport an `explore` item as `opt` (Copilot #13).
                mode_label = next(
                    (k for k in ("steer", "opt", "explore") if k in item),
                    "?",
                )
                raise KeyError(
                    f"metadiffusion[{i}].{mode_label}: "
                    f"CV '{cv_name}' requires 'reference_structure'."
                )
            # Reference chain filter: use `groups` or, if only region1 is
            # given, infer chain ID from the region spec.
            ref_chains = groups
            if ref_chains is None and regions:
                ref_chains = [str(regions[0]).split(":", 1)[0]]
            ref_coords = _load_reference_ca(ref_path, ref_chains)
            cv_kwargs["reference_coords"] = ref_coords
            sel_mask = cv_kwargs["atom_selection_mask"]
            if int(sel_mask.sum().item()) != ref_coords.shape[0]:
                logger.warning(
                    "[metadiffusion] term %s: query Cα count %d != "
                    "reference Cα count %d. CV will likely fail; check "
                    "groups / region / reference_structure.",
                    term_name, int(sel_mask.sum().item()), ref_coords.shape[0],
                )
            cv_kwargs["reference_mask"] = sel_mask

        # ── 3. CV-specific params (forward to CV function as kwargs) ──
        for k in ("contact_cutoff", "beta", "probe_radius", "atom_radius",
                  "n_quad", "chunk_size"):
            if k in body:
                cv_kwargs[k] = body[k]

        # ── 4. Silently-ignored Boltz advanced fields → WARN ──
        _warn_unsupported_fields(i, term_name, body)

        # ── 4. Gradient modifiers (Phase D) ──
        # Boltz-compatible `scaling` / `projection` / `modifier_order`
        # entries. Each scaler/projector is itself a CV spec with an
        # optional per-modifier atom selection — we resolve those the
        # same way we resolve the primary CV's masks.
        from protenix.tfg.metadiffusion.gradient_mods import (
            parse_scaling, parse_projection,
        )

        def _resolve_mod_cv_kwargs(mod_entry: dict) -> dict:
            """Build CV kwargs for a scaling/projection entry."""
            mod_kwargs: dict[str, Any] = {}
            mod_regions = [mod_entry.get(f"region{i}") for i in (1, 2, 3, 4)]
            mod_regions = [r for r in mod_regions if r]
            mod_groups = mod_entry.get("groups")
            if mod_regions:
                masks = []
                for r in mod_regions:
                    m_np = _parse_region_spec(r, atom_array)
                    masks.append(torch.from_numpy(m_np.astype(bool)))
                mod_kwargs["atom_selection_mask"] = masks[0]
                if len(masks) >= 2:
                    mod_kwargs.setdefault("chain1_mask", masks[0])
                    mod_kwargs.setdefault("chain2_mask", masks[1])
                    mod_kwargs.setdefault("mask1", masks[0])
                    mod_kwargs.setdefault("mask2", masks[1])
            elif mod_groups:
                m_np = _query_ca_mask_for_chains(atom_array, mod_groups)
                mod_kwargs["atom_selection_mask"] = torch.from_numpy(m_np.astype(bool))
            # Reference structure for rmsd/drmsd/etc. inside a scaler.
            ref_path = mod_entry.get("reference_structure")
            if ref_path:
                refs = _load_reference_ca(ref_path, mod_groups)
                mod_kwargs["reference_coords"] = refs
                sel = mod_kwargs.get("atom_selection_mask")
                if sel is not None:
                    mod_kwargs["reference_mask"] = sel
            return mod_kwargs

        scalers = parse_scaling(body, _resolve_mod_cv_kwargs)
        projectors = parse_projection(body, _resolve_mod_cv_kwargs)
        modifier_order = body.get("modifier_order") or ["scaling", "projection"]

        terms_info.append({
            "term": term_name,
            "cv": cv_name,
            "cv_kwargs": cv_kwargs,
            "mods": {
                "scaling": scalers,
                "projection": projectors,
                "modifier_order": modifier_order,
            },
        })

    out: dict[str, Any] = {"metadiffusion_terms": terms_info}
    # Per-term prefixed entries so a potential can route to its own
    # CV / kwargs via ``params["_term_name"]`` without interfering
    # with sibling terms that pick a different CV (issue #6).
    for t in terms_info:
        out[f"metadiffusion_cv_name__{t['term']}"] = t["cv"]
        out[f"metadiffusion_cv_kwargs__{t['term']}"] = t["cv_kwargs"]
        if t.get("mods"):
            out[f"metadiffusion_mods__{t['term']}"] = t["mods"]
    # Legacy single-term convenience: unchanged for backward compat.
    if len(terms_info) == 1:
        out["metadiffusion_cv_name"] = terms_info[0]["cv"]
        out["metadiffusion_cv_kwargs"] = terms_info[0]["cv_kwargs"]
    return out
