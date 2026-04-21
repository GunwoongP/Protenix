# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Template reference feature builder for TemplateReferencePotential.

Produces the feats consumed by `protenix.tfg.potentials.TemplateReferencePotential`
from user-provided template mmCIF files. JSON schema (Boltz-compatible):

    {
      "name": "target",
      "sequences": [...],
      "templates": [
        {
          "cif": "/abs/path/template.cif",
          "chain_mapping": {"A": "A", "B": "B"},   # optional; default identity
          "force": true,                              # optional; default True
          "threshold": 2.0                            # optional Å; default 2.0
        }
      ]
    }

Mapping rules
-------------
- Each query token has a centre atom (from Protenix's TokenArray annotation).
- For each query token, we try to find the matching atom in the template's
  AtomArray by (chain_id, res_id, atom_name).
- `chain_mapping` (optional) translates query chain_id → template chain_id.
  Default: identity (A→A, B→B).
- If the template has no matching atom, the mask for that token stays False.
- Per-template `force` and `threshold` let the potential selectively enforce
  some templates and skip others.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _read_template_atom_lookup(
    cif_path: str,
) -> tuple[dict[tuple[str, int, str], np.ndarray], dict[tuple[str, int], str]]:
    """Parse a template mmCIF into two lookups keyed by residue position.

    Returns
    -------
    atom_lookup : ``(chain_id, res_id, atom_name) -> coord``
    res_name_by_pos : ``(chain_id, res_id) -> residue 3-letter name``
        The residue-name map lets callers key embedding/atom-list decisions on
        the *template's* residue identity rather than the query's, which
        matters when the query and template have mismatched residues at the
        aligned position.

    Some CIFs (e.g. computationally generated ones without MODEL records)
    are missing the `_atom_site.pdbx_PDB_model_num` column. biotite's
    `get_structure` requires it; fall back to manual construction from
    the raw atom_site columns when the column is absent.
    """
    # Lazy import: biotite may not be needed if no templates are used.
    from biotite.structure.io.pdbx import CIFFile, get_structure

    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"Template CIF not found: {cif_path}")

    cif = CIFFile.read(cif_path)
    try:
        aa = get_structure(cif, model=1)
        chain_ids = aa.chain_id
        res_ids = aa.res_id
        res_names = aa.res_name
        atom_names = aa.atom_name
        coords = aa.coord
    except KeyError as e:
        if "pdbx_PDB_model_num" not in str(e):
            raise
        # Fallback: parse atom_site directly.
        block = cif.block
        atom_site = block["atom_site"]

        def _get(col: str) -> np.ndarray:
            if col in atom_site:
                return atom_site[col].as_array()
            return None

        auth_asym = _get("auth_asym_id")
        label_asym = _get("label_asym_id")
        chain_ids = auth_asym if auth_asym is not None else label_asym
        label_seq = _get("label_seq_id")
        auth_seq = _get("auth_seq_id")
        res_ids = label_seq if label_seq is not None else auth_seq
        # Convert res_id to int, treating "." (unknown) as -1 so lookups miss.
        res_ids = np.array(
            [int(r) if r not in (".", "?", "") else -1 for r in res_ids],
            dtype=np.int64,
        )
        res_names = _get("label_comp_id")
        atom_names = _get("label_atom_id")
        x = atom_site["Cartn_x"].as_array(np.float32)
        y = atom_site["Cartn_y"].as_array(np.float32)
        z = atom_site["Cartn_z"].as_array(np.float32)
        coords = np.stack([x, y, z], axis=-1)

    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    res_name_by_pos: dict[tuple[str, int], str] = {}
    for i in range(len(coords)):
        rid = int(res_ids[i])
        if rid < 0:
            continue
        chain = str(chain_ids[i])
        lookup[(chain, rid, str(atom_names[i]))] = coords[i]
        # First occurrence wins; subsequent atoms at the same residue share
        # the residue name, so skip the dict write when already seen.
        res_name_by_pos.setdefault((chain, rid), str(res_names[i]))
    return lookup, res_name_by_pos


# Three-letter residue code → one-letter code (proteins + common nucleic acids)
_RES3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "T",
    "DA": "A", "DC": "C", "DG": "G", "DT": "T",
}


def _compute_chain_offset(
    query_chain_seqs: dict[str, list[tuple[int, str]]],
    template_chain_seqs: dict[str, list[tuple[int, str]]],
    chain_map: dict[str, str],
) -> dict[str, int]:
    """Compute per-chain (query_res_id → template_res_id) offset using string alignment.

    Strategy: collapse each chain's residues into a one-letter string. Try every
    reasonable offset (template_res_id = query_res_id - off) and pick the one
    that maximises identity matches. Falls back to offset=0 when no chain match
    is reliable.

    Args:
        query_chain_seqs: {query_chain_id: [(res_id, res1_letter), ...]}
        template_chain_seqs: {template_chain_id: [(res_id, res1_letter), ...]}
        chain_map: query_chain → template_chain

    Returns:
        {query_chain_id: offset}  (template_res_id = query_res_id - offset)
    """
    offsets: dict[str, int] = {}
    for q_chain, q_pairs in query_chain_seqs.items():
        t_chain = chain_map.get(q_chain, q_chain)
        t_pairs = template_chain_seqs.get(t_chain)
        if not t_pairs or not q_pairs:
            offsets[q_chain] = 0
            continue

        # Convert to lookup and 1-letter strings for fast substring search.
        q_str = "".join(c for _, c in q_pairs)
        t_str = "".join(c for _, c in t_pairs)
        q_first = q_pairs[0][0]
        t_first = t_pairs[0][0]

        best_off, best_hits = 0, -1
        # Candidate: template exactly a substring of query (common case: query
        # has extra signal peptide / tag).
        hit_idx = q_str.find(t_str) if t_str and t_str in q_str else -1
        if hit_idx >= 0:
            # Query residue at position hit_idx (0-indexed) has res_id = q_first + hit_idx.
            # Template first residue has res_id t_first.
            # offset = q_res - t_res = (q_first + hit_idx) - t_first
            best_off = (q_first + hit_idx) - t_first
            best_hits = len(t_str)
        else:
            # Fallback: sliding window over a reasonable range.
            q_lookup = {rid: c for rid, c in q_pairs}
            t_lookup = {rid: c for rid, c in t_pairs}
            q_ids = sorted(q_lookup)
            t_ids = sorted(t_lookup)
            q_min, q_max = q_ids[0], q_ids[-1]
            t_min, t_max = t_ids[0], t_ids[-1]
            # offset = q_res - t_res, so for each candidate offset the overlap window is
            # max(q_min, t_min+offset) .. min(q_max, t_max+offset)
            search_min = q_min - t_max
            search_max = q_max - t_min
            step = 1
            # Cap search space for very long chains.
            if search_max - search_min > 500:
                step = max(1, (search_max - search_min) // 500)
            for off in range(search_min, search_max + 1, step):
                hits = 0
                start = max(q_min, t_min + off)
                end = min(q_max, t_max + off)
                for r in range(start, end + 1):
                    if q_lookup.get(r) and q_lookup[r] == t_lookup.get(r - off):
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_off = off

        if best_hits > 0:
            offsets[q_chain] = best_off
        else:
            offsets[q_chain] = 0
    return offsets


def build_template_reference_features(
    input_dict: dict[str, Any],
    atom_array,  # biotite AtomArray (query)
    token_array,  # Protenix TokenArray (query)
) -> dict[str, torch.Tensor]:
    """Build TemplateReferencePotential feats from `input_dict['templates']`.

    Returns an empty dict when no `templates` field is present (or empty list),
    so callers can unconditionally `feature_dict.update(...)`.

    Returned keys (only when templates are present):
        template_cb:              [N_tmpl, N_token, 3]  float32
        template_mask_cb:         [N_tmpl, N_token]     bool
        template_force:           [N_tmpl]              bool
        template_force_threshold: [N_tmpl]              float32
        token_centre_atom_idx:    [N_token]             int64
    """
    templates = input_dict.get("templates", []) or []
    if not templates:
        return {}

    # Token-level centre atom index (already computed by Protenix tokenizer).
    # Protenix's TokenArray.get_annotation may return either a list or ndarray
    # depending on backend, so normalize to int64 ndarray.
    centre_atom_idx = np.asarray(
        token_array.get_annotation("centre_atom_index"), dtype=np.int64
    )
    n_token = int(centre_atom_idx.shape[0])
    if n_token == 0:
        return {}

    # Build per-token (q_chain, q_res, q_atom, q_res3) keys from the query atom array.
    q_keys: list[tuple[str, int, str, str]] = []
    for tok_i in range(n_token):
        a_idx = centre_atom_idx[tok_i]
        q_keys.append((
            str(atom_array.chain_id[a_idx]),
            int(atom_array.res_id[a_idx]),
            str(atom_array.atom_name[a_idx]),
            str(atom_array.res_name[a_idx]),
        ))

    # Per-chain query residue sequence (for alignment).
    query_chain_seqs: dict[str, list[tuple[int, str]]] = {}
    _seen: set[tuple[str, int]] = set()
    for q_chain, q_res, _, q_res3 in q_keys:
        if (q_chain, q_res) in _seen:
            continue
        _seen.add((q_chain, q_res))
        c1 = _RES3_TO_1.get(q_res3)
        if c1 is None:
            continue
        query_chain_seqs.setdefault(q_chain, []).append((q_res, c1))
    for c in query_chain_seqs:
        query_chain_seqs[c].sort()

    n_tmpl = len(templates)
    template_cb = np.zeros((n_tmpl, n_token, 3), dtype=np.float32)
    template_mask_cb = np.zeros((n_tmpl, n_token), dtype=bool)
    template_force = np.zeros((n_tmpl,), dtype=bool)
    template_threshold = np.full((n_tmpl,), 2.0, dtype=np.float32)

    for k, tmpl in enumerate(templates):
        cif_path = tmpl["cif"]
        if not os.path.isabs(cif_path):
            raise ValueError(
                f"templates[{k}].cif must be an absolute path: {cif_path!r}"
            )
        template_force[k] = bool(tmpl.get("force", True))
        template_threshold[k] = float(tmpl.get("threshold", 2.0))
        chain_map = dict(tmpl.get("chain_mapping", {}) or {})

        lookup, res_name_by_pos = _read_template_atom_lookup(cif_path)

        # Derive per-chain template sequence from the residue-name map; CA
        # atoms exist for all protein residues so keyed iteration matches the
        # old CA-scan behaviour without a second CIF pass.
        template_chain_seqs: dict[str, list[tuple[int, str]]] = {}
        for (t_chain_id, t_res_id), res3 in res_name_by_pos.items():
            c1 = _RES3_TO_1.get(str(res3))
            if c1 is None:
                continue
            template_chain_seqs.setdefault(t_chain_id, []).append((t_res_id, c1))
        for c in template_chain_seqs:
            template_chain_seqs[c].sort()

        # Offset per query chain (template_res_id = query_res_id - offset).
        offsets = _compute_chain_offset(query_chain_seqs, template_chain_seqs, chain_map)

        hit = 0
        for tok_i, (q_chain, q_res, q_atom, _q_res3) in enumerate(q_keys):
            t_chain = chain_map.get(q_chain, q_chain)  # identity default
            off = offsets.get(q_chain, 0)
            t_res = q_res - off
            key = (t_chain, t_res, q_atom)
            coord = lookup.get(key)
            if coord is None:
                # Some protein residues use CA as centre atom even when CB would
                # be preferred — upstream templates may disagree. Fall through
                # to CB fallback only if this token is a standard protein CA.
                if q_atom == "CA":
                    fallback = lookup.get((t_chain, t_res, "CB"))
                    if fallback is not None:
                        template_cb[k, tok_i] = fallback
                        template_mask_cb[k, tok_i] = True
                        hit += 1
                        continue
                continue
            template_cb[k, tok_i] = coord
            template_mask_cb[k, tok_i] = True
            hit += 1

        logger.info(
            "[template_reference] template[%d] %s: matched %d / %d query tokens "
            "(force=%s, threshold=%.2f Å)",
            k, os.path.basename(cif_path), hit, n_token,
            bool(template_force[k]), float(template_threshold[k]),
        )

    return {
        "template_cb": torch.from_numpy(template_cb),
        "template_mask_cb": torch.from_numpy(template_mask_cb),
        "template_force": torch.from_numpy(template_force),
        "template_force_threshold": torch.from_numpy(template_threshold),
        "token_centre_atom_idx": torch.from_numpy(centre_atom_idx),
    }


# ════════════════════════════════════════════════════════════════════════════
# Trunk-level template features (Boltz-style; for models with
# `template_embedder.n_blocks >= 1`, e.g. protenix-v2).
# ════════════════════════════════════════════════════════════════════════════

def build_trunk_template_features(
    input_dict: dict[str, Any],
    atom_array,
    token_array,
) -> dict[str, torch.Tensor]:
    """Build trunk-level template feats for Protenix's template_embedder.

    Produces (template_aatype, template_atom_positions, template_atom_mask)
    from the same `input_dict['templates']` list used by the TFG potential.
    Downstream Protenix code (`Templates.as_protenix_dict`) will then compute
    `template_distogram`, `template_unit_vector`, `template_backbone_frame_mask`,
    `template_pseudo_beta_mask` from these three arrays.

    Returns empty dict when no templates, or for non-protein query tokens
    (atoms outside `ATOM14` convention contribute zero-masked entries).

    Shapes
    ------
    template_aatype         : [N_tmpl, N_token]               int64
    template_atom_positions : [N_tmpl, N_token, N_dense, 3]   float32
    template_atom_mask      : [N_tmpl, N_token, N_dense]      float32
        where N_dense = 24 (Protenix DENSE_ATOM convention, sized for NA+protein).
    """
    templates = input_dict.get("templates", []) or []
    if not templates:
        return {}

    from protenix.data.constants import (
        ATOM14,
        DENSE_ATOM,
        PROTEIN_COMMON_ONE_TO_THREE,
        PROTEIN_TYPES_ONE_LETTER,
    )

    # Restype index convention: PROTEIN_TYPES_ONE_LETTER[i] = 1-letter code for aatype=i.
    # UNK and non-standard residues → aatype=len(PROTEIN_TYPES_ONE_LETTER)=20 (UNK slot).
    _ONE_TO_IDX = {c: i for i, c in enumerate(PROTEIN_TYPES_ONE_LETTER)}
    _THREE_TO_ONE = {v: k for k, v in PROTEIN_COMMON_ONE_TO_THREE.items()}
    UNK_IDX = len(PROTEIN_TYPES_ONE_LETTER)  # 20

    # Centre atom → query residue (chain, res_id, res3name) per token.
    centre_atom_idx = np.asarray(
        token_array.get_annotation("centre_atom_index"), dtype=np.int64
    )
    n_token = int(centre_atom_idx.shape[0])
    if n_token == 0:
        return {}
    q_chain_ids: list[str] = []
    q_res_ids: list[int] = []
    q_res_names: list[str] = []
    for tok_i in range(n_token):
        a_idx = int(centre_atom_idx[tok_i])
        q_chain_ids.append(str(atom_array.chain_id[a_idx]))
        q_res_ids.append(int(atom_array.res_id[a_idx]))
        q_res_names.append(str(atom_array.res_name[a_idx]))

    # Per-chain query seq for offset alignment.
    query_chain_seqs: dict[str, list[tuple[int, str]]] = {}
    _seen_q: set[tuple[str, int]] = set()
    for q_chain, q_res, q_res3 in zip(q_chain_ids, q_res_ids, q_res_names):
        if (q_chain, q_res) in _seen_q:
            continue
        _seen_q.add((q_chain, q_res))
        c1 = _RES3_TO_1.get(q_res3)
        if c1 is not None:
            query_chain_seqs.setdefault(q_chain, []).append((q_res, c1))
    for c in query_chain_seqs:
        query_chain_seqs[c].sort()

    # Allocate output arrays (Protenix DENSE_ATOM max width).
    N_dense = max(len(v) for v in DENSE_ATOM.values())
    n_tmpl = len(templates)
    aatype = np.full((n_tmpl, n_token), UNK_IDX, dtype=np.int64)
    atom_positions = np.zeros((n_tmpl, n_token, N_dense, 3), dtype=np.float32)
    atom_mask = np.zeros((n_tmpl, n_token, N_dense), dtype=np.float32)

    for k, tmpl in enumerate(templates):
        cif_path = tmpl["cif"]
        if not os.path.isabs(cif_path):
            raise ValueError(
                f"templates[{k}].cif must be an absolute path: {cif_path!r}"
            )
        chain_map = dict(tmpl.get("chain_mapping", {}) or {})

        lookup, res_name_by_pos = _read_template_atom_lookup(cif_path)

        # Extract template per-chain sequence from the residue-name map.
        template_chain_seqs: dict[str, list[tuple[int, str]]] = {}
        for (t_chain_id, t_res_id), res3 in res_name_by_pos.items():
            c1 = _RES3_TO_1.get(str(res3))
            if c1 is None:
                continue
            template_chain_seqs.setdefault(t_chain_id, []).append((t_res_id, c1))
        for c in template_chain_seqs:
            template_chain_seqs[c].sort()

        offsets = _compute_chain_offset(query_chain_seqs, template_chain_seqs, chain_map)

        filled = 0
        for tok_i, (q_chain, q_res, q_res3) in enumerate(
            zip(q_chain_ids, q_res_ids, q_res_names)
        ):
            t_chain = chain_map.get(q_chain, q_chain)
            t_res = q_res - offsets.get(q_chain, 0)

            # Boltz convention: template aatype / atom list follow the
            # TEMPLATE residue identity, not the query's. Fall back to the
            # query residue only when the template has no residue at the
            # aligned position (leaves mask=0 for that token either way).
            t_res3 = res_name_by_pos.get((t_chain, t_res))
            if t_res3 is None or t_res3 not in ATOM14:
                t_res3 = q_res3 if q_res3 in ATOM14 else None
            if t_res3 is None:
                # Non-standard residue on both sides; cannot fill ATOM14 slots.
                continue
            one_letter = _THREE_TO_ONE.get(t_res3)
            if one_letter is None or one_letter not in _ONE_TO_IDX:
                continue
            aatype[k, tok_i] = _ONE_TO_IDX[one_letter]

            atom_names = ATOM14[t_res3]  # canonical per-residue order (template)
            res_hit = False
            for slot, name in enumerate(atom_names):
                key = (t_chain, t_res, name)
                coord = lookup.get(key)
                if coord is None:
                    continue
                atom_positions[k, tok_i, slot] = coord
                atom_mask[k, tok_i, slot] = 1.0
                res_hit = True
            if res_hit:
                filled += 1

        logger.info(
            "[template_reference/trunk] template[%d] %s: %d / %d tokens "
            "have at least one atom filled",
            k, os.path.basename(cif_path), filled, n_token,
        )

    # Compute derived features (distogram, unit_vector, pseudo_beta_mask,
    # backbone_frame_mask) using Protenix's Templates helper. The template
    # embedder expects these to be pre-computed in input_feature_dict.
    from protenix.data.template.template_featurizer import Templates

    t = Templates(aatype=aatype, atom_positions=atom_positions, atom_mask=atom_mask)
    derived = t.as_protenix_dict()  # includes aatype/atom_positions/atom_mask + 4 more

    # Convert to tensors
    out: dict[str, torch.Tensor] = {}
    for k_, v in derived.items():
        arr = np.asarray(v)
        if k_ == "template_aatype":
            out[k_] = torch.from_numpy(arr.astype(np.int64))
        elif arr.dtype == np.float64:
            out[k_] = torch.from_numpy(arr.astype(np.float32))
        else:
            out[k_] = torch.from_numpy(arr)
    return out
