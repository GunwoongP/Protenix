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


def _read_template_atom_lookup(cif_path: str) -> dict[tuple[str, int, str], np.ndarray]:
    """Parse a template mmCIF into a (chain_id, res_id, atom_name) → coord lookup."""
    # Lazy import: biotite may not be needed if no templates are used.
    from biotite.structure.io.pdbx import CIFFile, get_structure

    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"Template CIF not found: {cif_path}")

    cif = CIFFile.read(cif_path)
    aa = get_structure(cif, model=1)

    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for i in range(len(aa)):
        key = (str(aa.chain_id[i]), int(aa.res_id[i]), str(aa.atom_name[i]))
        lookup[key] = aa.coord[i]
    return lookup


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
    centre_atom_idx = token_array.get_annotation("centre_atom_index").astype(np.int64)
    n_token = int(centre_atom_idx.shape[0])
    if n_token == 0:
        return {}

    # Build per-token (q_chain, q_res, q_atom) keys from the query atom array.
    q_keys: list[tuple[str, int, str]] = []
    for tok_i in range(n_token):
        a_idx = centre_atom_idx[tok_i]
        q_keys.append((
            str(atom_array.chain_id[a_idx]),
            int(atom_array.res_id[a_idx]),
            str(atom_array.atom_name[a_idx]),
        ))

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

        lookup = _read_template_atom_lookup(cif_path)

        hit = 0
        for tok_i, (q_chain, q_res, q_atom) in enumerate(q_keys):
            t_chain = chain_map.get(q_chain, q_chain)  # identity default
            key = (t_chain, q_res, q_atom)
            coord = lookup.get(key)
            if coord is None:
                # Some protein residues use CA as centre atom even when CB would
                # be preferred — upstream templates may disagree. Fall through
                # to CB fallback only if this token is a standard protein CA.
                if q_atom == "CA":
                    fallback = lookup.get((t_chain, q_res, "CB"))
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
