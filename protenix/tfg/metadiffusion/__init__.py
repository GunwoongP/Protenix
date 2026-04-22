# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""MetaDiffusion steering / optimisation / exploration layer for Protenix TFG.

Native Protenix port of the ideas in `Boltz-Metadiffusion`: enhanced
sampling during diffusion via collective-variable–based potentials.

Public API (imported on package load so `@register` decorators fire):

    cv             — collective-variable functions returning
                     ``(value, dvalue/dcoord)``.  19 CVs cover the
                     Boltz-Metadiffusion default set (rg, distance,
                     inter_chain, rmsd, drmsd, d_tm, pair_drmsd,
                     native_contacts, max_diameter, sasa,
                     asphericity, min_distance, angle, dihedral,
                     hbond_count, salt_bridges, coordination,
                     contact_order, rmsf).
    potentials     — `SteeringPotential` (harmonic + gaussian),
                     `OptPotential` (min/max), and
                     `MetadynamicsPotential` (Gaussian-hill explore,
                     well-tempered). All three are registered into
                     ``protenix.tfg.potentials.CLASS_REGISTRY`` via
                     the ``@register`` decorator.
    schema         — translates the user-facing
                     ``metadiffusion: [...]`` YAML/JSON block into
                     the ``guidance.terms`` mapping consumed by
                     ``parse_tfg_config``. Honours steer, opt and
                     explore modes plus region-based atom selection.
    gradient_mods  — Phase-D gradient post-processors
                     (``GradientScaler`` + ``GradientProjector``) with
                     configurable ``modifier_order``.
"""

from protenix.tfg.metadiffusion import cv, potentials, schema  # noqa: F401
from protenix.tfg.metadiffusion.potentials import enable_diagnostics  # noqa: F401
from protenix.tfg.metadiffusion.schema import (  # noqa: F401
    build_metadiffusion_features,
    parse_metadiffusion_block,
)

__all__ = [
    "cv",
    "potentials",
    "schema",
    "enable_diagnostics",
    "build_metadiffusion_features",
    "parse_metadiffusion_block",
]
