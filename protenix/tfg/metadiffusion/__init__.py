# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""MetaDiffusion steering / optimisation / exploration layer for Protenix TFG.

Native Protenix port of the ideas in `Boltz-Metadiffusion`: enhanced
sampling during diffusion via collective-variable–based potentials.

Public API (imported on package load so `@register` decorators fire):

    cv                 — collective-variable functions returning
                         ``(value, dvalue/dcoord)``.
    potentials         — `SteeringPotential` (and later: `OptPotential`,
                         `MetadynamicsPotential`) registered into
                         `protenix.tfg.potentials.CLASS_REGISTRY`.
    schema             — translates the user-facing
                         ``metadiffusion: [...]`` YAML/JSON block into
                         the `guidance.terms` mapping consumed by
                         `parse_tfg_config`.
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
