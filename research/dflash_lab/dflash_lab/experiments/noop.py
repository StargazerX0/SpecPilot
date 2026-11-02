"""Control: loads the lab core (refine pipeline, step observer, fingerprints) and an
empty refiner. Outputs must be token-identical to unmodified DFlash2."""
from __future__ import annotations

from dflash_lab import core


def apply(params):
    if params:
        raise ValueError(f"noop takes no params, got {params}")
    core.add_refiner(lambda ctx: None)
