"""One module per experiment; each defines ``apply(params)``.

Rules:
  * never load or modify weights (checkpoints stay as published); draft-only derived copies
    (gathered rows, quantized copies) are allowed if declared and counted in memory;
  * the target's greedy verification decides what is emitted; a proposal change may not
    alter what the target computes, only which tokens it is asked to verify;
  * code inside the captured draft graph (backbone, conv, selector) is baked at capture
    time, so hooks there take effect for the whole server lifetime;
  * modules with ``DIAGNOSTIC = True`` sync the host (traces, dumps): untimed runs only.

Engine-level changes (block/tree shape, verify kernels, accept/commit, scheduling) belong in the
candidate's engine tree (``python/`` of its workspace), not in hooks.
"""
