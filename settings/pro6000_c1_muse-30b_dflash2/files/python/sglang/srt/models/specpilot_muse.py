"""SPECPILOT_D1: hook dispatcher for the Muse-Glimmer target (muse_glimmer.py, call sites marked ``# SPECPILOT_D1``).

Providers. A stream adds behaviour in its own module, never in this file or in muse_glimmer.py:

    stream  module (python/sglang/srt/models/)   fills
    W       specpilot_muse_w.py                  mlp_forward, qkv_gate, after_load
    A       specpilot_muse_attn.py               attend (attend.global / attend.sliding)
    G       specpilot_muse_glue.py               layer_forward, decoder_stack, qkv_gate (if agreed with W)
    H       specpilot_muse_head.py               logits (H's main path is logits_processor.py)

A provider module defines ``MUSE_HOOKS: dict[str, Callable]``, built once at its import from its own SPECPILOT_*
knobs and EMPTY when they are unset; a provider must import its kernels lazily. A missing module is skipped; a
provider that fails to import raises (never silently stock). The table is built once, at the import of this module;
muse_glimmer.py binds a handler per module in ``__init__`` (``resolve``), so with no provider active every bound
handler is None and the model runs the stock code path (same kernels, same order, same graphs).
``SPECPILOT_D1_HOOKS_OFF=1`` skips provider discovery altogether (all hooks None).

Keys. A per-layer hook key is ``<hook>`` (every layer) or ``<hook>.sliding`` / ``<hook>.global`` (one layer kind;
Muse layer_types = [sliding x3, full x1] x13, so the global layers are 3, 7, ..., 51). For a layer the kind-specific
key of a provider is preferred to its generic key. Several providers on one hook chain in the order W, A, G, H: the
first non-None result wins (``after_load``: every handler runs).

Hooks (``self`` is the Muse module at the call site; T tokens, H hidden 6656, HQ*D 4096, HKV*D 256):

    mlp_forward(mlp, x[T, H]) -> out[T, H] | None
        MuseGlimmerMLP.forward; replaces gate_up_proj -> SiluAndMul -> down_proj.
    qkv_gate(attn, hidden_states[T, H]) -> (q[T, HQ*D], k[T, HKV*D], v[T, HKV*D], gate[T, HQ*D] | None) | None
        MuseGlimmerAttention.forward, before qk_norm / RoPE; gate is the raw output_gate_proj output
        (pre-sigmoid), or None to let the call site compute it as stock.
    attend(attn, q, k, v, gate, forward_batch) -> out[T, HQ*D] | None
        after qk_norm / RoPE, instead of ``self.attn(...)`` + output gate; must store the step's K/V like the
        backend and return the GATED output (sigmoid(gate) * attn), i.e. the o_proj input. When a handler is bound
        for the layer the call site computes ``gate`` before attention (if the layer has an output gate), and a
        None result falls through to the stock attention reusing that gate.
    layer_forward(layer, positions, hidden_states[T, H], forward_batch) -> hidden_states[T, H] | None
        MuseGlimmerDecoderLayer.forward (norm / residual-add fusion point; the whole layer).
    decoder_stack(model, positions, hidden_states[T, H], forward_batch) -> (hidden_states, aux_hidden_states) | None
        MuseGlimmerModel.forward after the embedding (+ embed_norm), instead of the layer loop; returns the hidden
        states BEFORE the final norm and the list of captured aux states (``model.layers_to_capture``, empty list
        if none), for norm fusion across layer boundaries.
    after_load(model) -> None
        end of MuseGlimmerForCausalLM.load_weights (before KV sizing: freed bf16 bytes count); weight re-encoding.
    logits(model, input_ids, hidden_states, forward_batch, aux_hidden_states) -> LogitsProcessorOutput | None
        instead of ``model.logits_processor(input_ids, hidden_states, model.lm_head, forward_batch, aux)``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PER_LAYER_HOOKS = ("mlp_forward", "qkv_gate", "attend", "layer_forward")
MODEL_HOOKS = ("decoder_stack", "after_load", "logits")
HOOKS = PER_LAYER_HOOKS + MODEL_HOOKS
LAYER_KINDS = ("sliding", "global")
_RUN_ALL = frozenset({"after_load"})

# Chain order when several streams fill one hook.
PROVIDER_MODULES: Tuple[Tuple[str, str], ...] = (
    ("W", "sglang.srt.models.specpilot_muse_w"),
    ("A", "sglang.srt.models.specpilot_muse_attn"),
    ("G", "sglang.srt.models.specpilot_muse_glue"),
    ("H", "sglang.srt.models.specpilot_muse_head"),
)

_HOOKS_OFF = os.environ.get("SPECPILOT_D1_HOOKS_OFF", "").strip() not in ("", "0")

Table = Dict[Tuple[str, Optional[str]], Callable]


def layer_kind(*, layer_types: Sequence[str], layer_id: int) -> str:
    return "sliding" if layer_types[layer_id] == "sliding_attention" else "global"


def _valid_keys() -> frozenset:
    keys = set(HOOKS)
    keys.update(f"{h}.{k}" for h in PER_LAYER_HOOKS for k in LAYER_KINDS)
    return frozenset(keys)


def _chain(hook: str, handlers: Sequence[Callable]) -> Callable:
    if len(handlers) == 1:
        return handlers[0]
    handlers = tuple(handlers)
    if hook in _RUN_ALL:

        def run_all(*args, **kwargs):
            for fn in handlers:
                fn(*args, **kwargs)

        return run_all

    def first_result(*args, **kwargs):
        for fn in handlers:
            out = fn(*args, **kwargs)
            if out is not None:
                return out
        return None

    return first_result


def build_table(providers: Iterable[Tuple[str, Dict[str, Callable]]]) -> Table:
    """Map (hook, kind) -> handler; kind is None for model hooks. Pure: used at import and by the tests."""
    valid = _valid_keys()
    providers = [(name, dict(hooks)) for name, hooks in providers]
    for name, hooks in providers:
        bad = sorted(set(hooks) - valid)
        if bad:
            raise ValueError(f"SPECPILOT_D1: provider {name} declares unknown Muse hooks {bad}")
    table: Table = {}
    for hook in MODEL_HOOKS:
        found = [hooks[hook] for _, hooks in providers if hooks.get(hook) is not None]
        if found:
            table[(hook, None)] = _chain(hook, found)
    for hook in PER_LAYER_HOOKS:
        for kind in LAYER_KINDS:
            found = []
            for _, hooks in providers:
                fn = hooks.get(f"{hook}.{kind}")
                if fn is None:
                    fn = hooks.get(hook)
                if fn is not None:
                    found.append(fn)
            if found:
                table[(hook, kind)] = _chain(hook, found)
    return table


def _discover() -> list:
    if _HOOKS_OFF:
        return []
    providers = []
    for name, module_name in PROVIDER_MODULES:
        if importlib.util.find_spec(module_name) is None:
            continue
        hooks = importlib.import_module(module_name).MUSE_HOOKS
        if hooks:
            providers.append((name, hooks))
    return providers


_PROVIDERS = _discover()
_TABLE: Table = build_table(_PROVIDERS)
if _TABLE:
    logger.info(
        "SPECPILOT_D1 Muse hooks active: %s",
        {name: sorted(k for k, v in hooks.items() if v is not None) for name, hooks in _PROVIDERS},
    )


def resolve(hook: str, *, kind: Optional[str] = None, table: Optional[Table] = None) -> Optional[Callable]:
    """The handler bound for ``hook`` (per-layer hooks: ``kind`` is "sliding" or "global"), or None = stock."""
    if hook in PER_LAYER_HOOKS:
        if kind not in LAYER_KINDS:
            raise ValueError(f"SPECPILOT_D1: per-layer hook {hook} needs kind in {LAYER_KINDS}, got {kind!r}")
    elif hook in MODEL_HOOKS:
        if kind is not None:
            raise ValueError(f"SPECPILOT_D1: model hook {hook} takes no kind")
    else:
        raise ValueError(f"SPECPILOT_D1: unknown Muse hook {hook!r}")
    return (_TABLE if table is None else table).get((hook, kind))


def active() -> bool:
    return bool(_TABLE)
