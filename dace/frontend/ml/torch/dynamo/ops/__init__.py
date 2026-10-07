# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Registry of ATen/HOP lowerings.

A lowering has the signature ``lower(ctx: LoweringContext, node: torch.fx.Node, *args, **kwargs) -> Value``, where
``args``/``kwargs`` are the FX arguments with nodes replaced by their :mod:`context` values. The FX node gives access
to ``node.meta['val']`` (FakeTensor/SymInt) which is the authoritative source for output shapes, strides, and dtypes.
"""

from typing import Any, Callable, Dict, Optional

_LOWERINGS: Dict[Any, Callable] = {}


def register_lowering(*targets):
    """Registers a lowering for the given targets (``OpOverload``, ``OpOverloadPacket``, HOP, or Python callable)."""

    def decorator(fn: Callable) -> Callable:
        for target in targets:
            _LOWERINGS[target] = fn
        return fn

    return decorator


def lookup(target) -> Optional[Callable]:
    fn = _LOWERINGS.get(target)
    if fn is None and hasattr(target, "overloadpacket"):
        fn = _LOWERINGS.get(target.overloadpacket)
    return fn


def registered_targets():
    return list(_LOWERINGS.keys())


def resolve(*names):
    """
    Resolves operator names such as ``'aten.sym_size.default'`` under ``torch.ops`` (or ``'torch.sym_max'``),
    skipping names that do not exist in the installed torch version.
    """
    import torch

    result = []
    for name in names:
        obj = torch if name.startswith("torch.") else torch.ops
        parts = name.split(".")[1:] if name.startswith("torch.") else name.split(".")
        try:
            for part in parts:
                obj = getattr(obj, part)
        except (AttributeError, RuntimeError):
            continue
        result.append(obj)
    return result


# Import lowering modules so that they register themselves
from . import symint, pointwise, view, linalg, reduction, control_flow, nn, indexing, cfg  # noqa: E402,F401
