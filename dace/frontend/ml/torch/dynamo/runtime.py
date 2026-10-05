# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime wrapper: calls a compiled SDFG with torch tensors and SymInt arguments supplied by AOTAutograd."""
from typing import Any, Dict, List

import sympy
import torch

from dace import symbolic
from dace.codegen.compiled_sdfg import CompiledSDFG


def _evaluate(expr, symvals: Dict[str, int]) -> int:
    if isinstance(expr, (int, bool)):
        return int(expr)
    if isinstance(expr, sympy.Basic) and expr.is_Integer:
        return int(expr)
    return int(symbolic.evaluate(expr, symvals))


class CompiledGraph:
    """
    Callable returned to AOTAutograd for one captured graph.

    The inputs follow the FX placeholder order (SymInt sizes as Python ints, tensors). They are given either positionally
    or, in AOTAutograd's boxed calling convention, as one list that the call empties so that AOTAutograd can free saved
    activations as early as possible. Outputs are allocated as torch tensors (so torch owns the memory and the device) and passed to
    the SDFG as arguments.
    """

    def __init__(self, csdfg: CompiledSDFG, result):
        # Tells AOTAutograd to pass the inputs as one list. An instance attribute, so that it survives the
        # ``functools.wraps`` wrappers AOTAutograd puts around compiled functions (they copy ``__dict__`` only).
        self._boxed_call = True
        self.csdfg = csdfg
        self.result = result
        self.inputs = result.inputs
        self.outputs = result.outputs
        self.name = result.sdfg.name

    def __call__(self, *args) -> List[Any]:
        if len(args) == 1 and isinstance(args[0], list):  # Boxed: graph inputs are tensors and numbers, never lists
            inputs = args[0]
            args = tuple(inputs)
            inputs.clear()
        kwargs: Dict[str, Any] = {}
        symvals: Dict[str, int] = {}
        for spec in self.inputs:
            if spec.kind == 'sym':
                symvals[spec.name] = int(args[spec.position])
            elif spec.kind == 'tensor':
                t = args[spec.position]
                if t.requires_grad:
                    t = t.detach()
                if t.dim() == 0:
                    t = t.reshape(1)  # rank-0 tensors are shape-(1,) containers in the SDFG
                kwargs[spec.name] = t

        results: List[Any] = []
        for out in self.outputs:
            if out.kind == 'tensor':
                shape = [_evaluate(s, symvals) for s in out.tshape]
                strides = [_evaluate(s, symvals) for s in out.tstrides]
                if len(shape) == 0:
                    buf = torch.empty(1, dtype=out.torch_dtype, device=out.device)
                    kwargs[out.name] = buf
                    results.append(buf.reshape(()))
                    continue
                t = torch.empty_strided(shape, strides, dtype=out.torch_dtype, device=out.device)
                kwargs[out.name] = t
                results.append(t)
            elif out.kind == 'input':
                t = args[out.position]
                results.append(t)
            elif out.kind == 'sym':
                results.append(_evaluate(out.expr, symvals))
            elif out.kind == 'const':
                results.append(out.value)
            else:
                results.append(None)

        self.csdfg(**kwargs, **symvals)
        return results
