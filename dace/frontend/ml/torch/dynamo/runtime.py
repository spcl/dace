# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime wrapper: calls a compiled SDFG with torch tensors and SymInt arguments supplied by AOTAutograd."""
from typing import Any, Dict, List, Optional

import sympy
import torch

from dace import symbolic
from dace.codegen.compiled_sdfg import CompiledSDFG

from .training import SDFGPairFunction


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

    def __init__(self,
                 csdfg: CompiledSDFG,
                 inputs: List[Any],
                 outputs: List[Any],
                 fixed_arguments: Optional[Dict[str, Any]] = None):
        """
        :param csdfg: The compiled SDFG.
        :param inputs: How the placeholders map to containers and symbols (``InputSpec`` list of the importer).
        :param outputs: How the outputs are produced (``OutputSpec`` list of the importer).
        :param fixed_arguments: Further SDFG arguments passed on every call, e.g., the phase of a joint SDFG and
                                the (``None``) arrays of the other phase.
        """
        # Tells AOTAutograd to pass the inputs as one list. An instance attribute, so that it survives the
        # ``functools.wraps`` wrappers AOTAutograd puts around compiled functions (they copy ``__dict__`` only).
        self._boxed_call = True
        self.csdfg = csdfg
        self.inputs = inputs
        self.outputs = outputs
        self.fixed_arguments = dict(fixed_arguments or {})
        self.name = csdfg.sdfg.name

    def __call__(self, *args) -> List[Any]:
        if len(args) == 1 and isinstance(args[0], list):  # Boxed: graph inputs are tensors and numbers, never lists
            inputs = args[0]
            args = tuple(inputs)
            inputs.clear()
        kwargs: Dict[str, Any] = dict(self.fixed_arguments)
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
        scalars: List[Any] = []  #: (index, buffer) of outputs only known after the call
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
            elif out.kind == 'scalar':
                buf = torch.empty(1, dtype=out.torch_dtype, device=out.device)
                kwargs[out.name] = buf
                scalars.append((len(results), buf))
                results.append(None)
            elif out.kind == 'const':
                results.append(out.value)
            else:
                results.append(None)

        kwargs.update(symvals)
        self.csdfg(**kwargs)
        for index, buf in scalars:
            results[index] = buf.item()
        return results


class DifferentiableGraph:
    """
    Callable returned to Dynamo for a training graph differentiated by DaCe (see
    :meth:`~.backend.DaceBackend._compile_with_dace_autodiff`): it runs the forward SDFG as a PyTorch autograd function
    whose backward runs the backward SDFG.
    """

    def __init__(self, pair, inputs: List[Any], outputs: List[Any]):
        self.pair = pair
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, *args) -> List[Any]:
        if len(args) == 1 and isinstance(args[0], list):  # Boxed calling convention
            args = tuple(args[0])
        call: Dict[str, Any] = {}
        tensors: Dict[str, torch.Tensor] = {}
        for spec in self.inputs:
            if spec.kind == 'sym':
                call[spec.name] = int(args[spec.position])
            elif spec.kind == 'tensor':
                tensor = args[spec.position]
                tensors[spec.name] = tensor
                tensor = tensor.detach()
                call[spec.name] = tensor.reshape(1) if tensor.dim() == 0 else tensor
        results = SDFGPairFunction.apply(self.pair, call, *[tensors[name] for name in self.pair.differentiated])
        # Rank-0 tensors are shape-(1,) containers in the SDFG
        return [r.reshape(()) if len(spec.tshape) == 0 else r for r, spec in zip(results, self.outputs)]
