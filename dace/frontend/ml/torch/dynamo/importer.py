# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
FX graph walker: lowers an ATen FX graph (as produced by AOTAutograd) into a schedule tree and converts it to an SDFG.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import map_arg

from dace import dtypes
from dace.sdfg import SDFG
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from .context import (ConstValue, LoweringContext, SymValue, TensorValue, UnsupportedOpError, Value, dense_strides)
from .symbols import SymbolTable, SymExpr

_SYM_TYPES = (torch.SymInt, torch.SymBool, torch.SymFloat)


@dataclasses.dataclass
class InputSpec:
    kind: str  #: 'tensor' | 'sym'
    name: str  #: container or symbol name
    position: int = -1  #: index in the flat argument list


@dataclasses.dataclass
class OutputSpec:
    kind: str  #: 'tensor' | 'input' | 'sym' | 'const' | 'none'
    name: Optional[str] = None  #: output container name (for 'tensor')
    tshape: Tuple[SymExpr, ...] = ()
    tstrides: Tuple[SymExpr, ...] = ()
    torch_dtype: Optional[torch.dtype] = None
    device: Any = None
    position: int = -1  #: input position (for 'input')
    expr: Any = None  #: symbolic expression (for 'sym')
    value: Any = None  #: constant (for 'const')


@dataclasses.dataclass
class ImportResult:
    sdfg: SDFG
    stree: tn.ScheduleTreeRoot
    inputs: List[InputSpec]
    outputs: List[OutputSpec]
    symbols: Dict[str, Any]


class GraphImporter:
    """Lowers ATen FX graphs. One instance per compiled graph; HOP subgraphs reuse it via :meth:`lower_subgraph`."""

    def __init__(self, options):
        self.options = options

    # ------------------------------------------------------------------ entry points
    def import_graph(self,
                     gm: torch.fx.GraphModule,
                     example_inputs: Sequence[Any],
                     name: str,
                     symbol_names: Optional[Dict[str, str]] = None,
                     input_names: Optional[Sequence[Optional[str]]] = None,
                     return_arrays: bool = False) -> ImportResult:
        """
        Lowers an ATen graph into a schedule tree and converts it to an SDFG.

        :param gm: The functionalized ATen graph (``node.meta['val']`` must be populated).
        :param example_inputs: Fake or real example inputs in placeholder order.
        :param name: Name of the SDFG.
        :param symbol_names: Optional renaming of Dynamo shape symbols (``s77``) to user-facing DaCe symbol names.
        :param input_names: Optional container name per placeholder (``None`` entries keep the FX node name), e.g.
                            the qualified names of the arguments and parameters the placeholders come from.
        :param return_arrays: Write every output into its own ``__return`` (one output) or ``__return_<i>`` container,
                              the convention for SDFGs called from a ``@dace.program``. Symbolic and constant outputs
                              become rank-0 containers.
        """
        from . import ops  # noqa: F401 (registers lowerings)

        symtab = SymbolTable(names=symbol_names)
        storage = _storage_for(example_inputs)
        ctx = LoweringContext(name, symtab, self.options, importer=self, storage=storage)
        self.gm = gm

        inputs = self._bind_placeholders(ctx, gm, example_inputs, input_names)
        output_values = self.walk(ctx, gm)
        if return_arrays:
            outputs = self._return_outputs(ctx, output_values)
        else:
            outputs = self._finalize_outputs(ctx, output_values, inputs)

        arg_names = [spec.name for spec in inputs if spec.kind == 'tensor']
        arg_names += [spec.name for spec in outputs if spec.kind == 'tensor']
        stree = tn.ScheduleTreeRoot(name=name,
                                    containers=ctx.containers,
                                    constants=ctx.constants,
                                    symbols=symtab.symbol_types,
                                    arg_names=arg_names,
                                    children=ctx.root_children)
        if getattr(self.options, 'verbose', False):
            print(stree.as_string())
        sdfg = stree.as_sdfg(validate=getattr(self.options, 'validate', True),
                             simplify=getattr(self.options, 'simplify', True))
        return ImportResult(sdfg, stree, inputs, outputs, dict(symtab.symbols))

    def lower_subgraph(self, ctx: LoweringContext, sub_gm: torch.fx.GraphModule,
                       bindings: Sequence[Value]) -> List[Value]:
        """Lowers a HOP subgraph into the current scope, binding its placeholders to ``bindings`` positionally."""
        placeholders = [n for n in sub_gm.graph.nodes if n.op == 'placeholder']
        if len(placeholders) != len(bindings):
            raise ValueError(f'Subgraph expects {len(placeholders)} arguments, got {len(bindings)}')
        for node, value in zip(placeholders, bindings):
            ctx.env[node] = value
        return self.walk(ctx, sub_gm)

    # ------------------------------------------------------------------ placeholders
    def _bind_placeholders(self,
                           ctx: LoweringContext,
                           gm: torch.fx.GraphModule,
                           example_inputs: Sequence[Any],
                           input_names: Optional[Sequence[Optional[str]]] = None) -> List[InputSpec]:
        specs: List[InputSpec] = []
        position = 0
        for node in gm.graph.nodes:
            if node.op != 'placeholder':
                continue
            val = node.meta.get('val', None)
            if val is None and position < len(example_inputs):
                val = example_inputs[position]
            name = input_names[position] if input_names is not None and input_names[position] else node.name
            value, spec = self._bind_input(ctx, name, val, position)
            ctx.env[node] = value
            if spec is not None:
                specs.append(spec)
            position += 1
        return specs

    def _bind_input(self, ctx: LoweringContext, name: str, val: Any, position: int):
        if isinstance(val, torch.Tensor):
            tv = ctx.add_tensor_like(name, val, transient=False, contiguous=False, source='input')
            return tv, InputSpec('tensor', tv.name, position)
        if isinstance(val, _SYM_TYPES):
            expr = ctx.symtab.to_dace(val)
            if isinstance(expr, int):
                return ConstValue(expr), None
            if expr.is_Symbol:
                return SymValue(expr), InputSpec('sym', expr.name, position)
            # A compound expression as a placeholder: bind all its symbols cannot be inferred, treat as value
            return SymValue(expr), None
        if isinstance(val, (bool, int, float)):
            return ConstValue(val), None
        if val is None:
            return ConstValue(None), None
        raise UnsupportedOpError('placeholder', f'unsupported input type {type(val).__name__} for {name}')

    # ------------------------------------------------------------------ walking
    def walk(self, ctx: LoweringContext, gm: torch.fx.GraphModule) -> List[Value]:
        outputs: List[Value] = []
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                if node not in ctx.env:
                    raise ValueError(f'Unbound placeholder {node.name}')
                continue
            if node.op == 'output':
                args = map_arg(node.args[0], lambda n: ctx.env[n])
                if isinstance(args, (list, tuple)):
                    outputs = list(args)
                else:
                    outputs = [args]
                continue
            if node.op == 'get_attr':
                ctx.env[node] = self._lower_get_attr(ctx, gm, node)
                continue
            if node.op == 'call_function':
                ctx.env[node] = self._lower_call(ctx, node)
                continue
            raise UnsupportedOpError(node.op, f'FX node kind {node.op} ({node.target}) is not supported')
        return outputs

    def _lower_get_attr(self, ctx: LoweringContext, gm: torch.fx.GraphModule, node: torch.fx.Node) -> Value:
        target = node.target
        obj: Any = gm
        for part in str(target).split('.'):
            obj = getattr(obj, part)
        if isinstance(obj, torch.fx.GraphModule):
            return ConstValue(obj)
        if isinstance(obj, torch.Tensor):
            return self._lower_constant_tensor(ctx, node.name, obj)
        return ConstValue(obj)

    def _lower_constant_tensor(self, ctx: LoweringContext, name: str, tensor: torch.Tensor) -> TensorValue:
        """
        Lowers a constant tensor of the graph (e.g., ``torch.tensor([...])`` in the program) to a transient container
        that is also a compile-time constant of the SDFG (``ScheduleTreeRoot.constants``), so it is neither allocated
        nor passed at runtime.
        """
        # The compiler runs under AOTAutograd's fake mode, but the constant is a real tensor whose values are needed
        with unset_fake_temporarily():
            value = tensor.detach().cpu().contiguous().numpy()
        tshape = ctx.symtab.shape(tensor.shape)
        const = ctx.add_array('c_' + name, tshape, tensor.dtype, transient=True, device=tensor.device, source='const')
        ctx.constants[const.name] = (const.desc, value.reshape(const.desc.shape))
        if tensor.dim() == 0 or tensor.is_contiguous():
            return const
        # Downstream views use the constant's torch strides; copy it into a container with that layout
        tv = ctx.add_tensor_like('t_' + name, tensor, source='const')
        ctx.emit_copy(const, tv)
        return tv

    def _lower_call(self, ctx: LoweringContext, node: torch.fx.Node) -> Value:
        from . import ops
        target = node.target
        fn = ops.lookup(target)
        if fn is None and getattr(self.options, 'onnx_fallback', True):
            from .ops import onnx_fallback
            fn = onnx_fallback.lookup(target)
        if fn is None:
            raise UnsupportedOpError(target, 'no lowering registered (and no ONNX fallback available)')
        args = map_arg(node.args, lambda n: ctx.env[n])
        kwargs = map_arg(node.kwargs, lambda n: ctx.env[n])
        return fn(ctx, node, *args, **kwargs)

    # ------------------------------------------------------------------ outputs
    def _finalize_outputs(self, ctx: LoweringContext, values: List[Value], inputs: List[InputSpec]) -> List[OutputSpec]:
        specs: List[OutputSpec] = []
        input_positions = {spec.name: spec.position for spec in inputs if spec.kind == 'tensor'}
        already_returned = set()
        for i, v in enumerate(values):
            if isinstance(v, TensorValue):
                if v.source == 'input' and v.name in input_positions:
                    specs.append(OutputSpec('input', position=input_positions[v.name]))
                    continue
                if v.is_view or v.source is not None or v.name in already_returned:
                    strides = dense_strides(v.tstrides, v.tshape)
                    out = ctx.add_array(f'out_{i}', v.tshape, v.torch_dtype, strides, transient=False, device=v.device)
                    ctx.emit_copy(v, out)
                    v = out
                else:
                    v.desc.transient = False
                already_returned.add(v.name)
                specs.append(OutputSpec('tensor', v.name, v.tshape, v.tstrides, v.torch_dtype, v.device))
            elif isinstance(v, SymValue):
                specs.append(OutputSpec('sym', expr=v.expr))
            elif isinstance(v, ConstValue):
                specs.append(OutputSpec('none') if v.value is None else OutputSpec('const', value=v.value))
            elif v is None:
                specs.append(OutputSpec('none'))
            else:
                raise UnsupportedOpError('output', f'cannot return value of type {type(v).__name__}')
        return specs

    def _return_outputs(self, ctx: LoweringContext, values: List[Value]) -> List[OutputSpec]:
        """Copies every output into a ``__return``/``__return_<i>`` container (see ``return_arrays``)."""
        specs: List[OutputSpec] = []
        for i, v in enumerate(values):
            name = '__return' if len(values) == 1 else f'__return_{i}'
            if isinstance(v, TensorValue):
                strides = dense_strides(v.tstrides, v.tshape)
                out = ctx.add_array(name,
                                    v.tshape,
                                    v.torch_dtype,
                                    strides,
                                    transient=False,
                                    device=v.device,
                                    exact_name=True)
                ctx.emit_copy(v, out)
            elif isinstance(v, (SymValue, ConstValue)) and not (isinstance(v, ConstValue) and v.value is None):
                value = v.expr if isinstance(v, SymValue) else v.value
                dtype = torch.float64 if isinstance(value, float) else torch.bool if isinstance(value, bool) else \
                    torch.int64
                out = ctx.add_array(name, (), dtype, transient=False, exact_name=True)
                ctx.emit_tasklet(f'{name}_set', {}, f'__out = {value}', {'__out': out.memlet()})
            else:
                raise UnsupportedOpError('output', f'cannot return a value of type {type(v).__name__} as an array')
            specs.append(OutputSpec('tensor', out.name, out.tshape, out.tstrides, out.torch_dtype, out.device))
        return specs


def _storage_for(example_inputs: Sequence[Any]) -> dtypes.StorageType:
    for ex in example_inputs:
        if isinstance(ex, torch.Tensor) and ex.device.type == 'cuda':
            return dtypes.StorageType.GPU_Global
    return dtypes.StorageType.Default
