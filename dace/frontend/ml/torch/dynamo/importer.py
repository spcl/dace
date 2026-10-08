# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
FX graph walker: lowers an ATen FX graph (as produced by AOTAutograd) into a schedule tree and converts it to an SDFG.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sympy
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import map_arg

from dace import dtypes
from dace import symbolic as dsym
from dace.properties import CodeBlock
from dace.sdfg import SDFG
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from . import joint as jt
from .context import ConstValue, LoweringContext, SymValue, TensorValue, UnsupportedOpError, Value, dense_strides
from .symbols import SymbolTable, SymExpr

_SYM_TYPES = (torch.SymInt, torch.SymBool, torch.SymFloat)


@dataclasses.dataclass
class InputSpec:
    kind: str  #: 'tensor' | 'sym'
    name: str  #: container or symbol name
    position: int = -1  #: index in the flat argument list


@dataclasses.dataclass
class OutputSpec:
    kind: str  #: 'tensor' | 'input' | 'sym' | 'scalar' | 'const' | 'none'
    name: Optional[str] = None  #: output container name (for 'tensor' and 'scalar', a rank-0 container)
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


@dataclasses.dataclass
class PhaseInterface:
    """Calling convention of one phase of a joint SDFG (see :mod:`.joint`)."""

    phase: int
    inputs: List[InputSpec]
    outputs: List[OutputSpec]


@dataclasses.dataclass
class JointImportResult:
    sdfg: SDFG
    stree: tn.ScheduleTreeRoot
    forward: PhaseInterface
    backward: PhaseInterface
    symbols: Dict[str, Any]


class GraphImporter:
    """Lowers ATen FX graphs. One instance per compiled graph; HOP subgraphs reuse it via :meth:`lower_subgraph`."""

    def __init__(self, options):
        self.options = options

    # ------------------------------------------------------------------ entry points
    def import_graph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        name: str,
        symbol_names: Optional[Dict[str, str]] = None,
        input_names: Optional[Sequence[Optional[str]]] = None,
        return_arrays: bool = False,
    ) -> ImportResult:
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
            outputs, _ = self._finalize_outputs(ctx, output_values, inputs)

        arg_names = [spec.name for spec in inputs if spec.kind == "tensor"]
        arg_names += [spec.name for spec in outputs if spec.kind == "tensor"]
        stree = tn.ScheduleTreeRoot(
            name=name,
            containers=ctx.containers,
            constants=ctx.constants,
            symbols=symtab.symbol_types,
            arg_names=arg_names,
            children=ctx.root_children,
        )
        if getattr(self.options, "verbose", False):
            print(stree.as_string())
        sdfg = stree.as_sdfg(
            validate=getattr(self.options, "validate", True), simplify=getattr(self.options, "simplify", True)
        )
        return ImportResult(sdfg, stree, inputs, outputs, dict(symtab.symbols))

    def import_joint(
        self,
        plan: jt.JointPlan,
        name: str,
        symbol_names: Optional[Dict[str, str]] = None,
        input_names: Optional[Sequence[Optional[str]]] = None,
    ) -> JointImportResult:
        """
        Lowers a joint forward/backward graph into one SDFG with two phases, selected by the
        :data:`~.joint.PHASE_SYMBOL` symbol (see :mod:`.joint`). Each joint node is lowered in the phases that compute
        it. Saved values are containers both phases share: written by the forward phase (and returned to
        AOTAutograd, which keeps them) and read by the backward phase. Containers that only one phase uses are
        optional arguments.

        :param plan: The joint graph and its cut.
        :param name: Name of the SDFG.
        :param symbol_names: Optional renaming of Dynamo shape symbols to DaCe symbol names.
        :param input_names: Optional container name per forward placeholder (primal).
        """
        from . import ops  # noqa: F401 (registers lowerings)

        nodes = plan.nodes()
        forward_placeholders = [nodes[n] for n in jt.placeholder_names(plan.forward)]
        backward_placeholders = [nodes[n] for n in jt.placeholder_names(plan.backward)]
        symtab = SymbolTable(names=symbol_names)
        ctx = LoweringContext(
            name,
            symtab,
            self.options,
            importer=self,
            storage=_storage_for([n.meta.get("val") for n in forward_placeholders]),
        )
        self.gm = plan.joint
        if input_names is not None and len(input_names) != len(forward_placeholders):
            input_names = None

        # Forward phase: primals -> forward outputs, saved values
        forward_inputs: List[InputSpec] = []
        for position, node in enumerate(forward_placeholders):
            container = input_names[position] if input_names is not None and input_names[position] else node.name
            value, spec = self._bind_input(ctx, container, node.meta.get("val"), position)
            ctx.env[node] = value
            if spec is not None:
                forward_inputs.append(spec)
        primals = dict(ctx.env)
        forward_children: List[tn.ScheduleTreeNode] = []
        with ctx.scope(forward_children):
            self.walk(ctx, plan.joint, plan.forward_nodes())
            returned = [
                ctx.env[nodes[v.name]] if isinstance(v, torch.fx.Node) else ConstValue(v)
                for v in jt.flat_outputs(plan.forward)
            ]
            forward_outputs, finals = self._finalize_outputs(ctx, returned, forward_inputs)
        saved_values = {node.name: value for node, value in primals.items()}
        saved_values.update(
            {v.name: final for v, final in zip(jt.flat_outputs(plan.forward), finals) if isinstance(v, torch.fx.Node)}
        )

        # Backward phase: saved values, tangents -> gradients. Everything else it needs is recomputed.
        ctx.env = {}
        backward_inputs: List[InputSpec] = []
        for position, node in enumerate(backward_placeholders):
            if node.name in plan.saved:
                value, spec = self._bind_saved(saved_values[node.name], position)
            else:  # Tangent
                value, spec = self._bind_input(ctx, node.name, node.meta.get("val"), position)
            ctx.env[node] = value
            if spec is not None:
                backward_inputs.append(spec)
        gradients = jt.flat_outputs(plan.joint)[plan.num_fwd_outputs :]
        if len(gradients) != len(jt.flat_outputs(plan.backward)):
            raise UnsupportedOpError("joint", "the backward graph does not return one value per gradient")
        backward_children: List[tn.ScheduleTreeNode] = []
        with ctx.scope(backward_children):
            self.walk(ctx, plan.joint, plan.backward_nodes())
            returned = [ctx.env[v] if isinstance(v, torch.fx.Node) else ConstValue(v) for v in gradients]
            backward_outputs, _ = self._finalize_outputs(ctx, returned, backward_inputs)

        forward_args = _argument_names(forward_inputs, forward_outputs)
        backward_args = _argument_names(backward_inputs, backward_outputs)
        for container in set(forward_args) ^ set(backward_args):
            ctx.containers[container].optional = True
        arg_names = forward_args + [a for a in backward_args if a not in forward_args]
        phase = symtab.get(jt.PHASE_SYMBOL)
        stree = tn.ScheduleTreeRoot(
            name=name,
            containers=ctx.containers,
            constants=ctx.constants,
            symbols=symtab.symbol_types,
            arg_names=arg_names,
            children=[
                tn.IfScope(condition=CodeBlock(f"{phase} == {jt.FORWARD}"), children=forward_children),
                tn.ElseScope(children=backward_children),
            ],
        )
        if getattr(self.options, "verbose", False):
            print(stree.as_string())
        sdfg = stree.as_sdfg(
            validate=getattr(self.options, "validate", True), simplify=getattr(self.options, "simplify", True)
        )
        return JointImportResult(
            sdfg,
            stree,
            PhaseInterface(jt.FORWARD, forward_inputs, forward_outputs),
            PhaseInterface(jt.BACKWARD, backward_inputs, backward_outputs),
            dict(symtab.symbols),
        )

    @staticmethod
    def _bind_saved(value: Value, position: int) -> Tuple[Value, Optional[InputSpec]]:
        """Binds a value saved by the forward phase as an input of the backward phase."""
        if isinstance(value, TensorValue):
            return dataclasses.replace(value, source="input"), InputSpec("tensor", value.name, position)
        if isinstance(value, SymValue):
            if not value.expr.is_Symbol:  # An expression over symbols that are saved as well
                return value, None
            return value, InputSpec("sym", value.expr.name, position)
        if isinstance(value, ConstValue):
            return value, None
        raise UnsupportedOpError("joint", f"cannot save a value of type {type(value).__name__}")

    def lower_subgraph(
        self, ctx: LoweringContext, sub_gm: torch.fx.GraphModule, bindings: Sequence[Value]
    ) -> List[Value]:
        """Lowers a HOP subgraph into the current scope, binding its placeholders to ``bindings`` positionally."""
        placeholders = [n for n in sub_gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != len(bindings):
            raise ValueError(f"Subgraph expects {len(placeholders)} arguments, got {len(bindings)}")
        for node, value in zip(placeholders, bindings):
            ctx.env[node] = value
        return self.walk(ctx, sub_gm)

    # ------------------------------------------------------------------ placeholders
    def _bind_placeholders(
        self,
        ctx: LoweringContext,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        input_names: Optional[Sequence[Optional[str]]] = None,
    ) -> List[InputSpec]:
        specs: List[InputSpec] = []
        position = 0
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val", None)
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
            tv = ctx.add_tensor_like(name, val, transient=False, contiguous=False, source="input")
            return tv, InputSpec("tensor", tv.name, position)
        if isinstance(val, _SYM_TYPES):
            expr = ctx.symtab.to_dace(val)
            if isinstance(expr, int):
                return ConstValue(expr), None
            if expr.is_Symbol:
                return SymValue(expr), InputSpec("sym", expr.name, position)
            # A compound expression as a placeholder: bind all its symbols cannot be inferred, treat as value
            return SymValue(expr), None
        if isinstance(val, (bool, int, float)):
            return ConstValue(val), None
        if val is None:
            return ConstValue(None), None
        raise UnsupportedOpError("placeholder", f"unsupported input type {type(val).__name__} for {name}")

    # ------------------------------------------------------------------ walking
    def walk(
        self, ctx: LoweringContext, gm: torch.fx.GraphModule, include: Optional[Set[torch.fx.Node]] = None
    ) -> List[Value]:
        """
        Lowers the nodes of ``gm`` in order and returns the values of its outputs.

        :param include: If given, only lowers these nodes (and returns no outputs).
        """
        outputs: List[Value] = []
        for node in gm.graph.nodes:
            if include is not None and (node not in include or node.op in ("placeholder", "output")):
                continue
            if node.op == "placeholder":
                if node not in ctx.env:
                    raise ValueError(f"Unbound placeholder {node.name}")
                continue
            if node.op == "output":
                args = map_arg(node.args[0], lambda n: ctx.env[n])
                if isinstance(args, (list, tuple)):
                    outputs = list(args)
                else:
                    outputs = [args]
                continue
            if node.op == "get_attr":
                ctx.env[node] = self._lower_get_attr(ctx, gm, node)
                continue
            if node.op == "call_function":
                ctx.env[node] = self._lower_call(ctx, node)
                _check_data_dependent_sizes(ctx, node)
                continue
            raise UnsupportedOpError(node.op, f"FX node kind {node.op} ({node.target}) is not supported")
        return outputs

    def _lower_get_attr(self, ctx: LoweringContext, gm: torch.fx.GraphModule, node: torch.fx.Node) -> Value:
        target = node.target
        obj: Any = gm
        for part in str(target).split("."):
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
        const = ctx.add_array("c_" + name, tshape, tensor.dtype, transient=True, device=tensor.device, source="const")
        ctx.constants[const.name] = (const.desc, value.reshape(const.desc.shape))
        if tensor.dim() == 0 or tensor.is_contiguous():
            return const
        # Downstream views use the constant's torch strides; copy it into a container with that layout
        tv = ctx.add_tensor_like("t_" + name, tensor, source="const")
        ctx.emit_copy(const, tv)
        return tv

    def _lower_call(self, ctx: LoweringContext, node: torch.fx.Node) -> Value:
        from . import ops

        target = node.target
        fn = ops.lookup(target)
        if fn is None and getattr(self.options, "onnx_fallback", True):
            from .ops import onnx_fallback

            fn = onnx_fallback.lookup(target)
        if fn is None:
            raise UnsupportedOpError(target, "no lowering registered (and no ONNX fallback available)")
        args = map_arg(node.args, lambda n: ctx.env[n])
        kwargs = map_arg(node.kwargs, lambda n: ctx.env[n])
        return fn(ctx, node, *args, **kwargs)

    # ------------------------------------------------------------------ outputs
    def _finalize_outputs(
        self, ctx: LoweringContext, values: List[Value], inputs: List[InputSpec]
    ) -> Tuple[List[OutputSpec], List[Value]]:
        """
        Makes the outputs SDFG arguments: inputs are returned as-is, views, aliases, and constants are copied into new
        containers, and other transients become non-transient.

        :return: The output specifications and the value each output is read from after the SDFG returns.
        """
        specs: List[OutputSpec] = []
        finals: List[Value] = []
        input_positions = {spec.name: spec.position for spec in inputs if spec.kind == "tensor"}
        already_returned = set()
        for i, v in enumerate(values):
            finals.append(v)
            if isinstance(v, TensorValue):
                if v.source == "input" and v.name in input_positions:
                    specs.append(OutputSpec("input", position=input_positions[v.name]))
                    continue
                if v.is_view or v.source is not None or v.name in already_returned:
                    strides = dense_strides(v.tstrides, v.tshape)
                    out = ctx.add_array(f"out_{i}", v.tshape, v.torch_dtype, strides, transient=False, device=v.device)
                    ctx.emit_copy(v, out)
                    v = finals[-1] = out
                else:
                    v.desc.transient = False
                already_returned.add(v.name)
                specs.append(OutputSpec("tensor", v.name, v.tshape, v.tstrides, v.torch_dtype, v.device))
            elif isinstance(v, SymValue) and _data_dependent_symbols(ctx, v.expr):
                # Only the SDFG knows the value: return it through a rank-0 container
                dtype = _SCALAR_TORCH_DTYPES.get(_scalar_type(v.expr), torch.int64)
                out = ctx.add_array(f"out_{i}", (), dtype, transient=False)
                ctx.emit_tasklet(f"out_{i}_set", {}, f"__out = {v.expr}", {"__out": out.memlet()})
                specs.append(OutputSpec("scalar", out.name, torch_dtype=dtype, device=out.device))
            elif isinstance(v, SymValue):
                specs.append(OutputSpec("sym", expr=v.expr))
            elif isinstance(v, ConstValue):
                specs.append(OutputSpec("none") if v.value is None else OutputSpec("const", value=v.value))
            elif v is None:
                specs.append(OutputSpec("none"))
            else:
                raise UnsupportedOpError("output", f"cannot return value of type {type(v).__name__}")
            if isinstance(v, TensorValue) and _data_dependent_symbols(ctx, *v.tshape, *v.tstrides):
                raise UnsupportedOpError(
                    "output",
                    f"the shape {v.tshape} of output {i} depends on data "
                    f"({', '.join(sorted(_data_dependent_symbols(ctx, *v.tshape)))}); data-dependent output shapes "
                    "are not supported yet",
                )
        return specs, finals

    def _return_outputs(self, ctx: LoweringContext, values: List[Value]) -> List[OutputSpec]:
        """Copies every output into a ``__return``/``__return_<i>`` container (see ``return_arrays``)."""
        specs: List[OutputSpec] = []
        for i, v in enumerate(values):
            name = "__return" if len(values) == 1 else f"__return_{i}"
            if isinstance(v, TensorValue):
                strides = dense_strides(v.tstrides, v.tshape)
                out = ctx.add_array(
                    name, v.tshape, v.torch_dtype, strides, transient=False, device=v.device, exact_name=True
                )
                ctx.emit_copy(v, out)
            elif isinstance(v, (SymValue, ConstValue)) and not (isinstance(v, ConstValue) and v.value is None):
                value = v.expr if isinstance(v, SymValue) else v.value
                dtype = (
                    torch.float64
                    if isinstance(value, float)
                    else torch.bool
                    if isinstance(value, bool)
                    else torch.int64
                )
                out = ctx.add_array(name, (), dtype, transient=False, exact_name=True)
                ctx.emit_tasklet(f"{name}_set", {}, f"__out = {value}", {"__out": out.memlet()})
            else:
                raise UnsupportedOpError("output", f"cannot return a value of type {type(v).__name__} as an array")
            specs.append(OutputSpec("tensor", out.name, out.tshape, out.tstrides, out.torch_dtype, out.device))
        return specs


_SCALAR_TORCH_DTYPES = {dtypes.float64: torch.float64, dtypes.bool_: torch.bool, dtypes.int64: torch.int64}


def _data_dependent_symbols(ctx: LoweringContext, *exprs) -> Set[str]:
    """Names of the symbols in ``exprs`` that the program assigns from data (``.item()``)."""
    assigned = {symbol.name for symbol in ctx.assigned_symbols.values()}
    return {s.name for e in exprs if not isinstance(e, int) for s in e.free_symbols} & assigned


def _scalar_type(expr) -> dtypes.typeclass:
    if isinstance(expr, sympy.Basic) and (expr.is_Relational or expr.is_Boolean):
        return dtypes.bool_
    types = {s.dtype for s in expr.free_symbols if isinstance(s, dsym.symbol)}
    if dtypes.float64 in types or (isinstance(expr, sympy.Basic) and expr.has(sympy.Float)):
        return dtypes.float64
    return dtypes.int64


def _check_data_dependent_sizes(ctx: LoweringContext, node: torch.fx.Node) -> None:
    """Raises if the value of ``node`` has a data-dependent size that no lowering assigned (e.g., ``nonzero``)."""
    val = node.meta.get("val", None)
    if val is None:
        return
    missing = [s for s in free_unbacked_symbols(val) if not ctx.defines_unbacked(s.name)]
    if missing:
        raise UnsupportedOpError(
            node.target,
            f"the result has a data-dependent size ({', '.join(sorted(s.name for s in missing))}), "
            "which is not supported yet",
        )


def _argument_names(inputs: List[InputSpec], outputs: List[OutputSpec]) -> List[str]:
    names = [spec.name for spec in inputs if spec.kind == "tensor"]
    return names + [spec.name for spec in outputs if spec.kind == "tensor" and spec.name not in names]


def _storage_for(example_inputs: Sequence[Any]) -> dtypes.StorageType:
    for ex in example_inputs:
        if isinstance(ex, torch.Tensor) and ex.device.type == "cuda":
            return dtypes.StorageType.GPU_Global
    return dtypes.StorageType.Default
