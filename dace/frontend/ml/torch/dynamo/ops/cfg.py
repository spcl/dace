# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering of captured control-flow graphs (``dace::cfg``, see :mod:`..cfg.transport`) onto schedule-tree control flow.

Every traced block becomes a labeled segment of a general block (``GBlock``): its torch-level graph is converted to
ATen (with the frontend's decomposition table) and lowered into the segment. CFG variables live in per-block input
containers; an edge copies the block's outputs into the inputs of its successor and jumps to it. Branches are
conditional gotos on the predicate, and returns copy into the operator's outputs and jump to the end of the general
block. Loop-carried integers (e.g., the counters of for loops) are symbols assigned on the edges. Simplification then
raises the gotos into loops and conditionals.
"""

import dataclasses
from typing import Any, Dict, List, Sequence

import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.nn.attention import SDPBackend, sdpa_kernel

from dace import InterstateEdge, dtypes, symbolic
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ..cfg import transport
from ..cfg.transport import BRANCH, GOTO, RETURN, Binding, CfgRecord
from ..context import ConstValue, LoweringContext, SymValue, TensorValue, TupleValue, UnsupportedOpError, as_sym
from ..decompositions import build_decomposition_table
from . import register_lowering
from .control_flow import predicate_expr


def _items(v) -> list:
    return list(v.items) if isinstance(v, TupleValue) else list(v)


def _fake(arg) -> Any:
    """The fake value of an FX argument of the ``dace::cfg`` node."""
    return arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg


def _to_aten(graph: torch.fx.Graph, examples: Sequence[Any], decompositions: Dict) -> torch.fx.GraphModule:
    """Converts a block's torch-level graph to a functional ATen graph with ``node.meta['val']``."""
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    # Inputs that alias (e.g., after ``z = y``) must be distinct tensors: make_fx maps each tensor to one placeholder
    seen = set()
    distinct = []
    for example in examples:
        if isinstance(example, torch.Tensor) and id(example) in seen:
            with example.fake_mode:
                example = torch.empty_strided(
                    example.size(), example.stride(), dtype=example.dtype, device=example.device
                )
        seen.add(id(example))
        distinct.append(example)
    examples = distinct
    # ``.item()`` in a block (e.g., of a float attribute) is a data-dependent symbol, which the fake mode of the frame
    # only allows if it was created for scalar capture
    fake_modes = list({id(e.fake_mode): e.fake_mode for e in examples if isinstance(e, FakeTensor)}.values())
    allowed = [mode.allow_scalar_outputs for mode in fake_modes]
    try:
        for mode in fake_modes:
            mode.allow_scalar_outputs = True
        with sdpa_kernel(SDPBackend.MATH):
            return make_fx(
                torch.func.functionalize(gm, remove="mutations"),
                decomposition_table=decompositions,
                tracing_mode="real",
            )(*examples)
    finally:
        for mode, allow in zip(fake_modes, allowed):
            mode.allow_scalar_outputs = allow


@register_lowering(torch.ops.dace.cfg.default)
def lower_cfg(ctx: LoweringContext, node, cfg_id, tensors, symints):
    record: CfgRecord = transport.lookup(int(cfg_id.value if isinstance(cfg_id, ConstValue) else cfg_id))
    tensors, symints = _items(tensors), _items(symints)
    fake_tensors = [_fake(a) for a in node.args[1]]
    fake_syms = [_fake(a) for a in node.args[2]]
    decompositions = build_decomposition_table(ctx.options.extra_decompositions, ctx.options.native_ops)
    prefix = f"cfg{record.id}"

    def bound(binding: Binding):
        return tensors[binding.index] if binding.kind == "tensor" else symints[binding.index]

    def bound_fake(binding: Binding):
        return fake_tensors[binding.index] if binding.kind == "tensor" else fake_syms[binding.index]

    def own_examples(block_id: int) -> Dict[str, Any]:
        """Symbolic inputs keep the symbols they were traced with (loop counters differ between edges)."""
        block = record.block(block_id)
        return {
            name: example
            for name, example in zip(block.input_names, block.input_examples)
            if not isinstance(example, torch.Tensor)
        }

    # Example (fake) inputs of every block: tensors are propagated from the entry along the edges in discovery order
    examples: Dict[int, Dict[str, Any]] = {}
    for block_id in set(record.entry_successors.values()):
        examples[block_id] = {
            name: bound_fake(record.entry_bindings[name])
            for name in record.block(block_id).input_names
            if name in record.entry_bindings
        }
        examples[block_id].update(own_examples(block_id))
    aten: Dict[int, torch.fx.GraphModule] = {}
    order = sorted(set(record.entry_successors.values()))
    visited = set()
    while order:
        block_id = order.pop(0)
        if block_id in visited:
            continue
        visited.add(block_id)
        block = record.block(block_id)
        inputs = [examples[block_id][name] for name in block.input_names] + [bound_fake(b) for b in block.lifted]
        aten[block_id] = _to_aten(block.graph, inputs, decompositions)
        outputs = [n.meta["val"] for n in _output_nodes(aten[block_id])]
        if block.exit_kind == RETURN:
            continue
        values = dict(zip(block.output_names, outputs[1:] if block.exit_kind == BRANCH else outputs))
        successors = [block.successors] if block.exit_kind == GOTO else list(block.successors.values())
        for successor in successors:
            if successor not in examples:
                examples[successor] = {
                    name: values[name]
                    for name in record.block(successor).input_names
                    if isinstance(values.get(name), torch.Tensor)
                }
                examples[successor].update(own_examples(successor))
            order.append(successor)

    # Containers of the CFG variables (per block) and of the outputs
    inputs: Dict[int, Dict[str, Any]] = {}
    for block_id, values in examples.items():
        inputs[block_id] = {}
        for name, example in values.items():
            if isinstance(example, torch.Tensor):
                inputs[block_id][name] = ctx.add_tensor_like(f"{prefix}_b{block_id}_{name}", example)
            elif ctx.unassigned_unbacked(example) is not None or _is_assigned_unbacked(ctx, example):
                # A loop-carried integer: a symbol assigned on every edge into the block
                unbacked = example.node.expr
                symbol = ctx.symtab.define(unbacked.name, dtypes.int64, nonnegative=_nonnegative(example))
                ctx.assigned_symbols[unbacked.name] = symbol
                inputs[block_id][name] = _Carried(symbol)
            else:  # Other symbolic integers are identical on all incoming edges (blocks are specialized on them)
                inputs[block_id][name] = SymValue(ctx.symtab.to_dace(example))
    vals = node.meta["val"]
    outputs = [ctx.add_tensor_like(f"t_{node.name}_{k}", v) for k, v in enumerate(vals)]

    def label(block_id: int) -> str:
        return f"{prefix}_b{block_id}"

    exit_label = f"{prefix}_exit"

    def pass_values(values: Dict[str, Any], constants: Dict[str, Any], successor: int) -> Dict[str, Any]:
        """
        Copies tensors into the successor's containers; returns the assignments to its loop-carried symbols (to be
        made at once with :func:`_assign_at_once`).
        """
        assignments = {}
        for name, target in inputs[successor].items():
            source = values[name] if name in values else ConstValue(constants[name])
            if isinstance(target, TensorValue) and source is not target:
                ctx.emit_copy(source, target)
            elif isinstance(target, _Carried):
                value = source.expr if isinstance(source, SymValue) else as_sym(source)
                if value != target.symbol:
                    assignments[target.symbol.name] = value
        return assignments

    def goto(successor: int, assignments: Dict[str, Any]) -> None:
        """Jumps to ``successor``, assigning its loop-carried symbols on the way."""
        _assign_at_once(ctx, assignments)
        ctx.emit(tn.GotoNode(target=label(successor)))

    #: Segments that assign symbols on a taken conditional edge before jumping on: (label, assignments, successor)
    trampolines: List[Any] = []

    def branch(predicate, values: Dict[str, Any], constants: Dict[str, Any], successors: Dict[bool, int]) -> None:
        # Tensors are copied before branching (every block has its own containers); symbols are assigned on the edge
        # taken, after the predicate is evaluated
        assignments = {label: pass_values(values, constants, successor) for label, successor in successors.items()}
        target = label(successors[True])
        if assignments[True]:
            target = f"{prefix}_edge{len(trampolines)}"
            trampolines.append((target, assignments[True], successors[True]))
        ctx.emit(tn.StateIfScope(condition=CodeBlock(predicate_expr(predicate)), children=[tn.GotoNode(target=target)]))
        goto(successors[False], assignments[False])

    children: List[tn.ScheduleTreeNode] = [tn.StateLabel(state=f"{prefix}_entry")]
    with ctx.scope(children):
        entry_values = {name: bound(binding) for name, binding in record.entry_bindings.items()}
        if record.entry_condition is not None:
            condition = SymValue(ctx.symtab.to_dace(record.entry_condition))
            branch(condition, entry_values, record.entry_constants, record.entry_successors)
        elif record.entry_predicate is None:  # Into a loop
            successor = record.entry_successors[None]
            goto(successor, pass_values(entry_values, record.entry_constants, successor))
        else:
            branch(bound(record.entry_predicate), entry_values, record.entry_constants, record.entry_successors)

    for block_id in sorted(aten):
        block = record.block(block_id)
        children.append(tn.StateLabel(state=label(block_id)))
        with ctx.scope(children):
            bindings = [_binding(inputs[block_id][name]) for name in block.input_names] + [
                bound(b) for b in block.lifted
            ]
            results = ctx.importer.lower_subgraph(ctx, aten[block_id], bindings)
            if block.exit_kind == RETURN:
                for result, out in zip(results, outputs):
                    if not isinstance(result, TensorValue):
                        raise UnsupportedOpError(node.target, "captured control flow returns a non-tensor value")
                    ctx.emit_copy(result, out)
                ctx.emit(tn.GotoNode(target=exit_label))
            elif block.exit_kind == BRANCH:
                branch(results[0], dict(zip(block.output_names, results[1:])), block.output_constants, block.successors)
            else:
                goto(
                    block.successors,
                    pass_values(dict(zip(block.output_names, results)), block.output_constants, block.successors),
                )
    for target, assignments, successor in trampolines:
        children.append(tn.StateLabel(state=target))
        with ctx.scope(children):
            goto(successor, assignments)
    children.append(tn.StateLabel(state=exit_label))
    ctx.emit(tn.GBlock(children=children))
    return TupleValue(outputs)


@dataclasses.dataclass
class _Carried:
    """A symbolic block input assigned on the edges into the block (e.g., a loop counter)."""

    symbol: Any


def _assign_at_once(ctx: LoweringContext, assignments: Dict[str, Any]) -> None:
    """
    Assigns symbols as if at once, each value reading the symbols before any assignment. Assignments on one
    interstate edge may not read each other, so values that read an assigned symbol go through temporaries on a
    first edge, and the temporaries are copied on a second one.
    """
    reads = {name: {str(s) for s in sympy.sympify(value).free_symbols} for name, value in assignments.items()}
    first, second = {}, {}
    for name, value in assignments.items():
        # A symbol may read itself (``i = i + 1``), but not one another assignment of the edge changes
        if any(name in read for other, read in reads.items() if other != name):
            temporary = f"{name}_next"
            ctx.symtab.define(temporary, ctx.symtab.symbols[name].dtype if name in ctx.symtab.symbols else dtypes.int64)
            first[temporary] = value
            second[name] = temporary
        else:
            first[name] = value
    for batch in (first, second):
        for name, value in batch.items():
            code = value if isinstance(value, str) else symbolic.symstr(value, cpp_mode=False)
            ctx.emit(tn.AssignNode(name=name, value=CodeBlock(code), edge=InterstateEdge(assignments={name: code})))
        if batch is first and second:
            ctx.emit(tn.StateBoundaryNode())


def _binding(value: Any) -> Any:
    return SymValue(value.symbol) if isinstance(value, _Carried) else value


def _is_assigned_unbacked(ctx: LoweringContext, example: Any) -> bool:
    return (
        isinstance(example, torch.SymInt)
        and isinstance(example.node.expr, sympy.Symbol)
        and bool(free_unbacked_symbols(example))
        and ctx.defines_unbacked(example.node.expr.name)
    )


def _nonnegative(example: torch.SymInt) -> bool:
    value_range = example.node.shape_env.var_to_range.get(example.node.expr)
    return value_range is not None and bool(value_range.lower >= 0)


def _output_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    output = next(n for n in gm.graph.nodes if n.op == "output")
    args = output.args[0]
    return list(args) if isinstance(args, (list, tuple)) else [args]
