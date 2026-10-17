# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering of captured control-flow graphs (``dace::cfg``, see :mod:`..cfg.transport`) onto schedule-tree control flow.

Every traced block becomes a labeled segment of a general block (``GBlock``): its torch-level graph is converted to
ATen (with the frontend's decomposition table) and lowered into the segment. Every CFG variable lives in one container
(per shape, stride and dtype it takes), which the blocks read and write; an edge only copies when a block produced a
new tensor for the variable (a value passed on unchanged, or written in place, costs nothing) and jumps to the
successor. Branches are conditional gotos on the predicate, and returns copy into the operator's outputs and jump to
the end of the general block. Loop-carried integers (e.g., the counters of for loops) are symbols assigned on the
edges. Simplification then raises the gotos into loops and conditionals.

For that raising, a loop has to be entered at the block that tests its condition. Blocks are shared between the
specializations the capture traced (e.g., the first iteration of a loop, in which a buffer is still ``None``, and the
later ones), so a loop may be entered in the middle of its body; :class:`_Structure` then peels the blocks between the
entry and the test for the entering path (the blocks are lowered twice, the capture traced them once).
"""

import contextlib
import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any

import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensorMode, dispatch_functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.nn.attention import SDPBackend, sdpa_kernel

from dace import InterstateEdge, dtypes, symbolic
from dace.frontend.ml.torch.dynamo.cfg import transport
from dace.frontend.ml.torch.dynamo.cfg.transport import BRANCH, GOTO, RETURN, Binding, CfgRecord
from dace.frontend.ml.torch.dynamo.context import (
    ConstValue,
    LoweringContext,
    SymValue,
    TensorValue,
    TupleValue,
    UnsupportedOpError,
    as_sym,
)
from dace.frontend.ml.torch.dynamo.decompositions import build_decomposition_table
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from . import register_lowering
from .control_flow import predicate_expr

#: Key of the entry node of a :class:`_Structure`
ENTRY = ("entry", 0)


def _items(v) -> list:
    return list(v.items) if isinstance(v, TupleValue) else list(v)


def _fake(arg) -> Any:
    """The fake value of an FX argument of the ``dace::cfg`` node."""
    return arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg


def _to_aten(graph: torch.fx.Graph, examples: Sequence[Any], decompositions: dict) -> torch.fx.GraphModule:
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
    # only allows if it was created for scalar capture. The graph is replayed inside that fake mode: factory functions
    # whose only arguments are symbolic integers (``scalar_tensor(n)``, ``zeros((n,))``) have no fake tensor to
    # dispatch on and would otherwise run real kernels, which need concrete sizes. Mutations are removed by
    # ``FunctionalTensorMode`` (as AOTAutograd does), not by ``torch.func.functionalize``, whose ``*_copy`` view
    # replacements specialize data-dependent sizes (``slice_copy`` inside ``linear`` on a ragged chunk).
    fake_modes = list({id(e.fake_mode): e.fake_mode for e in examples if isinstance(e, FakeTensor)}.values())
    allowed = [mode.allow_scalar_outputs for mode in fake_modes]
    try:
        for mode in fake_modes:
            mode.allow_scalar_outputs = True
        with sdpa_kernel(SDPBackend.MATH), fake_modes[0] if fake_modes else contextlib.nullcontext():
            gm = make_fx(
                dispatch_functionalize(gm, FunctionalTensorMode()),
                decomposition_table=decompositions,
                tracing_mode="real",
            )(*examples)
    finally:
        for mode, allow in zip(fake_modes, allowed):
            mode.allow_scalar_outputs = allow
    # A mutated input is not written back (functionalization syncs only the outputs; a block has no caller that
    # observes its inputs, the new value reaches the successors as a block output), so the producing operator (e.g.,
    # ``slice_scatter`` for ``out[i:i+c] = v``) can update the variable's container in place instead of copying it.
    # Write-backs that other paths may still emit are dropped for the same reason.
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.copy_.default
            and isinstance(node.args[0], torch.fx.Node)
            and node.args[0].op == "placeholder"
            and not node.users
        ):
            gm.graph.erase_node(node)
    gm.recompile()
    return gm


# ---------------------------------------------------------------------------------------------- structure
class _Structure:
    """
    The graph of segments to lower: one per block, plus copies of blocks on paths that enter a loop away from the
    block that tests its condition (see the module documentation). Nodes are keyed by ``(block id, copy)``;
    :data:`ENTRY` stands for the frame's entry into the graph.
    """

    def __init__(self, record: CfgRecord):
        self.record = record
        self.nodes: dict[tuple, dict[Any, tuple]] = {}  #: Successor keys by edge label (``None`` for a goto)
        self._copies = itertools.count(1)
        self.nodes[ENTRY] = {label: (block_id, 0) for label, block_id in record.entry_successors.items()}
        for block in record.blocks:
            if block.exit_kind == GOTO:
                self.nodes[(block.id, 0)] = {None: (block.successors, 0)}
            elif block.exit_kind == BRANCH:
                self.nodes[(block.id, 0)] = {label: (s, 0) for label, s in block.successors.items()}
            else:
                self.nodes[(block.id, 0)] = {}
        for _ in range(len(record.blocks) + 1):  # Each round peels one loop
            if not self._peel_one():
                break

    def block(self, key: tuple):
        return self.record.block(key[0])

    def successors(self, key: tuple) -> list[tuple]:
        return list(self.nodes[key].values())

    def order(self) -> list[tuple]:
        """The nodes to lower (all but the entry), in a deterministic order: by block id, then copy."""
        return sorted((key for key in self.nodes if key != ENTRY), key=lambda k: (k[0], k[1]))

    # ------------------------------------------------------------------------------------------ analysis
    def _dominators(self) -> dict[tuple, set[tuple]]:
        nodes = list(self.nodes)
        predecessors: dict[tuple, list[tuple]] = {n: [] for n in nodes}
        for n in nodes:
            for s in self.successors(n):
                predecessors[s].append(n)
        dominators = {n: set(nodes) for n in nodes}
        dominators[ENTRY] = {ENTRY}
        changed = True
        while changed:
            changed = False
            for n in nodes:
                if n == ENTRY:
                    continue
                preds = [dominators[p] for p in predecessors[n]]
                new = (set.intersection(*preds) if preds else set()) | {n}
                if new != dominators[n]:
                    dominators[n] = new
                    changed = True
        return dominators

    def _loop(self, header: tuple, latches: list[tuple]) -> set[tuple]:
        """The natural loop of ``header`` with the back edges from ``latches``."""
        predecessors: dict[tuple, list[tuple]] = {}
        for n in self.nodes:
            for s in self.successors(n):
                predecessors.setdefault(s, []).append(n)
        body = {header}
        work = list(latches)
        while work:
            n = work.pop()
            if n in body:
                continue
            body.add(n)
            work.extend(predecessors.get(n, []))
        return body

    def _peel_one(self) -> bool:
        """
        Finds a loop whose only exiting block is not its header and peels the blocks between the header and that
        block for the paths entering the loop. Returns whether a loop was peeled.
        """
        dominators = self._dominators()
        headers: dict[tuple, list[tuple]] = {}
        for n in self.nodes:
            for s in self.successors(n):
                if s in dominators[n]:
                    headers.setdefault(s, []).append(n)
        for header, latches in sorted(headers.items(), key=lambda item: (item[0][0], item[0][1])):
            loop = self._loop(header, latches)
            exits = {n for n in loop if any(s not in loop for s in self.successors(n))}
            if header in exits or len(exits) != 1:
                continue  # Entered at its test (raised as a loop), or a shape the raising does not lift anyway
            (test,) = exits
            # Blocks between the header and the test, for the entering paths
            prefix = {header}
            work = [header]
            while work:
                n = work.pop()
                for s in self.successors(n):
                    if s in loop and s != test and s not in prefix:
                        prefix.add(s)
                        work.append(s)
            copies = {n: (n[0], next(self._copies)) for n in prefix}
            for n, copy in copies.items():
                self.nodes[copy] = {label: copies.get(s, s) for label, s in self.nodes[n].items()}
            for n in list(self.nodes):
                if n not in loop:
                    self.nodes[n] = {label: copies.get(s, s) for label, s in self.nodes[n].items()}
            return True
        return False


# ---------------------------------------------------------------------------------------------- lowering
@register_lowering(torch.ops.dace.cfg.default)
def lower_cfg(ctx: LoweringContext, node, cfg_id, tensors, symints):
    record: CfgRecord = transport.lookup(int(cfg_id.value if isinstance(cfg_id, ConstValue) else cfg_id))
    tensors, symints = _items(tensors), _items(symints)
    fake_tensors = [_fake(a) for a in node.args[1]]
    fake_syms = [_fake(a) for a in node.args[2]]
    decompositions = build_decomposition_table(ctx.options.extra_decompositions, ctx.options.native_ops)
    prefix = f"cfg{record.id}"

    def bound(binding: Binding):
        value = tensors[binding.index] if binding.kind == "tensor" else symints[binding.index]
        if isinstance(value, TensorValue):
            # Values of the frame are read-only inside the blocks (no lowering may write them in place)
            return dataclasses.replace(value, source="input")
        return value

    def bound_fake(binding: Binding):
        return fake_tensors[binding.index] if binding.kind == "tensor" else fake_syms[binding.index]

    def own_examples(block_id: int) -> dict[str, Any]:
        """Symbolic inputs keep the symbols they were traced with (loop counters differ between edges)."""
        block = record.block(block_id)
        return {
            name: example
            for name, example in zip(block.input_names, block.input_examples)
            if not isinstance(example, torch.Tensor)
        }

    # Example (fake) inputs of every block: tensors are propagated from the entry along the edges in discovery order
    examples: dict[int, dict[str, Any]] = {}
    for block_id in set(record.entry_successors.values()):
        examples[block_id] = {
            name: bound_fake(record.entry_bindings[name])
            for name in record.block(block_id).input_names
            if name in record.entry_bindings
        }
        examples[block_id].update(own_examples(block_id))
    aten: dict[int, torch.fx.GraphModule] = {}
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

    # One container per CFG variable and metadata (shape, strides, dtype) it takes; the blocks read and write it. A
    # tensor sized by a loop-carried symbol (e.g., the ragged last chunk of a loop) has a different size wherever the
    # symbol has been reassigned in between: such variables get a container per segment that reads them.
    containers: dict[tuple, TensorValue] = {}
    segment_inputs: dict[tuple, dict[str, Any]] = {}

    def container(segment: tuple, name: str, example: torch.Tensor) -> TensorValue:
        key = (name, _tensor_key(example))
        if free_unbacked_symbols(example.size()) or free_unbacked_symbols(example.stride()):
            key += (segment,)
        if key not in containers:
            containers[key] = ctx.add_tensor_like(f"{prefix}_{name}", example)
        return containers[key]

    def inputs(segment: tuple) -> dict[str, Any]:
        """The values bound to the inputs of a segment (by variable name)."""
        if segment in segment_inputs:
            return segment_inputs[segment]
        result: dict[str, Any] = {}
        for name, example in examples[segment[0]].items():
            if isinstance(example, torch.Tensor):
                result[name] = container(segment, name, example)
            elif ctx.unassigned_unbacked(example) is not None or _is_assigned_unbacked(ctx, example):
                # A loop-carried integer: a symbol assigned on every edge into the block
                unbacked = example.node.expr
                symbol = ctx.symtab.define(unbacked.name, dtypes.int64, nonnegative=_nonnegative(example))
                ctx.assigned_symbols[unbacked.name] = symbol
                result[name] = _Carried(symbol)
            else:  # Other symbolic integers are identical on all incoming edges (blocks are specialized on them)
                result[name] = SymValue(ctx.symtab.to_dace(example))
        segment_inputs[segment] = result
        return result

    vals = node.meta["val"]
    outputs = [ctx.add_tensor_like(f"t_{node.name}_{k}", v) for k, v in enumerate(vals)]

    structure = _Structure(record)

    def label(key: tuple) -> str:
        return f"{prefix}_b{key[0]}" if key[1] == 0 else f"{prefix}_b{key[0]}_{key[1]}"

    exit_label = f"{prefix}_exit"

    def pass_values(values: dict[str, Any], constants: dict[str, Any], successor: tuple) -> dict[str, Any]:
        """
        Copies the tensors a block produced for the variables into the variables' containers (as one parallel
        assignment: a source that is also a destination goes through a temporary); returns the assignments to the
        successor's loop-carried symbols (to be made at once with :func:`_assign_at_once`).
        """
        assignments = {}
        copies: list[tuple[TensorValue, TensorValue]] = []
        for name, target in inputs(successor).items():
            source = values[name] if name in values else ConstValue(constants[name])
            if isinstance(target, TensorValue):
                if source.name != target.name:
                    copies.append((source, target))
            elif isinstance(target, _Carried):
                value = source.expr if isinstance(source, SymValue) else as_sym(source)
                if value != target.symbol:
                    assignments[target.symbol.name] = value
        destinations = {target.name for _, target in copies}
        deferred = []
        for source, target in copies:
            if source.name in destinations or source.is_view:  # (A view may alias a destination)
                temporary = ctx.add_array(
                    ctx.unique(f"{prefix}_pass"),
                    target.tshape,
                    target.torch_dtype,
                    target.tstrides,
                    device=target.device,
                )
                ctx.emit_copy(source, temporary)
                deferred.append((temporary, target))
            else:
                ctx.emit_copy(source, target)
        for source, target in deferred:
            ctx.emit_copy(source, target)
        return assignments

    def goto(successor: tuple, assignments: dict[str, Any]) -> None:
        """Jumps to ``successor``, assigning its loop-carried symbols on the way."""
        _assign_at_once(ctx, assignments)
        ctx.emit(tn.GotoNode(target=label(successor)))

    #: Segments that assign symbols on a taken conditional edge before jumping on: (label, assignments, successor)
    trampolines: list[Any] = []

    def branch(predicate, values: dict[str, Any], constants: dict[str, Any], successors: dict[Any, tuple]) -> None:
        # Tensors are copied before branching (every variable has its own container); symbols are assigned on the
        # edge taken, after the predicate is evaluated
        assignments = {flag: pass_values(values, constants, successor) for flag, successor in successors.items()}
        target = label(successors[True])
        if assignments[True]:
            target = f"{prefix}_edge{len(trampolines)}"
            trampolines.append((target, assignments[True], successors[True]))
        ctx.emit(tn.StateIfScope(condition=CodeBlock(predicate_expr(predicate)), children=[tn.GotoNode(target=target)]))
        goto(successors[False], assignments[False])

    children: list[tn.ScheduleTreeNode] = [tn.StateLabel(state=f"{prefix}_entry")]
    with ctx.scope(children):
        entry_values = {name: bound(binding) for name, binding in record.entry_bindings.items()}
        entry_successors = structure.nodes[ENTRY]
        if record.entry_condition is not None:
            condition = SymValue(ctx.symtab.to_dace(record.entry_condition))
            branch(condition, entry_values, record.entry_constants, entry_successors)
        elif record.entry_predicate is None:  # Into a loop
            successor = entry_successors[None]
            goto(successor, pass_values(entry_values, record.entry_constants, successor))
        else:
            branch(bound(record.entry_predicate), entry_values, record.entry_constants, entry_successors)

    #: Data-dependent symbols a block's own lowering assigns (``item()``), by block id: a copy of the block (a peeled
    #: segment) must assign them again instead of taking the first segment's assignment as given
    own_assignments: dict[int, set[str]] = {}
    for key in structure.order():
        block = structure.block(key)
        successors = structure.nodes[key]
        children.append(tn.StateLabel(state=label(key)))
        with ctx.scope(children):
            bindings = [_binding(inputs(key)[name]) for name in block.input_names] + [bound(b) for b in block.lifted]
            if key[1] == 0:
                assigned_before = set(ctx.assigned_symbols)
            else:
                for name in own_assignments.get(block.id, ()):
                    ctx.assigned_symbols.pop(name, None)
            results = ctx.importer.lower_subgraph(ctx, aten[block.id], bindings)
            if key[1] == 0:
                own_assignments[block.id] = set(ctx.assigned_symbols) - assigned_before
            if block.exit_kind == RETURN:
                for result, out in zip(results, outputs):
                    if not isinstance(result, TensorValue):
                        raise UnsupportedOpError(node.target, "captured control flow returns a non-tensor value")
                    ctx.emit_copy(result, out)
                ctx.emit(tn.GotoNode(target=exit_label))
            elif block.exit_kind == BRANCH:
                branch(results[0], dict(zip(block.output_names, results[1:])), block.output_constants, successors)
            else:
                successor = successors[None]
                goto(successor, pass_values(dict(zip(block.output_names, results)), block.output_constants, successor))
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


def _tensor_key(example: torch.Tensor) -> tuple:
    return (
        str(example.dtype),
        str(example.device),
        tuple(str(s) for s in example.size()),
        tuple(str(s) for s in example.stride()),
    )


def _assign_at_once(ctx: LoweringContext, assignments: dict[str, Any]) -> None:
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


def _output_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    output = next(n for n in gm.graph.nodes if n.op == "output")
    args = output.args[0]
    return list(args) if isinstance(args, (list, tuple)) else [args]
