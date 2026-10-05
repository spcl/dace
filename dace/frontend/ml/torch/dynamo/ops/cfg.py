# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering of captured control-flow graphs (``dace::cfg``, see :mod:`..cfg.transport`) onto schedule-tree control flow.

Every traced block becomes a labeled segment of a general block (``GBlock``): its torch-level graph is converted to
ATen (with the frontend's decomposition table) and lowered into the segment. CFG variables live in per-block input
containers; an edge copies the block's outputs into the inputs of its successor and jumps to it. Branches are
conditional gotos on the predicate, and returns copy into the operator's outputs and jump to the end of the general
block. Simplification then raises the gotos into loops and conditionals.
"""
from typing import Any, Dict, List, Sequence

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention import SDPBackend, sdpa_kernel

from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ..cfg import transport
from ..cfg.transport import BRANCH, GOTO, RETURN, Binding, CfgRecord
from ..context import ConstValue, LoweringContext, SymValue, TensorValue, TupleValue, UnsupportedOpError
from ..decompositions import build_decomposition_table
from . import register_lowering
from .control_flow import predicate_expr


def _items(v) -> list:
    return list(v.items) if isinstance(v, TupleValue) else list(v)


def _fake(arg) -> Any:
    """The fake value of an FX argument of the ``dace::cfg`` node."""
    return arg.meta['val'] if isinstance(arg, torch.fx.Node) else arg


def _to_aten(graph: torch.fx.Graph, examples: Sequence[Any], decompositions: Dict) -> torch.fx.GraphModule:
    """Converts a block's torch-level graph to a functional ATen graph with ``node.meta['val']``."""
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    # Inputs that alias (e.g., after ``z = y``) must be distinct tensors: make_fx maps each tensor to one placeholder
    seen = set()
    distinct = []
    for example in examples:
        if isinstance(example, torch.Tensor) and id(example) in seen:
            with example.fake_mode:
                example = torch.empty_strided(example.size(),
                                              example.stride(),
                                              dtype=example.dtype,
                                              device=example.device)
        seen.add(id(example))
        distinct.append(example)
    examples = distinct
    with sdpa_kernel(SDPBackend.MATH):
        return make_fx(torch.func.functionalize(gm, remove='mutations'),
                       decomposition_table=decompositions,
                       tracing_mode='real')(*examples)


@register_lowering(torch.ops.dace.cfg.default)
def lower_cfg(ctx: LoweringContext, node, cfg_id, tensors, symints):
    record: CfgRecord = transport.lookup(int(cfg_id.value if isinstance(cfg_id, ConstValue) else cfg_id))
    tensors, symints = _items(tensors), _items(symints)
    fake_tensors = [_fake(a) for a in node.args[1]]
    fake_syms = [_fake(a) for a in node.args[2]]
    decompositions = build_decomposition_table(ctx.options.extra_decompositions, ctx.options.native_ops)
    prefix = f'cfg{record.id}'

    def bound(binding: Binding):
        return tensors[binding.index] if binding.kind == 'tensor' else symints[binding.index]

    def bound_fake(binding: Binding):
        return fake_tensors[binding.index] if binding.kind == 'tensor' else fake_syms[binding.index]

    # Example (fake) inputs of every block, propagated from the entry along the edges in discovery order
    examples: Dict[int, Dict[str, Any]] = {}
    for block_id in set(record.entry_successors.values()):
        examples[block_id] = {
            name: bound_fake(record.entry_bindings[name])
            for name in record.block(block_id).input_names
        }
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
        outputs = [n.meta['val'] for n in _output_nodes(aten[block_id])]
        if block.exit_kind == RETURN:
            continue
        values = dict(zip(block.output_names, outputs[1:] if block.exit_kind == BRANCH else outputs))
        successors = [block.successors] if block.exit_kind == GOTO else list(block.successors.values())
        for successor in successors:
            if successor not in examples:
                examples[successor] = {name: values[name] for name in record.block(successor).input_names}
            order.append(successor)

    # Containers of the CFG variables (per block) and of the outputs
    inputs: Dict[int, Dict[str, Any]] = {}
    for block_id, values in examples.items():
        inputs[block_id] = {}
        for name, example in values.items():
            if isinstance(example, torch.Tensor):
                inputs[block_id][name] = ctx.add_tensor_like(f'{prefix}_b{block_id}_{name}', example)
            else:  # Symbolic integers are identical on all incoming edges (blocks are specialized on them)
                inputs[block_id][name] = SymValue(ctx.symtab.to_dace(example))
    vals = node.meta['val']
    outputs = [ctx.add_tensor_like(f't_{node.name}_{k}', v) for k, v in enumerate(vals)]

    def label(block_id: int) -> str:
        return f'{prefix}_b{block_id}'

    exit_label = f'{prefix}_exit'

    def pass_values(values: Dict[str, Any], successor: int) -> None:
        for name, target in inputs[successor].items():
            source = values[name]
            if isinstance(target, TensorValue) and source is not target:
                ctx.emit_copy(source, target)

    def branch(predicate, values: Dict[str, Any], successors: Dict[bool, int]) -> None:
        for successor in successors.values():
            pass_values(values, successor)
        ctx.emit(
            tn.StateIfScope(condition=CodeBlock(predicate_expr(predicate)),
                            children=[tn.GotoNode(target=label(successors[True]))]))
        ctx.emit(tn.GotoNode(target=label(successors[False])))

    children: List[tn.ScheduleTreeNode] = [tn.StateLabel(state=f'{prefix}_entry')]
    with ctx.scope(children):
        entry_values = {name: bound(binding) for name, binding in record.entry_bindings.items()}
        branch(tensors[record.entry_predicate], entry_values, record.entry_successors)

    for block_id in sorted(aten):
        block = record.block(block_id)
        children.append(tn.StateLabel(state=label(block_id)))
        with ctx.scope(children):
            bindings = [inputs[block_id][name] for name in block.input_names] + [bound(b) for b in block.lifted]
            results = ctx.importer.lower_subgraph(ctx, aten[block_id], bindings)
            if block.exit_kind == RETURN:
                for result, out in zip(results, outputs):
                    if not isinstance(result, TensorValue):
                        raise UnsupportedOpError(node.target, 'captured control flow returns a non-tensor value')
                    ctx.emit_copy(result, out)
                ctx.emit(tn.GotoNode(target=exit_label))
            elif block.exit_kind == BRANCH:
                branch(results[0], dict(zip(block.output_names, results[1:])), block.successors)
            else:
                pass_values(dict(zip(block.output_names, results)), block.successors)
                ctx.emit(tn.GotoNode(target=label(block.successors)))
    children.append(tn.StateLabel(state=exit_label))
    ctx.emit(tn.GBlock(children=children))
    return TupleValue(outputs)


def _output_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    output = next(n for n in gm.graph.nodes if n.op == 'output')
    args = output.args[0]
    return list(args) if isinstance(args, (list, tuple)) else [args]
