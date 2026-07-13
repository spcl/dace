# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Elementwise computation mechanism: lowers a canonical flat expression into a
:class:`TaskletNode` (scalar result) or a :class:`MapScope` with a tasklet
(array result), with NumPy-style broadcasting of operands.

This mechanism serves *any* frontend construct that reduces to an elementwise
operation over data operands — Python operators, NumPy ufuncs, and future
registry entries all converge here.
"""
import ast
from typing import List

from dace import subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.lowering.access import (DataAccess, indexed_subset, nondegenerate_shape,
                                                          substitute_data_operands)
from dace.frontend.python.nextgen.lowering.registry import LoweringState


def emit_computation(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Emit a tasklet (scalar result) or map-with-tasklet (array result) that
    computes a canonical flat expression into the target access.
    """
    code, operands = substitute_data_operands(value, state)
    line = getattr(statement, 'lineno', 0)
    result_shape = nondegenerate_shape(target.subset)

    if not result_shape:
        # Scalar result: single tasklet
        tasklet = nodes.Tasklet(f'assign_{line}', {connector
                                                   for connector, _ in operands}, {'__out'}, f'__out = {code}')
        in_memlets = {connector: Memlet(data=access.container, subset=access.subset) for connector, access in operands}
        out_memlets = {'__out': Memlet(data=target.container, subset=target.subset)}
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
        return

    # Array result: elementwise map
    params = [f'__i{i}' for i in range(len(result_shape))]
    map_range = subsets.Range([(0, size - 1, 1) for size in result_shape])
    map_node = nodes.MapEntry(nodes.Map(f'map_{line}', params, map_range))
    tasklet = nodes.Tasklet(f'assign_{line}', {connector for connector, _ in operands}, {'__out'}, f'__out = {code}')

    in_memlets = {}
    for connector, access in operands:
        if access.is_scalar_access:
            in_memlets[connector] = Memlet(data=access.container, subset=access.subset)
        else:
            in_memlets[connector] = Memlet(data=access.container, subset=indexed_subset(access, params, result_shape))
    out_memlets = {'__out': Memlet(data=target.container, subset=target_indexed_subset(target.subset, params))}

    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def target_indexed_subset(subset: subsets.Range, params: List[str]) -> subsets.Range:
    """
    Index the non-degenerate dimensions of a write subset with map parameters,
    keeping degenerate dimensions pinned to their start.
    """
    ranges = []
    param_iterator = iter(params)
    for size, (start, _, step) in zip(subset.size(), subset.ranges):
        if size == 1:
            ranges.append((start, start, 1))
        else:
            param = symbolic.pystr_to_symbolic(next(param_iterator))
            index = start + param * step
            ranges.append((index, index, 1))
    return subsets.Range(ranges)
