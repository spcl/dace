# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Sequentialization of maps in schedule trees."""
from typing import List

from dace import data, symbolic
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion


def convert_map_to_loop(stree: tn.ScheduleTreeRoot) -> int:
    """
    Converts every map in the schedule tree into sequential for-loops, one per map dimension (outermost first).

    Dynamic map inputs (the ``dscopy`` statements that precede a map with, e.g., a data-dependent range) are read
    once before the map runs, so they become symbol assignments before the outermost loop. A map whose dynamic inputs
    would clash with an existing container or symbol name is left as-is.

    :param stree: The schedule tree to modify in place.
    :return: The number of maps converted.
    :raises ValueError: If a map has a provably negative step, which is invalid in the IR. The tree may then be
                        partially converted.
    """
    root = stree
    converted = 0

    def convert(scope: tn.ScheduleTreeScope) -> None:
        nonlocal converted
        new_children: List[tn.ScheduleTreeNode] = []
        dynamic_inputs: List[tn.DynScopeCopyNode] = []  # Pending dynamic inputs of the next dataflow scope
        for child in scope.children:
            if isinstance(child, tn.DynScopeCopyNode):
                dynamic_inputs.append(child)
                continue
            if isinstance(child, tn.ScheduleTreeScope):
                convert(child)
            if isinstance(child, tn.MapScope) and _can_convert(root, dynamic_inputs):
                new_children.extend(_dynamic_inputs_to_assignments(root, dynamic_inputs))
                new_children.append(_map_to_loops(child))
                converted += 1
            else:
                new_children.extend(dynamic_inputs)
                new_children.append(child)
            dynamic_inputs = []
        new_children.extend(dynamic_inputs)

        scope.children = []
        scope.add_children(new_children)

    convert(stree)
    return converted


def _can_convert(root: tn.ScheduleTreeRoot, dynamic_inputs: List[tn.DynScopeCopyNode]) -> bool:
    # Dynamic inputs are only visible inside the map; as symbol assignments they must not overwrite existing names.
    return all(dscopy.target not in root.containers and dscopy.target not in root.symbols for dscopy in dynamic_inputs)


def _dynamic_inputs_to_assignments(root: tn.ScheduleTreeRoot,
                                   dynamic_inputs: List[tn.DynScopeCopyNode]) -> List[tn.AssignNode]:
    result = []
    for dscopy in dynamic_inputs:
        desc = root.containers[dscopy.memlet.data]
        if isinstance(desc, data.Scalar):
            value = dscopy.memlet.data
        else:
            value = f'{dscopy.memlet.data}[{dscopy.memlet.subset}]'
        root.symbols[dscopy.target] = desc.dtype
        result.append(
            tn.AssignNode(name=dscopy.target,
                          value=CodeBlock(value),
                          edge=InterstateEdge(assignments={dscopy.target: value})))
    return result


def _map_to_loops(scope: tn.MapScope) -> tn.ForScope:
    dace_map = scope.node.map
    body = scope.children
    for param, (start, end, step) in reversed(list(zip(dace_map.params, dace_map.range))):
        if (step < 0) == True:
            raise ValueError(f'Map "{dace_map.label}" has a negative step ({step}) for parameter "{param}", '
                             'map steps must be positive')
        # Map ranges are inclusive (and map steps are positive), loop conditions follow the frontend's ``range`` form
        loop = LoopRegion(f'loop_{dace_map.label}_{param}',
                          condition_expr=f'{param} < {symbolic.symstr(end + 1)}',
                          loop_var=param,
                          initialize_expr=f'{param} = {symbolic.symstr(start)}',
                          update_expr=f'{param} = {symbolic.symstr(symbolic.symbol(param) + step)}',
                          unroll=dace_map.unroll,
                          unroll_factor=dace_map.unroll_factor or 0)
        body = [tn.ForScope(loop=loop, children=body)]
    return body[0]
