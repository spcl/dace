#!/usr/bin/env python

import sys
from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional, Tuple

from dataclasses_json import dataclass_json

import dace
from dace.codegen import control_flow
from dace.dtypes import ScheduleType
from dace.transformation.passes import Loops, LoopInfo
#from dace.transformation import helpers
#from dace.sdfg import utils as sdutil

import sympy

import graphviz

@dataclass_json
@dataclass
class Node:
    children: List['Node'] = field(default_factory=lambda : [])
    name: str = ""

    # un-frozen dataclasses are not hashable - but we have unique name!
    def __hash__(self):
        return hash(self.name)

@dataclass_json
@dataclass
class Loop(Node):
    execution_count: sympy.core.add.Add = field(default_factory=lambda : sympy.core.add.Add())
    def __hash__(self):
        return hash(self.name)

@dataclass_json
@dataclass
class Map(Node):
    schedule: dace.dtypes.ScheduleType = field(default_factory=lambda : ScheduleType.Default)
    execution_count: sympy.core.add.Add = field(default_factory=lambda : sympy.core.add.Add())

@dataclass_json
@dataclass
class SDFG(Node):
    pass

LoopMappingType = NewType('LoopMappingType', Dict[dace.SDFGState, control_flow.ForScope])
LoopsType = NewType('LoopMappingType', Dict[str, Tuple[control_flow.ForScope, Optional[Loop]]])

map_name_mapping = {}

def analyze_map(map_node: dace.nodes.MapEntry, parent: Node):

    map_entry = Map()
    map_entry.schedule = map_node.schedule
    parent.children.append(map_entry)

    expr = map_node.map.range.ranges[0][1] - map_node.map.range.ranges[0][0]
    from sympy import symbols
    x = symbols('NCLV') map_entry.execution_count = expr.subs(x, 5)

    # skip duplicated names
    if map_node.label.startswith('outer_fused'):
        map_name = f"{map_node.label}_{len(map_name_mapping) + 1}"
    else:
        map_name = map_node.label

    if map_name not in map_name_mapping:
        map_name_mapping[map_name] = 1
    else:
        map_name_mapping[map_name] += 1
        map_name = f"{map_name}_{map_name_mapping[map_name]}"

    map_entry.name = map_name

    return map_entry

def analyze_sdfg(sdfg: dace.SDFG, cur_node: Optional[Node] = None):

    if cur_node is None:
        cur_node = SDFG(name=sdfg.label)

    #loop_mapping, loop_nesting, loops = analyze_loops(sdfg, cur_node)
    loops = Loops.from_sdfg(sdfg)
    loop_nodes = {}

    """
        Mapping algorithm works as follows:
        (1) We find all loops in the SDFG, and create nodes for them.
        (2) If the loop is nested inside another loop, add the node under the outer loop.
        (3) For each new state, we check if it is located inside a loop. If yes,
        then all maps will be placed under this loop.
        (4) For each map, we insert a map node.
        Since maps often include a nested SDFG, we use the cur_map field to pass nested mapping.

        Missing: nested maps.
        Missing; loops nested inside map.

        Missing currently: nested maps.
    """
    for loop in loops.loops:

        loop_node = Loop(name=loop.name)

        from sympy import symbols
        x = symbols('NCLV')
        loop_node.execution_count = loop.guard.executions.subs(x, 5)

        loop_nodes[loop] = loop_node

    loops_without_parents = set()

    for loop, loop_node in loop_nodes.items():

        if not loop.is_nested:
            loops_without_parents.add(loop_node)
        else:
            loop_nodes[loop.parent_loop].children.append(loop_node)

    for state in sdfg.states():

        parent = cur_node
        # check if the state is inside a loop
        state_loop = loops.state_inside_loop(state)
        if state_loop is not None:
            parent = loop_nodes[state_loop]

        maps = {}
        # first, find all maps - we need to create nodes and then explore map bodies.
        for node in state.nodes():

            if isinstance(node, dace.nodes.MapEntry):
                map_entry = analyze_map(node, parent)
                for nested_map_node in state.scope_children()[node]:
                    maps[nested_map_node] = map_entry

        # now, we can explore everything else.
        # we check if the node is nested in map - this way, we can create proper nesting
        for node in state.nodes():
            match type(node):

                case dace.nodes.NestedSDFG:

                    if node in maps:
                        analyze_sdfg(node.sdfg, maps[node])
                    else:
                        analyze_sdfg(node.sdfg, parent)

    for loop in loops_without_parents:
        cur_node.children.append(loop)

    return cur_node

def visualize_sdfg(serialized: Node, graph: graphviz.Digraph, parent_name: Optional[str] = None):

    if isinstance(serialized, SDFG):

        with graph.subgraph(name=serialized.name) as subg:
            subg.attr(style='filled', color='grey')
            subg.node_attr.update(style='filled', color='white')
            subg.attr(label=serialized.name)

            for child in serialized.children:
                visualize_sdfg(child, subg, serialized.name)

    elif isinstance(serialized, Map):

        graph.edge(parent_name, serialized.name)

        label = f"""<
        <B>Map</B>: {serialized.name} <BR/>
        <B>Schedule:</B> {serialized.schedule} <BR/>
        <B>Executions:</B> {serialized.execution_count}>"""

        if serialized.schedule == ScheduleType.GPU_Device:
            graph.node(serialized.name, shape='parallelogram', fillcolor='green', color='black', label=label)
        else:
            graph.node(serialized.name, shape='parallelogram', fillcolor='white', color='black', label=label)

        for child in serialized.children:
            visualize_sdfg(child, graph, serialized.name)

    elif isinstance(serialized, Loop):

        graph.edge(parent_name, serialized.name)

        label = f"""<
        <B>Loop</B>: {serialized.name} <BR/>
        <B>Executions:</B> {serialized.execution_count}>"""

        graph.node(serialized.name, shape='ellipse', fillcolor='lightblue', color='black', label=label)

        for child in serialized.children:
            visualize_sdfg(child, graph, serialized.name)

if __name__ == "__main__":

    import sys
    sdfg = dace.SDFG.from_file(sys.argv[1])

    result = analyze_sdfg(sdfg)

    import graphviz
    g = graphviz.Digraph(sys.argv[2])
    visualize_sdfg(result, g)
    g.render()
