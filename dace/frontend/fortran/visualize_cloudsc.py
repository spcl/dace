
import sys
from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional, Tuple

from dataclasses_json import dataclass_json

import dace
from dace.codegen import control_flow
from dace.dtypes import ScheduleType
from dace.transformation import helpers
from dace.sdfg import utils as sdutil

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
    def __hash__(self):
        return hash(self.name)

@dataclass_json
@dataclass
class Map(Node):
    schedule: dace.dtypes.ScheduleType = field(default_factory=lambda : ScheduleType.Default)

@dataclass_json
@dataclass
class SDFG(Node):
    pass

LoopMappingType = NewType('LoopMappingType', Dict[dace.SDFGState, control_flow.ForScope])
LoopsType = NewType('LoopMappingType', Dict[str, Optional[Loop]])

def analyze_loops(sdfg: dace.SDFG, serialized: Node) -> Tuple[LoopMappingType, LoopsType]:

    loop_mapping = {}
    loop_nesting = {}
    loops = {}

    def add_state(state: dace.SDFGState, scope):

        nonlocal loop_mapping
        # We might have to support While/Do loops at some point
        assert isinstance(scope, control_flow.ForScope)

        if state not in loop_mapping:
            loop_mapping[state] = scope
        else:
            assert loop_mapping[state] == scope

    def process_if(scope: control_flow.IfScope, loop: control_flow.ForScope):

        add_state(scope.branch_state, loop)
        for state in scope.body.elements:
            process_scope(state, loop)

    def process_for_loop(scope: control_flow.ForScope, loop: control_flow.ForScope):

        nonlocal loops, loop_nesting

        loops[scope.guard.label] = None
        if scope != loop:
            loop_nesting[scope.guard.label] = loop.guard.label
        for body_state in scope.body.elements:
            process_scope(body_state, scope)

    def process_scope(scope, loop: control_flow.ForScope):

        match type(scope):
            case control_flow.ForScope:
                process_for_loop(scope, loop)
            case control_flow.IfScope:
                process_if(scope, loop)
            case control_flow.SingleState:
                add_state(scope.state, loop)

    result = helpers.find_sdfg_control_flow(sdfg)
    for k, v in result.items():

        scope = v[1]
        states = v[0]
        if isinstance(scope, control_flow.ForScope):
            process_scope(scope, scope)

    return loop_mapping, loop_nesting, loops

map_name_mapping = {}

def analyze_map(map_node: dace.nodes.MapEntry, parent: Node):

    map_entry = Map()
    map_entry.schedule = map_node.schedule
    parent.children.append(map_entry)

    # skip duplicated names
    if map_node.label.startswith('outer_fused'):
        if map_node.label not in map_name_mapping:
            map_name_mapping[map_node.label] = f"{map_node.label}_{len(map_name_mapping) + 1}"
        map_entry.name = map_name_mapping[map_node.label]
    else:
        map_entry.name = map_node.label

    return map_entry

def analyze_sdfg(sdfg: dace.SDFG, cur_node: Optional[Node] = None):

    if cur_node is None:
        cur_node = SDFG(name=sdfg.label)

    loop_mapping, loop_nesting, loops = analyze_loops(sdfg, cur_node)

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

    for loop in loops.keys():
        loop_node = Loop(name=loop)
        loops[loop] = loop_node

    loops_without_parents = set()

    for loop, loop_node in loops.items():
        if loop in loop_nesting:
            parent_loop = loop_nesting[loop]
            loops[parent_loop].children.append(loop_node)
        else:
            loops_without_parents.add(loop_node)
            #cur_node.children.append(loop_node)

    for state in sdfg.states():

        parent = cur_node
        # check if the state is inside a loop
        if state in loop_mapping:
            parent = loops[loop_mapping[state].guard.label]

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

                #case dace.nodes.MapEntry:

                #    print("Map", node.label)
                #    print(state.scope_children()[node])

    for loop in loops_without_parents:
        cur_node.children.append(loop)

    return cur_node

def visualize_sdfg(serialized: Node, graph: graphviz.Digraph, parent: Optional[Node] = None):

    if isinstance(serialized, SDFG):

        with graph.subgraph(name=serialized.name) as subg:
            subg.attr(style='filled', color='grey')
            subg.node_attr.update(style='filled', color='white')
            subg.attr(label=serialized.name)

            #print(f'Add graph {subg.name}')
            for child in serialized.children:
                visualize_sdfg(child, subg, serialized)

    elif isinstance(serialized, Map):

        # FIXME: find parent
        graph.edge(parent.name, serialized.name)

        label = f"""<<B>Map</B>: {serialized.name} <BR/>
        <B>Schedule:</B> {serialized.schedule}>"""

        if serialized.schedule == ScheduleType.GPU_Device:
            graph.node(serialized.name, shape='parallelogram', fillcolor='green', color='black', label=label)
        else:
            graph.node(serialized.name, shape='parallelogram', fillcolor='white', color='black', label=label)

        for child in serialized.children:
            visualize_sdfg(child, graph, serialized)

    elif isinstance(serialized, Loop):

        graph.edge(parent.name, serialized.name)

        label = f"""<<B>Loop</B>: {serialized.name}>"""

        graph.node(serialized.name, shape='ellipse', fillcolor='lightblue', color='black', label=label)

        for child in serialized.children:
            visualize_sdfg(child, graph, serialized)

if __name__ == "__main__":

    import sys
    #sdfg = dace.SDFG.from_file(sys.argv[1])
    sdfg = dace.SDFG.from_file('generation_full/test.sdfg')

    result = analyze_sdfg(sdfg)
    #print(sdfg.label)

    import graphviz
    g = graphviz.Digraph('test')
    visualize_sdfg(result, g)
