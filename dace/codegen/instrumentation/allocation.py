import json
import os

import dace
from dace.codegen.instrumentation.papi import MapEntry
from dace.codegen.instrumentation.provider import SDFG
from dace.sdfg.nodes import AccessNode, EntryNode
from dace.transformation.passes.symbol_propagation import SDFGState
from dace.sdfg.nodes import Node


type StateAlloc = dict[AccessNode,list[SDFGState]]
type NodeAlloc = dict[AccessNode, list[Node]]


def inScope(scopedict: dict[Node, SDFGState | Node | None], node: Node, scope: Node) -> bool:
    node_scope = scopedict[node]
    return node_scope != None and (node_scope == scope or inScope(scopedict, node_scope, scope) if isinstance(node_scope, Node) else False)



def create_allocation_report(to : dict[SDFG | SDFGState | EntryNode, list[tuple[SDFG, SDFGState | None,AccessNode | None, bool, bool, bool]]]):

    #state_alloc: dict[AccessNode,list[SDFGState]] = {}
    #node_alloc: dict[AccessNode, list[Node]] = {}

    state_alloc: dict[str, list[str]] = {}
    node_alloc: dict[str, list[str]] = {}

    all_alloc: dict[str, list[str]] = {}

    report: dict[SDFG, dict[str, list[str]]] = {}

    for scope in to:
        for alloc_info in to[scope]:

            sdfg: SDFG = alloc_info[0]
            state : SDFGState | None = alloc_info[1]
            access_node =alloc_info[2]

            nodes_allocated: list[Node] = []
            states_allocated: list[SDFGState] = []

            if issubclass(type(scope),SDFG):
                #TODO: find example where SDFG is the scope and implement
                pass
            elif issubclass(type(scope),SDFGState):
                #highlight all nodes and the state itself
                nodes_allocated = list(scope.nodes()) if isinstance(scope, SDFGState) else []
                states_allocated = [scope] if isinstance(scope, SDFGState) else []
            elif issubclass(type(scope),EntryNode):
                if isinstance(scope, MapEntry):
                    nodes_allocated = []
                    scope_dict = state.scope_dict() if state != None else {}
                    for node in state.nodes() if state != None else []:
                        if inScope(scope_dict, node, scope) or node == scope:
                            nodes_allocated.append(node)
                    states_allocated = [state] if state != None else []

            if access_node != None:
                state_alloc[access_node.guid] = [state.guid for state in states_allocated]
                node_alloc[access_node.guid] = [node.guid for node in nodes_allocated]
                all_alloc[access_node.guid] = state_alloc[access_node.guid] + node_alloc[access_node.guid]
                if sdfg in report.keys():
                    report[sdfg].update(all_alloc)
                else:
                    report[sdfg] = all_alloc




    for sdfg in report:
        os.makedirs(f"{sdfg.build_folder}/perf", exist_ok=True)
        with open(f"{sdfg.build_folder}/perf/allocation-report-{str(hash(str(report[sdfg])))}.json", "x") as f:
            json.dump(report[sdfg],f)





    return
