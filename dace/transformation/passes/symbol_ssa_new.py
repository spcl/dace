# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, Union
from dace.sdfg.graph import Edge
from dace.sdfg.sdfg import InterstateEdge
import networkx as nx
from dace import SDFG, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap


class StrictSymbolSSA(ppl.Pass):
    """
    Perform an SSA transformation on all symbols in the SDFG in a strict manner, i.e., without introducing phi nodes.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def depends_on(self):
        return {ap.SymbolWriteScopes, ap.AccessSets, ap.SymbolAccessSets}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Rename symbols in a restricted SSA manner.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each symbol.
        """
        def add_definition(new_def: str, original_name: str):
            if not definitions.get(original_name):
                definitions[original_name] = set()
            definitions[original_name].add(new_def)

        def update_last_def(state: SDFGState, new_def: str, original_name: str):
            if not last_defs.get(original_name):
                last_defs[original_name] = {}
            last_defs[original_name][state] = new_def

        def find_reaching_def(state: SDFGState, var: str):
            if last_defs.get(var):
                if state in last_defs[var].keys():
                    return last_defs[var][state]

            # otherwise return the last definition of the immediate dominator
            idom = immediate_dominators[state]
            if last_defs.get(var):
                if last_defs[var].get(idom):
                    return last_defs[var][idom]
                
            # in case the state is the initial state and there isn't any reaching definition in the current state just return the original variable
            return var

        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]
                          ] = pipeline_results[ap.AccessSets.__name__][sdfg.sdfg_id]
        
        immediate_dominators = nx.dominance.immediate_dominators(
            sdfg.nx, sdfg.start_state)
        
        symbol_access_sets: Dict[int, Dict[Union[SDFGState, Edge[InterstateEdge]], Tuple[Set[str], Set[str]]]] = pipeline_results[ap.SymbolAccessSets.__name__][sdfg.sdfg_id]
        dominance_frontiers = nx.dominance.dominance_frontiers(
            sdfg.nx, sdfg.start_state)
        
        phi_nodes: Dict[SDFGState, Dict[str, Dict]] = defaultdict(None)

        definitions: Dict[str, Set[str]] = defaultdict(None)

        last_defs: Dict[str, Dict[SDFGState, str]] = defaultdict(None)


        symbols = sdfg.symbols.copy()

        for state in sdfg.states():
            phi_nodes[state] = {}

        for sym in symbols.keys():
            defining_edges = set()
            phi_states = set()

            for edge in sdfg.edges():
                if not sym in symbol_access_sets[edge][1]:
                    continue
                defining_edges.add(edge)
                    
            while(defining_edges):
                current_edge = next(iter(defining_edges))
                dst_state = current_edge.dst
                dominance_frontier = dominance_frontiers[dst_state]

                if sdfg.in_degree(dst_state) > 1:
                    phi_nodes[dst_state][sym] = {
                        "name": sym,
                        "variables": [],
                        "descriptor": None
                    }

                for frontier_state in dominance_frontier:
                    if frontier_state in phi_states:
                        continue
                    phi_nodes[frontier_state][sym] = {
                        "name": sym,
                        "variables": [],
                        "descriptor": None
                    }
                    phi_states.add(frontier_state)
                    for edge in sdfg.out_edges(frontier_state):
                        defining_edges.add(edge)

                defining_edges.remove(current_edge)



                # create dictionary mapping each node to the nodes it immediatly dominates basically representing the dominator tree
        immediate_dominated = defaultdict(None)
        for node, dom_node in immediate_dominators.items():
            if dom_node not in immediate_dominated:
                immediate_dominated[dom_node] = set()
            immediate_dominated[dom_node].add(node)

        # traverse the dominator tree depth first and rename all variables
        # TODO: rename occurences of symbol reads within the state
        stack = []
        visited = set()
        stack.append(sdfg.start_state)
        while stack:
            current_state: SDFGState = stack.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            if immediate_dominated.get(current_state):
                children = immediate_dominated[current_state]
                for child in children:
                    if child not in visited:
                        stack.append(child)
        
            # rename phi nodes
            for sym in symbols.keys():
                if not phi_nodes[current_state].get(sym):
                    continue
            
                newname = sdfg.find_new_symbol(sym)
                sdfg.symbols[newname] = sdfg.symbols[sym]
                phi_nodes[current_state][sym]["name"] = newname

                add_definition(newname, sym)
                update_last_def(current_state, newname, sym)

            # rename conditions and RHS s of assignments of outgoing edges
            rename_dict = {}
            for sym in symbols.keys():
                last_def = find_reaching_def(current_state, sym)
                rename_dict[sym] = last_def
            
            for edge in sdfg.out_edges(current_state):
                inter_state_edge: InterstateEdge = edge.data
                inter_state_edge.replace_dict(rename_dict, False)


            current_state.replace_dict(rename_dict)

            
            for out_edge in sdfg.out_edges(current_state):
                for sym in symbols.keys():
                    newname = rename_dict[sym]
                    if sym in out_edge.data.assignments.keys():
                        newname = sdfg.find_new_symbol(sym)
                        sdfg.symbols[newname] = sdfg.symbols[sym]
                        try:
                            out_edge.data.assignments[newname] = out_edge.data.assignments[sym]
                            del out_edge.data.assignments[sym]
                        except KeyError:
                            pass
                    if phi_nodes[out_edge.dst].get(sym):
                        phi_nodes[out_edge.dst][sym]["variables"].append(newname)
                    else:
                        update_last_def(out_edge.dst, newname, sym)
            

        import pprint
        print("phi_nodes")
        pprint.pprint(phi_nodes)




        results: Dict[str, Set[str]] = defaultdict(lambda: set())

        # symbol_scope_dict: ap.SymbolScopeDict = pipeline_results[ap.SymbolWriteScopes.__name__][sdfg.sdfg_id]

        # for name, scope_dict in symbol_scope_dict.items():
        #     # If there is only one scope, don't do anything.
        #     if len(scope_dict) <= 1:
        #         continue

        #     for write, shadowed_reads in scope_dict.items():
        #         if write is not None:
        #             newname = sdfg.find_new_symbol(name)
        #             sdfg.symbols[newname] = sdfg.symbols[name]

        #             # Replace the write to this symbol with a write to the new symbol.
        #             try:
        #                 write.data.assignments[newname] = write.data.assignments[name]
        #                 del write.data.assignments[name]
        #             except KeyError:
        #                 # Ignore.
        #                 pass

        #             # Replace all dominated reads.
        #             for read in shadowed_reads:
        #                 if isinstance(read, SDFGState):
        #                     read.replace(name, newname)
        #                 else:
        #                     if read not in scope_dict:
        #                         read.data.replace(name, newname)
        #                     else:
        #                         read.data.replace(name, newname, replace_keys=False)

        #             results[name].add(newname)

        if len(results) == 0:
            return None
        else:
            return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} symbols: {pass_retval}.'
