# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Optional, Set

from dace import SDFG, SDFGState
import dace
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.transformation import pass_pipeline as ppl, transformation

from dace.symbolic import symbol, SymExpr


class InterstateAssignmentToMap(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Edges

    @staticmethod
    def _is_simple_assignment(expression_str):
        expr = SymExpr(expression_str)
        simplified_expr = expr.simplify()
        is_constant = simplified_expr.is_number
        return is_constant, simplified_expr

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for state in sdfg.states():
            for in_edge in sdfg.in_edges(state):
                assignments = in_edge.data._assignments
                condition = in_edge.data._cond_sympy
                simplifiable_assignments = []
                for var, assignment in assignments.items():
                    is_constant, constant = InterstateAssignmentToMap._is_simple_assignment(expression_str=assignment)
                    simplifiable_assignments.append((var, constant))
                    # If this state is a constant, then we remove this assignment from the assignments

                # Rm assignments
                for k, v in simplifiable_assignments:
                    assignments.pop(k, None)

                # Create cutout of the state as an SDFG (should have only one state)
                # Create NestedSDFG, add it to the current state
                cutout = SDFGCutout.singlestate_cutout(state, *state.nodes())
                new_nested_sdfg_content = cutout.sdfg
                for arr_name, arr in new_nested_sdfg_content.arrays.items():
                    arr.transient = False
                assert len(new_nested_sdfg_content.states()) == 1
                new_nested_sdfg_state = new_nested_sdfg_content.states()[0]

                # Collect input and output nodes that need to be connected from outside the nested SDFG
                in_nodes = [v for v in new_nested_sdfg_state.nodes() if len(new_nested_sdfg_state.in_edges(v)) == 0]
                out_nodes = [v for v in new_nested_sdfg_state.nodes() if len(new_nested_sdfg_state.out_edges(v)) == 0]

                in_node_names = set([v.data for v in in_nodes])
                out_node_names = set([v.data for v in out_nodes])

                # Remove all previous nodes from the state
                nodes_to_rm = state.nodes()
                for node in nodes_to_rm:
                    state.remove_node(node)

                # Create the map
                ndrange = dict()
                for var, constant in simplifiable_assignments:
                    ndrange[var] = dace.subsets.Range([(constant, constant, 1)])
                map_entry, map_exit = state.add_map(name="_symbol_assignment",
                                                    ndrange=ndrange,
                                                    schedule=dace.dtypes.ScheduleType.Sequential)

                nested_sdfg = state.add_nested_sdfg(new_nested_sdfg_content, state, in_node_names, out_node_names)

                # Connect the input and outputs to the nested SDFG, edges with None in Memlets
                # are necessary for the codegen to order array allocation / decleration at the correct place
                for in_node_data in in_node_names:
                    an = state.add_access(in_node_data)
                    state.add_edge(map_entry, None, an, None, dace.memlet.Memlet(None))
                    state.add_edge(an, None, nested_sdfg, in_node_data, dace.memlet.Memlet(in_node_data))
                for out_node_data in out_node_names:
                    an = state.add_access(out_node_data)
                    state.add_edge(nested_sdfg, out_node_data, an, None, dace.memlet.Memlet(out_node_data))
                    state.add_edge(an, None, map_exit, None, dace.memlet.Memlet(None))
