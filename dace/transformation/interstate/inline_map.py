# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
import sympy

from dace.sdfg import SDFG, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties
from dace import symbolic


@make_properties
class InlineMap(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(
        self, state: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive=False
    ):
        itervars = self.map_entry.map.params
        if len(itervars) > 1:
            return False

        nsdfg = self.nested_sdfg.sdfg
        if len(nsdfg.states()) == 1:
            return False

        start_state = nsdfg.start_state
        if len(start_state.nodes()) > 0:
            return False

        for oedge in nsdfg.out_edges(start_state):
            if oedge.data.assignments:
                return False

            condition = oedge.data.condition_sympy()
            if condition.__class__ not in [
                sympy.core.relational.StrictLessThan,
                sympy.core.relational.GreaterThan,
                sympy.core.relational.Unequality,
                sympy.core.relational.Equality
            ]:
                return False

            if str(condition.lhs) != itervars[0]:
                return False

            if isinstance(condition, (sympy.core.relational.Unequality,
                sympy.core.relational.Equality)):
                b, e, _ = self.map_entry.map.range[0]
                if not (str(condition.rhs) == str(b) or str(condition.rhs) == str(e)):
                    return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry
        map_exit = state.exit_node(self.map_entry)
        nsdfg_node = self.nested_sdfg

        ############################################
        # Isolate scope of map
        # TODO

        ############################################
        # Re-order nestes control-flow

        self._fission_maps_by_conditions(nsdfg_node, state, map_entry, map_exit)

        ############################################
        # Clean up

        state.remove_node(map_entry)
        state.remove_node(nsdfg_node)
        state.remove_node(map_exit)

        sdfg.reset_cfg_list()

    def _fission_maps_by_conditions(
        self,
        nsdfg_node: nodes.NestedSDFG,
        outer_state: SDFGState,
        map_entry: nodes.MapEntry,
        map_exit: nodes.MapExit
    ) -> None:
        nsdfg = nsdfg_node.sdfg
        start_state = nsdfg.start_state
        for oedge in nsdfg.out_edges(start_state):
            branch_nsdfg_node = copy.deepcopy(nsdfg_node)
            branch_nsdfg = branch_nsdfg_node.sdfg
            
            old_start_state = branch_nsdfg.start_state

            matching_edge = None
            for branch in branch_nsdfg.out_edges(old_start_state):
                if branch.data.condition.as_string == oedge.data.condition.as_string:
                    matching_edge = branch
                    break

            new_start_state = matching_edge.dst
            
            # Remove unreachable states
            branch_subgraph = set([e.dst for e in branch_nsdfg.bfs_edges(new_start_state)])
            branch_subgraph.add(new_start_state)
            states_to_remove = set(branch_nsdfg.states()) - branch_subgraph
            branch_nsdfg.remove_nodes_from(states_to_remove)

            branch_nsdfg.start_state = branch_nsdfg.node_id(new_start_state)
            outer_state.add_node(branch_nsdfg_node)
            
            # Add branch nsdfg to outer state
            branch_map_entry = copy.deepcopy(map_entry)
            outer_state.add_node(branch_map_entry)
            branch_map_exit = copy.deepcopy(map_exit)
            outer_state.add_node(branch_map_exit)

            for iedge in outer_state.in_edges(map_entry):
                outer_state.add_edge(iedge.src, iedge.src_conn, branch_map_entry, iedge.dst_conn, copy.deepcopy(iedge.data))

            for oedge in outer_state.out_edges(map_exit):
                outer_state.add_edge(branch_map_exit, oedge.src_conn, oedge.dst, oedge.dst_conn, copy.deepcopy(oedge.data))

            for oedge in outer_state.out_edges(map_entry):
                outer_state.add_edge(branch_map_entry, oedge.src_conn, branch_nsdfg_node, oedge.dst_conn, copy.deepcopy(oedge.data))

            for iedge in outer_state.in_edges(map_exit):
                outer_state.add_edge(branch_nsdfg_node, iedge.src_conn, branch_map_exit, iedge.dst_conn, copy.deepcopy(iedge.data))

            # Add condition to map definition
            condition = matching_edge.data.condition_sympy()
            b, e, s = branch_map_entry.map.range[0]
            if isinstance(condition, sympy.core.relational.StrictLessThan):
                e = min(e, condition.rhs - 1)
            elif isinstance(condition, sympy.core.relational.GreaterThan):
                b = max(b, condition.rhs)
            elif isinstance(condition, sympy.core.relational.Equality):
                b = condition.rhs
                e = condition.rhs
                s = 1
            elif isinstance(condition, sympy.core.relational.Unequality):
                if str(condition.rhs) == str(b):
                    b = condition.rhs + 1
                else:
                    e = condition.rhs - 1
            else:
                raise NotImplementedError
        
            branch_map_entry.map.range[0] = (b, e, s)

