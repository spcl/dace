# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace import properties, SDFG, nodes
from dace.sdfg.nodes import Dict
from dace.sdfg.state import ConditionalBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from typing import Set, Tuple, Union, Optional
import dace.sdfg.construction_utils as cutil


@properties.make_properties
@explicit_cf_compatible
class EliminateBranches(ppl.Pass):
    try_clean = properties.Property(dtype=bool, default=False, allow_none=False)
    clean_only = properties.Property(dtype=bool, default=False, allow_none=True)
    permissive = properties.Property(dtype=bool, default=False, allow_none=False)
    eps_operator_type_for_log_and_div = properties.Property(dtype=str, default="max", allow_none=True)
    apply_to_top_level_ifs = properties.Property(dtype=bool, default=False, allow_none=False)
    try_demote_and_fuse = properties.Property(dtype=bool, default=True, allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def _has_no_parent_loops_or_maps(self, root: SDFG, sdfg: SDFG, parent_nsdfg_state: Union[SDFG, None],
                                     node: ConditionalBlock) -> bool:
        parent_loops_and_maps = {
            m
            for m in cutil.get_parent_map_and_loop_scopes(
                parent_nsdfg_state.sdfg if parent_nsdfg_state is not None else sdfg, node, None)
        }
        return len(parent_loops_and_maps) == 0

    def _run_transformation(self,
                            root: SDFG,
                            sdfg: SDFG,
                            parent_nsdfg_state: Union[SDFG, None] = None) -> Tuple[int, Set[str]]:
        # Root SDFG is needed to collect all parent maps
        from dace.transformation.interstate import branch_elimination
        # Try applying without cleaning
        num_applied = 0
        added_scalar_names = set()
        for node in sdfg.all_control_flow_regions():
            if isinstance(node, ConditionalBlock):
                t = branch_elimination.BranchElimination()
                # If branch is top-level do not apply
                if not self.apply_to_top_level_ifs:
                    if self._has_no_parent_loops_or_maps(root, sdfg, parent_nsdfg_state, node):
                        continue

                t.conditional = node
                t.parent_nsdfg_state = parent_nsdfg_state
                t.eps_operator_type_for_log_and_div = self.eps_operator_type_for_log_and_div
                t.try_demote_and_fuse = self.try_demote_and_fuse
                if t.can_be_applied(graph=node.parent_graph, expr_index=0, sdfg=node.sdfg, permissive=self.permissive):
                    newly_added_scalar_names = t.apply(graph=node.parent_graph, sdfg=node.sdfg)
                    added_scalar_names = added_scalar_names.union(newly_added_scalar_names)
                    num_applied += 1
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    new_num_applied, newly_added_scalar_names = self._run_transformation(root, node.sdfg, state)
                    num_applied += new_num_applied
                    added_scalar_names = added_scalar_names.union(newly_added_scalar_names)
        return num_applied, added_scalar_names

    def _run_clean(self, root: SDFG, sdfg: SDFG, parent_nsdfg_state: Union[SDFG, None], lift_multi_state: bool):
        from dace.transformation.interstate import branch_elimination

        for node in sdfg.all_control_flow_regions():
            if isinstance(node, ConditionalBlock):
                if not self.apply_to_top_level_ifs:
                    if self._has_no_parent_loops_or_maps(root, sdfg, parent_nsdfg_state, node):
                        continue
                t = branch_elimination.BranchElimination()
                t.conditional = node
                t.parent_nsdfg_state = parent_nsdfg_state
                t.eps_operator_type_for_log_and_div = self.eps_operator_type_for_log_and_div
                t.try_demote_and_fuse = self.try_demote_and_fuse
                if all(t.only_top_level_tasklets(branch) for _, branch in node.branches):
                    # Try clean might generate multiple conditionals
                    t.try_clean(node.parent_graph, node.sdfg, lift_multi_state)

        # Try to sequentialize if-else branches that we can't apply right now to make the applyable
        if lift_multi_state:
            for node in sdfg.all_control_flow_regions():
                if isinstance(node, ConditionalBlock):
                    if not self.apply_to_top_level_ifs:
                        if self._has_no_parent_loops_or_maps(root, sdfg, parent_nsdfg_state, node):
                            continue
                    t = branch_elimination.BranchElimination()
                    t.conditional = node
                    t.parent_nsdfg_state = parent_nsdfg_state
                    t.try_demote_and_fuse = self.try_demote_and_fuse
                    first_conditional, second_conditional = t.sequentialize_if_else_branch_for_all_subsets(
                        node.parent_graph)
                    # We still can't apply the pass on these branches also try to lift states outside the branch
                    # because the branch elimination transformation requires the if branch to have one state
                    if first_conditional is not None and second_conditional is not None:
                        t.conditional = first_conditional
                        if not t.can_be_applied(first_conditional.parent_graph, 0, first_conditional.sdfg, False):
                            t.duplicate_condition_across_all_top_level_nodes_if_line_graph_and_empty_interstate_edges(
                                first_conditional.parent_graph)
                        t.conditional = second_conditional
                        if not t.can_be_applied(second_conditional.parent_graph, 0, second_conditional.sdfg, False):
                            t.duplicate_condition_across_all_top_level_nodes_if_line_graph_and_empty_interstate_edges(
                                second_conditional.parent_graph)

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    self._run_clean(root, node.sdfg, state, lift_multi_state)

    def _apply_eliminate_branches(self,
                                  root: SDFG,
                                  sdfg: SDFG,
                                  parent_nsdfg_state: Union[SDFG, None] = None) -> Tuple[int, Set[str]]:
        """Apply EliminateBranches transformation to all eligible conditionals."""
        from dace.transformation.interstate import branch_elimination
        # Pattern matching with conditional branches to not work (9.10.25), avoid it
        # Depending on the number of nestedness we need to apply that many times because
        # the transformation only runs on top-level ConditionalBlocks

        # Workflow:
        # 1. Try to apply as much as one can apply without trying to clean
        # 2. Run try clean without `lift_multi_state` on ifs that where the transformation can't apply to
        # 3. Try to apply again
        # 4. Run try clean with `lift_multi_state`

        # Root SDFG is needed to collect all parent maps, which is necessary to detect if a conditional is top level
        changed = True
        num_applied = 0
        added_scalar_names = set()

        while changed:
            changed = False

            cur_num_applied, newly_added_scalars = self._run_transformation(root, sdfg, parent_nsdfg_state)
            num_applied += cur_num_applied
            changed = changed or (cur_num_applied != 0)
            added_scalar_names = added_scalar_names.union(newly_added_scalars)

            # Run try clean, without lifting multistate
            if self.try_clean:
                self._run_clean(root, sdfg, parent_nsdfg_state, False)

            # Run transformation again
            cur_num_applied2, newly_added_scalars2 = self._run_transformation(root, sdfg, parent_nsdfg_state)
            num_applied += cur_num_applied2
            changed = changed or (cur_num_applied2 != 0)
            added_scalar_names = added_scalar_names.union(newly_added_scalars2)

            # Run try clean, with lifting multistate
            if self.try_clean:
                self._run_clean(root, sdfg, parent_nsdfg_state, True)

            # Run transformation again
            cur_num_applied3, newly_added_scalars3 = self._run_transformation(root, sdfg, parent_nsdfg_state)
            num_applied += cur_num_applied3
            changed = changed or (cur_num_applied3 != 0)
            added_scalar_names = added_scalar_names.union(newly_added_scalars3)

        return num_applied, added_scalar_names

    def _apply_symbol_removal(self, sdfg: SDFG):
        from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols

        changed = False
        ret_val = RemoveUnusedSymbols().apply_pass(sdfg, _={})
        changed = ret_val is not None

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    changed |= self._apply_symbol_removal(node.sdfg)

        return changed

    def apply_pass(self, sdfg: SDFG, d: Dict) -> Tuple[int, Set[str]]:
        if self.clean_only is True:
            self.try_clean = True
        cur_num_applied, cur_added_scalar_names = self._apply_eliminate_branches(sdfg, sdfg, None)

        num_applied = cur_num_applied
        added_scalar_names = copy.deepcopy(cur_added_scalar_names)
        while cur_num_applied:
            cur_num_applied, cur_added_scalar_names = self._apply_eliminate_branches(sdfg, sdfg, None)
            added_scalar_names = added_scalar_names.union(cur_added_scalar_names)
            num_applied += cur_num_applied

        sdfg.validate()
        return num_applied, added_scalar_names

    def report(self, pass_retval: int) -> str:
        return f'Fused (andd removed) {pass_retval} branches.'
