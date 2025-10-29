# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace import properties, SDFG, nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from typing import Union, Optional


@properties.make_properties
@explicit_cf_compatible
class EliminateBranches(ppl.Pass):
    try_clean = properties.Property(dtype=bool, default=False, allow_none=False)
    clean_only = properties.Property(dtype=bool, default=False, allow_none=True)
    permissive = properties.Property(dtype=bool, default=False, allow_none=False)
    eps_operator_type_for_log_and_div = properties.Property(dtype=str, default="add", allow_none=True)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _apply_eliminate_branches(self, root: SDFG, sdfg: SDFG, parent_nsdfg_state: Union[SDFG, None] = None):
        """Apply EliminateBranches transformation to all eligible conditionals."""
        from dace.transformation.interstate import branch_elimination
        # Pattern matching with conditional branches to not work (9.10.25), avoid it
        # Depending on the number of nestedness we need to apply that many times because
        # the transformation only runs on top-level ConditionalBlocks
        changed = True
        while changed:
            changed = False

            for node in sdfg.all_control_flow_blocks():
                if isinstance(node, ConditionalBlock):
                    t = branch_elimination.BranchElimination()
                    t.conditional = node
                    t.eps_operator_type_for_log_and_div = self.eps_operator_type_for_log_and_div

                    if self.try_clean:
                        t.try_clean(node.parent_graph, sdfg, True)
                        node = t.conditional

            if not self.clean_only:
                for node in sdfg.all_control_flow_blocks():
                    if isinstance(node, ConditionalBlock):
                        t = branch_elimination.BranchElimination()
                        t.conditional = node
                        if node.sdfg.parent_nsdfg_node is not None:
                            t.parent_nsdfg_state = parent_nsdfg_state
                        t.eps_operator_type_for_log_and_div = self.eps_operator_type_for_log_and_div
                        if t.can_be_applied(graph=node.parent_graph,
                                            expr_index=0,
                                            sdfg=node.sdfg,
                                            permissive=self.permissive):
                            t.apply(graph=node.parent_graph, sdfg=node.sdfg)
                            changed = True

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    changed |= self._apply_eliminate_branches(root, node.sdfg, state)

        return changed

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        if self.clean_only is True:
            self.try_clean = True
        self._apply_eliminate_branches(sdfg, sdfg, None)

    def report(self, pass_retval: int) -> str:
        return f'Fused (andd removed) {pass_retval} branches.'
