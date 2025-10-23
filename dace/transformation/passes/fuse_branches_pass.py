import dace
from dace import properties
from dace import Optional
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.interstate import fuse_branches
from typing import Union


@properties.make_properties
@explicit_cf_compatible
class FuseBranchesPass(ppl.Pass):
    try_clean = properties.Property(dtype=bool, default=False, allow_none=False)
    clean_only = properties.Property(dtype=bool, default=False, allow_none=True)
    permissive = properties.Property(dtype=bool, default=False, allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return (modified & ppl.Modifies.CFG)

    def _apply_fuse_branches(self, root: dace.SDFG, sdfg: dace.SDFG, parent_nsdfg_state: Union[dace.SDFG, None] = None):
        """Apply FuseBranches transformation to all eligible conditionals."""
        # Pattern matching with conditional branches to not work (9.10.25), avoid it
        # Depending on the number of nestedness we need to apply that many times because
        # the transformation only runs on top-level ConditionalBlocks
        changed = True
        while changed:
            changed = False

            for node in sdfg.all_control_flow_blocks():
                if isinstance(node, ConditionalBlock):
                    t = fuse_branches.FuseBranches()
                    t.conditional = node
                    if self.try_clean:
                        t.try_clean(node.parent_graph, sdfg)
                        node = t.conditional

            if not self.clean_only:
                for node in sdfg.all_control_flow_blocks():
                    if isinstance(node, ConditionalBlock):
                        t = fuse_branches.FuseBranches()
                        t.conditional = node
                        if node.sdfg.parent_nsdfg_node is not None:
                            t.parent_nsdfg_state = parent_nsdfg_state
                        if t.can_be_applied(graph=node.parent_graph,
                                            expr_index=0,
                                            sdfg=node.sdfg,
                                            permissive=self.permissive):
                            t.apply(graph=node.parent_graph, sdfg=node.sdfg)
                            changed = True

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    changed |= self._apply_fuse_branches(root, node.sdfg, state)

        return changed

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        if self.clean_only is True:
            self.try_clean = True
        self._apply_fuse_branches(sdfg, sdfg, None)

    def report(self, pass_retval: int) -> str:
        return f'Fused (andd removed) {pass_retval} branches.'
