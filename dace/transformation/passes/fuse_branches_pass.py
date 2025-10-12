import dace
from dace import properties
from dace import Optional
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.interstate import fuse_branches


@properties.make_properties
@explicit_cf_compatible
class FuseBranchesPass(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return (modified & ppl.Modifies.CFG)

    def _get_nestedness_of_conditional_blocks(self, cfg: ControlFlowRegion, current_depth: int) -> int:
        if isinstance(cfg, SDFGState):
            return current_depth

        for node in cfg.nodes():
            if isinstance(node, ConditionalBlock):
                return max([
                    self._get_nestedness_of_conditional_blocks(body, current_depth + 1) for cond, body in node.branches
                ])
            elif isinstance(node, SDFGState):
                return current_depth
            else:
                return max(
                    [self._get_nestedness_of_conditional_blocks(body, current_depth) for cond, body in node.branches])

        raise Exception("?")

    def _apply_fuse_branches(self, sdfg: dace.SDFG):
        """Apply FuseBranches transformation to all eligible conditionals."""
        # Pattern matching with conditional branches to not work (9.10.25), avoid it
        t = fuse_branches.FuseBranches()
        # Depending on the number of nestedness we need to apply that many times because
        # the transformation only runs on top-level ConditionalBlocks
        nestedness = self._get_nestedness_of_conditional_blocks(sdfg, 0)

        for _ in range(nestedness):
            for node in sdfg.all_control_flow_blocks():
                if isinstance(node, ConditionalBlock):
                    #print(node, t.can_be_applied_to(node.parent_graph, conditional=node))
                    if t.can_be_applied_to(node.parent_graph, conditional=node):
                        t.apply_to(node.parent_graph.sdfg, conditional=node)

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_fuse_branches(node.sdfg)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        self._apply_fuse_branches(sdfg)

    def report(self, pass_retval: int) -> str:
        return f'Fused (andd removed) {pass_retval} branches.'
