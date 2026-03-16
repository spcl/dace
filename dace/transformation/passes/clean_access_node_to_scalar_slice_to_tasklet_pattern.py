# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace.transformation import pass_pipeline as ppl
from dace.sdfg import utils as sdutil
from typing import Optional
import copy
from dace.transformation.transformation import explicit_cf_compatible


@dace.properties.make_properties
@explicit_cf_compatible
class CleanAccessNodeToScalarSliceToTaskletPattern(ppl.Pass):
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _check_pattern(self, state: dace.SDFGState, access_node: dace.nodes.AccessNode):
        # Requirements:
        # Path exists: an1 (access_node) -> an2 -> tasklet
        # In and out degree of an2 is 1
        # an2 is scalar or array of length 1, also transient
        # an2 access susbet is [0]
        # Optional: should check if it is reused
        sdfg = state.sdfg

        # an1 must be an AccessNode
        if not isinstance(access_node, dace.nodes.AccessNode):
            return None, None, None

        # For each successor of an1, check if it's a candidate an2
        for e1 in state.out_edges(access_node):
            an2 = e1.dst
            if not isinstance(an2, dace.nodes.AccessNode):
                continue

            # an2 must be transient
            desc = sdfg.arrays.get(an2.data)
            if desc is None or not desc.transient:
                continue

            # an2 must be scalar or array of total size 1
            if not (isinstance(desc, dace.data.Scalar) or (isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and desc.total_size == 1)):
                continue

            # an2 must have exactly one incoming and one outgoing edge
            if state.in_degree(an2) != 1 or state.out_degree(an2) != 1:
                continue

            # The outgoing edge of an2 must go to a Tasklet
            e2 = state.out_edges(an2)[0]
            tasklet = e2.dst
            if not isinstance(tasklet, dace.nodes.Tasklet):
                continue

            # an2's access subset (on the outgoing edge) should be [0]
            if e2.data.subset != dace.subsets.Range([(0,0,1)]):
                continue

            return access_node, an2, tasklet

        return None, None, None

    def _apply_recursive(self, sdfg: dace.SDFG):
        # TODO: Implement a check for when the scalar is reused later
        for state in sdfg.all_states():
            pre_transform_state_nodes = state.nodes()
            for node in pre_transform_state_nodes:
                # Might be already removed
                if node not in state.nodes():
                    continue
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)
                else:
                    # Look for the pattern
                    an1, an2, tasklet = self._check_pattern(state, node)
                    if an1 is not None and an2 is not None and tasklet is not None:
                        # Remove an2, rewrite to tasklet
                        ies = state.in_edges(an2)
                        oes = state.out_edges(an2)
                        assert len(oes) == 1 and len(ies) == 1
                        oe = oes[0]
                        ie = ies[0]
                        assert oe.dst == tasklet
                        state.remove_node(an2)
                        state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(ie.data))


    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        self._apply_recursive(sdfg)

