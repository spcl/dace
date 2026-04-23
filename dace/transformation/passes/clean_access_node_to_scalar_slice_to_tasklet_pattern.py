import dace
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from typing import Optional, Set, Dict
import copy
from dace.transformation.transformation import explicit_cf_compatible


@dace.properties.make_properties
@explicit_cf_compatible
class CleanAccessNodeToScalarSliceToTaskletPattern(ppl.Pass):
    permissive = dace.properties.Property(
        dtype=bool, default=False, desc="If permissive the pass does not check if scalar is used in other states")

    def __init__(self, permissive: bool = False):
        self.permissive = permissive

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _check_pattern(self, state: dace.SDFGState, out_edge: MultiConnectorEdge[dace.Memlet],
                       access_node: dace.nodes.AccessNode, used_elsewhere: Set[str]):
        sdfg = state.sdfg

        if not isinstance(access_node, dace.nodes.AccessNode):
            return None, None, None

        e1 = out_edge
        if e1.data is None or e1.data.subset is None:
            return None, None, None

        if e1.data.wcr is not None:
            return None, None, None

        an2 = e1.dst
        if not isinstance(an2, dace.nodes.AccessNode):
            return None, None, None

        desc = sdfg.arrays.get(an2.data)
        if desc is None or not desc.transient:
            return None, None, None

        if not (isinstance(desc, dace.data.Scalar) or
                (isinstance(desc, dace.data.Array) and len(desc.shape) == 1 and desc.total_size == 1)):
            return None, None, None

        if state.in_degree(an2) != 1 or state.out_degree(an2) != 1:
            return None, None, None

        e2 = state.out_edges(an2)[0]
        tasklet = e2.dst
        if not isinstance(tasklet, dace.nodes.Tasklet):
            return None, None, None

        if e2.data.subset != dace.subsets.Range([(0, 0, 1)]):
            return None, None, None

        # Do not remove if the scalar is read or written in any other state
        if not self.permissive:
            if an2.data in used_elsewhere:
                return None, None, None

        return access_node, an2, tasklet

    def _collect_other_state_data(self, sdfg: dace.SDFG, current_state: dace.SDFGState) -> Set[str]:
        """Collect all data names that appear in any state other than current_state."""
        names = set()
        for state in sdfg.all_states():
            if state is current_state:
                continue
            for dn in state.data_nodes():
                names.add(dn.data)
        return names

    def _apply_recursive(self, sdfg: dace.SDFG):
        # Pre-compute per-state: which data names appear in other states
        all_states = list(sdfg.all_states())
        # Global set of all data names across all states
        if not self.permissive:
            global_data: Dict[str, Set[dace.SDFGState]] = {}
            for state in all_states:
                for dn in state.data_nodes():
                    global_data.setdefault(dn.data, set()).add(state)

        for state in all_states:
            # Data names used in at least one other state
            if not self.permissive:
                used_elsewhere = {name for name, states in global_data.items() if states - {state}}
            else:
                used_elsewhere = set()

            pre_transform_state_nodes = list(state.nodes())
            for node in pre_transform_state_nodes:
                if node not in state.nodes():
                    continue
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply_recursive(node.sdfg)
                else:
                    for e in state.out_edges(node):
                        an1, an2, tasklet = self._check_pattern(state, e, node, used_elsewhere)
                        if an1 is not None and an2 is not None and tasklet is not None:
                            ies = state.in_edges(an2)
                            oes = state.out_edges(an2)
                            assert len(oes) == 1 and len(ies) == 1
                            oe = oes[0]
                            ie = ies[0]
                            assert oe.dst == tasklet
                            state.remove_node(an2)

                            # find correct subset
                            if ie.data.data == an1.data:
                                new_subset = copy.deepcopy(ie.data.subset)
                            else:
                                assert ie.data.data == an2.data
                                assert ie.data.other_subset is not None
                                new_subset = copy.deepcopy(ie.data.other_subset)
                            state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn,
                                           dace.memlet.Memlet(data=an1.data, subset=new_subset))

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        # TODO: add a test involving other subset and then one not involving
        # TODO: Add a test for multiple edges come out from the src
        self._apply_recursive(sdfg)
