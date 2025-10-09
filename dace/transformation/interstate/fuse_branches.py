import dace
from dace import properties, transformation
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
import dace.sdfg.utils as sdutil
import dace.sdfg.construction_utils as cutil
from typing import Tuple, Set

@properties.make_properties
class FuseBranches(transformation.MultiStateTransformation):
    """
    We have the pattern:
    ```
    if (cond){
        out1[address1] = computation1(...)
    } else {
        out1[address2] = computation2(...)
    }
    ```

    If all the write sets in the left and right branches are the same,
    menaing the address1 == address2,
    we can transformation the if branch to:
    ```
    fcond = float(cond)
    out1[address1] = computation1(...) * fcond + (1.0 - fcond) * computation2(...)
    ```

    For single branch case:
    ```
    out1[address1] = computation1(...) * fcond + (1.0 - fcond) * out1[address1]
    ```

    This eliminates branching by duplicating the computation of each branch
    but makes it possible to vectorize the computation.
    """
    conditional = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def _check_reuse(self, sdfg: dace.SDFG, orig_state: dace.SDFGState, diff_set: Set[str]):
        for graph in sdfg.all_control_flow_regions():
            if (not isinstance(graph, dace.SDFGState)) and orig_state in graph.all_states():
                continue
            if graph == orig_state:
                continue
            read_set, write_set = graph.read_and_write_sets()
            if any({k in read_set or k in write_set for k in diff_set}):
                return True

        return False

    def can_be_applied(self,
                       graph: ControlFlowRegion,
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:
        # Works for if-else branches or only if branches
        if len(self.conditional.branches) > 2:
            return False

        if len(self.conditional.branches) == 2:
            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]
            if cond0 is not None and cond1 is not None:
                return False
            assert not (cond0 is None and cond1 is None)

            # Works if the branch bodies have a single state each
            for body in [body0, body1]:
                if len(body.nodes()) != 1 or not isinstance(body.nodes()[0], SDFGState):
                    return False

            # Check write sets are equivalent
            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets0, write_sets1 = state1.read_and_write_stes()

            joint_writes = write_sets0.intersection(write_sets1)
            diff_state0 = write_sets0.difference(write_sets1)
            diff_state1 = write_sets1.difference(write_sets0)


            # For joint writes ensure the write subsets are always the same
            for write in joint_writes:
                state0_accesses = {n for n in state0.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}
                state1_accesses = {n for n in state1.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}

                # If there are more than one writes we can't fuse them together without knowing how to order
                if len(state0_accesses) > 0 or len(state1_accesses):
                    return False

                state0_writes = set()
                state1_writes = set()
                for state_writes, state_accesses, state in [(state0_writes, state0_accesses, state0), (state1_writes, state1_accesses, state1)]:
                    for an in state_accesses:
                        state_write_edges = state_write_edges.union({e for e in state.in_edges(an) if e.data.data is not None})
                        # If there are multiple write edges again we would need to know the order
                        if len(state_write_edges) > 0:
                            return False
                        state_writes = state_writes.union({e.data.data for e in state_write_edges})

                # If the subset of each branch is different then we can't fuse either
                # This can be extended (TODO)
                if state0_writes != state1_writes:
                    return False

            # If diff states only have transient scalars or arrays it is prob ok (permissive)
            if diff_state0 or diff_state1:
                if self._check_reuse(sdfg, state0, diff_state0):
                    return False
                if self._check_reuse(sdfg, state1, diff_state1):
                    return False

        elif len(self.conditional.branches) == 1:
            tup0 : Tuple[properties.CodeBlock, ControlFlowRegion] = self.conditional.branches[0:1]
            cond0, body0 = tup0[0], tup0[1]

            # Works if the branch body has a single state
            if len(body0.nodes()) != 1 or not isinstance(body0.nodes()[0], SDFGState):
                return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        # If CFG has 1 or two branches
        # If two branches then the write sets to sink nodes are the same

        # Strategy copy the nodes of the states to the new state
        if len(self.conditional.branches) == 2:

            tup0 = self.conditional.branches[0]
            tup1 = self.conditional.branches[1]
            (cond0, body0) = tup0[0], tup0[1]
            (cond1, body1) = tup1[0], tup1[1]
            
            cond_cond = cond0 if cond0 is not None else cond1

            state0: SDFGState = body0.nodes()[0]
            state1: SDFGState = body1.nodes()[0]

            new_state = dace.SDFGState(f"fused_{state0.label}_and_{state1.label}")
            state0_to_new_state_node_map = cutil.copy_state_contents(state0, new_state)
            state1_to_new_state_node_map = cutil.copy_state_contents(state1, new_state)

            cond_prep_state = None
            srcs = {e.src for e in graph.in_edges(self.conditional) if isinstance(e.src, dace.SDFGState)}
            if len(srcs) > 0:
                cond_prep_state = srcs.pop()
            else:
                cond_prep_state = graph.add_state_before(self.conditional, f"cond_prep_for_fused_{state0.label}_and_{state1.label}")

            assert cond_prep_state is not None
            # TODO normalize conditions before
            


            # State1, State0 write sets are data names which should be present in the new state too
            read_sets0, write_sets0 = state0.read_and_write_sets()
            read_sets0, write_sets1 = state1.read_and_write_stes()

            joint_writes = write_sets0.intersection(write_sets1)

            for write in joint_writes:
                state0_accesses = {n for n in new_state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}
                state1_accesses = {n for n in state1.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == write}

                state0_accesses_in_new_state = {state0_to_new_state_node_map[n] for n in state0_accesses}
                state1_accesses_in_new_state = {state1_to_new_state_node_map[n] for n in state1_accesses}

                assert len(state0_accesses_in_new_state) == 1
                assert len(state1_accesses_in_new_state) == 1

                state0_in_new_state_access = state0_accesses_in_new_state.pop()
                state1_in_new_state_access = state1_accesses_in_new_state.pop()







            # Need to fuse left and right states

        pass