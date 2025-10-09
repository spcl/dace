import dace
from dace import properties, transformation
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
import dace.sdfg.utils as sdutil

from typing import Tuple

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

            joint_writes = write_sets0.intersection(write_sets0)
            diff_state0 = write_sets0.difference(write_sets1)
            diff_state1 = write_sets1.difference(write_sets0)


            # For joint writes ensure the write subsets are always the same
            pass

            # If diff states are empty it is ok

            # If diff states only have transient scalars or arrays it is prob ok (permissive)
            if not permissive:
                if diff_state0 or diff_state1:
                    # TODO
                    pass

        elif len(self.conditional.branches) == 1:
            tup0 : Tuple[properties.CodeBlock, ControlFlowRegion] = self.conditional.branches[0:1]
            cond0, body0 = tup0[0], tup0[1]

            # Works if the branch body has a single state
            if len(body0.nodes()) != 1 or not isinstance(body0.nodes()[0], SDFGState):
                return False

    def apply(self, graph: ControlFlowRegion, sdfg: SDFG):
        pass