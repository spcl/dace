# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace import properties
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.loop_lifting import LoopLifting


@properties.make_properties
@transformation.experimental_cfg_block_compatible
class ControlFlowLifting(ppl.Pass):

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def apply_pass(self, top_sdfg: ppl.SDFG, _) -> ppl.Any | None:
        for sdfg in top_sdfg.all_sdfgs_recursive():
            sdfg.apply_transformations_repeated([LoopLifting], validate_all=False, validate=False)
