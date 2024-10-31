# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from dace.transformation import pass_pipeline as ppl, transformation
from dace import properties

@properties.make_properties
@transformation.experimental_cfg_block_compatible
class MemletPropagation(ppl.StatePass):
    """
    TODO
    """

    CATEGORY: str = 'Analysis'

    def modifies(self):
        return ppl.Modifies.Nothing

    def should_reapply(self, modified):
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets)

    def apply(self, state, pipeline_results):
        return super().apply(state, pipeline_results)
