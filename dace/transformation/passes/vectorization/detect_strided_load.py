# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply


_STRIDED_LOAD_TEMPLATE = """
{{
strided_load<{dtype}>(_in, _out, {vector_length}, {stride});
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectStridedLoad(ppl.Pass):
    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        detect_lane_fanout_apply(sdfg, direction="load", pattern="strided",
                                         intrinsic_template=_STRIDED_LOAD_TEMPLATE,
                                         intrinsic_tasklet_name="gather_load")
        sdfg.validate()
