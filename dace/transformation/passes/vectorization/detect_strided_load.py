# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that collapses a strided per-lane fan-out into a strided-load intrinsic."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply

_STRIDED_LOAD_TEMPLATE = """
{{
strided_load<{dtype}, {vector_length}>(_in, _out, {stride});
}}
"""

_STRIDED_LOAD_TEMPLATE_MASKED = """
{{
strided_load_masked<{dtype}, {vector_length}>(_in, _out, {stride}, _mask);
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectStridedLoad(ppl.Pass):
    """Replace a strided per-lane load fan-out with a single strided-load intrinsic."""

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        """Detect and rewrite strided load fan-outs in ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes.
        """
        detect_lane_fanout_apply(sdfg,
                                 direction="load",
                                 pattern="strided",
                                 intrinsic_template=_STRIDED_LOAD_TEMPLATE,
                                 intrinsic_template_masked=_STRIDED_LOAD_TEMPLATE_MASKED,
                                 intrinsic_tasklet_name="gather_load")
        sdfg.validate()
