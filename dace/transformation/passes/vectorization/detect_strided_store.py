# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that collapses a strided per-lane fan-out into a strided-store intrinsic."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply

_STRIDED_STORE_TEMPLATE = """
{{
strided_store<{dtype}, {vector_length}>(_in, _out, {stride});
}}
"""

_STRIDED_STORE_TEMPLATE_MASKED = """
{{
strided_store_masked<{dtype}, {vector_length}>(_in, _out, {stride}, _mask);
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectStridedStore(ppl.Pass):
    """Replace a strided per-lane store fan-out with a single strided-store intrinsic."""

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        """Detect and rewrite strided store fan-outs in ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes.
        """
        detect_lane_fanout_apply(sdfg,
                                 direction="store",
                                 pattern="strided",
                                 intrinsic_template=_STRIDED_STORE_TEMPLATE,
                                 intrinsic_template_masked=_STRIDED_STORE_TEMPLATE_MASKED,
                                 intrinsic_tasklet_name="scatter_store")
        sdfg.validate()
