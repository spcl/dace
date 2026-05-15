# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that collapses a multi-dim per-lane fan-out into a strided-store intrinsic."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_multi_dim_strided_apply

_MULTI_DIM_STRIDED_STORE_TEMPLATE = """
{{
strided_store<{dtype}>(_in, _out, {vector_length}, {stride});
}}
"""

_MULTI_DIM_STRIDED_STORE_TEMPLATE_MASKED = """
{{
strided_store_masked<{dtype}>(_in, _out, {vector_length}, {stride}, _mask);
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectMultiDimStridedStore(ppl.Pass):
    """Collapse a multi-dim linear-combination store fan-out into one strided-store intrinsic.

    Handles indices that are linear combinations of a single map parameter,
    e.g. ``A[i,i] = ...``, ``A[2*i,i] = ...``, ``A[i,2*i] = ...``.
    """

    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        """Report the SDFG elements this pass may change."""
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Report whether this pass should run again after other passes."""
        return False

    def depends_on(self):
        """Report the passes this pass depends on."""
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        """Detect and rewrite multi-dim strided store fan-outs in ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes.
        """
        detect_multi_dim_strided_apply(sdfg,
                                       direction="store",
                                       intrinsic_template=_MULTI_DIM_STRIDED_STORE_TEMPLATE,
                                       intrinsic_template_masked=_MULTI_DIM_STRIDED_STORE_TEMPLATE_MASKED,
                                       intrinsic_tasklet_name="multi_dim_strided_store")
        sdfg.validate()
