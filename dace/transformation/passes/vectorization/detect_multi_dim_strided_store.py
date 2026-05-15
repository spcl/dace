# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
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
    """
    Collapse the multi-dim linear-combo per-lane fan-out writing to a
    ``_packed`` access node (e.g. ``A[i,i] = ...``, ``A[2*i,i] = ...``,
    ``A[i,2*i] = ...``) into a single ``strided_store_double`` intrinsic call.

    Sibling of :class:`DetectStridedStore` (1D pattern).
    """
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        detect_multi_dim_strided_apply(sdfg,
                                       direction="store",
                                       intrinsic_template=_MULTI_DIM_STRIDED_STORE_TEMPLATE,
                                       intrinsic_template_masked=_MULTI_DIM_STRIDED_STORE_TEMPLATE_MASKED,
                                       intrinsic_tasklet_name="multi_dim_strided_store")
        sdfg.validate()
