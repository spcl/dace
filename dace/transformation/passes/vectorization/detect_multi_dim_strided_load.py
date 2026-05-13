# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_multi_dim_strided_apply


_MULTI_DIM_STRIDED_LOAD_TEMPLATE = """
{{
strided_load<{dtype}>(_in, _out, {vector_length}, {stride});
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectMultiDimStridedLoad(ppl.Pass):
    """
    Collapse the multi-dim linear-combo per-lane fan-out around a ``_packed``
    access node (e.g. ``A[i,i]``, ``A[2*i,i]``, ``A[i,2*i]``) into a single
    ``strided_load_double`` intrinsic call.

    Sibling of :class:`DetectStridedLoad` (1D pattern); both delegate to
    ``detect_lane_fanout_apply`` / ``detect_multi_dim_strided_apply``.
    """
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        detect_multi_dim_strided_apply(sdfg, direction="load",
                                               intrinsic_template=_MULTI_DIM_STRIDED_LOAD_TEMPLATE,
                                               intrinsic_tasklet_name="multi_dim_strided_load")
        sdfg.validate()
