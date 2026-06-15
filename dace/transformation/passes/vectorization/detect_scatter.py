# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that collapses a contiguous per-lane fan-out into a scatter intrinsic."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply

# Reserved ``__vec_lane_idx`` name (not bare ``idx``) — see the rationale
# in ``detect_gather.py``: a bare ``idx`` collides with the common
# map-param name and gets shadow-bound to a parent-scope symbol by
# ``SDFG.replace_dict`` when a scaffolding map is removed.
_SCATTER_TEMPLATE = """
{{
int64_t __vec_lane_idx[{vector_length}] = {{ {initializer_values} }};
scatter<{dtype}>(_in, __vec_lane_idx, _out, {vector_length});
}}
"""

_SCATTER_TEMPLATE_MASKED = """
{{
int64_t __vec_lane_idx[{vector_length}] = {{ {initializer_values} }};
scatter_masked<{dtype}>(_in, __vec_lane_idx, _out, {vector_length}, _mask);
}}
"""

# Index-array-direct variants (``collapse_laneid_index_loads``). The W
# per-lane scatter indices are read straight from the index array slice
# via the ``_idx`` connector instead of W interstate-edge laneid
# symbols.
#
# When the index array is already ``int64`` the ``_idx`` pointer is
# passed straight through. The ``*_CONV`` variant is the fallback for a
# narrower index dtype (e.g. ``int32``): the local buffer exists only
# for the element-width conversion the runtime signature requires.
_SCATTER_TEMPLATE_IDXARR = """
{{
scatter<{dtype}>(_in, _idx, _out, {vector_length});
}}
"""

_SCATTER_TEMPLATE_IDXARR_MASKED = """
{{
scatter_masked<{dtype}>(_in, _idx, _out, {vector_length}, _mask);
}}
"""

_SCATTER_TEMPLATE_IDXARR_CONV = """
{{
int64_t __vec_lane_idx[{vector_length}];
for (int __l = 0; __l < {vector_length}; ++__l) __vec_lane_idx[__l] = _idx[__l * {stride}];
scatter<{dtype}>(_in, __vec_lane_idx, _out, {vector_length});
}}
"""

# Masked counterpart: the remainder tile's index window extends past the
# valid index-array tail (lanes ``>= R`` are out of bounds). Read the index
# ONLY for active lanes; masked-off lanes copy lane 0's index (always in
# bounds -- the remainder has at least one valid lane), so the per-lane fill
# never dereferences ``_idx`` out of bounds. The destination written by a
# masked lane is irrelevant: ``scatter_masked`` skips it.
_SCATTER_TEMPLATE_IDXARR_CONV_MASKED = """
{{
int64_t __vec_lane_idx[{vector_length}];
for (int __l = 0; __l < {vector_length}; ++__l) __vec_lane_idx[__l] = _mask[__l] ? _idx[__l * {stride}] : _idx[0];
scatter_masked<{dtype}>(_in, __vec_lane_idx, _out, {vector_length}, _mask);
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectScatter(ppl.Pass):
    """Replace a contiguous per-lane store fan-out with a single scatter intrinsic."""

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    only_masked = properties.Property(dtype=bool,
                                      default=False,
                                      desc="Collapse only masked (vector-remainder) fan-outs; leave the "
                                      "main loop's per-lane scalar scatter untouched.")

    collapse_laneid_index_loads = properties.Property(
        dtype=bool,
        default=False,
        desc="Collapse the per-lane laneid index fan into a direct index-array "
        "slice read (_idx connector) and drop the now-dead laneid symbols + "
        "their interstate-edge assignments.")

    def __init__(self, only_masked: bool = False, collapse_laneid_index_loads: bool = False):
        super().__init__()
        self.only_masked = only_masked
        self.collapse_laneid_index_loads = collapse_laneid_index_loads

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        """Detect and rewrite scatter fan-outs in ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes.
        """
        detect_lane_fanout_apply(sdfg,
                                 direction="scatter",
                                 pattern="contiguous",
                                 intrinsic_template=_SCATTER_TEMPLATE,
                                 intrinsic_template_masked=_SCATTER_TEMPLATE_MASKED,
                                 intrinsic_tasklet_name="scatter_store",
                                 skip_unmasked=self.only_masked,
                                 collapse_laneid_index_loads=self.collapse_laneid_index_loads,
                                 intrinsic_template_idxarr=_SCATTER_TEMPLATE_IDXARR,
                                 intrinsic_template_idxarr_masked=_SCATTER_TEMPLATE_IDXARR_MASKED,
                                 intrinsic_template_idxarr_conv=_SCATTER_TEMPLATE_IDXARR_CONV,
                                 intrinsic_template_idxarr_conv_masked=_SCATTER_TEMPLATE_IDXARR_CONV_MASKED)
        sdfg.validate()
