# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that collapses a contiguous per-lane fan-out into a gather intrinsic."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply

# The lane-index buffer uses a DaCe-internal reserved name (``__``
# prefix) rather than a bare ``idx``: the latter collides with the very
# common map/loop parameter name ``idx``, and when a residual scaffolding
# map is removed, ``SDFG.replace_dict`` shadows that token in CPP tasklet
# bodies by prepending ``auto idx = <range-begin>;`` — binding to a
# parent-scope symbol that is not threaded into an out-of-line body
# NestedSDFG (the spmv ``auto idx = tile_idx;`` undeclared-symbol bug).
_GATHER_TEMPLATE = """
{{
int64_t __vec_lane_idx[{vector_length}] = {{ {initializer_values} }};
gather<{dtype}>(_in, __vec_lane_idx, _out, {vector_length});
}}
"""

_GATHER_TEMPLATE_MASKED = """
{{
int64_t __vec_lane_idx[{vector_length}] = {{ {initializer_values} }};
gather_masked<{dtype}>(_in, __vec_lane_idx, _out, {vector_length}, _mask);
}}
"""

# Index-array-direct variants (``collapse_laneid_index_loads``). The W
# per-lane indices are read straight from the index array slice via the
# ``_idx`` connector instead of W interstate-edge laneid symbols.
#
# When the index array is already ``int64`` (the runtime ``gather`` index
# type) the ``_idx`` pointer is passed straight through — no buffer, no
# per-lane fill. The ``*_CONV`` variant is the fallback for a narrower
# index dtype (e.g. ``int32``): the only reason a local buffer exists is
# the element-width conversion the runtime signature requires, not
# regeneration of the indices.
_GATHER_TEMPLATE_IDXARR = """
{{
gather<{dtype}>(_in, _idx, _out, {vector_length});
}}
"""

_GATHER_TEMPLATE_IDXARR_MASKED = """
{{
gather_masked<{dtype}>(_in, _idx, _out, {vector_length}, _mask);
}}
"""

_GATHER_TEMPLATE_IDXARR_CONV = """
{{
constexpr int __VL = {vector_length};
int64_t __vec_lane_idx[__VL];
DACE_UNROLL
for (int __l = 0; __l < __VL; ++__l) __vec_lane_idx[__l] = _idx[__l * {stride}];
gather<{dtype}>(_in, __vec_lane_idx, _out, __VL);
}}
"""

# Masked counterpart: the remainder tile's index window extends past the
# valid index-array tail (lanes ``>= R`` are out of bounds). Read the index
# ONLY for active lanes; masked-off lanes copy lane 0's index (always in
# bounds -- the remainder has at least one valid lane), so the per-lane fill
# never dereferences ``_idx`` out of bounds. The garbage that a masked lane
# would have produced is irrelevant: ``gather_masked`` skips it.
_GATHER_TEMPLATE_IDXARR_CONV_MASKED = """
{{
constexpr int __VL = {vector_length};
int64_t __vec_lane_idx[__VL];
DACE_UNROLL
for (int __l = 0; __l < __VL; ++__l) __vec_lane_idx[__l] = _mask[__l] ? _idx[__l * {stride}] : _idx[0];
gather_masked<{dtype}>(_in, __vec_lane_idx, _out, __VL, _mask);
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectGather(ppl.Pass):
    """Replace a contiguous per-lane load fan-out with a single gather intrinsic."""

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    only_masked = properties.Property(dtype=bool,
                                      default=False,
                                      desc="Collapse only masked (vector-remainder) fan-outs; leave the "
                                      "main loop's per-lane scalar gather untouched.")

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
        """Detect and rewrite gather fan-outs in ``sdfg``.

        :param sdfg: SDFG to transform in place.
        :param pipeline_results: Results of previously run pipeline passes.
        """
        detect_lane_fanout_apply(sdfg,
                                 direction="gather",
                                 pattern="contiguous",
                                 intrinsic_template=_GATHER_TEMPLATE,
                                 intrinsic_template_masked=_GATHER_TEMPLATE_MASKED,
                                 intrinsic_tasklet_name="gather_load",
                                 skip_unmasked=self.only_masked,
                                 collapse_laneid_index_loads=self.collapse_laneid_index_loads,
                                 intrinsic_template_idxarr=_GATHER_TEMPLATE_IDXARR,
                                 intrinsic_template_idxarr_masked=_GATHER_TEMPLATE_IDXARR_MASKED,
                                 intrinsic_template_idxarr_conv=_GATHER_TEMPLATE_IDXARR_CONV,
                                 intrinsic_template_idxarr_conv_masked=_GATHER_TEMPLATE_IDXARR_CONV_MASKED)
        sdfg.validate()
