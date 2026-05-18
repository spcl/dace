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

    def __init__(self, only_masked: bool = False):
        super().__init__()
        self.only_masked = only_masked

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
                                 skip_unmasked=self.only_masked)
        sdfg.validate()
