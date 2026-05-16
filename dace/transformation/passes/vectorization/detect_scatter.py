# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that collapses a contiguous per-lane fan-out into a scatter intrinsic."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply

_SCATTER_TEMPLATE = """
{{
int64_t idx[{vector_length}] = {{ {initializer_values} }};
scatter<{dtype}>(_in, idx, _out, {vector_length});
}}
"""

_SCATTER_TEMPLATE_MASKED = """
{{
int64_t idx[{vector_length}] = {{ {initializer_values} }};
scatter_masked<{dtype}>(_in, idx, _out, {vector_length}, _mask);
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
                                 skip_unmasked=self.only_masked)
        sdfg.validate()
