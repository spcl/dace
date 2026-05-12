# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl, transformation

from dace.transformation.passes.vectorization.utils.lane_fanout import detect_lane_fanout_apply


_SCATTER_TEMPLATE = """
{{
int64_t idx[{vector_length}] = {{ {initializer_values} }};
scatter_double(_in, idx, _out, {vector_length});
}}
"""


@properties.make_properties
@transformation.explicit_cf_compatible
class DetectScatter(ppl.Pass):
    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        found = detect_lane_fanout_apply(sdfg, direction="scatter", pattern="contiguous",
                                         intrinsic_template=_SCATTER_TEMPLATE, intrinsic_tasklet_name="scatter_store")
        if found > 0:
            sdfg.append_global_code('#include <stdint.h>\n#include "dace/vector_intrinsics/scatter.h"')
        sdfg.validate()
