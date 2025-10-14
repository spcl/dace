# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from dace import SDFG, InterstateEdge
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes import analysis as ap
from dace import properties
import dace.sdfg.construction_utils as cutil


@transformation.explicit_cf_compatible
class RemoveReduntantCopyTasklets(ppl.Pass):
    copy_tasklet_pattern = properties.Property(dtype=str, default='', allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _arr_appears_in_loop(self, arr_name: str, loop: LoopRegion):
        pass

    def _arr_appears_in_conditional(self, arr_name: str, conditional: ConditionalBlock):
        pass

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        # If AccessNode (A1) -> CopyTasklet -> AccessNode (A2)
        # If copy tasklet matches the pattern e.g. `{rhs} = {lhs}` or `vector_copy({rhs}, {lhs});`
        # If access node is not accessed anywhere at all (not appearing as a memlet data, interstate edge read, for/ifblocks)
        pass

    def report(self, pass_retval: Any) -> Optional[str]:
        return f''
