# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from typing import Dict, Optional, Set

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation

import ast
from dace.sdfg.nodes import CodeBlock

@transformation.explicit_cf_compatible
class DetectAndRenameSoftHierTasklets(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Tasklets

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.Tasklet):
                if n.code.as_string.strip().startswith("_softhier"):
                    n.label = "_softhier_" + n.label