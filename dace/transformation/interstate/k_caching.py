# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import sympy as sp
from typing import Set, Optional

from dace import sdfg as sd, symbolic, properties
from dace.sdfg import SDFG, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.sdfg.nodes import CodeBlock
from dace.symbolic import pystr_to_symbolic


@properties.make_properties
@xf.explicit_cf_compatible
class KCaching(xf.MultiStateTransformation):
    """
    Apply K-Caching on loops. K-Caching is a technique to replace thread-local accesses in a loop with a modulo-indexed access to reduce memory footprint.
    Example:
    
      for i in range(2, N):
        v[i] = a[i - 1] + a[i - 2] 
        a[i] = b[i] * 2 

      i, Reads, Writes:
      0, a1 a0, a2
      1, a2 a1, a3
      2, a3 a2, a4
  
    Transforms to:

      for i in range(2, N):
        v[i] = a[(i - 1) % K] + a[(i - 2) % K] 
        a[i % K] = b[i] * 2 

      i, Reads, Writes:
      0, a1 a0, a2
      1, a2 a1, a0
      2, a0 a2, a1

    With K = 3 the memory footprint is reduced from O(N) to O(K).
    We can only do this, if 
    """

    loop = xf.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        pass

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        pass
