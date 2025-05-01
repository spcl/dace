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

    Second example:

      for i in range(3, N-2):
          v[i] = a[i + 1] + a[i - 3]
          a[i - 1] = b[i] * 2
          a[i + 2] = b[i] * 2

      i, Reads, Writes:
      0, a4 a0, a2 a5
      1, a5 a1, a3 a6
      2, a6 a2, a4 a7
      3, a7 a3, a5 a8

    Transforms to:

      for i in range(3, N-2):
        v[i] = a[(i + 1) % K] + a[(i - 3) % K]
        a[(i - 1) % K] = b[i] * 2
        a[(i + 2) % K] = b[i] * 2

      i, Reads, Writes:
      0, a4 a0, a2 a5
      1, a5 a1, a3 a0
      2, a0 a2, a4 a1
      3, a1 a3, a5 a2

    With K = 6, which can be determined by the distance of lowest read and highest write: K = hw - lr + 1 = (i + 2) - (i - 3) + 1 = 6.
    If we also analyze the order of reads and writes, we can lower K even further.

    Works with:
      - Interleaved reads and writes
      - Gaps in the range of indices

    Might work with (need to check and how to compute K):
      - Constant indexes (i.e. independent of the loop variable)
      - Index expression (a*i + b) with a != 1
      - Backwards loops step < 0 or a < 0
      - Step != 1

    Does not work with:
      - Non-linear index expressions (a*i + b)
      - Non-linear and non-constant step expressions
      - Indirect accesses
    """

    loop = xf.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return False

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        pass

    def _can_apply_for_array(self, array_name: str):
        """
        1. Loop step expression must be constant.

        2. All read and write indices must be linear combinations of the loop variable. I.e. a*i + b, where a and b are constants.

        XXX: For now a must be 1. Need to figure out if scaling is possible and how to compute K.

        3. Outside of the loop, the written subset of the array must be written before read or not read at all.

        4. At least one write index must be higher than all read indices (i.e. K > 1).
        """
        pass

    def _apply_for_array(self, array_name: str):
        pass
