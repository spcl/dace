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
      - Constant indexes (a*i + b) with a = 0
      - Index expression (a*i + b) with a > 1
      - Backwards loops step < 0 or a < 0
      - Step > 1
      - Iteration variable on multiple dimensions, e.g. a[i, i]

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
        arrays = set(acc_node.data for acc_node in self.loop.data_nodes())
        return any(self._can_apply_for_array(arr) for arr in arrays)

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        arrays = set(acc_node.data for acc_node in self.loop.data_nodes())
        for arr in arrays:
            if self._can_apply_for_array(arr):
                self._apply_for_array(arr)

    def _get_edge_indices(self, subset):
        # tuples of (a, b) for a*i + b, None if cannot be determined
        indices = set()
        itervar = self.loop.loop_variable
        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild("a", exclude=[itersym])
        b = sp.Wild("b", exclude=[itersym])

        for rb, re, _ in subset.ndrange():
            m = rb.match(a * itersym + b)
            if m is not None and rb == re:
                indices.add((m[a], m[b]))
            else:
                indices.add(None)

        return indices

    def _get_read_write_indices(self, array_name: str):
        # tuples of (a, b) for a*i + b, None if cannot be determined
        read_indices = set()
        write_indices = set()

        access_nodes = set(an for an in self.loop.data_nodes() if an.data == array_name)
        read_edges = set(
            e
            for an in access_nodes
            for st in self.loop.all_states()
            for e in st.out_edges(an)
        )
        write_edges = set(
            e
            for an in access_nodes
            for st in self.loop.all_states()
            for e in st.in_edges(an)
        )

        for edge in read_edges:
            added = False
            for ei in self._get_edge_indices(edge.data.src_subset):
                if ei is not None and ei[0] != 0:
                    read_indices.add(ei)
                    added = True
                    break
            if not added:
                read_indices.add(None)

        for edge in write_edges:
            added = False
            for ei in self._get_edge_indices(edge.data.dst_subset):
                if ei is not None and ei[0] != 0:
                    write_indices.add(ei)
                    added = True
                    break
            if not added:
                write_indices.add(None)

        return read_indices, write_indices

    def _check_loop_params(self):
        itervar = self.loop.loop_variable
        step = loop_analysis.get_loop_stride(self.loop)
        if step is None:
            return False

        defined_syms = set()
        for edge in self.loop.all_interstate_edges():
            if isinstance(edge.data, InterstateEdge):
                defined_syms.update(edge.data.assignments.keys())

        return (
            itervar not in defined_syms
            and step.free_symbols.isdisjoint(defined_syms)
            and step.free_symbols.isdisjoint({itervar})
            and symbolic.resolve_symbol_to_constant(step, self.loop.sdfg) == 1
        )

    def _can_apply_for_array(self, array_name: str):
        """
        1. Loop step expression must be constant.

        2. All read and write indices must be linear combinations of the loop variable. I.e. a*i + b, where a and b are constants.

        XXX: For now we only support a = 1, step = 1.

        3. Outside of the loop, the written subset of the array must be written before read or not read at all.

        4. At least one write index must be higher than all read indices (i.e. K > 1).
        """
        if not self._check_loop_params():
            return False

        if any(
            a is None
            for a in self._get_read_write_indices(array_name)[0]
            | self._get_read_write_indices(array_name)[1]
        ):
            return False

        return False

    def _apply_for_array(self, array_name: str):
        pass
