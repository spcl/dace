# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import sympy as sp
from dace import sdfg as sd, symbolic, properties
from dace.sdfg import InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis, StateReachability
from dace.symbolic import pystr_to_symbolic
from dace.subsets import Range
import copy


@properties.make_properties
@xf.explicit_cf_compatible
class LoopLocalMemoryReduction(xf.MultiStateTransformation):
    """
    This transformation replaces thread-local array accesses in a loop with a modulo-indexed access to reduce memory footprint.
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
      - Iteration variable on multiple dimensions, e.g. a[i, i]

    Might work with (need to check and add support):
      - Constant indexes (a*i + b) with a = 0
      - Index expression (a*i + b) with a > 1
      - Backwards loops step < 0 or a < 0
      - Step > 1

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

    def can_be_applied(self, graph: ControlFlowRegion, expr_index, sdfg: sd.SDFG, permissive=False):
        if not isinstance(self.loop, LoopRegion):
            return False
        arrays = set(acc_node.data for acc_node in self.loop.data_nodes())
        return any(self._can_apply_for_array(arr, sdfg) for arr in arrays)

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        arrays = set(acc_node.data for acc_node in self.loop.data_nodes())
        for arr in arrays:
            if self._can_apply_for_array(arr, sdfg):
                self._apply_for_array(arr, sdfg)

    def _get_edge_indices(self, subset: Range) -> list[tuple | None]:
        # list of tuples of (a, b) for a*i + b, None if cannot be determined
        indices = list()
        itervar = self.loop.loop_variable
        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild("a", exclude=[itersym])
        b = sp.Wild("b", exclude=[itersym])

        for rb, re, _ in subset.ndrange():
            m = rb.match(a * itersym + b)
            if m is not None and rb == re:
                indices.append((m[a], m[b]))
            else:
                indices.append(None)

        return indices

    def _get_read_write_indices(self, array_name: str) -> tuple[list[list[tuple | None]], list[list[tuple | None]]]:
        # list of list of tuples of (a, b) for a*i + b
        read_indices = list()
        write_indices = list()

        access_nodes = set(an for an in self.loop.data_nodes() if an.data == array_name)
        read_edges = set(e for an in access_nodes for st in self.loop.all_states() if an in st.data_nodes()
                         for e in st.out_edges(an))
        write_edges = set(e for an in access_nodes for st in self.loop.all_states() if an in st.data_nodes()
                          for e in st.in_edges(an))

        for edge in read_edges:
            eri = self._get_edge_indices(edge.data.src_subset)
            read_indices.append(eri)

        for edge in write_edges:
            ewi = self._get_edge_indices(edge.data.dst_subset)
            write_indices.append(ewi)

        return read_indices, write_indices

    def _has_constant_loop_expressions(self, sdfg: sd.SDFG) -> bool:
        itervar = self.loop.loop_variable
        step = loop_analysis.get_loop_stride(self.loop)
        if step is None:
            return False

        defined_syms = set()
        for edge in self.loop.all_interstate_edges():
            defined_syms.update(edge.data.assignments.keys())

        # TODO: For now we only support step = 1.
        return (itervar not in defined_syms and step.free_symbols.isdisjoint(defined_syms)
                and step.free_symbols.isdisjoint({itervar}) and symbolic.resolve_symbol_to_constant(step, sdfg) == 1)

    def _get_K_values(self, array_name: str, read_indices: list[list[tuple | None]],
                      write_indices: list[list[tuple | None]], sdfg: sd.SDFG) -> list[int | None]:
        k_values = []
        max_indices = self._get_max_indices_before_loop(array_name, sdfg)

        # For each dimension
        for dim in range(len(read_indices[0])):
            # Get all read and write indices for this dimension
            dim_read_indices = [il[dim] for il in read_indices if il[dim] is not None]
            dim_write_indices = [il[dim] for il in write_indices if il[dim] is not None]

            if not dim_read_indices or not dim_write_indices:
                k_values.append(None)
                continue

            # Get the minimum read index and maximum write index
            read_lb = min([i[1] for i in dim_read_indices])
            read_ub = max([i[1] for i in dim_read_indices])
            write_ub = max([i[1] for i in dim_write_indices])

            # TODO: What we really should do here is find the write linear function, which eventually dominates all reads, yet leads to the smallest K.

            # K = hw - lr
            # Take maximum from previous accesses into account
            k = sp.Max(write_ub - read_lb, max_indices[dim])
            k = symbolic.resolve_symbol_to_constant(k, sdfg)

            # Write upper bound must be higher than read upper bound
            if k is None or read_ub >= write_ub or k >= sdfg.arrays[array_name].shape[dim]:
                k_values.append(None)
            else:
                k_values.append(k + 1)  # +1 because k is the highest index accessed, so size is k+1
        return k_values

    def _write_is_loop_local(self, array_name: str, write_indices: list[list[tuple]], sdfg: sd.SDFG) -> bool:
        # To obtain the written subset, use the lower bound and upper bound of the write indices.
        init = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)

        # Cannot be determined
        if init is None or end is None:
            return False

        # Must be transient (otherwise it's observable)
        if not sdfg.arrays[array_name].transient:
            return False

        # The (overapproximated) written subset must be written before read or not read at all.
        # TODO: This is overly conservative. Just checks if there are access nodes after the loop.

        can_reach = StateReachability().apply_pass(sdfg, {})
        loop_states = set(self.loop.all_states())

        for k1, v1 in can_reach.items():
            for k2, v2 in v1.items():
                if k2 not in loop_states:
                    continue
                for st in v2:
                    if st not in loop_states:
                        # Access outside of the loop
                        return False
        return True

    def _get_max_indices_before_loop(self, array_name: str, sdfg: sd.SDFG) -> list[int]:
        # Collect all read and write subsets of the array before the loop.
        can_reach = StateReachability().apply_pass(sdfg, {})
        loop_states = set(self.loop.all_states())
        subsets = set()
        for k1, v1 in can_reach.items():
            for k2, v2 in v1.items():
                if len(v2.intersection(loop_states)) == 0 or k2 in loop_states:
                    continue
                access_nodes = set(an for an in k2.data_nodes() if an.data == array_name)

                # Replace loop variables in subsets with their max value.
                pgraph = k2.parent_graph
                replacement = {}
                while isinstance(pgraph, LoopRegion):
                    loop_end = loop_analysis.get_loop_end(pgraph)
                    replacement[pgraph.loop_variable] = loop_end
                    pgraph = pgraph.parent_graph
                replacement.update(sdfg.constants)

                for acc in access_nodes:
                    write_subsets = set(e.data.dst_subset for e in k2.in_edges(acc))
                    read_subsets = set(e.data.src_subset for e in k2.out_edges(acc))
                    for s in write_subsets.union(read_subsets):
                        s2 = copy.deepcopy(s)
                        s2.replace(replacement)
                        subsets.add(s2)

        # For each dimension, get the maximum index accessed.
        indices = []
        for dim in range(len(sdfg.arrays[array_name].shape)):
            max_index = 0
            for subset in subsets:
                rb, re, _ = subset.ndrange()[dim]
                max_index = sp.Max(max_index, re)
            max_index = symbolic.resolve_symbol_to_constant(max_index, sdfg)

            if max_index is None:
                indices.append(sdfg.arrays[array_name].shape[dim])
            else:
                indices.append(max_index)
        return indices

    def _can_apply_for_array(self, array_name: str, sdfg: sd.SDFG) -> bool:
        # Loop step expression must be constant.
        if not self._has_constant_loop_expressions(sdfg):
            return False

        # All read and write indices must be linear combinations of the loop variable. I.e. a*i + b, where a and b are constants.
        # TODO: For now we only support a = 1.
        read_indices, write_indices = self._get_read_write_indices(array_name)
        if any(i is None or i[0] != 1 for il in read_indices + write_indices for i in il):
            return False
        
        # There needs to be at least one read and one write.
        if not read_indices or not write_indices:
            return False

        # Outside of the loop, the written subset of the array must be written before read or not read at all.
        if not self._write_is_loop_local(array_name, write_indices, sdfg):
            return False

        # At least one write index must be higher than all read indices for a dimension (i.e. K >= 1).
        if all([k is None for k in self._get_K_values(array_name, read_indices, write_indices, sdfg)]):
            return False

        # Otherwise, we can apply the transformation.
        return True

    def _apply_for_array(self, array_name: str, sdfg: sd.SDFG):
        read_indices, write_indices = self._get_read_write_indices(array_name)
        Ks = self._get_K_values(array_name, read_indices, write_indices, sdfg)

        # Replace all read and write edges in the loop with modulo accesses.
        read_edges = set(e for st in sdfg.all_states() for an in st.data_nodes() if an.data == array_name
                         for e in st.out_edges(an))
        write_edges = set(e for st in sdfg.all_states() for an in st.data_nodes() if an.data == array_name
                          for e in st.in_edges(an))

        for edge in read_edges:
            subset = edge.data.src_subset
            for i, k in enumerate(Ks):
                if k is None:
                    continue
                lb = pystr_to_symbolic(f"({subset[i][0]}) % ({k})")
                ub = pystr_to_symbolic(f"({subset[i][1]}) % ({k})")
                st = subset[i][2]
                subset[i] = (lb, ub, st)
            edge.data.src_subset = subset

        for edge in write_edges:
            subset = edge.data.dst_subset
            for i, k in enumerate(Ks):
                if k is None:
                    continue
                lb = pystr_to_symbolic(f"({subset[i][0]}) % ({k})")
                ub = pystr_to_symbolic(f"({subset[i][1]}) % ({k})")
                st = subset[i][2]
                subset[i] = (lb, ub, st)
            edge.data.dst_subset = subset

        # Reduce the size of the array in the SDFG.
        array = sdfg.arrays[array_name]
        new_shape = list(array.shape)
        for k in Ks:
            if k is not None:
                new_shape[i] = k
        array.set_shape(tuple(new_shape))
