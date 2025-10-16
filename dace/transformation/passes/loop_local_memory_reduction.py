# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import sympy as sp
from dace import sdfg as sd, symbolic, properties
from dace import data as dt
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ConditionalBlock
from dace.data import Scalar
from dace.transformation import transformation as xf
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis, StateReachability, FindAccessStates
from dace.symbolic import pystr_to_symbolic, issymbolic
from dace.subsets import Range
import copy
from typing import Union, Set, Optional, Dict, Any


@properties.make_properties
@xf.explicit_cf_compatible
class LoopLocalMemoryReduction(ppl.Pass):
    """
    This pass replaces thread-local array accesses in a loop with a modulo-indexed access to reduce memory footprint.
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

    # With K = 3 the memory footprint is reduced from O(N) to O(K).

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
      - Constant indexes (a*i + b) with a = 0
      - Index expression (a*i + b) with a > 1
      - Backwards loops step < 0 or a < 0
      - Step > 1

    Does not work with:
      - Non-linear index expressions (i.e. expression NOT in the form of a*i + b)
      - Non-linear and non-constant step expressions
      - Indirect accesses
    """

    bitmask_indexing = properties.Property(
        dtype=bool,
        default=True,
        desc="Whether or not to use bitmasking for modulo operations when the reduced memory size is a power of two.",
    )

    next_power_of_two = properties.Property(
        dtype=bool,
        default=True,
        desc=
        "Whether or not to round up the reduced memory size to the next power of two (enables bitmasking instead of modulo).",
    )

    num_applications = 0  # To track number of applications for testing

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def depends_on(self):
        return {StateReachability, FindAccessStates}

    def apply_pass(self, sdfg: sd.SDFG, pipeline_results: Dict[str, Any]) -> Optional[Set[str]]:
        self.num_applications = 0
        self.out_of_loop_states_cache = {}

        if StateReachability.__name__ in pipeline_results:
            self.states_reach = pipeline_results[StateReachability.__name__]
        else:
            self.states_reach = StateReachability().apply_pass(sdfg, {})

        if FindAccessStates.__name__ in pipeline_results:
            self.access_states = pipeline_results[FindAccessStates.__name__]
        else:
            self.access_states = FindAccessStates().apply_pass(sdfg, {})

        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, LoopRegion):
                # Loop step expression must be constant.
                if not self._has_constant_loop_expressions(sdfg, node):
                    continue

                arrays = set(acc_node.data for acc_node in node.data_nodes())
                write_states = {}
                for st in node.all_states():
                    for an in st.data_nodes():
                        if st.in_degree(an) > 0:
                            write_states.setdefault(an.data, set()).add(st)

                for arr in arrays:
                    self._apply_for_array(arr, sdfg, node, write_states)

        self.out_of_loop_states_cache = {}

    def _get_edge_indices(self, subset: Range, loop: LoopRegion) -> list[Union[tuple, None]]:
        # list of tuples of (a, b) for a*i + b, None if cannot be determined
        indices = list()
        itervar = loop.loop_variable
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

    def _get_read_write_indices(
            self, array_name: str,
            loop: LoopRegion) -> tuple[list[list[Union[tuple, None]]], list[list[Union[tuple, None]]]]:
        # list of list of tuples of (a, b) for a*i + b
        read_indices = list()
        write_indices = list()

        access_nodes = set(an for an in loop.data_nodes() if an.data == array_name)
        read_edges = set(e for an in access_nodes for st in loop.all_states() if an in st.data_nodes()
                         for e in st.out_edges(an))
        write_edges = set(e for an in access_nodes for st in loop.all_states() if an in st.data_nodes()
                          for e in st.in_edges(an))

        for edge in read_edges:
            eri = self._get_edge_indices(edge.data.src_subset, loop)
            read_indices.append(eri)

        for edge in write_edges:
            ewi = self._get_edge_indices(edge.data.dst_subset, loop)
            write_indices.append(ewi)

        return read_indices, write_indices

    def _has_constant_loop_expressions(self, sdfg: sd.SDFG, loop: LoopRegion) -> tuple[bool, Union[int, None]]:
        itervar = loop.loop_variable
        step = loop_analysis.get_loop_stride(loop)
        if step is None:
            return False

        defined_syms = set()
        for edge in loop.all_interstate_edges():
            defined_syms.update(edge.data.assignments.keys())

        resolved_step = symbolic.resolve_symbol_to_constant(step, sdfg)
        return (itervar not in defined_syms and step.free_symbols.isdisjoint(defined_syms)
                and step.free_symbols.isdisjoint({itervar}) and resolved_step is not None)

    def _get_K_values(
        self,
        array_name: str,
        read_indices: list[list[Union[tuple, None]]],
        write_indices: list[list[Union[tuple, None]]],
        step: int,
        sdfg: sd.SDFG,
        loop: LoopRegion,
    ) -> list[Union[int, None]]:
        k_values = []
        max_indices = self._get_max_indices_before_loop(array_name, sdfg, loop)

        # For each dimension
        for dim in range(len(read_indices[0])):
            # Get all read and write indices for this dimension
            dim_read_indices = [il[dim] for il in read_indices if il[dim] is not None]
            dim_write_indices = [il[dim] for il in write_indices if il[dim] is not None]

            if not dim_read_indices or not dim_write_indices:
                k_values.append(None)
                continue

            # Get the minimum read index and maximum write index
            try:
                read_lb = min([i[1] for i in dim_read_indices])
                read_ub = max([i[1] for i in dim_read_indices])
                write_lb = min([i[1] for i in dim_write_indices])
                write_ub = max([i[1] for i in dim_write_indices])
            except TypeError:
                # If the minimum or maximum cannot be determined (e.g. because of symbolic expressions being uncomparable), we cannot apply the transformation.
                k_values.append(None)
                continue

            # TODO: We could also handle cases where write_lb == read_ub if we can prove that the write happens before the read in the loop for larger spans.

            # We assume a is the same for all indices, so we can just take the first one.
            a = dim_read_indices[0][0] * step
            if a >= 1:
                span = (write_ub - read_lb) / a
                cond = (write_lb > read_ub)  # At least one write index must be higher than all read indices
            if a == 0:
                span = len(dim_read_indices + dim_write_indices)
                cond = (write_lb > read_ub)  # At least one write index must be higher than all read indices
            if a <= -1:
                span = (read_ub - write_ub) / (-a)
                cond = (write_ub < read_lb)  # At least one write index must be lower than all read indices

            # If we have a span of one, it's enough that reads happen after writes in the loop.
            if span == 0:
                cond = all(
                    st.in_degree(an) > 0 and st.out_degree(an) > 0 for st in loop.all_states()
                    for an in st.data_nodes() if an.data == array_name)

            # Take maximum from previous accesses into account
            try:
                k = max(span, max_indices[dim])
                not cond  # XXX: This ensures the condition can be evaluated. Do not remove.
            except TypeError:
                k_values.append(None)
                continue

            k = symbolic.resolve_symbol_to_constant(k, sdfg)
            if k is None:
                k_values.append(None)
                continue
            k = int(k)

            # Round up to next power of two if enabled
            if self.next_power_of_two:
                k_p2 = (1 << k.bit_length()) - 1

                # If we're larger than the array size, don't round up.
                if k_p2 + 1 < sdfg.arrays[array_name].shape[dim]:
                    k = k_p2

            # Condition must hold
            if not cond or k + 1 >= sdfg.arrays[array_name].shape[dim]:
                k_values.append(None)
            else:
                k_values.append(k + 1)  # +1 because k is the highest index accessed, so size is k+1
        return k_values

    def _write_is_loop_local(self, array_name: str, write_indices: list[list[tuple]], sdfg: sd.SDFG,
                             loop: LoopRegion) -> bool:
        # The (overapproximated) written subset must be written before read or not read at all.
        # TODO: This is overly conservative. Just checks if there are access nodes after the loop.

        if loop in self.out_of_loop_states_cache:
            out_of_loop_states = self.out_of_loop_states_cache[loop]
        else:
            loop_states = set(loop.all_states())
            states_reach = self.states_reach[sdfg.cfg_id]
            out_of_loop_states = set(v for st in loop_states for v in states_reach[st] if v not in loop_states)
            self.out_of_loop_states_cache[loop] = out_of_loop_states

        access_states = self.access_states[sdfg.cfg_id]
        array_states = access_states[array_name]

        # There should be no overlap between out_of_loop_states and states that read the array.
        return out_of_loop_states.isdisjoint(array_states)

    def _get_max_indices_before_loop(self, array_name: str, sdfg: sd.SDFG, loop: LoopRegion) -> list[int]:
        # Collect all read and write subsets of the array before the loop.
        loop_states = set(loop.all_states())
        subsets = set()
        for k1, v1 in self.states_reach.items():
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
                try:
                    max_index = max(max_index, re)
                except TypeError:
                    # If the maximum cannot be determined (e.g. because of symbolic expressions being uncomparable), we cannot determine the maximum index.
                    max_index = None
            max_index = symbolic.resolve_symbol_to_constant(max_index, sdfg)

            if max_index is None:
                indices.append(sdfg.arrays[array_name].shape[dim])
            else:
                indices.append(max_index)
        return indices

    def _apply_for_array(self, array_name: str, sdfg: sd.SDFG, loop: LoopRegion,
                         write_states: dict[str, set[sd.SDFGState]]) -> bool:
        # Must be transient (otherwise it's observable)
        if not sdfg.arrays[array_name].transient:
            return

        # Views and References are not supported
        if isinstance(sdfg.arrays[array_name], dt.View) or isinstance(sdfg.arrays[array_name], dt.Reference):
            return

        # There needs to be at least one read and one write.
        read_indices, write_indices = self._get_read_write_indices(array_name, loop)
        if not read_indices or not write_indices:
            return

        # All read and write indices must be linear combinations of the loop variable. I.e. a*i + b, where a and b are constants.
        if any(i is None for il in read_indices + write_indices for i in il):
            return

        # The scaling factor a must be the same for all indices if a != 0.
        a_values = set(i[0] for il in read_indices + write_indices for i in il if i[0] != 0)
        if len(a_values) > 1:
            return
        if len(a_values) == 0:
            a_values.add(0)

        # The offset b must be multiple of a if a != 0.
        step = symbolic.resolve_symbol_to_constant(loop_analysis.get_loop_stride(loop), sdfg)
        a = a_values.pop() * step
        if a != 0 and any(i[1] % a != 0 for il in read_indices + write_indices for i in il if i[0] != 0):
            return

        # All constants (a == 0) must be in the same dimension.
        for dim in range(len(read_indices[0])):
            if any(il[dim][0] == 0 for il in read_indices) and any(il[dim][0] != 0 for il in read_indices):
                return
            if any(il[dim][0] == 0 for il in write_indices) and any(il[dim][0] != 0 for il in write_indices):
                return

        # None of the write accesses must be within a conditional block. Reads are ok.
        for st in write_states.get(array_name, set()):
            pgraph = st.parent_graph
            while pgraph is not None and pgraph != loop:
                if isinstance(pgraph, ConditionalBlock):
                    return
                pgraph = pgraph.parent_graph

        # Outside of the loop, the written subset of the array must be written before read or not read at all.
        if not self._write_is_loop_local(array_name, write_indices, sdfg, loop):
            return

        # A K value must be found for at least one dimension.
        Ks = self._get_K_values(array_name, read_indices, write_indices, step, sdfg, loop)
        if all(k is None for k in Ks):
            return

        ### Otherwise, we can apply the transformation.
        self.num_applications += 1

        # Replace all read and write edges in the loop with modulo accesses.
        read_edges = set(e for st in sdfg.all_states() for an in st.data_nodes() if an.data == array_name
                         for e in st.out_edges(an))
        write_edges = set(e for st in sdfg.all_states() for an in st.data_nodes() if an.data == array_name
                          for e in st.in_edges(an))

        # XXX: We use abs() because pystr_to_symbolic() rewrites modulo operations, e.g. (-i + 32) % 31 -> Mod(1 - i, 31), which changes the behavior as C++ modulo differs from Python for negative numbers.
        for edge in read_edges:
            subset = edge.data.src_subset
            for i, k in enumerate(Ks):
                if k is None:
                    continue

                if k == 1:
                    # we can replace the array with a scalar, so no need for modulo.
                    lb = pystr_to_symbolic("0")
                    ub = pystr_to_symbolic("0")
                elif self.bitmask_indexing and k & (k - 1) == 0:
                    # if k is a power of two, we can use a bitmask instead of modulo.
                    lb = pystr_to_symbolic(f"{subset[i][0]} & ({k - 1})")
                    ub = pystr_to_symbolic(f"{subset[i][1]} & ({k - 1})")
                else:
                    lb = pystr_to_symbolic(f"abs({subset[i][0]}) % ({k})")
                    ub = pystr_to_symbolic(f"abs({subset[i][1]}) % ({k})")

                # If both k and lb / ub are constant, sympy can simplify with the modulo.
                if not issymbolic(k) and not issymbolic(lb):
                    lb = pystr_to_symbolic(f"abs({subset[i][0]}) % ({k})")
                if not issymbolic(k) and not issymbolic(ub):
                    ub = pystr_to_symbolic(f"abs({subset[i][1]}) % ({k})")

                st = subset[i][2]
                subset[i] = (lb, ub, st)
            edge.data.src_subset = subset

        for edge in write_edges:
            subset = edge.data.dst_subset
            for i, k in enumerate(Ks):
                if k is None:
                    continue

                if k == 1:
                    # we can replace the array with a scalar, so no need for modulo.
                    lb = pystr_to_symbolic("0")
                    ub = pystr_to_symbolic("0")
                elif self.bitmask_indexing and k & (k - 1) == 0:
                    # if k is a power of two, we can use a bitmask instead of modulo.
                    lb = pystr_to_symbolic(f"{subset[i][0]} & ({k - 1})")
                    ub = pystr_to_symbolic(f"{subset[i][1]} & ({k - 1})")
                else:
                    lb = pystr_to_symbolic(f"abs({subset[i][0]}) % ({k})")
                    ub = pystr_to_symbolic(f"abs({subset[i][1]}) % ({k})")

                # If both k and lb / ub are constant, sympy can simplify with the modulo.
                if not issymbolic(k) and not issymbolic(lb):
                    lb = pystr_to_symbolic(f"abs({subset[i][0]}) % ({k})")
                if not issymbolic(k) and not issymbolic(ub):
                    ub = pystr_to_symbolic(f"abs({subset[i][1]}) % ({k})")

                st = subset[i][2]
                subset[i] = (lb, ub, st)
            edge.data.dst_subset = subset

        # Reduce the size of the array in the SDFG.
        array = sdfg.arrays[array_name]
        new_shape = list(array.shape)
        for i, k in enumerate(Ks):
            if k is not None:
                new_shape[i] = k
        array.set_shape(tuple(new_shape))

        # If the new shape is a single element, we can replace the array with a scalar.
        if all(s == 1 for s in new_shape):
            sdfg.arrays[array_name] = Scalar(
                dtype=array.dtype,
                transient=array.transient,
                storage=array.storage,
                allow_conflicts=array.allow_conflicts,
                location=array.location,
                lifetime=array.lifetime,
                debuginfo=array.debuginfo,
            )
            for edge in read_edges:
                edge.data.src_subset = Range([(0, 0, 1)])
            for edge in write_edges:
                edge.data.dst_subset = Range([(0, 0, 1)])
