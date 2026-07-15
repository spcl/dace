# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ShuffleElements -- the Shuffle layout primitive: renumber a dimension's elements by a
user-supplied bijection ``sigma``.

For an array ``A`` shuffled on dimension ``d`` by ``sigma`` (registered via
:func:`~dace.libraries.layout.shuffle.register_shuffle`), the laid-out array ``A'`` holds

    A'[.., j, ..] = A[.., sigma(j), ..]              (physical reorder, a "gather")

so a body access to the LOGICAL element ``A[.., e, ..]`` reads ``A'[.., sigma^{-1}(e), ..]``
(since ``A'[sigma^{-1}(e)] = A[sigma(sigma^{-1}(e))] = A[e]``). The pass therefore:

  * clones ``A`` into a fresh transient ``A'`` of the same shape;
  * if ``A`` is READ in the body, inserts a ``shuffle_in`` state gathering ``A' = A o sigma``;
  * if ``A`` is WRITTEN in the body, inserts a ``shuffle_out`` state scattering
    ``A[j] = A'[sigma^{-1}(j)]`` so the caller sees the original order;
  * rewrites every body access ``A[.., e, ..] -> A'[.., sigma^{-1}(e), ..]`` (point accesses on
    ``d``; a full-extent range on ``d`` is a bijection invariant and is only renamed), recursing
    into nested SDFGs;
  * injects the C++ definitions of ``sigma`` / ``sigma^{-1}`` into the SDFG global code.

``sigma`` is CLOSED-FORM: the gather/scatter/consumer index expressions are plain
sympy-function terms that lower to literal C calls (``A[shuffle_<name>(j, ...)]``), so no
indirection subgraph is needed. Run after ``prepare_for_layout`` (Core Dialect). A shuffled
dimension must be accessed point-wise in the body; a strict partial range on ``d`` raises.
"""
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.core_dialect import require_core_dialect
from dace.transformation import pass_pipeline as ppl

from dace.libraries.layout.shuffle import get_shuffle, emit_shuffle_globals


@dataclass
class ShuffleElements(ppl.Pass):
    """Renumber a dimension's elements by a registered bijection ``sigma``.

    :param shuffle_map: ``{array_name: (shuffle_name, dim)}`` -- ``shuffle_name`` must be
                        registered with :func:`register_shuffle`; ``dim`` is the shuffled axis.
    """

    def __init__(self, shuffle_map: Dict[str, Tuple[str, int]]):
        self._shuffle_map = shuffle_map

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors
                | ppl.Modifies.NestedSDFGs | ppl.Modifies.Memlets)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        require_core_dialect(sdfg, source='ShuffleElements')
        used = []
        for arr, (name, dim) in self._shuffle_map.items():
            fn = get_shuffle(name)
            used.append(name)
            self._shuffle_array(sdfg, arr, fn, dim)
        emit_shuffle_globals(sdfg, used)
        return 0

    # ------------------------------------------------------------------ #
    #  Read/write detection
    # ------------------------------------------------------------------ #
    def _is_read(self, sdfg: dace.SDFG, arr: str) -> bool:
        return any(an.data == arr and state.out_degree(an) > 0 for state in sdfg.all_states()
                   for an in state.data_nodes())

    def _is_written(self, sdfg: dace.SDFG, arr: str) -> bool:
        return any(an.data == arr and state.in_degree(an) > 0 for state in sdfg.all_states()
                   for an in state.data_nodes())

    # ------------------------------------------------------------------ #
    #  Per-array driver
    # ------------------------------------------------------------------ #
    def _shuffle_array(self, sdfg: dace.SDFG, arr: str, fn, dim: int) -> None:
        self._guard_no_interstate(sdfg, arr)
        desc = sdfg.arrays[arr]
        if dim < 0 or dim >= len(desc.shape):
            raise ValueError(f"ShuffleElements: dim {dim} out of range for '{arr}' shape {desc.shape}")
        size = desc.shape[dim]
        shuffled = f"shuffled_{arr}"

        reads = self._is_read(sdfg, arr)
        writes = self._is_written(sdfg, arr)

        cloned = copy.deepcopy(desc)
        cloned.transient = True
        sdfg.add_datadesc(shuffled, cloned, find_new_name=False)

        skip = set()
        # Gather whenever the array is read OR written. A written array must start as a full
        # shuffled copy of its incoming values: a write that does not cover the whole array (e.g.
        # partial on a non-shuffled dim) otherwise leaves part of the transient uninitialized, and
        # the full-array scatter would clobber the caller's array there. A fully-overwritten output
        # re-reads its input redundantly (an extra O(size) boundary copy) -- correct, not wrong.
        if reads or writes:
            pre = sdfg.add_state_before(sdfg.start_state, f"shuffle_in_{arr}")
            skip.add(pre)
            self._emit_reorder(sdfg, pre, src=arr, dst=shuffled, fn=fn, dim=dim, forward=True)
        if writes:
            # Scatter back after EVERY sink block, so results are restored to the original order on
            # whichever terminating path executes (not just the first sink).
            for sink in [v for v in sdfg.nodes() if sdfg.out_degree(v) == 0]:
                post = sdfg.add_state_after(sink, f"shuffle_out_{arr}")
                skip.add(post)
                self._emit_reorder(sdfg, post, src=shuffled, dst=arr, fn=fn, dim=dim, forward=False)

        self._rewrite_body(sdfg, arr, shuffled, fn, dim, size, skip)

    # ------------------------------------------------------------------ #
    #  Boundary reorder (gather / scatter), closed-form sigma
    # ------------------------------------------------------------------ #
    def _emit_reorder(self, sdfg, state, src, dst, fn, dim, forward) -> None:
        """``dst[j] = src[sigma(j)]`` (forward gather) or ``dst[j] = src[sigma^{-1}(j)]`` (scatter),
        with the shuffled dimension's read index a closed-form sigma term."""
        shape = sdfg.arrays[src].shape
        ndim = len(shape)
        params = [f"__i{k}" for k in range(ndim)]
        map_ranges = {p: f"0:{s}" for p, s in zip(params, shape)}

        read_ranges = []
        for k in range(ndim):
            sym = dace.symbolic.symbol(params[k])
            if k == dim:
                ix = fn.apply_forward(sym) if forward else fn.apply_inverse(sym)
            else:
                ix = sym
            read_ranges.append((ix, ix, 1))
        write_ranges = [(dace.symbolic.symbol(p), dace.symbolic.symbol(p), 1) for p in params]

        state.add_mapped_tasklet(
            name=f"reorder_{src}_to_{dst}",
            map_ranges=map_ranges,
            inputs={"__inp": dace.Memlet(data=src, subset=dace.subsets.Range(read_ranges))},
            code="__out = __inp",
            outputs={"__out": dace.Memlet(data=dst, subset=dace.subsets.Range(write_ranges))},
            external_edges=True,
        )

    # ------------------------------------------------------------------ #
    #  Body rewrite: A[.., e, ..] -> A'[.., sigma^{-1}(e), ..] (recurse nested SDFGs)
    # ------------------------------------------------------------------ #
    def _compose_subset(self, subset: dace.subsets.Range, fn, dim: int, size) -> dace.subsets.Range:
        ranges = list(subset.ranges)
        b, e, s = ranges[dim]
        if dace.symbolic.simplify(e - b) == 0:  # point access -> compose sigma^{-1}
            inv = fn.apply_inverse(b)
            ranges[dim] = (inv, inv, 1)
        elif self._is_full_extent(b, e, s, size):  # bijection over the whole axis -> invariant
            pass
        else:
            raise NotImplementedError(
                f"ShuffleElements: a shuffled dimension must be accessed point-wise or full; got range "
                f"({b}:{e}:{s}) on dim {dim} of size {size}.")
        return dace.subsets.Range(ranges)

    def _is_full_extent(self, b, e, s, size) -> bool:
        return (dace.symbolic.simplify(b) == 0
                and dace.symbolic.simplify(e - (dace.symbolic.pystr_to_symbolic(size) - 1)) == 0
                and dace.symbolic.simplify(s - 1) == 0)

    def _rewrite_body(self, sdfg, arr, shuffled, fn, dim, size, skip) -> None:
        # Collect nested boundaries BEFORE the rename below rewrites the boundary memlet's data
        # (otherwise the edge no longer reads ``arr`` and the recursion misses it).
        nested = list(self._nested_targets(sdfg, arr))
        for state in sdfg.all_states():
            if state in skip:
                continue
            for node in state.nodes():
                if isinstance(node, nd.AccessNode) and node.data == arr:
                    node.data = shuffled
            for edge in state.edges():
                if edge.data is None or edge.data.data != arr:
                    continue
                if edge.data.other_subset is not None:
                    raise NotImplementedError("ShuffleElements: other_subset memlets are unsupported.")
                self._rename_connectors(edge, arr, shuffled)
                new_subset = self._compose_subset(edge.data.subset, fn, dim, size)
                edge.data = dace.memlet.Memlet(data=shuffled, subset=new_subset)
        # Nested SDFGs: the inner array keeps its connector name; only its body memlets compose
        # sigma^{-1} (the shape is unchanged, so no descriptor edits and no boundary states).
        for nsdfg, inner in nested:
            self._rewrite_inner(nsdfg, inner, fn, dim, size)

    def _rewrite_inner(self, sdfg, arr, fn, dim, size) -> None:
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is None or edge.data.data != arr:
                    continue
                if edge.data.other_subset is not None:
                    raise NotImplementedError("ShuffleElements: other_subset memlets are unsupported.")
                new_subset = self._compose_subset(edge.data.subset, fn, dim, size)
                edge.data = dace.memlet.Memlet(data=arr, subset=new_subset)
        for nsdfg, inner in self._nested_targets(sdfg, arr):
            self._rewrite_inner(nsdfg, inner, fn, dim, size)

    def _rename_connectors(self, edge, arr, shuffled) -> None:
        if edge.dst_conn == "IN_" + arr:
            edge.dst.remove_in_connector("IN_" + arr)
            edge.dst.add_in_connector("IN_" + shuffled)
            edge.dst_conn = "IN_" + shuffled
        if edge.src_conn == "OUT_" + arr:
            edge.src.remove_out_connector("OUT_" + arr)
            edge.src.add_out_connector("OUT_" + shuffled)
            edge.src_conn = "OUT_" + shuffled

    def _nested_targets(self, sdfg, arr):
        """Yield ``(nested_sdfg, inner_connector_name)`` for every nested SDFG ``arr`` flows into."""
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, nd.NestedSDFG):
                    for ie in state.in_edges(node):
                        if ie.data is not None and ie.data.data == arr and len(node.sdfg.arrays[ie.dst_conn].shape) \
                                == len(sdfg.arrays[arr].shape):
                            yield node.sdfg, ie.dst_conn
                    for oe in state.out_edges(node):
                        if oe.data is not None and oe.data.data == arr and len(node.sdfg.arrays[oe.src_conn].shape) \
                                == len(sdfg.arrays[arr].shape):
                            yield node.sdfg, oe.src_conn

    def _guard_no_interstate(self, sdfg, arr) -> None:
        token = re.compile(rf"\b{re.escape(arr)}\b")
        for edge in sdfg.all_interstate_edges():
            for k, v in edge.data.assignments.items():
                if token.search(str(v)):
                    raise NotImplementedError(
                        f"ShuffleElements: '{arr}' is referenced in an interstate edge assignment ('{k} = {v}'); "
                        f"shuffling interstate-referenced arrays is not supported.")
