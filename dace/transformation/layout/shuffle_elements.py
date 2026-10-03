# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ShuffleElements -- the Shuffle layout primitive: renumbers one or more full array dimensions via registered bijections ``sigma`` (one per dimension), rewriting body accesses and inserting gather/scatter boundary states."""
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import dace
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl

from dace.libraries.layout.shuffle import get_shuffle, emit_shuffle_globals

#: A per-array shuffle spec: one ``(sigma_name, dim)`` pair, or a list of them for multiple dimensions.
ShuffleSpec = Union[Tuple[str, int], List[Tuple[str, int]]]


@dataclass
class ShuffleElements(ppl.Pass):
    """Renumber one or more full dimensions via registered bijections. ``shuffle_map`` maps each array to
    either a single ``(sigma_name, dim)`` or a list of them (each dim renumbered by its own sigma)."""

    def __init__(self, shuffle_map: Dict[str, ShuffleSpec]):
        self._shuffle_map = shuffle_map

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors
                | ppl.Modifies.NestedSDFGs | ppl.Modifies.Memlets)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    @staticmethod
    def _normalize(spec: ShuffleSpec) -> List[Tuple[str, int]]:
        """A single ``(name, dim)`` tuple or a list of them -> a list (backward compatible)."""
        if isinstance(spec, tuple):
            return [spec]
        return list(spec)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        used = []
        for arr, spec in self._shuffle_map.items():
            pairs = self._normalize(spec)
            fns = {}
            for name, dim in pairs:
                if dim in fns:
                    raise ValueError(f"ShuffleElements: dimension {dim} of '{arr}' is shuffled twice")
                fns[dim] = get_shuffle(name)
                used.append(name)
            self._shuffle_array(sdfg, arr, fns)
        emit_shuffle_globals(sdfg, used)
        return 0

    def _is_read(self, sdfg: dace.SDFG, arr: str) -> bool:
        return any(an.data == arr and state.out_degree(an) > 0 for state in sdfg.all_states()
                   for an in state.data_nodes())

    def _is_written(self, sdfg: dace.SDFG, arr: str) -> bool:
        return any(an.data == arr and state.in_degree(an) > 0 for state in sdfg.all_states()
                   for an in state.data_nodes())

    def _shuffle_array(self, sdfg: dace.SDFG, arr: str, fns: Dict[int, Any]) -> None:
        self._guard_no_interstate(sdfg, arr)
        desc = sdfg.arrays[arr]
        for dim in fns:
            if dim < 0 or dim >= len(desc.shape):
                raise ValueError(f"ShuffleElements: dim {dim} out of range for '{arr}' shape {desc.shape}")
        sizes = {dim: desc.shape[dim] for dim in fns}
        shuffled = f"shuffled_{arr}"

        reads = self._is_read(sdfg, arr)
        writes = self._is_written(sdfg, arr)

        cloned = copy.deepcopy(desc)
        cloned.transient = True
        sdfg.add_datadesc(shuffled, cloned, find_new_name=False)

        skip = set()
        # gather on read or write: partial writes need the pre-existing values preserved
        if reads or writes:
            pre = sdfg.add_state_before(sdfg.start_state, f"shuffle_in_{arr}")
            skip.add(pre)
            self._emit_reorder(sdfg, pre, src=arr, dst=shuffled, fns=fns, forward=True)
        if writes:
            # scatter after every sink, not just the first, so all exit paths restore order
            for sink in [v for v in sdfg.nodes() if sdfg.out_degree(v) == 0]:
                post = sdfg.add_state_after(sink, f"shuffle_out_{arr}")
                skip.add(post)
                self._emit_reorder(sdfg, post, src=shuffled, dst=arr, fns=fns, forward=False)

        self._rewrite_body(sdfg, arr, shuffled, fns, sizes, skip)

    def _emit_reorder(self, sdfg, state, src, dst, fns, forward) -> None:
        """``dst[j] = src[sigma(j)]`` (gather) or ``src[sigma^{-1}(j)]`` (scatter), per shuffled dim."""
        shape = sdfg.arrays[src].shape
        ndim = len(shape)
        params = [f"__i{k}" for k in range(ndim)]
        map_ranges = {p: f"0:{s}" for p, s in zip(params, shape)}

        read_ranges = []
        for k in range(ndim):
            sym = dace.symbolic.symbol(params[k])
            if k in fns:
                ix = fns[k].apply_forward(sym) if forward else fns[k].apply_inverse(sym)
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

    # body rewrite: A[e] -> A'[sigma^{-1}(e)] on each shuffled dim; recurses into nested SDFGs
    def _compose_subset(self, subset: dace.subsets.Range, fns: Dict[int, Any], sizes: Dict[int,
                                                                                           Any]) -> dace.subsets.Range:
        ranges = list(subset.ranges)
        for dim, fn in fns.items():
            b, e, s = ranges[dim]
            if dace.symbolic.simplify(e - b) == 0:  # point access -> compose sigma^{-1}
                inv = fn.apply_inverse(b)
                ranges[dim] = (inv, inv, 1)
            elif self._is_full_extent(b, e, s, sizes[dim]):  # bijection over the whole axis -> invariant
                pass
            else:
                raise NotImplementedError(
                    f"ShuffleElements: a shuffled dimension must be accessed point-wise or full; got range "
                    f"({b}:{e}:{s}) on dim {dim} of size {sizes[dim]}.")
        return dace.subsets.Range(ranges)

    def _is_full_extent(self, b, e, s, size) -> bool:
        return (dace.symbolic.simplify(b) == 0
                and dace.symbolic.simplify(e - (dace.symbolic.pystr_to_symbolic(size) - 1)) == 0
                and dace.symbolic.simplify(s - 1) == 0)

    def _rewrite_body(self, sdfg, arr, shuffled, fns, sizes, skip) -> None:
        # collect nested boundaries before the rename below, or the recursion misses them
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
                new_subset = self._compose_subset(edge.data.subset, fns, sizes)
                # preserve wcr: reduction into the shuffled target keeps accumulating
                edge.data = dace.memlet.Memlet(data=shuffled,
                                               subset=new_subset,
                                               wcr=edge.data.wcr,
                                               wcr_nonatomic=edge.data.wcr_nonatomic,
                                               dynamic=edge.data.dynamic)
        # nested SDFGs keep the inner connector name; only body memlets compose sigma^{-1}
        for nsdfg, inner in nested:
            self._rewrite_inner(nsdfg, inner, fns, sizes)

    def _rewrite_inner(self, sdfg, arr, fns, sizes) -> None:
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is None or edge.data.data != arr:
                    continue
                if edge.data.other_subset is not None:
                    raise NotImplementedError("ShuffleElements: other_subset memlets are unsupported.")
                new_subset = self._compose_subset(edge.data.subset, fns, sizes)
                edge.data = dace.memlet.Memlet(data=arr,
                                               subset=new_subset,
                                               wcr=edge.data.wcr,
                                               wcr_nonatomic=edge.data.wcr_nonatomic,
                                               dynamic=edge.data.dynamic)
        for nsdfg, inner in self._nested_targets(sdfg, arr):
            self._rewrite_inner(nsdfg, inner, fns, sizes)

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
        """Yield ``(nested_sdfg, inner_connector_name)`` once per inner array (dedup avoids double sigma^{-1})."""
        seen = set()
        for state in sdfg.all_states():
            for node in state.nodes():
                if not isinstance(node, nd.NestedSDFG):
                    continue
                boundary = ([(ie, ie.dst_conn) for ie in state.in_edges(node)] + [(oe, oe.src_conn)
                                                                                  for oe in state.out_edges(node)])
                for edge, conn in boundary:
                    if edge.data is None or edge.data.data != arr or conn is None:
                        continue
                    if len(node.sdfg.arrays[conn].shape) != len(sdfg.arrays[arr].shape):
                        continue
                    key = (id(node.sdfg), conn)
                    if key not in seen:
                        seen.add(key)
                        yield node.sdfg, conn

    def _guard_no_interstate(self, sdfg, arr) -> None:
        """Raise if ``arr`` appears in an interstate edge assignment or condition -- those aren't rewritten."""
        token = re.compile(rf"\b{re.escape(arr)}\b")
        for edge in sdfg.all_interstate_edges():
            for k, v in edge.data.assignments.items():
                if token.search(str(v)):
                    raise NotImplementedError(
                        f"ShuffleElements: '{arr}' is referenced in an interstate edge assignment ('{k} = {v}'); "
                        f"shuffling interstate-referenced arrays is not supported.")
            condition = edge.data.condition
            if condition is not None and token.search(condition.as_string):
                raise NotImplementedError(
                    f"ShuffleElements: '{arr}' is referenced in an interstate edge condition "
                    f"('{condition.as_string}'); shuffling interstate-referenced arrays is not supported.")
