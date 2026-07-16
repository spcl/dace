# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""UnblockDimensions -- inverse of SplitDimensions (the Block primitive).

Block turns a shape ``[M, N, K]`` with mask ``[F, T, T]`` and factors ``[_, 16, 32]``
into ``[M, N//16, K//32, 16, 32]`` (outer/block dims in place, inner/offset dims
appended in mask order), rewriting an access ``[i, j, k]`` to
``[i, j//16, k//32, j%16, k%32]``.

Unblock reverses that, using the SAME ``(masks, factors)`` the Block used: it merges
each ``(outer, inner)`` pair back into one dimension ``outer*factor + inner`` and drops
the appended inner dims, so ``[M, N//16, K//32, 16, 32] -> [M, N, K]`` and the access
``[i, jo, ko, ji, ki] -> [i, jo*16 + ji, ko*32 + ki]``.

Assumptions (mirrors SplitDimensions):
  * The layout normal form (no views/other_subset). Runs after ``prepare_for_layout``.
  * The blocked array is packed. Reconstructed extent is ``outer_count * factor``
    (exact when the original extent was divisible by the factor; over-approximates
    by the pad amount otherwise).
  * The outer/block index of a masked dimension is a single-tile (point) access,
    which is what Block emits for element-wise nests. Multi-tile outer accesses raise.
"""
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import dace
from dace.sdfg.graph import Edge, EdgeT
from dace.transformation import pass_pipeline as ppl


@dataclass
class UnblockDimensions(ppl.Pass):
    """Merge blocked (outer, inner) dimension pairs back into flat dimensions.

    :param unblock_map: ``{array_name: (masks, factors)}`` -- the same masks/factors
                        the Block (``SplitDimensions``) used to produce the array.
                        ``masks``/``factors`` have length = the ORIGINAL (unblocked) rank.
    """

    def __init__(self, unblock_map: Dict[str, Tuple[List[bool], List[int]]], verbose: bool = False):
        self._unblock_map = unblock_map
        self._verbose = verbose

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.InterstateEdges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    # ------------------------------------------------------------------ #
    #  Shape / subset reconstruction
    # ------------------------------------------------------------------ #
    def _inner_slot(self, masks: List[bool], d: int) -> int:
        """Appended-inner-dimension index for original masked dimension ``d``."""
        return len(masks) + sum(1 for m in masks[:d] if m)

    def _recover_extent(self, blocked_dim, factor: int):
        """Invert Block's per-dimension division to recover the original extent.

        Block emits ``int_ceil(E, factor)`` (non-divisible / bare symbol) or
        ``int_floor(E, factor)`` (divisible symexpr) for the outer block count; a concrete
        divisible int folds to a plain integer. Both ``int_ceil(E, factor)`` and
        ``int_floor(E, factor)`` invert to ``E`` exactly -- Block only emits them where it
        assumed the extent maps back to ``E`` under blocking (``floor(E/f)*f`` would NOT fold
        to ``E`` symbolically, so we unwrap the wrapper rather than multiply). A folded
        integer count hits the fallback ``count * factor``.
        """
        expr = dace.symbolic.pystr_to_symbolic(blocked_dim)
        if type(expr).__name__ in ('int_ceil', 'int_floor') and len(
                expr.args) == 2 and dace.symbolic.simplify(expr.args[1] - factor) == 0:
            return expr.args[0]
        return dace.symbolic.simplify(expr * factor)

    def _unblocked_shape(self, blocked_shape, masks: List[bool], factors: List[int]) -> List:
        new_shape = []
        for d, m in enumerate(masks):
            if m:
                new_shape.append(self._recover_extent(blocked_shape[d], factors[d]))
            else:
                new_shape.append(blocked_shape[d])
        return new_shape

    def _unblocked_subset(self, subset: dace.subsets.Range, masks: List[bool],
                          factors: List[int]) -> dace.subsets.Range:
        ranges = list(subset.ranges)
        expected = len(masks) + sum(1 for m in masks if m)
        if len(ranges) != expected:
            raise ValueError(f"UnblockDimensions: subset rank {len(ranges)} != expected blocked rank {expected} "
                             f"for masks {masks}")

        # Merge each masked (outer, inner) pair: index = outer*factor + inner.
        # This is exact for both the map-body point access ``[i//f, i%f]`` (outer
        # single tile) and the full-array map-boundary memlet, since
        # ``floor(x, f)*f + (x % f) == x``. When inner is a strict sub-block of a
        # multi-tile outer range the bounding box over-approximates the read set
        # -- safe for data-layout dependency analysis, and not a form Block emits.
        new_ranges = []
        for d, m in enumerate(masks):
            if not m:
                new_ranges.append(ranges[d])
                continue
            ob, oe, os = ranges[d]
            ib, ie, iss = ranges[self._inner_slot(masks, d)]
            factor = factors[d]
            # Fold composite affine forms: (i//f)*f + i%f -> i (symbols are nonnegative).
            new_b = dace.symbolic.simplify(ob * factor + ib)
            new_e = dace.symbolic.simplify(oe * factor + ie)
            new_ranges.append((new_b, new_e, iss))
        return dace.subsets.Range(new_ranges)

    # ------------------------------------------------------------------ #
    #  Descriptor / memlet / interstate rewrites (recurse into nested SDFGs)
    # ------------------------------------------------------------------ #
    def _replace_array(self, sdfg: dace.SDFG, arr_name: str, new_shape: List):
        arr = sdfg.arrays[arr_name]
        datadesc = copy.deepcopy(arr)
        sdfg.remove_data(arr_name, validate=False)
        sdfg.add_array(name=arr_name,
                       shape=new_shape,
                       dtype=datadesc.dtype,
                       transient=datadesc.transient,
                       storage=datadesc.storage,
                       lifetime=datadesc.lifetime,
                       alignment=datadesc.alignment,
                       debuginfo=datadesc.debuginfo,
                       find_new_name=False)

    def _nested_targets(self, sdfg: dace.SDFG, arr_name: str):
        """Yield ``(nested_sdfg, inner_name)`` for every nested SDFG ``arr_name`` flows into, each
        inner array ONCE. A read-write array uses the same inner name on its in-edge and out-edge;
        yielding it twice would unblock the inner descriptor / memlets a second time."""
        seen = set()
        for state in sdfg.all_states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.NestedSDFG):
                    continue
                boundary = ([(ie, ie.dst_conn) for ie in state.in_edges(node)] +
                            [(oe, oe.src_conn) for oe in state.out_edges(node)])
                for edge, conn in boundary:
                    if edge.data is None or edge.data.data != arr_name or conn is None:
                        continue
                    key = (id(node.sdfg), conn)
                    if key not in seen:
                        seen.add(key)
                        yield node.sdfg, conn

    def _replace_array_recursive(self, sdfg: dace.SDFG, arr_name: str, new_shape: List):
        self._replace_array(sdfg, arr_name, new_shape)
        for nsdfg, inner in self._nested_targets(sdfg, arr_name):
            self._replace_array_recursive(nsdfg, inner, new_shape)

    def _replace_memlets_recursive(self, sdfg: dace.SDFG, arr_name: str, masks, factors):
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data == arr_name:
                    if edge.data.other_subset is not None:
                        raise NotImplementedError("UnblockDimensions: other_subset memlets are unsupported.")
                    new_subset = self._unblocked_subset(edge.data.subset, masks, factors)
                    # Preserve wcr: a reduction into the unblocked target keeps accumulating.
                    edge.data = dace.memlet.Memlet(data=edge.data.data,
                                                   subset=new_subset,
                                                   wcr=edge.data.wcr,
                                                   wcr_nonatomic=edge.data.wcr_nonatomic,
                                                   dynamic=edge.data.dynamic)
        for nsdfg, inner in self._nested_targets(sdfg, arr_name):
            self._replace_memlets_recursive(nsdfg, inner, masks, factors)

    def _extract_indices(self, expr: str, name: str) -> List[str]:
        m = re.search(rf'{re.escape(name)}\[(.*)\]', expr)
        if not m:
            return []
        inside = m.group(1)
        parts, depth, current = [], 0, []
        for ch in inside:
            if ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())
        return parts

    def _replace_interstate_edges_recursive(self, sdfg: dace.SDFG, arr_name: str, masks, factors):
        for edge in sdfg.all_interstate_edges():
            new_assignments = dict()
            for k, v in edge.data.assignments.items():
                idx = self._extract_indices(v, arr_name)
                expected = len(masks) + sum(1 for m in masks if m)
                if len(idx) == expected:
                    merged = []
                    for d, m in enumerate(masks):
                        if m:
                            outer = idx[d]
                            inner = idx[self._inner_slot(masks, d)]
                            merged.append(f"(({outer}) * {factors[d]} + ({inner}))")
                        else:
                            merged.append(idx[d])
                    new_assignments[k] = re.sub(rf'{re.escape(arr_name)}\[(.*)\]', f"{arr_name}[{', '.join(merged)}]",
                                                v)
                else:
                    new_assignments[k] = v
            edge.data.assignments = new_assignments
        for nsdfg, inner in self._nested_targets(sdfg, arr_name):
            self._replace_interstate_edges_recursive(nsdfg, inner, masks, factors)

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        for arr_name, (masks, factors) in self._unblock_map.items():
            arr = sdfg.arrays[arr_name]
            expected = len(masks) + sum(1 for m in masks if m)
            if len(arr.shape) != expected:
                raise ValueError(f"UnblockDimensions: array '{arr_name}' has rank {len(arr.shape)}, expected blocked "
                                 f"rank {expected} for masks {masks}")
            new_shape = self._unblocked_shape(tuple(arr.shape), masks, factors)
            self._replace_array_recursive(sdfg, arr_name, new_shape)
            self._replace_memlets_recursive(sdfg, arr_name, masks, factors)
            self._replace_interstate_edges_recursive(sdfg, arr_name, masks, factors)
        return 0
