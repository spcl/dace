# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split a small transient buffer whose every access is a constant element into one scalar per element.

A fused map keeps a per-iteration buffer such as CloudSC's ``__map_fusion_zpfplsx[2, 1, 1]``: tasklets write and
read single elements (``B[k, 0, 0]``) and one copy moves a range of them out (``B[0:2, 0, 0] -> A[2:4, k, i]``).
Vectorized, every element holds one value per lane, so the buffer would have to grow a lane axis and every
consumer would have to slice it. As one scalar per element it is the per-lane temporary the tile pipeline already
widens, and the range copy becomes one element copy per scalar::

    B[0, 0, 0] = x; B[1, 0, 0] = y; A[2:4, k, i] <- B[0:2, 0, 0]
    ->  B_0 = x;  B_1 = y;  A[2, k, i] <- B_0;  A[3, k, i] <- B_1

Only exact cases are split: a transient ``Array`` of constant shape (at most :data:`MAX_ELEMENTS` elements) whose
accesses are all access nodes connected to tasklets or to other access nodes, every edge naming constant elements
of it and, for a range, a matching range of the other array (the same extents once unit dimensions are dropped).
Anything else -- a map, nested SDFG or library node neighbour, a WCR, an ordering edge, an interstate read -- leaves
the buffer as it is.
"""

from itertools import product
from typing import Any

import sympy

from dace import data, properties, subsets
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.narrowing import as_expr
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: The largest buffer split: one scalar per element.
MAX_ELEMENTS = 32


def _constant_ranges(subset: subsets.Range | None) -> list[tuple[int, int, int]] | None:
    """``subset`` as integer ``(begin, end, step)`` triples, or ``None`` if any bound is not a constant."""
    if not isinstance(subset, subsets.Range):
        return None
    ranges = []
    for begin, end, step in subset.ranges:
        values = [as_expr(v) for v in (begin, end, step)]
        if not all(isinstance(v, sympy.Integer) for v in values):
            return None
        ranges.append(tuple(int(v) for v in values))
    return ranges


def _elements(ranges: list[tuple[int, int, int]]) -> list[tuple[int, ...]]:
    """The element indices ``ranges`` covers, row-major."""
    return list(product(*(range(b, e + 1, s) for b, e, s in ranges)))


def _extents(subset: subsets.Range) -> list:
    """The extent of every non-unit dimension of ``subset`` with constant bounds: a dimension with symbolic bounds (the
    lane axis of a vectorized copy) is kept whole on every element, so it is not one the elements walk."""
    extents = []
    for begin, end, step in subset.ranges:
        if as_expr(end - begin) == 0:
            continue
        if _constant_ranges(subsets.Range([(begin, end, step)])) is not None:
            extents.append(as_expr((end - begin) // step + 1))
    return extents


def _other_elements(other: subsets.Range, count: int) -> list[subsets.Range] | None:
    """The single-element subsets of ``other`` in row-major order over its non-unit dimensions; ``None`` unless it has
    ``count`` of them. A non-constant dimension of extent one (a lane index) is kept as it is."""
    axes = []
    for begin, end, step in other.ranges:
        if as_expr(end - begin) == 0:
            axes.append([(begin, end, step)])
            continue
        ranges = _constant_ranges(subsets.Range([(begin, end, step)]))
        if ranges is None:
            # A non-constant extent: the lane axis a vectorized copy spans keeps its whole range on every element.
            axes.append([(begin, end, step)])
            continue
        ((b, e, s),) = ranges
        axes.append([(v, v, 1) for v in range(b, e + 1, s)])
    picks = [list(choice) for choice in product(*axes)]
    if len(picks) != count:
        return None
    return [subsets.Range(pick) for pick in picks]


def _buffer_accesses(sdfg: SDFG, name: str) -> list[tuple[SDFGState, nodes.AccessNode]] | None:
    """Every access node of ``name`` in ``sdfg``, or ``None`` if an access is not splittable (see the module)."""
    if name in {str(s) for e in sdfg.all_interstate_edges() for s in e.data.free_symbols}:
        return None
    found = []
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            for edge in state.all_edges(node):
                other = edge.dst if edge.src is node else edge.src
                if not isinstance(other, (nodes.Tasklet, nodes.AccessNode)):
                    return None
                if edge.data.is_empty() or edge.data.wcr is not None:
                    return None
            found.append((state, node))
    return found


def _side(memlet: Memlet, name: str) -> tuple[subsets.Range | None, subsets.Range | None]:
    """``(subset of name, subset of the other end)`` of ``memlet``."""
    if memlet.data == name:
        return memlet.subset, memlet.other_subset
    return memlet.other_subset, memlet.subset


@properties.make_properties
@transformation.explicit_cf_compatible
class SplitConstantIndexedBuffers(ppl.Pass):
    """Split each small transient buffer accessed only at constant elements into one scalar per element."""

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        split = 0
        for nested in list(sdfg.all_sdfgs_recursive()):
            for name, desc in list(nested.arrays.items()):
                if self._split(nested, name, desc):
                    split += 1
        return split or None

    @staticmethod
    def _plan(sdfg: SDFG, name: str, accesses) -> list | None:
        """Per edge, ``(state, edge, buffer elements, other-end element subsets or None)``; ``None`` if any edge is
        not exact."""
        plan = []
        for state, node in accesses:
            for edge in state.all_edges(node):
                own, other = _side(edge.data, name)
                ranges = _constant_ranges(own)
                if ranges is None:
                    return None
                elements = _elements(ranges)
                if isinstance(edge.src, nodes.Tasklet) or isinstance(edge.dst, nodes.Tasklet):
                    if len(elements) != 1:
                        return None
                    plan.append((state, edge, elements, None))
                    continue
                if other is None or _extents(other) and len(_extents(other)) != len(_extents(own)):
                    return None
                others = _other_elements(other, len(elements))
                if others is None:
                    return None
                plan.append((state, edge, elements, others))
        return plan

    def _split(self, sdfg: SDFG, name: str, desc: data.Data) -> bool:
        return split_constant_indexed_buffer(sdfg, name, desc)


def split_constant_indexed_buffer(sdfg: SDFG, name: str, desc: data.Data | None = None) -> bool:
    """Split the buffer ``name`` of ``sdfg`` into one scalar per element if it qualifies (see the module)."""
    desc = sdfg.arrays[name] if desc is None else desc
    if not isinstance(desc, data.Array) or isinstance(desc, data.View) or not desc.transient:
        return False
    shape = [as_expr(s) for s in desc.shape]
    if not all(isinstance(s, sympy.Integer) for s in shape) or len(shape) == 0:
        return False
    if int(sympy.prod(shape)) <= 1 or int(sympy.prod(shape)) > MAX_ELEMENTS:
        return False
    accesses = _buffer_accesses(sdfg, name)
    if not accesses:
        return False
    plan = SplitConstantIndexedBuffers._plan(sdfg, name, accesses)
    if plan is None:
        return False
    scalars: dict[tuple[int, ...], str] = {}

    def scalar(element: tuple[int, ...]) -> str:
        if element not in scalars:
            scalars[element] = sdfg.add_scalar(
                f"{name}_{'_'.join(map(str, element))}",
                desc.dtype,
                storage=desc.storage,
                transient=True,
                find_new_name=True,
            )[0]
        return scalars[element]

    # One access node per element and original node, so each keeps its node's position in the state.
    per_node: dict[tuple[int, tuple[int, ...]], nodes.AccessNode] = {}

    def element_node(state: SDFGState, node: nodes.AccessNode, element: tuple[int, ...]) -> nodes.AccessNode:
        key = (id(node), element)
        if key not in per_node:
            per_node[key] = state.add_access(scalar(element))
        return per_node[key]

    for state, edge, elements, others in plan:
        on_src = isinstance(edge.src, nodes.AccessNode) and edge.src.data == name
        node = edge.src if on_src else edge.dst
        for index, element in enumerate(elements):
            piece = element_node(state, node, element)
            if others is None:
                memlet = Memlet(data=scalar(element), subset="0")
            else:
                other_node = edge.dst if on_src else edge.src
                memlet = Memlet(data=other_node.data, subset=others[index], other_subset="0")
            if on_src:
                state.add_edge(piece, edge.src_conn, edge.dst, edge.dst_conn, memlet)
            else:
                state.add_edge(edge.src, edge.src_conn, piece, edge.dst_conn, memlet)
        state.remove_edge(edge)
    for state, node in accesses:
        state.remove_node(node)
    del sdfg.arrays[name]
    return True
