# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map that reads ``A[S]`` and writes a transient that is only copied back to ``A[S]`` writes ``A[S]`` itself.

The frontend spells an update ``A[S] = A[S] + e`` as ``A[S] -> map -> t -> A[S]``: the operator's result, then the
assignment's copy. Each iteration writes the element it read, so writing ``A`` in place is exact; the transient and
the copy are gone, and the map is the read-modify-write ``AugAssignToWCR`` turns into a WCR write.
"""

from dataclasses import dataclass

import sympy

from dace import SDFG, SDFGState, data, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@dataclass
class CopyBack:
    """``map_exit -> transient -> target``: the map's writes of ``transient`` placed in ``target[subset]``."""

    map_exit: nodes.MapExit
    transient: nodes.AccessNode
    copy: MultiConnectorEdge[Memlet]
    #: The dimensions of ``subset`` the transient spans, in order; the others are single indices.
    kept: list[int]


def same_index(a: subsets.Range, b: subsets.Range) -> bool:
    """Whether ``a`` and ``b`` are the same single element."""
    if a.dims() != b.dims():
        return False
    return all(
        symbolic.simplify(sympy.sympify(x) - sympy.sympify(y)) == 0
        for (ab, ae, _), (bb, be, _) in zip(a.ranges, b.ranges)
        for x, y in ((ab, bb), (ae, be), (ab, ae))
    )


def placed(write: subsets.Range, subset: subsets.Range, kept: list[int]) -> subsets.Range:
    """The element of the target a write of the transient at ``write`` lands on."""
    begins = [begin for begin, _, _ in subset.ranges]
    for dim, (begin, _, _) in zip(kept, write.ranges):
        begins[dim] = begins[dim] + begin
    return subsets.Range([(begin, begin, 1) for begin in begins])


def copy_back(state: SDFGState, sdfg: SDFG, transient: nodes.AccessNode) -> CopyBack | None:
    """The copy-back ``transient`` closes, or ``None``."""
    desc = sdfg.arrays[transient.data]
    if not desc.transient or isinstance(desc, data.View) or state.in_degree(transient) != 1:
        return None
    if state.out_degree(transient) != 1:
        return None
    (write,), (copy,) = state.in_edges(transient), state.out_edges(transient)
    map_exit, target = write.src, copy.dst
    if not isinstance(map_exit, nodes.MapExit) or not isinstance(target, nodes.AccessNode):
        return None
    if target.data == transient.data or isinstance(sdfg.arrays[target.data], data.View) or copy.data.wcr is not None:
        return None
    if sum(1 for st in sdfg.states() for n in st.data_nodes() if n.data == transient.data) != 1:
        return None
    whole, subset = copy.data.get_src_subset(copy, state), copy.data.get_dst_subset(copy, state)
    if not isinstance(whole, subsets.Range) or not isinstance(subset, subsets.Range):
        return None
    if whole != subsets.Range.from_array(desc):
        return None
    kept = [dim for dim, (begin, end, _) in enumerate(subset.ranges) if sympy.sympify(end - begin) != 0]
    if len(kept) != len(desc.shape):
        return None
    return CopyBack(map_exit, transient, copy, kept)


def in_place(state: SDFGState, found: CopyBack) -> bool:
    """Whether every element the map writes is read, if at all, only by the iteration writing it: one write, straight
    into the map exit, and every read of the target at that element."""
    target = found.copy.dst
    subset = found.copy.data.get_dst_subset(found.copy, state)
    entry = state.entry_node(found.map_exit)
    writes = [
        e
        for e in state.in_edges(found.map_exit)
        if e.dst_conn == "IN_" + state.in_edges(found.transient)[0].src_conn[4:]
    ]
    if len(writes) != 1 or writes[0].data.is_empty() or writes[0].data.wcr is not None:
        return False
    write = writes[0].data.get_dst_subset(writes[0], state)
    if not isinstance(write, subsets.Range) or write.num_elements() != 1:
        return False
    element = placed(write, subset, found.kept)
    for edge in state.in_edges(entry):
        if edge.data.is_empty() or edge.data.data != target.data:
            continue
        for leaf in state.memlet_tree(edge).leaves():
            read = leaf.data.get_src_subset(leaf, state)
            if not isinstance(read, subsets.Range) or not same_index(read, element):
                return False
    # Anything else in the scope naming the target (a nested write, a second access node) is not this pattern
    inner = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    return not any(isinstance(n, nodes.AccessNode) and n.data == target.data for n in inner.nodes())


def eliminate(state: SDFGState, sdfg: SDFG, found: CopyBack) -> None:
    target, copy = found.copy.dst, found.copy
    subset = copy.data.get_dst_subset(copy, state)
    (outer,) = state.in_edges(found.transient)
    (inner,) = [e for e in state.in_edges(found.map_exit) if e.dst_conn == "IN_" + outer.src_conn[4:]]
    write = inner.data.get_dst_subset(inner, state)
    inner.data = Memlet(data=target.data, subset=placed(write, subset, found.kept))
    state.add_edge(found.map_exit, outer.src_conn, target, copy.dst_conn, Memlet(data=target.data, subset=subset))
    state.remove_node(found.transient)
    if not any(n.data == found.transient.data for st in sdfg.states() for n in st.data_nodes()):
        sdfg.remove_data(found.transient.data, validate=False)


@transformation.explicit_cf_compatible
class EliminateCopyBack(ppl.Pass):
    """A map reading ``A[S]`` and writing a transient whose only use copies it back to ``A[S]``, element by element,
    writes ``A[S]`` in place."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets)

    def apply_pass(self, sdfg: SDFG, _) -> int | None:
        """:returns: The number of copy-backs removed, or ``None`` if none was."""
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for transient in list(state.data_nodes()):
                    if transient not in state.nodes():
                        continue
                    found = copy_back(state, sd, transient)
                    if found is not None and in_place(state, found):
                        eliminate(state, sd, found)
                        count += 1
        return count or None
