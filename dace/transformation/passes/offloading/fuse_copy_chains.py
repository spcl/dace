# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A chain of copies within one array, none writing what another reads or writes, becomes one map.

fv3's corner fill copies the columns of a halo one statement at a time (``q[0, 0, :] = q[0, 5, :]``, 36 of them per
call). The frontend chains such statements as access nodes of ``q`` joined by copy edges, and the device offload
turns every copy into its own transfer: 204 ``hipMemcpyAsync`` per fv3 time step. When no copy of the chain writes
an element another copy reads or writes, their order is immaterial: the chain is one data-parallel map over the
copied extent, one tasklet per copy, and the offload makes it one kernel.
"""

from typing import Any

from dace import SDFG, Memlet, data, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState, SymbolResolver
from dace.transformation import pass_pipeline as ppl


def is_copy(state: SDFGState, edge) -> bool:
    """An edge copying part of an array into another access node of the same array."""
    return (
        isinstance(edge.src, nodes.AccessNode)
        and isinstance(edge.dst, nodes.AccessNode)
        and edge.src.data == edge.dst.data
        and not edge.data.is_empty()
        and edge.data.wcr is None
    )


def is_link(state: SDFGState, node: nodes.AccessNode) -> bool:
    """A node inside a chain: one copy leads in, one copy leads on, nothing else touches it."""
    return (
        state.in_degree(node) == 1
        and state.out_degree(node) == 1
        and is_copy(state, state.in_edges(node)[0])
        and is_copy(state, state.out_edges(node)[0])
    )


def copy_chains(state: SDFGState) -> list[list]:
    """Every maximal chain of at least two copies, as its edges in order."""
    chains = []
    for start in state.data_nodes():
        if is_link(state, start):
            continue
        for first in state.out_edges(start):
            if not is_copy(state, first):
                continue
            chain = [first]
            while is_link(state, chain[-1].dst):
                chain.append(state.out_edges(chain[-1].dst)[0])
            if len(chain) > 1:
                chains.append(chain)
    return chains


def copied_axes(subset: subsets.Subset) -> list[tuple[int, Any]] | None:
    """The ``(dimension, extent)`` of every axis of ``subset`` wider than one element; ``None`` unless a Range."""
    if not isinstance(subset, subsets.Range):
        return None
    return [(dim, size) for dim, size in enumerate(subset.size()) if (size == 1) != True]


def element(subset: subsets.Range, axes: list[tuple[int, Any]], params: list[str]) -> subsets.Range:
    """The element of ``subset`` at ``params`` along its copied ``axes``."""
    ranges = list(subset.ranges)
    for (dim, _), param in zip(axes, params, strict=True):
        begin, _, step = ranges[dim]
        point = begin + symbolic.pystr_to_symbolic(param) * step
        ranges[dim] = (point, point, 1)
    return subsets.Range(ranges)


def fusible(state: SDFGState, chain: list) -> list[tuple] | None:
    """Per copy, its source and destination subsets and copied axes, when they all copy the same extents and every
    destination is disjoint from every other destination and from every source; ``None`` otherwise."""
    if isinstance(state.sdfg.arrays[chain[0].src.data], data.View):
        return None
    ends = [(e.data.get_src_subset(e, state), e.data.get_dst_subset(e, state)) for e in chain]
    shaped = [(src, dst, copied_axes(src), copied_axes(dst)) for src, dst in ends]
    if any(src_axes is None or dst_axes is None for _, _, src_axes, dst_axes in shaped):
        return None
    extents = [size for _, size in shaped[0][2]]
    for _, _, src_axes, dst_axes in shaped:
        for axes in (src_axes, dst_axes):
            if len(axes) != len(extents) or any(
                symbolic.simplify(size - want) != 0 for (_, size), want in zip(axes, extents)
            ):
                return None
    facts = SymbolResolver().facts_at(state, chain[0].src)
    for i, (_, dst, _, _) in enumerate(shaped):
        for j, (src, other, _, _) in enumerate(shaped):
            touched = [src] if i == j else [src, other]
            if any(subsets.intersects(dst, t, facts) is not symbolic.Truth.FALSE for t in touched):
                return None
    return shaped


class FuseCopyChains(ppl.Pass):
    """Turn every chain of independent copies within one array into one map, one tasklet per copy."""

    CATEGORY: str = "Device Offloading"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many chains became maps, or ``None`` if none did."""
        fused = 0
        for state in sdfg.states():
            for chain in copy_chains(state):
                shaped = fusible(state, chain)
                if shaped is None:
                    continue
                name, start, end = chain[0].src.data, chain[0].src, chain[-1].dst
                links = [e.dst for e in chain[:-1]]
                for edge in chain:
                    state.remove_edge(edge)
                state.remove_nodes_from(links)
                params = [f"__copy{d}" for d in range(len(shaped[0][2]))]
                entry, exit_node = state.add_map(
                    "copy_chain", {p: f"0:{size}" for p, (_, size) in zip(params, shaped[0][2], strict=True)}
                )
                for src, dst, src_axes, dst_axes in shaped:
                    copy = state.add_tasklet("copy", {"__in"}, {"__out"}, "__out = __in")
                    state.add_memlet_path(
                        start,
                        entry,
                        copy,
                        dst_conn="__in",
                        memlet=Memlet(data=name, subset=element(src, src_axes, params)),
                    )
                    state.add_memlet_path(
                        copy,
                        exit_node,
                        end,
                        src_conn="__out",
                        memlet=Memlet(data=name, subset=element(dst, dst_axes, params)),
                    )
                fused += 1
        return fused or None
