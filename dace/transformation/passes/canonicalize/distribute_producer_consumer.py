# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Producer->consumer loop distribution: split a loop whose body is a linear
chain of statements into one loop per ordered statement group, allowing a
FORWARD flow dependence (an earlier statement writes a container a later one
reads) to cross a group boundary.

``LoopFission`` distributes only *independent* groups: any container written by
one block and touched by another forces a merge, so a producer->consumer pair
(atax ``for i: { tmp[i] = A[i,:]@x ; y += A[i,:]*tmp[i] }`` -- ``tmp`` produced
then consumed) stays fused. That fusion blocks the downstream contraction lift
(``LoopToEinsum``), which needs each matvec in its own perfect nest.

This pass adds the forward case. Distribution around a statement partition is
legal iff no dependence *cycle* spans the split (Allen & Kennedy): keep every
strongly-connected component of the loop-body dependence graph together and
emit the resulting loops in the graph's topological order. Because we preserve
the ORIGINAL block order, every forward (loop-independent or forward
loop-carried) dependence is already satisfied; the only thing that forms a
cycle is a BACKWARD loop-carried dependence -- a *later* block writing a
container an *earlier* block reads or writes (an anti/WAR, output/WAW, or
flow-back edge from the next iteration). So the merge rule is exactly:

    merge(Bi, Bj) with i < j  iff  writes(Bj) & (reads(Bi) | writes(Bi)) != {}

Everything else (earlier writes, later only reads -- a forward producer; or
read-only sharing; or no sharing) may split. The producer loop fully
materializes the shared container before the consumer loop reads it, so the
split is value-preserving.

The analysis is container-name level and therefore CONSERVATIVE: two blocks
that write the SAME array at provably-disjoint subsets (e.g. covariance's
``cov[i,j]`` compute vs ``cov[j,i]`` mirror) are kept together here; splitting
those needs subset-disjointness reasoning and is out of scope for this pass.
"""

import functools
import weakref
from collections.abc import Callable
from typing import Any

import networkx as nx

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.loop_fission import _is_per_iter_subset, _linear_blocks


def _interstate_reads(block: ControlFlowBlock) -> list[str]:
    """Containers ``block`` reads through an interstate edge or a branch condition.

    Such a read never becomes an AccessNode: ``if mask[i]:`` lowers to an assignment
    ``mask_index = mask[i]`` on the edge into a ConditionalBlock, plus the branch condition on the
    symbol. A walk over AccessNodes therefore reports the container as UNREAD by this block, and a
    pass deciding a split on that answer separates the mask from the statements that read it.
    """
    if isinstance(block, SDFGState):
        return []
    arrays = block.sdfg.arrays
    found: dict[str, None] = {}
    for edge in block.all_interstate_edges(recursive=True):
        # Sorted: free symbols are a set, and their hash order would order the reads the grouping visits.
        found.update(dict.fromkeys(sorted(name for name in edge.data.free_symbols if name in arrays)))
    for region in block.all_control_flow_regions(recursive=True):
        if not isinstance(region, ConditionalBlock):
            continue
        for condition, _ in region.branches:
            if condition is None:
                continue
            found.update(dict.fromkeys(sorted(name for name in condition.get_free_symbols() if name in arrays)))
    return list(found)


def _rw_subsets(block: ControlFlowBlock, loop_var: str | None) -> tuple[dict[str, bool], dict[str, bool]]:
    """For every container ``block`` writes / reads, whether ALL of its accesses
    are per-iteration (single-point at ``loop_var`` with zero offset).

    :returns: ``(writes, reads)`` dicts mapping container name -> ``True`` iff
        every write / read memlet of that container is per-iteration w.r.t.
        ``loop_var`` (``_is_per_iter_subset``). A container absent from a dict
        is not written / read by the block.
    """
    writes: dict[str, bool] = {}
    reads: dict[str, bool] = {}
    states = [block] if isinstance(block, SDFGState) else list(block.states())
    for st in states:
        for n in st.nodes():
            if not isinstance(n, nodes.AccessNode):
                continue
            for e in st.in_edges(n):
                if e.data is None:
                    continue
                sub = e.data.get_dst_subset(e, st) if e.data.subset is None else e.data.subset
                writes[n.data] = writes.get(n.data, True) and _is_per_iter_subset(sub, loop_var)
            if st.out_degree(n) > 0 or st.in_degree(n) == 0:
                for e in st.out_edges(n):
                    if e.data is None:
                        continue
                    sub = e.data.get_src_subset(e, st) if e.data.subset is None else e.data.subset
                    reads[n.data] = reads.get(n.data, True) and _is_per_iter_subset(sub, loop_var)
                if st.out_degree(n) == 0 and st.in_degree(n) == 0:
                    reads.setdefault(n.data, False)
    # False, not True: the subset an interstate read uses is not analyzable here, and the forward
    # producer->consumer split is only legal when BOTH sides are provably per-iteration. Recording
    # it as unanalyzable keeps such a pair fused.
    for name in _interstate_reads(block):
        reads[name] = False
    # A symbol is never a per-iteration subset: one assigned inside the block (KMP's state ``q``) is written, one
    # the block uses is read, so a block reading a symbol another block assigns stays fused with it.
    if not isinstance(block, SDFGState):
        for edge in block.all_interstate_edges(recursive=True):
            writes.update(dict.fromkeys(edge.data.assignments, False))
    for name in map(str, block.free_symbols):
        if name not in block.sdfg.arrays:
            reads[name] = False
    return writes, reads


def charge_body_edges(loop: LoopRegion, rw: dict) -> None:
    """Charge each body edge's assignments and the names they read to the block the edge enters.

    :func:`assignment_coupled` already keeps the edge's ends in one group; this makes the rest of the body see
    what the edge reads and writes.
    """
    for edge in loop.edges():
        writes, reads = rw[edge.dst]
        writes.update(dict.fromkeys(edge.data.assignments, False))
        reads.update(dict.fromkeys(edge.data.free_symbols, False))


def assignment_coupled(loop: LoopRegion, order: list[ControlFlowBlock]) -> dict[ControlFlowBlock, ControlFlowBlock]:
    """Union-find parents that already join the blocks a body-edge assignment couples.

    A symbol assigned on a body edge exists only in the clone holding that edge, so the edge's ends and every
    block reading the symbol stay in one group (CSR ``row_ptr_index := row_ptr[i]`` feeding an inner loop).
    """
    parent: dict[ControlFlowBlock, ControlFlowBlock] = {b: b for b in order}

    def root(x: ControlFlowBlock) -> ControlFlowBlock:
        while parent[x] != x:
            x = parent[x]
        return x

    reads = {b: {str(sym) for sym in b.free_symbols} for b in order}
    for edge in loop.edges():
        if not edge.data.assignments:
            continue
        readers = [b for b in order if set(edge.data.assignments) & reads[b]]
        for block in [edge.src, *readers]:
            parent[root(block)] = root(edge.dst)
    # A symbol a block assigns inside itself (kmp's ``q = fail[q - 1]`` in a branch) is read by the next iteration
    # of every block that reads it: assigner and readers carry it together, so they stay in one loop
    assigns = {b: assigned_in(b) for b in order}
    for block in order:
        assigned = assigns[block]
        for other in order:
            if other is not block and assigned & (reads[other] | assigns[other]):
                parent[root(other)] = root(block)
    return parent


def assigned_in(block: ControlFlowBlock) -> set[str]:
    """The symbols the interstate edges inside ``block`` assign."""
    if not isinstance(block, AbstractControlFlowRegion):
        return set()
    return {name for edge in block.all_interstate_edges(recursive=True) for name in edge.data.assignments}


def _forward_flow_groups(loop: LoopRegion, allow_assignments: bool = False) -> list[list[ControlFlowBlock]] | None:
    """Partition ``loop``'s linear body into ordered groups split only across aligned forward flow.

    ``Bi`` and ``Bj`` (``i < j``) sharing ``X`` merge if ``Bj`` writes ``X`` (backward WAR / WAW /
    flow), or if ``Bi`` writes and ``Bj`` reads ``X`` without a provably aligned per-iteration index
    on both sides (scalar accumulators, ``a[i+1]``, offsets like ``b[i-1]``). The aligned case (atax
    ``tmp[i]``, covariance ``cov[i,j]``) splits. ``None`` if not a linear chain of >= 2 blocks or
    nothing splits.

    :param allow_assignments: Also split a body whose edges assign symbols, each such edge kept with the blocks
                              that read them. Only the expansion fallback asks for it: a plain sequential body split
                              there loses the fusion its pieces get as one loop (CloudSC's ``iphase`` nest).
    """
    order = _linear_blocks(loop, allow_assignments=allow_assignments)
    if order is None or len(order) < 2:
        return None
    lv = loop.loop_variable
    rw = {b: _rw_subsets(b, lv) for b in order}
    charge_body_edges(loop, rw)
    parent = assignment_coupled(loop, order)
    flows: list[tuple[ControlFlowBlock, ControlFlowBlock]] = []

    def find(x: ControlFlowBlock) -> ControlFlowBlock:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, bi in enumerate(order):
        writes_i, reads_i = rw[bi]
        # writes_i/reads_i are already ordered dicts; dict.fromkeys unions them without
        # dropping into a hash-order set (the old `set(a) | set(b)` did).
        touched_i = dict.fromkeys([*writes_i, *reads_i])
        for bj in order[i + 1 :]:
            writes_j, reads_j = rw[bj]
            merge = False
            for x in dict.fromkeys(k for k in touched_i if k in writes_j or k in reads_j):
                if x in writes_j:
                    merge = True  # backward WAR / WAW / flow-back
                    break
                if x in writes_i and x in reads_j:
                    # forward producer -> consumer: safe only if aligned per-iter
                    # on both the producing write and the consuming read.
                    if not (writes_i[x] and reads_j[x]):
                        merge = True
                        break
                    flows.append((bi, bj))
            if merge:
                parent[find(bi)] = find(bj)

    groups = flow_ordered_groups(order, find, flows)
    return groups if len(groups) >= 2 else None


def flow_ordered_groups(
    order: list[ControlFlowBlock],
    find: Callable[[ControlFlowBlock], ControlFlowBlock],
    flows: list[tuple[ControlFlowBlock, ControlFlowBlock]],
) -> list[list[ControlFlowBlock]]:
    """The union-find classes of ``order``, each producer's class before its consumers'.

    A class split from the middle of the body can feed a class that starts before it; classes a flow cycle joins
    merge into one.
    """
    pos = {b: i for i, b in enumerate(order)}
    graph = nx.DiGraph()
    graph.add_nodes_from(find(b) for b in order)
    graph.add_edges_from((find(a), find(b)) for a, b in flows if find(a) is not find(b))
    condensed = nx.condensation(graph)
    members: dict[int, list[ControlFlowBlock]] = {c: [] for c in condensed.nodes}
    for b in order:
        members[condensed.graph["mapping"][find(b)]].append(b)
    return [members[c] for c in nx.lexicographical_topological_sort(condensed, key=lambda c: pos[members[c][0]])]


@transformation.explicit_cf_compatible
class DistributeProducerConsumerLoop(ppl.Pass):
    """Distribute a linear-chain loop across forward (producer->consumer) deps.

    The complement of ``LoopFission``: where fission separates only independent
    groups, this also separates a producer statement from the consumer that
    reads its output, emitting the producer loop first. Runs in the loops-only
    window so a coupled ``for i: { produce; consume }`` becomes two perfect
    nests each liftable on its own.
    """

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        # Function-local: ``perfect_loop_nesting`` imports this module.
        from dace.transformation.passes.canonicalize.perfect_loop_nesting import distribute_loops, groups_with_expansion

        return (
            distribute_loops(sdfg, grouping=functools.partial(groups_with_expansion, tried=weakref.WeakSet())) or None
        )


__all__ = ["DistributeProducerConsumerLoop", "_forward_flow_groups"]
