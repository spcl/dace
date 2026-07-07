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
from typing import Dict, List, Optional, Tuple

from dace.sdfg import nodes
from dace import SDFG
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.loop_fission import LoopFission, _linear_blocks, _is_per_iter_subset
from dace.sdfg.utils import set_nested_sdfg_parent_references


def _rw_subsets(block, loop_var: Optional[str]) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """For every container ``block`` writes / reads, whether ALL of its accesses
    are per-iteration (single-point at ``loop_var`` with zero offset).

    :returns: ``(writes, reads)`` dicts mapping container name -> ``True`` iff
        every write / read memlet of that container is per-iteration w.r.t.
        ``loop_var`` (``_is_per_iter_subset``). A container absent from a dict
        is not written / read by the block.
    """
    writes: Dict[str, bool] = {}
    reads: Dict[str, bool] = {}
    states = [block] if isinstance(block, SDFGState) else list(block.all_states())
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
    return writes, reads


def _forward_flow_groups(loop: LoopRegion) -> Optional[List[List]]:
    """Partition ``loop``'s linear body blocks into ordered groups, allowing a
    FORWARD flow dependence (an earlier block writes a container -- at the
    aligned per-iteration index -- that a later block reads at that same index)
    to cross a group boundary. Everything else merges. ``None`` if the body is
    not a plain linear chain of >= 2 blocks or nothing splits.

    Merge ``Bi``, ``Bj`` (``i < j``) sharing container ``X`` iff:

    - ``X`` is WRITTEN by the later ``Bj`` -- a backward anti (WAR) / output
      (WAW) / flow-back edge (later group -> earlier group), always fatal; or
    - ``X`` is a forward producer->consumer (``Bi`` writes, ``Bj`` reads) whose
      dependence is NOT provably aligned per-iteration on BOTH sides. A scalar
      accumulator (``s = s + a[i]``; not per-iter), a future read (``a[i+1]``),
      or any cross-iteration / unanalyzable subset falls here: distributing it
      would let the consumer see the wrong (final, or not-yet-overwritten)
      value. Only the aligned same-index case (atax ``tmp[i]``, covariance
      ``cov[i,j]``) -- where the producer loop fully materialises ``X`` before
      the consumer loop reads exactly its own index -- is safe to split.

    (Conservative: a legal-but-offset forward read like ``b[i-1]`` is kept
    fused rather than reasoned about -- soundness over completeness.)
    """
    order = _linear_blocks(loop)
    if order is None or len(order) < 2:
        return None
    lv = loop.loop_variable
    pos = {b: i for i, b in enumerate(order)}
    rw = {b: _rw_subsets(b, lv) for b in order}
    parent: Dict = {b: b for b in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, bi in enumerate(order):
        writes_i, reads_i = rw[bi]
        touched_i = set(writes_i) | set(reads_i)
        for bj in order[i + 1:]:
            writes_j, reads_j = rw[bj]
            merge = False
            for x in touched_i & (set(writes_j) | set(reads_j)):
                if x in writes_j:
                    merge = True  # backward WAR / WAW / flow-back
                    break
                if x in writes_i and x in reads_j:
                    # forward producer -> consumer: safe only if aligned per-iter
                    # on both the producing write and the consuming read.
                    if not (writes_i[x] and reads_j[x]):
                        merge = True
                        break
            if merge:
                parent[find(bi)] = find(bj)

    classes: Dict = {}
    for b in order:
        classes.setdefault(find(b), []).append(b)
    groups = sorted((sorted(g, key=lambda b: pos[b]) for g in classes.values()), key=lambda g: pos[g[0]])
    return groups if len(groups) >= 2 else None


@transformation.explicit_cf_compatible
class DistributeProducerConsumerLoop(ppl.Pass):
    """Distribute a linear-chain loop across forward (producer->consumer) deps.

    The complement of ``LoopFission``: where fission separates only independent
    groups, this also separates a producer statement from the consumer that
    reads its output, emitting the producer loop first. Runs in the loops-only
    window so a coupled ``for i: { produce; consume }`` becomes two perfect
    nests each liftable on its own.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        count = 0
        changed = True
        # Each split rebuilds the CFG, so restart the scan after applying one
        # (mirrors ``LoopFission.apply_pass``); distributing an outer loop can
        # expose a newly-splittable inner one on the next sweep.
        while changed:
            changed = False
            for loop in list(sdfg.all_control_flow_regions(recursive=True)):
                if not isinstance(loop, LoopRegion):
                    continue
                groups = _forward_flow_groups(loop)
                if groups is None:
                    continue
                LoopFission._fission_blocks(loop, groups)
                set_nested_sdfg_parent_references(sdfg)
                count += 1
                changed = True
                break
        return count or None


__all__ = ['DistributeProducerConsumerLoop', '_forward_flow_groups']
