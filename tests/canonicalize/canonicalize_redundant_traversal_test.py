# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The EXTRA passes over memory canonicalization buys its parallelism with, pinned per kernel.

Three focus-40 kernels canonicalized into a correct parallel form that measured SLOWER than the
sequential baseline at the XL preset, and the reason was the same in all three: the parallel form
streams the same arrays more times than the one sequential loop did. Run on ONE thread, where
neither page placement nor fork/join can be blamed, the canonical form of ``ext_break_post_body``
cost 1.90x the sequential loop's time, ``s1244`` 1.68x and ``s319`` 1.11x -- each within reach of
the extra traversals counted here, which were 1.5x, 1.4x and 1.4x of the sequential form's array
traffic. The traversals are the durable half of that: whether they add up to a LOSS turned out to
depend entirely on the machine, and on one of these three kernels the answer flipped.

Two of the three were fixed in their lowering; the third had its lowering withdrawn. The break
kernels are pinned here anyway, because the surcharge is what any future break lowering has to
buy back, and a lowering that reappears without a cost model has to fail this file first.

That surcharge is what a parallel form has to buy back, and how much it can buy back is a property
of the run rather than of the graph. On a 64-core EPYC 7A53 with 4 NUMA domains -- the cluster's
DEFAULT partition, and NOT the 24-physical-core MI300A quadrant the grading contract pins
submissions to -- the harness first-touches its arrays on one thread, and the same three kernels
measured 0.71x / 0.78x / 0.92x at 64 threads but 1.97x / 1.63x / 3.04x under
``numactl --interleave=all``. On one MI300A quadrant under ``--localalloc``, where a rank's first
touch is already local, ``s1244`` measured 2.62x before the fix below and was never a loss at all.
So a speedup number here says as much about where it ran as about what canonicalization did, which
is exactly why nothing below is a timing: the counts are the part canonicalization owns, and the
part that survives being measured somewhere else.

What each kernel pays, and what one of them stopped paying:

* ``ext_break_post_body`` (TSVC ``s482``) and ``ext_break_find_first`` (``s481``) -- both stay
  SEQUENTIAL, so neither buys any traversal at all. The lowering that used to lift them put a
  whole ``FindFirst`` pass over the predicate arrays in front of a map that read them again, and
  it matched on the break's shape alone: on ``s482``, whose predicate ``c[i] > b[i]`` names the
  same arrays as the body ``a[i] += b[i] * c[i]``, that was three extra streams for a form
  measuring 0.70x, while ``s481``'s predicate over ``d`` -- an array its body never reads --
  added none and measured 1.03x. The overlap, not the lowering, was the cost, and nothing in the
  rewrite ever priced it, so the rewrite was withdrawn rather than gated. The case against
  withdrawing it is worth keeping: the re-stream is a CPU cost paid in a device-neutral stage,
  and on a GPU the break is divergent control flow whose removal is the whole point -- which is
  why ``FindFirst`` keeps its CUDA expansion over ``dace::find_first_index_device``. A lift that
  returns on that argument needs somewhere to hand the parallelism back, and the
  ``cpu_specialize`` band that would do it only re-schedules Maps today.
* ``s1244_d_single`` -- FIXED, and pinned here so it stays fixed. The ``d[i] = a[i] + a[i + 1]``
  anti dependence used to be broken with a FULL-LENGTH snapshot of ``a``: one extra read of ``a``
  and one extra write plus read of the snapshot.
  :class:`~dace.transformation.passes.cpu_specialization.chunk_anti_dependence.ChunkAntiDependence`
  exists to replace exactly that with a one-element-per-chunk seam, and ``ext_war_unit`` (the same
  dependence with no second statement) always got the seam. What separated them was not the
  dependence but two EMPTY ordering edges hanging off the snapshot node, which its ``_match``
  counted as readers still needing the whole window.
* ``s319_d_single`` -- FIXED, and pinned here so it stays fixed. The two elementwise stores and
  the accumulation over what they just wrote used to land in two states, so ``a`` and ``b`` were
  written by one map and read back by the next. The accumulation now rides the stores, and the
  kernel streams only its three inputs.

The counts below are structural, never timings: a wall-clock assertion is noise in CI, and the
traversal count is the thing that actually decides the wall clock here. Sizes are evaluated with
every free symbol pinned to :data:`N`, so a pass over the whole array and a pass over one element
per chunk are told apart by their trip count rather than by their name.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import Iterable

import dace
from dace import symbolic
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.ordered import OrderedSet
from dace.sdfg import nodes as nd
from dace.sdfg.state import BreakBlock, LoopRegion, SDFGState
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc_2_5 import tsvc_2_5

#: The recipe settings the rest of this directory canonicalizes the corpora at.
PEEL_LIMIT = 4
UNROLL_LIMIT = 4

#: Kernel-name suffix that routes a name to the TSVC corpus rather than to TSVC-2.5.
TSVC_SUFFIX = "_d_single"

#: Value every free symbol is pinned to when a trip count is evaluated. A power of two well above
#: the 4096 chunk width, so a whole-array pass and a per-chunk pass cannot land in the same bucket.
N = 1 << 20

#: Thread count the seam sizing is evaluated at -- the grading contract's width, so a per-thread
#: buffer is judged at the size it will really have.
THREADS = 24

#: Trip count at or above which a pass counts as streaming the whole array.
FULL_LENGTH = N // 2


def canonical_sdfg(name: str) -> dace.SDFG:
    """A FRESH SDFG for ``name``, canonicalized and finalized for the CPU.

    Built per kernel and never reused: canonicalize mutates in place and dace's pass state is
    process-global, so a graph another case has already been through is not the kernel described
    here.
    """
    if name.endswith(TSVC_SUFFIX):
        kernel = tsvc.collect(name=name)[0]
        sdfg = tsvc.to_sdfg(kernel, "traversal", simplify=True)
        canonicalize(sdfg, target="cpu", validate=True, peel_limit=PEEL_LIMIT)
    else:
        program = next(p for p in tsvc_2_5.collect() if p.f.__name__ == name)
        sdfg = program.to_sdfg(simplify=True)
        canonicalize(sdfg, target="cpu", validate=True, peel_limit=PEEL_LIMIT, unroll_limit=UNROLL_LIMIT)
    finalize_for_target(sdfg, "cpu")
    return sdfg


def evaluated(expr) -> int:
    """``expr`` with every free symbol pinned to :data:`N`, as an int.

    A data-dependent bound is as much a whole-array pass as a static one -- the search that
    produced it can land anywhere -- so its symbol is pinned like any other rather than making
    the pass uncountable.
    """
    parsed = symbolic.pystr_to_symbolic(str(expr))
    # ``__dace_num_threads`` is a MACHINE property, not a problem dimension: frame code defines it
    # from ``omp_get_max_threads``. Pinning it to N like an extent makes a per-thread buffer look
    # problem-sized, which is the opposite of what it is.
    values = {str(s): (THREADS if str(s) == symbolic.NUM_THREADS_SYMBOL else N) for s in parsed.free_symbols}
    return int(symbolic.evaluate(parsed, values))


def scope_trips(state: SDFGState, entry: nd.MapEntry) -> int:
    """Iterations of ``entry``'s whole scope: its own range times every map range nested in it.

    The product, not the outer range alone: ``ChunkAntiDependence`` leaves a chunk map whose 256
    iterations each run 4096 more, and reading only the outer one would call a whole-array sweep a
    per-chunk touch.
    """
    trips = entry.map.range.num_elements()
    for node in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nd.MapEntry):
            trips = trips * node.map.range.num_elements()
    return evaluated(trips)


def read_names(edges: Iterable[MultiConnectorEdge[Memlet]]) -> OrderedSet:
    """The arrays a scope reads, deduplicated -- one scope streams one array once."""
    return OrderedSet(e.data.data for e in edges if e.data is not None and not e.data.is_empty())


def full_length_reads(sdfg: dace.SDFG) -> list[str]:
    """Every whole-array READ the finalized form makes, one entry per pass, array names sorted.

    An array read by two different top-level scopes appears twice: that repetition IS the cost this
    file is about. Reads inside a scope are not counted separately -- a map that names ``b`` on
    three of its edges still streams ``b`` once.
    """
    passes: list[str] = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nd.MapEntry) and state.entry_node(node) is None:
                if scope_trips(state, node) >= FULL_LENGTH:
                    passes.extend(read_names(state.in_edges(node)))
            elif isinstance(node, nd.LibraryNode):
                for edge in state.in_edges(node):
                    if edge.data is not None and evaluated(edge.data.volume) >= FULL_LENGTH:
                        passes.append(edge.data.data)
            elif isinstance(node, nd.AccessNode):
                for edge in state.out_edges(node):
                    if not isinstance(edge.dst, nd.AccessNode) or edge.data is None or edge.data.is_empty():
                        continue
                    if evaluated(edge.data.volume) >= FULL_LENGTH:
                        passes.append(edge.data.data)
    return sorted(passes)


def sequential_break_loops(sdfg: dace.SDFG) -> int:
    """Loop regions that still carry a ``break``, i.e. the shape no lowering claimed."""
    return sum(
        1 for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, LoopRegion) and any(isinstance(b, BreakBlock) for b, _ in node.all_nodes_recursive()))


def anti_dependence_buffers(sdfg: dace.SDFG) -> dict[str, int]:
    """Every anti-dependence buffer canonicalization allocated, by its element count at :data:`N`."""
    return {
        name: evaluated(desc.total_size)
        for name, desc in sdfg.arrays.items()
        if desc.transient and (name.endswith("_split_snap") or "_antidep_seam" in name)
    }


def test_s482_break_loop_buys_no_traversal():
    """``s482`` stays one sequential loop, so it streams each array exactly as the source does.

    It used to make six whole-array reads and writes where the sequential loop made four -- the
    +50% traffic behind 1.90x the sequential time on ONE thread, where no threading effect can be
    blamed for it. The predicate was not what made the search long: under the corpus fill it fires
    at 94% of the range, so the search and the sequential loop visited the same elements, and the
    surcharge was the second pass over ``b`` and ``c``, not the first.

    The list is also asserted to hold no name twice: the property the withdrawn lowering never
    reached is that no array is read by both a search and the body it clips.
    """
    sdfg = canonical_sdfg("ext_break_post_body")
    reads = full_length_reads(sdfg)
    assert reads == [], reads
    assert len(reads) == len(set(reads)), reads
    assert sequential_break_loops(sdfg) == 1, "the break loop must survive as one sequential LoopRegion"


def test_s481_break_loop_buys_no_traversal():
    """``s481`` stays sequential too, and for the same reason: nothing rewrites a break.

    Pinned alongside ``s482`` because the two kernels are what separated a sound rewrite from a
    profitable one. ``s481``'s predicate reads ``d``, an array its body never touches, so the
    withdrawn lowering added no stream here and measured 1.03x where ``s482`` measured 0.70x. A
    break lowering that returns may well be right for this shape and wrong for the other, which is
    the distinction it has to make rather than inherit.
    """
    sdfg = canonical_sdfg("ext_break_find_first")
    reads = full_length_reads(sdfg)
    assert reads == [], reads
    assert sequential_break_loops(sdfg) == 1, "the break loop must survive as one sequential LoopRegion"


def test_war_unit_snapshot_is_a_per_chunk_seam():
    """``ext_war_unit``'s anti dependence costs one buffered element per chunk, not one per index.

    The reference the s1244 case is held against: same dependence, same offset, same 1-D unit
    stride, and this is the size such a buffer should be.
    """
    seams = anti_dependence_buffers(canonical_sdfg("ext_war_unit"))
    assert list(seams) == ["a_antidep_seam"], seams
    assert seams["a_antidep_seam"] < N // 1000, seams


def test_s1244_gets_the_per_chunk_seam():
    """``s1244`` reaches the seam too, so its anti dependence costs no whole-array copy.

    It did not until ``ChunkAntiDependence._match`` stopped counting the snapshot's two EMPTY
    ORDERING edges (``a_split_snap -> b``, ``-> c``) as readers that would still need the whole
    window. Those edges name no data, so no rewrite can be unsound for cutting the copy they
    order. What it was worth at XL on the 24-physical-core MI300A quadrant submissions are graded
    on, geomean over the node's four quadrants with the answer checked against the numpy oracle
    every time: 2.62x -> 3.87x of the sequential baseline, the canonical form's own time 48.1 ms
    -> 32.6 ms.
    """
    buffers = anti_dependence_buffers(canonical_sdfg("s1244_d_single"))
    assert list(buffers) == ["a_antidep_seam"], buffers
    assert buffers["a_antidep_seam"] < N // 1000, buffers


def test_s319_accumulates_in_the_map_that_writes():
    """``s319`` streams its three inputs and nothing else -- no pass back over ``a`` or ``b``.

    Seven whole-array reads and writes where the sequential loop made five was what the two-state
    form cost; the accumulation rides the stores now, so the two extra passes are gone. Pinned so
    the read-back cannot come back.
    """
    reads = full_length_reads(canonical_sdfg("s319_d_single"))
    assert reads == ["c", "d", "e"], reads
