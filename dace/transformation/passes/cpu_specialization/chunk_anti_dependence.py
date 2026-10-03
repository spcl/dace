# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Trade an anti-dependence snapshot for a per-chunk seam buffer, on CPU only.

:class:`~dace.transformation.passes.break_anti_dependence.BreakAntiDependence` leaves
the device-neutral canonical form of a read-ahead WAR loop::

    snap[W] = arr[W]                       # one copy of the window the loop reads
    parallel map i in [lo, hi]: arr[i] = f(snap[i + 1], ...)

Every iteration is independent, so it offloads to a GPU unchanged. On a CPU it costs
two extra streams: the snapshot is read and written once more than the sweep needs.
For ``arr[i] = arr[i+1] + b[i]`` that is 5N of traffic where the sequential loop moves
3N, and it measures 0.52x of the plain sequential loop at N = 589824 on 8 threads.

The CPU rewrite keeps the loop parallel but restores the 3N traffic: run contiguous
CHUNKS in parallel and each chunk's own iterations in order. Sequential order inside a
chunk already satisfies a read-AHEAD dependence -- iteration ``i`` reads ``arr[i+1]``,
which only iteration ``i+1`` overwrites, and that runs later -- so only the read that
crosses into the next chunk needs the original value. That is ONE element per chunk, so
the buffer is indexed by CHUNK rather than by array index and holds ``nchunks + 1``
elements instead of the whole window::

    seam[0 : nchunks] = arr[lo+1 : hi : C]   # chunk k's first element, gathered
    seam[nchunks]     = arr[hi+1]            # the final read, on no chunk boundary
    parallel map i in [lo, lo]:                        arr[i] = f(seam[0], ...)
    parallel map t:  sequential map i in [t, e(t)-1]:  arr[i] = f(arr[i+1], ...)
    parallel map t:  map i in [e(t), e(t)]:            arr[i] = f(seam[k(t) + 1], ...)

with ``e(t) = min(t + C - 1, hi)`` the last index of chunk ``t`` and ``k(t)`` its index.
Chunk ``k`` reads across into chunk ``k+1``'s first element, uniformly slot ``k + 1`` --
the last chunk included, whose read lands on ``arr[hi+1]`` in the final slot. Both the
copy and the buffer drop from ``hi - lo`` elements to ``(hi - lo) / C + 1``.

This is a DEVICE SPECIALIZATION, not a canonical form: sequential-within-chunk is a CPU
scheduling decision (one thread per chunk, per-chunk ordering), and it does not offload.
It runs in the CPU lowering band and never matches a GPU-scheduled map, which keeps the
snapshot form.

Restricted to a read-ahead offset of exactly 1 on a 1-D array under unit stride. A wider
offset ``D`` needs ``2D`` contiguous seam elements per chunk, which is a strided copy of
a run rather than of a point and is not expressible as one memlet on a 1-D array; those
loops keep the snapshot. Pure data movement either way -- no arithmetic is reassociated,
so the result is bit-identical.
"""
import copy
from typing import Any, Dict, Optional, Tuple

from dace import data, dtypes, properties, subsets, symbolic, Memlet
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl

#: Schedules that lower through the CPU path. A GPU-scheduled map never matches.
_CPU_SCHEDULES = (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent,
                  dtypes.ScheduleType.Sequential)


def _min(a, b):
    """``min(a, b)`` built through DaCe's parser, not raw sympy."""
    return symbolic.pystr_to_symbolic(f'Min({symbolic.symstr(a)}, {symbolic.symstr(b)})')


def _diff(a, b):
    """``a - b`` with both sides reparsed, so equally-named symbols cancel."""
    return symbolic.simplify(symbolic.pystr_to_symbolic(f'({a}) - ({b})'))


def _exact(bound):
    """The exact expression of a tiled range bound.

    ``StripMining`` (under ``MapTiling``) writes range bounds as :class:`~dace.symbolic.SymExpr`
    pairs (exact, over-approximated). Reusing one verbatim in a fresh range leaves a ``SymExpr``
    where downstream symbol replacement (``TrivialMapElimination`` on the single-iteration seam
    map) sympifies the substitution value and fails. The seam index must be the exact chunk end,
    so take ``.expr``.
    """
    return bound.expr if isinstance(bound, symbolic.SymExpr) else bound


def _clone_contents(src: SDFGState, dst: SDFGState, sdfg: SDFG) -> None:
    """Copy every node and edge of ``src`` into the empty state ``dst``, privatizing the temporaries.

    The node list is deepcopied as ONE object so a MapEntry and its MapExit keep sharing
    the single ``Map`` they describe; deepcopying node by node would hand them two.

    A transient every access to which sits INSIDE the map scope is one instance per iteration --
    the privatized WCR accumulators, the split-statement temporaries. Sharing its name across the
    clones would leave one descriptor serving three different map scopes, and codegen declares such
    a transient in the scope it meets first: the prologue's loop body then declares it and the chunk
    body and the seam iterations reference a name that is not in scope there (``s212``, ``'...' was
    not declared in this scope``). So each clone gets its own descriptor. The seam buffer and the
    arrays the map reads are entered from the state's top level, not from inside the scope, and stay
    shared -- being read by all three states is what they are for.
    """
    src_nodes = src.nodes()
    scope_local = {}
    for node in src_nodes:
        if isinstance(node, nodes.AccessNode):
            scope_local[node.data] = scope_local.get(node.data, True) and src.entry_node(node) is not None
    private = {
        name: sdfg.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]), find_new_name=True)
        for name, only_inside in scope_local.items() if only_inside and sdfg.arrays[name].transient
    }

    clones = copy.deepcopy(src_nodes)
    mapping = dict(zip(src_nodes, clones))
    for n in clones:
        if isinstance(n, nodes.AccessNode) and n.data in private:
            n.data = private[n.data]
        dst.add_node(n)
    for e in src.edges():
        memlet = copy.deepcopy(e.data)
        if memlet is not None and memlet.data in private:
            memlet.data = private[memlet.data]
        dst.add_edge(mapping[e.src], e.src_conn, mapping[e.dst], e.dst_conn, memlet)


@properties.make_properties
class ChunkAntiDependence(ppl.Pass):
    """Rewrite a CPU snapshot-broken read-ahead map into parallel chunks with seam buffers."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _match(self, state: SDFGState, sdfg: SDFG) -> Optional[Tuple]:
        """``(snap_node, arr, map_entry, lo, hi)`` for the canonical snapshot pattern, else None."""
        from dace.sdfg.scope import is_devicelevel_gpu
        for snap_node in state.data_nodes():
            desc = sdfg.arrays[snap_node.data]
            if not (desc.transient and isinstance(desc, data.Array) and len(desc.shape) == 1):
                continue
            in_edges = state.in_edges(snap_node)
            # An EMPTY out-edge is an ORDERING edge, not a reader: it names no data at all, so it
            # cannot be the one thing this guard is looking for -- a consumer that would still need
            # the whole window the rewrite stops copying. Counting one as a reader is what left
            # s1244 (whose snapshot carries two, a_split_snap -> b and -> c) on the full-length copy.
            out_edges = [e for e in state.out_edges(snap_node) if e.data is not None and not e.data.is_empty()]
            if len(in_edges) != 1 or not out_edges:
                continue
            src = in_edges[0].src
            if not isinstance(src, nodes.AccessNode) or in_edges[0].src_conn is not None:
                continue
            arr = src.data
            if len(sdfg.arrays[arr].shape) != 1:
                continue
            # The snapshot must be private to this state: a reader elsewhere would still
            # need the whole window this rewrite is about to stop copying.
            others = [
                n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, nodes.AccessNode) and n.data == snap_node.data and n is not snap_node
            ]
            if others:
                continue
            me = out_edges[0].dst
            if any(e.dst is not me for e in out_edges):
                continue
            if not isinstance(me, nodes.MapEntry) or state.entry_node(me) is not None:
                continue
            if me.map.schedule not in _CPU_SCHEDULES or is_devicelevel_gpu(sdfg, state, me):
                continue
            if len(me.map.params) != 1 or len(me.map.range) != 1:
                continue
            lo, hi, step = me.map.range[0]
            if step != 1:
                continue
            if not self._offsets_are_unit(state, sdfg, me, snap_node.data, arr):
                continue
            return snap_node, arr, me, lo, hi
        return None

    def _offsets_are_unit(self, state: SDFGState, sdfg: SDFG, me: nodes.MapEntry, snap: str, arr: str) -> bool:
        """Every snapshot read is ``[i + 1]`` and every live ``arr`` access is ``[i]``.

        A read at a wider offset would need a run of seam elements per chunk; a live read
        at a nonzero offset would cross a chunk boundary without a seam behind it.
        """
        param = me.map.params[0]
        seen_snap = False
        for e in state.edges():
            if e.data is None or e.data.is_empty() or e.data.data not in (snap, arr):
                continue
            sub = e.data.subset
            if sub is None or len(sub) != 1:
                continue
            beg, end, _ = sub[0]
            # Print-and-reparse both sides: a symbol carries its dtype in its identity, so the
            # iterator read off a memlet and one built here do not cancel until DaCe's own
            # parser has given both the same identity.
            if _diff(beg, end) != 0:
                continue  # a scope-level range, not a point access
            off = _diff(beg, param)
            if param in {str(s) for s in off.free_symbols}:
                continue  # not an affine point access in the iterator: a scope range
            if e.data.data == snap:
                if off != 1:
                    return False
                seen_snap = True
            elif off != 0:
                return False
        return seen_snap

    def _chunk_stride(self, extent):
        """The per-chunk iteration count, so the chunk COUNT lands on the thread count.

        The seam holds one slot per chunk, so whichever of (size, count) is fixed decides how much
        this pass copies. Fixing the SIZE makes the seam grow with the array -- at 4096 an XL
        extent of 5.2e8 still buys 127,000 chunks and a 127,001-element copy. Fixing the COUNT at
        the thread count makes the seam thread-sized no matter how long the array is: one boundary
        element per thread, which is the least a chunked anti-dependence break can copy.

        ``__dace_num_threads`` is defined by frame code and excluded from the ABI, so the stride is
        symbolic but the allocation is a couple of dozen elements. ``int_ceil``, never ``/``: the
        last chunk is short and a floor would drop its iterations.
        """
        threads = symbolic.pystr_to_symbolic(symbolic.NUM_THREADS_SYMBOL)
        return symbolic.int_ceil(symbolic.pystr_to_symbolic(str(extent)), threads)

    def _redirect_to_seam(self, state: SDFGState, snap: str, seam: str, slot, outer: Optional[Tuple]) -> None:
        """Repoint every ``snap`` read in ``state`` at seam slot ``slot``.

        ``outer`` is ``(entry, union)`` for the state whose reads sit inside a chunk map: the
        edge entering ``entry`` is outside that scope, so it cannot name the chunk parameter
        ``slot`` is built from and carries the union of the slots instead.
        """
        point = subsets.Range([(slot, slot, 1)])
        outer_entry, union = outer if outer is not None else (None, None)
        for n in state.data_nodes():
            if n.data == snap:
                n.data = seam
        for e in state.edges():
            if e.data is None or e.data.data != snap:
                continue
            sub = union if e.dst is outer_entry else point
            e.data.data = seam
            e.data.subset = copy.deepcopy(sub)
            e.data.volume = sub.num_elements()

    def _tile(self, state: SDFGState, sdfg: SDFG, me: nodes.MapEntry, chunk_size) -> nodes.MapEntry:
        """Wrap ``me`` in an outer chunk map and return the outer entry.

        ``MapTiling`` is the existing orthogonal-tiling transformation; it leaves ``me`` as
        the inner map over ``[t, min(t + C - 1, hi)]`` and adds the outer map over the chunk
        starts, with the memlets already propagated.
        """
        from dace.transformation.dataflow.tiling import MapTiling
        MapTiling.apply_to(sdfg,
                           options=dict(prefix='antidep_chunk', tile_sizes=(chunk_size, ), tile_trivial=True),
                           map_entry=me,
                           save=False,
                           verify=False)
        outer = state.entry_node(me)
        outer.map.schedule = dtypes.ScheduleType.CPU_Multicore
        # Sequential inner, parallel outer. The differing schedules also stop MapCollapse
        # from fusing the two back into one parallel map, which would race the seam.
        me.map.schedule = dtypes.ScheduleType.Sequential
        return outer

    def _sequentialize(self, state: SDFGState, sdfg: SDFG, me: nodes.MapEntry) -> None:
        """Turn the in-chunk sweep into a ``LoopRegion``, leaving only the chunks a map.

        A Map asserts that its iterations carry NO dependence, and the in-chunk sweep carries one:
        iteration ``i`` reads ``arr[i + 1]``, which iteration ``i + 1`` overwrites, so it is the
        ORDER inside a chunk that makes reading the live array legal here. A ``Sequential`` schedule
        does not say that -- it is a lowering hint, and every consumer is still entitled to act on
        the map's parallelism claim. The tile vectorizer does: it widens the sweep and sinks the
        ``arr[i + 1]`` load past the ``arr[i]`` store, so seven lanes out of eight read values the
        same tile has just overwritten (``s212``, silently wrong ``b``).

        The chunk map above it stays a map, which is where the parallelism actually is: on a
        many-core node the chunks fill the machine, and the sweep inside one is a plain C loop that
        the host compiler is free to vectorize on its own dependence analysis.
        """
        # Avoid import loop: dataflow transformations import the pass pipeline this module defines.
        from dace.transformation.dataflow.map_for_loop import MapToForLoop
        to_loop = MapToForLoop()
        # The loop belongs inside the chunk map's scope, so keep the wrapping NestedSDFG rather
        # than inlining it up to the parent region, where a map scope cannot hold it.
        to_loop.inline_after = False
        to_loop.map_entry = me
        to_loop.apply(state, sdfg)

    def _rewrite(self, state: SDFGState, sdfg: SDFG, match: Tuple) -> None:
        snap_node, arr, me, lo, hi = match
        snap = snap_node.data
        parent = state.parent_graph
        chunk = self._chunk_stride(_diff(hi, lo))
        # Read the chunk count off the gather range itself rather than spelling the same
        # ceiling a second way: validation compares the two forms syntactically, and DaCe's
        # ``Range.size`` builds a sympy ``ceiling`` that never matches a fresh ``int_ceil``
        # (both print as ``int_ceil`` in C++, so this costs nothing downstream).
        gather = subsets.Range([(lo + 1, hi, chunk)])
        nchunks = gather.num_elements()
        # The seam is sized by the THREAD COUNT, not by the chunk count it happens to produce:
        # ``nchunks`` is ``int_ceil(E, int_ceil(E, T))``, which is never above ``T`` and equals it
        # for any extent worth chunking. Sizing on ``T`` keeps the allocation one slot per thread
        # plus the trailing read -- the least a chunked anti-dependence break can copy -- and keeps
        # the shape a plain sum instead of a nested rounding.
        threads = symbolic.pystr_to_symbolic(symbolic.NUM_THREADS_SYMBOL)

        # Drop the whole-window snapshot copy; the seam state replaces it.
        copy_edge = state.in_edges(snap_node)[0]
        src = copy_edge.src
        state.remove_edge(copy_edge)
        if state.degree(src) == 0:
            state.remove_node(src)

        # Clone the map BEFORE redirecting it, so the prologue and the seam iterations keep
        # reading the snapshot while the chunk body moves to the live array.
        pro = parent.add_state_before(state, label=f'{arr}_antidep_prologue')
        seam = parent.add_state_before(pro, label=f'{arr}_antidep_seams')
        tail = parent.add_state_after(state, label=f'{arr}_antidep_seam_iters')
        _clone_contents(state, pro, sdfg)
        _clone_contents(state, tail, sdfg)

        # Seam elements: each chunk's first element, gathered into consecutive slots, plus the
        # final read that no chunk boundary lands on. Strided source, contiguous destination.
        # SCOPE lifetime, not the default. The size names ``__dace_num_threads``, which a graph
        # tasklet supplies once the program is running -- a persistent buffer is allocated in
        # ``__dace_init``, where that symbol does not yet exist and the C++ does not compile. The
        # seam is a couple of dozen elements, so allocating it per call costs nothing.
        buf, desc = sdfg.add_transient(f'{arr}_antidep_seam', [threads + 1],
                                       sdfg.arrays[snap].dtype,
                                       storage=sdfg.arrays[snap].storage,
                                       lifetime=dtypes.AllocationLifetime.State,
                                       find_new_name=True)
        gathered = (gather, subsets.Range([(0, nchunks - 1, 1)]))
        trailing = (subsets.Range([(hi + 1, hi + 1, 1)]), subsets.Range([(nchunks, nchunks, 1)]))
        for src_sub, dst_sub in (gathered, trailing):
            seam.add_nedge(seam.add_read(arr), seam.add_write(buf),
                           Memlet(data=arr, subset=src_sub, other_subset=dst_sub))

        # Prologue: the one iteration whose read-ahead has no chunk in front of it, so it
        # reads chunk 0's own first element -- slot 0.
        pro_entry = next(n for n in pro.nodes() if isinstance(n, nodes.MapEntry))
        pro_entry.map.range = subsets.Range([(lo, _min(lo, hi), 1)])
        self._redirect_to_seam(pro, snap, buf, symbolic.pystr_to_symbolic('0'), None)

        # Chunk body: sequential inside a chunk, so every read-ahead that stays inside the
        # chunk sees the value the sequential loop would have seen -- read the live array.
        me.map.range = subsets.Range([(lo + 1, hi, 1)])
        self._tile(state, sdfg, me, chunk)
        inner_lo, inner_hi, _ = me.map.range[0]
        me.map.range = subsets.Range([(_exact(inner_lo), _exact(inner_hi) - 1, 1)])
        for n in state.data_nodes():
            if n.data == snap:
                n.data = arr
        for e in state.edges():
            if e.data is not None and e.data.data == snap:
                e.data.data = arr
        self._sequentialize(state, sdfg, me)

        # Seam iterations: the last index of every chunk, whose read-ahead crosses into the
        # next chunk and must come from the buffer -- chunk ``k`` reads slot ``k + 1``.
        tail_entry = next(n for n in tail.nodes() if isinstance(n, nodes.MapEntry) and tail.entry_node(n) is None)
        tail_entry.map.range = subsets.Range([(lo + 1, hi, 1)])
        tail_outer = self._tile(tail, sdfg, tail_entry, chunk)
        t_lo, t_hi, _ = tail_entry.map.range[0]
        seam_idx = _exact(t_hi)
        tail_entry.map.range = subsets.Range([(seam_idx, seam_idx, 1)])
        # Off the OUTER chunk parameter, whose stride is C, not off the inner index: the inner
        # map is a single point whose bound is already a Min, which no division would survive.
        chunk_id = symbolic.int_floor(symbolic.pystr_to_symbolic(tail_outer.map.params[0]) - (lo + 1), chunk)
        self._redirect_to_seam(tail, snap, buf, symbolic.simplify(chunk_id + 1),
                               (tail_outer, subsets.Range([(1, nchunks, 1)])))

        # Nothing reads the whole-window snapshot any more; ``remove_data`` validates that.
        sdfg.remove_data(snap)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Rewrite every CPU snapshot-broken read-ahead map; returns how many, or ``None``."""
        applied = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in list(sd.states()):
                match = self._match(state, sd)
                if match is None:
                    continue
                self._rewrite(state, sd, match)
                applied += 1
        return applied or None
