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

A read-ahead offset ``D >= 1`` generalizes this: the last ``D`` iterations of a chunk read
the first ``D`` cells of the next one, so the seam holds a ROW of ``D`` cells per chunk,
gathered by one parallel map, and the chunk size is at least ``D`` so a full chunk's row
never reaches past its neighbour. Restricted to a constant ``D`` on a 1-D array under unit
stride. Pure data movement either way -- no arithmetic is reassociated, so the result is
bit-identical.
"""

import copy
from typing import Any

from dace import Memlet, data, dtypes, properties, subsets, symbolic
from dace.optionals import required
from dace.sdfg import SDFG, nodes
from dace.sdfg.narrowing import as_basic, as_expr, as_map_entry
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl

#: Schedules that lower through the CPU path. A GPU-scheduled map never matches.
_CPU_SCHEDULES = (
    dtypes.ScheduleType.Default,
    dtypes.ScheduleType.CPU_Multicore,
    dtypes.ScheduleType.CPU_Persistent,
    dtypes.ScheduleType.Sequential,
)


def _min(a, b):
    """``min(a, b)`` built through DaCe's parser, not raw sympy."""
    return symbolic.pystr_to_symbolic(f"Min({symbolic.symstr(a)}, {symbolic.symstr(b)})")


def _diff(a, b):
    """``a - b`` with both sides reparsed, so equally-named symbols cancel."""
    return symbolic.simplify(symbolic.pystr_to_symbolic(f"({a}) - ({b})"))


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
        for name, only_inside in scope_local.items()
        if only_inside and sdfg.arrays[name].transient
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

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _match(self, state: SDFGState, sdfg: SDFG) -> tuple | None:
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
                n
                for n, _ in sdfg.all_nodes_recursive()
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
            offset = self._read_ahead_offset(state, me, snap_node.data, arr)
            if offset is None:
                continue
            return snap_node, arr, me, lo, hi, offset
        return None

    def _read_ahead_offset(self, state: SDFGState, me: nodes.MapEntry, snap: str, arr: str) -> int | None:
        """The one constant ``D >= 1`` every snapshot read is ``[i + D]`` at, every live ``arr`` access being ``[i]``.

        A live read at a nonzero offset would cross a chunk boundary without a seam behind it.
        """
        param = me.map.params[0]
        offsets: set[int] = set()
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
            if param in {str(s) for s in as_basic(off).free_symbols}:
                continue  # not an affine point access in the iterator: a scope range
            if e.data.data == snap:
                if not (as_basic(off).is_Integer and int(off) >= 1):
                    return None
                offsets.add(int(off))
            elif off != 0:
                return None
        return offsets.pop() if len(offsets) == 1 else None

    def _chunk_stride(self, extent, offset: int):
        """The per-chunk iteration count, so the chunk COUNT lands on the thread count.

        The seam holds one slot per chunk, so whichever of (size, count) is fixed decides how much
        this pass copies. Fixing the SIZE makes the seam grow with the array -- at 4096 an XL
        extent of 5.2e8 still buys 127,000 chunks and a 127,001-element copy. Fixing the COUNT at
        the thread count makes the seam thread-sized no matter how long the array is: one boundary
        element per thread, which is the least a chunked anti-dependence break can copy.

        ``__dace_num_threads`` is defined by frame code and excluded from the ABI, so the stride is
        symbolic but the allocation is a couple of dozen elements. ``int_ceil``, never ``/``: the
        last chunk is short and a floor would drop its iterations.

        A chunk is never shorter than the offset ``D``: the last ``D`` iterations of a full chunk then read the first
        ``D`` cells of the next chunk and nothing beyond, which is the row the seam keeps for it.
        """
        threads = symbolic.pystr_to_symbolic(symbolic.NUM_THREADS_SYMBOL)
        per_thread = symbolic.int_ceil(symbolic.pystr_to_symbolic(str(extent)), threads)
        return (
            per_thread if offset == 1 else symbolic.pystr_to_symbolic(f"Max({symbolic.symstr(per_thread)}, {offset})")
        )

    def _redirect_to_seam(
        self, state: SDFGState, snap: str, seam: str, slot, base, union: subsets.Range, outer_entry
    ) -> None:
        """Repoint every ``snap`` read in ``state`` at seam row ``slot``, columns ``(read indices) - base``.

        The edges entering a map scope sit outside it, so they cannot name the parameters ``slot`` and the read
        index are built from, and carry ``union`` instead; the edges inside name one cell.
        """
        for n in state.data_nodes():
            if n.data == snap:
                n.data = seam
        for e in state.edges():
            if e.data is None or e.data.data != snap:
                continue
            if isinstance(e.dst, nodes.MapEntry) and (outer_entry is None or e.dst is outer_entry):
                sub = union
            elif len(union) == 1:
                sub = subsets.Range([(slot, slot, 1)])  # offset 1: one cell per row, the row is the slot
            else:
                first, last, _ = e.data.subset[0]
                sub = subsets.Range([(slot, slot, 1), (first - base, last - base, 1)])
            e.data.data = seam
            e.data.subset = copy.deepcopy(sub)
            e.data.volume = required(sub).num_elements()

    def _tile(self, state: SDFGState, sdfg: SDFG, me: nodes.MapEntry, chunk_size) -> nodes.MapEntry:
        """Wrap ``me`` in an outer chunk map and return the outer entry.

        ``MapTiling`` is the existing orthogonal-tiling transformation; it leaves ``me`` as
        the inner map over ``[t, min(t + C - 1, hi)]`` and adds the outer map over the chunk
        starts, with the memlets already propagated.
        """
        from dace.transformation.dataflow.tiling import MapTiling

        MapTiling.apply_to(
            sdfg,
            options=dict(prefix="antidep_chunk", tile_sizes=(chunk_size,), tile_trivial=True),
            map_entry=me,
            save=False,
            verify=False,
        )
        outer = state.entry_node(me)
        as_map_entry(outer).map.schedule = dtypes.ScheduleType.CPU_Multicore
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

    def _rewrite(self, state: SDFGState, sdfg: SDFG, match: tuple) -> None:
        snap_node, arr, me, lo, hi, offset = match
        snap = snap_node.data
        parent = state.parent_graph
        chunk = self._chunk_stride(_diff(hi, lo + offset - 1), offset)
        # Read the chunk count off the gather range itself rather than spelling the same
        # ceiling a second way: validation compares the two forms syntactically, and DaCe's
        # ``Range.size`` builds a sympy ``ceiling`` that never matches a fresh ``int_ceil``
        # (both print as ``int_ceil`` in C++, so this costs nothing downstream).
        gather = subsets.Range([(lo + offset, hi, chunk)])
        nchunks = gather.num_elements()
        if offset > 1:
            # The chunk size is a ``Max`` here, and sympy's ``ceiling`` over it reaches C++ as an integer division that
            # rounds down; ``int_ceil`` is the count the tiled chunk map runs.
            nchunks = symbolic.int_ceil(as_expr(_diff(hi, lo + offset - 1)), chunk)
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
        pro = parent.add_state_before(state, label=f"{arr}_antidep_prologue")
        seam = parent.add_state_before(pro, label=f"{arr}_antidep_seams")
        tail = parent.add_state_after(state, label=f"{arr}_antidep_seam_iters")
        _clone_contents(state, pro, sdfg)
        _clone_contents(state, tail, sdfg)

        # Seam elements: each chunk's first element, gathered into consecutive slots, plus the
        # final read that no chunk boundary lands on. Strided source, contiguous destination.
        # SCOPE lifetime, not the default. The size names ``__dace_num_threads``, which a graph
        # tasklet supplies once the program is running -- a persistent buffer is allocated in
        # ``__dace_init``, where that symbol does not yet exist and the C++ does not compile. The
        # seam is a couple of dozen elements, so allocating it per call costs nothing.
        buf, desc = sdfg.add_transient(
            f"{arr}_antidep_seam",
            [as_expr(threads) + 1] + ([offset] if offset > 1 else []),
            sdfg.arrays[snap].dtype,
            storage=sdfg.arrays[snap].storage,
            lifetime=dtypes.AllocationLifetime.State,
            find_new_name=True,
        )
        row = (lambda r: [(r, r, 1)]) if offset == 1 else (lambda r: [(r, r, 1), (0, offset - 1, 1)])
        if offset == 1:
            seam.add_nedge(
                seam.add_read(arr),
                seam.add_write(buf),
                Memlet(data=arr, subset=gather, other_subset=subsets.Range([(0, nchunks - 1, 1)])),
            )
        else:
            seam.add_mapped_tasklet(
                f"{arr}_antidep_gather",
                {"_seam_j": f"0:{symbolic.symstr(nchunks)}", "_seam_m": f"0:{offset}"},
                {
                    "_in": Memlet(
                        f"{arr}[{symbolic.symstr(lo + offset)} + _seam_j * ({symbolic.symstr(chunk)}) + _seam_m]"
                    )
                },
                "_out = _in",
                {"_out": Memlet(f"{buf}[_seam_j, _seam_m]")},
                external_edges=True,
            )
        trailing = Memlet(
            data=arr,
            subset=subsets.Range([(hi + 1, hi + offset, 1)]),
            other_subset=subsets.Range(row(nchunks)),
        )
        seam.add_nedge(seam.add_read(arr), seam.add_write(buf), trailing)
        every_row = subsets.Range([(0, nchunks, 1)] + row(0)[1:])

        # Prologue: the one iteration whose read-ahead has no chunk in front of it, so it
        # reads chunk 0's own first element -- slot 0.
        pro_entry = next(n for n in pro.nodes() if isinstance(n, nodes.MapEntry))
        pro_entry.map.range = subsets.Range([(lo, _min(lo + offset - 1, hi), 1)])
        # Row 0 is chunk 0's first cells, or the trailing row when the range is too short for any chunk.
        self._redirect_to_seam(
            pro, snap, buf, symbolic.pystr_to_symbolic("0"), _min(lo + offset, hi + 1), every_row, None
        )

        # Chunk body: sequential inside a chunk, so every read-ahead that stays inside the
        # chunk sees the value the sequential loop would have seen -- read the live array.
        me.map.range = subsets.Range([(lo + offset, hi, 1)])
        self._tile(state, sdfg, me, chunk)
        inner_lo, inner_hi, _ = me.map.range[0]
        me.map.range = subsets.Range([(_exact(inner_lo), as_expr(_exact(inner_hi)) - offset, 1)])
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
        tail_entry.map.range = subsets.Range([(lo + offset, hi, 1)])
        tail_outer = self._tile(tail, sdfg, tail_entry, chunk)
        t_lo, t_hi, _ = tail_entry.map.range[0]
        chunk_end = _exact(t_hi)
        # The last ``D`` iterations of the chunk, never before its own start: a short last chunk is all seam.
        first = symbolic.pystr_to_symbolic(
            f"Max({symbolic.symstr(as_expr(chunk_end) - offset + 1)}, {symbolic.symstr(_exact(t_lo))})"
        )
        tail_entry.map.range = subsets.Range([(first, chunk_end, 1)])
        # Off the OUTER chunk parameter, whose stride is C, not off the inner index, whose bounds are a Min and a
        # Max that no division would survive.
        chunk_id = symbolic.int_floor(symbolic.pystr_to_symbolic(tail_outer.map.params[0]) - (lo + offset), chunk)
        self._redirect_to_seam(
            tail,
            snap,
            buf,
            symbolic.simplify(chunk_id + 1),
            as_expr(chunk_end) + 1,
            subsets.Range([(1, nchunks, 1)] + row(0)[1:]),
            tail_outer,
        )

        # Nothing reads the whole-window snapshot any more; ``remove_data`` validates that.
        sdfg.remove_data(snap)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
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
