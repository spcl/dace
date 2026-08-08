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
crosses into the next chunk needs the original value. That is ONE element per chunk::

    snap[lo+1 : hi : C] = arr[...]         # one seam element per chunk
    snap[hi+1]          = arr[hi+1]
    parallel map i in [lo, lo]:            # prologue, reads the snapshot
    parallel map t:  sequential map i in [t, e(t)-1]:  arr[i] = f(arr[i+1], ...)
    parallel map t:  map i in [e(t), e(t)]:            arr[i] = f(snap[i+1], ...)

with ``e(t) = min(t + C - 1, hi)`` the last index of chunk ``t``. The copy drops from
``hi - lo`` elements to ``(hi - lo) / C + 1``.

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


def _clone_contents(src: SDFGState, dst: SDFGState) -> None:
    """Copy every node and edge of ``src`` into the empty state ``dst``.

    The node list is deepcopied as ONE object so a MapEntry and its MapExit keep sharing
    the single ``Map`` they describe; deepcopying node by node would hand them two.
    """
    src_nodes = src.nodes()
    clones = copy.deepcopy(src_nodes)
    mapping = dict(zip(src_nodes, clones))
    for n in clones:
        dst.add_node(n)
    for e in src.edges():
        dst.add_edge(mapping[e.src], e.src_conn, mapping[e.dst], e.dst_conn, copy.deepcopy(e.data))


@properties.make_properties
class ChunkAntiDependence(ppl.Pass):
    """Rewrite a CPU snapshot-broken read-ahead map into parallel chunks with seam buffers."""

    CATEGORY: str = 'Device Specialization'

    chunk_size = properties.Property(dtype=int,
                                     default=4096,
                                     desc='Iterations per chunk. Large enough that the per-chunk seam element is '
                                     'negligible, small enough to leave many chunks for the thread pool.')

    def __init__(self, chunk_size: int = 4096) -> None:
        super().__init__()
        self.chunk_size = chunk_size

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
            out_edges = state.out_edges(snap_node)
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

    def _tile(self, state: SDFGState, sdfg: SDFG, me: nodes.MapEntry) -> nodes.MapEntry:
        """Wrap ``me`` in an outer chunk map and return the outer entry.

        ``MapTiling`` is the existing orthogonal-tiling transformation; it leaves ``me`` as
        the inner map over ``[t, min(t + C - 1, hi)]`` and adds the outer map over the chunk
        starts, with the memlets already propagated.
        """
        from dace.transformation.dataflow.tiling import MapTiling
        MapTiling.apply_to(sdfg,
                           options=dict(prefix='antidep_chunk', tile_sizes=(self.chunk_size, ), tile_trivial=True),
                           map_entry=me,
                           save=False,
                           verify=False)
        outer = state.entry_node(me)
        outer.map.schedule = dtypes.ScheduleType.CPU_Multicore
        # Sequential inner, parallel outer. The differing schedules also stop MapCollapse
        # from fusing the two back into one parallel map, which would race the seam.
        me.map.schedule = dtypes.ScheduleType.Sequential
        return outer

    def _rewrite(self, state: SDFGState, sdfg: SDFG, match: Tuple) -> None:
        snap_node, arr, me, lo, hi = match
        snap = snap_node.data
        parent = state.parent_graph
        chunk = symbolic.pystr_to_symbolic(str(self.chunk_size))

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
        _clone_contents(state, pro)
        _clone_contents(state, tail)

        # Seam elements: the first read (``lo + 1``, the prologue's) and one per chunk
        # boundary, plus the final read that no chunk boundary lands on.
        strided = subsets.Range([(lo + 1, hi, chunk)])
        last = subsets.Range([(hi + 1, hi + 1, 1)])
        for sub in (strided, last):
            seam.add_nedge(seam.add_read(arr), seam.add_write(snap),
                           Memlet(data=arr, subset=copy.deepcopy(sub), other_subset=copy.deepcopy(sub)))

        # Prologue: the one iteration whose read-ahead has no chunk in front of it.
        pro_entry = next(n for n in pro.nodes() if isinstance(n, nodes.MapEntry))
        pro_entry.map.range = subsets.Range([(lo, _min(lo, hi), 1)])

        # Chunk body: sequential inside a chunk, so every read-ahead that stays inside the
        # chunk sees the value the sequential loop would have seen -- read the live array.
        me.map.range = subsets.Range([(lo + 1, hi, 1)])
        self._tile(state, sdfg, me)
        inner_lo, inner_hi, _ = me.map.range[0]
        me.map.range = subsets.Range([(_exact(inner_lo), _exact(inner_hi) - 1, 1)])
        for n in state.data_nodes():
            if n.data == snap:
                n.data = arr
        for e in state.edges():
            if e.data is not None and e.data.data == snap:
                e.data.data = arr

        # Seam iterations: the last index of every chunk, whose read-ahead crosses into the
        # next chunk and must come from the snapshot.
        tail_entry = next(n for n in tail.nodes() if isinstance(n, nodes.MapEntry) and tail.entry_node(n) is None)
        tail_entry.map.range = subsets.Range([(lo + 1, hi, 1)])
        self._tile(tail, sdfg, tail_entry)
        t_lo, t_hi, _ = tail_entry.map.range[0]
        seam_idx = _exact(t_hi)
        tail_entry.map.range = subsets.Range([(seam_idx, seam_idx, 1)])

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
