# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize a loop-local transient buffer by expanding it with an extra dimension.

A transient array that is reused as a scratch buffer inside a loop -- fully (re)written
and then read on *every* iteration -- carries no value between iterations, yet because
all iterations touch the same elements it reads as a loop-carried write/write conflict and
blocks :class:`~dace.transformation.interstate.loop_to_map.LoopToMap`. Giving the buffer a
private copy per iteration (an extra dimension indexed by the loop variable) removes the
false dependence so the loop parallelizes::

    for i in range(M):          for i in range(M):        # now a Map
        buf[0:N] = f(i)    ->       buf[i, 0:N] = f(i)
        c[i] = g(buf)               c[i] = g(buf[i, 0:N])

This is the classic *array (scalar) expansion* / *array privatization* transformation.

The pass only expands a buffer when doing so turns a loop that ``LoopToMap`` currently
*refuses* into one it *accepts* (verified by re-running the match): a safe expansion that
would not help is left alone so the SDFG does not grow needlessly. A static safety guard
refuses any buffer that is genuinely carried (read before it is written in an iteration,
e.g. an accumulator) -- those stay sequential.

Layout: the new (slowest-varying) dimension is placed so the buffer's storage order is
preserved and each iteration's slice stays contiguous -- prepended (axis 0) for a
C-contiguous (row-major) array, appended (last axis) for a Fortran-contiguous (column-major)
array. A 1-D array's packing is ambiguous (its single stride is ``1`` in both orders), so it
follows the majority packing of the SDFG's multi-dimensional arrays.
"""
import contextlib
import functools
import io
import operator
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Storage-order tag for the new dimension's placement.
_C_ORDER = 'C'
_F_ORDER = 'F'

#: Array lifetimes we may re-shape (allocated per scope / per SDFG call, so a private
#: copy per iteration is sound). Persistent/Global arrays outlive the call and are left alone.
_REINDEXABLE_LIFETIMES = (dtypes.AllocationLifetime.Scope, dtypes.AllocationLifetime.SDFG)


def _prod(values) -> Any:
    """Product of ``values`` (symbolic-safe; empty product is ``1``)."""
    return functools.reduce(operator.mul, values, 1)


def _insert_index(subset: subsets.Range, index, axis: int) -> subsets.Range:
    """Return ``subset`` with a single-point range ``(index, index, 1)`` inserted at ``axis``.

    :param subset: The original memlet range.
    :param index: The symbolic index for the new dimension (the loop variable, normalized).
    :param axis: The position at which to insert the new dimension.
    :returns: A new :class:`~dace.subsets.Range` with the extra dimension.
    """
    ranges = list(subset.ranges)
    tiles = list(subset.tile_sizes)
    ranges.insert(axis, (index, index, 1))
    tiles.insert(axis, 1)
    out = subsets.Range(ranges)
    out.tile_sizes = tiles
    return out


class _Expansion:
    """An applied buffer expansion, with the information needed to undo it.

    :param desc: The expanded array descriptor.
    :param shape: The descriptor's pre-expansion shape/strides/total_size/offset.
    :param edits: The memlet subset edits as ``(memlet, attribute_name, old_subset)`` triples.
    """

    def __init__(self, desc: data.Array, shape, strides, total_size, offset, edits: List[Tuple[Any, str, Any]]):
        self.desc = desc
        self._shape = shape
        self._strides = strides
        self._total_size = total_size
        self._offset = offset
        self._edits = edits

    def undo(self):
        """Restore the descriptor shape and every reindexed memlet to its pre-expansion form."""
        self.desc.set_shape(self._shape, strides=self._strides, total_size=self._total_size, offset=self._offset)
        for memlet, attr, old in self._edits:
            setattr(memlet, attr, old)


@properties.make_properties
class BufferExpansion(ppl.Pass):
    """Expand loop-local scratch buffers by one loop-indexed dimension to unblock LoopToMap.

    See the module docstring. The pass scans every loop the SDFG (and its nested SDFGs)
    contains; for a loop that ``LoopToMap`` currently refuses, it expands the safe
    privatizable buffers accessed in that loop and keeps the expansion only if the loop
    then becomes parallelizable.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Re-running on a structurally changed SDFG can expose freshly-isolated buffers.
        return bool(modified & (ppl.Modifies.CFG | ppl.Modifies.Memlets | ppl.Modifies.Descriptors))

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
        """Expand the buffers that unblock a loop, in ``sdfg`` and every nested SDFG.

        :param sdfg: The SDFG to transform in place.
        :returns: ``{loop_label: [expanded_array, ...]}`` for the loops that became
                  parallelizable, or ``None`` if nothing was expanded.
        """
        expanded: Dict[str, List[str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            expanded.update(self._expand_sdfg(sd))
        return expanded or None

    def report(self, pass_retval: Any) -> Optional[str]:
        if not pass_retval:
            return None
        arrays = sum(len(v) for v in pass_retval.values())
        return f'BufferExpansion: expanded {arrays} buffer(s) to unblock {len(pass_retval)} loop(s)'

    # -- core ---------------------------------------------------------------------------

    def _expand_sdfg(self, sdfg: SDFG) -> Dict[str, List[str]]:
        """Expand beneficial buffers for the loops of a single SDFG (speculate, then verify).

        A loop ``LoopToMap`` already accepts needs no privatization; a loop it *refuses* is the
        only place a buffer expansion can help. So the (expensive, ``arrays x states``) buffer
        analysis runs only on the refused loops -- probing mappability separates the two.

        That mappability probe (``LoopToMap.can_be_applied_to``) is itself the pass's whole cost:
        it topological-sorts the loop and computes its dominators/branch-merges, ~0.3s per loop, so
        on a kernel with a few hundred post-MapToForLoop loops (channel_flow: 159) it dominates
        everything else by orders of magnitude. It is therefore gated behind a *cheap* candidate
        test (:meth:`_has_candidate_buffer`, pure set lookups against per-SDFG indices): a loop with
        no transient buffer that is both written inside it and confined to it has nothing to
        privatize and is never expanded, whatever ``LoopToMap`` would say -- so probing it only to
        throw the answer away is wasted. Skipping the probe on those loops leaves the expanded set
        identical while cutting almost all of the probe cost (channel_flow: 159 probes -> 1).
        """
        loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
        if not loops:
            return {}

        # Per-SDFG indices, built once from the pristine SDFG (the candidate gate consults all three
        # on every loop). Expansion never moves an access between states nor touches an interstate
        # edge, so the indices -- and the ambient storage order -- stay valid across the expansions
        # applied below.
        access_states = self._access_state_index(sdfg)
        interstate_syms = self._interstate_symbols(sdfg)
        write_index = self._write_state_index(sdfg)
        order: Optional[str] = None  # ambient storage order: built lazily, only when actually expanding

        kept: Dict[str, List[str]] = {}
        for loop in loops:
            ix = self._loop_index(loop)
            if ix is None:
                continue
            loop_states = set(loop.all_states())
            if not self._has_candidate_buffer(loop_states, write_index, access_states, interstate_syms):
                continue  # no privatizable-buffer candidate -> never expanded, so skip the probe
            if self._loop_mappable(sdfg, loop):
                continue  # already parallelizable -> expanding would only grow the SDFG
            buffers = self._privatizable_buffers(sdfg, loop, access_states, interstate_syms, loop_states)
            if not buffers:
                continue
            if order is None:
                order = self._ambient_order(sdfg)
            index, size = ix
            applied = [self._expand(sdfg, loop, arr, index, size, order) for arr in buffers]
            if self._loop_mappable(sdfg, loop):
                kept[loop.label] = list(buffers)
            else:
                for exp in applied:
                    exp.undo()
        return kept

    @staticmethod
    def _write_state_index(sdfg: SDFG) -> Dict[str, Set[SDFGState]]:
        """Map each *privatizable-kind* transient array to the states that WRITE it (one pass).

        A key is a transient, non-view, reindexable-lifetime Array of rank >= 1 -- exactly the
        descriptor-level filters :meth:`_privatizable_buffers` applies -- with at least one
        AccessNode fed by an incoming edge (a write). Read-only and never-accessed transients are
        omitted so the candidate gate iterates a small key set. The plain "has an incoming edge"
        test is a broad, sound stand-in for ``_defined_before_read``'s per-edge write check (it can
        only over-include), keeping the gate a necessary precondition for privatizability.
        """
        reindexable = {
            name
            for name, desc in sdfg.arrays.items() if isinstance(desc, data.Array) and not isinstance(desc, data.View)
            and desc.transient and desc.lifetime in _REINDEXABLE_LIFETIMES and len(desc.shape) >= 1
        }
        index: Dict[str, Set[SDFGState]] = {}
        if not reindexable:
            return index
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode) and n.data in reindexable and state.in_edges(n):
                    index.setdefault(n.data, set()).add(state)
        return index

    @staticmethod
    def _has_candidate_buffer(loop_states: Set[SDFGState], write_index: Dict[str, Set[SDFGState]],
                              access_states: Dict[str, Set[SDFGState]], interstate_syms: Set[str]) -> bool:
        """Whether ``loop_states`` could hold a privatizable buffer -- the cheap gate on the probe.

        True iff some reindexable transient is written inside the loop *and* used only inside it: the
        two conditions a privatizable buffer must meet that need no dominator or ``LoopToMap``
        analysis, so this is a pure set-lookup test against the precomputed indices.
        :meth:`_privatizable_buffers` additionally verifies define-before-read (dominators) and the
        library-node guard, but those run behind the mappability probe. Since every privatizable
        buffer is a written, loop-local, reindexable transient, this is a *necessary* precondition
        for a non-empty ``_privatizable_buffers`` -- a loop it rejects is never expanded, so gating
        the probe on it changes only which loops are probed, never which are expanded.
        """
        for name, write_states in write_index.items():
            if not write_states.isdisjoint(loop_states) \
                    and BufferExpansion._is_loop_local(loop_states, name, access_states, interstate_syms):
                return True
        return False

    @staticmethod
    def _access_state_index(sdfg: SDFG) -> Dict[str, Set[SDFGState]]:
        """Map every data name to the set of states holding an AccessNode for it (one pass).

        Built once per SDFG so ``_is_loop_local`` is a set lookup instead of an
        ``all_states x nodes`` rescan for every (loop, array) pair.
        """
        index: Dict[str, Set[SDFGState]] = {}
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode):
                    index.setdefault(n.data, set()).add(state)
        return index

    @staticmethod
    def _interstate_symbols(sdfg: SDFG) -> Set[str]:
        """Names appearing free on any interstate edge (a buffer used there is not loop-local)."""
        syms: Set[str] = set()
        for edge in sdfg.all_interstate_edges():
            syms |= {str(s) for s in edge.data.free_symbols}
        return syms

    @staticmethod
    def _loop_mappable(sdfg: SDFG, loop: LoopRegion) -> bool:
        """Whether ``LoopToMap`` would parallelize ``loop`` right now (refusals silenced)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap  # avoid an import cycle
        with contextlib.redirect_stdout(io.StringIO()):
            return LoopToMap.can_be_applied_to(sdfg, loop=loop)

    @staticmethod
    def _loop_index(loop: LoopRegion):
        """The ``(normalized_index, size)`` for a unit-step loop, or ``None`` if not analyzable.

        ``normalized_index`` is ``loop_var - start`` (so the new dimension is 0-based) and
        ``size`` is the iteration count; only unit-step loops with a known start/end qualify.
        """
        from dace.transformation.passes.analysis import loop_analysis  # avoid an import cycle

        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        step = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or step is None:
            return None
        if step != 1:  # only unit-step loops: the new dimension is indexed by ``loop_var - start``
            return None
        var = symbolic.pystr_to_symbolic(loop.loop_variable)
        # ``get_loop_end`` returns the INCLUSIVE last value (condition ``i < N`` -> ``end = N - 1``),
        # so the loop runs i in [start, end] and the private dimension -- indexed by the 0-based
        # ``i - start`` -- needs ``end - start + 1`` slots. (Dropping the +1 under-sizes the axis by
        # one and the final iteration's slice writes/reads out of bounds.)
        return var - start, end - start + 1

    # -- buffer detection ---------------------------------------------------------------

    def _privatizable_buffers(self,
                              sdfg: SDFG,
                              loop: LoopRegion,
                              access_states: Optional[Dict[str, Set[SDFGState]]] = None,
                              interstate_syms: Optional[Set[str]] = None,
                              loop_states: Optional[Set[SDFGState]] = None) -> List[str]:
        """Transient arrays in ``loop`` that are safe to privatize by expansion.

        A buffer qualifies when it is a transient array used only inside ``loop``, has no
        write-conflict (reduction) edge, and is *defined before read* on every iteration --
        i.e. never carries a value across iterations. ``access_states`` / ``interstate_syms``
        are the per-SDFG indices from :meth:`_access_state_index` / :meth:`_interstate_symbols`
        and ``loop_states`` is ``set(loop.all_states())``; each is built here when not supplied
        (so the helper is usable standalone).
        """
        if access_states is None:
            access_states = self._access_state_index(sdfg)
        if interstate_syms is None:
            interstate_syms = self._interstate_symbols(sdfg)
        if loop_states is None:
            loop_states = set(loop.all_states())
        candidates: List[str] = []
        for name, desc in sdfg.arrays.items():
            if not isinstance(desc, data.Array) or not desc.transient:
                continue
            # A View is a reshape/alias of another array (a ``data.View`` IS a
            # ``data.Array`` subclass, so the check above does not exclude it), often a
            # GEMM/MatMul operand slice. Expanding it with a loop dimension corrupts the
            # shape it presents to its consumer -- scattering_self_energies' 2D
            # ``Norb x Norb`` matmul-operand view widened to 6D, which the GEMM expansion
            # then rejects. Never expand a view; expand only real buffers.
            if isinstance(desc, data.View):
                continue
            # Do not expand a buffer that feeds (or is produced by) a library node. It
            # is a fixed-shape operand (e.g. a GEMM's 2D ``Norb x Norb`` slice), so
            # expanding it widens the operand shape the library node's expansion
            # requires (scattering_self_energies' GEMM -> "matrix-matrix product only
            # supported on matrices"). It is also pointless: parallelizing a loop whose
            # body is a library-node call gives no speedup -- a loop of (tiny) BLAS/
            # cuBLAS GEMMs serializes on one stream / oversubscribes BLAS threads -- so
            # the loop should stay sequential rather than become a Map of library calls.
            if self._is_library_node_operand(loop_states, name):
                continue
            if desc.lifetime not in _REINDEXABLE_LIFETIMES:
                continue
            if len(desc.shape) < 1:
                continue
            if self._is_loop_local(loop_states, name, access_states, interstate_syms) \
                    and self._defined_before_read(loop, name):
                candidates.append(name)
        return candidates

    @staticmethod
    def _is_library_node_operand(loop_states: Set[SDFGState], name: str) -> bool:
        """True if ``name`` is a direct input/output of a library node in ``loop_states``.

        Such a buffer is a shape-constrained operand (e.g. a GEMM's 2D slice); expanding
        it both breaks the library node's expansion and yields no parallelism (the
        library calls serialize), so it must not be expanded.
        """
        for state in loop_states:
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == name:
                    if any(isinstance(e.src, nodes.LibraryNode) for e in state.in_edges(n)):
                        return True
                    if any(isinstance(e.dst, nodes.LibraryNode) for e in state.out_edges(n)):
                        return True
        return False

    @staticmethod
    def _is_loop_local(loop_states: Set[SDFGState], name: str, access_states: Dict[str, Set[SDFGState]],
                       interstate_syms: Set[str]) -> bool:
        """True if ``name`` is accessed only within ``loop_states`` (not live in/out of the loop).

        Uses the precomputed per-SDFG indices: ``name`` is loop-local iff every state that
        accesses it is inside ``loop_states`` and it is not carried on any interstate edge.
        """
        if any(state not in loop_states for state in access_states.get(name, ())):
            return False
        # A buffer carried on an interstate edge (e.g. a lifted index symbol) is not loop-local.
        return name not in interstate_syms

    @staticmethod
    def _defined_before_read(loop: LoopRegion, name: str) -> bool:
        """True if every read of ``name`` in the loop body observes a value written in the
        *same* iteration -- i.e. ``name`` is never loop-carried, so a private copy per
        iteration preserves semantics.

        Sound (conservative). A read is accepted when a covering write to ``name`` either feeds
        the read's own access node (written then read in that state) or sits in an
        *unconditional* loop-body state that *dominates* the reading block (so it always runs
        earlier in the iteration). A write-conflict (WCR/reduction) edge, or a read that no such
        write covers, marks the buffer as carried (e.g. an accumulator, or ``q`` at level
        ``n-1``) and is refused. Conditional writes are not credited (they may not run), and a
        write/read split across two unconditional states is handled via dominance.
        """
        from dace.sdfg.analysis import cfg as cfg_analysis  # avoid an import cycle

        saw_read = saw_write = False
        # (reading_block, read_subset, node_local_write_subsets) triples needing a dominator.
        exposed_reads: List[Tuple[Any, Any, List[Any]]] = []
        write_blocks: Dict[Any, List[Any]] = {}  # unconditional state -> covering write subsets
        for state in loop.all_states():
            block = BufferExpansion._top_block(loop, state)
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode) or node.data != name:
                    continue
                in_edges, out_edges = state.in_edges(node), state.out_edges(node)
                if any(e.data is not None and e.data.wcr is not None for e in in_edges):
                    return False  # reduction edge -> genuine accumulation, not a scratch buffer
                in_subsets = [
                    s for s in (BufferExpansion._arr_subset(state, e, node) for e in in_edges) if s is not None
                ]
                if in_subsets:
                    saw_write = True
                    if isinstance(block, SDFGState):  # only unconditional writes are credited for dominance
                        write_blocks.setdefault(block, []).extend(in_subsets)
                if not out_edges:
                    continue
                saw_read = True
                for e in out_edges:
                    read = BufferExpansion._arr_subset(state, e, node)
                    if read is None:
                        continue
                    # Written then read at this node -> defined this iteration. A single covering
                    # write, or several node-local writes that together tile the read region
                    # (multi-edge fill), both qualify.
                    if BufferExpansion._union_covers(in_subsets, read):
                        continue
                    exposed_reads.append((block, read, in_subsets))
        if not (saw_read and saw_write):
            return False  # a reused buffer is both written and read inside the loop
        if not exposed_reads:
            return True

        doms = cfg_analysis.all_dominators(loop)
        for block, read, node_writes in exposed_reads:
            dominators = doms.get(block, set())
            # A read is defined-this-iteration when the writes that provably precede it -- the ones
            # feeding its own access node plus every unconditional write block that dominates it --
            # TOGETHER cover the read region. Crediting the union (not just one write) lets a buffer
            # filled across several statements (``buf[0:M] = ...; buf[M:2*M] = ...``) still qualify,
            # while a read no such writes cover is a genuine loop-carried value and is refused.
            covering = list(node_writes)
            for wb, subs in write_blocks.items():
                if wb is not block and wb in dominators:
                    covering.extend(subs)
            if not BufferExpansion._union_covers(covering, read):
                return False  # read reaches the loop entry without a covering write -> carried
        return True

    @staticmethod
    def _union_covers(writes: List[Any], read) -> bool:
        """True iff the ``writes`` TOGETHER cover ``read`` (sound; conservative when unsure).

        Accepts either a single write that already covers ``read``, or several unit-step writes
        that tile ``read`` as exactly-adjacent contiguous chunks along a single axis -- the classic
        multi-statement scratch fill ``buf[0:M] = ...; buf[M:2*M] = ...`` read back as
        ``buf[0:2*M]``. Only exact (gap-free, overlap-free) tilings are credited: every chunk
        boundary is matched by symbolic EQUALITY, never by an inequality that could silently
        over-approximate a hole and mistake a partially-written (carried) buffer for a full fill.
        """
        if not isinstance(read, subsets.Range):
            return False
        dims = read.dims()
        # ``covers_precise``, NOT ``covers``: the latter is a *bounding-box* test that ignores the
        # step, so a strided write ``buf[0:2*H:2]`` -- which touches only the even slots -- reports
        # covering the dense read ``buf[0:2*H]`` its box happens to span. Crediting that leaves the
        # skipped slots carrying an earlier iteration's value while the buffer is declared private,
        # and every iteration then reads its own never-written slice. ``covers_precise`` accounts
        # for the step and, like ``covers``, answers False when it cannot prove coverage.
        # Both return a (truthy) ValueError on a dimensionality mismatch instead of raising, so gate
        # the fast path on matching dims -- every subset here is for the same buffer, so this only
        # rejects malformed input, never a real cover.
        if any(w is not None and w.dims() == dims and w.covers_precise(read) for w in writes):
            return True
        chunks = [w for w in writes if isinstance(w, subsets.Range) and w.dims() == dims]
        if len(chunks) < 2:
            return False
        read_ranges = list(read.ranges)
        # Exactly one axis may vary between the chunks; on every other axis each chunk must match
        # the read's range, so stacking the chunks along the one axis reconstructs the read box.
        varying = None
        for ax in range(dims):
            if all(BufferExpansion._axis_eq(c.ranges[ax], read_ranges[ax]) for c in chunks):
                continue
            if varying is not None:
                return False  # chunks differ on more than one axis -> not a clean 1-axis tiling
            varying = ax
        if varying is None:
            return False
        rb, re_, rs = read_ranges[varying]
        if rs != 1:
            return False
        intervals = []
        for c in chunks:
            b, e, s = c.ranges[varying]
            if s != 1:
                return False
            intervals.append((b, e))
        # Greedy exact chain from the read's start: each step consumes a chunk whose start is
        # exactly the current frontier; the read is covered iff the frontier reaches its end + 1.
        frontier = rb
        progressed = True
        while progressed:
            progressed = False
            for i, (b, e) in enumerate(intervals):
                if BufferExpansion._sym_eq(b, frontier):
                    frontier = e + 1
                    intervals.pop(i)
                    progressed = True
                    break
        return BufferExpansion._sym_eq(frontier, re_ + 1)

    @staticmethod
    def _axis_eq(a, b) -> bool:
        """True if two ``(begin, end, step)`` axis ranges are provably identical."""
        return all(BufferExpansion._sym_eq(x, y) for x, y in zip(a, b))

    @staticmethod
    def _sym_eq(a, b) -> bool:
        """True only when ``a`` and ``b`` are provably equal (symbolic difference simplifies to 0)."""
        try:
            return bool(symbolic.simplify(a - b) == 0)
        except (TypeError, AttributeError):
            return False

    @staticmethod
    def _top_block(loop: LoopRegion, block):
        """The direct child block of ``loop`` containing ``block`` (walk up the CFG hierarchy)."""
        cur = block
        while cur.parent_graph is not None and cur.parent_graph is not loop:
            cur = cur.parent_graph
        return cur

    @staticmethod
    def _arr_subset(state, edge, node):
        """The subset of ``edge``'s memlet on ``node``'s side.

        Which of ``subset`` / ``other_subset`` names the source is carried by the memlet's own
        ``_is_data_src`` flag, NOT by the endpoint names. On a self-copy ``A -> A`` both endpoints
        match ``memlet.data``, so a name test hands the READ region back as the destination node's
        write -- a level shift ``A[k] = A[k-1]`` then looks like a write that covers its own read and
        the loop-carried dependence is expanded away.
        """
        memlet = edge.data
        if memlet is None:
            return None
        return memlet.get_src_subset(edge, state) if edge.src is node else memlet.get_dst_subset(edge, state)

    # -- expansion ----------------------------------------------------------------------

    def _expand(self, sdfg: SDFG, loop: LoopRegion, name: str, index, size, ambient: str) -> _Expansion:
        """Add the loop-indexed dimension to ``name``'s descriptor and every access in ``loop``."""
        desc = sdfg.arrays[name]
        old = (list(desc.shape), list(desc.strides), desc.total_size, list(desc.offset))

        order = self._array_order(desc, ambient)
        axis = 0 if order == _C_ORDER else len(desc.shape)
        new_shape = list(desc.shape)
        new_shape.insert(axis, size)
        new_offset = list(desc.offset)
        new_offset.insert(axis, 0)
        desc.set_shape(new_shape,
                       strides=self._contiguous_strides(new_shape, order),
                       total_size=_prod(new_shape),
                       offset=new_offset)

        edits: List[Tuple[Any, str, Any]] = []
        seen: Set[int] = set()
        for state in loop.all_states():
            for edge in state.edges():
                if id(edge) in seen or edge.data is None:
                    continue
                memlet = edge.data
                # ``subset`` addresses ``memlet.data``, ``other_subset`` the opposite endpoint. On a
                # self-copy ``A -> A`` BOTH sides address ``name``, so both must gain the new index;
                # an if/elif keyed on the data name rewrites one and leaves the other at the old rank.
                self_copy = (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)
                             and edge.src.data == name and edge.dst.data == name)
                rewrote = False
                if memlet.data == name and memlet.subset is not None:
                    edits.append((memlet, 'subset', memlet.subset))
                    memlet.subset = _insert_index(memlet.subset, index, axis)
                    rewrote = True
                if memlet.other_subset is not None and self._touches(edge, name) and (self_copy or not rewrote):
                    edits.append((memlet, 'other_subset', memlet.other_subset))
                    memlet.other_subset = _insert_index(memlet.other_subset, index, axis)
                    rewrote = True
                if rewrote:
                    seen.add(id(edge))
        return _Expansion(desc, old[0], old[1], old[2], old[3], edits)

    @staticmethod
    def _touches(edge, name: str) -> bool:
        """True if either endpoint of ``edge`` is an AccessNode for ``name``."""
        return ((isinstance(edge.src, nodes.AccessNode) and edge.src.data == name)
                or (isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name))

    @staticmethod
    def _contiguous_strides(shape, order: str):
        """Packed strides for ``shape`` in C (row-major) or Fortran (column-major) order."""
        if order == _C_ORDER:
            return [_prod(shape[i + 1:]) for i in range(len(shape))]
        return [_prod(shape[:i]) for i in range(len(shape))]

    # -- layout ------------------------------------------------------------------------

    @staticmethod
    def _ambient_order(sdfg: SDFG) -> str:
        """Majority storage order of the SDFG's multi-dimensional arrays (ties -> C)."""
        c = f = 0
        for desc in sdfg.arrays.values():
            if isinstance(desc, data.Array) and len(desc.shape) >= 2:
                if desc.is_packed_fortran_strides():
                    f += 1
                elif desc.is_packed_c_strides():
                    c += 1
        return _F_ORDER if f > c else _C_ORDER

    @staticmethod
    def _array_order(desc: data.Array, ambient: str) -> str:
        """``desc``'s own storage order when unambiguous, else the SDFG's ambient order.

        A multi-dimensional array reveals its order through its strides; a 1-D array (single
        stride ``1`` in both orders) does not, so it follows ``ambient`` (the majority vote).
        """
        if len(desc.shape) >= 2:
            is_c, is_f = desc.is_packed_c_strides(), desc.is_packed_fortran_strides()
            if is_f and not is_c:
                return _F_ORDER
            if is_c and not is_f:
                return _C_ORDER
        return ambient
