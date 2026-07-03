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
        """Expand beneficial buffers for the loops of a single SDFG (speculate, then verify)."""
        loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

        # Cheap structural pre-filter first: collect the analyzable loops that actually have a
        # privatizable buffer. Only if some do is the (expensive) LoopToMap match worth running.
        candidates: Dict[LoopRegion, Tuple[Any, List[str]]] = {}
        for loop in loops:
            ix = self._loop_index(loop)
            if ix is None:
                continue
            buffers = self._privatizable_buffers(sdfg, loop)
            if buffers:
                candidates[loop] = (ix, buffers)
        if not candidates:
            return {}

        before = self._mappable_labels(sdfg)
        order = self._ambient_order(sdfg)

        # Speculatively expand the safe buffers of every currently-unmappable loop. Buffers
        # are loop-local (verified below), so different loops never share a buffer and their
        # expansions do not interfere -- one LoopToMap re-match then decides each loop.
        plan: Dict[str, Tuple[LoopRegion, List[_Expansion], List[str]]] = {}
        for loop, ((index, size), buffers) in candidates.items():
            if loop.label in before:
                continue
            applied = [self._expand(sdfg, loop, arr, index, size, order) for arr in buffers]
            plan[loop.label] = (loop, applied, list(buffers))

        if not plan:
            return {}
        after = self._mappable_labels(sdfg)

        kept: Dict[str, List[str]] = {}
        for label, (loop, applied, buffers) in plan.items():
            if label in after:
                kept[label] = buffers
            else:
                for exp in applied:
                    exp.undo()
        return kept

    @staticmethod
    def _mappable_labels(sdfg: SDFG) -> Set[str]:
        """Labels of the loops ``LoopToMap`` would parallelize right now (refusals silenced)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap  # avoid an import cycle
        from dace.transformation.passes.pattern_matching import match_patterns

        with contextlib.redirect_stdout(io.StringIO()):
            return {m.loop.label for m in match_patterns(sdfg, LoopToMap)}

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
        # ``end`` is the exclusive bound (condition ``i < end``); the loop runs i in [start, end).
        return var - start, end - start

    # -- buffer detection ---------------------------------------------------------------

    def _privatizable_buffers(self, sdfg: SDFG, loop: LoopRegion) -> List[str]:
        """Transient arrays in ``loop`` that are safe to privatize by expansion.

        A buffer qualifies when it is a transient array used only inside ``loop``, has no
        write-conflict (reduction) edge, and is *defined before read* on every iteration --
        i.e. never carries a value across iterations.
        """
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
            if self._is_loop_local(sdfg, loop_states, name) and self._defined_before_read(loop, name):
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
    def _is_loop_local(sdfg: SDFG, loop_states: Set[SDFGState], name: str) -> bool:
        """True if ``name`` is accessed only within ``loop_states`` (not live in/out of the loop)."""
        for state in sdfg.all_states():
            if state in loop_states:
                continue
            if any(isinstance(n, nodes.AccessNode) and n.data == name for n in state.nodes()):
                return False
        # A buffer carried on an interstate edge (e.g. a lifted index symbol) is not loop-local.
        for edge in sdfg.all_interstate_edges():
            if name in edge.data.free_symbols:
                return False
        return True

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
        exposed_reads: List[Tuple[Any, Any]] = []  # (reading_block, read_subset) needing a dominator
        write_blocks: Dict[Any, List[Any]] = {}  # unconditional state -> covering write subsets
        for state in loop.all_states():
            block = BufferExpansion._top_block(loop, state)
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode) or node.data != name:
                    continue
                in_edges, out_edges = state.in_edges(node), state.out_edges(node)
                if any(e.data is not None and e.data.wcr is not None for e in in_edges):
                    return False  # reduction edge -> genuine accumulation, not a scratch buffer
                in_subsets = [s for s in (BufferExpansion._arr_subset(e, name) for e in in_edges) if s is not None]
                if in_subsets:
                    saw_write = True
                    if isinstance(block, SDFGState):  # only unconditional writes are credited for dominance
                        write_blocks.setdefault(block, []).extend(in_subsets)
                if not out_edges:
                    continue
                saw_read = True
                for e in out_edges:
                    read = BufferExpansion._arr_subset(e, name)
                    if read is None:
                        continue
                    if any(w.covers(read) for w in in_subsets):
                        continue  # written then read at this node -> defined this iteration
                    exposed_reads.append((block, read))
        if not (saw_read and saw_write):
            return False  # a reused buffer is both written and read inside the loop
        if not exposed_reads:
            return True

        doms = cfg_analysis.all_dominators(loop)
        for block, read in exposed_reads:
            dominators = doms.get(block, set())
            if not any(wb is not block and wb in dominators and any(w.covers(read) for w in subs)
                       for wb, subs in write_blocks.items()):
                return False  # read reaches the loop entry without a covering write -> carried
        return True

    @staticmethod
    def _top_block(loop: LoopRegion, block):
        """The direct child block of ``loop`` containing ``block`` (walk up the CFG hierarchy)."""
        cur = block
        while cur.parent_graph is not None and cur.parent_graph is not loop:
            cur = cur.parent_graph
        return cur

    @staticmethod
    def _arr_subset(edge, name: str):
        """The subset of ``edge``'s memlet that refers to data ``name`` (subset or other_subset)."""
        memlet = edge.data
        if memlet is None:
            return None
        if memlet.data == name:
            return memlet.subset
        return memlet.other_subset

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
                if memlet.data == name and memlet.subset is not None:
                    edits.append((memlet, 'subset', memlet.subset))
                    memlet.subset = _insert_index(memlet.subset, index, axis)
                    seen.add(id(edge))
                elif memlet.other_subset is not None and self._touches(edge, name):
                    edits.append((memlet, 'other_subset', memlet.other_subset))
                    memlet.other_subset = _insert_index(memlet.other_subset, index, axis)
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
