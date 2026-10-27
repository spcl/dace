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
*refuses* into one it *accepts* (asked before the rewrite, with the buffers assumed
iteration-private): a safe expansion that would not help is left alone so the SDFG does not
grow needlessly. A static safety guard
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
from typing import Any

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.sdfg import memoize_facts
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.scalar_fission import (
    AccessIndex,
    MustDefinition,
    SubsetAccesses,
    has_wcr_write,
)

#: Storage-order tag for the new dimension's placement.
_C_ORDER = "C"
_F_ORDER = "F"

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


def iteration_private_buffers(sdfg: SDFG, loop: LoopRegion, only: set[str]) -> tuple[list[str], Any] | None:
    """The transients of ``only`` one iteration of ``loop`` defines before reading it, which
    :func:`expand_iteration_private` would give one slot each, and the index of that slot; ``None`` when none
    qualifies. A transient scalar is judged as a length-1 array. Reads the graph only.

    The privatization :class:`BufferExpansion` applies for ``LoopToMap``, offered to loop distribution: a
    per-iteration accumulator ``s`` shared by a producer and a consumer keeps them in one loop, and ``s[i]``
    lets the producer run to completion first. The safety guard is the pass's own (loop-local, defined before
    read, no view, no library-node operand).

    :param only: The containers that may be expanded, those that couple two body blocks; a temporary used by one
                 block stays private where it is, so the loops inside that block keep privatizing it.
    """
    expander = BufferExpansion()
    ix = expander._loop_index(loop)
    if ix is None:
        return None
    scalars = {
        name
        for name in only
        if isinstance(sdfg.arrays.get(name), data.Scalar)
        and sdfg.arrays[name].transient
        and sdfg.arrays[name].lifetime in _REINDEXABLE_LIFETIMES
    }
    # A nested SDFG's connector must see a descriptor equal to its own: ``x[i]`` of an expanded ``x`` would need a view
    nested_operands = {
        e.data.data
        for state in loop.states()
        for node in state.nodes()
        if isinstance(node, nodes.NestedSDFG)
        for e in state.all_edges(node)
        if e.data is not None and e.data.data is not None
    }
    buffers = expander._privatizable_buffers(sdfg, loop, scalars=scalars, names=only - nested_operands)
    return (buffers, ix[0]) if buffers else None


def rescues(sdfg: SDFG, loop: LoopRegion, refused_on: str | None) -> bool:
    """Whether :class:`BufferExpansion` may expand ``refused_on``, the container ``LoopToMap`` refused ``loop`` on: a
    privatizable buffer of ``loop``. Reads the graph only."""
    desc = sdfg.arrays.get(refused_on) if refused_on is not None else None
    if (
        not isinstance(desc, data.Array)
        or isinstance(desc, data.View)
        or not desc.transient
        or desc.lifetime not in _REINDEXABLE_LIFETIMES
    ):
        return False
    expander = BufferExpansion()
    if expander._loop_index(loop) is None:
        return False
    return bool(expander._privatizable_buffers(sdfg, loop, names={refused_on}))


def expand_iteration_private(sdfg: SDFG, loop: LoopRegion, names: list[str]) -> None:
    """Give each of ``names``, which :func:`iteration_private_buffers` admitted, one slot per iteration of ``loop``:
    ``x`` -> ``x[i]``."""
    expander = BufferExpansion()
    index, size = expander._loop_index(loop)
    for name in names:
        desc = sdfg.arrays[name]
        if isinstance(desc, data.Scalar):
            sdfg.arrays[name] = data.Array(
                desc.dtype, [1], transient=True, storage=desc.storage, lifetime=desc.lifetime
            )
    order = expander._ambient_order(sdfg)
    touching = expander._edges_by_name(loop)
    for name in names:
        expander._expand(sdfg, loop, name, index, size, order, touching.get(name, ()))


@properties.make_properties
class BufferExpansion(ppl.Pass):
    """Expand loop-local scratch buffers by one loop-indexed dimension to unblock LoopToMap.

    See the module docstring. The pass scans every loop the SDFG (and its nested SDFGs)
    contains; for a loop that ``LoopToMap`` currently refuses, it expands the safe
    privatizable buffers accessed in that loop when ``LoopToMap`` accepts the loop with
    those buffers assumed iteration-private.
    """

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Re-running on a structurally changed SDFG can expose freshly-isolated buffers.
        return bool(modified & (ppl.Modifies.CFG | ppl.Modifies.Memlets | ppl.Modifies.Descriptors))

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> dict[str, list[str]] | None:
        """Expand the buffers that unblock a loop, in ``sdfg`` and every nested SDFG.

        :param sdfg: The SDFG to transform in place.
        :returns: ``{loop_label: [expanded_array, ...]}`` for the loops that became
                  parallelizable, or ``None`` if nothing was expanded.
        """
        expanded: dict[str, list[str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            expanded.update(self._expand_sdfg(sd))
        return expanded or None

    def report(self, pass_retval: Any) -> str | None:
        if not pass_retval:
            return None
        arrays = sum(len(v) for v in pass_retval.values())
        return f"BufferExpansion: expanded {arrays} buffer(s) to unblock {len(pass_retval)} loop(s)"

    # core

    def _expand_sdfg(self, sdfg: SDFG) -> dict[str, list[str]]:
        """Expand beneficial buffers for the loops of a single SDFG (analyze, decide, then expand).

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
        accesses = AccessIndex(sdfg)
        write_index = self._write_state_index(sdfg, accesses)
        order: str | None = None  # ambient storage order: built lazily, only when actually expanding

        kept: dict[str, list[str]] = {}
        # avoid an import cycle
        from dace.transformation.passes.parallelization.parallelize_loops import ProbeSession

        # Every probe asks the graph as it stands, so they share one session until an expansion changes it.
        session = ProbeSession()
        for loop in loops:
            ix = self._loop_index(loop)
            if ix is None:
                continue
            loop_states = set(loop.states())
            if not self._has_candidate_buffer(loop_states, write_index, accesses):
                continue  # no privatizable-buffer candidate -> never expanded, so skip the probe
            accepted, refused_on = self._loop_verdict(sdfg, loop, session)
            if accepted:
                continue  # already parallelizable -> expanding would only grow the SDFG
            buffers = self._privatizable_buffers(sdfg, loop, accesses, loop_states)
            # Only a refusal on a buffer can be lifted by expanding it; any other survives the expansion unchanged.
            if refused_on not in buffers:
                continue
            if not self._loop_verdict(sdfg, loop, session, frozenset(buffers))[0]:
                continue
            if order is None:
                order = self._ambient_order(sdfg)
            index, size = ix
            touching = self._edges_by_name(loop)
            for arr in buffers:
                self._expand(sdfg, loop, arr, index, size, order, touching.get(arr, ()))
            session.forget(sdfg)
            kept[loop.label] = list(buffers)
        return kept

    @staticmethod
    def _write_state_index(sdfg: SDFG, index: AccessIndex) -> dict[str, set[SDFGState]]:
        """Map each *privatizable-kind* transient array to the states that WRITE it (one pass).

        A key is a transient, non-view, reindexable-lifetime Array of rank >= 1 -- exactly the
        descriptor-level filters :meth:`_privatizable_buffers` applies -- with at least one
        AccessNode fed by an incoming edge (a write). Read-only and never-accessed transients are
        omitted so the candidate gate iterates a small key set. The plain "has an incoming edge"
        test is a broad, sound stand-in for ``defined_before_read``'s per-edge write check (it can
        only over-include), keeping the gate a necessary precondition for privatizability.
        """
        reindexable = {
            name
            for name, desc in sdfg.arrays.items()
            if isinstance(desc, data.Array)
            and not isinstance(desc, data.View)
            and desc.transient
            and desc.lifetime in _REINDEXABLE_LIFETIMES
            and len(desc.shape) >= 1
        }
        written: dict[str, set[SDFGState]] = {}
        for name in reindexable:
            states = {st for st in index.states(name) if any(st.in_degree(n) for n in index.nodes(st, name))}
            if states:
                written[name] = states
        return written

    @staticmethod
    def _has_candidate_buffer(
        loop_states: set[SDFGState], write_index: dict[str, set[SDFGState]], index: AccessIndex
    ) -> bool:
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
            if not write_states.isdisjoint(loop_states) and index.confined_to(name, loop_states):
                return True
        return False

    @staticmethod
    def _loop_mappable(sdfg: SDFG, loop: LoopRegion, session: Any = None) -> bool:
        """Whether ``LoopToMap`` would parallelize ``loop`` right now (refusals silenced)."""
        return BufferExpansion._loop_verdict(sdfg, loop, session)[0]

    @staticmethod
    def _loop_verdict(
        sdfg: SDFG,
        loop: LoopRegion,
        session: Any = None,
        assume_private: frozenset[str] = frozenset(),
    ) -> tuple[bool, str | None]:
        """Whether ``LoopToMap`` would parallelize ``loop`` with ``assume_private`` made iteration-private, and on a
        refusal the container it refused on (``None`` for any other refusal).

        :param session: the :class:`ProbeSession` the caller's probes share, ``None`` to probe without one.
        """
        # avoid an import cycle
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        with contextlib.redirect_stdout(io.StringIO()):
            xform = LoopToMap()
            accepted = ParallelizeLoops().accepts(loop, xform, assume_private, session=session)
        return accepted, None if accepted or xform.last_refusal is None else xform.last_refusal.container

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

    # buffer detection

    def _privatizable_buffers(
        self,
        sdfg: SDFG,
        loop: LoopRegion,
        index: AccessIndex | None = None,
        loop_states: set[SDFGState] | None = None,
        scalars: set[str] = frozenset(),
        names: set[str] | None = None,
    ) -> list[str]:
        """Transient arrays in ``loop`` that are safe to privatize by expansion; ``scalars`` are transient scalars
        judged as length-1 arrays, ``names``, when given, the only containers judged.

        A buffer qualifies when it is a transient array used only inside ``loop``, has no
        write-conflict (reduction) edge, and is *defined before read* on every iteration --
        i.e. never carries a value across iterations. ``index`` is the SDFG's access index and
        ``loop_states`` is ``set(loop.states())``; each is built here when not supplied.
        """
        index = index or AccessIndex(sdfg)
        if loop_states is None:
            loop_states = set(loop.states())
        candidates: list[str] = []
        library_operands: set[str] | None = None
        # Nothing changes while the buffers are judged, so every cover test shares the facts of the SDFG.
        with memoize_facts():
            for name, desc in sdfg.arrays.items():
                if names is not None and name not in names:
                    continue
                if not (isinstance(desc, data.Array) or name in scalars) or not desc.transient:
                    continue
                # A View is a reshape/alias of another array (a ``data.View`` IS a
                # ``data.Array`` subclass, so the check above does not exclude it), often a
                # GEMM/MatMul operand slice. Expanding it with a loop dimension corrupts the
                # shape it presents to its consumer -- scattering_self_energies' 2D
                # ``Norb x Norb`` matmul-operand view widened to 6D, which the GEMM expansion
                # then rejects. Never expand a view; expand only real buffers.
                if isinstance(desc, data.View):
                    continue
                # Descriptor tests and index lookups before the walks over the loop's states.
                if desc.lifetime not in _REINDEXABLE_LIFETIMES:
                    continue
                if len(desc.shape) < 1:
                    continue
                if not index.confined_to(name, loop_states):
                    continue
                # Do not expand a buffer that feeds (or is produced by) a library node. It
                # is a fixed-shape operand (e.g. a GEMM's 2D ``Norb x Norb`` slice), so
                # expanding it widens the operand shape the library node's expansion
                # requires (scattering_self_energies' GEMM -> "matrix-matrix product only
                # supported on matrices"). It is also pointless: parallelizing a loop whose
                # body is a library-node call gives no speedup -- a loop of (tiny) BLAS/
                # cuBLAS GEMMs serializes on one stream / oversubscribes BLAS threads -- so
                # the loop should stay sequential rather than become a Map of library calls.
                if library_operands is None:
                    library_operands = self._library_node_operands(loop_states)
                if name in library_operands:
                    continue
                if self.defined_before_read(loop, name, index, loop_states):
                    candidates.append(name)
        return candidates

    @staticmethod
    def _library_node_operands(loop_states: set[SDFGState]) -> set[str]:
        """The containers that are a direct input/output of a library node in ``loop_states``.

        Such a buffer is a shape-constrained operand (e.g. a GEMM's 2D slice); expanding
        it both breaks the library node's expansion and yields no parallelism (the
        library calls serialize), so it must not be expanded.
        """
        operands: set[str] = set()
        for state in loop_states:
            for n in state.data_nodes():
                if any(isinstance(e.src, nodes.LibraryNode) for e in state.in_edges(n)) or any(
                    isinstance(e.dst, nodes.LibraryNode) for e in state.out_edges(n)
                ):
                    operands.add(n.data)
        return operands

    @staticmethod
    def defined_before_read(loop: LoopRegion, name: str, index: AccessIndex | None = None, loop_states=None) -> bool:
        """True if every read of ``name`` in the loop body observes a value written in the *same*
        iteration, so a private copy per iteration preserves semantics.

        Sound (conservative): a write-conflict (WCR/reduction) write marks an accumulator and is refused,
        a buffer must be both written and read in the loop, and every read must be covered by the writes
        that precede it on every path of one iteration (the shared
        :class:`~dace.transformation.passes.scalar_fission.MustDefinition` over memlet subsets, where
        several writes together may tile a read: ``buf[0:M] = ...; buf[M:2*M] = ...``).

        :param index: The SDFG's access index; ``None`` builds one for this call.
        :param loop_states: ``loop.states()``, when the caller already has it.
        """
        index = index or AccessIndex(loop.sdfg)
        accesses = SubsetAccesses(name, loop.sdfg.facts())
        saw_read = saw_write = False
        for state in loop.states() if loop_states is None else loop_states:
            for node in index.nodes(state, name):
                if has_wcr_write(state, node):
                    return False  # reduction edge -> genuine accumulation, not a scratch buffer
                saw_write = saw_write or bool(accesses.writes(state, node))
                saw_read = saw_read or state.out_degree(node) > 0
        if not (saw_read and saw_write):
            return False  # a reused buffer is both written and read inside the loop
        return not MustDefinition(accesses, index).exposed(loop)

    # expansion

    @staticmethod
    def _edges_by_name(loop: LoopRegion) -> dict[str, list[Any]]:
        """The edges of ``loop``'s states, in state and edge order, under every name :meth:`_expand` can rewrite
        them for: their memlet's container and the container of each access-node endpoint."""
        touching: dict[str, list[Any]] = {}
        for state in loop.states():
            for edge in state.edges():
                if edge.data is None:
                    continue
                names = {edge.data.data}
                for node in (edge.src, edge.dst):
                    if isinstance(node, nodes.AccessNode):
                        names.add(node.data)
                for name in names:
                    touching.setdefault(name, []).append(edge)
        return touching

    def _expand(self, sdfg: SDFG, loop: LoopRegion, name: str, index, size, ambient: str, touching: Any = None) -> None:
        """Add the loop-indexed dimension to ``name``'s descriptor and every access in ``loop``.

        :param touching: ``name``'s entry of :meth:`_edges_by_name`, built before the first of a batch of
                         expansions; an expansion rewrites subsets only, so it stays exact for the batch. ``None``
                         walks every edge of ``loop``.
        """
        desc = sdfg.arrays[name]
        order = self._array_order(desc, ambient)
        axis = 0 if order == _C_ORDER else len(desc.shape)
        new_shape = list(desc.shape)
        new_shape.insert(axis, size)
        new_offset = list(desc.offset)
        new_offset.insert(axis, 0)
        desc.set_shape(
            new_shape,
            strides=self._contiguous_strides(new_shape, order),
            total_size=_prod(new_shape),
            offset=new_offset,
        )

        seen: set[int] = set()
        if touching is None:
            touching = [edge for state in loop.states() for edge in state.edges()]
        for edge in touching:
            if id(edge) in seen or edge.data is None:
                continue
            memlet = edge.data
            # ``subset`` addresses ``memlet.data``, ``other_subset`` the opposite endpoint. On a
            # self-copy ``A -> A`` BOTH sides address ``name``, so both must gain the new index;
            # an if/elif keyed on the data name rewrites one and leaves the other at the old rank.
            self_copy = (
                isinstance(edge.src, nodes.AccessNode)
                and isinstance(edge.dst, nodes.AccessNode)
                and edge.src.data == name
                and edge.dst.data == name
            )
            rewrote = False
            if memlet.data == name and memlet.subset is not None:
                memlet.subset = _insert_index(memlet.subset, index, axis)
                rewrote = True
            if memlet.other_subset is not None and self._touches(edge, name) and (self_copy or not rewrote):
                memlet.other_subset = _insert_index(memlet.other_subset, index, axis)
                rewrote = True
            if rewrote:
                seen.add(id(edge))

    @staticmethod
    def _touches(edge, name: str) -> bool:
        """True if either endpoint of ``edge`` is an AccessNode for ``name``."""
        return (isinstance(edge.src, nodes.AccessNode) and edge.src.data == name) or (
            isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name
        )

    @staticmethod
    def _contiguous_strides(shape, order: str):
        """Packed strides for ``shape`` in C (row-major) or Fortran (column-major) order."""
        if order == _C_ORDER:
            return [_prod(shape[i + 1 :]) for i in range(len(shape))]
        return [_prod(shape[:i]) for i in range(len(shape))]

    # layout

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
