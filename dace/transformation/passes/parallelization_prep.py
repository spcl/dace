# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Standalone parallelization-preparation passes.

These rewrite loops so that ``LoopToMap`` can parallelize more of them. They are
plain :class:`~dace.transformation.pass_pipeline.Pass` objects so the
``parallelize`` pipeline (and anyone else) can just compose them:

- :class:`ShortLoopUnroll` -- fully unroll constant-trip loops with at most
  ``unroll_limit`` iterations, turning small recurrence / reduction loops into
  inline straight-line code instead of atomically-parallelized maps.
- :class:`BestEffortLoopPeeling` -- index-set-split a loop at a point DERIVED from its body
  (an ``if i == x`` guard, a broadcast read's conflicting index), and peel a wrapping body
  modulo to its floor-correct affine form, pruning the now-dead boundary guard from each
  segment. ``peel_limit`` bounds the modulo peel and underwrites the large-trip-count
  assumption the split's range proofs lean on (0 disables the pass).

Transformation classes are imported lazily inside the methods: importing them at
module load would cycle (this package is imported by the transformations those
imports pull in).
"""
import copy
from typing import Any, Dict, Optional

from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Default trip-count threshold below which a constant-trip loop is unrolled.
DEFAULT_UNROLL_LIMIT = 8
#: Default maximum number of iterations peeled (per side) when searching for a
#: peel that unblocks parallelization.
DEFAULT_PEEL_LIMIT = 8
#: Names under which a (floor) modulo may be defined in a subset expression. The
#: peel modulo-rewrite recognises all of these -- ``sympy.Mod`` (the ``%`` operator)
#: and the equivalent helper-function spellings -- so it folds a wrap-around access
#: regardless of which representation introduced it.
_MODULO_FUNC_NAMES = frozenset({'Mod', 'py_mod', 'Modulo', 'mod', 'floor_mod'})


def _loops(sdfg: SDFG):
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _unique_block_label(sdfg: SDFG, base: str) -> str:
    """A control-flow-block label not currently used anywhere in ``sdfg``."""
    import itertools
    existing = {b.label for b in sdfg.all_control_flow_blocks()}
    for n in itertools.count():
        cand = f'{base}_p{n}'
        if cand not in existing:
            return cand


def _constant_trip_count(loop: LoopRegion, sdfg: SDFG) -> Optional[int]:
    """The exact iteration count of ``loop`` if it is constant, else ``None``
    (matches ``len(range(0, end - start + 1, stride))``, i.e. LoopUnroll's count)."""
    from dace.transformation.passes.analysis import loop_analysis
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or not loop.loop_variable:
        return None
    if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
        return None
    try:
        stride_val = int(symbolic.evaluate(stride, sdfg.constants))
        diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
    except (TypeError, ValueError):
        return None
    if stride_val <= 0 or diff <= 0:
        return None
    return len(range(0, diff, stride_val))


@properties.make_properties
class ShortLoopUnroll(ppl.Pass):
    """Fully unroll every constant-trip loop with at most ``unroll_limit`` iterations."""

    CATEGORY: str = 'Optimization Preparation'

    unroll_limit = properties.Property(
        dtype=int,
        default=DEFAULT_UNROLL_LIMIT,
        desc='Fully unroll constant-trip loops with at most this many iterations (0 disables).')

    def __init__(self, unroll_limit: int = DEFAULT_UNROLL_LIMIT):
        self.unroll_limit = unroll_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Unroll short constant-trip loops; returns the number unrolled or None.

        Re-collects after each unroll since unrolling rewrites the control-flow
        structure (and may expose newly-constant inner loops).
        """
        if self.unroll_limit <= 0:
            return None
        from dace.transformation.interstate.loop_unroll import LoopUnroll
        unrolled = 0
        changed = True
        while changed:
            changed = False
            for loop in _loops(sdfg):
                trip = _constant_trip_count(loop, sdfg)
                if trip is None or trip > self.unroll_limit:
                    continue
                try:
                    # ``annotate=False``: skip the per-apply full-SDFG memlet/state
                    # propagation. The transformation framework otherwise re-runs it
                    # after EVERY unroll -- O(unrolls x sdfg_size) redundant work
                    # (the dominant ~83% cost of unrolling a d-deep trip-t tile
                    # nest, whose SDFG grows to ~t^d blocks). The trip-count / loop
                    # re-collection below reads only loop bounds, not memlets, so
                    # the interim annotations are never observed; one propagation
                    # after the whole fixpoint (below) refreshes them.
                    LoopUnroll().apply_to(sdfg=loop.sdfg, loop=loop, annotate=False)
                except Exception:
                    continue  # not unrollable in this context; leave it for LoopToMap
                unrolled += 1
                changed = True
                break
        if unrolled:
            # Propagate once, at the end of the pass (not per-apply).
            from dace.sdfg.propagation import propagate_memlets_sdfg
            propagate_memlets_sdfg(sdfg)
        return unrolled or None


@properties.make_properties
class BestEffortLoopPeeling(ppl.Pass):
    """Best-effort index-set splitting and peeling that unblocks parallelization.

    Split points are DERIVED from the body -- an equality guard's value, a broadcast read's
    conflicting index -- rather than enumerated blindly, so the body nominates the few cuts
    worth trying. Those candidates are still probed (each on a ``copy.deepcopy``, revertible
    by construction) and only the one unblocking the most maps reaches the real SDFG. A loop
    the body nominates nothing for is left alone: sequential and correct.

    Peeling is what a split at a boundary point degenerates to (``_split_loop_at`` drops the
    empty side), so there is no separate boundary-peel stage. The wrapping-modulo peel does
    search a bounded ``(count, direction)`` grid, because it is a CORRECTNESS fix rather than
    a parallelism one -- a wrap-around read maps fine but then emits C's truncated ``%`` and
    computes the wrong boundary value -- and so deliberately runs even when ``LoopToMap``
    already accepts the loop.

    This pass is a preparation pass: it only COUNTS what ``LoopToMap`` *could* parallelize via
    ``can_be_applied_to``, never applying it. The actual parallelization is the pipeline's job.
    """

    CATEGORY: str = 'Optimization Preparation'

    peel_limit = properties.Property(
        dtype=int,
        default=DEFAULT_PEEL_LIMIT,
        desc='Bounds the wrapping-modulo peel to 1..peel_limit iterations (front/back/both), keeping '
        'the smallest peel that folds the wrap away. Also underwrites the split range proofs: a loop '
        'worth peeling by k <= peel_limit is assumed to run more than peel_limit times (0 disables).')

    def __init__(self, peel_limit: int = DEFAULT_PEEL_LIMIT):
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _peel_one_loop(self, sdfg: SDFG, loop: LoopRegion, count: int, direction: str) -> bool:
        """Peel ``count`` iterations off ``loop`` (``'front'`` / ``'back'`` /
        ``'both'``). Returns whether it peeled. Does NOT prune -- callers prune.

        Uses ``verify=False`` so loops with symbolic bounds (where peeling a
        boundary is most useful) are not rejected by ``LoopUnroll``'s
        constant-size gate; infeasible loops simply raise and are skipped.
        """
        from dace.transformation.interstate.loop_peeling import LoopPeeling
        sides = {'front': [True], 'back': [False], 'both': [True, False]}[direction]
        # A loop short enough to be fully consumed by the peel is the unroll
        # pass's job, not peeling's.
        trip = _constant_trip_count(loop, sdfg)
        if trip is not None and trip <= count * len(sides):
            return False
        did = False
        for idx, begin in enumerate(sides):
            try:
                if idx > 0:
                    # LoopPeeling names the peeled iteration regions after the loop
                    # label; a second peel on the same loop would reuse those names.
                    # Relabel the remainder loop first so front/back peels differ.
                    loop.label = _unique_block_label(loop.sdfg, loop.label)
                # Properties must go through ``options=`` -- bare kwargs are not
                # applied, leaving ``count`` at LoopUnroll's default 0 (a no-op).
                LoopPeeling().apply_to(sdfg=loop.sdfg,
                                       loop=loop,
                                       verify=False,
                                       options={
                                           'count': count,
                                           'begin': begin
                                       })
                did = True
            except Exception:
                continue
        return did

    def _peel_loops(self, sdfg: SDFG, count: int, direction: str) -> int:
        """Peel every peelable loop in ``sdfg`` (used directly in tests). Prunes
        the dead boundary guards a peel leaves behind."""
        peeled = sum(int(self._peel_one_loop(sdfg, loop, count, direction)) for loop in _loops(sdfg))
        if peeled:
            self._clean_peeled_remainder(sdfg)
        return peeled

    def _isolate_loop(self, loop: LoopRegion, sdfg: SDFG):
        """Build a throwaway mini-SDFG containing only a copy of ``loop`` and the
        arrays / symbols it references, so the peel search can experiment cheaply
        without deep-copying the whole SDFG. Returns ``(mini, mini_loop)`` or
        ``(None, None)`` if the nest cannot be isolated cleanly."""
        import dace
        from dace import serialize
        try:
            mini = dace.SDFG(sdfg.name + '_peelprobe')
            for sname, stype in sdfg.symbols.items():
                mini.add_symbol(sname, stype)
            # Arrays referenced by the loop body and its interstate edges.
            needed = set()
            for st in loop.all_states():
                for n in st.data_nodes():
                    needed.add(n.data)
                for e in st.edges():
                    if e.data is not None and e.data.data is not None:
                        needed.add(e.data.data)
            for e in loop.all_interstate_edges():
                needed |= e.data.used_arrays(sdfg.arrays)
            for name in needed:
                if name in sdfg.arrays and name not in mini.arrays:
                    mini.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]))
            # Copy the loop region via a JSON round-trip (reparents to `mini`).
            # ``context['version']`` is required for symbolic deserialization (the
            # symbolic-property deserializer refuses without it); a loop with a
            # symbolic subset such as ``a[N // 2]`` (``int_floor``) would otherwise
            # fail the round-trip and the nest be wrongly treated as un-isolable.
            mini_loop = serialize.from_json(serialize.to_json(loop),
                                            context={
                                                'sdfg': mini,
                                                'version': dace.__version__
                                            })
            mini.add_node(mini_loop, is_start_block=True)
            mini.reset_cfg_list()
            mini.validate()
            return mini, mini_loop
        except Exception:
            return None, None

    def _equality_guard_values(self, loop: LoopRegion):
        """Loop-invariant values ``x`` for which the body has an ``if i == x`` guard
        (or ``x == i``). These are index-set-split points: a special-case iteration
        that blocks parallelization can be carved out as [start, x-1] + {x} + [x+1,
        end] (a boundary ``x`` simply drops the empty side -- see
        :meth:`_split_loop_at`). ``x`` must not depend on the loop variable."""
        import ast
        ivar_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
        values = []
        for cb in [b for b in loop.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]:
            for cond, _ in cb.branches:
                if cond is None:
                    continue
                try:
                    node = cond.code[0]
                except (AttributeError, IndexError, TypeError):
                    continue
                if isinstance(node, ast.Expr):
                    node = node.value
                if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
                    continue
                left, right = node.left, node.comparators[0]

                def is_ivar(n):
                    return isinstance(n, ast.Name) and n.id == loop.loop_variable

                if is_ivar(left) and not is_ivar(right):
                    other = right
                elif is_ivar(right) and not is_ivar(left):
                    other = left
                else:
                    continue
                try:
                    x = symbolic.pystr_to_symbolic(ast.unparse(other))
                except Exception:
                    continue
                if ivar_sym not in x.free_symbols and x not in values:
                    values.append(x)
        return values

    def _broadcast_conflict_split_points(self, loop: LoopRegion):
        """Iteration values ``x`` where a loop-invariant *broadcast read* collides
        with a per-iteration *write* to the same array.

        The body reads ``A[c]`` at a loop-invariant (constant) index ``c`` and also
        writes ``A[f(i)]`` at a loop-variable-dependent index; the single iteration
        ``x`` solving ``f(x) == c`` is the sole producer of the element every other
        iteration reads. It is NOT an ``if i == x`` guard, so
        :meth:`_equality_guard_values` misses it -- but the same index-set split
        unblocks it: ``[start, x-1]`` reads the original ``A[c]``, ``{x}`` writes it,
        ``[x+1, end]`` reads the new value, which is exactly the sequential order, and
        the two range segments become conflict-free parallel maps (TSVC s1113
        ``a[i] = a[N//2] + b[i]``).

        Returns loop-invariant split values (``!=`` the loop variable). Liberal by
        design: :meth:`_best_split_for` probes each candidate on an isolated copy and
        keeps only one that raises the mappable-loop count, so a non-helping candidate
        is harmless.
        """
        ivar = symbolic.pystr_to_symbolic(loop.loop_variable)
        reads: Dict[Any, list] = {}  # array -> loop-invariant single-point read subsets
        writes: Dict[Any, list] = {}  # array -> loop-var-dependent single-point write subsets
        for state in loop.all_states():
            if not isinstance(state, SDFGState):
                continue
            for node in state.data_nodes():
                for e in state.in_edges(node):
                    m = e.data
                    if m is not None and m.data is not None and m.subset is not None:
                        writes.setdefault(m.data, []).append(m.subset)
                for e in state.out_edges(node):
                    m = e.data
                    if m is not None and m.data is not None and m.subset is not None:
                        reads.setdefault(m.data, []).append(m.subset)
        values = []
        for data in set(reads) & set(writes):
            for rsub in reads[data]:
                # A broadcast read touches no dimension that varies with the loop var.
                if any(ivar in symbolic.pystr_to_symbolic(str(b)).free_symbols for (b, _e, _s) in rsub.ndrange()):
                    continue
                for wsub in writes[data]:
                    if len(wsub) != len(rsub):
                        continue
                    x = self._solve_write_eq_read(wsub, rsub, ivar)
                    if x is not None and ivar not in x.free_symbols and x not in values:
                        values.append(x)
        return values

    def _solve_write_eq_read(self, wsub, rsub, ivar):
        """Solve ``write_index(i) == read_const`` for the single ``i`` at which the
        per-iteration write hits the broadcast-read element, or ``None`` if there is
        no clean affine solution. Every non-loop-var dimension must already match
        (else the write never touches the read element); the one loop-var dimension
        must be affine ``a*i + b`` with ``|a| == 1`` (so the solution is an exact
        integer). Single-point accesses only."""
        sol = None
        for (wb, we, _ws), (rb, re_, _rs) in zip(wsub.ndrange(), rsub.ndrange()):
            w = symbolic.pystr_to_symbolic(str(wb))
            r = symbolic.pystr_to_symbolic(str(rb))
            if symbolic.simplify(symbolic.pystr_to_symbolic(str(we)) - w) != 0:
                return None  # multi-element write range in this dim -> not a clean point
            if symbolic.simplify(symbolic.pystr_to_symbolic(str(re_)) - r) != 0:
                return None  # multi-element read range in this dim
            if ivar in w.free_symbols:
                a = w.coeff(ivar, 1)
                b = symbolic.simplify(w - a * ivar)
                if ivar in a.free_symbols or ivar in b.free_symbols:
                    return None  # non-affine in the loop variable
                if not (a.is_number and symbolic.simplify(a * a - 1) == 0):
                    return None  # |a| != 1 -> solution may be non-integer
                xi = symbolic.simplify((r - b) / a)
                if sol is not None and symbolic.simplify(xi - sol) != 0:
                    return None  # inconsistent solution across dimensions
                sol = xi
            elif symbolic.simplify(w - r) != 0:
                return None  # non-loop-var dimension does not match -> no collision
        return sol

    def _split_loop_at(self, sdfg: SDFG, loop: LoopRegion, x, middle_singleton: bool = True) -> bool:
        """Index-set-split ``loop`` at iteration ``x`` into range segments, each a
        clone of the body wired in sequence in place of the loop. Unit stride only;
        returns whether it split.

        With ``middle_singleton=True`` (the default) the split carves out the single
        iteration ``x``: [start, x-1] + {x} + [x+1, end], for peeling a special-case
        iteration out of a range. A boundary ``x`` drops the side that would be empty
        (``x == start`` emits only {x} + [x+1, end]; ``x == end`` only [start, x-1] +
        {x}). The enclosing-range-aware ``LiftTrivialIf`` (run by
        :meth:`_clean_peeled_remainder` afterwards) then resolves the ``if i == x``
        guard per segment: a contradiction in the range segments, a tautology in the
        single middle iteration.

        With ``middle_singleton=False`` the split is the two halves [start, x-1] +
        [x, end] (``x`` joins the second half), for a band-boundary split where each
        half's wrap-around modulo folds to a different affine index -- the symbolic
        modular split ``a[(i + K) % N]`` at ``x = N - K``. A no-op ``x == start``
        (whole loop is the second half) returns ``False`` without splitting.

        CONTRACT: the segments regroup the SAME iterations only while ``start <= x <= end``. An
        ``x`` outside the range would invent iterations the loop never ran (the middle at ``i = x``)
        or run past its end (``before``'s ``i < x`` REPLACES the original bound), so a caller that
        cannot prove the range membership must guard the split with it -- see
        :meth:`_split_range_relations` and :meth:`_specialize_index_set_split`."""
        import copy
        from dace.properties import CodeBlock
        from dace.sdfg.sdfg import InterstateEdge
        from dace.transformation.passes.analysis import loop_analysis
        stride = loop_analysis.get_loop_stride(loop)
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        try:
            if stride is None or int(symbolic.evaluate(stride, sdfg.constants)) != 1:
                return False
        except (TypeError, ValueError):
            return False
        if start is None or end is None:
            return False
        ivar = loop.loop_variable
        parent = loop.parent_graph
        # Drop a range segment that is provably empty at a boundary split point.
        want_before = symbolic.simplify(x - start) != 0  # x != start -> [start, x-1] is non-empty
        want_after = symbolic.simplify(x - end) != 0  # x != end   -> [x+1, end] is non-empty
        if not middle_singleton and not want_before:
            return False  # [x, end] would be the whole loop -> nothing to split

        def clone_segment() -> LoopRegion:
            seg = copy.deepcopy(loop)
            seg.label = _unique_block_label(sdfg, loop.label)
            parent.add_node(seg)  # register so the next unique-label query sees it
            return seg

        chain = []
        if want_before:
            before = clone_segment()
            before.loop_condition = CodeBlock(f'{ivar} < ({x})')  # [start, x-1]
            chain.append(before)
        if middle_singleton:
            at = clone_segment()  # {x} intersected with [start, end]: at most a single iteration
            # Clamped to the range, so an out-of-range ``x`` runs nothing here rather than an
            # iteration the loop never had. Nobody maps a singleton, so the min/max costs no
            # parallelism -- unlike the range segments, whose bounds must stay bare for
            # ``LoopToMap`` and are guarded by :meth:`_split_range_relations` instead.
            at.init_statement = CodeBlock(f'{ivar} = max(({x}), ({start}))')
            at.loop_condition = CodeBlock(f'{ivar} < min(({x}), ({end})) + 1')
            chain.append(at)
            if want_after:
                after = clone_segment()
                after.init_statement = CodeBlock(f'{ivar} = ({x}) + 1')  # [x+1, end], original condition
                chain.append(after)
        else:
            after = clone_segment()  # [x, end]: x joins the second half, original condition
            after.init_statement = CodeBlock(f'{ivar} = ({x})')
            chain.append(after)

        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        is_start = parent.start_block is loop
        for ie in in_edges:
            parent.add_edge(ie.src, chain[0], ie.data)
            parent.remove_edge(ie)
        for prev, nxt in zip(chain, chain[1:]):
            parent.add_edge(prev, nxt, InterstateEdge())
        for oe in out_edges:
            parent.add_edge(chain[-1], oe.dst, oe.data)
            parent.remove_edge(oe)
        parent.remove_node(loop)
        if is_start:
            parent.start_block = parent.node_id(chain[0])
        parent.reset_cfg_list()
        return True

    def _split_range_relations(self, loop: LoopRegion, x):
        """The range-membership relations an index-set split at ``x`` needs for ``loop`` but cannot
        prove -- the missing half of :meth:`_split_loop_at`'s contract, for
        :meth:`_specialize_index_set_split` to emit as the ``if cond: par else: seq`` guard.

        A split point is only known to be loop-INVARIANT: a free ``K`` in ``if i == K``, or a
        ``N // 2`` broadcast point against a symbolic ``N``, says nothing about where it lands
        relative to the bounds. Only the two RANGE segments need guarding, and only in the one
        direction each can run away in (the middle singleton is clamped in
        :meth:`_split_loop_at`, and either range segment is simply empty in its other direction):

        - ``before`` = [start, x-1], emitted as ``i < x``, REPLACES the original end bound -- it
          overruns only when it runs an iteration ABOVE ``end``, i.e. when ``x - 1 > end``. Safe for
          ``x <= end + 1`` (at ``x == end + 1`` ``before`` is exactly the whole loop and the middle
          singleton and ``after`` are empty). Proven against that true bound; the emitted fallback
          guard stays the conservative ``x <= end`` (stricter, so always safe) when it is not proven.
        - ``after`` = [x+1, end], entered at ``i = x + 1`` -- an ``x < start`` re-runs the
          iterations below the start. Needs ``start <= x``.

        Empty means every needed side is provable and the split applies unconditionally (the
        boundary ``x == start`` of a front conflict emits no ``before`` at all, so nothing is left
        to prove). ``None`` means the bounds are unreadable and the caller must not split."""
        import sympy
        from dace.transformation.passes.analysis import loop_analysis
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None:
            return None  # bounds unreadable -> membership cannot even be stated
        relations = set()
        if symbolic.simplify(x - start) != 0 and not self._nonneg_in_loop(loop, end + 1 - x, start, end):
            relations.add(sympy.LessThan(x, end))  # a `before` segment exists -> it must not overrun
        if symbolic.simplify(x - end) != 0 and not self._nonneg_in_loop(loop, x - start, start, end):
            relations.add(sympy.LessThan(start, x))  # an `after` segment exists -> it must not underrun
        return frozenset(relations)

    def _nonneg_in_loop(self, loop: LoopRegion, expr, start, end) -> bool:
        """Whether ``expr >= 0`` holds -- either unconditionally
        (:meth:`_provably_nonneg`) or under the same large-trip-count assumption
        :meth:`_nonneg_assuming_large_modulus` already encodes: a symbolic loop
        being peeled/split by ``k <= peel_limit`` iterations presupposes it runs
        more than ``peel_limit`` times, so its trip count ``end - start + 1``
        exceeds ``peel_limit``.

        A near-boundary split point ``x`` carving off ``c <= peel_limit`` tail
        iterations has ``x - start = trip - 1 - c``, non-negative once
        ``trip > peel_limit`` -- so its range-membership is statically true and
        needs no ``if start <= x`` runtime guard (the ext_peel_multi_back back-peel
        at ``x = LEN_1D - 2``: ``x - start = LEN_1D - 2 = trip - 2 >= 0`` for
        ``trip = LEN_1D > peel_limit``). Applying the assumption with the trip
        count as the modulus (and NO wrap offsets, so no offset facts are relied
        on) proves only such affine-in-trip boundary points; a point leaning on a
        free offset (``N - K``) is left guarded. A non-affine floor/ceil point
        (``N // 2``) falls through to :meth:`_provably_nonneg_symbolic`, which
        bounds it from the nonnegative-symbol contract."""
        if self._provably_nonneg(expr):
            return True
        import sympy
        trip = symbolic.simplify(end - start + 1)
        if isinstance(trip, sympy.Symbol) and self._nonneg_assuming_large_modulus(expr, trip) is not None:
            return True
        # Floor/ceil-of-a-nonnegative-bound split points (``x = int_floor(N, 2)``) are not affine in
        # the trip count, so the large-modulus path leaves them guarded; the symbolic nonnegativity
        # prover discharges them from the canonicalization nonnegative-symbol contract + floor/ceil
        # bounds. See :meth:`_provably_nonneg_symbolic`.
        return self._provably_nonneg_symbolic(expr)

    def _specialize_index_set_split(self, sdfg: SDFG, loop: LoopRegion, x, relations) -> None:
        """Replace ``loop`` with ``if (start <= x <= end) { index-set split } else { original loop }``.

        The split form regroups the same iterations only inside the range (see
        :meth:`_split_loop_at`), so it is emitted as the true branch guarded by ``relations`` and
        the untouched loop -- correct wherever ``x`` lands -- is the sequential fallback. With no
        relation (a provably in-range ``x``) the split is applied in place, no branch. Mirrors
        :meth:`_specialize_modulo_split`, whose far-half fold is likewise conditional."""
        from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
        if not relations:
            if self._split_loop_at(sdfg, loop, x):
                self._clean_peeled_remainder(sdfg)
            return
        # A ``CodeBlock`` condition is PYTHON (``and``, not ``&&``), and the relation is rendered by
        # sympy's own printer -- not ``sym2cpp``, whose C ``/`` would turn the ``N // 2`` split point
        # into a true division. ``int_floor`` and friends round-trip through ``pystr_to_symbolic``.
        condition = ' and '.join(f'({r})' for r in sorted(relations, key=str))

        def _parallelize(par_loop, par_region, _owner):
            if self._split_loop_at(sdfg, par_loop, x):
                self._clean_peeled_remainder(par_region)

        specialize_loop_under_condition(loop, condition, _parallelize, sdfg)

    def _inner_loop_variables(self, loop: LoopRegion) -> set:
        """Iterator names bound by a ``LoopRegion`` nested strictly inside ``loop``.

        A split point that names one of these is not loop-invariant at ``loop``'s level: it is only
        defined per-iteration deeper in the nest, so embedding it in ``loop``'s segment bounds is
        out of scope. The reference stays hidden while the inner loop shares the name (the symbol
        table still ``defines`` it), then leaks the instant ``UniqueLoopIterators`` gives that inner
        loop a fresh unique name -- exactly the durbin ``_loop_it_1`` codegen failure."""
        return {
            r.loop_variable
            for r in loop.all_control_flow_regions() if r is not loop and isinstance(r, LoopRegion) and r.loop_variable
        }

    def _best_split_for(self, loop: LoopRegion, sdfg: SDFG):
        """The ``if i == x`` guard value whose index-set split unblocks the most maps
        for ``loop`` (probed on an isolated copy), or ``None`` if splitting does not
        help or the loop already maps."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        # Cheap structural gate FIRST: a loop with no ``if i == x`` equality guard and no
        # broadcast-conflict split point has no split candidate, so a split can never unblock it --
        # skip the can_be_applied probe AND the isolate-and-search (behaviour-identical: both paths
        # return None here). On a stencil like channel_flow this drops the whole split search to a
        # cheap syntactic scan instead of a per-loop can_be_applied probe.
        candidates = self._equality_guard_values(loop)
        for x in self._broadcast_conflict_split_points(loop):
            if x not in candidates:
                candidates.append(x)
        # A split point is baked into ``loop``'s segment bounds (see :meth:`_split_loop_at`), so it
        # must be in scope at ``loop``'s own level. Drop any candidate naming an iterator an INNER
        # loop binds -- a value that varies per inner iteration is undefined where the outer segments
        # test it (durbin's broadcast conflict ``y[k] == y[i]`` solves to ``x = i``, the inner loop
        # variable). Splitting there embeds the inner name into the outer bounds, valid only by
        # accident until ``UniqueLoopIterators`` renames the inner loop and the reference dangles as a
        # free symbol (``SDFG.arglist`` -> ``KeyError``). See :meth:`_inner_loop_variables`.
        inner = self._inner_loop_variables(loop)
        candidates = [x for x in candidates if inner.isdisjoint(str(s) for s in x.free_symbols)]
        if not candidates:
            return None
        try:
            if LoopToMap.can_be_applied_to(sdfg, loop=loop):
                return None
        except Exception:
            pass
        mini, _ = self._isolate_loop(loop, sdfg)
        if mini is None:
            return None
        try:
            baseline = self._mappable_loop_count(copy.deepcopy(mini))
        except Exception:
            return None
        best_count, best = baseline, None
        for x in candidates:
            cand = copy.deepcopy(mini)
            cloops = _loops(cand)
            if not cloops or not self._split_loop_at(cand, cloops[0], x):
                continue
            self._clean_peeled_remainder(cand)
            try:
                cand.validate()
                n_mappable = self._mappable_loop_count(cand)
            except Exception:
                continue
            if n_mappable > best_count:
                best_count, best = n_mappable, x
        return best

    def _prune_dead_loop_branches(self, sdfg: SDFG):
        """Collapse the boundary guards a peel leaves in the remainder loop body so
        what is left is clean affine code LoopToMap can map.

        Delegates to the iteration-range-aware :class:`LiftTrivialIf`: a guard that
        is false over the whole remainder range (``if i == N-1`` once the remainder
        is ``[0, N-2]``, the ``i == 0`` / ``i == 1`` arms of an if/elif chain peeled
        off the front) is a contradiction whose branch is dropped, and the surviving
        always-true guard (the if/elif's else body, lowered to ``if not(i==0): ...``)
        is a tautology that gets lifted up. The frontend lowers an if/elif chain to
        *nested* conditionals, and one ``LiftTrivialIf`` pass collapses one nesting
        level (lifting the outer guard exposes the inner one), so it is run to a
        fixpoint -- the conditional count strictly decreases, so this terminates --
        and ``EmptyStateElimination`` then tidies the empty boundary states a lift
        leaves behind."""
        from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf

        def num_conditionals() -> int:
            return sum(1 for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock))

        lift = LiftTrivialIf()
        prev = num_conditionals()
        while prev > 0:
            lift.apply_pass(sdfg, {})
            cur = num_conditionals()
            if cur >= prev:
                break  # no further trivial conditional to collapse
            prev = cur
        EmptyStateElimination().apply_pass(sdfg, {})

    def _clean_peeled_remainder(self, sdfg: SDFG, collect: Optional[set] = None):
        """Post-peel cleanup so the remainder body is affine and mappable: collapse
        the now-dead boundary guards (:meth:`_prune_dead_loop_branches`) and rewrite
        the modulo subsets the peel/split made affine over the shortened range
        (:meth:`_rewrite_modulo_over_range`). When ``collect`` is given, the
        ``offset < modulus`` relations a fold relied on are added to it -- the
        symbolic-split search uses that set as the ``if cond: par else: seq`` branch
        condition (a constant / no-offset fold contributes nothing)."""
        self._prune_dead_loop_branches(sdfg)
        self._rewrite_modulo_over_range(sdfg, collect)

    @staticmethod
    def _provably_nonneg(x) -> bool:
        """Whether ``x`` is provably ``>= 0`` -- a concrete non-negative number once
        simplified (the deciding differences of a range bound reduce to numbers)."""
        s = symbolic.simplify(x)
        return s.is_number and s >= 0

    @staticmethod
    def _provably_nonneg_symbolic(x) -> bool:
        """Whether ``x`` is provably ``>= 0`` for EVERY value of its free symbols, given the
        canonicalization contract that those symbols are nonnegative -- the symbolic superset of
        :meth:`_provably_nonneg`, which only decides concrete numbers.

        This discharges the range-membership relations of an index-set split whose split point is
        a floor/ceil of a nonnegative loop bound -- the archetype being the ``a[N // 2]`` broadcast
        conflict of ``s1113``, split at ``x = int_floor(N, 2)``. Both membership sides
        (``0 <= int_floor(N, 2)`` and ``int_floor(N, 2) <= N``) hold for all ``N >= 0``, so the
        guard is redundant and the split applies unconditionally.

        Sound because every step only ever REPLACES a subterm by a valid bound that makes the whole
        smaller, then asks sympy whether the weakened expression is still nonnegative:

        - Free symbols are re-asserted nonnegative (the canonicalization contract, already runtime
          guarded), and ``int_floor``/``int_ceil`` are rewritten to sympy's ``floor``/``ceiling``
          so its own sign reasoning applies. If that alone proves it, done.
        - Otherwise relax each rounding atom to the extreme that MINIMIZES the (affine) expression:
          ``floor(t) in [t - 1, t]`` and ``ceiling(t) in [t, t + 1]``, so a term with a negative
          coefficient takes the upper bound and one with a nonnegative coefficient the lower. If the
          resulting bound is nonnegative, so is ``x``. A non-numeric coefficient (nonlinear in the
          atom) is unprovable here -> ``False`` (conservative: the guard stays)."""
        import sympy
        s = symbolic.simplify(x)
        if s.is_number:
            return bool(s >= 0)
        nonneg = {sym: sympy.Symbol(sym.name, nonnegative=True, integer=bool(sym.is_integer)) for sym in s.free_symbols}
        e = s.replace(lambda t: isinstance(t, symbolic.int_floor), lambda t: sympy.floor(t.args[0] / t.args[1]))
        e = e.replace(lambda t: isinstance(t, symbolic.int_ceil), lambda t: sympy.ceiling(t.args[0] / t.args[1]))
        e = e.subs(nonneg)
        if e.is_nonnegative is True:
            return True
        bounds = {sympy.floor: (lambda a: a - 1, lambda a: a), sympy.ceiling: (lambda a: a, lambda a: a + 1)}
        atoms = list(e.atoms(sympy.floor, sympy.ceiling))
        if not atoms:
            return False
        relaxed = {}
        for a in atoms:
            c = e.coeff(a, 1)
            if not c.is_number:
                return False  # nonlinear in the rounding atom -> cannot bound
            low, high = bounds[a.func]
            relaxed[a] = high(a.args[0]) if c < 0 else low(a.args[0])
        return symbolic.simplify(e.subs(relaxed)).is_nonnegative is True

    def _nonneg_assuming_large_modulus(self, x, m, offsets=frozenset()):
        """Prove ``x >= 0`` given the modulus ``m`` is at least ``peel_limit + 1`` and
        every symbol in ``offsets`` lies in ``[0, m - 1]``. Returns the (possibly empty)
        subset of ``offsets`` the bound *leaned on* -- i.e. the ``offset < m`` facts a
        caller must record and runtime-check -- when ``x`` is provably ``>= 0``, else
        ``None`` (so ``result is None`` means "not proven"; an empty set means "proven
        without any offset assumption").

        Peeling/splitting a symbolic loop by ``k <= peel_limit`` iterations already
        presupposes the loop runs at least ``k`` times, so when ``m`` is the trip count
        it is ``> peel_limit``. The ``offset < m`` facts model the modular-wrap contract
        that a wrap offset is smaller than the modulus (``K < N`` for ``a[(i + K) % N]``);
        offsets are ``>= 0`` unconditionally (the canonicalization nonnegative-symbol
        contract, already runtime-guarded). ``x`` must be affine in ``m`` and the offset
        symbols with numeric coefficients; the minimum is taken at ``o = 0`` for a
        non-negative coefficient and ``o = m - 1`` for a negative one (the latter is what
        needs ``o < m``), then checked at the smallest admissible ``m``."""
        import sympy
        s = symbolic.simplify(x)
        if s.is_number:
            return frozenset() if s >= 0 else None
        if not isinstance(m, sympy.Symbol):
            return None
        # Affine decomposition in m and each offset symbol; every coefficient numeric.
        coeffs: Dict[Any, Any] = {}
        rem = s
        for sym in (m, *offsets):
            c = s.coeff(sym, 1)
            if not c.is_number:
                return None  # nonlinear or a cross term (e.g. m*offset) -- cannot bound
            coeffs[sym] = c
            rem = rem - c * sym
        c0 = symbolic.simplify(rem)
        if not c0.is_number:
            return None  # a free symbol we have no bound for remains
        # Fold each offset into the worst-case (C1*m + C0): a negative coefficient is
        # worst at o = m - 1 (contributing c*m - c, and needing o < m); a non-negative
        # one at o = 0. Record exactly the offsets whose lower extreme we relied on.
        C1 = coeffs[m]
        C0 = c0
        relied = set()
        for o in offsets:
            if coeffs[o] < 0:
                C1 += coeffs[o]
                C0 -= coeffs[o]
                relied.add(o)
        if not (C1.is_number and C0.is_number):
            return None
        if C1 < 0:
            return None  # decreasing in m -> not bounded below as m grows
        if bool(C0 + C1 * (self.peel_limit + 1) >= 0):
            return frozenset(relied)
        return None

    def _enclosing_loop_ranges(self, block) -> Dict[Any, Any]:
        """``{loop_variable: (start, end)}`` for every ``LoopRegion`` enclosing
        ``block``, with inclusive bounds. Empty for a peeled iteration region (no
        enclosing loop), whose body holds a fixed, already-substituted index."""
        from dace.transformation.passes.analysis import loop_analysis
        ranges: Dict[Any, Any] = {}
        graph = block.parent_graph
        seen = set()
        while graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            if isinstance(graph, LoopRegion) and graph.loop_variable:
                start = loop_analysis.get_init_assignment(graph)
                end = loop_analysis.get_loop_end(graph)
                if start is not None and end is not None:
                    ranges[symbolic.pystr_to_symbolic(graph.loop_variable)] = (start, end)
            graph = getattr(graph, 'parent_graph', None)
        return ranges

    @staticmethod
    def _modulo_operands(node):
        """``(arg, m)`` if ``node`` is a modulo of two operands -- either
        ``sympy.Mod`` (the ``%`` operator) or a recognised floor-mod helper-function
        call (see :data:`_MODULO_FUNC_NAMES`) -- else ``None``. Lets the rewrite
        accept whichever spelling defines the modulo."""
        import sympy
        if isinstance(node, sympy.Mod) and len(node.args) == 2:
            return node.args[0], node.args[1]
        name = getattr(getattr(node, 'func', None), '__name__', None)
        if name in _MODULO_FUNC_NAMES and len(node.args) == 2:
            return node.args[0], node.args[1]
        return None

    @staticmethod
    def _modulo_nodes(expr) -> set:
        """Every modulo subexpression of ``expr``: both the ``%`` operator
        (``sympy.Mod``) and the floor-mod helper-function spellings (see
        :data:`_MODULO_FUNC_NAMES`), so a wrap-around index is found regardless of
        which representation introduced it."""
        import sympy
        mods = set(expr.atoms(sympy.Mod))
        mods |= {f for f in expr.atoms(sympy.Function) if getattr(f.func, '__name__', None) in _MODULO_FUNC_NAMES}
        return mods

    def _modulo_to_affine(self, mod, ranges: Dict[Any, Any]):
        """If ``mod`` is a modulo ``arg % m`` (operator or helper function) with
        ``arg`` affine in the enclosing loop variables and provably confined to a
        single band ``[t*m, (t+1)*m - 1]`` over their ranges, return ``(arg - t*m,
        relations)`` -- the equivalent affine index (landing in ``[0, m-1]``, i.e.
        ``arg % m``) and the ``frozenset`` of ``offset < m`` relations the band
        proof leaned on (empty when none were needed); else ``None``. This makes a
        wrap-around access affine WITHOUT relying on C's truncated ``%`` (which
        would mis-handle a negative or beyond-``m`` argument): the remainder of
        ``arr[(i + k) % N]`` rewrites to ``i + k`` (band ``t = 0``), a peeled
        wrapping iteration ``Mod(N, N)`` to ``0`` (band ``t = 1``), ``Mod(-1, N)``
        to ``N - 1`` (band ``t = -1``); the far half of a symbolic-boundary split
        ``a[(i + K) % N]`` rewrites to ``i + K - N`` under the assumption ``K < N``,
        returned in ``relations`` for a caller to record and runtime-check. ``m``
        may be symbolic."""
        operands = self._modulo_operands(mod)
        if operands is None:
            return None
        arg, m = operands
        # Symbols in ``arg`` that are neither the modulus nor a ranged loop variable
        # are wrap OFFSETS: modelled as lying in ``[0, m - 1]`` (the modular-wrap
        # contract, ``offset < m``), which lets the band proof fold the far segment.
        offsets = frozenset(arg.free_symbols - set(ranges.keys()) - {m})
        # Reduce ``arg`` to its extremes over the enclosing loop box: affine in each
        # ranged variable, so the min/max sit at the range ends (per slope sign). A
        # peeled region has no enclosing loop, so ``arg`` is already a fixed point.
        lo = hi = arg
        for lv, (start, end) in ranges.items():
            if lv not in arg.free_symbols:
                continue
            a = arg.coeff(lv, 1)
            if lv in a.free_symbols or lv in (arg - a * lv).free_symbols:
                return None  # not affine in this loop variable
            if self._provably_nonneg(a):
                lo, hi = lo.subs(lv, start), hi.subs(lv, end)
            elif self._provably_nonneg(-a):
                lo, hi = lo.subs(lv, end), hi.subs(lv, start)
            else:
                return None  # indeterminate slope sign
        # Find the band ``t`` such that ``arg - t*m`` stays in ``[0, m-1]`` over the
        # whole range. Peeling/splitting shifts the argument by a bounded number of
        # strides, so the band index is within +/-(peel_limit + 1).
        import sympy
        for t in range(-(self.peel_limit + 1), self.peel_limit + 2):
            lo_relied = self._nonneg_assuming_large_modulus(lo - t * m, m, offsets)
            if lo_relied is None:
                continue
            hi_relied = self._nonneg_assuming_large_modulus(m - 1 - (hi - t * m), m, offsets)
            if hi_relied is None:
                continue
            relations = frozenset(sympy.StrictLessThan(o, m) for o in (lo_relied | hi_relied))
            return arg - t * m, relations
        return None

    def _loop_own_ranges(self, loop: LoopRegion) -> Dict[Any, Any]:
        """``{loop_variable: (start, end)}`` for ``loop`` itself (inclusive), the
        single-entry range box :meth:`_modulo_to_affine` reduces a modulo argument
        over; empty if the bounds are not recoverable."""
        from dace.transformation.passes.analysis import loop_analysis
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None or not loop.loop_variable:
            return {}
        return {symbolic.pystr_to_symbolic(loop.loop_variable): (start, end)}

    def _affine_body_modulos(self, loop: LoopRegion):
        """Yield ``(mod, arg, m, a, b)`` for every memlet-subset modulo ``Mod(arg,
        m)`` in ``loop``'s body whose argument ``arg = a*ivar + b`` is affine in the
        loop variable -- the only modulos a bounded peel or a band split can fold.
        Data-dependent / non-affine arguments are skipped: no bounded rewrite folds
        them. ``a`` and ``b`` may be symbolic (e.g. a symbolic stride or offset)."""
        import sympy
        from dace import subsets
        ranges = self._loop_own_ranges(loop)
        if not ranges:
            return
        (ivar, _), = ranges.items()
        for st in loop.all_states():
            for e in st.edges():
                if e.data is None:
                    continue
                for r in (e.data.subset, e.data.other_subset):
                    if not isinstance(r, subsets.Range):
                        continue
                    for rng in r.ranges:
                        for expr in rng:
                            if not isinstance(expr, sympy.Basic):
                                continue
                            for mod in self._modulo_nodes(expr):
                                operands = self._modulo_operands(mod)
                                if operands is None:
                                    continue
                                arg, m = operands
                                if ivar not in arg.free_symbols:
                                    continue
                                a = arg.coeff(ivar, 1)
                                b = arg - a * ivar
                                if ivar in a.free_symbols or ivar in b.free_symbols:
                                    continue  # arg not affine in the loop variable
                                yield mod, arg, m, a, symbolic.simplify(b)

    def _has_wrapping_modulo(self, loop: LoopRegion) -> bool:
        """Whether ``loop``'s body holds an affine memlet-subset modulo ``Mod(arg, m)``
        whose ``arg`` spans more than one band over the loop's full range -- so it does
        *not* reduce to a single affine index there (:meth:`_modulo_to_affine` returns
        ``None``). Such a wrap reads/writes the wrong element under C's truncated ``%``
        at the boundary iteration; peeling or splitting that boundary lets the band
        fold make each segment affine and floor-correct."""
        ranges = self._loop_own_ranges(loop)
        for mod, _arg, _m, _a, _b in self._affine_body_modulos(loop):
            if self._modulo_to_affine(mod, ranges) is None:
                return True  # affine but genuinely wrapping
        return False

    def _modulo_split_points(self, loop: LoopRegion):
        """Iteration values ``x`` where an affine wrapping body modulo ``arg = a*i +
        b`` (mod ``m``) crosses a band boundary ``t*m`` over ``loop``'s range: the
        solution of ``a*x + b == t*m``, i.e. ``x = (t*m - b)/a``, for each band the
        argument might reach. These are the SYMBOLIC split points that carve a wrap
        with a symbolic boundary (``a[(i + K) % N]`` crosses at ``x = N - K``) into
        single-band affine halves -- unreachable by a bounded, constant-count peel.

        Only unit-stride arguments (``|a| == 1``) yield a candidate: then ``x`` is an
        exact integer for every ``t``. Points provably at/outside the loop range are
        dropped (their split is a no-op or leaves an empty segment); the rest are
        returned liberally -- :meth:`_best_modulo_split_for` probes each on an
        isolated copy and keeps only one whose fold actually removes the wrap, so a
        non-splitting candidate is harmless."""
        import sympy
        ranges = self._loop_own_ranges(loop)
        if not ranges:
            return []
        (ivar, (start, end)), = ranges.items()
        points = []
        for _mod, _arg, m, a, b in self._affine_body_modulos(loop):
            if not (a.is_number and symbolic.simplify(a * a - 1) == 0):
                continue  # |a| != 1: the crossing is not an exact integer in general
            for t in range(-(self.peel_limit + 1), self.peel_limit + 2):
                x = symbolic.simplify((t * m - b) / a)
                if ivar in x.free_symbols:
                    continue
                # Drop x at/left of the start (empty/no-op before-segment) or right of
                # the end (empty after-segment) when that is PROVABLE; keep the rest.
                if self._provably_nonneg(start - x) or self._provably_nonneg(x - end - 1):
                    continue
                if x not in points:
                    points.append(x)
        return points

    def _best_modulo_peel_for(self, loop: LoopRegion, sdfg: SDFG):
        """The smallest ``(count, direction)`` peel that turns a genuinely-wrapping
        body modulo into an affine, floor-correct index for ``loop``, probed on an
        isolated copy; ``None`` if the loop has no such wrap or no bounded peel folds
        it. Unlike the index-set split above, this runs *even when* ``LoopToMap`` already
        maps the loop: a wrap-around read maps as-is but then emits C's truncated
        ``%``, computing the wrong boundary value, so the peel is for correctness, not
        to unblock a map. A peel is accepted as soon as it removes every wrapping
        modulo (so no C ``%`` survives) while staying valid -- folding the wrap to its
        affine, floor-correct form is value-preserving whether or not it also maps."""
        if not self._has_wrapping_modulo(loop):
            return None
        mini, _ = self._isolate_loop(loop, sdfg)
        if mini is None:
            return None
        # Smallest peel first (count ascending, single-side before both) so the
        # correctness fix touches as few iterations as possible.
        for count in range(1, self.peel_limit + 1):
            for direction in ('front', 'back', 'both'):
                cand = copy.deepcopy(mini)
                cloops = _loops(cand)
                if not cloops:
                    return None
                if not self._peel_one_loop(cand, cloops[0], count, direction):
                    continue
                self._clean_peeled_remainder(cand)
                try:
                    cand.validate()
                    if any(self._has_wrapping_modulo(l) for l in _loops(cand)):
                        continue  # a wrapping modulo still survives -> C ``%`` remains
                except Exception:
                    continue
                return (count, direction)
        return None

    def _best_modulo_split_for(self, loop: LoopRegion, sdfg: SDFG):
        """``(x, condition_relations)`` for the band-crossing split that turns a
        genuinely-wrapping body modulo into affine, floor-correct indices on BOTH
        halves of ``loop``, probed on an isolated copy; ``None`` if the loop has no
        such wrap or no candidate split folds it. This is the symbolic-boundary
        counterpart of :meth:`_best_modulo_peel_for`: a bounded peel can only reach a
        boundary a constant number of iterations from an end, so a wrap whose
        boundary is symbolic (``a[(i + K) % N]``, boundary ``i = N - K``) needs a
        split at the symbolic crossing ``x = N - K`` instead. Each half's wrap then
        folds to a single-band affine index -- the near half ``i + K``
        unconditionally, the far half ``i + K - N`` under ``K < N``. Those
        ``offset < modulus`` relations are returned as the ``if cond: par else: seq``
        branch condition: the split (parallel) form is emitted under them and the
        original modular loop is the sequential fallback for the rest. A split is
        accepted as soon as it removes every wrapping modulo while staying valid."""
        if not self._has_wrapping_modulo(loop):
            return None
        points = self._modulo_split_points(loop)
        if not points:
            return None
        mini, _ = self._isolate_loop(loop, sdfg)
        if mini is None:
            return None
        for x in points:
            cand = copy.deepcopy(mini)
            cloops = _loops(cand)
            if not cloops:
                return None
            if not self._split_loop_at(cand, cloops[0], x, middle_singleton=False):
                continue
            relations: set = set()
            self._clean_peeled_remainder(cand, collect=relations)  # capture the branch condition
            try:
                cand.validate()
                if any(self._has_wrapping_modulo(l) for l in _loops(cand)):
                    continue  # a wrapping modulo still survives -> C ``%`` remains
            except Exception:
                continue
            return x, frozenset(relations)
        return None

    def _rewrite_modulo_over_range(self, sdfg: SDFG, collect: Optional[set] = None):
        """Rewrite every ``Mod(arg, m)`` memlet subset to an equivalent affine form
        when ``arg`` provably stays in one band over its enclosing loops' ranges (see
        :meth:`_modulo_to_affine`). A peel that removes the wrapping boundary
        iterations leaves a remainder whose wrap-around modulo is the identity, and
        the peeled iterations themselves carry the wrapped constant index -- both are
        folded here so the body is affine and C's truncated ``%`` is never emitted.

        When ``collect`` is given, every ``offset < modulus`` relation a fold leaned
        on is added to it. The symbolic-split search reads that set back as the
        ``if cond: par else: seq`` branch condition (the fold is applied inside the
        parallel branch, where the condition already holds, so the relation guards
        the branch rather than aborting)."""
        import sympy
        from dace import subsets
        for st in sdfg.all_states():
            ranges = self._enclosing_loop_ranges(st)
            repl: Dict[Any, Any] = {}
            ranges_seen = []
            for e in st.edges():
                if e.data is None:
                    continue
                for r in (e.data.subset, e.data.other_subset):
                    if not isinstance(r, subsets.Range):
                        continue
                    ranges_seen.append(r)
                    for rng in r.ranges:
                        for expr in rng:
                            if not isinstance(expr, sympy.Basic):
                                continue
                            for mod in self._modulo_nodes(expr):
                                if mod in repl:
                                    continue
                                result = self._modulo_to_affine(mod, ranges)
                                if result is not None:
                                    affine, relations = result
                                    repl[mod] = affine
                                    if collect is not None:
                                        collect.update(relations)
            if repl:
                for r in ranges_seen:
                    r.replace(repl)

    def _mappable_loop_count(self, candidate: SDFG) -> int:
        """Cheap proxy for "does the peel unblock parallelization?": run scalar
        fission -> symbol propagation -> constant propagation (no reduction passes),
        then COUNT the loops ``LoopToMap`` *could* parallelize -- via
        ``can_be_applied_to``, WITHOUT applying it. Peeling is a preparation pass;
        the actual ``LoopToMap`` is the pipeline's ``parallelize`` stage, so the
        search only probes ``can_be_applied``. Mutates ``candidate`` (the prep)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.constant_propagation import ConstantPropagation
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        PrivatizeScalars().apply_pass(candidate, {})
        SymbolPropagation().apply_pass(candidate, {})
        ConstantPropagation().apply_pass(candidate, {})
        count = 0
        for loop in _loops(candidate):
            try:
                if LoopToMap.can_be_applied_to(candidate, loop=loop):
                    count += 1
            except Exception:
                pass
        return count

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Unblock stuck loops, splitting or peeling each at a point taken from its body.
        Returns the number of loops rewritten or None.

        An index-set split comes first. Its point ``x`` is solved for, not guessed: from an
        interior equality guard (``if i == x``), or from a broadcast/self-conflict read
        (``a[i] = a[c]``, solving ``f(x) == c``). The carve is [start, x-1] + {x} + [x+1, end],
        and ``_split_loop_at`` drops whichever side is empty -- so a point ON a boundary yields
        a peel of one, which is why there is no separate peel stage. The carve regroups the
        loop's own iterations only while ``start <= x <= end``, and a loop-invariant ``x`` need
        not satisfy it (a free ``K`` in ``if i == K``), so an unprovable membership is emitted
        as ``if (start <= x <= end) { split } else { original loop }`` rather than assumed.

        Next a wrapping-modulo peel fixes a boundary that reads/writes the wrong element under
        C's truncated ``%`` -- a correctness fix, so it runs even when the loop already maps.
        This is the only stage that searches (bounded by ``peel_limit``), and the only one that
        produces peels in practice. A wrap whose boundary is symbolic -- unreachable by any
        constant-count peel -- is instead *specialized* into ``if (K < N) { band-split into
        affine maps } else { original modular loop }`` (``a[(i + K) % N]`` split at
        ``x = N - K``): the parallel split is value-correct only below the modulus, so the
        untouched modular loop is kept as the sequential fallback. A loop no derived point
        reaches is left alone, sequential and correct."""
        if self.peel_limit <= 0 or not _loops(sdfg):
            return None
        applied = 0
        # 1. Index-set splitting for interior equality guards. The split only regroups the loop's
        #    own iterations while the split point lies inside the range, which a loop-invariant
        #    point does not have to (``if i == K`` with a free ``K``), so an unprovable membership
        #    becomes the `if in-range: split else: original loop` branch condition.
        for loop in list(_loops(sdfg)):
            x = self._best_split_for(loop, sdfg)
            if x is None:
                continue
            relations = self._split_range_relations(loop, x)
            if relations is None:
                continue
            self._specialize_index_set_split(sdfg, loop, x, relations)
            applied += 1
        # 2. Peel a genuinely-wrapping body modulo to its floor-correct affine form,
        #    even for loops LoopToMap already maps (the wrap-around access otherwise
        #    emits C's truncated ``%`` and computes the wrong boundary value).
        for loop in list(_loops(sdfg)):
            best = self._best_modulo_peel_for(loop, sdfg)
            if best is not None and self._peel_one_loop(sdfg, loop, *best):
                self._clean_peeled_remainder(sdfg)
                applied += 1
        # 2b. Specialize a symbolic-boundary wrap the constant peel could not reach
        #     into `if (offset < modulus) { band split } else { original loop }`. Only
        #     fires when a wrap survived the peel (constant offsets are handled there).
        for loop in list(_loops(sdfg)):
            res = self._best_modulo_split_for(loop, sdfg)
            if res is not None:
                x, relations = res
                self._specialize_modulo_split(sdfg, loop, x, relations)
                applied += 1
        # There is deliberately no separate boundary-peel stage. A boundary peel IS an index-set split
        # at a boundary point: ``_split_loop_at`` drops the side that would be empty, so ``x == end``
        # emits [start, end-1] + {end} -- a back-peel of one -- and stage 1 already reaches it from the
        # guard. The old stage searched a blind 3-directions x peel_limit grid per loop (24 speculative
        # peels, each a deepcopy + peel + validate + a full mappability recount) and, measured across
        # 151 TSVC kernels, all of tsvc_2_5 and 6 polybench kernels, produced ZERO peels while costing
        # 86.5% of the whole canonicalization pipeline on CloudSC (5398s of 6242s). Every peel that
        # does fire comes from the modulo stage above; every boundary case that resolves comes from the
        # split. Nor is an iteration-variable INEQUALITY guard promoted to a split point to compensate:
        # it bounds the run rather than marking a boundary between a special case and the rest, so
        # there is nothing for a split to carve out. No corpus kernel asks for one, and an attempt to
        # add them regressed nussinov (a point derived in the ENCLOSING loop's variable; three
        # InvalidSDFGEdgeError failures, root cause never pinned down) -- so the evidence for the
        # feature was zero and the evidence against it was three tests.
        return applied or None

    def _specialize_modulo_split(self, sdfg: SDFG, loop: LoopRegion, x, relations) -> None:
        """Replace ``loop`` with ``if (offset < modulus) { band split into affine
        maps } else { original modular loop }``.

        The far split half's fold to ``i + K - N`` is value-correct only below the
        modulus, so the split (parallel) form is emitted as the true branch guarded
        by ``relations`` (``K < N``) and the untouched modular loop -- correct for
        any nonnegative offset via C's floor-mod -- is the sequential fallback. With
        no relation (an unconditional fold) the split is applied in place, no branch.
        """
        from dace.codegen.common import sym2cpp
        from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
        if not relations:
            if self._split_loop_at(sdfg, loop, x, middle_singleton=False):
                self._clean_peeled_remainder(sdfg)
            return
        condition = ' && '.join(sym2cpp(r) for r in sorted(relations, key=str))

        def _parallelize(par_loop, par_region, _owner):
            if self._split_loop_at(sdfg, par_loop, x, middle_singleton=False):
                self._clean_peeled_remainder(par_region)

        specialize_loop_under_condition(loop, condition, _parallelize, sdfg)
