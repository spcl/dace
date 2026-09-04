# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Break loop-carried anti-dependences by snapshotting the read array.

A loop such as ``a[i] = a[i+1] + b[i]`` carries a write-after-read (WAR)
anti-dependence on ``a``: iteration ``i`` reads ``a[i+1]``, which a *later*
iteration overwrites. It cannot become a map as written. Copying ``a`` into a
fresh transient before the loop and reading the snapshot inside the loop removes
the WAR -- reads then come from a distinct, read-only array and writes go to
disjoint elements of ``a`` -- so ``LoopToMap`` can parallelize it.

This is **only sound for a pure anti-dependence (WAR with no RAW)**: the loop must
not read a value an *earlier* iteration wrote (a read-behind ``a[i-1]`` is a true
recurrence and must stay sequential). The pass therefore renames only when, for
the array's affine point accesses, every read/write pair is WAR or non-aliasing
and at least one is WAR.

It trades an extra array + a copy of the window the loop reads (see
:meth:`BreakAntiDependence._snapshot_window`) for parallelism, so it is meant to run
**optionally** (a tuning knob), not as part of the default pipeline.

Two admission policies share the one classifier, selected by the ``forward_reads``
property. Breaking an anti-dependence is this pass's job in BOTH; nothing else in
the codebase snapshots, and in particular ``SplitStatements`` does distribution
only:

* **whole array** (``forward_reads=False``, the default) -- the array's ENTIRE
  read/write cross product must be free of true dependences, so after the rename
  every read comes from the snapshot and ``LoopToMap`` can parallelize the loop.
* **forward reads** (``forward_reads=True``) -- admit an individual read EDGE that
  is purely read-ahead even when the array's OTHER reads are not. The loop may well
  stay sequential; what this buys is that the read-ahead no longer binds two
  otherwise-independent statements, so fission can distribute them (TSVC ``s1244``,
  ``A[i]=..; d[i]=A[i]+A[i+1]``). Its symbolic-offset guard is STRICTLY positive
  rather than nonnegative -- see :meth:`BreakAntiDependence._snapshot_forward_reads`.

ORDER: run this AFTER ``SplitStatements``, not before. Distribution is what makes
an anti-dependence go away for free -- once the read-ahead and the write sit in
different loops there is nothing left to break -- so breaking first pays a
whole-array copy for arrays the distribution was about to separate anyway. The
converse does not hold on the shapes that motivate either pass: in the mixed
``A[i]=..; d[i]=A[i]+A[i+1]``, the same-index read ``A[i]`` STAYS on the live
array (redirecting it would read the stale original), so ``A`` is still both read
and written by ``d``'s group afterwards and ``SplitStatements`` refuses it exactly
as it did before -- breaking first exposes no split that distribution-first
misses. Checked against the motivating carry kernel (no anti-dependence at all,
order irrelevant), TSVC ``s421`` (single output, distribution never applies) and
the ``fission_dep_const_offset`` / ``fission_dep_sym_offset`` shapes (refused by
the split either way, distributed later by ``LoopFission``).

Breaking is never declined because the copy looks expensive: the canonical form
prefers the parallel version, and cost is the tuner's call, not this pass's.
"""
import copy
import zlib
from functools import lru_cache
from typing import Any, Dict, Optional, Set

from dace.ordered import OrderedSet
import sympy

from dace import data, dtypes, properties, subsets, symbolic, Memlet
from dace.sdfg import SDFG, nodes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl


def _subset_key(subset):
    """A hashable identity for a subset that captures exactly what
    :meth:`BreakAntiDependence._dep_class` reads off it -- its ``ndrange``.

    Used to drop duplicate subsets before the read x write cross product. Falls
    back to the string form for the odd subset whose bounds are not hashable.
    """
    if subset is None:
        return None
    key = tuple(subset.ndrange())
    try:
        hash(key)
    except TypeError:
        return str(subset)
    return key


@lru_cache(maxsize=8192, typed=True)
def _reparsed_index(raw):
    """A memlet subset bound re-read through DaCe's own parser.

    The print-and-reparse round trip normalises the expression the way the rest of
    the pass expects (DaCe's converter maps ``/`` onto ``int_floor`` and friends).
    Printing a SymPy expression is expensive and the same handful of index bounds
    recurs across the whole read x write cross product, so the round trip is
    memoized on the incoming expression.
    """
    return symbolic.pystr_to_symbolic(str(raw))


@lru_cache(maxsize=8192, typed=True)
def _index_under_bindings(expr, bindings):
    """``expr`` with the loop's iedge bindings inlined (see
    :meth:`BreakAntiDependence._collect_iedge_substitutions`).

    ``bindings`` is the substitution map as a tuple of pairs so the result can be
    memoized: the map is the same for every subset of a given loop, and the same
    few index bounds recur across the read x write cross product, while
    ``Basic.subs`` over a body-sized binding map is not cheap.
    """
    return symbolic.simplify(expr.subs(dict(bindings)))


@lru_cache(maxsize=8192, typed=True)
def _is_identically_zero(diff) -> bool:
    """``True`` iff ``diff`` is the zero expression, i.e. the two indices alias.

    ``sympy.simplify`` decides this but costs about a millisecond per call, and the
    subset-difference test runs once per (read, write, dimension) triple -- it
    dominated the pass. Two exact shortcuts avoid it:

    * ``expand`` is a COMPLETE zero test for a polynomial: a polynomial is
      identically zero iff its expanded form is, and DaCe symbols are mutually
      independent, so a nonzero expanded linear form (``jk - jl``, the shape
      essentially every memlet-index difference takes) is decided outright.
    * anything non-polynomial (``int_floor``, ``Mod``, an opaque function) falls
      back to the full ``simplify``, so no verdict changes.

    Memoized because index differences repeat heavily across the read x write
    cross product. SymPy hashing ignores DaCe symbol dtype metadata, which is
    irrelevant here: the result is a purely algebraic yes/no.
    """
    expanded = sympy.expand(diff)
    if expanded.is_zero:
        return True
    if expanded.is_polynomial(*expanded.free_symbols):
        return False
    return symbolic.simplify(diff) == 0


def _provably_nonnegative_under_nonneg_symbols(expr) -> bool:
    """``True`` iff ``expr >= 0`` for every nonnegative value of its free symbols.

    Canonicalization assumes symbols (array sizes, strides, offsets) are
    nonnegative, but DaCe symbols carry no sign assumption (``is_nonnegative`` is
    ``None``), so we re-evaluate the expression with each free symbol replaced by
    a nonnegative one and ask sympy. ``>= 0`` -- not ``> 0`` -- is exactly the
    soundness condition for the snapshot-and-redirect rewrite: a read at
    ``a[i + offset]`` with ``offset >= 0`` is never written by an *earlier*
    iteration (which writes ``a[i' ], i' < i``), so reading the snapshot is sound;
    an ``offset < 0`` read-behind is a true RAW recurrence.

    A bare symbol (``K``) or a sum of nonnegatives (``K + N``) is provably
    nonnegative; a negation (``-K``) is not; and -- importantly -- a difference of
    two nonnegatives (``K - M``) is *not* provably nonnegative (its sign is
    undecidable even under the assumption), so it is rejected rather than renamed.
    Returns ``False`` on any sympy uncertainty (``is_nonnegative`` is ``None``).
    """
    from dace import symbolic
    try:
        # Fresh DaCe symbols (uncached ``__xnew__``) carrying the nonnegativity
        # assumption for a LOCAL proof; ``_eval_subs`` matches by name, so the
        # substitution lands without polluting the global symbol registry.
        subs = {s: symbolic.symbol(s.name, nonnegative=True) for s in expr.free_symbols}
        return bool(expr.subs(subs).is_nonnegative)
    except (AttributeError, TypeError):
        return False


def _provably_nonpositive_under_nonneg_symbols(expr) -> bool:
    """``True`` iff ``expr <= 0`` for every nonnegative value of its free symbols.

    Mirror of :func:`_provably_nonnegative_under_nonneg_symbols` (``expr <= 0`` iff
    ``-expr >= 0``). Used to separate a *provable* read-behind offset (``a[i - K]``, a
    true recurrence -- read-behind, sequential, and safe for a consumer to fuse) from an
    offset whose sign is undecidable even under the assumption (``K - M``), which is
    neither provably read-ahead nor provably read-behind and must not be mistaken for
    either. Returns ``False`` on any sympy uncertainty.
    """
    return _provably_nonnegative_under_nonneg_symbols(-expr)


def referenced_arrays(expr) -> Set[str]:
    """Data containers an index expression reads once its interstate bindings are expanded.

    The solver models each container as an immutable array, so a container the loop WRITES
    cannot be modelled this way and the caller must refuse.
    """
    names: Set[str] = set()
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, symbolic.Subscript):
            names.add(str(node.args[0]))
        elif isinstance(node, sympy.Indexed):
            names.add(str(node.base))
    return names


def written_data(loop: LoopRegion) -> Set[str]:
    """Every data container written anywhere in ``loop``'s body."""
    written: Set[str] = set()
    for st in loop.all_states():
        for n in st.data_nodes():
            for e in st.in_edges(n):
                if e.data is not None and not e.data.is_empty():
                    written.add(n.data)
                    break
    return written


def point_index(subset):
    """The single index of a one-dimensional point subset, or ``None``."""
    nd = list(subset.ndrange())
    if len(nd) != 1:
        return None
    rb, re_, _ = nd[0]
    if rb != re_:
        return None
    return _reparsed_index(rb)


@properties.make_properties
class BreakAntiDependence(ppl.Pass):
    """Snapshot-rename loops with a pure WAR anti-dependence so they can map.

    Off by default in pipelines (it adds a transient + a copy); enable it as a
    tuning knob when the extra buffer is worth the parallelism.
    """

    CATEGORY: str = 'Optimization Preparation'

    forward_reads = properties.Property(
        dtype=bool,
        default=False,
        desc="Break a read-ahead EDGE even when the array's other reads are true dependences, so the "
        "read-ahead stops binding two otherwise-independent statements (TSVC s1244).")

    def __init__(self, forward_reads: bool = False) -> None:
        super().__init__()
        self.forward_reads = forward_reads

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    @staticmethod
    def _loops(sdfg: SDFG):
        return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

    def _safe_stride(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """``True`` for loops whose stride is forward (numeric > 0 or symbolic).

        The per-iteration WAR analysis in :meth:`_dep_class` is direction-aware
        only by the sign of the carried offset, so reverse-iteration loops
        would misclassify (a forward-read pattern under stride < 0 is RAW, not
        WAR). Reverse loops are normalised away by
        :class:`~dace.transformation.passes.canonicalize.normalize_negative_stride.NormalizeNegativeStride`
        before this pass runs; here we only require the stride to be forward.

        For symbolic strides the actual positivity is deferred to the
        ``WAR_symbolic`` runtime guard the caller emits -- when the carried
        offset equals the stride (the canonical ``a[i] = a[i + inc] + b[i]``
        shape of TSVC s175), the existing ``inc > 0`` guard already implies
        ``stride > 0``; when they differ the guard collected via
        :meth:`_renamable_arrays` already conjoins all per-array offset
        constraints, which subsumes the stride-positivity ask.
        """
        from dace.transformation.passes.analysis import loop_analysis
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None:
            return False
        try:
            v = int(symbolic.evaluate(stride, sdfg.constants))
            return v > 0
        except (TypeError, ValueError):
            # Symbolic stride: defer the positivity check to the runtime guard.
            return True

    def _dep_class(self, read, write, ivar, loop=None, sdfg=None, iedge_subs=None):
        """Classify the dependence between a read and a write subset (both affine
        point accesses) of one array under unit-stride iteration of ``ivar``.

        :returns: One of:

            * ``('WAR', None)``             -- read-ahead anti-dep with constant +offset
            * ``('WAR_symbolic', expr)``    -- carried offset is a non-numeric symbolic
              expression ``expr`` independent of the iteration variable; only sound
              to rename if ``expr > 0`` at runtime (the caller emits the guard).
            * ``('WAR_indirected', name)``  -- carried offset reduces to ``arr[i]`` for
              some array ``arr``; only sound to rename if every element of ``arr`` is
              positive at runtime (the caller emits a per-element guard loop).
            * ``('RAW', None)``             -- read-behind true dep (sequential)
            * ``('none', None)``            -- never alias, or the same element of the
              same iteration (no dependence is CARRIED either way)
            * ``('invariant', None)``       -- both accesses hit the same LOOP-INVARIANT
              location (no subset mentions the iterator), so there is no carried offset
              to speak of. Distinct from ``'none'``: the accesses DO alias, every
              iteration. This pass treats it exactly as ``'none'`` -- neither is a
              carried anti-dependence, so neither is renamable -- but a caller reasoning
              about reordering (e.g. fusing two loops) must not: unfused, a later loop
              reads the FINAL value left in that location; interleaved, it reads the
              RUNNING one.
            * ``('complex', None)``         -- give up

        ``iedge_subs`` is the loop's iedge substitution map (see
        :meth:`_collect_iedge_substitutions`). It depends only on ``loop`` and
        ``ivar``, never on the two subsets, so callers that classify many pairs
        of the same loop compute it once and pass it in; ``None`` means "derive
        it here".
        """
        isym = symbolic.pystr_to_symbolic(ivar)
        rr, wr = list(read.ndrange()), list(write.ndrange())
        if len(rr) != len(wr):
            return ('complex', None)
        # Inline iedge symbol bindings (``i := LEN_1D - _loop_pos_0 - 2`` is
        # what :class:`NormalizeNegativeStride` plants for a reversed loop) so
        # the matcher sees memlet subsets in terms of the actual loop
        # iterator ``isym`` rather than indirect frontend-bound symbols.
        # Single ``.subs(...)`` + ``simplify(...)`` is sufficient for the
        # patterns we target (the bindings we admit reference the iterator
        # directly; see :meth:`_collect_iedge_substitutions` for the gate).
        if iedge_subs is None:
            iedge_subs = self._collect_iedge_substitutions(loop, isym, sdfg) if loop is not None else {}
        bindings = tuple(iedge_subs.items())
        carried_offset = None
        for (rb, re_, _), (wb, we_, _) in zip(rr, wr):
            if rb != re_ or wb != we_:
                return ('complex', None)  # not a single-element (point) access
            rb = _reparsed_index(rb)
            wb = _reparsed_index(wb)
            if bindings:
                rb = _index_under_bindings(rb, bindings)
                wb = _index_under_bindings(wb, bindings)
            r_has = isym in rb.free_symbols
            w_has = isym in wb.free_symbols
            if not r_has and not w_has:
                if not _is_identically_zero(rb - wb):
                    return ('none', None)  # different fixed index -> never alias
                continue
            # carried dimension: decompose ``wb`` as ``alpha * isym + beta`` with
            # ``alpha in {+1, -1}`` and ``beta`` loop-invariant. ``alpha = +1`` is
            # the standard forward-stride case; ``alpha = -1`` arises after
            # :class:`NormalizeNegativeStride` rewrites a ``range(hi, lo, -1)``
            # loop -- the body's memlets are then in the form ``a[c - k]`` for
            # the new positive-stride iterator ``k``. Both cases are sound for
            # the snapshot-and-redirect rewrite; only the iteration-direction
            # interpretation of ``carried_offset`` differs.
            # ``simplify`` never introduces a free symbol that was not already there,
            # so an ``isym`` the raw (already term-collected) difference has dropped
            # stays dropped -- test that first and only simplify when it has not.
            wb_minus_i = wb - isym
            if isym not in wb_minus_i.free_symbols or isym not in symbolic.simplify(wb_minus_i).free_symbols:
                alpha = 1
            else:
                wb_plus_i = wb + isym
                if isym not in wb_plus_i.free_symbols or isym not in symbolic.simplify(wb_plus_i).free_symbols:
                    alpha = -1
                else:
                    return ('complex', None)
            if carried_offset is not None:
                return ('complex', None)  # more than one carried dimension
            carried_offset = symbolic.simplify(rb - wb)
            # Effective offset in iteration-time space: solving ``rb(i1) = wb(i2)``
            # under ``rb = alpha*i + gamma``, ``wb = alpha*i + beta`` gives
            # ``i2 - i1 = (gamma - beta) / alpha = carried_offset / alpha``.
            # For alpha = -1 the iteration-time direction flips; multiply by
            # alpha so the downstream sign tests stay uniform.
            carried_offset = symbolic.simplify(alpha * carried_offset)
            # TODO(break-anti-dep, pre-existing): the alpha=-1 post-NNS reverse
            # cases (tests test_break_anti_dependence_alpha_minus_one_with_larger_offset
            # and ..._post_normalize_negative_stride_reverse_scan) classify as WAR
            # and snapshot-rename correctly, but the *resulting* loop still carries
            # the ``i := N - _loop_pos_0 - 1`` reverse rebinding, and LoopToMap then
            # refuses it (0 maps). Per the positive-symbol assumption, that negative
            # reverse index test can be normalized to a positive forward form so
            # LoopToMap can map the snapshotted loop. Not addressed here.
        if carried_offset is None:
            # No dimension mentioned the iterator, and every one was the same fixed index (a differing
            # one already returned 'none' above): the read and the write hit the SAME loop-invariant
            # location. Not a carried anti-dependence, so not our case -- but it is an alias, which
            # ``'none'`` would deny to callers who need to know.
            return ('invariant', None)
        if carried_offset.is_number:
            if carried_offset > 0:
                return ('WAR', None)
            if carried_offset < 0:
                return ('RAW', None)
            return ('none', None)
        # Symbolic offset path. Three sub-cases depending on what's left in the
        # carried_offset's free symbols:
        #
        #   (a) ``isym`` (the iter var) is NOT present  -> straightforward
        #       symbolic positive offset; emit a single positive-check guard.
        #
        #   (b) ``isym`` IS present but the offset resolves to ``arr[isym]`` after
        #       walking back symbol definitions through interstate edges and
        #       tasklets (the ``a[i + idx[i]]`` family) -> WAR_indirected with the
        #       indirection array name. Caller emits a per-element array guard.
        #
        #   (c) ``isym`` is present and resolution fails  -> conservative complex.
        if isym not in carried_offset.free_symbols:
            # Read-ahead (offset >= 0) is a renamable WAR; anything else is a true
            # recurrence (read-behind) or an offset whose sign we cannot establish,
            # both of which must stay sequential -> RAW. Canonicalization assumes
            # symbols are nonnegative, so we test the offset *under that assumption*
            # (``+K`` -> WAR; ``-K`` -> RAW). Critically, a difference like ``K - M``
            # is NOT provably nonnegative even with that assumption, so it is refused
            # as RAW rather than renamed: the old test (``could_extract_minus_sign``)
            # is canonical-ordering-dependent and let ``K - M`` through as a guarded
            # WAR (while refusing the algebraically equivalent ``M - K``), emitting an
            # unsatisfiable runtime ``> 0`` guard that traps and, once DCE'd, silently
            # corrupts the result.
            if _provably_nonnegative_under_nonneg_symbols(carried_offset):
                return ('WAR_symbolic', carried_offset)
            if _provably_nonpositive_under_nonneg_symbols(carried_offset):
                return ('RAW', None)  # provable read-behind (``a[i - K]``): a true recurrence
            # Sign undecidable even under the nonneg-symbol assumption (``K - M``): keep
            # sequential, but do NOT report 'RAW'. RAW means a *proven* read-behind, and a
            # consumer that fuses on that (FuseLoops) would then wrongly permit fusing a possible
            # read-ahead. 'complex' is the honest verdict -- every in-module consumer already
            # treats it exactly like RAW (keep sequential), so this is a no-op for renaming and
            # only tightens the fusion oracle.
            return ('complex', None)
        if loop is not None and sdfg is not None:
            arr = self._try_recognize_indirected(carried_offset, isym, loop, sdfg)
            if arr is not None:
                return ('WAR_indirected', arr)
        return ('complex', None)

    def _smt_dep_class(self, read, write, loop: LoopRegion, sdfg: SDFG, read_state, internal_syms: Set[str],
                       written: Set[str]):
        """The verdict :meth:`_dep_class` could not reach, asked of the SMT oracle.

        Only ever consulted where the affine matcher already gave up (``'complex'``), and only
        ever ANSWERS ``'WAR'`` or ``'none'``. ``'WAR'`` is a proof that no iteration up to and
        including the reader's own writes the element it reads, which is exactly what makes the
        snapshot-and-redirect rewrite value-preserving; ``'none'`` additionally proves that no
        LATER iteration writes it either, so the accesses never alias and there is nothing to
        break. Everything else stays ``'complex'``, i.e. sequential.

        Two things must be true before the query means anything, and both are enforced here
        rather than in the oracle:

        * every symbol left in the index after expanding its interstate bindings is
          loop-INVARIANT (or the iterator itself). An unexpanded loop-varying symbol would be
          one z3 variable standing for two different iterations' values, which turns the
          absence of a solution into a false proof.
        * no container the index reads is written by the loop. The oracle models a container as
          an immutable array, so a loop that writes it is outside what the encoding says.
        """
        from dace.transformation.passes.analysis import loop_analysis, smt_dependence
        from dace.transformation.passes.symbol_propagation import resolve_bindings
        if not smt_dependence.has_z3() or read_state is None:
            return ('complex', None)
        rb, wb = point_index(read), point_index(write)
        if rb is None or wb is None:
            return ('complex', None)
        ivar = loop.loop_variable
        guard = cfg_analysis.collect_enclosing_conditions(read_state, stop=loop)
        exprs = [resolve_bindings(e, sdfg, expand_data_reads=True) for e in (rb, wb, guard)]
        for e in exprs:
            if ({str(sym) for sym in e.free_symbols} - {ivar}) & internal_syms:
                return ('complex', None)
            if referenced_arrays(e) & written:
                return ('complex', None)
        rb, wb, guard = exprs
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return ('complex', None)
        read_guard = None if guard is sympy.true else guard
        ahead = smt_dependence.prove_read_ahead(rb, wb, ivar, start, end, stride, read_guard=read_guard)
        if ahead is not True:
            return ('complex', None)
        # Read-ahead only rules out the writes up to and including the reader's own iteration. When
        # no LATER iteration writes the element either, the two accesses never alias at all -- s115's
        # ``a[i] -= aa[j, i] * a[j]`` reads a[j] from below an ``i`` range that starts at j + 1. That
        # is 'none', not 'WAR': snapshotting it would pay a copy to break a dependence that is not
        # there, and the copy costs the caller an extra transient and an extra (degenerate) map.
        late = smt_dependence.prove_no_write_after_read(rb, wb, ivar, start, end, stride, read_guard=read_guard)
        return ('none', None) if late is True else ('WAR', None)

    def _walk_back_symbol_def(self, loop: LoopRegion, sym_name: str):
        """Find ``sym_name := expr`` on any interstate edge in the loop body.
        Returns the RHS string, or ``None``.

        ``loop.all_states()`` recurses into nested control-flow regions
        (LoopRegion / ConditionalBlock / etc.); a state inside a nested
        region is NOT in ``loop._nodes`` directly, so asking
        ``loop.in_edges(st)`` for such a state raises ``KeyError``. Use
        ``st.parent_graph`` instead -- each state's parent CFR knows
        about that state.
        """
        for st in loop.all_states():
            parent = st.parent_graph
            if parent is None:
                continue
            for e in parent.in_edges(st):
                if e.data is not None and sym_name in (e.data.assignments or {}):
                    return e.data.assignments[sym_name]
        return None

    def _collect_iedge_substitutions(self, loop: LoopRegion, isym=None, sdfg: Optional[SDFG] = None):
        """Build ``{sym: rhs_expr}`` for every iedge assignment in the loop
        body whose RHS is a *pure* symbolic expression (loop iterator +
        loop-invariant symbols, no array reads anywhere in the dependency
        chain). Lets the WAR matcher see memlet subsets in terms of the
        actual iterator after :class:`NormalizeNegativeStride`-style iedge
        rebindings (the post-NNS body indexes via a bound symbol
        ``i := c - k``, not the new iterator ``k`` directly).

        Crucially we EXCLUDE any binding that transitively touches a
        data-array read: the indirected-gather chain
        ``__sym := i + idx_slice ; idx_slice := idx[i]`` must NOT be
        substituted into the memlet, because :meth:`_try_recognize_indirected`
        relies on walking that chain to recognise the ``a[i + idx[i]]`` shape
        and emit the per-element array guard. Substituting would erase the
        chain by collapsing ``__sym`` to ``i + idx_slice`` (with
        ``idx_slice`` an opaque symbol), and the downstream code couldn't
        distinguish it from a benign ``WAR_symbolic`` case.

        Scope safety: we also refuse to substitute a binding whose RHS
        introduces a free symbol that is not already defined at the loop's
        SDFG scope (``sdfg.symbols`` + ``sdfg.arrays`` + ``{isym}``). A
        binding referencing an unknown name would produce an unbound symbol
        in the matcher's algebra, leading to spurious 'complex' verdicts at
        best and silent misclassification at worst.

        Returns a dict suitable for ``sympy_expr.subs(...)``.
        """
        # Symbols in scope at this loop: the iteration variable + everything
        # in sdfg.symbols + every data-array name (for array-references on
        # the RHS we keep them but mark the binding tainted further down).
        in_scope: set = set()
        if sdfg is not None:
            in_scope.update(sdfg.symbols.keys())
            in_scope.update(sdfg.arrays.keys())
        if isym is not None:
            in_scope.add(str(isym))
        # First pass: collect every iedge binding as a candidate, and note
        # which ones have an array gather (``[`` in the RHS) -- those are
        # tainted, and any binding whose RHS transitively references a tainted
        # symbol is also tainted.
        candidates = {}
        tainted_syms = set()
        for e in loop.all_interstate_edges():
            for lhs, rhs in (e.data.assignments or {}).items():
                rhs_str = str(rhs)
                if '[' in rhs_str:
                    tainted_syms.add(lhs)
                    continue
                try:
                    expr = symbolic.pystr_to_symbolic(rhs_str)
                except Exception:
                    continue
                candidates[lhs] = expr
        # Transitive taint propagation: if any candidate's RHS references a
        # tainted symbol, the candidate becomes tainted too.
        changed = True
        while changed:
            changed = False
            for lhs, expr in list(candidates.items()):
                if lhs in tainted_syms:
                    continue
                if any(str(s) in tainted_syms for s in expr.free_symbols):
                    tainted_syms.add(lhs)
                    changed = True
        # Final substitution map: untainted, non-self-referential bindings
        # whose RHS only references in-scope symbols AND mentions the loop
        # iterator. The iterator-mention requirement is what distinguishes
        # the case we want to handle (``i := N-1-k`` -- a re-expression of
        # the iterator we want inlined) from opaque renames
        # (``__sym := tasklet_output_sym`` -- the indirected-gather chain
        # ``_try_recognize_indirected`` needs to walk symbolically). Inlining
        # the latter would erase the chain and lose the WAR_indirected
        # recognition.
        subs = {}
        all_binding_names = set(candidates.keys())
        isym_str = str(isym) if isym is not None else None
        for lhs, expr in candidates.items():
            if lhs in tainted_syms:
                continue
            if symbolic.pystr_to_symbolic(lhs) in expr.free_symbols:
                continue
            unknown = [s for s in expr.free_symbols if str(s) not in in_scope and str(s) not in all_binding_names]
            if unknown:
                continue
            # Only inline when the RHS references the loop iterator; otherwise
            # the binding is an opaque rename whose substitution would lose
            # information the downstream matcher needs.
            if isym_str is not None and isym_str not in (str(s) for s in expr.free_symbols):
                continue
            subs[symbolic.pystr_to_symbolic(lhs)] = expr
        return subs

    def _try_recognize_indirected(self, offset_expr, isym, loop: LoopRegion, sdfg: SDFG) -> Optional[str]:
        """Recognise ``offset_expr == arr[isym]`` after walking back through
        interstate-edge assignments and a single ``__out = isym + Y`` tasklet
        in the loop body.

        Recognised chain (the frontend's expansion of ``a[i + idx[i]]``):

            interstate:   sym1 := scalar_name          (binds the read subset)
            tasklet:      __out = (isym + sym2)         (writes to scalar_name)
            interstate:   sym2 := arr[isym]             (the indirection)

        The carried offset is then ``arr[isym]`` and the rename is sound iff every
        element of ``arr`` is positive.

        Returns the array name if matched, ``None`` otherwise.
        """
        import ast

        isym_name = str(isym)

        # 1. offset_expr must contain exactly one non-isym free symbol.
        free = list(offset_expr.free_symbols)
        non_isym = [s for s in free if str(s) != isym_name]
        if len(non_isym) != 1:
            return None
        sym1 = non_isym[0]
        # Offset must reduce to sym1 - isym (i.e. the read is ``sym1``).
        if symbolic.simplify(offset_expr - sym1 + isym) != 0:
            return None

        # 2. Walk back: sym1 -> scalar_name (an SDFG array name, typically a
        #    transient Scalar).
        sym1_def = self._walk_back_symbol_def(loop, str(sym1))
        if sym1_def is None:
            return None
        scalar_name = sym1_def.strip()
        if scalar_name not in sdfg.arrays:
            return None

        # 3. Find the tasklet that writes ``scalar_name`` inside the loop body.
        writer_tasklet = None
        for st in loop.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == scalar_name and st.in_degree(n) > 0:
                    for e in st.in_edges(n):
                        if isinstance(e.src, nodes.Tasklet):
                            writer_tasklet = (e.src, st)
                            break
                if writer_tasklet is not None:
                    break
            if writer_tasklet is not None:
                break
        if writer_tasklet is None:
            return None
        tasklet, _ = writer_tasklet

        # 4. Parse the tasklet body: must be ``__out = (isym + Y)`` (possibly with
        #    a type cast around Y).
        try:
            tree = ast.parse((tasklet.code.as_string or "").strip())
        except SyntaxError:
            return None
        if not tree.body or not isinstance(tree.body[0], ast.Assign):
            return None
        rhs = tree.body[0].value
        if not isinstance(rhs, ast.BinOp) or not isinstance(rhs.op, ast.Add):
            return None

        # One operand must be ``isym``, the other is ``Y``. Either may be wrapped in an
        # INTEGER-CAST call (the frontend emits ``dace.int32(i) + idx_index``, so the iterator
        # side is ``dace.int32(i)``, NOT a bare ``ast.Name``); strip ONLY recognised int casts.
        # A non-cast single-arg call (``min(i, C)``, ``abs(i)``, any intrinsic) must NOT be
        # unwrapped: doing so mis-reads its argument as the bare iterator and would unsoundly
        # break an unrelated anti-dependence guarded on the wrong offset.
        int_cast_callees = frozenset(
            {'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'intc', 'intp'})

        def _strip_casts(node):
            while isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if name not in int_cast_callees or len(node.args) != 1:
                    return node  # not a recognised int cast -> leave as-is (won't match the iterator)
                node = node.args[0]
            return node

        y_node = None
        has_i = False
        for side in (rhs.left, rhs.right):
            stripped = _strip_casts(side)
            if isinstance(stripped, ast.Name) and stripped.id == isym_name:
                has_i = True
            else:
                y_node = stripped
        if not (has_i and y_node is not None):
            return None
        if not isinstance(y_node, ast.Name):
            return None
        y_name = y_node.id

        # 5. Walk back ``y_name`` to find ``arr[isym]``.
        y_def = self._walk_back_symbol_def(loop, y_name)
        if y_def is None:
            return None
        try:
            y_tree = ast.parse(y_def)
        except SyntaxError:
            return None
        if not y_tree.body or not isinstance(y_tree.body[0], ast.Expr):
            return None
        sub = y_tree.body[0].value
        if not isinstance(sub, ast.Subscript):
            return None
        if not isinstance(sub.value, ast.Name):
            return None
        arr_name = sub.value.id
        if arr_name not in sdfg.arrays:
            return None
        # The subscript must be exactly ``isym``.
        idx_part = sub.slice
        if isinstance(idx_part, ast.Index):  # py < 3.9 compatibility
            idx_part = idx_part.value
        if not (isinstance(idx_part, ast.Name) and idx_part.id == isym_name):
            return None

        # All checks passed.
        return arr_name

    @staticmethod
    def _loop_internal_symbols(loop: LoopRegion) -> Set[str]:
        """Symbols defined *within* ``loop`` -- the loop variable plus every nested
        map parameter and every nested loop variable. A symbolic carried offset
        whose free symbols intersect this set is NOT loop-invariant and the
        rename would be unsound (the read position varies inside the loop body
        in a way that may overlap the write).
        """
        internal: Set[str] = set()
        if loop.loop_variable:
            internal.add(loop.loop_variable)
        for st in loop.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry):
                    internal.update(str(p) for p in n.map.params)
        for cfr in loop.all_control_flow_regions():
            if isinstance(cfr, LoopRegion) and cfr is not loop and cfr.loop_variable:
                internal.add(cfr.loop_variable)
        return internal

    def _renamable_arrays(self, loop: LoopRegion, sdfg: SDFG):
        """Arrays in ``loop`` whose read/write pattern is a pure WAR (read-ahead)
        anti-dependence -- renamable -- and not a RAW recurrence.

        :returns: A list of ``(name, guard_exprs)`` pairs. ``guard_exprs`` is the
            set of non-numeric symbolic carried-offset expressions that must be
            asserted ``> 0`` at runtime for the rename to be sound; empty when
            the offset is a numeric positive constant.
        """
        # Subsets are DEDUPED by their ndrange -- the only thing :meth:`_dep_class`
        # ever reads off a subset -- so two subsets with the same ndrange classify
        # identically. The verdicts are consumed as a set, so dropping the
        # duplicates cannot change the outcome; it only avoids re-deriving the same
        # answer (an array touched from dozens of states otherwise blows the read x
        # write cross product up quadratically).
        reads: Dict[str, Dict[Any, Any]] = {}
        read_states: Dict[str, Dict[Any, Any]] = {}
        writes: Dict[str, Dict[Any, Any]] = {}
        for st in loop.all_states():
            for n in st.data_nodes():
                if not isinstance(sdfg.arrays.get(n.data), data.Array):
                    continue
                for e in st.out_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        sub = e.data.get_src_subset(e, st) or e.data.subset
                        reads.setdefault(n.data, {}).setdefault(_subset_key(sub), sub)
                        # The state a read sits in is what carries its branch guards, which the
                        # SMT fallback below needs and the affine matcher never looks at.
                        read_states.setdefault(n.data, {}).setdefault(_subset_key(sub), st)
                for e in st.in_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        sub = e.data.get_dst_subset(e, st) or e.data.subset
                        writes.setdefault(n.data, {}).setdefault(_subset_key(sub), sub)

        internal_syms = self._loop_internal_symbols(loop)
        # Loop-invariant: depends on the loop and its iterator only, not on the
        # subsets being classified. Computed once here instead of once per
        # (read, write) pair -- it walks every interstate edge of the loop body,
        # which dominated the pass on deeply nested SDFGs.
        iedge_subs = self._collect_iedge_substitutions(loop, symbolic.pystr_to_symbolic(loop.loop_variable), sdfg)
        written = written_data(loop)

        renamable = []
        for name, read_subsets in reads.items():
            write_subsets = writes.get(name)
            if not write_subsets:
                continue  # read-only in loop -> no anti-dependence to break
            # A single RAW / complex verdict disqualifies the array, so stop at the
            # first one rather than classifying the whole cross product.
            classes = []
            disqualified = False
            for rkey, r in read_subsets.items():
                for w in write_subsets.values():
                    c = self._dep_class(r, w, loop.loop_variable, loop=loop, sdfg=sdfg, iedge_subs=iedge_subs)
                    if c[0] == 'complex':
                        c = self._smt_dep_class(r, w, loop, sdfg, read_states[name].get(rkey), internal_syms, written)
                    if c[0] == 'RAW' or c[0] == 'complex':
                        disqualified = True
                        break
                    classes.append(c)
                if disqualified:
                    break
            if disqualified:
                continue  # true dependence (or unanalyzable) -> not sound to rename
            verdicts = {c[0] for c in classes}
            # WAR_symbolic offsets must be LOOP-INVARIANT -- free symbols may not
            # intersect any iteration variable of this loop OR of any nested
            # map / loop. Otherwise the read position varies inside the body in
            # a way that may overlap the write (e.g. offset ``-j-1`` for a
            # nested map over ``j`` is NOT a safe forward-only read).
            ok = True
            sym_guards: OrderedSet = OrderedSet()
            array_guards: OrderedSet[str] = OrderedSet()
            for kind, payload in classes:
                if kind == 'WAR_symbolic':
                    free = {str(s) for s in payload.free_symbols}
                    if free & internal_syms:
                        ok = False
                        break
                    sym_guards.add(payload)
                elif kind == 'WAR_indirected':
                    array_guards.add(payload)  # payload is the array name
            if not ok:
                continue
            if not (verdicts & {'WAR', 'WAR_symbolic', 'WAR_indirected'}):
                continue
            renamable.append((name, sym_guards, array_guards))
        return renamable

    def _emit_array_positive_guard(self, pre, arr_name: str, sdfg: SDFG) -> None:
        """Add a guard tasklet to ``pre`` that traps if any element of ``arr_name``
        is ``<= 0``. Mirrors :meth:`_emit_positive_guard` but for a per-element
        check over an array.

        Implementation: a guard tasklet calling :cpp:func:`dace::detect_all_positive`
        (``dace/runtime/include/dace/detect.h``), which folds a per-element 0/1 flag with a
        ``min`` reduction under ``omp parallel for simd``. A hand-written loop that aborted on
        the first violation would be SERIAL, and this sits right in front of the loop the
        snapshot exists to parallelize; the check that trips is a program bug, never the hot
        path, so there is nothing to gain by exiting early and a whole parallel sweep to lose.
        The index also lives inside the runtime function rather than in the tasklet body.
        """
        desc = sdfg.arrays[arr_name]
        n_str = symbolic.symstr(desc.shape[0])
        conn = f'__arr_{arr_name}'
        code = f'if (!dace::detect_all_positive({conn}, ({n_str}))) {{ std::abort(); }}'
        tlet = pre.add_tasklet(
            name=f'_break_antidep_array_guard_{arr_name}',
            inputs={conn: None},
            outputs={},
            code=code,
            language=dtypes.Language.CPP,
        )
        # No output connector, so mark side-effecting to survive dead-code
        # elimination (mirrors :meth:`_emit_positive_guard`).
        tlet.side_effects = True
        pre.add_edge(pre.add_read(arr_name), None, tlet, conn, Memlet.from_array(arr_name, desc))

    def _emit_positive_guard(self, pre, expr) -> None:
        """Add a side-effect-only tasklet to ``pre`` that traps when ``expr < 0``.

        The tasklet has zero connectors and is allowed to read free SDFG
        symbols by name (per the SDFG convention "init / symbol-only tasklets
        may have no src connectors"). Trips ``std::abort`` on violation so
        the failure is loud at runtime and does not corrupt downstream output.

        The soundness condition for the snapshot rename is ``offset >= 0`` (a
        read ahead of, or at, the write index never aliases an earlier write), so
        the guard tests ``>= 0`` -- a renamed ``+K`` offset that happens to be 0
        at runtime is still sound and must not trap. The classifier only routes
        provably-nonnegative offsets here, so in practice the guard never trips;
        it is a defensive backstop.
        """
        expr_str = symbolic.symstr(expr)
        # Not `assert(...)`: NDEBUG compiles it out. `std::abort()` is standard and faults at any
        # optimization level; SIGABRT also reads as deliberate, unlike a trap's misleading SIGILL.
        code = f'if (!(({expr_str}) >= 0)) {{ std::abort(); }}'
        # Tasklet with no input/output connectors. The CPU codegen still emits
        # its body; the symbols referenced in the code are resolved against
        # the enclosing scope.
        guard = pre.add_tasklet(
            # crc32, NOT hash(): ``hash()`` of a str is randomized per process by PYTHONHASHSEED,
            # so the same guard would get a different tasklet label, different emitted C symbols
            # and a different build hash on every run. crc32 is a stable digest of the same text.
            name=f'_break_antidep_guard_{zlib.crc32(expr_str.encode()) & 0xfffffff:x}',
            inputs={},
            outputs={},
            code=code,
            language=dtypes.Language.CPP,
        )
        # Carry no edges -- the tasklet is purely a side-effect node. Mark it
        # side-effecting so dead-code elimination cannot prune the connector-less
        # guard (which would silently restore the unsound assume-nonneg behaviour).
        guard.side_effects = True
        return guard

    def _snapshot_window(self, loop: LoopRegion, name: str, sdfg: SDFG, read_subsets) -> Optional[Memlet]:
        """The copy memlet for ``name -> snap`` restricted to the elements the
        redirected reads actually touch, or ``None`` to fall back to a whole-array copy.

        Only the redirected read edges ever source from the snapshot, so copying more
        than the image of their subsets over the loop's iteration space is pure data
        movement. The image is the standard memlet propagation
        (:func:`~dace.sdfg.propagation.propagate_subset`), which over-approximates --
        the copy can only come out too large, never too small.

        The saving is proportional, not constant: on a flat loop that sweeps a whole
        1-D array the window IS the array. It matters when the loop touches a slice --
        a blocked/tiled inner loop over ``a[t*B : (t+1)*B]`` copied the entire array
        once per outer iteration, an O(outer x N) memcpy for an O(N) sweep.

        The snapshot descriptor keeps the FULL shape so the redirected reads keep
        indexing it exactly as they indexed ``name`` -- no index rewriting, and the
        pages outside the window are never touched, so the untouched tail costs
        address space rather than memory.

        Returns ``None`` (whole array) when the loop bounds are not parseable or when
        the window would index by a symbol that only exists inside the loop -- the copy
        state sits BEFORE the loop, where such a symbol is not defined.
        """
        from dace.sdfg import propagation
        from dace.transformation.passes.analysis import loop_analysis
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None
        desc = sdfg.arrays[name]
        memlets = [Memlet(data=name, subset=copy.deepcopy(s)) for s in read_subsets]
        window = propagation.propagate_subset(memlets, desc, [loop.loop_variable],
                                              subsets.Range([(start, end, stride)])).subset
        if window is None:
            return None
        # Symbols the loop DEFINES -- its iterator, nested map/loop iterators, and anything
        # an interstate edge of the body assigns -- do not exist in the pre-loop state.
        internal = self._loop_internal_symbols(loop)
        internal.update(k for e in loop.all_interstate_edges() for k in (e.data.assignments or {}))
        if {str(s) for s in window.free_symbols} & internal:
            return None
        return Memlet(data=name, subset=copy.deepcopy(window), other_subset=copy.deepcopy(window))

    def _snapshot_and_redirect(self, loop: LoopRegion, name: str, sdfg: SDFG, guards=None, array_guards=None):
        """Insert ``snap = name`` before ``loop`` and point the loop's
        *read-ahead* reads of ``name`` at ``snap``. Also plants runtime
        positive-check guards:

        * ``guards``        -- symbolic expressions (each asserted ``> 0``).
        * ``array_guards``  -- array names (each element asserted ``> 0``).

        Both guard kinds emit a side-effect ``std::abort`` tasklet into the
        snapshot pre-state.

        Redirection is PER EDGE and restricted to strict read-ahead reads
        (``a[i + k], k > 0``). A same-index read ``a[i]`` classifies as ``none``
        (offset 0), NOT as a WAR -- and it may consume a value an *earlier state*
        of the SAME iteration wrote (an intra-iteration flow dependence, e.g. a
        later branch-body state reading the ``a[i]`` the loop just produced).
        Redirecting such a read to the pre-loop snapshot would read the stale
        original and corrupt the result, so those edges stay on the live array
        (which always holds the correct -- original or freshly written -- value
        and remains per-iteration-local, so ``LoopToMap`` still maps the loop
        once the read-ahead edges are broken). Reading a genuine read-ahead
        element off a node that was also *written* this iteration is still the
        cross-iteration original (this iteration only wrote its own index), so
        those edges are moved regardless of the node's in-degree; but an element
        that is ALSO written at the same index this iteration classifies ``none``
        against that write and is therefore left live."""
        desc = sdfg.arrays[name]
        ivar = loop.loop_variable

        # Collect every write subset of `name` in the loop body (same criterion
        # as :meth:`_renamable_arrays`) so each read edge can be classified and
        # only the strict read-ahead ones moved.
        unique_writes: Dict[Any, Any] = {}
        for st in loop.all_states():
            for n in st.data_nodes():
                if n.data != name:
                    continue
                for e in st.in_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        ws = e.data.get_dst_subset(e, st) or e.data.subset
                        if ws is not None:
                            unique_writes.setdefault(_subset_key(ws), ws)
        writes = list(unique_writes.values())
        iedge_subs = self._collect_iedge_substitutions(loop, symbolic.pystr_to_symbolic(ivar), sdfg)
        internal_syms = self._loop_internal_symbols(loop)
        written = written_data(loop)

        # Read edges to redirect: those whose subset is a strict read-ahead
        # against EVERY write (WAR / WAR_symbolic / WAR_indirected). A read that
        # is `none` (same index) or otherwise not purely read-ahead stays live.
        ahead = {'WAR', 'WAR_symbolic', 'WAR_indirected'}
        to_move = []
        # Read subsets with the same ndrange classify identically, so the verdict is
        # cached per subset instead of re-derived for every edge that carries it.
        is_ahead: Dict[Any, bool] = {}
        for st in loop.all_states():
            for n in list(st.data_nodes()):
                if n.data != name:
                    continue
                for e in st.out_edges(n):
                    if e.data is None or e.data.is_empty():
                        continue
                    rs = e.data.get_src_subset(e, st) or e.data.subset
                    if rs is None:
                        continue
                    key = _subset_key(rs)
                    verdict = is_ahead.get(key)
                    if verdict is None:
                        kinds = set()
                        for w in writes:
                            kind = self._dep_class(rs, w, ivar, loop=loop, sdfg=sdfg, iedge_subs=iedge_subs)[0]
                            if kind == 'complex':
                                kind = self._smt_dep_class(rs, w, loop, sdfg, st, internal_syms, written)[0]
                            kinds.add(kind)
                        verdict = bool(kinds) and kinds <= ahead
                        is_ahead[key] = verdict
                    if verdict:
                        to_move.append((st, e))
        if not to_move:
            return  # no genuine read-ahead edge to break -> nothing (and no snapshot)

        snap, _ = sdfg.add_transient(f'{name}_antidep_snap',
                                     desc.shape,
                                     desc.dtype,
                                     storage=desc.storage,
                                     find_new_name=True)

        # Snapshot copy `name -> snap` in a fresh state right before the loop, over the
        # window the redirected reads touch (see :meth:`_snapshot_window`).
        read_subsets = [e.data.get_src_subset(e, st) or e.data.subset for st, e in to_move]
        copy_mem = self._snapshot_window(loop, name, sdfg, read_subsets) or Memlet.from_array(name, desc)
        pre = loop.parent_graph.add_state_before(loop, label=f'{name}_snapshot')
        pre.add_nedge(pre.add_read(name), pre.add_write(snap), copy_mem)

        # Emit runtime positive-check tasklets for any symbolic guards.
        for expr in (guards or ()):
            self._emit_positive_guard(pre, expr)
        for arr_name in (array_guards or ()):
            self._emit_array_positive_guard(pre, arr_name, sdfg)

        # The snapshot is the device-neutral resolution: it costs a full copy of the read window
        # and buys unconditional parallelism. A CPU specialization has a cheaper option -- buffer
        # only the seam between chunks, where a full copy of the window is bandwidth the loop
        # itself would not have spent. A GPU has the bandwidth and would pay for the seam in
        # synchronization instead, so it keeps this form. Recorded, not decided, here.
        loop.specialization_hint = (f'anti-dependence on {name} broken by snapshotting the read window.\n'
                                    'Alternative: buffer only the seam between chunks.\n'
                                    'CPU: the seam buffer is worth trying -- the full copy is the expensive '
                                    'half here.\n'
                                    'GPU: the snapshot is usually the cheaper of the two; a seam costs '
                                    'synchronization.\n'
                                    'Both are correct. Measure before choosing.')

        # Redirect only the read-ahead edges to a fresh `snap` source, keeping any
        # destination subset (copy edges carry an `other_subset`).
        for st, e in to_move:
            snap_node = st.add_access(snap)
            new_mem = Memlet(data=snap, subset=e.data.get_src_subset(e, st) or e.data.subset)
            if isinstance(e.dst, nodes.AccessNode):
                new_mem.other_subset = e.data.get_dst_subset(e, st)
            src = e.src
            st.add_edge(snap_node, e.src_conn, e.dst, e.dst_conn, new_mem)
            st.remove_edge(e)
            if st.degree(src) == 0:
                st.remove_node(src)

    def _snapshot_forward_reads(self, loop: LoopRegion, sdfg: SDFG) -> int:
        """Break only the READ-AHEAD edges of ``loop``, leaving the array's other reads live.

        The whole-array policy above disqualifies an array the moment ONE read/write pair is a true
        dependence, because its goal is a loop that maps. This policy has a different goal: the
        MIXED shape ``A[i] = ..; d[i] = A[i] + A[i + 1]`` (TSVC s1244), where the read-ahead is the
        only thing binding two otherwise-independent statements. Moving just that edge to the
        snapshot unbinds them -- the ``A[i]`` read stays on the live array and keeps its value, and
        the loop is free to be distributed even though it stays sequential.

        Restricted to a single-compute-state body: the edge-level verdict is computed against the
        writes of that one state, so a producer chain spanning states is not analyzable here.

        :param loop: The loop to break.
        :param sdfg: The SDFG owning ``loop``.
        """
        from dace.transformation.passes.loop_fission import _single_compute_state

        state = _single_compute_state(loop)
        if state is None:
            return 0
        ivar = loop.loop_variable
        # Forward stride only. ``_dep_class`` reads direction off the sign of the carried
        # offset alone, so under a reverse stride it calls ``a[i + 1]`` read-ahead when it is
        # really the value the PREVIOUS iteration wrote -- redirecting it to the pre-loop
        # snapshot then silently computes the wrong thing.
        if not self._safe_stride(loop, sdfg):
            return 0
        internal_syms = self._loop_internal_symbols(loop)
        applied = 0

        written = sorted(
            dict.fromkeys(n.data for n in state.data_nodes()
                          if state.in_degree(n) > 0 and not sdfg.arrays[n.data].transient))
        for arr in written:
            write_subsets = []
            for n in state.data_nodes():
                if n.data != arr:
                    continue
                for e in state.in_edges(n):
                    ws = e.data.get_dst_subset(e, state) if e.data is not None else None
                    if ws is not None:
                        write_subsets.append(ws)
            if not write_subsets:
                continue

            fwd_edges = []
            sym_guards = OrderedSet()
            for n in list(state.data_nodes()):
                if n.data != arr:
                    continue
                for e in state.out_edges(n):
                    rs = e.data.get_src_subset(e, state) if e.data is not None else None
                    if rs is None:
                        continue
                    verdicts = [self._dep_class(rs, ws, ivar, loop=loop, sdfg=sdfg) for ws in write_subsets]
                    kinds = dict.fromkeys(v[0] for v in verdicts)
                    # Redirect to the pre-loop snapshot ONLY when EVERY verdict is a read-ahead
                    # (WAR / WAR_symbolic). A RAW/complex producer, OR a 'none' (offset-0, same-index
                    # producer THIS iteration), means the read consumes a value made within the sweep
                    # and must keep its live-array value -- moving it to the stale snapshot is a silent
                    # miscompile. (The old gate only skipped RAW/complex and required *some* WAR, so a
                    # read that was WAR vs one sibling write but 'none' vs another --
                    # ``A[i]=..; A[i+1]=..; d[i]=A[i+1]`` -- slipped through and read the stale value.)
                    if not (kinds and all(k in ('WAR', 'WAR_symbolic') for k in kinds)):
                        continue
                    guards = {p for k, p in verdicts if k == 'WAR_symbolic'}
                    if any(str(s) in internal_syms for g in guards for s in g.free_symbols):
                        continue
                    sym_guards |= guards
                    fwd_edges.append((n, e))
            if not fwd_edges:
                continue

            # Snapshot before the loop and redirect only the read-ahead edges to it. The copy
            # covers the window those edges read (see :meth:`_snapshot_window`), not the whole
            # array. The break itself is never declined because the copy looked expensive.
            desc = sdfg.arrays[arr]
            snap, _ = sdfg.add_transient(f'{arr}_split_snap',
                                         desc.shape,
                                         desc.dtype,
                                         storage=desc.storage,
                                         find_new_name=True)
            read_subsets = [e.data.get_src_subset(e, state) for _, e in fwd_edges]
            copy_mem = self._snapshot_window(loop, arr, sdfg, read_subsets) or Memlet.from_array(arr, desc)
            pre = loop.parent_graph.add_state_before(loop, label=f'{arr}_split_snapshot')
            pre.add_nedge(pre.add_read(arr), pre.add_write(snap), copy_mem)
            # sorted: ``sym_guards`` is a set of sympy exprs (hashed via symbol-name strings). It is iterated
            # to EMIT tasklets into ``pre``, so its order fixes their node names/ids and the emitted C order.
            for expr in sorted(sym_guards, key=symbolic.symstr):
                # STRICT (>0) guard -- NOT the >=0 the whole-array policy uses. There every read of
                # ``arr`` moves to the snapshot and a same-index read ``arr[i]`` equals the pre-loop
                # original (only iteration i writes ``arr[i]``), so a symbolic offset of 0 is sound.
                # HERE the shape is MIXED -- ``arr[i]=..; d[i]=arr[i]+arr[i+sym]`` -- so a SIBLING
                # statement writes ``arr[i]`` earlier in the SAME iteration, and a read
                # ``arr[i+sym]`` with ``sym == 0`` aliases that just-written live value and must NOT
                # be redirected to the stale snapshot. Trap unless ``sym >= 1`` (offsets are integer,
                # so ``sym - 1 >= 0`` is exactly the strict ``sym > 0``); ``sym == 0`` is then a loud
                # runtime fault instead of a silent miscompile.
                self._emit_positive_guard(pre, expr - 1)

            for src, e in fwd_edges:
                snap_node = state.add_access(snap)
                new_mem = Memlet(data=snap, subset=e.data.get_src_subset(e, state))
                if isinstance(e.dst, nodes.AccessNode):
                    new_mem.other_subset = e.data.get_dst_subset(e, state)
                state.add_edge(snap_node, e.src_conn, e.dst, e.dst_conn, new_mem)
                state.remove_edge(e)
                if state.degree(src) == 0:
                    state.remove_node(src)
            applied += 1
        return applied

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Snapshot-rename every loop with a read-ahead anti-dependence; returns the
        number of arrays renamed, or ``None``.

        ``forward_reads`` picks the admission policy (see the module docstring): the default
        whole-array pure-WAR rename, or the per-edge read-ahead break that unbinds two statements
        of a loop that stays sequential."""
        if self.forward_reads:
            applied = 0
            for loop in self._loops(sdfg):
                applied += self._snapshot_forward_reads(loop, sdfg)
            return applied or None
        renamed = 0
        for loop in self._loops(sdfg):
            if not self._safe_stride(loop, sdfg):
                continue
            n_before = renamed
            for name, sym_guards, array_guards in self._renamable_arrays(loop, sdfg):
                self._snapshot_and_redirect(loop, name, sdfg, guards=sym_guards, array_guards=array_guards)
                renamed += 1
            if renamed > n_before:
                self._forwardize_reverse_iterator(loop, sdfg)
        return renamed or None

    def _forwardize_reverse_iterator(self, loop: LoopRegion, sdfg: SDFG) -> None:
        """Inline a renamed loop's reverse-iterator binding (alpha -1 -> +1).

        A loop that :class:`NormalizeNegativeStride` reversed indexes its body via
        an iedge binding ``i := c - loop_var`` -- a body-defined symbol. That is
        what makes the access a reverse (alpha=-1) form and what blocks
        ``LoopToMap`` on the now-snapshot-renamed (hence carry-free) loop. Inlining
        the binding into the body rewrites every memlet to index via the forward
        iterator directly, so ``LoopToMap`` maps it. Scoped to *this* loop (the one
        just renamed) and to *only* the iterator re-expression bindings
        :meth:`_collect_iedge_substitutions` admits, so it does not disturb
        unrelated loops the way a whole-SDFG ``SymbolPropagation`` would. A no-op
        for genuinely forward loops (no such binding exists)."""
        isym = symbolic.pystr_to_symbolic(loop.loop_variable)
        subs = self._collect_iedge_substitutions(loop, isym, sdfg)
        if not subs:
            return
        from dace.sdfg.replace import replace_dict
        inlined = {str(k) for k in subs}
        str_repl = {str(k): f'({symbolic.symstr(v)})' for k, v in subs.items()}
        # Substitute the binding's RHS into every body memlet / tasklet so the
        # body indexes via the forward iterator directly.
        for st in loop.all_states():
            replace_dict(st, str_repl)
        # Substitute into interstate-edge conditions and other assignments' RHS
        # (``replace_keys=False`` keeps the binding's own key intact), then drop
        # the now-dead binding assignment(s).
        for e in loop.all_interstate_edges():
            e.data.replace_dict(str_repl, replace_keys=False)
            for k in list((e.data.assignments or {}).keys()):
                if k in inlined:
                    del e.data.assignments[k]
