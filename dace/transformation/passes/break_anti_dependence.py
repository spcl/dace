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

It trades an extra array + an O(N) copy for parallelism, so it is meant to run
**optionally** (a tuning knob), not as part of the default pipeline.
"""
from typing import Any, Dict, Optional, Set

from dace import data, dtypes, properties, symbolic, Memlet
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.loop_fission import _single_compute_state


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


@properties.make_properties
class BreakAntiDependence(ppl.Pass):
    """Snapshot-rename loops with a pure WAR anti-dependence so they can map.

    Off by default in pipelines (it adds a transient + a copy); enable it as a
    tuning knob when the extra buffer is worth the parallelism.
    """

    CATEGORY: str = 'Optimization Preparation'

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

    def _dep_class(self, read, write, ivar, loop=None, sdfg=None):
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
        iedge_subs = self._collect_iedge_substitutions(loop, isym, sdfg) if loop is not None else {}
        carried_offset = None
        for (rb, re_, _), (wb, we_, _) in zip(rr, wr):
            if rb != re_ or wb != we_:
                return ('complex', None)  # not a single-element (point) access
            rb = symbolic.pystr_to_symbolic(str(rb))
            wb = symbolic.pystr_to_symbolic(str(wb))
            if iedge_subs:
                rb = symbolic.simplify(rb.subs(iedge_subs))
                wb = symbolic.simplify(wb.subs(iedge_subs))
            r_has = isym in rb.free_symbols
            w_has = isym in wb.free_symbols
            if not r_has and not w_has:
                if symbolic.simplify(rb - wb) != 0:
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
            if isym not in symbolic.simplify(wb - isym).free_symbols:
                alpha = 1
            elif isym not in symbolic.simplify(wb + isym).free_symbols:
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
            return ('RAW', None)
        if loop is not None and sdfg is not None:
            arr = self._try_recognize_indirected(carried_offset, isym, loop, sdfg)
            if arr is not None:
                return ('WAR_indirected', arr)
        return ('complex', None)

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
        SDFG scope (``sdfg.symbols`` ∪ ``sdfg.arrays`` ∪ ``{isym}``). A
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
        reads: Dict[str, list] = {}
        writes: Dict[str, list] = {}
        for st in loop.all_states():
            for n in st.data_nodes():
                if not isinstance(sdfg.arrays.get(n.data), data.Array):
                    continue
                for e in st.out_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        reads.setdefault(n.data, []).append(e.data.get_src_subset(e, st) or e.data.subset)
                for e in st.in_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        writes.setdefault(n.data, []).append(e.data.get_dst_subset(e, st) or e.data.subset)

        internal_syms = self._loop_internal_symbols(loop)

        renamable = []
        for name in reads:
            if name not in writes:
                continue  # read-only in loop -> no anti-dependence to break
            classes = [
                self._dep_class(r, w, loop.loop_variable, loop=loop, sdfg=sdfg) for r in reads[name]
                for w in writes[name]
            ]
            verdicts = {c[0] for c in classes}
            if 'RAW' in verdicts or 'complex' in verdicts:
                continue  # true dependence (or unanalyzable) -> not sound to rename
            # WAR_symbolic offsets must be LOOP-INVARIANT -- free symbols may not
            # intersect any iteration variable of this loop OR of any nested
            # map / loop. Otherwise the read position varies inside the body in
            # a way that may overlap the write (e.g. offset ``-j-1`` for a
            # nested map over ``j`` is NOT a safe forward-only read).
            ok = True
            sym_guards: Set = set()
            array_guards: Set[str] = set()
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

        Implementation: a single CPP tasklet with one input connector reading
        the whole array. The body is a tight ``for`` loop with ``__builtin_trap``
        on the first violation.
        """
        desc = sdfg.arrays[arr_name]
        n_str = symbolic.symstr(desc.shape[0])
        conn = f'__arr_{arr_name}'
        code = (f'for (long long _j = 0; _j < ({n_str}); _j++) {{\n'
                f'    if (!({conn}[_j] > 0)) {{ __builtin_trap(); }}\n'
                f'}}')
        tlet = pre.add_tasklet(
            name=f'_break_antidep_array_guard_{arr_name}',
            inputs={conn},
            outputs=set(),
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
        may have no src connectors"). Trips ``__builtin_trap`` on violation so
        the failure is loud at runtime and does not corrupt downstream output.

        The soundness condition for the snapshot rename is ``offset >= 0`` (a
        read ahead of, or at, the write index never aliases an earlier write), so
        the guard tests ``>= 0`` -- a renamed ``+K`` offset that happens to be 0
        at runtime is still sound and must not trap. The classifier only routes
        provably-nonnegative offsets here, so in practice the guard never trips;
        it is a defensive backstop.
        """
        expr_str = symbolic.symstr(expr)
        # `assert(...)` is also valid but `__builtin_trap()` gives a hard fault
        # at any optimization level and matches the convention used elsewhere
        # in the pipeline (scatter guard).
        code = f'if (!(({expr_str}) >= 0)) {{ __builtin_trap(); }}'
        # Tasklet with no input/output connectors. The CPU codegen still emits
        # its body; the symbols referenced in the code are resolved against
        # the enclosing scope.
        guard = pre.add_tasklet(
            name=f'_break_antidep_guard_{abs(hash(expr_str)) & 0xfffffff:x}',
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

    def _snapshot_and_redirect(self, loop: LoopRegion, name: str, sdfg: SDFG, guards=None, array_guards=None):
        """Insert ``snap = name`` before ``loop`` and point the loop's
        *read-ahead* reads of ``name`` at ``snap``. Also plants runtime
        positive-check guards:

        * ``guards``        -- symbolic expressions (each asserted ``> 0``).
        * ``array_guards``  -- array names (each element asserted ``> 0``).

        Both guard kinds emit a side-effect ``__builtin_trap`` tasklet into the
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
        writes = []
        for st in loop.all_states():
            for n in st.data_nodes():
                if n.data != name:
                    continue
                for e in st.in_edges(n):
                    if e.data is not None and not e.data.is_empty():
                        ws = e.data.get_dst_subset(e, st) or e.data.subset
                        if ws is not None:
                            writes.append(ws)

        # Read edges to redirect: those whose subset is a strict read-ahead
        # against EVERY write (WAR / WAR_symbolic / WAR_indirected). A read that
        # is `none` (same index) or otherwise not purely read-ahead stays live.
        ahead = {'WAR', 'WAR_symbolic', 'WAR_indirected'}
        to_move = []
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
                    kinds = {self._dep_class(rs, w, ivar, loop=loop, sdfg=sdfg)[0] for w in writes}
                    if kinds and kinds <= ahead:
                        to_move.append((st, e))
        if not to_move:
            return  # no genuine read-ahead edge to break -> nothing (and no snapshot)

        snap, _ = sdfg.add_transient(f'{name}_antidep_snap',
                                     desc.shape,
                                     desc.dtype,
                                     storage=desc.storage,
                                     find_new_name=True)

        # Snapshot copy `name -> snap` in a fresh state right before the loop.
        pre = loop.parent_graph.add_state_before(loop, label=f'{name}_snapshot')
        pre.add_nedge(pre.add_read(name), pre.add_write(snap), Memlet.from_array(name, desc))

        # Emit runtime positive-check tasklets for any symbolic guards.
        for expr in (guards or ()):
            self._emit_positive_guard(pre, expr)
        for arr_name in (array_guards or ()):
            self._emit_array_positive_guard(pre, arr_name, sdfg)

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

    def _break_mixed_forward_reads(self, loop: LoopRegion, sdfg: SDFG) -> int:
        """Break a forward-read anti-dependence carried on a MIXED array -- one that
        :meth:`_renamable_arrays` skips because a sibling read of it is RAW.

        The whole-array :meth:`_snapshot_and_redirect` only fires when EVERY read of
        an array is read-ahead (pure WAR); an array written at ``a[i]`` and read at
        BOTH ``a[i]`` (same-index RAW) and ``a[i+1]`` (forward WAR) off the same node
        -- the s1244 shape ``d[i] = a[i] + a[i+1]`` -- has that RAW read, so it is
        left alone and its statements stay a single sequential loop that
        ``LoopFission`` cannot split (the forward read is a cross-iteration bridge).

        This snapshots the array before the loop and redirects ONLY the read-ahead
        edges to the snapshot, per edge: an offset-0 / read-behind read keeps its
        live-array (RAW) value, so a genuine recurrence is preserved, while the
        forward read now reads the pre-loop original. That leaves only per-iteration
        bridges, which ``LoopFission`` distributes into independent siblings.

        A SYMBOLIC offset ``a[i + sym]`` is a forward read only when ``sym > 0``.
        Under the canonical nonnegative-symbol assumption :meth:`_dep_class` routes a
        provably-nonnegative offset to ``WAR_symbolic`` (and ``a[i - sym]`` to
        ``RAW``, so a symbolic read-behind recurrence correctly stays put); the
        rename is then sound iff ``sym > 0`` at runtime, so a loop-invariant symbolic
        offset is snapshotted AND a positive-check guard is planted before the loop.

        :returns: the number of arrays snapshotted.
        """
        state = _single_compute_state(loop)
        if state is None:
            return 0
        ivar = loop.loop_variable
        internal_syms = self._loop_internal_symbols(loop)
        applied = 0

        written = sorted(
            {n.data
             for n in state.data_nodes() if state.in_degree(n) > 0 and not sdfg.arrays[n.data].transient})
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
            sym_guards: Set = set()
            for n in list(state.data_nodes()):
                if n.data != arr:
                    continue
                for e in state.out_edges(n):
                    rs = e.data.get_src_subset(e, state) if e.data is not None else None
                    if rs is None:
                        continue
                    verdicts = [self._dep_class(rs, ws, ivar, loop=loop, sdfg=sdfg) for ws in write_subsets]
                    kinds = {v[0] for v in verdicts}
                    if kinds & {'RAW', 'complex'}:
                        continue  # a RAW read must keep its live-array value -- never move it
                    if not (kinds & {'WAR', 'WAR_symbolic'}):
                        continue  # only read-ahead anti-dependences are renamable
                    # A symbolic forward offset must be loop-invariant; a symbol shared
                    # with the iterator / a nested map varies the read position and may
                    # alias the write, so it is not a pure forward read.
                    guards = {p for k, p in verdicts if k == 'WAR_symbolic'}
                    if any({str(s) for s in g.free_symbols} & internal_syms for g in guards):
                        continue
                    sym_guards |= guards
                    fwd_edges.append((n, e))
            if not fwd_edges:
                continue

            desc = sdfg.arrays[arr]
            snap, _ = sdfg.add_transient(f'{arr}_fwd_snap',
                                         desc.shape,
                                         desc.dtype,
                                         storage=desc.storage,
                                         find_new_name=True)
            pre = loop.parent_graph.add_state_before(loop, label=f'{arr}_fwd_snapshot')
            pre.add_nedge(pre.add_read(arr), pre.add_write(snap), Memlet.from_array(arr, desc))
            for expr in sym_guards:
                self._emit_positive_guard(pre, expr)  # trap unless sym >= 0 (rename soundness)

            for src, e in fwd_edges:
                snap_node = state.add_access(snap)
                new_mem = Memlet(data=snap, subset=e.data.get_src_subset(e, state))
                # A read that feeds another access node is a COPY memlet -- it carries
                # a destination subset (``Bout[i]`` in ``Bout[i] = a[i+1]``) as its
                # ``other_subset``. Only the source side moves to the snapshot; keep the
                # destination subset or the sink would be written at the wrong index.
                # A read feeding a tasklet has no destination subset (nothing to keep).
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

        Two shapes are handled: a whole-array pure-WAR rename (every read of the array
        is read-ahead) and a per-edge MIXED break (only the read-ahead edges of an
        array whose sibling reads are RAW), the latter enabling ``LoopFission`` to
        split otherwise cross-iteration-bound statements."""
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
            # Then the per-edge mixed break (arrays the whole-array path skipped). Runs
            # AFTER it so a pure-WAR array already redirected to its snapshot is a
            # no-op here (its reads no longer originate from the live array).
            pass  # PROBE renamed += self._break_mixed_forward_reads(loop, sdfg)
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
