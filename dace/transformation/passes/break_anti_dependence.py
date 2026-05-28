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
from typing import Any, Dict, List, Optional, Set

from dace import data, properties, symbolic, Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl


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

    def _unit_stride(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        from dace.transformation.passes.analysis import loop_analysis
        stride = loop_analysis.get_loop_stride(loop)
        try:
            return stride is not None and int(symbolic.evaluate(stride, sdfg.constants)) == 1
        except (TypeError, ValueError):
            return False

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
            * ``('none', None)``            -- never alias or same element
            * ``('complex', None)``         -- give up
        """
        isym = symbolic.pystr_to_symbolic(ivar)
        rr, wr = list(read.ndrange()), list(write.ndrange())
        if len(rr) != len(wr):
            return ('complex', None)
        carried_offset = None
        for (rb, re_, _), (wb, we_, _) in zip(rr, wr):
            if rb != re_ or wb != we_:
                return ('complex', None)  # not a single-element (point) access
            rb = symbolic.pystr_to_symbolic(str(rb))
            wb = symbolic.pystr_to_symbolic(str(wb))
            r_has = isym in rb.free_symbols
            w_has = isym in wb.free_symbols
            if not r_has and not w_has:
                if symbolic.simplify(rb - wb) != 0:
                    return ('none', None)  # different fixed index -> never alias
                continue
            # carried dimension: require coefficient 1 on the iteration variable
            # in the WRITE (the read may contain a data-indirected term we resolve
            # below). The write's free symbols outside of ``isym`` (after
            # subtracting ``isym``) must be empty.
            if symbolic.simplify(wb - isym).free_symbols & {isym}:
                return ('complex', None)
            if carried_offset is not None:
                return ('complex', None)  # more than one carried dimension
            carried_offset = symbolic.simplify(rb - wb)
        if carried_offset is None:
            return ('none', None)  # loop-invariant read of the array, not our case
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
            return ('WAR_symbolic', carried_offset)
        if loop is not None and sdfg is not None:
            arr = self._try_recognize_indirected(carried_offset, isym, loop, sdfg)
            if arr is not None:
                return ('WAR_indirected', arr)
        return ('complex', None)

    def _walk_back_symbol_def(self, loop: LoopRegion, sym_name: str):
        """Find ``sym_name := expr`` on any interstate edge in the loop body.
        Returns the RHS string, or ``None``."""
        for st in loop.all_states():
            for e in loop.in_edges(st):
                if e.data is not None and sym_name in (e.data.assignments or {}):
                    return e.data.assignments[sym_name]
        return None

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
        from dace.sdfg import nodes as _nd

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
                if isinstance(n, _nd.AccessNode) and n.data == scalar_name and st.in_degree(n) > 0:
                    for e in st.in_edges(n):
                        if isinstance(e.src, _nd.Tasklet):
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
        # One operand must be ``isym``, the other is ``Y``.
        y_node = None
        has_i = False
        for side in (rhs.left, rhs.right):
            if isinstance(side, ast.Name) and side.id == isym_name:
                has_i = True
            else:
                y_node = side
        if not (has_i and y_node is not None):
            return None
        # Strip leading type casts: ``dace.int64(idx_index)`` -> ``idx_index``.
        while isinstance(y_node, ast.Call):
            if not y_node.args:
                return None
            y_node = y_node.args[0]
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
        from dace.sdfg.state import LoopRegion as _LR
        internal: Set[str] = set()
        if loop.loop_variable:
            internal.add(loop.loop_variable)
        for st in loop.all_states():
            for n in st.nodes():
                from dace.sdfg import nodes as _nd
                if isinstance(n, _nd.MapEntry):
                    internal.update(str(p) for p in n.map.params)
        for cfr in loop.all_control_flow_regions():
            if isinstance(cfr, _LR) and cfr is not loop and cfr.loop_variable:
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
        from dace import dtypes as _dt
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
            language=_dt.Language.CPP,
        )
        pre.add_edge(pre.add_read(arr_name), None, tlet, conn, Memlet.from_array(arr_name, desc))

    def _emit_positive_guard(self, pre, expr) -> None:
        """Add a side-effect-only tasklet to ``pre`` that traps when ``expr <= 0``.

        The tasklet has zero connectors and is allowed to read free SDFG
        symbols by name (per the SDFG convention "init / symbol-only tasklets
        may have no src connectors"). Trips ``__builtin_trap`` on violation so
        the failure is loud at runtime and does not corrupt downstream output.
        """
        from dace import dtypes as _dt
        expr_str = symbolic.symstr(expr)
        # `assert(...)` is also valid but `__builtin_trap()` gives a hard fault
        # at any optimization level and matches the convention used elsewhere
        # in the pipeline (scatter guard).
        code = f'if (!(({expr_str}) > 0)) {{ __builtin_trap(); }}'
        # Tasklet with no input/output connectors. The CPU codegen still emits
        # its body; the symbols referenced in the code are resolved against
        # the enclosing scope.
        guard = pre.add_tasklet(
            name=f'_break_antidep_guard_{abs(hash(expr_str)) & 0xfffffff:x}',
            inputs={},
            outputs={},
            code=code,
            language=_dt.Language.CPP,
        )
        # Carry no edges -- the tasklet is purely a side-effect node.
        return guard

    def _snapshot_and_redirect(self, loop: LoopRegion, name: str, sdfg: SDFG, guards=None, array_guards=None):
        """Insert ``snap = name`` before ``loop`` and point the loop's reads of
        ``name`` at ``snap``. Also plants runtime positive-check guards:

        * ``guards``        -- symbolic expressions (each asserted ``> 0``).
        * ``array_guards``  -- array names (each element asserted ``> 0``).

        Both guard kinds emit a side-effect ``__builtin_trap`` tasklet into the
        snapshot pre-state."""
        desc = sdfg.arrays[name]
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

        # Redirect every pure read of `name` inside the loop body to `snap`.
        for st in loop.all_states():
            for n in list(st.data_nodes()):
                if n.data != name or st.in_degree(n) != 0:
                    continue  # only pure-read sources (writes stay on `name`)
                n.data = snap
                for e in st.out_edges(n):
                    if e.data is not None and e.data.data == name:
                        e.data.data = snap

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Snapshot-rename every loop with a pure WAR anti-dependence; returns the
        number of arrays renamed, or ``None``."""
        renamed = 0
        for loop in self._loops(sdfg):
            if not self._unit_stride(loop, sdfg):
                continue
            for name, sym_guards, array_guards in self._renamable_arrays(loop, sdfg):
                self._snapshot_and_redirect(loop, name, sdfg, guards=sym_guards, array_guards=array_guards)
                renamed += 1
        return renamed or None
