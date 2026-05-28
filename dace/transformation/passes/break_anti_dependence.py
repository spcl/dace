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

    def _dep_class(self, read, write, ivar):
        """Classify the dependence between a read and a write subset (both affine
        point accesses) of one array under unit-stride iteration of ``ivar``.

        :returns: One of:

            * ``('WAR', None)``           -- read-ahead anti-dep with constant +offset
            * ``('WAR_symbolic', expr)``  -- carried offset is a non-numeric symbolic
              expression ``expr`` independent of the iteration variable; only sound
              to rename if ``expr > 0`` at runtime (the caller emits the guard).
            * ``('RAW', None)``           -- read-behind true dep (sequential)
            * ``('none', None)``          -- never alias or same element
            * ``('complex', None)``       -- give up
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
            if symbolic.simplify(rb - isym).free_symbols & {isym} or symbolic.simplify(wb - isym).free_symbols & {isym}:
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
        # Symbolic offset: independent of the iteration variable (already checked
        # above) and not a numeric constant. We can rename UNDER the assumption
        # the offset is positive; the caller plants a runtime guard.
        if isym in carried_offset.free_symbols:
            return ('complex', None)
        return ('WAR_symbolic', carried_offset)

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

        renamable = []
        for name in reads:
            if name not in writes:
                continue  # read-only in loop -> no anti-dependence to break
            classes = [self._dep_class(r, w, loop.loop_variable) for r in reads[name] for w in writes[name]]
            verdicts = {c[0] for c in classes}
            if 'RAW' in verdicts or 'complex' in verdicts:
                continue  # true dependence (or unanalyzable) -> not sound to rename
            if 'WAR' not in verdicts and 'WAR_symbolic' not in verdicts:
                continue
            guards: Set = set()
            for kind, payload in classes:
                if kind == 'WAR_symbolic' and payload is not None:
                    guards.add(payload)
            renamable.append((name, guards))
        return renamable

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

    def _snapshot_and_redirect(self, loop: LoopRegion, name: str, sdfg: SDFG, guards=None):
        """Insert ``snap = name`` before ``loop`` and point the loop's reads of
        ``name`` at ``snap``. If ``guards`` is non-empty, also emit a runtime
        positive-check tasklet (per expression) into the same pre-state -- the
        rename is only sound when each guarded expression is ``> 0``."""
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
            for name, guards in self._renamable_arrays(loop, sdfg):
                self._snapshot_and_redirect(loop, name, sdfg, guards=guards)
                renamed += 1
        return renamed or None
