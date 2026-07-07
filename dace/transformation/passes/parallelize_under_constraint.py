# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parallelize loops that are data-parallel only under an algebraic constraint.

Some loops are safe to map iff a symbolic quantity satisfies a side condition
that cannot be discharged at compile time. Canonicalization historically *assumed*
such conditions (e.g. symbols are nonnegative), which is unsound when the
assumption is violated at runtime. This pass instead makes the assumption
explicit and checked: it emits a **runtime guard** -- a side-effecting CPP
tasklet that calls ``__builtin_trap()`` when the constraint is violated -- that
dominates the map, then parallelizes the loop on the assumption that the
constraint holds. The contract is "parallelize-or-abort": the result is either
the correct parallel computation or a hard trap, never a silent miscompile. (A
later post-cleanup may replace the trap with a sequential fallback so a violated
assumption degrades instead of aborting; the trap is the always-correct floor.)

This is the same guarded-parallelization shape as
:class:`~dace.transformation.passes.scatter_to_guarded_maps.ScatterToGuardedMaps`
(whose guard is a sorted duplicate-count), but the guard here is an algebraic
predicate over loop symbols.

Guard types:

* **``coeff != 0``** -- a write whose index has a *symbolic* coefficient on the
  loop variable. Two sub-shapes qualify:

  * an in-place read-modify-write (``a[i * inc] = a[i * inc] + b[i]``; TSVC
    ``s171``), and
  * a plain injective store (``dst[i * SSYM] = src[i] * scale``; the TSVC-2.5
    ``ext_strided_store_ssym`` symbolic-stride scatter *store*).

  In both cases the write is injective -- and the loop therefore data-parallel --
  exactly when that coefficient is nonzero; permissive :class:`LoopToMap` then
  lifts it to a plain (WCR-free, vectorizable) Map. The trap fires on ``inc ==
  0`` / ``SSYM == 0``. A loop-carried recurrence (``aa[c*i] = aa[c*i - 1]``, the
  array read at a subset *other* than the one written) is deliberately excluded:
  a ``coeff != 0`` guard does not make that data-parallel.

Each guarded map gets its own trap state, spliced to immediately dominate it, so
the (constraint, map) association is unambiguous when several guards coexist.
"""
from typing import Optional, Set

import dace
from dace import SDFG, dtypes, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


@explicit_cf_compatible
class ParallelizeUnderConstraint(ppl.Pass):
    """Guard-and-parallelize loops that are data-parallel only under a constraint."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        """Guard-and-parallelize every constraint-parallel loop.

        :param sdfg: SDFG to mutate in place.
        :returns: The number of loops guarded + parallelized, or ``None`` if none.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap

        guarded = 0
        for sd in sdfg.all_sdfgs_recursive():
            owner = sd
            while owner.parent_sdfg is not None:
                owner = owner.parent_sdfg
            for loop in list(sd.all_control_flow_regions(recursive=True)):
                if not (isinstance(loop, LoopRegion) and loop.loop_variable):
                    continue
                parent = loop.parent_graph
                if parent is None or loop not in parent.nodes():
                    continue
                # Cheap, read-only structural filter FIRST: only the symbolic-
                # stride in-place RMW shape is a candidate. Probing every loop with
                # LoopToMap (below) is both wasteful and unsafe on shapes it cannot
                # handle (e.g. 2-D recurrences), so gate on the structure first.
                trap_cond = self._symbolic_stride_violation(loop, sd)
                if trap_cond is None:
                    continue
                # Confirm the only blocker is the guarded condition: LoopToMap
                # refuses it outright but accepts permissively.
                probe = LoopToMap()
                probe.loop = loop
                if probe.can_be_applied(parent, 0, owner, permissive=False):
                    continue
                if not probe.can_be_applied(parent, 0, owner, permissive=True):
                    continue
                self._insert_trap_guard(parent, loop, trap_cond)
                # Parallelize on the (now trap-guarded) assumption. Permissive
                # LoopToMap lifts the injective-under-the-guard write to a plain
                # Map with no WCR -- directly vectorizable.
                inst = LoopToMap()
                inst.loop = loop
                inst.apply(parent, owner)
                guarded += 1
        return guarded or None

    def _symbolic_stride_violation(self, loop: LoopRegion, sdfg: SDFG) -> Optional[str]:
        """Runtime trap condition for a symbolic-stride **injective write**.

        Targets a non-transient array written at a subset with a *symbolic*
        (non-numeric) coefficient ``c`` on the loop variable, where that write is
        injective iff ``c != 0`` -- so the assumption is ``c != 0`` and the trap
        fires on its negation, ``c == 0``. Two write shapes qualify:

        * the in-place read-modify-write ``a[c * i + d] = a[c * i + d] <op> ...``
          (TSVC ``s171``): the array is read and written at the *same* subset, and
        * the plain store ``dst[c * i + d] = f(i)`` (TSVC-2.5
          ``ext_strided_store_ssym``): the array is written but not read at these
          positions.

        Non-aliasing recurrences (``aa[c*i] = aa[c*i - 1] ...`` -- the array read
        at a subset *other* than the one written) are excluded: a ``c != 0`` guard
        does not make a loop-carried recurrence data-parallel. Returns ``"(c) ==
        0"`` when exactly one such coefficient governs the body, else ``None``.

        :param loop: The candidate loop region.
        :param sdfg: The SDFG owning ``loop``'s arrays.
        :returns: The trap condition string, or ``None``.
        """
        loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
        # (array, subset-string) pairs read from / written to non-transient arrays.
        # Resolve descriptors against each state's OWN SDFG -- ``loop.all_states()``
        # may descend into nested SDFGs whose data names are absent from ``sdfg``.
        reads: Set = set()
        writes: dict = {}
        for state in loop.all_states():
            arrays = state.sdfg.arrays
            for e in state.edges():
                if e.data is None or e.data.is_empty() or e.data.subset is None:
                    continue
                src_desc = arrays.get(e.src.data) if isinstance(e.src, nodes.AccessNode) else None
                if src_desc is not None and not src_desc.transient:
                    reads.add((e.src.data, str(e.data.subset)))
                dst_desc = arrays.get(e.dst.data) if isinstance(e.dst, nodes.AccessNode) else None
                if dst_desc is not None and not dst_desc.transient:
                    writes[(e.dst.data, str(e.data.subset))] = e.data.subset
        reads_by_arr: dict = {}
        for rarr, rkey in reads:
            reads_by_arr.setdefault(rarr, set()).add(rkey)
        coeffs: Set = set()
        for (arr, key), subset in writes.items():
            # Admit an injective write: either a same-subset read-modify-write
            # (``key`` is among the array's read subsets) or a plain store (the
            # array is not read at all here). Refuse when the array is read at a
            # subset OTHER than the one written -- that is a loop-carried
            # recurrence (``aa[c*i] = aa[c*i - 1]``) which a ``c != 0`` guard
            # cannot parallelize.
            if reads_by_arr.get(arr, set()) - {key}:
                continue
            for rb, re_, _ in subset.ndrange():
                if rb != re_:
                    continue
                expr = symbolic.pystr_to_symbolic(rb)
                if loop_var not in expr.free_symbols:
                    continue
                coeff = expr.coeff(loop_var)
                # Symbolic (not provably numeric) nonzero coefficient only.
                if coeff == 0 or not coeff.free_symbols:
                    continue
                coeffs.add(coeff)
        if len(coeffs) != 1:
            return None
        return f"({symbolic.symstr(next(iter(coeffs)))}) == 0"

    def _insert_trap_guard(self, parent, loop: LoopRegion, trap_cond: str) -> None:
        """Splice a side-effecting trap state to immediately dominate ``loop``.

        The new state holds a connector-less CPP tasklet ``if (trap_cond) {
        __builtin_trap(); }`` over the constraint's free symbols, and inherits
        ``loop``'s incoming edges (so it runs before the loop / its lifted Map).
        Named after ``loop`` so each guarded map carries its own uniquely-tagged
        guard.

        :param parent: The control-flow region holding ``loop``.
        :param loop: The loop about to be parallelized under the guard.
        :param trap_cond: The CPP condition that, when true, traps.
        """
        was_start = parent.start_block is loop
        in_edges = list(parent.in_edges(loop))
        guard_state = parent.add_state(f'_pconstraint_guard_{loop.label}', is_start_block=was_start)
        trap = guard_state.add_tasklet(f'trap_{loop.label}', {}, {},
                                       f'if ({trap_cond}) {{ __builtin_trap(); }}',
                                       language=dtypes.Language.CPP)
        # The trap has no data connectors, so dead-code elimination would prune
        # it; mark it side-effecting so the runtime guard always survives.
        trap.side_effects = True
        for e in in_edges:
            parent.add_edge(e.src, guard_state, e.data)
            parent.remove_edge(e)
        parent.add_edge(guard_state, loop, dace.InterstateEdge())


__all__ = ['ParallelizeUnderConstraint']
