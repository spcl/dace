# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parallelize loops that are data-parallel only under an algebraic constraint.

Some loops are safe to map iff a symbolic quantity satisfies a side condition
that cannot be discharged at compile time. Canonicalization historically *assumed*
such conditions (e.g. symbols are nonnegative), which is unsound when the
assumption is violated at runtime. This pass instead makes the assumption
explicit by *specializing*: it replaces the loop with a two-way conditional
``if (constraint) { parallel Map } else { original sequential loop }`` (see
:func:`~dace.transformation.passes.loop_specialization.specialize_loop_under_condition`).
A runtime value that satisfies the constraint takes the parallel path; one that
violates it still computes correctly on the sequential fallback -- ``if cond:
par else: seq``, never a silent miscompile and never an abort.

This is the same conditional-parallelization shape as
:class:`~dace.transformation.passes.scatter_to_guarded_maps.ScatterToGuardedMaps`,
but the predicate here is an algebraic condition over loop symbols.

Constraint types:

* **``coeff != 0``** -- a write whose index has a *symbolic* coefficient on the
  loop variable. Two sub-shapes qualify:

  * an in-place read-modify-write (``a[i * inc] = a[i * inc] + b[i]``; TSVC
    ``s171``), and
  * a plain injective store (``dst[i * SSYM] = src[i] * scale``; the TSVC-2.5
    ``ext_strided_store_ssym`` symbolic-stride scatter *store*).

  In both cases the write is injective -- and the loop therefore data-parallel --
  exactly when that coefficient is nonzero; permissive :class:`LoopToMap` lifts
  the true branch to a plain (WCR-free, vectorizable) Map, while the else branch
  keeps the sequential loop for ``inc == 0`` / ``SSYM == 0``. A loop-carried
  recurrence (``aa[c*i] = aa[c*i - 1]``, the array read at a subset *other* than
  the one written) is deliberately excluded: a ``coeff != 0`` condition does not
  make that data-parallel.
"""
from typing import Optional, Set

from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.loop_specialization import specialize_loop_under_condition
from dace.transformation.transformation import explicit_cf_compatible


@explicit_cf_compatible
class ParallelizeUnderConstraint(ppl.Pass):
    """Specialize into ``if cond: parallel-Map else: sequential-loop`` the loops
    that are data-parallel only under a symbolic constraint."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        """Specialize every constraint-parallel loop into ``if cond: par else: seq``.

        :param sdfg: SDFG to mutate in place.
        :returns: The number of loops specialized, or ``None`` if none.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap

        specialized = 0
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
                # stride injective-write shape is a candidate. Probing every loop
                # with LoopToMap (below) is both wasteful and unsafe on shapes it
                # cannot handle (e.g. 2-D recurrences), so gate on structure first.
                condition = self._symbolic_stride_condition(loop, sd)
                if condition is None:
                    continue
                # Confirm the only blocker is the constrained condition: LoopToMap
                # refuses it outright but accepts permissively.
                probe = LoopToMap()
                probe.loop = loop
                if probe.can_be_applied(parent, 0, owner, permissive=False):
                    continue
                if not probe.can_be_applied(parent, 0, owner, permissive=True):
                    continue

                # Specialize: the true branch is the loop lifted to a plain
                # (WCR-free, vectorizable) Map -- injective under ``condition`` --
                # and the else branch keeps the original sequential loop for when
                # the condition fails at runtime.
                def _parallelize(par_loop, par_region, own):
                    inst = LoopToMap()
                    inst.loop = par_loop
                    inst.apply(par_region, own)

                specialize_loop_under_condition(loop, condition, _parallelize, owner)
                specialized += 1
        return specialized or None

    def _symbolic_stride_condition(self, loop: LoopRegion, sdfg: SDFG) -> Optional[str]:
        """The parallel-validity condition for a symbolic-stride **injective write**.

        Targets a non-transient array written at a subset with a *symbolic*
        (non-numeric) coefficient ``c`` on the loop variable, where that write is
        injective -- hence the loop data-parallel -- iff ``c != 0``. Two write
        shapes qualify:

        * the in-place read-modify-write ``a[c * i + d] = a[c * i + d] <op> ...``
          (TSVC ``s171``): the array is read and written at the *same* subset, and
        * the plain store ``dst[c * i + d] = f(i)`` (TSVC-2.5
          ``ext_strided_store_ssym``): the array is written but not read at these
          positions.

        Non-aliasing recurrences (``aa[c*i] = aa[c*i - 1] ...`` -- the array read
        at a subset *other* than the one written) are excluded: a ``c != 0``
        condition does not make a loop-carried recurrence data-parallel. Returns
        ``"(c) != 0"`` when exactly one such coefficient governs the body, else
        ``None``.

        :param loop: The candidate loop region.
        :param sdfg: The SDFG owning ``loop``'s arrays.
        :returns: The parallel-validity condition string, or ``None``.
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
        return f"({symbolic.symstr(next(iter(coeffs)))}) != 0"


__all__ = ['ParallelizeUnderConstraint']
