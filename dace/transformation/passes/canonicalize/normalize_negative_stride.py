# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite negative-stride ``LoopRegion`` loops into semantically equivalent
positive-stride form.

A loop ``for i in range(start, end, -k)`` (with ``k > 0``, ``end < start``)
visits the same values of ``i`` as

    for _j in range(0, trip): i = start + (-k) * _j

where ``trip = (start - end_inclusive) // k + 1``. The body sees ``i`` having
the same per-iteration values in the same order, so downstream passes that
require positive stride (``LoopToMap``'s linear-affine subset classifier,
``LoopToScan``'s ``stride != 1`` refusal, ``RerollUnrolledLoops``) can match
without changing semantics. The loop's iteration *order* is preserved -- the
original symbol ``i`` is rebound on every iteration via an interstate-edge
assignment from the fresh positive iterator -- so loop-carried recurrences and
anti-dependences continue to behave exactly as before.

Scope:

* Loops with a literal/numeric *negative* stride. Symbolic-stride loops whose
  sign is unknown are left untouched (a runtime ``abs`` would force the
  rewrite to insert a guard, which is out of scope here).
* The loop variable, ``init``, ``end``, and stride must all be resolvable via
  the existing :mod:`dace.transformation.passes.analysis.loop_analysis` helpers.

Out of scope:

* Loops whose init / condition / update statements assign to additional
  symbols beyond the loop variable; the rewrite would have to thread those
  through the new positive-iterator form.
* While loops (no ``loop_variable``).
"""
from typing import Optional, Set

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Prefix for the fresh positive-direction iterator the rewrite introduces.
_POS_ITER_PREFIX = '_loop_pos_'


def _is_negative(value) -> bool:
    """``True`` iff ``value`` simplifies to a concrete negative number."""
    try:
        s = symbolic.simplify(value)
    except Exception:
        return False
    return s.is_number and s.is_negative


def _next_id(sdfg: SDFG) -> int:
    """Lowest ``<N>`` no existing ``_loop_pos_<N>`` symbol uses anywhere in the SDFG tree."""
    used: Set[int] = set()
    for sd in sdfg.all_sdfgs_recursive():
        for s in list(sd.symbols.keys()) + list(sd.free_symbols):
            if s.startswith(_POS_ITER_PREFIX):
                tail = s[len(_POS_ITER_PREFIX):]
                if tail.isdigit():
                    used.add(int(tail))
        for cfg in sd.all_control_flow_regions():
            if isinstance(cfg, LoopRegion) and cfg.loop_variable and cfg.loop_variable.startswith(_POS_ITER_PREFIX):
                tail = cfg.loop_variable[len(_POS_ITER_PREFIX):]
                if tail.isdigit():
                    used.add(int(tail))
    n = 0
    while n in used:
        n += 1
    return n


@properties.make_properties
@xf.explicit_cf_compatible
class NormalizeNegativeStride(ppl.Pass):
    """Rewrite negative-stride loops into positive-stride form with rebinding."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Rewrite every eligible negative-stride loop in ``sdfg`` (and nested SDFGs).

        :returns: The number of loops rewritten, or ``None`` if none.
        """
        rewritten = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                if self._try_normalize(cfg, sd):
                    rewritten += 1
        return rewritten or None

    def _try_normalize(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Rewrite one loop if its stride is a numeric negative; return whether it ran."""
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None or not _is_negative(stride):
            return False
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        if start is None or end is None:
            return False
        try:
            trip = symbolic.simplify(symbolic.int_floor(start - end, -stride) + 1)
        except Exception:
            return False

        old_var = loop.loop_variable
        new_var = f"{_POS_ITER_PREFIX}{_next_id(sdfg)}"
        # Declare the new iterator. Inherit the old variable's dtype where
        # known so downstream type-inference doesn't have to redo the work.
        new_var_dtype = sdfg.symbols.get(old_var, dace.int64)
        sdfg.add_symbol(new_var, new_var_dtype)

        # Rebind the original symbol on the *first body block* via the existing
        # loop_variable-to-body assignment edge; this preserves every reference
        # to ``old_var`` in memlets / tasklet bodies without rewriting them.
        # ``i = start + stride * new_var`` (stride is the original negative
        # value, so ``start + (-k) * new_var`` walks back through the same
        # iteration values in the same order).
        sub_expr = symbolic.symstr(symbolic.simplify(start + stride * symbolic.pystr_to_symbolic(new_var)))
        body_start = loop.start_block
        if body_start is None:
            return False
        # Add the binding to every interstate edge that enters ``body_start``
        # from inside the loop (i.e., the implicit "iteration entry" edges).
        # If no such edges exist (degenerate single-block body), splice a fresh
        # entry state that carries the binding on its in-edge.
        in_edges_into_start = list(loop.in_edges(body_start))
        if in_edges_into_start:
            for e in in_edges_into_start:
                e.data.assignments[old_var] = sub_expr
        else:
            entry = loop.add_state(f"{loop.label}_neg_inv_entry")
            loop.add_edge(entry, body_start, dace.InterstateEdge(assignments={old_var: sub_expr}))
            loop.start_block = loop.node_id(entry)

        # Rewrite the LoopRegion's iteration descriptors to drive ``new_var`` forward.
        loop.loop_variable = new_var
        loop.init_statement = dace.properties.CodeBlock(f"{new_var} = 0")
        loop.loop_condition = dace.properties.CodeBlock(f"{new_var} < ({symbolic.symstr(trip)})")
        loop.update_statement = dace.properties.CodeBlock(f"{new_var} = {new_var} + 1")
        return True


__all__ = ['NormalizeNegativeStride']
