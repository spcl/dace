# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite negative-stride ``LoopRegion`` loops into semantically equivalent positive-stride form."""

from typing import Any

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import ConditionalBlock, LoopRegion, SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.fresh_names import lowest_free_suffix

#: Prefix for the fresh positive-direction iterator the rewrite introduces.
POS_ITER_PREFIX = "_loop_pos_"


def is_negative_number(value: symbolic.SymbolicType, facts: symbolic.Facts) -> bool:
    """``True`` iff ``facts`` prove ``value`` negative."""
    relation = symbolic.predicate_relation(symbolic.Predicate.NEGATIVE, as_expr(value))
    return symbolic.ask(relation, facts) is symbolic.Truth.TRUE


def read_after_loop(loop: LoopRegion, name: str) -> bool:
    """Whether a block or interstate edge that may run after ``loop`` reads ``name``."""
    block, region = loop, loop.parent_graph
    while True:
        if isinstance(region, ConditionalBlock):
            block, region = region, region.parent_graph
            continue
        if isinstance(region, LoopRegion):
            # The enclosing loop's header and next iteration run after ``loop`` as well.
            return region.loop_variable == name or name in region.free_symbols
        reached = list(region.bfs_nodes(block))
        if any(name in b.free_symbols for b in reached if b is not block):
            return True
        if any(name in e.data.free_symbols for b in reached for e in region.out_edges(b)):
            return True
        if isinstance(region, SDFG):
            return False
        block, region = region, region.parent_graph


@properties.make_properties
@xf.explicit_cf_compatible
class NormalizeNegativeStride(ppl.Pass):
    """Rewrite negative-stride loops into positive-stride form with rebinding."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Rewrite every eligible negative-stride loop in ``sdfg`` (and nested SDFGs)."""
        rewritten = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                if self._try_normalize(cfg, sd):
                    rewritten += 1
        return rewritten or None

    def _try_normalize(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Rewrite one loop if its stride is provably negative; return whether it ran."""
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None:
            return False
        # A literal stride (almost every loop's ``1``) needs no facts to read its sign
        if stride.is_number:
            if not stride < 0:
                return False
        elif not is_negative_number(stride, SymbolResolver().facts_at(loop)):
            return False
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        # The rebinding ``i = start + stride * j`` reads ``start`` in every iteration
        if start is None or end is None or loop_analysis.body_writes(loop, start):
            return False
        # The rewrite leaves the old iterator at its last body value (unset on a zero-trip loop) instead of the first
        # failing value, so a later read of it would change.
        if read_after_loop(loop, loop.loop_variable):
            return False
        try:
            trip = symbolic.simplify(symbolic.int_floor(start - end, -stride) + 1)
        except Exception:
            return False

        old_var = loop.loop_variable
        new_var = f"{POS_ITER_PREFIX}{lowest_free_suffix(sdfg, (POS_ITER_PREFIX,), with_free_symbols=True)}"
        # Rebind the original symbol on the *first body block* via the existing loop_variable-to-body assignment edge
        sub_expr = symbolic.symstr(symbolic.simplify(start + stride * symbolic.pystr_to_symbolic(new_var)))
        body_start = loop.start_block
        if body_start is None:
            return False
        # Add the binding to every interstate edge that enters ``body_start`` from inside the loop (i.e., the implicit
        # "iteration entry" edges).
        in_edges_into_start = list(loop.in_edges(body_start))
        if in_edges_into_start:
            for e in in_edges_into_start:
                e.data.assignments[old_var] = sub_expr
        else:
            entry = loop.add_state(f"{loop.label}_neg_inv_entry", is_start_block=True)
            loop.add_edge(entry, body_start, dace.InterstateEdge(assignments={old_var: sub_expr}))

        # Rewrite the LoopRegion's iteration descriptors to drive ``new_var`` forward.
        loop.loop_variable = new_var
        loop.init_statement = dace.properties.CodeBlock(f"{new_var} = 0")
        loop.loop_condition = dace.properties.CodeBlock(f"{new_var} < ({symbolic.symstr(trip)})")
        loop.update_statement = dace.properties.CodeBlock(f"{new_var} = {new_var} + 1")
        return True


__all__ = ["NormalizeNegativeStride"]
