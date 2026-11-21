# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite residual ``sympy.floor(a / b)`` into ``int_floor(a, b)``."""

from typing import Any

import sympy

from dace import SDFG, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_expr
from dace.subsets import Range, Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def normalize(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """``expr`` with every ``sympy.floor`` replaced by the equivalent ``int_floor``."""
    if not isinstance(expr, sympy.Basic):
        return expr
    floors = expr.atoms(sympy.floor)
    if not floors:
        return expr
    replacements = {}
    for node in floors:
        numerator, denominator = as_expr(sympy.together(node.args[0])).as_numer_denom()
        replacements[node] = symbolic.int_floor(numerator, denominator)
    return expr.subs(replacements)


def normalize_subset(subset: Subset | None) -> bool:
    """Normalize a subset in place; returns whether anything changed."""
    changed = False
    if isinstance(subset, Range):
        for i, dim in enumerate(subset.ranges):
            rewritten = tuple(normalize(bound) for bound in dim)
            if rewritten != tuple(dim):
                subset.ranges[i] = rewritten
                changed = True
    return changed


def normalize_descriptor(desc: dt.Data) -> int:
    """Normalize a descriptor's shape (and an array's strides/offset). Returns the rewrite count."""
    count = 0
    reshaped = tuple(normalize(dim) for dim in desc.shape)
    if reshaped != tuple(desc.shape):
        desc.shape = reshaped
        count += 1
    if isinstance(desc, dt.Array):
        rewritten_strides = [normalize(v) for v in desc.strides]
        if rewritten_strides != list(desc.strides):
            desc.strides = rewritten_strides
            count += 1
        rewritten_offset = [normalize(v) for v in desc.offset]
        if rewritten_offset != list(desc.offset):
            desc.offset = rewritten_offset
            count += 1
        rewritten_total = normalize(desc.total_size)
        if rewritten_total != desc.total_size:
            desc.total_size = rewritten_total
            count += 1
    return count


@transformation.explicit_cf_compatible
class NormalizeFloorDivision(ppl.Pass):
    """Replace ``sympy.floor(a / b)`` with ``int_floor(a, b)`` everywhere it can reach codegen."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # A new floor can only arrive with new memlets, descriptors or nodes.
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Normalize descriptors, map ranges and memlet subsets, recursively."""
        count = 0
        for sub in sdfg.all_sdfgs_recursive():
            for desc in sub.arrays.values():
                count += normalize_descriptor(desc)
            for state in sub.states():
                for node in state.nodes():
                    if isinstance(node, nodes.MapEntry) and normalize_subset(node.map.range):
                        count += 1
                for edge in state.edges():
                    if edge.data is None:
                        continue
                    count += normalize_subset(edge.data.subset)
                    count += normalize_subset(edge.data.other_subset)
        return count or None
