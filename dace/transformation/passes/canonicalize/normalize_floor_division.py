# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite residual ``sympy.floor(a / b)`` into ``int_floor(a, b)``.

Two things spell floor division, and only one of them is safe:

* **Parsed DaCe source** -- a ``@dace.program`` body, a memlet string, an interstate assignment --
  goes through ``pystr_to_symbolic``, which already maps ``//`` onto ``int_floor``. Writing
  ``int_floor`` there instead fails ("not registered with an SDFG implementation"), so ``//`` is
  the correct thing to write in a kernel.
* **Python code operating on sympy objects** -- transformation and pass code -- gets sympy's own
  ``__floordiv__``, i.e. ``sympy.floor(a / b)``. sympy distributes the division over the sum INSIDE
  the floor, and ``sym2cpp`` prints the argument WITHOUT the floor, so each term truncates alone::

      ((N + 1) * 4) // 8         ->  floor(N/2 + 1/2)  ->  C: (((N / 2) + (1 / 2)) / 1)  # N=3: 1
      int_floor((N + 1) * 4, 8)                        ->  C: (((4 * N) + 4) / 8)        # N=3: 2

  A bare ``i // 2`` survives by luck (C integer division coincides with floor for non-negatives),
  so this only corrupts once the numerator is a SUM -- what tiling and delinearization produce.

Call sites use ``int_floor`` directly; this pass is the net that keeps one missed ``//`` from
reaching codegen as a wrong index. Recovery is exact: ``together`` puts the distributed argument
back over a common denominator before it is split into numerator and denominator.
"""
from typing import Any, Dict, Optional, Set

import sympy

from dace import SDFG, data as dt, symbolic
from dace.sdfg import nodes
from dace.subsets import Indices, Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def normalize(expr):
    """``expr`` with every ``sympy.floor`` replaced by the equivalent ``int_floor``."""
    if not isinstance(expr, sympy.Basic):
        return expr
    floors = expr.atoms(sympy.floor)
    if not floors:
        return expr
    replacements = {}
    for node in floors:
        numerator, denominator = sympy.together(node.args[0]).as_numer_denom()
        replacements[node] = symbolic.int_floor(numerator, denominator)
    return expr.subs(replacements)


def normalize_subset(subset) -> bool:
    """Normalize a subset in place; returns whether anything changed."""
    changed = False
    if isinstance(subset, Range):
        for i, dim in enumerate(subset.ranges):
            rewritten = tuple(normalize(bound) for bound in dim)
            if rewritten != tuple(dim):
                subset.ranges[i] = rewritten
                changed = True
    elif isinstance(subset, Indices):
        rewritten = [normalize(index) for index in subset.indices]
        if rewritten != list(subset.indices):
            subset.indices = rewritten
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
        for values, assign in ((desc.strides, "strides"), (desc.offset, "offset")):
            rewritten = [normalize(v) for v in values]
            if rewritten != list(values):
                setattr(desc, assign, rewritten)
                count += 1
        rewritten_total = normalize(desc.total_size)
        if rewritten_total != desc.total_size:
            desc.total_size = rewritten_total
            count += 1
    return count


@transformation.explicit_cf_compatible
class NormalizeFloorDivision(ppl.Pass):
    """Replace ``sympy.floor(a / b)`` with ``int_floor(a, b)`` everywhere it can reach codegen."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # A new floor can only arrive with new memlets, descriptors or nodes.
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Normalize descriptors, map ranges and memlet subsets, recursively.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of expressions rewritten, or ``None`` if none were.
        """
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
