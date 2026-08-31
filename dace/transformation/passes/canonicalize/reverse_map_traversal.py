# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Re-express a Map whose accesses run BACKWARDS through memory so that they run forwards.

A ``Map`` is unordered by definition, so the direction its parameter sweeps carries no meaning --
but the direction its *accesses* sweep carries all the performance. A reversed source loop reaches
codegen as an ascending parameter over descending addresses:

    for i in range(N - 1, -1, -1): a[i] = b[i] + 1.0

    #pragma omp parallel for simd
    for (auto _loop_it_0 = 0; _loop_it_0 < N; _loop_it_0 += 1)
        a[((N - _loop_it_0) - 1)] = (b[((N - _loop_it_0) - 1)] + 1.0);

:mod:`~dace.transformation.passes.canonicalize.normalize_negative_stride` produces that binding on
purpose: it keeps the original ITERATION ORDER so a loop-carried recurrence still behaves. Once
``LoopToMap`` has proven there is no carried dependence and made a Map of it, the order is free and
the binding is pure cost -- every thread walks its chunk from high address to low, which defeats
the hardware prefetcher and, on a streaming kernel, costs about 30% (TSVC ``s1112`` /
``neg_stride_rev`` at 0.78x against the un-canonicalized form).

The rewrite substitutes ``p -> lo + hi - p`` in the map's SCOPE only, leaving the range alone.
That is a bijection of ``[lo, hi]`` onto itself for a unit stride, so the same index set is visited
and only the association between parameter value and address changes. It fires only when every
access is affine in the parameter and every non-zero coefficient is negative -- a map that reads
forwards and writes backwards has no good direction and is left alone.

Direction, then order: this runs before
:mod:`~dace.transformation.passes.minimize_stride_permutation`, which decides which parameter
belongs innermost by scoring unit coefficients. Orienting first means it scores the coefficients
the emitted code will actually use.
"""
from typing import List, Optional

import sympy

from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


def access_coefficients(state: SDFGState, entry: nodes.MapEntry, param: str) -> Optional[List[Optional[int]]]:
    """Sign of ``param``'s coefficient in every data access inside ``entry``'s scope.

    ``None`` when any access uses the parameter non-affinely (a modulo, a call, an indirection):
    the direction of such an access is not a property this pass can read, so the map is left alone
    rather than guessed at.

    :returns: one entry per (edge, dimension) the parameter appears in -- ``1``/``-1`` for an
        ascending/descending access, or ``None`` for the whole map if any use is non-affine.
    """
    sym = symbolic.pystr_to_symbolic(param)
    signs: List[Optional[int]] = []
    for edge in state.scope_subgraph(entry).edges():
        if edge.data is None or edge.data.is_empty():
            continue
        for subset in (edge.data.subset, edge.data.other_subset):
            if subset is None:
                continue
            for rng in getattr(subset, 'ranges', ()):
                for expr in rng:
                    if not isinstance(expr, sympy.Basic) or sym not in expr.free_symbols:
                        continue
                    coeff = expr.coeff(sym, 1)
                    # Affine iff removing the linear term leaves the parameter behind entirely.
                    if sym in coeff.free_symbols or sym in symbolic.simplify(expr - coeff * sym).free_symbols:
                        return None
                    if coeff.is_negative:
                        signs.append(-1)
                    elif coeff.is_positive:
                        signs.append(1)
                    else:
                        return None  # a sign we cannot decide is not a direction we may flip
    return signs


def reverse_descending_maps(sdfg: SDFG) -> Optional[int]:
    """Flip every map parameter whose accesses all descend, in place.

    :param sdfg: the SDFG to rewrite.
    :returns: the number of parameters flipped, or ``None`` if none were.
    """
    flipped = 0
    for g in sdfg.all_sdfgs_recursive():
        for state in g.all_states():
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                for dim, param in enumerate(entry.map.params):
                    begin, end, step = entry.map.range[dim]
                    if symbolic.simplify(step - 1) != 0:
                        continue  # lo + hi - p only re-covers the range at unit stride
                    signs = access_coefficients(state, entry, param)
                    if not signs or any(s > 0 for s in signs):
                        continue
                    state.scope_subgraph(entry).replace(param, f'({begin} + {end} - {param})')
                    flipped += 1
    return flipped or None


@xf.explicit_cf_compatible
class ReverseMapTraversal(ppl.Pass):
    """Re-express a Map whose data accesses all run backwards so that they run forwards.

    See the module docstring: a Map is unordered, so this changes no result -- only which address
    a given thread touches first.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        return reverse_descending_maps(sdfg)
