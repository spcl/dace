# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RelaxIntegerPowers`` -- lower ``base ** exp`` to ``ipow`` where the exponent
is a non-negative integer.

DaCe's symbolic C++ printer emits ``dace::math::pow`` (libm, ``double``) for a
non-constant exponent, which is illegal where an integer is required -- an array
size, a subscript, or a loop bound.  ``dace::math::ipow`` (repeated multiply,
exact integer) is correct there iff the exponent is a non-negative integer (its
C++ exponent is ``unsigned``, so a negative one is catastrophic).

This pass rewrites ``Pow(base, exp) -> ipow(base, exp)`` whenever ``exp`` is
provably a non-negative integer:

* a non-negative integer constant (``3``) or an integer-valued float constant
  (``2.0``, ``3.0``); or
* a symbolic integer expression proven ``>= 0`` by interval analysis over the
  enclosing loop / map iterator ranges -- e.g. ``K - i - 1`` with the loop
  ``for i in range(K)`` giving ``i in [0, K-1]``, so the exponent bottoms out at
  ``0``.

A provably-negative exponent (a genuine reciprocal) is left on the ``pow`` path.
See :class:`dace.symbolic.ipow`.
"""
from typing import Any, Dict, Iterator, Optional, Tuple

import sympy

from dace import SDFG, data, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.symbolic import ipow
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.analysis import loop_analysis

#: A per-symbol inclusive iteration range ``name -> (low, high)``.  Keyed by
#: symbol *name* (not object) so a mismatch in SymPy assumptions between the
#: iterator declaration and its use in an exponent cannot hide the range.
_RangeMap = Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]]


def _iterator_ranges(sdfg: SDFG) -> _RangeMap:
    """Collect inclusive ``[low, high]`` ranges of every loop / map iterator.

    A symbol seen with two different ranges (e.g. the map parameter ``__i0``
    reused by unrelated maps) is dropped -- an ambiguous bound is no bound.
    """
    ranges: _RangeMap = {}
    dropped = set()

    def add(name, low, high):
        key = str(name)
        if key in dropped:
            return
        val = (low, high)
        if key in ranges and ranges[key] != val:
            del ranges[key]
            dropped.add(key)
            return
        ranges[key] = val

    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    for param, (low, high, _step) in zip(node.map.params, node.map.range.ranges):
                        add(param, low, high)  # DaCe map ranges are inclusive
        for region in sd.all_control_flow_regions():
            if not isinstance(region, LoopRegion) or not region.loop_variable:
                continue
            start = loop_analysis.get_init_assignment(region)
            end = loop_analysis.get_loop_end(region)  # inclusive
            if start is None or end is None:
                continue
            step = loop_analysis.get_loop_stride(region)
            if step is not None and step.is_negative:
                add(region.loop_variable, end, start)
            else:
                add(region.loop_variable, start, end)
    return ranges


def _proven_nonnegative(exp: sympy.Expr, ranges: _RangeMap) -> bool:
    """Is ``exp`` provably ``>= 0``?

    First trust SymPy's own assumptions (covers ``R**K`` / ``R**(K-1)`` when the
    base symbols are positive).  Otherwise minimise ``exp`` over the iterator
    box: for an exponent affine in each iterator, its minimum sits at a corner
    -- the iterator's high end where its coefficient is negative, its low end
    otherwise -- so substitute those corners and test the residual's sign.
    """
    if exp.is_nonnegative:
        return True
    if exp.is_negative:
        return False

    # The exponent's symbols carry the SDFG's real assumptions; a range bound
    # parsed out of a loop condition (via ``loop_analysis``) may spell the same
    # symbol with different assumptions, so ``K_exp - K_loop`` would not cancel.
    # Rebind range-bound symbols onto the exponent's like-named symbols first.
    canonical = {s.name: s for s in exp.free_symbols}

    def rebind(bound):
        if not symbolic.issymbolic(bound):
            return bound
        return bound.subs({s: canonical[s.name] for s in bound.free_symbols if s.name in canonical})

    corners = {}
    for sym in exp.free_symbols:
        if sym.name not in ranges:
            continue  # a free symbol we cannot bound stays in the residual
        coeff = sympy.diff(exp, sym)
        if not coeff.is_number:
            return False  # non-affine in an iterator -> no simple corner minimum
        low, high = ranges[sym.name]
        corners[sym] = rebind(high) if coeff.is_negative else rebind(low)
    if not corners:
        return False  # not provable by assumptions and no iterator to bound
    return exp.subs(corners).is_nonnegative is True


def _authoritative_symbols(sdfg: SDFG) -> Dict[str, sympy.Symbol]:
    """Map each symbol name to its strongest sign assumption seen anywhere in the
    SDFG.

    DaCe loses a symbol's assumptions across some copies -- the same ``K``
    declared ``positive`` on an array shape can reappear sign-less on a map
    range, so ``R**K`` would relax in one place and not the other.  A symbol
    declared positive/nonnegative *anywhere* holds that everywhere (it is one
    logical symbol), so this rebinds every occurrence to the strongest witness.
    """
    auth: Dict[str, sympy.Symbol] = {}
    for expr in _size_expressions(sdfg):
        for sym in expr.free_symbols:
            if not isinstance(sym, sympy.Symbol):
                continue
            current = auth.get(sym.name)
            if current is None:
                auth[sym.name] = sym
            elif sym.is_positive and not current.is_positive:
                auth[sym.name] = sym
            elif sym.is_nonnegative and not current.is_nonnegative and not current.is_positive:
                auth[sym.name] = sym
    return auth


def _relaxed_exponent(exp: sympy.Expr, ranges: _RangeMap, auth: Dict[str, sympy.Symbol]) -> Optional[sympy.Expr]:
    """Return the integer exponent to feed ``ipow``, or ``None`` to keep ``pow``."""
    if exp.is_Number:
        if exp.is_integer:
            value = int(exp)
        elif exp.is_real and float(exp) == int(float(exp)):
            value = int(float(exp))  # integer-valued float literal (2.0 -> 2)
        else:
            return None  # genuinely fractional (0.5 -> sqrt) -> not an integer power
        return sympy.Integer(value) if value >= 0 else None  # negative -> reciprocal
    # Prove on the assumption-canonicalized exponent, but emit ``ipow`` with the
    # original symbols so it matches its surrounding expression.
    canon = exp.xreplace({s: auth[s.name] for s in exp.free_symbols if s.name in auth})
    if canon.is_integer and _proven_nonnegative(canon, ranges):
        return exp
    return None


def _subset_exprs(sub) -> Iterator[sympy.Expr]:
    """Yield the symbolic components of a memlet subset."""
    if isinstance(sub, subsets.Range):
        for rng in sub.ranges:
            yield from rng
    elif isinstance(sub, subsets.Indices):
        yield from sub.indices


def _size_expressions(sdfg: SDFG) -> Iterator[sympy.Expr]:
    """Yield every symbolic size / subscript / bound expression (read-only)."""
    for sd in sdfg.all_sdfgs_recursive():
        for desc in sd.arrays.values():
            for seq in (desc.shape, desc.strides, desc.offset):
                for item in seq:
                    if symbolic.issymbolic(item):
                        yield item
            if symbolic.issymbolic(desc.total_size):
                yield desc.total_size
        for state in sd.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    for item in _subset_exprs(node.map.range):
                        if symbolic.issymbolic(item):
                            yield item
            for edge in state.edges():
                if edge.data is None:
                    continue
                for sub in (edge.data.subset, edge.data.other_subset):
                    if sub is not None:
                        for item in _subset_exprs(sub):
                            if symbolic.issymbolic(item):
                                yield item


def _relax_subset(sub, relax) -> None:
    """Rewrite a memlet subset's components in place with ``relax`` (expr -> expr)."""
    if isinstance(sub, subsets.Range):
        sub.ranges = [tuple(relax(component) for component in rng) for rng in sub.ranges]
    elif isinstance(sub, subsets.Indices):
        sub.indices = [relax(idx) for idx in sub.indices]


@transformation.explicit_cf_compatible
class RelaxIntegerPowers(ppl.Pass):
    """Lower non-negative-integer ``Pow`` to ``ipow`` across the SDFG's size,
    subscript and bound expressions."""

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        # Idempotent: an ``ipow`` is never re-relaxed.
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        # Fast out (this pass runs on every SimplifyPass iteration): nothing to do
        # without a power to relax.
        if not any(expr.has(sympy.Pow) for expr in _size_expressions(sdfg)):
            return None
        ranges = _iterator_ranges(sdfg)
        auth = _authoritative_symbols(sdfg)
        relaxed = [0]

        def relax(expr):
            """Rewrite each provable ``Pow`` in ``expr`` to ``ipow`` (SymPy-native)."""
            if not symbolic.issymbolic(expr):
                return expr

            def to_ipow(base, exp):
                new_exp = _relaxed_exponent(exp, ranges, auth)
                if new_exp is None:
                    return base**exp
                relaxed[0] += 1
                return ipow(base, new_exp)

            return expr.replace(sympy.Pow, to_ipow)

        for sd in sdfg.all_sdfgs_recursive():
            for desc in sd.arrays.values():
                if not isinstance(desc, data.Array):
                    continue  # Scalar / Stream have no settable shape / strides / offset
                desc.shape = tuple(relax(item) for item in desc.shape)
                desc.strides = tuple(relax(item) for item in desc.strides)
                desc.offset = tuple(relax(item) for item in desc.offset)
                desc.total_size = relax(desc.total_size)
            for state in sd.all_states():
                for node in state.nodes():
                    if isinstance(node, nodes.MapEntry):
                        _relax_subset(node.map.range, relax)
                for edge in state.edges():
                    if edge.data is not None:
                        for sub in (edge.data.subset, edge.data.other_subset):
                            if sub is not None:
                                _relax_subset(sub, relax)
        return relaxed[0] or None


__all__ = ['RelaxIntegerPowers']
