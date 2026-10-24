# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Injectivity of a per-iteration WRITE index over ONE enclosing parameter.

Two consumers reduce to "distinct values of one parameter address distinct elements" and must
answer it the same way, so both call :func:`write_subset_is_injective`: the WCR-revert legality
check (``dace/transformation/dataflow/wcr_conversion.py``) and the vectorizer's scatter-store gate
(``map_predicates.map_body_is_tile_lowerable``).
"""

import sympy

from dace import data as dt
from dace import subsets, symbolic
from dace.sdfg.narrowing import as_basic, as_expr


def write_subset_is_injective(write_subset: subsets.Range, params: list[str], nonzero: frozenset = frozenset()) -> bool:
    """Write at ``write_subset`` hits a DISTINCT element per distinct tuple of the enclosing ``params``
    -> conflict-free.

    Conservative. Every param needs a dedicated single-element dim ``c*p+d`` with ``c`` a nonzero
    numeric constant and no other param in it: two iterations that differ in any param then differ in
    that dim. A loop-varying multi-element dim may overlap between iterations and refuses; a dim over
    several params (``i*N+j``) decides nothing on its own. A reduction (``c == 0``, or a param in no
    dim) is NOT injective; a symbolic stride ``c`` counts only when every factor of it is a nonzero
    number or a name in ``nonzero`` (see :func:`guarded_nonzero_symbols`). Symbols match by name, so an
    ``int64`` iterator and its untyped spelling are one parameter.

    :param write_subset: the written range.
    :param params: enclosing iteration parameters.
    :param nonzero: names known nonzero where the write runs.
    :returns: ``True`` only when the write is provably injective.
    """
    if not params:
        return False
    ranges = equalized_range(write_subset).ranges
    by_name = {str(sym): sym for rng in ranges for bound in rng for sym in as_basic(bound).free_symbols}
    loop_syms = {name: by_name.get(name, symbolic.pystr_to_symbolic(name)) for name in params}
    covered = set()
    for begin, end, _step in ranges:
        varying = [
            name
            for name, sym in loop_syms.items()
            if sym in as_basic(begin).free_symbols or sym in as_basic(end).free_symbols
        ]
        if begin != end:
            # Multi-element range: if loop-varying, per-iter windows may overlap -> not injective.
            if varying:
                return False
            continue
        if len(varying) != 1:
            continue
        slope = sympy.diff(begin, loop_syms[varying[0]])
        if all((f.is_number and f != 0) or (f.is_Symbol and str(f) in nonzero) for f in sympy.Mul.make_args(slope)):
            covered.add(varying[0])
    return covered == set(params)


def guarded_nonzero_symbols(block) -> frozenset:
    """The names an enclosing ``if`` branch of ``block`` asserts nonzero (``s != 0`` conjuncts of its condition).

    ``ParallelizeUnderConstraint`` lifts a symbolic-stride loop to a map only under ``inc != 0``; inside that
    branch the stride is nonzero, which the injectivity proof otherwise cannot know. A name the branch
    reassigns on an interstate edge does not keep the guard's value and is left out.

    :param block: a state or region, searched upward within its SDFG.
    :returns: the guarded names.
    """
    from dace.sdfg.state import ConditionalBlock

    names = set()
    while block is not None:
        parent = block.parent_graph
        if isinstance(parent, ConditionalBlock):
            for condition, region in parent.branches:
                if region is not block or condition is None:
                    continue
                expr = symbolic.pystr_to_symbolic(condition.as_string)
                for term in expr.args if isinstance(expr, sympy.And) else (expr,):
                    if isinstance(term, sympy.Ne):
                        lhs, rhs = term.args
                        side = lhs if rhs == 0 else rhs if lhs == 0 else None
                        if side is not None and side.is_Symbol:
                            names.add(str(side))
                reassigned = {
                    name for edge in region.all_interstate_edges(recursive=True) for name in edge.data.assignments
                }
                names -= reassigned
        block = parent
    return frozenset(names)


def equalized_range(write_subset: subsets.Range) -> subsets.Range:
    """``write_subset`` with every same-named symbol collapsed onto ONE instance, group-wise.

    A name denotes one value in an SDFG, but sympy keeps two instances of it apart whenever their
    DaCe dtype or assumptions differ, and nothing downstream raises -- the analysis just answers
    wrong. A diagonal ``a[i, i]`` nested into a body NestedSDFG propagates to a bound spelled
    ``Min(i, i)`` over an ``int64`` and an ``int`` instance of ``i``: sympy will not fold it, and
    differentiating it yields ``Heaviside(i - i)`` rather than the constant 1 the affine test wants.
    Equalized as a GROUP, so a dim's begin and end stay comparable to each other.

    :param write_subset: the range to normalize.
    :returns: an equivalent range whose bounds are parsed sympy expressions over merged symbols.
    """
    bounds = symbolic.equalize_symbols_across(*(as_expr(str(bound)) for rng in write_subset.ranges for bound in rng))
    return subsets.Range([bounds[d : d + 3] for d in range(0, len(bounds), 3)])


def scatter_write_is_injective(write_subset: subsets.Range, lane_var: str, desc: dt.Data) -> bool:
    """True when concurrent tile lanes of ``lane_var`` provably write DISTINCT elements of ``desc``.

    Two preconditions the WCR consumer does not need: a ``dt.Array`` maps distinct in-bounds index
    TUPLES to distinct elements while a ``View`` may alias its strides, and a data-dependent begin
    (``a[idx[i]]``) is an author assertion rather than a proof.

    Soundness: let ``d`` be the dimension :func:`write_subset_is_injective` found separating. Its
    index is ``c*lane + off`` with ``c`` a nonzero integer and ``off`` free of ``lane``, so two
    lanes differ in position ``d`` over the integers and the boxes they write are disjoint. No tile
    width, trip count or array bound enters, so the verdict holds for every lane window.

    :param write_subset: the per-lane write subset.
    :param lane_var: the ONE widened map parameter, spelled as ``write_subset`` spells it.
    :param desc: the destination data descriptor.
    :returns: ``True`` only on a proof; ``False`` keeps the caller closed.
    """
    if not isinstance(desc, dt.Array) or isinstance(desc, dt.View):
        return False
    try:
        equalized = equalized_range(write_subset)
        for beg, end, _step in equalized.ranges:
            for bound in (beg, end):
                if len(as_basic(bound).atoms(symbolic.Subscript)) > 0:
                    return False
        return write_subset_is_injective(equalized, [lane_var])
    except Exception:  # noqa: BLE001 -- a bound we cannot parse is not a bound we can prove
        return False
