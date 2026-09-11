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


def write_subset_is_injective(write_subset: subsets.Range, params: list[str]) -> bool:
    """Write at ``write_subset`` hits a DISTINCT element per distinct enclosing ``params`` value
    -> conflict-free.

    Conservative: only the single-param affine case is decided (the ``a[c*i+d]`` in-place shape).
    Injective iff some dim is ``c*i+d`` with ``c`` a nonzero numeric constant and no dim is a
    loop-varying multi-element range. A reduction (``c == 0``) is NOT injective; a symbolic stride
    (``c`` not provably nonzero) is left to the guarded parallelization passes.

    :param write_subset: the written range.
    :param params: enclosing iteration parameters; anything but a single one is refused.
    :returns: ``True`` only when the write is provably injective.
    """
    if len(params) != 1:
        return False
    p = symbolic.pystr_to_symbolic(params[0])
    monotone_dim = False
    for (b, e, _step) in write_subset.ranges:
        be = symbolic.pystr_to_symbolic(b)
        en = symbolic.pystr_to_symbolic(e)
        depends = p in be.free_symbols or p in en.free_symbols
        if be != en:
            # Multi-element range: if loop-varying, per-iter windows may overlap -> not injective.
            if depends:
                return False
            continue
        if depends:
            slope = sympy.diff(be, p)
            if slope.is_number and slope != 0:
                monotone_dim = True
            else:
                # Non-affine, or a symbolic slope we cannot prove nonzero.
                return False
    return monotone_dim


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
    bounds = symbolic.equalize_symbols_across(*(symbolic.pystr_to_symbolic(str(bound)) for rng in write_subset.ranges
                                                for bound in rng))
    return subsets.Range([bounds[d:d + 3] for d in range(0, len(bounds), 3)])


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
                if len(bound.atoms(symbolic.Subscript)) > 0:
                    return False
        return write_subset_is_injective(equalized, [lane_var])
    except Exception:  # noqa: BLE001 -- a bound we cannot parse is not a bound we can prove
        return False
