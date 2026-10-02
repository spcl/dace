# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileMaskGen`` can carry a branch guard as an extra conjunct of the iteration mask.

A branch that is neither ITE-able (two arms, same write subset) nor IT-able (one arm whose accesses
are in range regardless) has to execute UNDER A MASK: the guard becomes a per-lane predicate, and
the lanes it excludes neither store (masked ``TileScatter``) nor dereference their source (a masked
``TileGather`` guards the read). That is what makes ``if i < N-1: s += a[i+1]*a[i+1]`` safe to run
unconditionally -- lane ``i = N-1`` never touches ``a[N]``.

The mask generator previously encoded only the iteration bound. ``guard_predicate`` AND-s the guard
into the same mask, so every tile op already wired to it is masked by both.
"""
import pytest

from dace.libraries.tileops import TileMaskGen
from dace.libraries.tileops.nodes.tile_mask_gen import ExpandTileMaskGenPure

GUARD = '((i) + __l0) < (mid)'


def _mask_line(node: TileMaskGen) -> str:
    """The generated per-lane mask assignment."""
    code = ExpandTileMaskGenPure.expansion(node, None, None).code.as_string
    return next(line.strip() for line in code.splitlines() if '_o[' in line)


def test_bounds_only_mask_is_unchanged():
    """Without a guard the mask is exactly the iteration bound -- no behaviour change."""
    node = TileMaskGen('mg', widths=(8, ), iter_vars=('i', ), global_ubs=('N', ))
    assert node.guard_predicate is None
    assert _mask_line(node) == '_o[__l0] = (((i) + __l0) < (N));'


def test_guard_is_anded_into_the_mask():
    """The guard rides along as one more conjunct, per lane."""
    node = TileMaskGen('mg', widths=(8, ), iter_vars=('i', ), global_ubs=('N', ), guard_predicate=GUARD)
    assert _mask_line(node) == '_o[__l0] = (((i) + __l0) < (N)) && (((i) + __l0) < (mid));'


def test_a_guarded_mask_lowers_pure():
    """The ISA backends build the mask from the bounds alone, so they would silently DROP the guard and run every
    lane. A guarded node has no header lowering, which leaves the pure expansion, the only one that emits the extra
    conjunct. An unguarded node keeps the header lowering."""
    guarded = TileMaskGen('mg', widths=(8, ), iter_vars=('i', ), global_ubs=('N', ), guard_predicate=GUARD)
    unguarded = TileMaskGen('mg2', widths=(8, ), iter_vars=('i', ), global_ubs=('N', ))
    assert not guarded.can_lower_to_isa(None, None)
    assert unguarded.can_lower_to_isa(None, None)


def test_guard_composes_with_multi_dim_bounds():
    """Every tiled dim keeps its own bound term; the guard is appended once."""
    node = TileMaskGen('mg',
                       widths=(4, 8),
                       iter_vars=('i', 'j'),
                       global_ubs=('M', 'N'),
                       guard_predicate='((j) + __l1) != 0')
    line = _mask_line(node)
    assert '(((i) + __l0) < (M))' in line
    assert '(((j) + __l1) < (N))' in line
    assert '(((j) + __l1) != 0)' in line


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
