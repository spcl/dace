"""Mode-C reductions (COUNT / ANY / ALL over a comparison-derived
mask) when the source array is itself a *section* of a higher-rank
array — e.g. ``m(:, pos1)`` where ``pos1`` is a scalar dim of a
rank-2 array.

The bridge's elemental walker materialises the comparison into a
transient mask, then reduces that mask.  The walker that collects
read accesses inside the mask body must walk the parent designate
chain (``expandDesignateChain``) so the access list matches the
underlying array's full rank, not just the inner element designate's
rank.  Without that walk the access for ``m(:, pos1)`` would record
``index_exprs=['ei0']`` (rank 1) against rank-2 ``m`` and produce a
malformed memlet.

These tests pin the regression: a future divergence between the
elementals.inc and control_flow.inc walkers would re-introduce the
rank mismatch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_count_parent_designate_scalar_dim(tmp_path: Path):
    """``COUNT(m(:, pos1) .eq. 5)`` — the comparison's elemental walks
    a section whose parent designate has a scalar dim (``pos1``).
    The collectReadAccesses helper must thread that scalar through to
    the underlying access."""
    src = """
SUBROUTINE count_parent_dg(m, pos1, res)
  integer, dimension(7,4) :: m
  integer :: pos1
  integer, dimension(2) :: res
  res(1) = COUNT(m(:, pos1) .eq. 5)
END SUBROUTINE count_parent_dg
"""
    sdfg = build_sdfg(src, tmp_path, name='count_parent_dg').build()

    m = np.zeros((7, 4), order='F', dtype=np.int32)
    m[2, 1] = 5
    m[5, 1] = 5
    m[3, 2] = 5
    res = np.zeros(2, order='F', dtype=np.int32)

    sdfg(m=m, pos1=2, res=res)
    assert res[0] == 2

    sdfg(m=m, pos1=3, res=res)
    assert res[0] == 1

    sdfg(m=m, pos1=4, res=res)
    assert res[0] == 0


def test_any_parent_designate_scalar_dim(tmp_path: Path):
    """``ANY(m(:, pos1) .gt. 0)`` — Mode-C ANY over a parent-section
    designate.  Same shape as the COUNT case above; pins
    ``buildElementalAnyAllReduce``'s walker."""
    src = """
SUBROUTINE any_parent_dg(m, pos1, res)
  integer, dimension(5,3) :: m
  integer :: pos1
  logical, dimension(2) :: res
  res(1) = ANY(m(:, pos1) .gt. 0)
END SUBROUTINE any_parent_dg
"""
    sdfg = build_sdfg(src, tmp_path, name='any_parent_dg').build()

    m = np.zeros((5, 3), order='F', dtype=np.int32)
    m[2, 1] = 7
    res = np.zeros(2, order='F', dtype=np.int32)

    sdfg(m=m, pos1=2, res=res)
    assert res[0] != 0

    sdfg(m=m, pos1=3, res=res)
    assert res[0] == 0


def test_all_parent_designate_scalar_dim(tmp_path: Path):
    """``ALL(m(:, pos1) .gt. 0)`` — Mode-C ALL counterpart.  All
    elements of the parent-designate slice must satisfy the
    predicate; if any don't, ALL is false."""
    src = """
SUBROUTINE all_parent_dg(m, pos1, res)
  integer, dimension(4,3) :: m
  integer :: pos1
  logical, dimension(2) :: res
  res(1) = ALL(m(:, pos1) .gt. 0)
END SUBROUTINE all_parent_dg
"""
    sdfg = build_sdfg(src, tmp_path, name='all_parent_dg').build()

    m = np.ones((4, 3), order='F', dtype=np.int32)
    res = np.zeros(2, order='F', dtype=np.int32)

    sdfg(m=m, pos1=2, res=res)
    assert res[0] != 0

    m[1, 1] = 0
    sdfg(m=m, pos1=2, res=res)
    assert res[0] == 0
