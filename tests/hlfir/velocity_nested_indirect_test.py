"""Velocity-tendencies nested-struct indirect-access pattern.

Distilled minimal repro of the failing memlet seen when running the
full ``mo_velocity_advection.velocity_tendencies`` through the bridge:

    p_prog%w(p_patch%edges%cell_idx(je, jb, 1), jk,
             p_patch%edges%cell_blk(je, jb, 1))

The struct ``t_patch`` carries a nested ``t_edges`` whose member
arrays (``cell_idx`` / ``cell_blk``) are themselves used as the OUTER
dim 0 / dim 2 indices into another array (``w``).  The bridge's
struct-flatten pass currently bails on dummy-arg structs whose
members include a nested record (``allMembersFlattenable`` is false)
— the nested designate chain survives into the AST extractor, and
``build_memlet_index`` produces a memlet string with a raw
sub-subscript ``arr[other[i]]``.  ``Memlet._parse_from_subexpr``
splits on the outermost ``[`` and crashes with
``ValueError: too many values to unpack (expected 2)``.

Test sizes / data shape (per user spec):
    * ``nproma = nlev = nblks = 32``
    * indirection arrays carry values in ``[1, 31]`` so every
      ``w(idx_arr(...), jk, jb)`` access is in-bounds.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_test_types
  implicit none
  integer, parameter :: nproma = 32, nblks = 32
  type :: t_edges
    integer :: cell_idx(nproma, nblks, 2)
    integer :: cell_blk(nproma, nblks, 2)
  end type
  type :: t_patch
    type(t_edges) :: edges
  end type
end module

subroutine kernel(p_patch, w, out, nlev)
  use mo_test_types
  implicit none
  integer, intent(in) :: nlev
  type(t_patch), intent(in) :: p_patch
  real(8), intent(in) :: w(nproma, nlev, nblks)
  real(8), intent(out) :: out(nproma, nlev, nblks)
  integer :: je, jk, jb
  do jb = 1, nblks
    do jk = 1, nlev
      do je = 1, nproma
        out(je, jk, jb) = w(p_patch % edges % cell_idx(je, jb, 1), jk, &
                            p_patch % edges % cell_blk(je, jb, 1))
      end do
    end do
  end do
end subroutine kernel
"""


def test_velocity_nested_struct_indirection(tmp_path: Path):
    """End-to-end numerical check on the velocity-tendencies indirect
    pattern.  Sizes match the user's request: ``nproma = nlev = nblks
    = 32``; indirection arrays carry values in ``[1, 31]``."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name='kernel').build()

    nproma, nblks, nlev = 32, 32, 32
    rng = np.random.default_rng(0)
    w = np.asfortranarray(rng.standard_normal((nproma, nlev, nblks)))
    cell_idx = np.asfortranarray(rng.integers(1, 32, size=(nproma, nblks, 2), dtype=np.int32))
    cell_blk = np.asfortranarray(rng.integers(1, 32, size=(nproma, nblks, 2), dtype=np.int32))
    out_sdfg = np.zeros((nproma, nlev, nblks), dtype=np.float64, order='F')

    # Reference: NumPy gather (Fortran 1-based → 0-based on the indirect
    # axes).  This is the exact arithmetic ``kernel`` performs for every
    # ``(je, jk, jb)``.
    cell_i = cell_idx[..., 0] - 1
    cell_b = cell_blk[..., 0] - 1
    out_ref = np.empty_like(out_sdfg)
    for jb in range(nblks):
        for jk in range(nlev):
            for je in range(nproma):
                out_ref[je, jk, jb] = w[cell_i[je, jb], jk, cell_b[je, jb]]

    # Pack the nested-struct dummy.  numpy doesn't have a native
    # Fortran-derived-type binding; for the bridge call the struct
    # arg is materialised through the flattened companions
    # (``p_patch_edges_cell_idx`` / ``_cell_blk``) once flatten-structs
    # learns the nested-dummy path.
    sdfg(p_patch_edges_cell_idx=cell_idx, p_patch_edges_cell_blk=cell_blk, w=w, out=out_sdfg, nlev=nlev)

    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)
