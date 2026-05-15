"""Targeted test for pointer-array member on a DUMMY-ARG struct.

Distinct from ``derived_type_test::test_struct_pointer_member_slice_rebind``
which covers Phase 5b for a LOCAL struct instance (rebound to a section
via ``s%w => src(1:n)``).  This test exercises the dummy-arg path:
``type(holder), intent(in) :: h`` with ``real, pointer :: data(:, :, :)``
inside the type, accessed from the kernel as ``h%data(i, j, k)``.

The bridge route is ``FlattenStructs::replaceStructArg`` (non-nested,
non-AoS).  Before the fix at FlattenStructs.cpp:2199 the flat
companion ``h_data`` was synthesised without ``fortran_attrs<pointer>``,
so ``extract_vars`` couldn't peel the ``fir.box<fir.ptr<>>`` wrappers
to find the inner ``SequenceType`` and classified ``h_data`` as a
Scalar of dtype ``!fir.box<!fir.ptr<...>>`` -- arglist later raised
``KeyError: 'h_data_d0'`` looking up the deferred-shape extent symbol.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_holder
  implicit none
  type :: holder_t
    real(kind = 8), pointer, contiguous :: data(:, :, :)
  end type holder_t
end module mo_holder

subroutine read_holder(h, out, n)
  use mo_holder, only: holder_t
  implicit none
  integer, intent(in) :: n
  type(holder_t), intent(in) :: h
  real(kind = 8), intent(out) :: out(n, n, n)
  integer :: i, j, k
  do k = 1, n
    do j = 1, n
      do i = 1, n
        out(i, j, k) = h % data(i, j, k)
      end do
    end do
  end do
end subroutine read_holder
"""


def test_pointer_array_member_dummy_arg_flattens(tmp_path: Path):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="read_holder", entry="_QPread_holder").build()
    sdfg.validate()

    # The flat companion must be a rank-3 Array, not a Scalar (which is
    # what the pre-fix classification produced).
    desc = sdfg.arrays.get('h_data')
    assert desc is not None, (f"expected flat companion `h_data` in SDFG; arrays: {sorted(sdfg.arrays.keys())}")
    assert type(desc).__name__ == 'Array', (f"`h_data` must classify as rank-3 Array (not Scalar) after pointer-attr "
                                            f"propagation; got {type(desc).__name__}")
    assert len(desc.shape) == 3, (f"`h_data` rank must be 3, got shape={desc.shape}")

    n = 4
    rng = np.random.default_rng(0)
    data = np.asfortranarray(rng.standard_normal((n, n, n)))
    out = np.zeros((n, n, n), dtype=np.float64, order='F')

    # Dummy-arg deferred-shape POINTER -- bridge leaves the per-dim
    # offset as a free symbol when the body has no literal-index hint.
    # Pass the actual 1-based lower bound here.
    sdfg(h_data=data,
         out=out,
         n=np.int32(n),
         h_data_d0=np.int64(n),
         h_data_d1=np.int64(n),
         offset_h_data_d0=np.int64(1),
         offset_h_data_d1=np.int64(1),
         offset_h_data_d2=np.int64(1))

    np.testing.assert_array_equal(out, data)
