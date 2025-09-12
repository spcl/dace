# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_struct():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type test_type
    integer :: start
    integer :: end
  end type
end module lib

subroutine main(res, startidx, endidx)
  use lib
  implicit none
  integer, dimension(6) :: res
  integer :: startidx
  integer :: endidx
  type(test_type) :: indices
  indices%start = startidx
  indices%end = endidx
  call fun(res, indices)
end subroutine main

subroutine fun(res, idx)
  use lib
  implicit none
  integer, dimension(6) :: res
  type(test_type) :: idx
  res(idx%start:idx%end) = 42
end subroutine fun
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, entry_point='main', normalize_offsets=False)
    sdfg.save('before.sdfg')
    sdfg.simplify(verbose=True)
    sdfg.save('after.sdfg')
    sdfg.compile()

    size = 6
    res = np.full([size], 42, order="F", dtype=np.int32)
    res[:] = 0
    sdfg(res=res, startidx=2, endidx=5)
    print(res)


if __name__ == "__main__":
    test_fortran_struct()
