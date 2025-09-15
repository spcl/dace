# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace import dtypes, symbolic
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import AccessNode
from dace.sdfg.state import LoopRegion
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_fortran_frontend_array_access():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(4)
  d(2) = 5.5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.5, 42, 42])


def test_fortran_frontend_array_ranges():
    """
    Tests that the Fortran frontend can parse multidimenstional arrays with vectorized ranges and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(3, 4, 5), e(3, 4, 5), f(3, 4, 5)
  e(:, :, :) = 1.0
  f(:, :, :) = 2.0
  f(:, 2:4, :) = 3.0
  f(1, 1, :) = 4.0
  d(:, :, :) = e(:, :, :) + f(:, :, :)
  d(1, 2:4, 1) = e(1, 2:4, 1)*10.0
  d(1, 1, 1) = sum(e(:, 1, :))
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 15)
    assert (d[0, 1, 0] == 10)
    assert (d[1, 0, 0] == 3)
    assert (d[2, 3, 3] == 4)
    assert (d[0, 0, 2] == 5)


def test_fortran_frontend_array_multiple_ranges_with_symbols():
    """
    Tests that the Fortran frontend can parse multidimenstional arrays with vectorized ranges and that the accessed indices are correct.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(a, lu, iend, m)
  integer, intent(in) :: iend, m
  double precision, intent(inout) :: a(iend, m, m), lu(iend, m, m)
  lu(1:iend,1:m,1:m) = a(1:iend,1:m,1:m)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    sdfg.compile()

    iend, m = 3, 4
    lu = np.full([iend, m, m], 0, order="F", dtype=np.float64)
    a = np.full([iend, m, m], 42, order="F", dtype=np.float64)

    sdfg(a=a, lu=lu, sym_iend=iend, m=m, iend=iend, sym_m=m)
    assert np.allclose(lu, 42)


def test_fortran_frontend_array_3dmap():
    """
    Tests that the normalization of multidimensional array indices works correctly.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(4, 4, 4)
  d(:, :, :) = 7
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    sdutil.normalize_offsets(sdfg)
    from dace.transformation.auto import auto_optimize as aopt
    aopt.auto_optimize(sdfg, dtypes.DeviceType.CPU)
    a = np.full([4, 4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0, 0] == 7)
    assert (a[3, 3, 3] == 7)


def test_fortran_frontend_twoconnector():
    """
    Tests that the multiple connectors to one array are handled correctly.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(4)
  d(2) = d(1) + d(3)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 84)
    assert (a[2] == 42)


def test_fortran_frontend_input_output_connector():
    """
    Tests that the presence of input and output connectors for the same array is handled correctly.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  double precision d(2, 3)
  integer a, b
  a = 1
  b = 2
  d(:, :) = 0.0
  d(a, b) = d(1, 1) + 5
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


def test_fortran_frontend_memlet_in_map_test():
    """
    Tests that no assumption is made where the iteration variable is inside a memlet subset
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(INP, OUT)
  real INP(100, 10)
  real OUT(100, 10)
  integer I
  do I = 1, 100
    call inner_loops(INP(I, :), OUT(I, :))
  end do
end subroutine main

subroutine inner_loops(INP, OUT)
  real INP(10)
  real OUT(10)
  integer J


  do J = 1, 10
    OUT(J) = INP(J) + 1
  end do
end subroutine inner_loops
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    sdfg.simplify()
    # Expect that the start is a for loop
    assert len(sdfg.nodes()) == 1
    loop = sdfg.nodes()[0]
    assert isinstance(loop, LoopRegion)
    iter_var = symbolic.symbol(loop.loop_variable)

    for state in sdfg.states():
        if len(state.nodes()) > 1:
            for node in state.nodes():
                if isinstance(node, AccessNode) and node.data in ['INP', 'OUT']:
                    edges = [*state.in_edges(node), *state.out_edges(node)]
                    # There should be only one edge in/to the access node
                    assert len(edges) == 1
                    memlet = edges[0].data
                    # Check that the correct memlet has the iteration variable
                    assert memlet.subset[0] == (iter_var, iter_var, 1)
                    assert memlet.subset[1] == (1, 10, 1)


def test_pass_an_arrayslice_that_looks_like_a_scalar_from_outside_with_literal_size():
    sources, main = SourceCodeBuilder().add_file("""
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    ! TODO: We currently cannot handle an assumed shape `(:)` here.
    real, intent(in) :: d(dz)  ! Dependency on `dz`.
    f = sum(d)
  end function f
end module lib

subroutine main(d)
  use lib
  integer :: i
  integer :: sz
  real, intent(inout) :: d(50)
  do i=1, 50
    d(i) = i * 1.0
  end do
  sz = 0
  do i=2,3
    sz = sz + i
  end do
  d(1) = f(d(11), 5)  ! Passing literal `5`
end subroutine main
""").check_with_gfortran().get()
    g = create_singular_sdfg_from_string(sources, 'main')
    g.simplify()
    g.compile()

    d = np.full([50], 42, order="F", dtype=np.float32)
    g(d=d)
    assert d[0] == 65


def test_pass_an_arrayslice_that_looks_like_a_scalar_from_outside_with_symbolic_size():
    sources, main = SourceCodeBuilder().add_file("""
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    ! TODO: We currently cannot handle an assumed shape `(:)` here.
    real, intent(in) :: d(dz)  ! Dependency on `dz`.
    f = sum(d)
  end function f
end module lib

subroutine main(d)
  use lib
  integer :: i
  integer :: sz
  real, intent(inout) :: d(50)
  do i=1, 50
    d(i) = i * 1.0
  end do
  sz = 0
  do i=2,3
    sz = sz + i
  end do
  d(1) = f(d(11), sz)  ! `sz == 5`, but we are passing it symbolically.
end subroutine main
""").check_with_gfortran().get()
    g = create_singular_sdfg_from_string(sources, 'main')
    g.simplify()
    g.compile()

    d = np.full([50], 42, order="F", dtype=np.float32)
    g(d=d)
    assert d[0] == 65


if __name__ == "__main__":
    test_fortran_frontend_array_3dmap()
    test_fortran_frontend_array_access()
    test_fortran_frontend_input_output_connector()
    test_fortran_frontend_array_ranges()
    test_fortran_frontend_array_multiple_ranges_with_symbols()
    test_fortran_frontend_twoconnector()
    test_fortran_frontend_memlet_in_map_test()
