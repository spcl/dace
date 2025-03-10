from typing import Dict

import numpy as np

import dace
from dace.frontend.fortran.ast_desugaring import ConstTypeInjection
from dace.frontend.fortran.fortran_parser import ParseConfig, create_internal_ast, create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def construct_internal_ast(sources: Dict[str, str]):
    cfg = ParseConfig(sources=sources)
    iast, prog = create_internal_ast(cfg)
    return iast, prog


def test_minimal():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer, allocatable :: a(:, :)
    real, allocatable :: b(:, :, :)
  end type config
contains
  subroutine fun(cfg)
    implicit none
    type(config), intent(inout) :: cfg
    cfg%a = -1
    cfg%b = 5.1
  end subroutine fun
end module lib

subroutine main(cfg, c)
  use lib
  implicit none
  type(config), intent(in) :: cfg
  real, intent(out) :: c(2)
  c(1) = 1
  c(1) = size(cfg%a, 1) + c(1) * size(cfg%b, 1)
end subroutine main
""").check_with_gfortran().get()
    g = create_singular_sdfg_from_string(
        sources, entry_point='main', normalize_offsets=False,
        config_injections=[
            ConstTypeInjection(
                scope_spec=None, type_spec=('lib', 'config'),component_spec=('__f2dace_SA_a_d_0_s',), value='3'),
            ConstTypeInjection(
                scope_spec=None, type_spec=('lib', 'config'), component_spec=('__f2dace_SA_a_d_1_s',), value='4'),
            ConstTypeInjection(
                scope_spec=None, type_spec=('lib', 'config'), component_spec=('__f2dace_SA_b_d_0_s',), value='5'),
            ConstTypeInjection(
                scope_spec=None, type_spec=('lib', 'config'), component_spec=('__f2dace_SA_b_d_1_s',), value='6'),
            ConstTypeInjection(
                scope_spec=None, type_spec=('lib', 'config'), component_spec=('__f2dace_SA_b_d_2_s',), value='7'),
        ])
    g.simplify(verbose=True)
    g.compile()

    # As per the injection, the result should be 3 (first dimension size of a) + 5 (first dimension size of b)
    cfg_T = g.arrays['cfg'].dtype.base_type.as_ctypes()
    a = np.full([3, 4], 42, order="F", dtype=np.int32)
    b = np.full([5, 6, 7], 42, order="F", dtype=np.int32)
    cfg = cfg_T(a=a.ctypes.data, b=b.ctypes.data)
    c = np.zeros(2, dtype=np.float32)
    g(cfg=cfg, c=c)
    assert c[0] == 3 + 5

    # Even if we now pass a different value (which we shouldn't), the result stays unchanged, since the values are
    # already injected.
    a = np.full([1, 1], 42, order="F", dtype=np.int32)
    b = np.full([1, 1, 1], 42, order="F", dtype=np.int32)
    cfg = cfg_T(a=a.ctypes.data, b=b.ctypes.data)
    c = np.zeros(2, dtype=np.float32)
    g(cfg=cfg, c=c)
    assert c[0] == 3 + 5


if __name__ == "__main__":
    test_minimal()
