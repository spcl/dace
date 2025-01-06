from typing import Dict

import numpy as np

import dace
from dace.frontend.fortran.ast_desugaring import ConstTypeInjection
from dace.frontend.fortran.fortran_parser import ParseConfig, create_internal_ast, SDFGConfig, \
    create_sdfg_from_internal_ast
from tests.fortran.fortran_test_helper import SourceCodeBuilder, create_singular_sdfg_from_string


def construct_internal_ast(sources: Dict[str, str]):
    assert 'main.f90' in sources
    cfg = ParseConfig(sources['main.f90'], sources, [])
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
""").add_file("""
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
        sources, entry_point='main',  normalize_offsets=False,
        config_injections=[
            ConstTypeInjection(scope_spec=None, type_spec=('lib', 'config'), component_spec=('a_d0_s',), value='3'),
            ConstTypeInjection(scope_spec=None, type_spec=('lib', 'config'), component_spec=('a_d1_s',), value='4'),
            ConstTypeInjection(scope_spec=None, type_spec=('lib', 'config'), component_spec=('b_d0_s',), value='5'),
            ConstTypeInjection(scope_spec=None, type_spec=('lib', 'config'), component_spec=('b_d1_s',), value='6'),
            ConstTypeInjection(scope_spec=None, type_spec=('lib', 'config'), component_spec=('b_d2_s',), value='7'),
        ])
    g.simplify(verbose=True)
    g.compile()
    g.save('/Users/pmz/Downloads/bleh.sdfg')
    print(g.free_symbols)
    print(g.arglist())

    # As per the injection, the result should be 3 (first dimension size of a) + 5 (first dimension size of b)
    cfg_T = dace.data.Structure({'a': dace.int32[3, 4], 'b': dace.float32[5, 6, 7]}, 'config')
    cfg = cfg_T.dtype._typeclass.as_ctypes()()
    c = np.zeros(2, dtype=np.float32)
    g(cfg=cfg, c=c)
    assert c[0] == 3 + 5

    # Even if we now pass a different value (which we shouldn't), the result stays unchanged, since the values are
    # already injected.
    cfg_T = dace.data.Structure({'a': dace.int32[1, 1], 'b': dace.float32[1, 1, 1]}, 'config')
    cfg = cfg_T.dtype._typeclass.as_ctypes()()
    c = np.zeros(2, dtype=np.float32)
    g(cfg=cfg, c=c)
    assert c[0] == 3 + 5
