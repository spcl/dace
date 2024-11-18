# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, List

import numpy as np

from dace.frontend.fortran.ast_components import InternalFortranAst
from dace.frontend.fortran.ast_internal_classes import FNode
from dace.frontend.fortran.fortran_parser import ParseConfig, create_internal_ast, SDFGConfig, \
    create_sdfg_from_internal_ast
from tests.fortran.fotran_test_helper import SourceCodeBuilder


def construct_internal_ast(sources: Dict[str, str]):
    assert 'main.f90' in sources
    cfg = ParseConfig(sources['main.f90'], sources, [])
    iast, prog = create_internal_ast(cfg)
    return iast, prog


def construct_sdfg(iast: InternalFortranAst, prog: FNode, entry_points: List[str]):
    entry_points = [list(ep.split('.')) for ep in entry_points]
    entry_points = {ep[-1]: ep for ep in entry_points}
    cfg = SDFGConfig(entry_points)
    g = create_sdfg_from_internal_ast(iast, prog, cfg)
    return g


def test_minimal():
    """
    A simple program to just verify that we can produce compilable SDFGs.
    """
    sources, main = SourceCodeBuilder().add_file("""
program main
  implicit none
  double precision d(4)
  d(2) = 5.5
end program main
subroutine fun(d)
  implicit none
  double precision, intent(inout) :: d(4)
  d(2) = 4.2
end subroutine fun
""").check_with_gfortran().get()
    # Construct
    iast, prog = construct_internal_ast(sources)
    gmap = construct_sdfg(iast, prog, ['main', 'fun'])

    # Verify
    assert gmap.keys() == {'main', 'fun'}
    gmap['main'].compile()
    # We will do nothing else here, since it's just a sanity check test.


def test_standalone_subroutines():
    """
    A standalone subroutine, with no program or module in sight.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine fun(d)
  implicit none
  double precision, intent(inout) :: d(4)
  d(2) = 4.2
end subroutine fun

subroutine not_fun(d, val)
  implicit none
  double precision, intent(in) :: val
  double precision, intent(inout) :: d(4)
  d(4) = val
end subroutine not_fun
""").check_with_gfortran().get()
    # Construct
    iast, prog = construct_internal_ast(sources)
    gmap = construct_sdfg(iast, prog, ['fun', 'not_fun'])

    # Verify
    assert gmap.keys() == {'fun', 'not_fun'}
    d = np.full([4], 0, dtype=np.float64)

    fun = gmap['fun'].compile()
    fun(d=d)
    assert np.allclose(d, [0, 4.2, 0, 0])
    not_fun = gmap['not_fun'].compile()
    not_fun(d=d, val=5.5)
    assert np.allclose(d, [0, 4.2, 0, 5.5])


def test_subroutines_from_module():
    """
    A standalone subroutine, with no program or module in sight.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
contains
  subroutine fun(d)
    implicit none
    double precision, intent(inout) :: d(4)
    d(2) = 4.2
  end subroutine fun

  subroutine not_fun(d, val)
    implicit none
    double precision, intent(in) :: val
    double precision, intent(inout) :: d(4)
    d(4) = val
  end subroutine not_fun
end module lib
""").add_file("""
program main
  use lib
  implicit none
end program main
""").check_with_gfortran().get()
    # Construct
    iast, prog = construct_internal_ast(sources)
    gmap = construct_sdfg(iast, prog, ['lib.fun', 'lib.not_fun'])

    # Verify
    assert gmap.keys() == {'fun', 'not_fun'}
    d = np.full([4], 0, dtype=np.float64)

    fun = gmap['fun'].compile()
    fun(d=d)
    assert np.allclose(d, [0, 4.2, 0, 0])
    not_fun = gmap['not_fun'].compile()
    not_fun(d=d, val=5.5)
    assert np.allclose(d, [0, 4.2, 0, 5.5])
