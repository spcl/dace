import os
from pathlib import Path
from tempfile import TemporaryFile, NamedTemporaryFile
from typing import Dict

from fparser.api import get_reader
from fparser.common.readfortran import FortranStringReader
from fparser.two.Fortran2003 import Program, Module, Execution_Part, Specification_Part, Use_Stmt, Call_Stmt, \
    Main_Program
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from dace.frontend.fortran.ast_desugaring import correct_for_function_calls, append_children, prepend_children
from dace.frontend.fortran.ast_utils import singular
from dace.frontend.fortran.fortran_parser import recursive_ast_improver
from dace.frontend.fortran.gen_serde import generate_serde_module
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    assert 'main.f90' in sources
    reader = FortranStringReader(sources['main.f90'])
    ast = parser(reader)
    ast = recursive_ast_improver(ast, sources, [], parser)
    ast = correct_for_function_calls(ast)
    assert isinstance(ast, Program)
    return ast


def test_gen_serde():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none

  type T3
    integer :: a = 1
    integer :: aA(3) = 4
    integer, allocatable :: aAZ(:)
    real :: b = 2.
    real :: bB(3:5) = 5.
    real, allocatable :: bBZ(:, :)
    double precision :: c = 3.d0
    double precision :: cC(3:4, 5:6) = 6.d0
    double precision, pointer :: cCP(:, :) => null()
    logical :: d = .true.
    logical :: dD(3) = .false.
    logical, pointer :: dDP(:) => null()
  end type T3

  type T2
    ! TODO: Find a way to use generic functions to serialize arbitrary types.
    ! type(T3) :: w(7:12, 8:13)
    type(T3) :: w
  end type T2

  type T
    type(T2) :: name
  end type T
end module lib
""").add_file("""
program main
  use lib
  implicit none
  real :: d(5, 5)
  type(T), target :: s
  allocate(s%name%w%bBZ(2,2))
  s%name%w%bBZ = 5.1
  s%name%w%dDP => s%name%w%dD
  call f2(s)
  ! TODO: Find a way to use generic functions to serialize arbitrary types.
  ! d(1, 1) = s%name%w(8, 10)%a
  d(1, 1) = s%name%w%a
end program main

subroutine f2(s)
  use lib
  implicit none
  type(T) :: s
  ! TODO: Find a way to use generic functions to serialize arbitrary types.
  ! s%name%w(8, 10)%a = 42
  s%name%w%a = 42
end subroutine f2
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    with NamedTemporaryFile() as s_data:
        test_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        serde_base = Module(get_reader(
            test_dir.joinpath('../../dace/frontend/fortran/conf_files/serde_base.f90').read_text()))
        serde_mod = generate_serde_module(serde_base, ast)

        # Modify the AST to use the serializer.
        y = singular(y for p in walk(ast, Main_Program) for y in walk(p, Specification_Part))
        x = singular(x for p in walk(ast, Main_Program) for x in walk(p, Execution_Part))
        prepend_children(y, Use_Stmt(f"use serde"))
        append_children(x, Call_Stmt(f'call write_to("{s_data.name}", trim(serialize(s)))'))
        ast = recursive_ast_improver(
            ast, {'serde.f90': serde_mod.tofortran()}, [], ParserFactory().create(std="f2008"))

        code = ast.tofortran()
        SourceCodeBuilder().add_file(code).run_with_gfortran()

        got = Path(s_data.name).read_text().strip()
        # TODO: Get rid of unwanted whitespaces in the generated fortran code.
        got = '\n'.join(l.strip() for l in got.split('\n') if l.strip())
        print(got)
        want = """
# name
# w
# a
42
# aA
# rank
1
# size
3
# lbound
1
# entries
4
4
4
# aAZ
# alloc
F
# b
2.00000000
# bB
# rank
1
# size
3
# lbound
3
# entries
5.00000000
5.00000000
5.00000000
# bBZ
# alloc
T
# rank
2
# size
2
2
# lbound
1
1
# entries
5.09999990
5.09999990
5.09999990
5.09999990
# c
3.0000000000000000
# cC
# rank
2
# size
2
2
# lbound
3
5
# entries
6.0000000000000000
6.0000000000000000
6.0000000000000000
6.0000000000000000
# cCP
# assoc
F
# d
T
# dD
# rank
1
# size
3
# lbound
1
# entries
F
F
F
# dDP
# assoc
T
""".strip()
        assert want == got
