from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

from fparser.two.Fortran2003 import Program, Execution_Part, Specification_Part, Use_Stmt, Call_Stmt, \
    Main_Program
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from dace.frontend.fortran.ast_desugaring import correct_for_function_calls, append_children, prepend_children
from dace.frontend.fortran.ast_utils import singular
from dace.frontend.fortran.fortran_parser import construct_full_ast
from dace.frontend.fortran.gen_serde import generate_serde_code, gen_serde_module_skeleton, minimal_preprocessing
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(sources, parser)
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
    real, dimension(3:5) :: bB = 5.
    real, allocatable :: bBZ(:, :)
    double precision :: c = 3.d0
    double precision :: cC(3:4, 5:6) = 6.d0
    double precision, pointer :: cCP(:) => null()
    logical :: d = .true.
    logical :: dD(3) = .false.
    logical, pointer :: dDP(:) => null()
  end type T3

  type T2
    type(T3) :: w(1)
  end type T2

  type T
    type(T2) :: name
  end type T
end module lib

program main
  use lib
  implicit none
  real :: d(5, 5)
  type(T), target :: s
  allocate(s%name%w(1)%bBZ(2,2))
  s%name%w(1)%bBZ = 5.1
  s%name%w(1)%cCP => s%name%w(1)%cC(:, 5)
  call f2(s)
  ! TODO: Find a way to use generic functions to serialize arbitrary types.
  d(1, 1) = s%name%w(1)%a
end program main

subroutine f2(s)
  use lib
  implicit none
  type(T) :: s
  s%name%w(1)%a = 42
end subroutine f2
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    with NamedTemporaryFile() as s_data:
        ast = minimal_preprocessing(ast)
        serde_mod = generate_serde_code(ast)

        # Modify the AST to use the serializer.
        # 1. Reconstruct the original AST, since we have run some preprocessing on the existing one.
        ast = parse_and_improve(sources)
        # 2. Instrument the module usage, to serialize certain data into the path `s_data`.
        y = singular(y for p in walk(ast, Main_Program) for y in walk(p, Specification_Part))
        x = singular(x for p in walk(ast, Main_Program) for x in walk(p, Execution_Part))
        prepend_children(y, Use_Stmt(f"use serde"))
        append_children(x, Call_Stmt(f'call write_to("{s_data.name}", trim(serialize(s)))'))

        # Now reconstruct the AST again, this time with serde module in place.
        ast = parse_and_improve({'serde.f90': serde_mod.tofortran(), 'main.f90': ast.tofortran()})

        code = ast.tofortran()
        SourceCodeBuilder().add_file(code).run_with_gfortran()

        got = Path(s_data.name).read_text().strip()
        # TODO: Get rid of unwanted whitespaces in the generated fortran code.
        got = '\n'.join(l.strip() for l in got.split('\n') if l.strip())
        want = """
# name
# w
# rank
1
# size
1
# lbound
1
# entries
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
T
=> missing
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
F
""".strip()
        assert want == got
