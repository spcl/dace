from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict

from fparser.two.Fortran2003 import Program, Execution_Part, Specification_Part, Use_Stmt, Call_Stmt, \
    Main_Program, Component_Decl
from fparser.two.utils import walk

import dace
from dace.config import Config
from dace.frontend.fortran.ast_desugaring import correct_for_function_calls, append_children, prepend_children, \
    identifier_specs
from dace.frontend.fortran.ast_utils import singular
from dace.frontend.fortran.config_propagation_data import deserialize
from dace.frontend.fortran.fortran_parser import ParseConfig, \
    create_fparser_ast, create_internal_ast, SDFGConfig, create_sdfg_from_internal_ast, \
    run_fparser_transformations
from dace.frontend.fortran.gen_serde import generate_serde_code, generate_type_injection_code
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def parse_and_improve(sources: Dict[str, str]):
    ast = create_fparser_ast(ParseConfig(sources=sources))
    ast = correct_for_function_calls(ast)
    # NOTE: We don't run `run_fparser_transformations(ast, cfg)` here since we use this function to produce AST that
    # retains interfaces and procedures too.
    assert isinstance(ast, Program)
    return ast


def run_main_cpp(cpp_code: str, root_dir: str, libname: str) -> str:
    DACE_ROOT = Path(dace.__file__).parent
    return SourceCodeBuilder().add_file(cpp_code, 'main.cc').run_with_gcc([
        *Config.get_default('compiler', 'cpu', 'args').split(),
        f"-std=c++17",
        f"-I{root_dir}/include",
        f"-I{DACE_ROOT}/runtime/include/",
        f"-L{root_dir}/build",
        f"-l{libname}",
        f"-Wl,-rpath,{root_dir}/build",
    ])


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
    type(T3) :: w  ! TODO: How to make `w` work?
  end type T2

  type T
    type(T2) :: name
  end type T
end module lib

program main  ! Original entry point.
  use lib
  implicit none
  real :: d(5, 5)
  type(T), target :: s
  s%name%w%a = 0
  allocate(s%name%w%bBZ(2,2))
  s%name%w%bBZ = 5.1
  s%name%w%cCP => s%name%w%cC(:, 5)
  call f2(s)
  ! TODO: Find a way to use generic functions to serialize arbitrary types.
  d(1, 1) = s%name%w%a
end program main

logical function f1(s)
  use lib
  implicit none
  type(T3) :: s
  f1 = allocated(s % bbz)
end function f1

subroutine f2(s)  ! Entry point for the SDFG.
  use lib
  implicit none
  type(T) :: s
  s%name%w%a = s%name%w%a + 42
end subroutine f2
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    # We want a particular SDFG where the typical prunings do not happen, since we want to test them.
    # TODO: We have to discard the `pointer` type components, because internal AST cannot handle them.
    do_not_prune = ({k for k, v in identifier_specs(ast).items() if isinstance(v, Component_Decl)}
                    - {('lib', 't3', 'ccp'), ('lib', 't3', 'ddp')})
    cfg = ParseConfig(sources={'main.f90': ast.tofortran()}, entry_points=('f2',), do_not_prune=list(do_not_prune))
    own_ast, program = create_internal_ast(cfg)
    gmap = create_sdfg_from_internal_ast(own_ast, program, SDFGConfig({'f2': 'f2'}))
    assert gmap.keys() == {'f2'}
    g = list(gmap.values())[0]
    g.simplify()

    with TemporaryDirectory() as t_dir, NamedTemporaryFile() as s_data:
        # TODO: We cannot compile with `allocatable` type components, because the generated C++ code is broken.
        g.build_folder = t_dir
        g.compile()
        serde_code = generate_serde_code(ast, g)
        ti_code = generate_type_injection_code(ast)

        # Modify the AST to use the serializer.
        # 1. Reconstruct the original AST, since we have run some preprocessing on the existing one.
        ast = parse_and_improve(sources)
        # 2. Instrument the module usage, to serialize certain data into the path `s_data`.
        y = singular(y for p in walk(ast, Main_Program) for y in walk(p, Specification_Part))
        x = singular(x for p in walk(ast, Main_Program) for x in walk(p, Execution_Part))
        prepend_children(y, [Use_Stmt(f"use serde"), Use_Stmt(f"use type_injection")])
        append_children(x, Call_Stmt(f'call serialize(at("{s_data.name}", .true.), s)'))
        append_children(x, Call_Stmt(f'call serialize(at("{s_data.name}.bbz", .true.), s%name%w%bBZ)'))
        append_children(x, Call_Stmt(f'call type_inject(at("{s_data.name}.ti"), s%name%w)'))

        # Now reconstruct the AST again, this time with serde module in place. Then we will run the test and ensure that
        # the serialization is as expected.
        ast = parse_and_improve({'serde.f90': serde_code.f90_serializer, 'ti.f90': ti_code, 'main.f90': ast.tofortran()})
        SourceCodeBuilder().add_file(ast.tofortran()).run_with_gfortran()

        got = Path(s_data.name).read_text().strip()
        want = """
# name
# w
# a
42
# aa
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
# aaz
# alloc
0
# b
2.00000000
# bb
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
# bbz
# alloc
1
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
# cc
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
# d
1
# dd
# rank
1
# size
3
# lbound
1
# entries
0
0
0
""".strip()
        assert want == got

        # Now, verify that it can be deserialized from C++ by re-serializing and comparing.
        cpp_code = f"""
{serde_code.cpp_serde}
#include "{g.name}.h"

#include <fstream>
#include <iostream>

int main() {{
    std::ifstream data("{s_data.name}");
    std::ifstream data_bbz("{s_data.name}.bbz");


    t x;
    serde::deserialize(&x, data);
    // Just checking if we can read the plain array too.
    auto [m, y] = serde::read_array<float>(data_bbz);

    auto* h = __dace_init_f2(&x);
    __program_f2(h, &x);

    std::cout << serde::serialize(&x) << std::endl;

    return __dace_exit_f2(h);
}}
"""
        output = run_main_cpp(cpp_code, t_dir, g.name)
        assert output.strip() == (f"""
# name
# w
# a
84
# aa
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
# aaz
# alloc
0
# b
2.000000
# bb
# rank
1
# size
3
# lbound
3
# entries
5.000000
5.000000
5.000000
# bbz
# alloc
1
# rank
2
# size
2
2
# lbound
1
1
# entries
5.100000
5.100000
5.100000
5.100000
# c
3.000000
# cc
# rank
2
# size
2
2
# lbound
3
5
# entries
6.000000
6.000000
6.000000
6.000000
# d
1
# dd
# rank
1
# size
3
# lbound
1
# entries
0
0
0
""").strip()

        cinjs = [deserialize(l.strip()) for l in Path(f"{s_data.name}.ti").read_text().splitlines() if l.strip()]
        cfg = ParseConfig(sources=sources, entry_points=[('f1',), ('f2',)], config_injections=cinjs)
        ast = create_fparser_ast(cfg)
        ast = run_fparser_transformations(ast, cfg)
        assert f"""
LOGICAL FUNCTION f1(s)
  USE lib, ONLY: t3
  IMPLICIT NONE
  TYPE(t3) :: s
  f1 = .TRUE.
END FUNCTION f1
""".strip() in ast.tofortran()
