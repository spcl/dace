# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dace.frontend.fortran.ast_internal_classes import Program_Node, Main_Program_Node, Subroutine_Subprogram_Node, \
    Module_Node, Specification_Part_Node
from dace.frontend.fortran.ast_transforms import Structures, Structure
from dace.frontend.fortran.fortran_parser import ParseConfig, create_internal_ast
from tests.fortran.fortran_test_helper import SourceCodeBuilder, InternalASTMatcher as M


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
    cfg = ParseConfig(sources=sources, rename_uniquely=False)
    iast, prog = create_internal_ast(cfg)

    # Verify
    assert not iast.fortran_intrinsics().transformations()
    m = M(Program_Node,
          has_attr={
              'main_program': M(Main_Program_Node),
              'subroutine_definitions': [
                  M(Subroutine_Subprogram_Node, {
                      'name': M.NAMED('fun'),
                      'args': [M.NAMED('d')],
                  }),
              ],
              'structures': M(Structures, has_empty_attr={'structures'})
          },
          has_empty_attr={'function_definitions', 'modules', 'placeholders', 'placeholders_offsets'})
    m.check(prog)


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
    cfg = ParseConfig(sources=sources, rename_uniquely=False)
    iast, prog = create_internal_ast(cfg)

    # Verify
    assert not iast.fortran_intrinsics().transformations()
    m = M(Program_Node,
          has_attr={
              'subroutine_definitions': [
                  M(Subroutine_Subprogram_Node, {
                      'name': M.NAMED('fun'),
                      'args': [M.NAMED('d')],
                  }),
                  M(Subroutine_Subprogram_Node, {
                      'name': M.NAMED('not_fun'),
                      'args': [M.NAMED('d'), M.NAMED('val')],
                  }),
              ],
              'structures':
              M(Structures, has_empty_attr={'structures'})
          },
          has_empty_attr={'main_program', 'function_definitions', 'modules', 'placeholders', 'placeholders_offsets'})
    m.check(prog)


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

program main
  use lib
  implicit none
  double precision :: d(4)
  call fun(d)
  call not_fun(d, 2.1d0)
end program main
""").check_with_gfortran().get()
    # Construct
    cfg = ParseConfig(sources=sources, rename_uniquely=False)
    iast, prog = create_internal_ast(cfg)

    # Verify
    assert not iast.fortran_intrinsics().transformations()
    m = M(Program_Node,
          has_attr={
              'main_program':
              M(Main_Program_Node),
              'modules': [
                  M(Module_Node,
                    has_attr={
                        'subroutine_definitions': [
                            M(Subroutine_Subprogram_Node, {
                                'name': M.NAMED('fun'),
                                'args': [M.NAMED('d')],
                            }),
                            M(Subroutine_Subprogram_Node, {
                                'name': M.NAMED('not_fun'),
                                'args': [M.NAMED('d'), M.NAMED('val')],
                            }),
                        ],
                    },
                    has_empty_attr={'function_definitions', 'interface_blocks'})
              ],
              'structures':
              M(Structures, has_empty_attr={'structures'})
          },
          has_empty_attr={'function_definitions', 'subroutine_definitions', 'placeholders', 'placeholders_offsets'})
    m.check(prog)


def test_subroutine_with_local_variable():
    """
    A standalone subroutine, with no program or module in sight.
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine fun(d)
  implicit none
  double precision, intent(inout) :: d(4)
  double precision :: e(4)
  e(:) = 1.0
  e(2) = 4.2
  d(:) = e(:)
end subroutine fun
""").check_with_gfortran().get()
    # Construct
    cfg = ParseConfig(sources=sources, rename_uniquely=False)
    iast, prog = create_internal_ast(cfg)

    # Verify
    assert not iast.fortran_intrinsics().transformations()
    m = M(Program_Node,
          has_attr={
              'subroutine_definitions': [
                  M(Subroutine_Subprogram_Node, {
                      'name': M.NAMED('fun'),
                      'args': [M.NAMED('d')],
                  }),
              ],
              'structures': M(Structures, has_empty_attr={'structures'})
          },
          has_empty_attr={'main_program', 'function_definitions', 'modules', 'placeholders', 'placeholders_offsets'})
    m.check(prog)


def test_subroutine_contains_function():
    """
    A function is defined inside a subroutine that calls it. A main program uses the top-level subroutine.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
contains
  subroutine fun(d)
    implicit none
    double precision d(4)
    d(2) = fun2()

  contains
    real function fun2()
      implicit none
      fun2 = 5.5
    end function fun2
  end subroutine fun
end module lib

program main
  use lib, only: fun
  implicit none

  double precision d(4)
  call fun(d)
end program main
""").check_with_gfortran().get()
    # Construct
    cfg = ParseConfig(sources=sources, rename_uniquely=False)
    iast, prog = create_internal_ast(cfg)

    # Verify
    assert not iast.fortran_intrinsics().transformations()
    m = M(Program_Node,
          has_attr={
              'main_program':
              M(Main_Program_Node),
              'modules': [
                  M(Module_Node,
                    has_attr={
                        'subroutine_definitions': [
                            M(Subroutine_Subprogram_Node, {
                                'name': M.NAMED('fun'),
                                'args': [M.NAMED('d')],
                            }),
                        ],
                    },
                    has_empty_attr={'function_definitions', 'interface_blocks'})
              ],
              'structures':
              M(Structures, has_empty_attr={'structures'})
          },
          has_empty_attr={'function_definitions', 'subroutine_definitions', 'placeholders', 'placeholders_offsets'})
    m.check(prog)

    # TODO: We cannot handle during the internal AST construction (it works just fine before during parsing etc.) when a
    #  subroutine contains other subroutines. This needs to be fixed.
    mod = prog.modules[0]
    # Where could `fun2`'s definition could be?
    assert not mod.function_definitions  # Not here!
    assert 'fun2' not in [f.name.name for f in mod.subroutine_definitions]  # Not here!
    fn = mod.subroutine_definitions[0]
    assert not hasattr(fn, 'function_definitions')  # Not here!
    assert not hasattr(fn, 'subroutine_definitions')  # Not here!


def test_module_contains_types():
    """
    Module has type definition that the program does not use, so it gets pruned.
    """
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type used_type
    real :: w(5, 5, 5), z(5)
    integer :: a
    real :: name
  end type used_type
end module lib

program main
  implicit none
  real :: d(5, 5)
  call fun(d)
end program main
subroutine fun(d)
  use lib, only : used_type
  real d(5, 5)
  type(used_type) :: s
  s%w(1, 1, 1) = 5.5
  d(2, 1) = 5.5 + s%w(1, 1, 1)
end subroutine fun
""").check_with_gfortran().get()
    # Construct
    cfg = ParseConfig(sources=sources, rename_uniquely=False)
    iast, prog = create_internal_ast(cfg)

    # Verify
    assert not iast.fortran_intrinsics().transformations()
    m = M(Program_Node,
          has_attr={
              'main_program':
              M(Main_Program_Node),
              'modules': [
                  M(Module_Node,
                    has_attr={'specification_part': M(Specification_Part_Node, {'typedecls': M.IGNORE(1)})},
                    has_empty_attr={'function_definitions', 'interface_blocks'})
              ],
              'subroutine_definitions': [
                  M(Subroutine_Subprogram_Node, {
                      'name': M.NAMED('fun'),
                      'args': [M.NAMED('d')],
                  }),
              ],
              'structures':
              M(Structures, {
                  'structures': {
                      'used_type': M(Structure)
                  },
              })
          },
          has_empty_attr={'function_definitions', 'placeholders', 'placeholders_offsets'})
    m.check(prog)


if __name__ == '__main__':
    test_minimal()
    test_standalone_subroutines()
    test_subroutines_from_module()
    test_subroutine_with_local_variable()
    test_subroutine_contains_function()
    test_module_contains_types()
