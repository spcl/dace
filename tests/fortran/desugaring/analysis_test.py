# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace.frontend.fortran.ast_desugaring import analysis
from tests.fortran.desugaring.common import parse_and_improve
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_spec_mapping_of_abstract_interface():
    sources, main = SourceCodeBuilder().add_file("""
module lib  ! should be present
  abstract interface  ! should NOT be present
    subroutine fun  ! should be present
    end subroutine fun
  end interface
end module lib
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {('lib', ), ('lib', '__interface__', 'fun')}

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {('lib', ), ('lib', '__interface__', 'fun')}


def test_spec_mapping_of_type_extension():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  type base
    integer :: a
  end type base
  type, extends(base) :: ext
    integer :: b
  end type ext
end module lib
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {('lib', ), ('lib', 'base'), ('lib', 'base', 'a'), ('lib', 'ext'), ('lib', 'ext', 'b')}

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {('lib', ), ('lib', 'base'), ('lib', 'base', 'a'), ('lib', 'ext'), ('lib', 'ext', 'b'),
                                ('lib', 'ext', 'base'), ('lib', 'ext', 'base', 'a')}


def test_spec_mapping_of_procedure_pointers():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  type T
    procedure(fun), nopass, pointer :: fun
    procedure(fun), nopass, pointer :: nofun
  end type T
  procedure(fun), pointer :: real_fun => null()
contains
  real function fun()
    fun = 1.1
  end function fun
end module lib
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert (ident_map.keys() == {('lib', ), ('lib', 'T'), ('lib', 'T', 'fun'), ('lib', 'T', 'nofun'), ('lib', 'fun'),
                                 ('lib', 'real_fun')})

    alias_map = analysis.alias_specs(ast)
    assert (alias_map.keys() == {('lib', ), ('lib', 'T'), ('lib', 'T', 'fun'), ('lib', 'T', 'nofun'), ('lib', 'fun'),
                                 ('lib', 'real_fun')})
