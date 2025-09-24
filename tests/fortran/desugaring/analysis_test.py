# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace.frontend.fortran.ast_desugaring import analysis
from tests.fortran.fortran_test_helper import SourceCodeBuilder, parse_and_improve


def test_spec_mapping_of_abstract_interface():
    """
    Tests that the spec mapping correctly identifies and maps symbols within an
    abstract interface. It ensures that the interface and the subroutine inside
    are correctly added to the identifier and alias maps.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib  ! should be present
  abstract interface  ! should NOT be present
    subroutine fun  ! should be present
    end subroutine fun
  end interface
end module lib
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {("lib", ), ("lib", "__interface__", "fun")}

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {("lib", ), ("lib", "__interface__", "fun")}


def test_spec_mapping_of_type_extension():
    """
    Tests that the spec mapping correctly handles type extensions. It checks that
    the components of the base type are correctly included in the extended type's
    alias map.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
module lib
  type base
    integer :: a
  end type base
  type, extends(base) :: ext
    integer :: b
  end type ext
end module lib
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {
        ("lib", ),
        ("lib", "base"),
        ("lib", "base", "a"),
        ("lib", "ext"),
        ("lib", "ext", "b"),
    }

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {
        ("lib", ),
        ("lib", "base"),
        ("lib", "base", "a"),
        ("lib", "ext"),
        ("lib", "ext", "b"),
        ("lib", "ext", "base"),
        ("lib", "ext", "base", "a"),
    }


def test_spec_mapping_of_procedure_pointers():
    """
    Tests that the spec mapping correctly handles procedure pointers, both as
    components of a derived type and as standalone variables.
    """
    sources, _ = (SourceCodeBuilder().add_file("""
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
""").check_with_gfortran().get())
    ast = parse_and_improve(sources)

    ident_map = analysis.identifier_specs(ast)
    assert ident_map.keys() == {
        ("lib", ),
        ("lib", "T"),
        ("lib", "T", "fun"),
        ("lib", "T", "nofun"),
        ("lib", "fun"),
        ("lib", "real_fun"),
    }

    alias_map = analysis.alias_specs(ast)
    assert alias_map.keys() == {
        ("lib", ),
        ("lib", "T"),
        ("lib", "T", "fun"),
        ("lib", "T", "nofun"),
        ("lib", "fun"),
        ("lib", "real_fun"),
    }


if __name__ == "__main__":
    test_spec_mapping_of_abstract_interface()
    test_spec_mapping_of_type_extension()
    test_spec_mapping_of_procedure_pointers()
