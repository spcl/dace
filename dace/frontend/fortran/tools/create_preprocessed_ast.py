# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import argparse
from pathlib import Path

from dace.frontend.fortran.fortran_parser import (
    ParseConfig,
    create_fparser_ast,
    run_fparser_transformations,
)
from dace.frontend.fortran.tools.helpers import find_all_f90_files

BUILTINS = """
module iso_c_binding
  integer, parameter :: c_int8_t = 1, c_int16_t = 2, c_int32_t = 4, c_int64_t = 8
  integer, parameter :: c_char = c_int8_t, c_signed_char = c_char, c_bool = c_int8_t, c_int = c_int32_t, c_long = c_int, c_size_t = c_int64_t
  integer, parameter :: c_float = 4, c_double = 8
  type c_ptr
  end type c_ptr
  type c_funptr
  end type c_funptr
  type(c_ptr), parameter :: c_null_ptr = c_ptr()
  character(kind=c_char), parameter :: c_null_char = char(0)
  interface c_f_pointer
    module procedure :: cfp_logical_r3
  end interface c_f_pointer
  interface c_f_procpointer
  end interface c_f_procpointer
  interface c_loc
  end interface c_loc
  interface c_associated
    module procedure :: cass_cptr
  end interface c_associated
contains
  subroutine cfp_logical_r3(cptr, fptr, shape, lower)
    type(c_ptr), intent(in) :: cptr
    logical, pointer, intent(out) :: fptr(:, :, :)
    integer, optional :: shape(:)
    integer, optional :: lower(:)
  end subroutine cfp_logical_r3
  logical function cass_cptr(a, b)
    type(c_ptr), intent(in) :: a
    type(c_ptr), optional, intent(in) :: b
  end function cass_cptr
end module iso_c_binding
module iso_fortran_env
  integer, parameter :: real32 = 4
  integer, parameter :: real64 = 8
  integer, parameter :: int32 = 4
  integer, parameter :: int64 = 8
  character, parameter :: compiler_version = "", compiler_options = ""
end module iso_fortran_env
"""


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "-i",
        "--in_src",
        type=str,
        required=True,
        action="append",
        default=[],
        help="The files or directories containing Fortran source code (absolute path or relative to CWD)."
        "Can be repeated to include multiple files and directories.",
    )
    argp.add_argument(
        "-k",
        "--entry_point",
        type=str,
        required=True,
        action="append",
        default=[],
        help="The entry points which should be kept with their dependencies (can be repeated)."
        "Specify each entry point as a `dot` separated path through named objects from the top.",
    )
    argp.add_argument(
        "-o",
        "--output_ast",
        type=str,
        required=False,
        default=None,
        help="(Optional) A file to write the preprocessed AST into (absolute path or relative to CWD)."
        "If nothing is given, then will write to STDOUT.",
    )
    argp.add_argument(
        "--noop",
        type=str,
        required=False,
        action="append",
        default=[],
        help="(Optional) Functions or subroutine to make no-op.",
    )
    argp.add_argument(
        "-d",
        "--checkpoint_dir",
        type=str,
        required=False,
        default=None,
        help="(Optional) If specified, the AST in various stages of preprocessing will be written as"
        "Fortran code in there.",
    )
    argp.add_argument(
        "--consolidate_global_data",
        type=bool,
        required=False,
        default=False,
        help="Whether to consolidate the global data into one structure.",
    )
    argp.add_argument(
        "--rename_uniquely",
        type=bool,
        required=False,
        default=False,
        help="Whether to rename the variables and the functions to have globally unique names.",
    )
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    entry_points = [tuple(ep.split(".")) for ep in args.entry_point]
    print(f"Will be keeping these as entry points: {entry_points}")

    noops = [tuple(np.split(".")) for np in args.noop]
    print(f"Will be making these as no-ops: {noops}")

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        print(f"Will be writing the checkpoint ASTs in: {checkpoint_dir}")

    consolidate_global_data = args.consolidate_global_data
    if consolidate_global_data:
        print(f"Will be consolidating the global data into one structure")
    else:
        print(f"Will leave the global data in their own modules")

    rename_uniquely = args.rename_uniquely
    if rename_uniquely:
        print(f"Will be renaming the variables and the functions to have globally unique names")
    else:
        print(f"Will leave the variable names as they are")

    cfg = ParseConfig(
        sources=input_f90s,
        entry_points=entry_points,
        make_noop=noops,
        ast_checkpoint_dir=checkpoint_dir,
        consolidate_global_data=consolidate_global_data,
        rename_uniquely=rename_uniquely,
    )
    cfg.sources["_builtins.f90"] = BUILTINS

    ast = create_fparser_ast(cfg)
    ast = run_fparser_transformations(ast, cfg)
    assert ast.children, f"Nothing remains in this AST after pruning."
    f90 = ast.tofortran()

    if args.output_ast:
        with open(args.output_ast, "w") as f:
            f.write(f90)
    else:
        print("Preprocessed Fortran AST:\n===")
        print(f90)


if __name__ == "__main__":
    main()
