import argparse
from itertools import chain
from pathlib import Path
from typing import Generator

from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast, run_fparser_transformations
from dace.frontend.fortran.gen_serde import find_all_f90_files


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-i', '--in_src', type=str, required=True, action='append', default=[],
                      help='The files or directories containing Fortran source code (absolute path or relative to CWD).'
                           'Can be repeated to include multiple files and directories.')
    argp.add_argument('-k', '--entry_point', type=str, required=True, action='append', default=[],
                      help='The entry points which should be kept with their dependencies (can be repeated).'
                           'Specify each entry point as a `dot` separated path through named objects from the top.')
    argp.add_argument('-o', '--output_ast', type=str, required=False, default=None,
                      help='(Optional) A file to write the preprocessed AST into (absolute path or relative to CWD).'
                           'If nothing is given, then will write to STDOUT.')
    argp.add_argument('--noop', type=str, required=False, action='append', default=[],
                      help='(Optional) Functions or subroutine to make no-op.')
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    entry_points = [tuple(ep.split('.')) for ep in args.entry_point]
    print(f"Will be keeping these as entry points: {entry_points}")

    noops = [tuple(np.split('.')) for np in args.noop]
    print(f"Will be making these as no-ops: {noops}")

    cfg = ParseConfig(sources=input_f90s, entry_points=entry_points, make_noop=noops)

    ast = create_fparser_ast(cfg)
    ast = run_fparser_transformations(ast, cfg)
    assert ast.children, f"Nothing remains in this AST after pruning."
    f90 = ast.tofortran()

    if args.output_ast:
        with open(args.output_ast, 'w') as f:
            f.write(f90)
    else:
        print(f90)


if __name__ == "__main__":
    main()
