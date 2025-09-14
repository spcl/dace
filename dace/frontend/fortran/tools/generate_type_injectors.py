# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Given a complete Fortran codebase (without C++ preprocessor statements, from which a fully resolveable AST can be
produced), this script will generate a Fortran module that can serialize "configuration injection" entries for every
derived type in that AST.

Example:
```sh
python -m dace.frontend.fortran.tools.generate_type_injectors \
    -i ~/gitspace/icon-dace-for-serde/icon-dace/externals/ecrad \
    -f dace/frontend/fortran/conf_files/ti.F90
```
"""

import argparse
from pathlib import Path

from dace.frontend.fortran.fortran_parser import create_fparser_ast, ParseConfig
from dace.frontend.fortran.gen_serde import find_all_f90_files, generate_type_injection_code


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-i',
                      '--in_src',
                      type=str,
                      required=True,
                      action='append',
                      default=[],
                      help='The files or directories containing Fortran source code (absolute path or relative to CWD).'
                      'Can be repeated to include multiple files and directories.')
    argp.add_argument('-f',
                      '--out_f90',
                      type=str,
                      required=False,
                      default=None,
                      help='A file to write the generated F90 module into (absolute path or relative to CWD).')
    argp.add_argument('-m',
                      '--module_name',
                      type=str,
                      required=False,
                      default='type_injection',
                      help="The name of the generated type injection module's name.")
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    ast = create_fparser_ast(ParseConfig(sources=input_f90s))
    ti_code = generate_type_injection_code(ast, args.module_name)

    if args.out_f90:
        with open(args.out_f90, 'w') as f:
            f.write(ti_code)
    else:
        print(f"=== F90 TYPE INJECTOR CODE BEGINS ===")
        print(ti_code)
        print(f"=== F90 TYPE INJECTOR CODE ENDS ===")


if __name__ == "__main__":
    main()
