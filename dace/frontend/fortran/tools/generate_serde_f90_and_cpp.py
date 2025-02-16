"""
Given a complete Fortran codebase (without C++ preprocessor statements, from which a fully resolveable AST can be
produced) and an SDFG that is the final product of this codebase, this script will generate a Fortran module that can
serialize various data (scalars, array and structures) that are still present in the SDFG (i.e., after the
transformations and pruning), and a C++ module that can deserialize such data.

Example:
```sh
python -m dace.frontend.fortran.tools.generate_serde_f90_and_cpp \
    -i ~/gitspace/icon-dace-for-serde/icon-dace/externals/ecrad \
    -g ~/calc_surface_spectral_simplified_dbg22full.sdfg \
    -f dace/frontend/fortran/conf_files/serde.F90 \
    -c dace/frontend/fortran/conf_files/serde.h
```
"""

import argparse
from pathlib import Path
from typing import List

from dace import SDFG
from dace.frontend.fortran.ast_desugaring import const_eval_nodes, ConstTypeInjection, inject_const_evals
from dace.frontend.fortran.config_propagation_data import deserialize
from dace.frontend.fortran.create_preprocessed_ast import find_all_f90_files
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast
from dace.frontend.fortran.gen_serde import generate_serde_code, _keep_only_derived_types


def config_injection_list(root: str = 'dace/frontend/fortran/conf_files') -> List[ConstTypeInjection]:
    cfgs = [Path(root).joinpath(f).read_text() for f in [
        'config.ti', 'aerosol.ti', 'cloud.ti', 'flux.ti', 'gas.ti', 'single_level.ti', 'thermodynamics.ti']]
    injs = [deserialize(l.strip()) for c in cfgs for l in c.splitlines() if l.strip()]
    return injs


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-i', '--in_src', type=str, required=True, action='append', default=[],
                      help='The files or directories containing Fortran source code (absolute path or relative to CWD).'
                           'Can be repeated to include multiple files and directories.')
    argp.add_argument('-g', '--in_sdfg', type=str, required=True, default=None,
                      help='The SDFG file containing the final product of DaCe. We need this to know the structures '
                           'and their members that are present in the end (aftre further pruning etc.)')
    argp.add_argument('-f', '--out_f90', type=str, required=False, default=None,
                      help='A file to write the generated F90 functions into (absolute path or relative to CWD).')
    argp.add_argument('-c', '--out_cpp', type=str, required=False, default=None,
                      help='A file to write the generated C++ functions into (absolute path or relative to CWD).')
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    print(f"Will be using the SDFG as the deserializer target: {args.in_sdfg}")
    g = SDFG.from_file(args.in_sdfg)

    cfg = ParseConfig(sources=input_f90s, config_injections=config_injection_list())
    ast = create_fparser_ast(cfg)
    ast = _keep_only_derived_types(ast)
    ast = const_eval_nodes(ast)
    ast = inject_const_evals(ast, cfg.config_injections)
    # Generated serde code from the processed code.
    serde_code = generate_serde_code(ast, g)

    if args.out_f90:
        with open(args.out_f90, 'w') as f:
            f.write(serde_code.f90_serializer)
    else:
        print(f"=== F90 SERIALIZER CODE BEGINS ===")
        print(serde_code.f90_serializer)
        print(f"=== F90 SERIALIZER CODE ENDS ===")

    if args.out_cpp:
        with open(args.out_cpp, 'w') as f:
            f.write(serde_code.cpp_serde)
    else:
        print(f"=== C++ SERDE CODE BEGINS ===")
        print(serde_code.cpp_serde)
        print(f"=== C++ SERDE CODE ENDS ===")


if __name__ == "__main__":
    main()
