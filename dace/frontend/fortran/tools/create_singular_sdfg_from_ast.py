# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import argparse
from pathlib import Path

from dace import SDFG
from dace.frontend.fortran.ast_desugaring import types
from dace.frontend.fortran.ast_utils import singular
from dace.frontend.fortran.config_propagation_data import (
    find_all_config_injection_files,
    find_all_config_injections,
)
from dace.frontend.fortran.fortran_parser import (
    ParseConfig,
    create_internal_ast,
    SDFGConfig,
    create_sdfg_from_internal_ast,
)
from dace.frontend.fortran.tools.helpers import find_all_f90_files


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "-i",
        "--in_src",
        type=str,
        required=True,
        action="append",
        default=[],
        help="The files or directories containing self-contained AST in its Fortran source code "
        "representation (absolute path or relative to CWD). Can be repeated to include multiple "
        "files and directories.",
    )
    argp.add_argument(
        "-k",
        "--entry_point",
        type=str,
        required=True,
        help="The single entry point which should be kept with their dependencies (can be repeated)."
        "Specify the entry point as a `dot` separated path through named objects from the top.",
    )
    argp.add_argument(
        "-c",
        "--config_inject",
        type=str,
        required=False,
        action="append",
        default=[],
        help="The files or directories containing config injection data. "
        "Can be repeated to include multiple files or directories.",
    )
    argp.add_argument(
        "-o",
        "--output_sdfg",
        type=str,
        required=True,
        help="A file to write the SDFG into (absolute path or relative to CWD).",
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
        "--keep_components",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="(Optional) If specified, will avoid pruning struct components.",
    )
    args = argp.parse_args()

    input_dirs = [Path(p) for p in args.in_src]
    input_f90s = [f for p in input_dirs for f in find_all_f90_files(p)]
    print(f"Will be reading from {len(input_f90s)} Fortran files in directories: {input_dirs}")

    entry_point: types.SPEC = tuple(args.entry_point.split("."))
    print(f"Will be using this as entry points: {entry_point}")

    keep_components: bool = args.keep_components
    if keep_components:
        print(f"Will avoid pruning struct components")
    else:
        print(f"Will prune unused struct components")

    output_sdfg = args.output_sdfg
    print(f"Will be writing SDFG to: {output_sdfg}")

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir:
        print(f"Will be writing the checkpoint ASTs in: {checkpoint_dir}")

    ti_dirs = [Path(p) for p in args.config_inject]
    ti_files = [f for p in ti_dirs for f in find_all_config_injection_files(p)]
    config_injections = list(find_all_config_injections(ti_files))

    cfg = ParseConfig(
        sources=input_f90s,
        entry_points=[entry_point],
        config_injections=config_injections,
        ast_checkpoint_dir=checkpoint_dir,
        do_not_prune_type_components=keep_components,
    )
    own_ast, program = create_internal_ast(cfg)

    cfg = SDFGConfig({entry_point[-1]: entry_point}, config_injections=config_injections)
    gmap = create_sdfg_from_internal_ast(own_ast, program, cfg)
    assert gmap.keys() == {entry_point[-1]}
    g: SDFG = singular(v for v in gmap.values())
    # Save once simplifying, in case simplification fails.
    g.save(output_sdfg, compress=output_sdfg.endswith(".sdfgz"))
    g.simplify()
    # Save once before compiling, in case compilation fails.
    g.save(output_sdfg, compress=output_sdfg.endswith(".sdfgz"))
    g.compile()


if __name__ == "__main__":
    main()
