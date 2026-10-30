# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compiles a ``.sdfgz`` of save_baselines.py into the library velocity_driver.F90 links, without the frontend or the
optimization pipeline. Transform the loaded SDFG between ``from_file`` and the build to try something on it."""

import argparse
import json
import os
from pathlib import Path

from build import BLOCK_SIZE, LIBRARY, NAME, SOURCE, STRICT_FLAGS
from dace_fortran.bindings.build_fortran_library import BuildMode, build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface

import dace


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("sdfg", type=Path)
    argp.add_argument("--dace-fortran", type=Path, default=os.environ.get("DACE_FORTRAN_ROOT"))
    argp.add_argument("--out", type=Path, required=True)
    argp.add_argument("--chiplets", type=int, default=0)
    argp.add_argument("--strict", action="store_true")
    args = argp.parse_args()
    if args.strict:
        flags = dace.Config.get("compiler", "cpu", "args")
        dace.Config.set("compiler", "cpu", "args", value=f"{flags} {STRICT_FLAGS}")
    dace.Config.set("compiler", "cuda", "default_block_size", value=BLOCK_SIZE)
    dace.Config.set("compiler", "cuda", "chiplet_number", value=args.chiplets)
    sdfg = dace.SDFG.from_file(str(args.sdfg))
    sdfg.validate()
    bindings = json.loads(args.sdfg.with_suffix(".bindings.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    sdfg.build_folder = str(args.out / "sdfg")
    build_fortran_library(
        sdfg,
        iface=build_auto_interface(bindings["interface"], NAME),
        plan=FlattenPlan.from_dict(bindings["flatten_plan"] or {}),
        out_dir=str(args.out),
        name=LIBRARY,
        prelude_sources=[args.dace_fortran / SOURCE],
        mode=BuildMode.DEBUG if args.strict else BuildMode.RELEASE,
    )


if __name__ == "__main__":
    main()
