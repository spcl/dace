# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Saves ``velocity_tendencies`` as reloadable ``.sdfgz`` files, one per namelist combination of the data set:

``*_optimized``, parallelized and on no device yet; ``*_cpu``, with levels shared by the team; ``*_gpu``, offloaded. No
megakernel and no timers: these are what build.py starts from. A loaded GPU SDFG needs ``compiler.cuda.default_block_size``
set to 256,1,1 before it is compiled, as build.py does.
"""

import argparse
import copy
import json
import os
from pathlib import Path

from build import (
    ENTRY,
    FIXED_CONFIG,
    NAME,
    SOURCE,
    block_maps,
    keep_device_transients,
    schedule_levels,
    share_levels_not_blocks,
)
from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.build import build_sdfg
from dace_fortran.pipelines import num_loops, num_maps, optimize

from dace.transformation.passes.config_propagation import specialize_variant

VARIANTS = ((0, 1), (0, 2), (1, 1))


def save(sdfg, out: Path, name: str) -> None:
    refreeze(sdfg)
    sdfg.save(str(out / f"{name}.sdfgz"), compress=True)
    bindings = {"interface": sdfg._fortran_interface_raw, "flatten_plan": sdfg._flatten_plan_raw}
    (out / f"{name}.bindings.json").write_text(json.dumps(bindings))
    print(f"{name}: {num_maps(sdfg)} maps, {num_loops(sdfg)} loops", flush=True)


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--dace-fortran", type=Path, default=os.environ.get("DACE_FORTRAN_ROOT"))
    argp.add_argument("--source", default=SOURCE, help="The Fortran, relative to --dace-fortran")
    argp.add_argument("--out", type=Path, required=True)
    args = argp.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = (args.dace_fortran / args.source).read_text()
    for lvn_only, istep in VARIANTS:
        stem = f"velocity_l{lvn_only}i{istep}"
        sdfg = build_sdfg(source, entry=ENTRY, name=NAME, out_dir=str(args.out / "frontend" / stem))
        specialize_variant(sdfg, {"lvn_only": lvn_only, "istep": istep, **FIXED_CONFIG})
        optimize(sdfg)
        gpu = copy.deepcopy(sdfg)
        save(sdfg, args.out, f"{stem}_optimized")
        share_levels_not_blocks(sdfg)
        schedule_levels(sdfg, 0)
        save(sdfg, args.out, f"{stem}_cpu")
        gpu.apply_gpu_transformations(host_maps=block_maps(gpu))
        keep_device_transients(gpu)
        save(gpu, args.out, f"{stem}_gpu")


if __name__ == "__main__":
    main()
