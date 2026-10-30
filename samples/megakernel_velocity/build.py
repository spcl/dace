# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Builds ICON ``velocity_tendencies`` from Fortran through dace-fortran for one configuration of the benchmark.

The Fortran is parsed, one namelist combination is baked in, the SDFG is parallelized, optionally moved to the device
and fused into a megakernel, and compiled into a library with the generated Fortran binding, which ``velocity_driver.F90``
links. ``--strict`` turns off value-changing floating-point optimization, for a bit-exact comparison with Fortran.
"""

import argparse
import os
from pathlib import Path

from dace_fortran.bindings.build_fortran_library import BuildMode, build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.build import build_sdfg
from dace_fortran.pipelines import num_loops, num_maps, optimize

import dace
from dace.dtypes import AllocationLifetime, DeviceType, ScheduleType, StorageType
from dace.sdfg import nodes
from dace.transformation.passes.config_propagation import specialize_variant
from dace.transformation.passes.megakernel import MegaKernelOptimization

ENTRY = "mo_velocity_advection::velocity_tendencies"
NAME = "velocity_tendencies"
#: The layout without ``__LOOP_EXCHANGE``: levels outermost, so ``nlev`` is what a CPU team shares out
SOURCE = "tests/icon/atmosphere/velocity_advection_inlined_no_loop_exchange_single_tu.f90"
#: Namelist values every dump of the data set shares
FIXED_CONFIG = {"ldeepatmo": 0, "lextra_diffu": 1}
LIBRARY = "velocity_inner_wrap"
#: Thread blocks of every GPU kernel, with or without a megakernel
BLOCK_SIZE = "256,1,1"
STRICT_FLAGS = "-fno-fast-math -ffp-contract=off"


def is_block_map(entry: nodes.MapEntry) -> bool:
    """Whether ``entry`` iterates over ICON's blocks, whose bounds are named ``i_startblk`` and ``i_endblk``."""
    names = {str(s) for s in entry.map.range.free_symbols}
    return bool(names) and all("startblk" in name or "endblk" in name for name in names)


def share_levels_not_blocks(sdfg: dace.SDFG) -> None:
    """Runs the block loop of every nest sequentially and makes the maps inside it the parallel kernels: the data set
    has one or two blocks of 20480 columns, so all the parallelism is in the levels and columns."""
    for state in sdfg.states():
        for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]:
            kernels = [
                n
                for node in state.scope_children()[entry]
                if isinstance(node, nodes.NestedSDFG)
                for body in node.sdfg.states()
                for n in body.nodes()
                if isinstance(n, nodes.MapEntry) and body.entry_node(n) is None
            ]
            if kernels and is_block_map(entry):
                entry.map.schedule = ScheduleType.Sequential
                for kernel in kernels:
                    kernel.map.schedule = ScheduleType.CPU_Multicore
            else:
                entry.map.schedule = ScheduleType.CPU_Multicore


def schedule_levels(sdfg: dace.SDFG, chunk: int) -> None:
    """Hands out the levels of every parallel map ``chunk`` at a time; 0 is ``schedule(static)``'s equal blocks."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == ScheduleType.CPU_Multicore:
            node.map.omp_schedule = dace.dtypes.OMPScheduleType.Static
            node.map.omp_chunk_size = chunk


def block_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    """The maps over ICON's blocks. There are few blocks, so each stays on the host and the maps under it are kernels."""
    found = []
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and state.entry_node(node) is None:
                if is_block_map(node):
                    found.append(node)
    return found


def keep_device_transients(sdfg: dace.SDFG) -> None:
    """Allocates the device copies of the arguments once: ICON keeps these fields on the device across calls."""
    for desc in sdfg.arrays.values():
        if desc.transient and desc.storage == StorageType.GPU_Global and desc.free_symbols <= set(sdfg.free_symbols):
            desc.lifetime = AllocationLifetime.Persistent


def crosses(state: dace.SDFGState, source: frozenset[StorageType], target: frozenset[StorageType]) -> bool:
    """Whether ``state`` copies a container of a storage in ``source`` to one in ``target``."""
    arrays = state.sdfg.arrays
    return any(
        arrays[e.src.data].storage in source and arrays[e.dst.data].storage in target
        for e in state.edges()
        if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)
    )


def time_compute_region(sdfg: dace.SDFG, null_region: bool) -> None:
    """Prints the device time between the last copy to the device and the first copy back, every call: device events on
    the stream of the kernels, with the device idle before the first. The copies are an artifact of this benchmark, whose
    fields start on the host, and a megakernel changes only what lies between them. With ``null_region``, the events
    enclose nothing, which measures their own overhead."""
    device, host = frozenset({StorageType.GPU_Global}), frozenset({StorageType.Default, StorageType.CPU_Heap})
    states = [b for b in sdfg.bfs_nodes(sdfg.start_block) if isinstance(b, dace.SDFGState)]
    last_in = [s for s in states if crosses(s, host, device)][-1]
    first_out = next(s for s in states[states.index(last_in) + 1 :] if crosses(s, device, host))

    def add(state: dace.SDFGState, code: str, code_global: str = "") -> None:
        state.add_tasklet(
            state.label, {}, {}, code, language=dace.Language.CPP, code_global=code_global, side_effects=True
        )

    record = """
#include <cstdio>
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#define VT_GPU(name) hip##name
typedef hipEvent_t vt_event_t;
#else
#include <cuda_runtime.h>
#define VT_GPU(name) cuda##name
typedef cudaEvent_t vt_event_t;
#endif
static vt_event_t vt_events[2];
"""
    stream = "gpu_streams[0]"
    start = (
        "VT_GPU(DeviceSynchronize)();\n"
        "if (!vt_events[0]) { VT_GPU(EventCreate)(&vt_events[0]); VT_GPU(EventCreate)(&vt_events[1]); }\n"
        f"VT_GPU(EventRecord)(vt_events[0], {stream});"
    )
    stop = (
        f"VT_GPU(EventRecord)(vt_events[1], {stream});\n"
        "VT_GPU(DeviceSynchronize)();\n"
        "float milliseconds = 0;\n"
        "VT_GPU(EventElapsedTime)(&milliseconds, vt_events[0], vt_events[1]);\n"
        'printf("COMPUTE seconds=%.6f\\n", milliseconds / 1e3);\n'
        "fflush(stdout);"
    )
    if null_region:
        add(sdfg.add_state_after(last_in, "compute_timer"), f"{start}\n{stop}", record)
    else:
        add(sdfg.add_state_after(last_in, "compute_start"), start, record)
        add(sdfg.add_state_before(first_out, "compute_stop"), stop)


def build(
    source: Path,
    out: Path,
    lvn_only: int,
    istep: int,
    device: DeviceType,
    megakernel: bool,
    strict: bool,
    chunk: int,
    block_size: int,
    sync_all: bool,
    null_timer: bool = False,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(source.read_text(), entry=ENTRY, name=NAME, out_dir=str(out / "frontend"))
    specialize_variant(sdfg, {"lvn_only": lvn_only, "istep": istep, **FIXED_CONFIG})
    optimize(sdfg)
    print(f"{out.name}: {num_maps(sdfg)} maps, {num_loops(sdfg)} loops", flush=True)
    if device == DeviceType.CPU:
        share_levels_not_blocks(sdfg)
    if device == DeviceType.GPU:
        sdfg.apply_gpu_transformations(host_maps=block_maps(sdfg))
        keep_device_transients(sdfg)
    if megakernel:
        fuse = MegaKernelOptimization(device=device)
        fuse.block_size = block_size
        fuse.sync_all = sync_all
        chains = fuse.apply_pass(sdfg, {})
        print(f"{out.name}: {len(chains or [])} megakernels", flush=True)
    if device == DeviceType.CPU:
        schedule_levels(sdfg, chunk)
    if device == DeviceType.GPU:
        time_compute_region(sdfg, null_timer)
    refreeze(sdfg)
    sdfg.build_folder = str(out / "sdfg")
    sdfg.save(str(out / f"{NAME}.sdfgz"), compress=True)
    iface = build_auto_interface(sdfg._fortran_interface_raw, NAME)
    plan = FlattenPlan.from_dict(sdfg._flatten_plan_raw or {})
    mode = BuildMode.DEBUG if strict else BuildMode.RELEASE
    build_fortran_library(
        sdfg, iface=iface, plan=plan, out_dir=str(out), name=LIBRARY, prelude_sources=[source], mode=mode
    )


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argp.add_argument("--dace-fortran", type=Path, default=os.environ.get("DACE_FORTRAN_ROOT"))
    argp.add_argument("--out", type=Path, required=True)
    argp.add_argument("--lvn-only", type=int, required=True, choices=[0, 1])
    argp.add_argument("--istep", type=int, required=True, choices=[1, 2])
    argp.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    argp.add_argument("--megakernel", action="store_true")
    argp.add_argument("--strict", action="store_true")
    argp.add_argument("--block-size", type=int, default=256, help="Threads per block of a GPU megakernel")
    argp.add_argument("--fewer-syncs", action="store_true", help="Grid barriers only between dependent stages")
    argp.add_argument("--chiplets", type=int, default=0, help="GPU: chiplets to spread thread blocks over; 0 detects")
    argp.add_argument("--null-timer", action="store_true", help="GPU: time an empty region, for the timer's overhead")
    argp.add_argument("--chunk", type=int, default=0, help="CPU: levels per schedule(static, chunk); 0 is equal blocks")
    args = argp.parse_args()
    if args.strict:
        flags = dace.Config.get("compiler", "cpu", "args")
        dace.Config.set("compiler", "cpu", "args", value=f"{flags} {STRICT_FLAGS}")
    device = DeviceType.GPU if args.device == "gpu" else DeviceType.CPU
    if device == DeviceType.GPU:
        dace.Config.set("compiler", "cuda", "default_block_size", value=BLOCK_SIZE)
        dace.Config.set("compiler", "cuda", "chiplet_number", value=args.chiplets)
    build(
        args.dace_fortran / SOURCE,
        args.out,
        args.lvn_only,
        args.istep,
        device,
        args.megakernel,
        args.strict,
        args.chunk,
        args.block_size,
        not args.fewer_syncs,
        args.null_timer,
    )


if __name__ == "__main__":
    main()
