import dace
import typing
import os
import numpy as np
from dace.transformation.dataflow import DoubleBuffering, MapTiling
from dace.transformation.soft_hier import SystolocTransformer, SystolicTransformer, SystolicSplitStore, SummaTransformer, BSPTransformer, SplitKReduction

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")


def _my_gen_summa_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                              global_storage: dace.dtypes.StorageType,
                              local_storage: dace.dtypes.StorageType,
                              device_schedule: dace.dtypes.ScheduleType,
                              thread_group_schedule: dace.dtypes.ScheduleType,
                              thread_group_dims: typing.Tuple,
                              hbm_split_scheme: typing.List[typing.Tuple[int, int]],
                              hbm_placement_scheme: typing.List[typing.Tuple[int, int]],
                              input_float,
                              output_float,
                              coarsening_factor,
                              mmad_tasklet_str: str,
                              GEMM_shape: typing.Tuple = None,
                              is_hbm_interleaved: bool = False):
    if GEMM_shape is not None:
        (M, N, K) = GEMM_shape
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        if arr_name == "A":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[0],
                                       hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[1],
                                       hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[2],
                                       hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(coarsening_factor * coarsening_factor, tM // coarsening_factor,
                                      tN // coarsening_factor),
                               dtype=ftype,
                               storage=local_storage,
                               transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(name="gemm_entry",
                                                     ndrange={
                                                         "i": dace.subsets.Range([(0, M - 1, tM * gM)]),
                                                         "j": dace.subsets.Range([(0, N - 1, tN * gN)])
                                                     },
                                                     schedule=device_schedule)
    i = dace.symbol("i")
    j = dace.symbol("j")

    for name in ["A", "B", "C"]:
        # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(name="thread_group_mmad",
                                                                       ndrange={
                                                                           "gi": dace.subsets.Range([(0, gM - 1, 1)]),
                                                                           "gj": dace.subsets.Range([(0, gN - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

    gi = dace.symbol("gi")
    gj = dace.symbol("gj")

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={
            "ci": dace.subsets.Range([(0, tM - 1, tM // coarsening_factor)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN // coarsening_factor)])
        },
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            elif name == "C":
                access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_entry.add_out_connector(f"OUT_{name}")
            thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK": dace.subsets.Range([(0, K - 1, tK // coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                "0:K"
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM // coarsening_factor, tK // coarsening_factor)),
                        ("B", (tK // coarsening_factor, tN // coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}",
                                   shape=shape,
                                   dtype=input_float,
                                   storage=local_storage,
                                   transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                f"bK:bK+{tK//coarsening_factor}"
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK//coarsening_factor}",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule",
                                       inputs={"_in_local_a", "_in_local_b", "_in_accumulator"},
                                       outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str,
                                       language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator",
                   dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    # assign_tasklet = state.add_tasklet(name="assign", inputs={"_in_accumulator"}, outputs={"_out_C"}, code="_out_C = _in_accumulator")
    # state.add_edge(block_tiled_map_exit, "OUT_C", assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator")) , "_in_C"

    # accumulator_an3 = state.add_access("accumulator")
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an3, None, assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", assign_tasklet, "_in_accumulator", dace.memlet.Memlet())

    # c_an2 = state.add_access("C")
    accumulator_an3 = state.add_access("accumulator")

    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    SummaTransformer.apply_to(sdfg,
                              map_entry=block_tiled_map_entry,
                              transient=local_access_nodes["local_A"],
                              options={
                                  "npe": gM,
                                  "gi": gi,
                                  "gj": gj,
                                  "i": i,
                                  "j": j,
                                  "M": M,
                                  "N": N,
                                  "K": K,
                                  "tM": tM,
                                  "tN": tN,
                                  "tK": tK
                              })

    return sdfg


def _my_gen_systolic_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                                 global_storage: dace.dtypes.StorageType,
                                 local_storage: dace.dtypes.StorageType,
                                 device_schedule: dace.dtypes.ScheduleType,
                                 thread_group_schedule: dace.dtypes.ScheduleType,
                                 thread_group_dims: typing.Tuple,
                                 hbm_split_scheme: typing.List[typing.Tuple[int, int]],
                                 hbm_placement_scheme: typing.List[typing.Tuple[int, int]],
                                 input_float,
                                 output_float,
                                 coarsening_factor,
                                 mmad_tasklet_str: str,
                                 GEMM_shape: typing.Tuple = None,
                                 is_hbm_interleaved: bool = False):
    if GEMM_shape is not None:
        (M, N, K) = GEMM_shape
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        if arr_name == "A":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[0],
                                       hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[1],
                                       hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[2],
                                       hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(coarsening_factor * coarsening_factor, tM // coarsening_factor,
                                      tN // coarsening_factor),
                               dtype=ftype,
                               storage=local_storage,
                               transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(name="gemm_entry",
                                                     ndrange={
                                                         "i": dace.subsets.Range([(0, M - 1, tM * gM)]),
                                                         "j": dace.subsets.Range([(0, N - 1, tN * gN)])
                                                     },
                                                     schedule=device_schedule)
    i = dace.symbol("i")
    j = dace.symbol("j")

    for name in ["A", "B", "C"]:
        # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(name="thread_group_mmad",
                                                                       ndrange={
                                                                           "gi": dace.subsets.Range([(0, gM - 1, 1)]),
                                                                           "gj": dace.subsets.Range([(0, gN - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

    gi = dace.symbol("gi")
    gj = dace.symbol("gj")

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={
            "ci": dace.subsets.Range([(0, tM - 1, tM // coarsening_factor)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN // coarsening_factor)])
        },
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            elif name == "C":
                access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_entry.add_out_connector(f"OUT_{name}")
            thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK": dace.subsets.Range([(0, K - 1, tK // coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                "0:K"
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM // coarsening_factor, tK // coarsening_factor)),
                        ("B", (tK // coarsening_factor, tN // coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}",
                                   shape=shape,
                                   dtype=input_float,
                                   storage=local_storage,
                                   transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                f"bK:bK+{tK//coarsening_factor}"
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK//coarsening_factor}",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule",
                                       inputs={"_in_local_a", "_in_local_b", "_in_accumulator"},
                                       outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str,
                                       language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator",
                   dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    # assign_tasklet = state.add_tasklet(name="assign", inputs={"_in_accumulator"}, outputs={"_out_C"}, code="_out_C = _in_accumulator")
    # state.add_edge(block_tiled_map_exit, "OUT_C", assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator")) , "_in_C"

    # accumulator_an3 = state.add_access("accumulator")
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an3, None, assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", assign_tasklet, "_in_accumulator", dace.memlet.Memlet())

    # c_an2 = state.add_access("C")
    accumulator_an3 = state.add_access("accumulator")

    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    SystolicTransformer.apply_to(sdfg,
                                 map_entry=block_tiled_map_entry,
                                 transient=local_access_nodes["local_A"],
                                 options={
                                     "npe": gM,
                                     "gi": gi,
                                     "gj": gj,
                                     "i": i,
                                     "j": j,
                                     "M": M,
                                     "N": N,
                                     "tM": tM,
                                     "tN": tN
                                 })
    SystolicSplitStore.apply_to(sdfg,
                                map_entry=thread_coarsened_map_entry,
                                accumulator=accumulator_an,
                                options={
                                    "npe": gM,
                                    "gi": gi,
                                    "gj": gj,
                                    "i": i,
                                    "j": j,
                                    "M": M,
                                    "N": N,
                                    "K": K,
                                    "tM": tM,
                                    "tN": tN,
                                    "tK": tK
                                })
    return sdfg


def _my_gen_baseline_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                                 global_storage: dace.dtypes.StorageType,
                                 local_storage: dace.dtypes.StorageType,
                                 device_schedule: dace.dtypes.ScheduleType,
                                 thread_group_schedule: dace.dtypes.ScheduleType,
                                 thread_group_dims: typing.Tuple,
                                 hbm_split_scheme: typing.List[typing.Tuple[int, int]],
                                 hbm_placement_scheme: typing.List[typing.Tuple[int, int]],
                                 input_float,
                                 output_float,
                                 coarsening_factor,
                                 mmad_tasklet_str: str,
                                 is_hbm_interleaved: bool = False,
                                 GEMM_shape: typing.Tuple = None):
    if GEMM_shape is not None:
        (M, N, K) = GEMM_shape
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        if arr_name == "A":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[0],
                                       hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[1],
                                       hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[2],
                                       hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(coarsening_factor * coarsening_factor, tM // coarsening_factor,
                                      tN // coarsening_factor),
                               dtype=ftype,
                               storage=local_storage,
                               transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(name="gemm_entry",
                                                     ndrange={
                                                         "i": dace.subsets.Range([(0, M - 1, tM * gM)]),
                                                         "j": dace.subsets.Range([(0, N - 1, tN * gN)])
                                                     },
                                                     schedule=device_schedule)
    i = dace.symbol("i")
    j = dace.symbol("j")

    for name in ["A", "B", "C"]:
        # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(name="thread_group_mmad",
                                                                       ndrange={
                                                                           "gi": dace.subsets.Range([(0, gM - 1, 1)]),
                                                                           "gj": dace.subsets.Range([(0, gN - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

    gi = dace.symbol("gi")
    gj = dace.symbol("gj")

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={
            "ci": dace.subsets.Range([(0, tM - 1, tM // coarsening_factor)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN // coarsening_factor)])
        },
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            elif name == "C":
                access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_entry.add_out_connector(f"OUT_{name}")
            thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK": dace.subsets.Range([(0, K - 1, tK // coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                "0:K"
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM // coarsening_factor, tK // coarsening_factor)),
                        ("B", (tK // coarsening_factor, tN // coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}",
                                   shape=shape,
                                   dtype=input_float,
                                   storage=local_storage,
                                   transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                f"bK:bK+{tK//coarsening_factor}"
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK//coarsening_factor}",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule",
                                       inputs={"_in_local_a", "_in_local_b", "_in_accumulator"},
                                       outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str,
                                       language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator",
                   dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    # assign_tasklet = state.add_tasklet(name="assign", inputs={"_in_accumulator"}, outputs={"_out_C"}, code="_out_C = _in_accumulator")
    # state.add_edge(block_tiled_map_exit, "OUT_C", assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator")) , "_in_C"

    # accumulator_an3 = state.add_access("accumulator")
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an3, None, assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", assign_tasklet, "_in_accumulator", dace.memlet.Memlet())

    # c_an2 = state.add_access("C")
    accumulator_an3 = state.add_access("accumulator")

    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    return sdfg


def _my_gen_double_buffer_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                                      global_storage: dace.dtypes.StorageType,
                                      local_storage: dace.dtypes.StorageType,
                                      device_schedule: dace.dtypes.ScheduleType,
                                      thread_group_schedule: dace.dtypes.ScheduleType,
                                      thread_group_dims: typing.Tuple,
                                      hbm_split_scheme: typing.List[typing.Tuple[int, int]],
                                      hbm_placement_scheme: typing.List[typing.Tuple[int, int]],
                                      input_float,
                                      output_float,
                                      coarsening_factor,
                                      mmad_tasklet_str: str,
                                      GEMM_shape=None,
                                      is_hbm_interleaved: bool = False):
    sdfg = dace.SDFG("GEMM")
    if GEMM_shape is not None:
        (M, N, K) = GEMM_shape
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        if arr_name == "A":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[0],
                                       hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[1],
                                       hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[2],
                                       hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(coarsening_factor * coarsening_factor, tM // coarsening_factor,
                                      tN // coarsening_factor),
                               dtype=ftype,
                               storage=local_storage,
                               transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(name="gemm_entry",
                                                     ndrange={
                                                         "i": dace.subsets.Range([(0, M - 1, tM * gM)]),
                                                         "j": dace.subsets.Range([(0, N - 1, tN * gN)])
                                                     },
                                                     schedule=device_schedule)
    i = dace.symbol("i")
    j = dace.symbol("j")

    for name in ["A", "B", "C"]:
        # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(name="thread_group_mmad",
                                                                       ndrange={
                                                                           "gi": dace.subsets.Range([(0, gM - 1, 1)]),
                                                                           "gj": dace.subsets.Range([(0, gN - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

    gi = dace.symbol("gi")
    gj = dace.symbol("gj")

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={
            "ci": dace.subsets.Range([(0, tM - 1, tM // coarsening_factor)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN // coarsening_factor)])
        },
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            elif name == "C":
                access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_entry.add_out_connector(f"OUT_{name}")
            thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK": dace.subsets.Range([(0, K - 1, tK // coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                "0:K"
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM // coarsening_factor, tK // coarsening_factor)),
                        ("B", (tK // coarsening_factor, tN // coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}",
                                   shape=shape,
                                   dtype=input_float,
                                   storage=local_storage,
                                   transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                f"bK:bK+{tK//coarsening_factor}"
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK//coarsening_factor}",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule",
                                       inputs={"_in_local_a", "_in_local_b", "_in_accumulator"},
                                       outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str,
                                       language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator",
                   dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    # assign_tasklet = state.add_tasklet(name="assign", inputs={"_in_accumulator"}, outputs={"_out_C"}, code="_out_C = _in_accumulator")
    # state.add_edge(block_tiled_map_exit, "OUT_C", assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator")) , "_in_C"

    # accumulator_an3 = state.add_access("accumulator")
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an3, None, assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", assign_tasklet, "_in_accumulator", dace.memlet.Memlet())

    # c_an2 = state.add_access("C")
    accumulator_an3 = state.add_access("accumulator")

    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    DoubleBuffering.apply_to(sdfg, map_entry=block_tiled_map_entry, transient=local_access_nodes["local_A"])
    return sdfg


def _my_gen_BSP_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                            global_storage: dace.dtypes.StorageType,
                            local_storage: dace.dtypes.StorageType,
                            device_schedule: dace.dtypes.ScheduleType,
                            thread_group_schedule: dace.dtypes.ScheduleType,
                            thread_group_dims: typing.Tuple,
                            hbm_split_scheme: typing.List[typing.Tuple[int, int]],
                            hbm_placement_scheme: typing.List[typing.Tuple[int, int]],
                            input_float,
                            output_float,
                            coarsening_factor,
                            mmad_tasklet_str: str,
                            BSP_generator_func: typing.Callable[..., any],
                            summa_range=None,
                            systolic_range=None,
                            n_streams=None,
                            direction='y',
                            GEMM_shape=None,
                            is_hbm_interleaved: bool = False):
    if GEMM_shape is not None:
        (M, N, K) = GEMM_shape
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        if arr_name == "A":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[0],
                                       hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[1],
                                       hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[2],
                                       hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(coarsening_factor * coarsening_factor, tM // coarsening_factor,
                                      tN // coarsening_factor),
                               dtype=ftype,
                               storage=local_storage,
                               transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(name="gemm_entry",
                                                     ndrange={
                                                         "i": dace.subsets.Range([(0, M - 1, tM * gM)]),
                                                         "j": dace.subsets.Range([(0, N - 1, tN * gN)])
                                                     },
                                                     schedule=device_schedule)
    i = dace.symbol("i")
    j = dace.symbol("j")

    for name in ["A", "B", "C"]:
        # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(name="thread_group_mmad",
                                                                       ndrange={
                                                                           "gi": dace.subsets.Range([(0, gM - 1, 1)]),
                                                                           "gj": dace.subsets.Range([(0, gN - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

    gi = dace.symbol("gi")
    gj = dace.symbol("gj")

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={
            "ci": dace.subsets.Range([(0, tM - 1, tM // coarsening_factor)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN // coarsening_factor)])
        },
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            elif name == "C":
                access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_entry.add_out_connector(f"OUT_{name}")
            thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK": dace.subsets.Range([(0, K - 1, tK // coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                "0:K"
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM // coarsening_factor, tK // coarsening_factor)),
                        ("B", (tK // coarsening_factor, tN // coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}",
                                   shape=shape,
                                   dtype=input_float,
                                   storage=local_storage,
                                   transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([
                f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                f"bK:bK+{tK//coarsening_factor}"
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK//coarsening_factor}",
                f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule",
                                       inputs={"_in_local_a", "_in_local_b", "_in_accumulator"},
                                       outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str,
                                       language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator",
                   dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    # assign_tasklet = state.add_tasklet(name="assign", inputs={"_in_accumulator"}, outputs={"_out_C"}, code="_out_C = _in_accumulator")
    # state.add_edge(block_tiled_map_exit, "OUT_C", assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator")) , "_in_C"

    # accumulator_an3 = state.add_access("accumulator")
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an3, None, assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", assign_tasklet, "_in_accumulator", dace.memlet.Memlet())

    # c_an2 = state.add_access("C")
    accumulator_an3 = state.add_access("accumulator")

    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    if summa_range is not None:
        (pre_shift_code_block, BSP_stride, BSP_init_code_block, BSP_loop_code_block, BSP_compute_code_block,
         BSP_communication_code_block, BSP_sync,
         post_shift_code_block) = BSP_generator_func(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K, summa_range)
    elif systolic_range is not None:
        (pre_shift_code_block, BSP_stride, BSP_init_code_block, BSP_loop_code_block, BSP_compute_code_block,
         BSP_communication_code_block, BSP_sync,
         post_shift_code_block) = BSP_generator_func(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K, systolic_range)
    elif n_streams is not None:
        (pre_shift_code_block, BSP_stride, BSP_init_code_block, BSP_loop_code_block, BSP_compute_code_block,
         BSP_communication_code_block, BSP_sync, post_shift_code_block) = BSP_generator_func(i,
                                                                                             j,
                                                                                             gi,
                                                                                             gj,
                                                                                             gM,
                                                                                             gN,
                                                                                             tM,
                                                                                             tN,
                                                                                             tK,
                                                                                             M,
                                                                                             N,
                                                                                             K,
                                                                                             n_streams=n_streams,
                                                                                             direction=direction)
    else:
        (pre_shift_code_block, BSP_stride, BSP_init_code_block, BSP_loop_code_block, BSP_compute_code_block,
         BSP_communication_code_block, BSP_sync,
         post_shift_code_block) = BSP_generator_func(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K)
    # BSP_compute_code_block = None
    BSPTransformer.apply_to(sdfg,
                            accumulator=accumulator_an,
                            map_entry=block_tiled_map_entry,
                            transient=local_access_nodes["local_A"],
                            options={
                                "npe_x": gM,
                                "npe_y": gN,
                                "gi": gi,
                                "gj": gj,
                                "i": i,
                                "j": j,
                                "M": M,
                                "N": N,
                                "K": K,
                                "tM": tM,
                                "tN": tN,
                                "tK": tK,
                                "pre_shift": pre_shift_code_block,
                                "BSP_stride": BSP_stride,
                                "BSP_init": BSP_init_code_block,
                                "BSP_loop": BSP_loop_code_block,
                                "BSP_compute": BSP_compute_code_block,
                                "BSP_communication": BSP_communication_code_block,
                                "BSP_sync": BSP_sync,
                                "post_shift": post_shift_code_block,
                            })

    return sdfg


def _my_gen_split_K_BSP_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                                    global_storage: dace.dtypes.StorageType,
                                    local_storage: dace.dtypes.StorageType,
                                    device_schedule: dace.dtypes.ScheduleType,
                                    thread_group_schedule: dace.dtypes.ScheduleType,
                                    thread_group_dims: typing.Tuple,
                                    k_group_dims: typing.Tuple,
                                    hbm_split_scheme: typing.List[typing.Tuple[int, int]],
                                    hbm_placement_scheme: typing.List[typing.Tuple[int, int]],
                                    input_float,
                                    output_float,
                                    mmad_tasklet_str: str,
                                    BSP_generator_func: typing.Callable[..., any],
                                    coarsening_factor=1,
                                    is_hbm_interleaved: bool = False,
                                    GEMM_shape=None,
                                    reduce_condition_func=None,
                                    summa_range=(1, 32)):
    if GEMM_shape is not None:
        M, N, K = GEMM_shape

    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    kg_m, kg_n = k_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        if arr_name == "A":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[0],
                                       hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[1],
                                       hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False,
                                       is_hbm_interleaved=is_hbm_interleaved,
                                       hbm_split_scheme=hbm_split_scheme[2],
                                       hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(coarsening_factor * coarsening_factor, tM // coarsening_factor,
                                      tN // coarsening_factor),
                               dtype=output_float,
                               storage=local_storage,
                               transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(name="gemm_entry",
                                                     ndrange={
                                                         "i": dace.subsets.Range([(0, M - 1, tM * gM // kg_m)]),
                                                         "j": dace.subsets.Range([(0, N - 1, tN * gN // kg_n)])
                                                     },
                                                     schedule=device_schedule)
    i = dace.symbol("i")
    j = dace.symbol("j")

    for name in ["A", "B", "C"]:
        # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(name="thread_group_mmad",
                                                                       ndrange={
                                                                           "gi": dace.subsets.Range([(0, gM - 1, 1)]),
                                                                           "gj": dace.subsets.Range([(0, gN - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

    gi = dace.symbol("gi")
    gj = dace.symbol("gj")
    kg_i = gi // kg_m
    kg_j = gj // kg_n
    kg_oi = gi % kg_m
    kg_oj = gj % kg_n
    kg_num = kg_m * kg_n
    kg_off = kg_oi * kg_n + kg_oj
    bK_start = kg_off * (K // kg_num)
    bK_end = (kg_off + 1) * (K // kg_num)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM} / {kg_m}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN} / {kg_n}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM} / {kg_m}", f"j:j + {gN} * {tN} / {kg_n}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM} / {kg_m}", f"j:j + {gN} * {tN} / {kg_n}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={
            "ci": dace.subsets.Range([(0, tM - 1, tM // coarsening_factor)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN // coarsening_factor)])
        },
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i + {kg_i} * {tM}:i + {kg_i} * {tM} + {tM}", "0:K"])
                access_subsets = dace.subsets.Range([(f"i + {kg_i} * {tM}", f"i + {kg_i} * {tM} + {tM} - 1", 1),
                                                     (0, K - 1, 1)])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j + {kg_j} * {tN}:j + {kg_j} * {tN} + {tN}"])
                access_subsets = dace.subsets.Range([(0, K - 1, 1),
                                                     (f"j + {kg_j} * {tN}", f"j + {kg_j} * {tN} + {tN} - 1", 1)])
            state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}",
                           dace.memlet.Memlet(data=f"{name}", subset=access_subsets))
            thread_group_map_entry.add_out_connector(f"OUT_{name}")
            thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join(
                [f"i + {kg_i} * {tM}:i + {kg_i} * {tM} + {tM}", f"j + {kg_j} * {tN}:j + {kg_j} * {tN} + {tN}"])
            access_subsets = dace.subsets.Range([(f"i + {kg_i} * {tM}", f"i + {kg_i} * {tM} + {tM} - 1", 1),
                                                 (f"j + {kg_j} * {tN}", f"j + {kg_j} * {tN} + {tN} - 1", 1)])
            # state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}",
                           dace.memlet.Memlet(data=f"{name}", subset=access_subsets))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        # ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        ndrange={"bK": dace.subsets.Range([(bK_start, bK_end - 1, tK // coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor}:i + {kg_i} * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                "0:K"
            ])
            access_subsets = dace.subsets.Range([
                (f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor}",
                 f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor} - 1", 1), (0, K - 1, 1)
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor}:j + {kg_j} * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
            access_subsets = dace.subsets.Range([
                (0, K - 1, 1),
                (f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor}",
                 f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor} - 1", 1)
            ])

        # state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(data=f"{name}", subset=access_subsets))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM // coarsening_factor, tK // coarsening_factor)),
                        ("B", (tK // coarsening_factor, tN // coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}",
                                   shape=shape,
                                   dtype=input_float,
                                   storage=local_storage,
                                   transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([
                f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor}:i + {kg_i} * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
                f"bK:bK+{tK//coarsening_factor}"
            ])
            access_subsets = dace.subsets.Range([
                (f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor}",
                 f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}-1", 1),
                (f"bK", f"bK+{tK//coarsening_factor}-1", 1)
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK//coarsening_factor}",
                f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor}:j + {kg_j} * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
            ])
            access_subsets = dace.subsets.Range([
                (f"bK", f"bK+{tK//coarsening_factor}-1", 1),
                (f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor}",
                 f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}-1", 1)
            ])
        # state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None,
                       dace.memlet.Memlet(data=f"{name}", subset=access_subsets))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule",
                                       inputs={"_in_local_a", "_in_local_b", "_in_accumulator"},
                                       outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str,
                                       language=dace.dtypes.Language.CPP)

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator",
                   dace.memlet.Memlet("accumulator"))
    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    accumulator_an3 = state.add_access("accumulator")

    access_str = ", ".join(
        [f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor}:i + {kg_i} * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor}:j + {kg_j} * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    access_subsets = dace.subsets.Range([
        (f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor}",
         f"i + {kg_i} * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}-1", 1),
        (f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor}",
         f"j + {kg_j} * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}-1", 1)
    ])
    # state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C",
                   dace.memlet.Memlet(data="C", subset=access_subsets, wcr="lambda a, b: a + b"))
    thread_coarsened_map_exit.add_in_connector("IN_C")
    if reduce_condition_func is None:
        reduce_condition = None
    else:
        reduce_condition = reduce_condition_func(gi, gj, kg_i, kg_j, kg_m, kg_n, gM, gN)
    SplitKReduction.apply_to(sdfg,
                             accumulator=accumulator_an3,
                             global_hbm=anc3,
                             options={
                                 "npe_x": gM,
                                 "npe_y": gN,
                                 "gi": gi,
                                 "gj": gj,
                                 "i": i,
                                 "j": j,
                                 "M": M,
                                 "N": N,
                                 "K": K,
                                 "tM": tM,
                                 "tN": tN,
                                 "tK": tK,
                                 "kg_m": kg_m,
                                 "kg_n": kg_n,
                                 "reduce_cond": reduce_condition
                             })
    # return sdfg
    if summa_range is not None:
        (pre_shift_code_block, BSP_stride, BSP_init_code_block, BSP_loop_code_block, BSP_compute_code_block,
         BSP_communication_code_block, BSP_sync, post_shift_code_block) = BSP_generator_func(i,
                                                                                             j,
                                                                                             gi,
                                                                                             gj,
                                                                                             gM,
                                                                                             gN,
                                                                                             tM,
                                                                                             tN,
                                                                                             tK,
                                                                                             M,
                                                                                             N,
                                                                                             K,
                                                                                             k_group_dims=k_group_dims,
                                                                                             summa_range=summa_range)
    else:
        (pre_shift_code_block, BSP_stride, BSP_init_code_block, BSP_loop_code_block, BSP_compute_code_block,
         BSP_communication_code_block, BSP_sync,
         post_shift_code_block) = BSP_generator_func(i, j, gi, gj, gM, gN, tM, tN, tK, M, N, K)

    BSPTransformer.apply_to(sdfg,
                            accumulator=accumulator_an,
                            map_entry=block_tiled_map_entry,
                            transient=local_access_nodes["local_A"],
                            options={
                                "npe_x": gM,
                                "npe_y": gN,
                                "gi": gi,
                                "gj": gj,
                                "i": i,
                                "j": j,
                                "M": M,
                                "N": N,
                                "K": K,
                                "tM": tM,
                                "tN": tN,
                                "tK": tK,
                                "pre_shift": pre_shift_code_block,
                                "BSP_stride": BSP_stride,
                                "BSP_init": BSP_init_code_block,
                                "BSP_loop": BSP_loop_code_block,
                                "BSP_compute": BSP_compute_code_block,
                                "BSP_communication": BSP_communication_code_block,
                                "BSP_sync": BSP_sync,
                                "post_shift": post_shift_code_block,
                            })

    return sdfg
