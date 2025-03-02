import dace
import typing
import os
import numpy as np
from dace.transformation.dataflow import DoubleBuffering, MapTiling
from dace.transformation.soft_hier import SystolocTransformer, SystolicTransformer, SystolicSplitStore, SummaTransformer

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
                     is_hbm_interleaved: bool = False):
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
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[0], hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[1], hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[2], hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator", shape=(coarsening_factor*coarsening_factor, tM//coarsening_factor, tN//coarsening_factor), dtype=ftype, storage=local_storage, transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(
        name="gemm_entry",
        ndrange={"i" : dace.subsets.Range([(0, M-1, tM*gM)]),
                 "j" : dace.subsets.Range([(0, N-1, tN*gN)])},
        schedule=device_schedule
    )
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

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(
        name="thread_group_mmad",
        ndrange={"gi" : dace.subsets.Range([(0, gM-1, 1)]),
                 "gj" : dace.subsets.Range([(0, gM-1, 1)])},
        schedule=thread_group_schedule
    )

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
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={"ci" : dace.subsets.Range([(0, tM-1, tM//coarsening_factor)]),
                 "cj" : dace.subsets.Range([(0, tN-1, tN//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
                elif name == "B":
                    access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                thread_group_map_entry.add_out_connector(f"OUT_{name}")
                thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry


    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", "0:K"])
        elif name == "B":
            access_str = ", ".join(["0:K", f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")


    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM//coarsening_factor, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN//coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                                    f"bK:bK+{tK//coarsening_factor}"])
        elif name == "B":
            access_str = ", ".join([f"bK:bK+{tK//coarsening_factor}", 
                                    f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule", inputs={"_in_local_a", "_in_local_b", "_in_accumulator"}, outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str, language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
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
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                            f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")
    
    SummaTransformer.apply_to(sdfg, map_entry=block_tiled_map_entry, transient=local_access_nodes["local_A"], options={"npe": gM, "gi": gi, "gj": gj,
                                                                                                                          "i": i, "j": j,
                                                                                                                          "M": M, "N": N, "K": K,
                                                                                                                          "tM": tM, "tN": tN, "tK": tK})
    
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
                     is_hbm_interleaved: bool = False):
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
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[0], hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[1], hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[2], hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator", shape=(coarsening_factor*coarsening_factor, tM//coarsening_factor, tN//coarsening_factor), dtype=ftype, storage=local_storage, transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(
        name="gemm_entry",
        ndrange={"i" : dace.subsets.Range([(0, M-1, tM*gM)]),
                 "j" : dace.subsets.Range([(0, N-1, tN*gN)])},
        schedule=device_schedule
    )
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

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(
        name="thread_group_mmad",
        ndrange={"gi" : dace.subsets.Range([(0, gM-1, 1)]),
                 "gj" : dace.subsets.Range([(0, gM-1, 1)])},
        schedule=thread_group_schedule
    )

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
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={"ci" : dace.subsets.Range([(0, tM-1, tM//coarsening_factor)]),
                 "cj" : dace.subsets.Range([(0, tN-1, tN//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
                elif name == "B":
                    access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                thread_group_map_entry.add_out_connector(f"OUT_{name}")
                thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry


    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", "0:K"])
        elif name == "B":
            access_str = ", ".join(["0:K", f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")


    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM//coarsening_factor, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN//coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                                    f"bK:bK+{tK//coarsening_factor}"])
        elif name == "B":
            access_str = ", ".join([f"bK:bK+{tK//coarsening_factor}", 
                                    f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule", inputs={"_in_local_a", "_in_local_b", "_in_accumulator"}, outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str, language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
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
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                            f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")
    
    SystolicTransformer.apply_to(sdfg, map_entry=block_tiled_map_entry, transient=local_access_nodes["local_A"], options={"npe": gM, "gi": gi, "gj": gj, 
                                                                                                                          "i": i, "j": j, 
                                                                                                                          "M": M, "N": N, 
                                                                                                                          "tM": tM, "tN": tN})
    SystolicSplitStore.apply_to(sdfg, map_entry=thread_coarsened_map_entry, accumulator=accumulator_an, options={"npe": gM, "gi": gi, "gj": gj,
                                                                                                                          "i": i, "j": j,
                                                                                                                          "M": M, "N": N, "K": K,
                                                                                                                          "tM": tM, "tN": tN, "tK": tK})
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
                     is_hbm_interleaved: bool = False):
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
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[0], hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[1], hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[2], hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator", shape=(coarsening_factor*coarsening_factor, tM//coarsening_factor, tN//coarsening_factor), dtype=ftype, storage=local_storage, transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(
        name="gemm_entry",
        ndrange={"i" : dace.subsets.Range([(0, M-1, tM*gM)]),
                 "j" : dace.subsets.Range([(0, N-1, tN*gN)])},
        schedule=device_schedule
    )
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

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(
        name="thread_group_mmad",
        ndrange={"gi" : dace.subsets.Range([(0, gM-1, 1)]),
                 "gj" : dace.subsets.Range([(0, gM-1, 1)])},
        schedule=thread_group_schedule
    )

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
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={"ci" : dace.subsets.Range([(0, tM-1, tM//coarsening_factor)]),
                 "cj" : dace.subsets.Range([(0, tN-1, tN//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
                elif name == "B":
                    access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                thread_group_map_entry.add_out_connector(f"OUT_{name}")
                thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry


    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", "0:K"])
        elif name == "B":
            access_str = ", ".join(["0:K", f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")


    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM//coarsening_factor, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN//coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                                    f"bK:bK+{tK//coarsening_factor}"])
        elif name == "B":
            access_str = ", ".join([f"bK:bK+{tK//coarsening_factor}", 
                                    f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule", inputs={"_in_local_a", "_in_local_b", "_in_accumulator"}, outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str, language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
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
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                            f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
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
                     is_hbm_interleaved: bool = False):
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
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[0], hbm_placement_scheme=hbm_placement_scheme[0])
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[1], hbm_placement_scheme=hbm_placement_scheme[1])
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False, is_hbm_interleaved=is_hbm_interleaved, hbm_split_scheme=hbm_split_scheme[2], hbm_placement_scheme=hbm_placement_scheme[2])
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator", shape=(coarsening_factor*coarsening_factor, tM//coarsening_factor, tN//coarsening_factor), dtype=ftype, storage=local_storage, transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(
        name="gemm_entry",
        ndrange={"i" : dace.subsets.Range([(0, M-1, tM*gM)]),
                 "j" : dace.subsets.Range([(0, N-1, tN*gN)])},
        schedule=device_schedule
    )
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

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(
        name="thread_group_mmad",
        ndrange={"gi" : dace.subsets.Range([(0, gM-1, 1)]),
                 "gj" : dace.subsets.Range([(0, gM-1, 1)])},
        schedule=thread_group_schedule
    )

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
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={"ci" : dace.subsets.Range([(0, tM-1, tM//coarsening_factor)]),
                 "cj" : dace.subsets.Range([(0, tN-1, tN//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
                elif name == "B":
                    access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                thread_group_map_entry.add_out_connector(f"OUT_{name}")
                thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry


    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", "0:K"])
        elif name == "B":
            access_str = ", ".join(["0:K", f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")


    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM//coarsening_factor, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN//coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                                    f"bK:bK+{tK//coarsening_factor}"])
        elif name == "B":
            access_str = ", ".join([f"bK:bK+{tK//coarsening_factor}", 
                                    f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule", inputs={"_in_local_a", "_in_local_b", "_in_accumulator"}, outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str, language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
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
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                            f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")
    
    DoubleBuffering.apply_to(sdfg, map_entry=block_tiled_map_entry, transient=local_access_nodes["local_A"])
    return sdfg


