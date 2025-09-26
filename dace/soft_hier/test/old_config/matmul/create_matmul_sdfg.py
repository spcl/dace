import dace
import typing
import numpy as np

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")


def _my_gen_matmul_sdfg(hardware_matmul_mnk: typing.Tuple, global_storage: dace.dtypes.StorageType,
                        local_storage: dace.dtypes.StorageType, device_schedule: dace.dtypes.ScheduleType,
                        thread_group_schedule: dace.dtypes.ScheduleType, thread_group_dims: typing.Tuple, input_float,
                        output_float, coarsening_factor, mmad_tasklet_str: str):
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
        arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False)
        arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(tM // coarsening_factor, tN // coarsening_factor),
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
                                                                           "gj": dace.subsets.Range([(0, gM - 1, 1)])
                                                                       },
                                                                       schedule=thread_group_schedule)

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
        schedule=dace.dtypes.ScheduleType.Sequential)

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
        schedule=dace.dtypes.ScheduleType.Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[0:{tM//coarsening_factor}, 0:{tN//coarsening_factor}]"))
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
    accumulator_an2 = state.add_access("accumulator")
    state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    #state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
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
    access_str = ", ".join([
        f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}",
        f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"
    ])
    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator"))
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    return sdfg


if __name__ == "__main__":
    sdfg = _my_gen_matmul_sdfg(
        hardware_matmul_mnk=(32, 32, 32),
        global_storage=dace.dtypes.StorageType.SoftHier_HBM,
        local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        device_schedule=dace.dtypes.ScheduleType.SoftHier_Device,
        thread_group_schedule=dace.dtypes.ScheduleType.SoftHier_Cluster,
        thread_group_dims=(4, 4),
        input_float=dace.float16,
        output_float=dace.float16,
        coarsening_factor=2,
        mmad_tasklet_str="flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_FP_16);")
    sdfg.save("my_gemm.sdfgz")

    sdfg(A=np.random.rand(128, 128).astype(np.float16),
         B=np.random.rand(128, 128).astype(np.float16),
         C=np.zeros((128, 128), dtype=np.float16),
         M=128,
         N=128,
         K=128)
