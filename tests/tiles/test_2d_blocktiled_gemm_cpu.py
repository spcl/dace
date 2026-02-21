import dace
import typing
import os
import numpy as np
import ast
from dace.properties import CodeBlock


from dace.libraries.tiles.nodes.load import Load
from dace.libraries.tiles.nodes.store import Store
from dace.libraries.tiles.nodes.mma import MMA


M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")

def _my_gen_matmul_sdfg(hardware_matmul_mnk: typing.Tuple,
                        global_storage: dace.dtypes.StorageType,
                        local_storage: dace.dtypes.StorageType,
                        device_schedule: dace.dtypes.ScheduleType,
                        thread_group_schedule: dace.dtypes.ScheduleType,
                        thread_group_dims: typing.Tuple,
                        k_group_dims: typing.Tuple,
                        input_float,
                        output_float):
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
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
                                       transient=False)
            arrs[arrn] = arr
        if arr_name == "B":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False)
            arrs[arrn] = arr
        if arr_name == "C":
            arrn, arr = sdfg.add_array(name=arr_name,
                                       shape=shape,
                                       dtype=ftype,
                                       storage=global_storage,
                                       transient=False)
            arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator",
                               shape=(tM,
                                      tN),
                               dtype=ftype,
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
            "ci": dace.subsets.Range([(0, tM - 1, tM)]),
            "cj": dace.subsets.Range([(0, tN - 1, tN)])
        },
        schedule=dace.dtypes.ScheduleType.Sequential)

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
        ndrange={"bK": dace.subsets.Range([(bK_start, bK_end - 1, tK - 1)])},
        schedule=dace.dtypes.ScheduleType.Sequential)

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join(
        [f"0:{tM}", f"0:{tN}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry

    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([
                f"i + {kg_i} * {tM} + ci * {tM}:i + {kg_i} * {tM} + ci * {tM} + {tM}",
                "0:K"
            ])
            access_subsets = dace.subsets.Range([
                (f"i + {kg_i} * {tM} + ci * {tM}",
                 f"i + {kg_i} * {tM} + ci * {tM} + {tM} - 1", 1), (0, K - 1, 1)
            ])
        elif name == "B":
            access_str = ", ".join([
                "0:K",
                f"j + {kg_j} * {tN} + cj * {tN}:j + {kg_j} * {tN} + cj * {tN} + {tN}"
            ])
            access_subsets = dace.subsets.Range([
                (0, K - 1, 1),
                (f"j + {kg_j} * {tN} + cj * {tN}",
                 f"j + {kg_j} * {tN} + cj * {tN} + {tN} - 1", 1)
            ])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}",
                       dace.memlet.Memlet(data=f"{name}", subset=access_subsets))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")

    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM, tK)),
                        ("B", (tK, tN))]:
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
                f"i + {kg_i} * {tM} + ci * {tM}:i + {kg_i} * {tM} + ci * {tM} + {tM}",
                f"bK:bK+{tK}"
            ])
            access_subsets = dace.subsets.Range([
                (f"i + {kg_i} * {tM} + ci * {tM}",
                 f"i + {kg_i} * {tM} + ci * {tM} + {tM}-1", 1),
                (f"bK", f"bK+{tK}-1", 1)
            ])
        elif name == "B":
            access_str = ", ".join([
                f"bK:bK+{tK}",
                f"j + {kg_j} * {tN} + cj * {tN}:j + {kg_j} * {tN} + cj * {tN} + {tN}"
            ])
            access_subsets = dace.subsets.Range([
                (f"bK", f"bK+{tK}-1", 1),
                (f"j + {kg_j} * {tN} + cj * {tN}",
                 f"j + {kg_j} * {tN} + cj * {tN} + {tN}-1", 1)
            ])
        # state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None,
                       dace.memlet.Memlet(data=f"{name}", subset=access_subsets))

    # Connect local_A + local_B -> matmul -> accumulator
    assert input_float == output_float, "For simplicity, this example assumes input and output have the same dtype."
    matmul_node = MMA(
        name="mmad",
        dtype=input_float,
    )

    state.add_node(matmul_node)

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_node, "_in_" + name.lower().replace("local_", ""), dace.memlet.Memlet(name))

    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_node, "_in_acc",
                   dace.memlet.Memlet("accumulator"))

    access_str = ", ".join(
        [f"0:{tM}", f"0:{tN}"])
    state.add_edge(matmul_node, "_out_c", block_tiled_map_exit, "IN_accumulator",
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")

    accumulator_an3 = state.add_access("accumulator")

    access_str = ", ".join(
        [f"0:{tM}", f"0:{tN}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None,
                   dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([
        f"i + {kg_i} * {tM} + ci * {tM}:i + {kg_i} * {tM} + ci * {tM} + {tM}",
        f"j + {kg_j} * {tN} + cj * {tN}:j + {kg_j} * {tN} + cj * {tN} + {tN}"
    ])
    access_subsets = dace.subsets.Range([
        (f"i + {kg_i} * {tM} + ci * {tM}",
         f"i + {kg_i} * {tM} + ci * {tM} + {tM}-1", 1),
        (f"j + {kg_j} * {tN} + cj * {tN}",
         f"j + {kg_j} * {tN} + cj * {tN} + {tN}-1", 1)
    ])
    # state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C",
                   dace.memlet.Memlet(data="C", subset=access_subsets))
    thread_coarsened_map_exit.add_in_connector("IN_C")
    return sdfg


if __name__ == "__main__":

    dim_x = 1
    dim_y = 64
    cluster_dims = (dim_x, dim_y)
    k_dims = (1, 8)

    K = 64*320
    M = 64
    N = 64*20

    tK = 64
    tM = 64
    tN = 64

    A_host = np.ones((M, K), dtype=np.float16)
    B_host = np.ones((K, N), dtype=np.float16)
    C_host = np.zeros((M, N), dtype=np.float16)

    sdfg = _my_gen_matmul_sdfg(
        hardware_matmul_mnk=(tM, tN, tK),
        global_storage=dace.dtypes.StorageType.CPU_Heap,
        local_storage=dace.dtypes.StorageType.Register,
        device_schedule=dace.dtypes.ScheduleType.CPU_Multicore,
        thread_group_schedule=dace.dtypes.ScheduleType.Sequential,
        thread_group_dims=cluster_dims,
        k_group_dims=k_dims,
        input_float=dace.float64,
        output_float=dace.float64)
    sdfg.save("my_gemm.sdfgz")
    sdfg.validate()
    sdfg.compile()