import dace
import cupy as cp
import numpy as np


from dace.sdfg.container_group import ContainerGroupFlatteningMode
from dace.transformation.passes.struct_to_container_group import StructToContainerGroups


N = dace.symbol("N")
TSTEPS = dace.symbol("TSTEPS")

_N = 8192

A = np.random.rand(_N, _N).astype(np.float32)
B = np.random.rand(_N, _N).astype(np.float32)
A2 = cp.asarray(A, cp.float32)
B2 = cp.asarray(B, cp.float32)
C2 = cp.zeros((_N, _N), cp.float32)

def _get_jacobi_sdfg(use_container_array: bool):
    jacobi_sdfg = dace.SDFG("jacobi")

    initialize = jacobi_sdfg.add_state("initialize")
    for_guard = jacobi_sdfg.add_state("for_guard")
    kernel = jacobi_sdfg.add_state("kernel")
    finalize = jacobi_sdfg.add_state("finalize")

    step_init = dace.InterstateEdge(assignments={"step": "0"})
    step_check_true = dace.InterstateEdge(condition="not (step < NUM_STEPS)")
    step_check_false = dace.InterstateEdge(condition="(step < NUM_STEPS)")
    step_increment = dace.InterstateEdge(assignments={"step": "(step + 1)"})

    jacobi_sdfg.add_edge(initialize, for_guard, step_init)
    jacobi_sdfg.add_edge(for_guard, kernel, step_check_true)
    jacobi_sdfg.add_edge(for_guard, finalize, step_check_false)
    jacobi_sdfg.add_edge(kernel, for_guard, step_increment)

    N = dace.symbol("N")
    jacobi_sdfg.add_symbol(name="N", stype=np.int64)
    jacobi_sdfg.add_symbol(name="NUM_STEPS", stype=np.int64)

    if use_container_array:
        struct = dace.data.Structure(
            members={
                "As": dace.data.ContainerArray(
                    stype=dace.data.Array(
                        shape=[N, N],
                        storage=dace.dtypes.StorageType.CPU_Heap,
                        dtype=dace.typeclass(np.float32),
                    ), shape=(2,), transient=False
                ),
            },
            name="AB",
            storage=dace.dtypes.StorageType.CPU_Heap,
        )
    else:
        struct = dace.data.Structure(
            members={
                "A": dace.data.Array(
                    shape=[N, N],
                    storage=dace.dtypes.StorageType.CPU_Heap,
                    dtype=dace.typeclass(np.float32),
                ),
                "B": dace.data.Array(
                    shape=[N, N],
                    storage=dace.dtypes.StorageType.CPU_Heap,
                    dtype=dace.typeclass(np.float32),
                ),
            },
            name="AB",
            storage=dace.dtypes.StorageType.CPU_Heap,
        )

    jacobi_sdfg.add_datadesc(name="AB", datadesc=struct)

    v_A_name, v_A = jacobi_sdfg.add_view(
        name="v_A",
        shape=[N, N],
        storage=dace.dtypes.StorageType.CPU_Heap,
        dtype=dace.typeclass(np.float32),
    )

    v_B_name, v_B = jacobi_sdfg.add_view(
        name="v_B",
        shape=[N, N],
        storage=dace.dtypes.StorageType.CPU_Heap,
        dtype=dace.typeclass(np.float32),
    )

    ab_access = dace.nodes.AccessNode(data="AB")
    kernel.add_node(ab_access)
    ab2_access = dace.nodes.AccessNode(data="AB")
    kernel.add_node(ab2_access)
    ab3_access = dace.nodes.AccessNode(data="AB")
    kernel.add_node(ab3_access)

    if use_container_array:
        jacobi_sdfg.add_view(
            name="v_AB_As",
            shape=[N, N],
            storage=dace.dtypes.StorageType.CPU_Heap,
            dtype=dace.typeclass(np.float32),
        )
        ab4_access = dace.nodes.AccessNode(data="v_AB_As")
        kernel.add_node(ab4_access)
        ab5_access = dace.nodes.AccessNode(data="v_AB_As")
        kernel.add_node(ab5_access)


    a_access = dace.nodes.AccessNode(data="v_A")
    a_access.add_in_connector("views")
    b_dst_access = dace.nodes.AccessNode(data="v_B")
    b_dst_access.add_out_connector("views")
    kernel.add_node(a_access)
    b_access = dace.nodes.AccessNode(data="v_B")
    b_access.add_in_connector("views")
    a_dst_access = dace.nodes.AccessNode(data="v_A")
    a_dst_access.add_out_connector("views")
    kernel.add_node(b_access)

    if not use_container_array:
        kernel.add_edge(ab_access, None, a_access, "views",
                        dace.Memlet(data="AB.A",
                                    subset=dace.subsets.Range.from_string("0:N, 0:N")))
        kernel.add_edge(ab2_access, None, b_access, "views",
                        dace.Memlet(data="AB.B",
                                    subset=dace.subsets.Range.from_string("0:N, 0:N")))
    else:
        kernel.add_edge(ab_access, None, ab4_access, "views",
                        dace.Memlet(data="AB.As",
                                    subset=dace.subsets.Range.from_string("0:1")))
        kernel.add_edge(ab2_access, None, ab5_access, "views",
                        dace.Memlet(data="AB.As",
                                    subset=dace.subsets.Range.from_string("1:2")))
        kernel.add_edge(ab4_access, None, a_access, "views",
                        dace.Memlet(data="v_AB_As",
                                    subset=dace.subsets.Range.from_string("0:N, 0:N")))
        kernel.add_edge(ab5_access, None, b_access, "views",
                        dace.Memlet(data="v_AB_As",
                                    subset=dace.subsets.Range.from_string("0:N, 0:N")))

    for j, (src, dst, src_access, dst_access) in enumerate(
        [("A", "B", a_access, b_dst_access), ("B", "A", b_access, a_dst_access)]
    ):
        update_map_entry, update_map_exit = kernel.add_map(
            name=f"{dst}_update",
            ndrange={
                "i": dace.subsets.Range(ranges=[(0, N - 3, 1)]),
                "j": dace.subsets.Range(ranges=[(0, N - 3, 1)]),
            },
        )

        update_map_entry.add_in_connector(f"IN_v_{src}")
        update_map_entry.add_out_connector(f"OUT_v_{src}")
        update_map_exit.add_in_connector(f"IN_v_{dst}")
        update_map_exit.add_out_connector(f"OUT_v_{dst}")

        kernel.add_edge(
            src_access,
            None,
            update_map_entry,
            f"IN_v_{src}",
            dace.Memlet(expr=f"v_{src}[0:N,0:N]"),
        )

        jacobi_sdfg.add_scalar(
            name=f"acc{j}",
            dtype=dace.float32,
            transient=True,
            storage=dace.dtypes.StorageType.Register,
            lifetime=dace.dtypes.AllocationLifetime.Scope,
        )
        san =  kernel.add_access(f"acc{j}")

        kernel.add_edge(
            update_map_entry,
            None,
            san,
            None,
            dace.Memlet(None),
        )

        sub_domain_access = dace.nodes.AccessNode(data=f"v_{src}")
        sub_domain_access_2 = dace.nodes.AccessNode(data=f"v_{dst}")
        kernel.add_edge(
            update_map_entry,
            f"OUT_v_{src}",
            sub_domain_access,
            None,
            dace.Memlet(expr=f"v_{src}[i:i+2,j:j+2]"),
        )

        inner_map_entry, inner_map_exit = kernel.add_map(
            name=f"{dst}_inner_stencil",
            ndrange={
                "_i":  dace.subsets.Range(ranges=[(1, 3, 2)]),
                "_j":  dace.subsets.Range(ranges=[(1, 3, 2)]),
            },
            schedule=dace.dtypes.ScheduleType.Sequential
        )
        inner_map_entry.add_in_connector(f"IN_v_{src}")
        inner_map_entry.add_out_connector(f"OUT_v_{src}")
        inner_map_entry.add_in_connector(f"IN_acc")
        inner_map_entry.add_out_connector(f"OUT_acc")
        inner_map_exit.add_in_connector(f"IN_v_{dst}")
        inner_map_exit.add_out_connector(f"OUT_v_{dst}")

        kernel.add_edge(
            sub_domain_access,
            None,
            inner_map_entry,
            f"IN_v_{src}",
            dace.Memlet(expr=f"v_{src}[i:i+2,j:j+2]"),
        )
        kernel.add_edge(
            san,
            None,
            inner_map_entry,
            f"IN_acc",
            dace.Memlet(expr=f"acc{j}"),
        )
        kernel.add_edge(
            inner_map_exit,
            f"OUT_v_{dst}",
            sub_domain_access_2,
            None,
            dace.Memlet(expr=f"v_{dst}[i:i+2,j:j+2]"),
        )
        kernel.add_edge(
            sub_domain_access_2,
            None,
            update_map_exit,
            f"IN_v_{dst}",
            dace.Memlet(expr=f"v_{dst}[i:i+2,j:j+2]"),
        )

        access_str = f"v_{src}[i+_i,j+_j]"
        t1 = kernel.add_tasklet(
            name="Add", inputs={"_in"}, outputs={"_out"}, code=f"_out = 0.2 * _in "
        )
        t2 = kernel.add_tasklet(
            name="Acc", inputs={"_in", "_acc_in"}, outputs={"_out"}, code=f"_out = _acc_in + _in"
        )
        e1 = kernel.add_edge(
            inner_map_entry, f"OUT_v_{src}", t1, "_in", dace.Memlet(expr=access_str)
        )
        e2 = kernel.add_edge(t1, "_out", t2, "_in", dace.Memlet(expr=f"acc{j}"))
        e3 = kernel.add_edge(t2, "_out", inner_map_exit, f"IN_v_{dst}", dace.Memlet(expr=f"v_{dst}[i+_i,j+_j]"))
        e4 = kernel.add_edge(
            inner_map_entry, f"OUT_acc", t2, "_acc_in", dace.Memlet(expr=f"acc{j}")
        )

        update_map_exit.add_out_connector(f"OUT_v_{dst}")
        kernel.add_edge(update_map_exit, f"OUT_v_{dst}", dst_access, None,  dace.Memlet(expr=f"v_{dst}[0:N,0:N]"))

        if j == 0:
            if use_container_array:
                ab6_access = kernel.add_access("v_AB_As")
                kernel.add_edge(dst_access, f"views", ab6_access, None,  dace.Memlet(expr=f"v_AB_As[0:N, 0:N]"))
                kernel.add_edge(ab6_access, f"views", ab2_access, None,  dace.Memlet(expr=f"AB.As[{j}:{j}+1]"))
            else:
                kernel.add_edge(dst_access, f"views", ab2_access, None,  dace.Memlet(expr=f"AB.{dst}[0:N,0:N]"))
        if j == 1:
            if use_container_array:
                ab7_access = kernel.add_access("v_AB_As")
                kernel.add_edge(dst_access, f"views", ab7_access, None,  dace.Memlet(expr=f"v_AB_As[0:N, 0:N]"))
                kernel.add_edge(ab7_access, f"views", ab3_access, None,  dace.Memlet(expr=f"AB.As[{j}:{j}+1]"))
            else:
                kernel.add_edge(dst_access, f"views", ab3_access, None,  dace.Memlet(expr=f"AB.{dst}[0:N,0:N]"))

    jacobi_sdfg.save("jacobi_with_structs_sdfg.sdfg")
    jacobi_sdfg.validate()
    return jacobi_sdfg


def test_struct_to_container_group(use_container_array:bool):
    N = dace.symbol("N")
    TSTEPS = dace.symbol("TSTEPS")
    _N = 8192

    sdfg = _get_jacobi_sdfg(use_container_array)

    if use_container_array is False:
        sdfg.save("jacobi_with_container_array.sdfg")

    StructToContainerGroups().apply_pass(sdfg, {})

    sdfg.save("jacobi_with_data_groups.sdfg")

    sdfg.validate()

    #for arr in sdfg.arrays:
    #    assert isinstance(arr, dace.data.Array) or isinstance(arr, dace.data.Scalar)


if __name__ == "__main__":
    print("===========1")
    test_struct_to_container_group(False)
    print("===========2")
    test_struct_to_container_group(True)