import pytest
import dace
import numpy as np

from dace.transformation.passes.struct_to_container_group import ContainerGroupFlatteningMode
from dace.transformation.passes.struct_to_container_group import StructToContainerGroups


def _get_jacobi_sdfg(container_variant: str):
    jacobi_sdfg = dace.SDFG("jacobi_" + container_variant)

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

    if container_variant == "ContainerArray":
        struct = dace.data.Structure(
            members={
                "As": dace.data.ContainerArray(
                    stype=dace.data.Structure(
                        name="AsInternal",
                        members={
                            "AsArray": dace.data.Array(
                                shape=[N, N],
                                storage=dace.dtypes.StorageType.CPU_Heap,
                                dtype=dace.typeclass(np.float32),
                            ),
                        }
                    ),
                    shape=(2,),
                    transient=False,
                ),
            },
            name="AB",
            storage=dace.dtypes.StorageType.CPU_Heap,
        )
    elif container_variant == "Struct":
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
    else:
        assert container_variant == "Baseline"
        for n in ["v_A", "v_B"]:
            jacobi_sdfg.add_array(
                name=n,
                shape=[N, N],
                storage=dace.dtypes.StorageType.CPU_Heap,
                dtype=dace.typeclass(np.float32),
            )

    if container_variant != "Baseline":
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

    if container_variant == "ContainerArray":
        jacobi_sdfg.add_datadesc_view(
            name="v_AB_As",
            datadesc=struct.members["As"],
        )
        print(jacobi_sdfg.arrays)
        jacobi_sdfg.add_datadesc_view(
            name="v_AB_As_AsInternal",
            datadesc=jacobi_sdfg.arrays["v_AB_As"].stype,
        )
        jacobi_sdfg.add_datadesc_view(
            name="v_AB_As_AsInternal_AsArray",
            datadesc=jacobi_sdfg.arrays["v_AB_As"].stype.members["AsArray"],
        )
        ab4_access = dace.nodes.AccessNode(data="v_AB_As")
        kernel.add_node(ab4_access)
        ab5_access = dace.nodes.AccessNode(data="v_AB_As")
        kernel.add_node(ab5_access)
        ab4_is_access = dace.nodes.AccessNode(data="v_AB_As_AsInternal")
        kernel.add_node(ab4_is_access)
        ab5_is_access = dace.nodes.AccessNode(data="v_AB_As_AsInternal")
        kernel.add_node(ab5_is_access)
        ab4_is2_access = dace.nodes.AccessNode(data="v_AB_As_AsInternal_AsArray")
        kernel.add_node(ab4_is2_access)
        ab5_is2_access = dace.nodes.AccessNode(data="v_AB_As_AsInternal_AsArray")
        kernel.add_node(ab5_is2_access)


    if container_variant != "Baseline":
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

    if container_variant == "Baseline":
        a_access = dace.nodes.AccessNode(data="v_A")
        kernel.add_node(a_access)
        a_dst_access = dace.nodes.AccessNode(data="v_A")
        b_dst_access = dace.nodes.AccessNode(data="v_B")

    if container_variant == "Struct":
        kernel.add_edge(
            ab_access,
            None,
            a_access,
            "views",
            dace.Memlet(data="AB.A", subset=dace.subsets.Range.from_string("0:N, 0:N")),
        )
        kernel.add_edge(
            ab2_access,
            None,
            b_access,
            "views",
            dace.Memlet(data="AB.B", subset=dace.subsets.Range.from_string("0:N, 0:N")),
        )
    elif container_variant == "ContainerArray":
        kernel.add_edge(
            ab_access,
            None,
            ab4_access,
            "views",
            dace.Memlet(data="AB.As"),
        )
        kernel.add_edge(
            ab2_access,
            None,
            ab5_access,
            "views",
            dace.Memlet(data="AB.As"),
        )
        kernel.add_edge(
            ab4_access,
            None,
            ab4_is_access,
            "views",
            dace.Memlet(expr="v_AB_As[0]"),
        )
        kernel.add_edge(
            ab5_access,
            None,
            ab5_is_access,
            "views",
            dace.Memlet(expr="v_AB_As[1]"),
        )
        kernel.add_edge(
            ab4_is_access,
            None,
            ab4_is2_access,
            "views",
            dace.Memlet(expr="v_AB_As_AsInternal.AsArray"),
        )
        kernel.add_edge(
            ab5_is_access,
            None,
            ab5_is2_access,
            "views",
            dace.Memlet(expr="v_AB_As_AsInternal.AsArray"),
        )
        kernel.add_edge(
            ab4_is2_access,
            None,
            a_access,
            "views",
            dace.Memlet(
                data="v_AB_As_AsInternal_AsArray", subset=dace.subsets.Range.from_string("0:N, 0:N")
            ),
        )
        kernel.add_edge(
            ab5_is2_access,
            None,
            b_access,
            "views",
            dace.Memlet(
                data="v_AB_As_AsInternal_AsArray", subset=dace.subsets.Range.from_string("0:N, 0:N")
            ),
        )
    else:
        assert container_variant == "Baseline"

    if container_variant != "Baseline":
        vars = [("A", "B", a_access, b_dst_access), ("B", "A", b_access, a_dst_access)]
    else:
        vars = [
            ("A", "B", a_access, b_dst_access),
            ("B", "A", b_dst_access, a_dst_access),
        ]
    for j, (src, dst, src_access, dst_access) in enumerate(vars):
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
        jacobi_sdfg.add_scalar(
            name=f"acc_2_{j}",
            dtype=dace.float32,
            transient=True,
            storage=dace.dtypes.StorageType.Register,
            lifetime=dace.dtypes.AllocationLifetime.Scope,
        )
        san = kernel.add_access(f"acc{j}")

        kernel.add_edge(
            update_map_entry,
            None,
            san,
            None,
            dace.Memlet(None),
        )

        inner_map_entry, inner_map_exit = kernel.add_map(
            name=f"{dst}_inner_stencil",
            ndrange={
                "_i": dace.subsets.Range(ranges=[(1, 3, 2)]),
                "_j": dace.subsets.Range(ranges=[(1, 3, 2)]),
            },
            schedule=dace.dtypes.ScheduleType.Sequential,
        )
        inner_map_entry.add_in_connector(f"IN_v_{src}")
        inner_map_entry.add_out_connector(f"OUT_v_{src}")
        inner_map_entry.add_in_connector(f"IN_acc")
        inner_map_entry.add_out_connector(f"OUT_acc")
        inner_map_exit.add_in_connector(f"IN_v_{dst}")
        inner_map_exit.add_out_connector(f"OUT_v_{dst}")

        kernel.add_edge(
            update_map_entry,
            f"OUT_v_{src}",
            inner_map_entry,
            f"IN_v_{src}",
            dace.Memlet(expr=f"v_{src}[i:i+2,j:j+2]"),
        )

        kernel.add_edge(
            san,
            None,
            inner_map_entry,
            f"IN_acc",
            dace.Memlet(expr=f"acc{j}[0]"),
        )


        kernel.add_edge(
            inner_map_exit,
            f"OUT_v_{dst}",
            update_map_exit,
            f"IN_v_{dst}",
            dace.Memlet(expr=f"v_{dst}[i:i+2,j:j+2]"),
        )

        access_str = f"v_{src}[i+_i,j+_j]"
        t1 = kernel.add_tasklet(
            name="Add", inputs={"_in"}, outputs={"_out"}, code=f"_out = 0.2 * _in "
        )
        t2 = kernel.add_tasklet(
            name="Acc",
            inputs={"_in", "_acc_in"},
            outputs={"_out"},
            code=f"_out = _acc_in + _in",
        )
        e1 = kernel.add_edge(
            inner_map_entry, f"OUT_v_{src}", t1, "_in", dace.Memlet(expr=access_str)
        )
        e2 = kernel.add_edge(t1, "_out", t2, "_in", dace.Memlet(expr=f"acc_2_{j}"))
        e3 = kernel.add_edge(
            t2,
            "_out",
            inner_map_exit,
            f"IN_v_{dst}",
            dace.Memlet(expr=f"v_{dst}[i+_i,j+_j]"),
        )
        e4 = kernel.add_edge(
            inner_map_entry, f"OUT_acc", t2, "_acc_in", dace.Memlet(expr=f"acc{j}")
        )

        update_map_exit.add_out_connector(f"OUT_v_{dst}")
        kernel.add_edge(
            update_map_exit,
            f"OUT_v_{dst}",
            dst_access,
            None,
            dace.Memlet(expr=f"v_{dst}[0:N,0:N]"),
        )

        if j == 0:
            if container_variant == "ContainerArray":
                ab6_access = kernel.add_access("v_AB_As")
                _ab6_access = kernel.add_access("v_AB_As_AsInternal")
                __ab6_access = kernel.add_access("v_AB_As_AsInternal_AsArray")
                kernel.add_edge(
                    dst_access,
                    f"views",
                    __ab6_access,
                    None,
                    dace.Memlet(expr=f"v_AB_As_AsInternal_AsArray[0:N, 0:N]"),
                )
                kernel.add_edge(
                    __ab6_access,
                    f"views",
                    _ab6_access,
                    None,
                    dace.Memlet(expr=f"v_AB_As_AsInternal.AsArray"),
                )
                kernel.add_edge(
                    _ab6_access,
                    f"views",
                    ab6_access,
                    None,
                    dace.Memlet(expr=f"v_AB_As[{j}:{j}+1]"),
                )
                kernel.add_edge(
                    ab6_access,
                    f"views",
                    ab2_access,
                    None,
                    dace.Memlet(expr=f"AB.As[{j}:{j}+1]"),
                )
            elif container_variant == "Struct":
                kernel.add_edge(
                    dst_access,
                    f"views",
                    ab2_access,
                    None,
                    dace.Memlet(expr=f"AB.{dst}[0:N,0:N]"),
                )
            else:
                # kernel.add_edge(dst_access, f"views", bb_access, None,  dace.Memlet(expr=f"v_B[0:N,0:N]"))
                pass
        if j == 1:
            if container_variant == "ContainerArray":
                ab7_access = kernel.add_access("v_AB_As")
                _ab7_access = kernel.add_access("v_AB_As_AsInternal")
                __ab7_access = kernel.add_access("v_AB_As_AsInternal_AsArray")
                kernel.add_edge(
                    dst_access,
                    f"views",
                    __ab7_access,
                    None,
                    dace.Memlet(expr=f"v_AB_As_AsInternal_AsArray[0:N, 0:N]"),
                )
                kernel.add_edge(
                    __ab7_access,
                    f"views",
                    _ab7_access,
                    None,
                    dace.Memlet(expr=f"v_AB_As_AsInternal.AsArray"),
                )
                kernel.add_edge(
                    _ab7_access,
                    f"views",
                    ab7_access,
                    None,
                    dace.Memlet(expr=f"v_AB_As[{j}:{j}+1]"),
                )
                kernel.add_edge(
                    ab7_access,
                    f"views",
                    ab3_access,
                    None,
                    dace.Memlet(expr=f"AB.As[{j}:{j}+1]"),
                )
            elif container_variant == "Struct":
                kernel.add_edge(
                    dst_access,
                    f"views",
                    ab3_access,
                    None,
                    dace.Memlet(expr=f"AB.{dst}[0:N,0:N]"),
                )
            else:
                # kernel.add_edge(src_access, None, b_dst_access, None,  dace.Memlet(expr=f"v_A[0:N,0:N]"))
                pass

    jacobi_sdfg.validate()
    return jacobi_sdfg


@pytest.fixture(params=["ContainerArray", "Struct"])
def container_variant(request):
    return request.param


def test_struct_to_container_group(container_variant: str):
    baseline_sdfg = _get_jacobi_sdfg("Baseline")
    baseline_sdfg.simplify(validate_all=True)
    _N = 256
    _NS = 512
    np.random.seed(42)
    A_ref = np.random.rand(_N, _N).astype(np.float32)
    B_ref = np.random.rand(_N, _N).astype(np.float32)
    baseline_sdfg(v_A=A_ref, v_B=B_ref, N=_N, NUM_STEPS=_NS)

    sdfg = _get_jacobi_sdfg(container_variant)
    use_container_array = container_variant == "ContainerArray"
    sdfg.simplify(validate_all=True)

    StructToContainerGroups(
        flattening_mode=ContainerGroupFlatteningMode.StructOfArrays,
        simplify=True,
        validate=True,
        validate_all=True,
        save_steps=True,
        interface_with_struct_copy=True,
        verbose=True,
    ).apply_pass(sdfg, {})

    #for arr in sdfg.arrays.values():
    #    assert isinstance(arr, (dace.data.Array, dace.data.Scalar))

    np.random.seed(42)
    if use_container_array is True:
        AB = np.random.rand(2, _N, _N).astype(np.float32)
        sdfg.compile()
        #sdfg(__CG_AB__CA_As__CG_AsInternal__m_AsArray=AB, NUM_STEPS=_NS, N=_N)

        A_view = AB[0, :, :]
        B_view = AB[1, :, :]
        assert np.allclose(A_ref, A_view)
        assert np.allclose(B_ref, B_view)
    else:
        A = np.random.rand(_N, _N).astype(np.float32)
        B = np.random.rand(_N, _N).astype(np.float32)
        sdfg.compile()
        #sdfg(__CG_AB__m_A=A, __CG_AB__m_B=B, NUM_STEPS=_NS, N=_N)
        assert np.allclose(A_ref, A)
        assert np.allclose(B_ref, B)


if __name__ == "__main__":
    test_struct_to_container_group("Struct")

    test_struct_to_container_group("ContainerArray")
