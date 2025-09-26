import dace


def vec_copy():
    sdfg = dace.SDFG("soft_hier_vec_copy")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (256 * 32, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    a_dev = sdfg.add_array("soft_hier_A", (256 * 32, ),
                           dace.float16,
                           dace.dtypes.StorageType.SoftHier_HBM,
                           transient=True)
    b_host = sdfg.add_array("B", (256 * 32, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    b_dev = sdfg.add_array("soft_hier_B", (256 * 32, ),
                           dace.float16,
                           dace.dtypes.StorageType.SoftHier_HBM,
                           transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("soft_hier_A")
    bhc = state.add_access("B")
    bdc = state.add_access("soft_hier_B")
    frag_a = sdfg.add_array("frag_A", (256, ), dace.float16, dace.dtypes.StorageType.SoftHier_TCDM, transient=True)
    frag_b = sdfg.add_array("frag_B", (256, ), dace.float16, dace.dtypes.StorageType.SoftHier_TCDM, transient=True)
    #af = state.add_access("frag_A")
    #bf = state.add_access("frag_B")

    dev_entry, dev_exit = state.add_map(name="copy_map_outer",
                                        ndrange={"i": dace.subsets.Range([(0, 256 * 32 - 1, 256 * 32)])},
                                        schedule=dace.dtypes.ScheduleType.SoftHier_Device)
    tblock_entry, tblock_exit = state.add_map(name="copy_map_inner",
                                              ndrange={"ii": dace.subsets.Range([(0, 256 * 32 - 1, 256)])},
                                              schedule=dace.dtypes.ScheduleType.SoftHier_Cluster)

    #glb_to_vecin = GlobalToVECIN(name="glb_to_vecin_a", input_names=["IN_A"], output_names=["OUT_frag_A"], queue_length=1, load_length=256)
    glb_to_vecin = state.add_access("frag_A")
    #vecout_to_glb = VECOUTToGlobal(name="vecout_to_glb_b", input_names=["IN_frag_B"], output_names=["OUT_B"], queue_length=1, load_length=256)
    libnode = state.add_access("frag_B")
    #libnode = VecUnit(name="vecout_to_glb_b", input_names=["IN_frag_A"], output_names=["OUT_frag_B"], queue_length=1, load_length=256)

    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet(f"A[0:{256*32}]"))
    state.add_edge(adc, None, dev_entry, "IN_A", dace.memlet.Memlet(f"soft_hier_A[0:{256*32}]"))
    state.add_edge(dev_entry, "OUT_A", tblock_entry, "IN_A", dace.memlet.Memlet(f"soft_hier_A[0:{256*32}]"))
    state.add_edge(tblock_entry, "OUT_A", glb_to_vecin, None, dace.memlet.Memlet(f"soft_hier_A[i + ii:i + ii + 256]"))
    state.add_edge(glb_to_vecin, None, libnode, None, dace.memlet.Memlet(f"frag_A[0:256]"))
    state.add_edge(libnode, None, tblock_exit, "IN_B", dace.memlet.Memlet(f"soft_hier_B[i + ii:i + ii + 256]"))
    state.add_edge(tblock_exit, "OUT_B", dev_exit, "IN_B", dace.memlet.Memlet(f"soft_hier_B[i + ii:i + ii + 256]"))
    state.add_edge(dev_exit, "OUT_B", bdc, None, dace.memlet.Memlet(f"soft_hier_B[0:{256*32}]"))
    state.add_edge(bdc, None, bhc, None, dace.memlet.Memlet(f"B[0:{256*32}]"))

    for n in [dev_entry, tblock_entry]:
        n.add_in_connector("IN_A")
        n.add_out_connector("OUT_A")

    for n in [dev_exit, tblock_exit]:
        n.add_in_connector("IN_B")
        n.add_out_connector("OUT_B")

    #libnode.add_in_connector("IN_A")
    #libnode.add_out_connector("OUT_B")

    #t = state.add_tasklet(name="assign", inputes={"_in"}, outputs={"_out"}, code="_out = _in")

    sdfg.save("soft_hier_2.sdfgz")
    return sdfg


if __name__ == "__main__":
    s = vec_copy()
    #   print(list(dace.ScheduleType))
    s.compile()
