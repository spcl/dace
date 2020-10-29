import dace
from dace.transformation.dataflow import PruneConnectors


def make_sdfg():
    """ Creates three SDFG nested within each other, where four input arrays and
        four output arrays are fed throughout the hierarchy. Two inputs and two
        output are not used for anything in the innermost SDFG, and can thus be
        removed in all nestings.
    """

    n = dace.symbol("N")

    sdfg_outer = dace.SDFG("prune_connectors_test")
    state_outer = sdfg_outer.add_state("state_outer")
    sdfg_outer.add_symbol("N", dace.int32)

    sdfg_middle = dace.SDFG("middle")
    sdfg_middle.add_symbol("N", dace.int32)
    nsdfg_middle = state_outer.add_nested_sdfg(
        sdfg_middle,
        sdfg_outer, {"read_used_middle", "read_unused_middle"},
        {"write_used_middle", "write_unused_middle"},
        name="middle")
    state_middle = sdfg_middle.add_state("middle")

    sdfg_inner = dace.SDFG("inner")
    sdfg_inner.add_symbol("N", dace.int32)
    nsdfg_inner = state_middle.add_nested_sdfg(
        sdfg_inner,
        sdfg_middle, {"read_used_inner", "read_unused_inner"},
        {"write_used_inner", "write_unused_inner"},
        name="inner")
    state_inner = sdfg_inner.add_state("inner")

    entry, exit = state_inner.add_map("map", {"i": "0:N"})
    tasklet = state_inner.add_tasklet("tasklet", {"read_tasklet"}, {"write_tasklet"},
                                      "write_tasklet = read_tasklet + 1")

    for s in ["unused", "used"]:

        # Read

        sdfg_outer.add_array(f"read_{s}", [n], dace.uint16)
        sdfg_middle.add_array(f"read_{s}_middle", [n], dace.uint16)
        sdfg_inner.add_array(f"read_{s}_inner", [n], dace.uint16)

        read_outer = state_outer.add_read(f"read_{s}")
        read_middle = state_middle.add_read(f"read_{s}_middle")
        read_inner = state_inner.add_read(f"read_{s}_inner")

        state_outer.add_memlet_path(
            read_outer,
            nsdfg_middle,
            dst_conn=f"read_{s}_middle",
            memlet=dace.Memlet(f"read_{s}[0:N]"))
        state_middle.add_memlet_path(
            read_middle,
            nsdfg_inner,
            dst_conn=f"read_{s}_inner",
            memlet=dace.Memlet(f"read_{s}_middle[0:N]"))

        # Write

        sdfg_outer.add_array(f"write_{s}", [n], dace.uint16)
        sdfg_middle.add_array(f"write_{s}_middle", [n], dace.uint16)
        sdfg_inner.add_array(f"write_{s}_inner", [n], dace.uint16)

        write_outer = state_outer.add_write(f"write_{s}")
        write_middle = state_middle.add_write(f"write_{s}_middle")
        write_inner = state_inner.add_write(f"write_{s}_inner")

        state_outer.add_memlet_path(
            nsdfg_middle,
            write_outer,
            src_conn=f"write_{s}_middle",
            memlet=dace.Memlet(f"write_{s}[0:N]"))
        state_middle.add_memlet_path(
            nsdfg_inner,
            write_middle,
            src_conn=f"write_{s}_inner",
            memlet=dace.Memlet(f"write_{s}_middle[0:N]"))

    state_inner.add_memlet_path(
        read_inner,
        entry,
        tasklet,
        dst_conn=f"read_tasklet",
        memlet=dace.Memlet(f"read_{s}_inner[i]"))

    state_inner.add_memlet_path(
        tasklet,
        exit,
        write_inner,
        src_conn=f"write_tasklet",
        memlet=dace.Memlet(f"write_{s}_inner[i]"))

    return sdfg_outer

if __name__ == "__main__":

    sdfg = make_sdfg()

    sdfg.apply_transformations_repeated(PruneConnectors)
