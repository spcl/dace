
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Test for an issue where streams would be allocated globally (to a state) and locally.
    Expected behavior is to never allocate streams locally to a scope. """
import dace

def test_stream_allocation_scope():
    sdfg = dace.SDFG("stream_allocation")
    N = dace.symbol("N")
    P = dace.symbol("P")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_stream("AtoB",
                    dace.float64,
                    transient=True,
                    shape=(P, ),
                    storage=dace.dtypes.StorageType.FPGA_Local
                    )
    state = sdfg.add_state()
    a = state.add_read("A")
    b = state.add_write("B")
    read_AtoB = state.add_read("AtoB")
    write_AtoB = state.add_write("AtoB")

    read_entry, read_exit = state.add_map(
        "read_A", {"n": "0:N/P"},
        schedule=dace.ScheduleType.FPGA_Device)

    unroll_read_entry, unroll_read_exit = state.add_map(
        "unroll_read", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    write_entry, write_exit = state.add_map(
        "write_B", {"n": "0:N/P"},
        schedule=dace.ScheduleType.FPGA_Device)

    unroll_write_entry, unroll_write_exit = state.add_map(
        "unroll_write", {"p": "0:P"},
        schedule=dace.ScheduleType.FPGA_Device,
        unroll=True)

    copy_tasklet1 = state.add_tasklet(
        "copy1", {"in_data"}, {"out_data"}, """\
out_data = in_data""")

    state.add_memlet_path(unroll_read_entry, a, memlet=dace.memlet.Memlet())

    state.add_memlet_path(
        a,
        read_entry,
        copy_tasklet1,
        dst_conn="in_data",
        memlet=dace.Memlet("A[p+n]")
    )
    state.add_memlet_path(
        copy_tasklet1,
        read_exit,
        write_AtoB,
        src_conn="out_data",
        memlet=dace.Memlet("AtoB[p]")
    )

    state.add_memlet_path(write_AtoB, unroll_read_exit, memlet=dace.memlet.Memlet())

    copy_tasklet2 = state.add_tasklet(
        "copy2", {"in_data"}, {"out_data"}, """\
out_data = in_data""")

    state.add_memlet_path(unroll_write_entry, read_AtoB, memlet=dace.memlet.Memlet())

    state.add_memlet_path(
        read_AtoB,
        write_entry,
        copy_tasklet2,
        dst_conn="in_data",
        memlet=dace.Memlet("AtoB[p]")
    )

    state.add_memlet_path(
        copy_tasklet2,
        write_exit,
        b,
        src_conn="out_data",
        memlet=dace.Memlet("B[p+n]")
    )

    state.add_memlet_path(b, unroll_write_exit, memlet=dace.memlet.Memlet())

    sdfg.specialize(dict(P=2))
    assert sdfg.generate_code()[2].clean_code.count("dace::SetNames(AtoB") == 1
    # Expecting the stream AtoB to be only defined once - looking for this code:
    #    dace::FIFO<float, 1, 1> AtoB[P];
    #    dace::SetNames(AtoB, "AtoB", P);
    # by searching for the substring used in the assertion

if __name__ == '__main__':
    test_stream_allocation_scope()