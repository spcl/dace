
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Test for an issue where streams would be allocated globally (to a state) and locally.
    Expected behavior is to never allocate streams locally to a scope. """
import dace
import numpy as np

def test_stream_only_used_in_one_scope():
    sdfg = dace.SDFG("stream_only_in_one_scope")
    N = dace.symbol("N")
    sdfg.add_array("A", [N], dace.float32)
    sdfg.add_array("B", [N], dace.float32)
    sdfg.add_stream("accumulator_stream",
                    dace.float64,
                    transient=True,
                    storage=dace.dtypes.StorageType.FPGA_Local
                    )
    sdfg.add_array("A_device", [N],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [N],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)
    prep_state = sdfg.add_state()
    A_host = prep_state.add_read("A")
    A_device = prep_state.add_write("A_device")

    prep_state.add_memlet_path(A_host,
                          A_device,
                          memlet=dace.Memlet("A_device[0:N]"))
    state = sdfg.add_state()
    a = state.add_read("A_device")
    b = state.add_write("B_device")
    read_accumulator_stream = state.add_read("accumulator_stream")
    write_accumulator_stream = state.add_write("accumulator_stream")

    entry, exit = state.add_map(
        "map", {"n": "0:N"},
        schedule=dace.ScheduleType.FPGA_Device)

    acc_tasklet = state.add_tasklet(
        "accumulate", {"in_data", "acc"}, {"out_data", "acc_out"}, """\
out_data = in_data if (n == 0) else (in_data + acc)
acc_out=out_data""")


    state.add_memlet_path(
        a,
        entry,
        acc_tasklet,
        dst_conn="in_data",
        memlet=dace.Memlet("A_device[n]")
    )
    state.add_memlet_path(
        read_accumulator_stream,
        entry,
        acc_tasklet,
        dst_conn="acc",
        memlet=dace.Memlet("accumulator_stream[0]", dynamic=True)
    )

    state.add_memlet_path(
        acc_tasklet,
        exit,
        b,
        src_conn="out_data",
        memlet=dace.Memlet("B_device[n]")
    )

    state.add_memlet_path(
        acc_tasklet,
        exit,
        write_accumulator_stream,
        src_conn="acc_out",
        memlet=dace.Memlet("accumulator_stream[0]", dynamic=True)
    )

    post_state = sdfg.add_state()
    B_host = post_state.add_write("B")
    B_device = post_state.add_read("B_device")
    post_state.add_memlet_path(B_device,
                          B_host,
                          memlet=dace.Memlet("B[0:N]"))

    sdfg.add_edge(prep_state, state, dace.sdfg.InterstateEdge())
    sdfg.add_edge(state, post_state, dace.sdfg.InterstateEdge())

    N.set(5)

    A = np.ndarray([N.get()], dtype=dace.float32.type)
    B = np.ndarray([N.get()], dtype=dace.float32.type)
    A[:] = [dace.float32(0),dace.float32(1),dace.float32(2),dace.float32(3),dace.float32(4),]

    sdfg(A, B)

    print(A)
    print(B)

    #assert sdfg.generate_code()[2].clean_code.count('dace::FIFO<double, 1, 1> accumulator_stream("accumulator_stream");') == 1

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
    test_stream_only_used_in_one_scope()
    test_stream_allocation_scope()