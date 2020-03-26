# Computes vector addition between three vectors:
#   - the first addiction is computed on rank sender
#   - the second in rank_receiver who send the result back to rank sender

import argparse
import dace
import numpy as np
from mpi4py import MPI

N = dace.symbol("N")

def make_receiver_sdfg():
    parent_sdfg = dace.SDFG("smi_sample0_receiver")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = parent_sdfg.add_state("copy_to_device")

    parent_sdfg.add_array("in_C", [N], dtype=dace.float32)
    in_host_C = copy_in_state.add_read("in_C")

    parent_sdfg.add_array(
        "in_device_C", [N],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    in_device_C = copy_in_state.add_write("in_device_C")

    copy_in_state.add_edge(in_host_C, None, in_device_C, None,
                           dace.memlet.Memlet.simple(in_device_C, "0:N"))


    ###########################################################################
    # FPGA:

    ###### AXPY 2 (B+C) part. Result is sent to rank 1 ######
    nested_axpy_2_sdfg = dace.SDFG('axpy_2_sdfg')
    nested_axpy_2_state = nested_axpy_2_sdfg.add_state("axpy2_state")

    nested_axpy_2_sdfg.add_array("mem_C", shape=[N], dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Global)
    in_read_C = nested_axpy_2_state.add_read("mem_C")
    nested_axpy_2_sdfg.add_stream("stream_in", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Remote)
    stream_read = nested_axpy_2_state.add_read("stream_in")
    nested_axpy_2_sdfg.add_stream('stream_out', dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Remote)
    stream_write = nested_axpy_2_state.add_write("stream_out")


    tasklet_wr, map_entry_wr, map_exit_wr = nested_axpy_2_state.add_mapped_tasklet(
        'write',  # name
        dict(i='0:N'),  # map range
        dict(inp_stream=dace.Memlet.simple(stream_read.data,'i'),  # input memlets
             inp_mem=dace.Memlet.simple(in_read_C.data, 'i')),
        '''                                                 # code
out = inp_stream + inp_mem
        ''',
        dict(out=dace.Memlet.simple(stream_write.data, 'i')),  # output memlets,
        schedule=dace.dtypes.ScheduleType.FPGA_Device
    )

    nested_axpy_2_state.add_edge(
        stream_read, None,
        map_entry_wr, None,
        memlet=dace.Memlet.simple(stream_read.data, '0:N'))
    nested_axpy_2_state.add_edge(
        in_read_C, None,
        map_entry_wr, None,
        memlet=dace.Memlet.simple(in_read_C.data, '0:N'))

    # Add output path (exit->dst)
    nested_axpy_2_state.add_edge(
        map_exit_wr, None,
        stream_write, None,
        memlet=dace.Memlet.simple(stream_write.data, '0:N'))

    nested_axpy_2_sdfg.fill_scope_connectors()
    nested_axpy_2_sdfg.validate()

    ##### make fpga state and nest SDFGs

    parent_nested_axpy = parent_sdfg.add_state("axpy")

    nested_axpy_2_node = parent_nested_axpy.add_nested_sdfg(nested_axpy_2_sdfg, parent_sdfg, {"stream_in", "mem_C"}, {"stream_out"})
    in_data_C = parent_nested_axpy.add_read("in_device_C")
    _, stream_node = parent_sdfg.add_stream("stream", dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Remote)
    #####################################################
    # set SMI properties
    stream_node.remote = True
    stream_node.location["snd_rank"] = "0"
    stream_node.location["port"] = "0"

    stream_rd = parent_nested_axpy.add_read("stream")


    _, stream_node_wr = parent_sdfg.add_stream("stream_wr", dtype=dace.float32, transient=True,
                                            storage=dace.dtypes.StorageType.FPGA_Remote)
    #####################################################
    # set SMI properties
    stream_node_wr.remote = True
    stream_node_wr.location["rcv_rank"] = "0"
    stream_node_wr.location["port"] = "1"

    stream_wr = parent_nested_axpy.add_write("stream_wr")



    parent_nested_axpy.add_memlet_path(in_data_C,
                                           nested_axpy_2_node,
                                           memlet=dace.Memlet.simple(in_data_C.data, '0:N'),
                                           dst_conn="mem_C")
    parent_nested_axpy.add_memlet_path(stream_rd,
                                           nested_axpy_2_node,
                                           memlet=dace.Memlet.simple(stream_rd.data, '0:N'),
                                           dst_conn="stream_in")
    parent_nested_axpy.add_memlet_path(nested_axpy_2_node,
                                 stream_wr,
                                 memlet=dace.Memlet.simple(stream_wr.data, '0:N'),
                                 src_conn="stream_out")

    parent_sdfg.add_edge(copy_in_state, parent_nested_axpy, dace.graph.edges.InterstateEdge())
    parent_sdfg.validate()

    return parent_sdfg

def make_sender_sdfg():
    parent_sdfg = dace.SDFG("smi_sample0_sender")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = parent_sdfg.add_state("copy_to_device")

    parent_sdfg.add_array("in_A", [N], dtype=dace.float32)
    in_host_A = copy_in_state.add_read("in_A")
    parent_sdfg.add_array("in_B", [N], dtype=dace.float32)
    in_host_B = copy_in_state.add_read("in_B")

    parent_sdfg.add_array(
        "in_device_A", [N],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    parent_sdfg.add_array(
        "in_device_B", [N],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    parent_sdfg.add_array(
        "in_device_C", [N],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    in_device_A = copy_in_state.add_write("in_device_A")
    in_device_B = copy_in_state.add_write("in_device_B")

    copy_in_state.add_edge(in_host_A, None, in_device_A, None,
                           dace.memlet.Memlet.simple(in_device_A, "0:N"))
    copy_in_state.add_edge(in_host_B, None, in_device_B, None,
                           dace.memlet.Memlet.simple(in_device_B, "0:N"))

    ###########################################################################
    # Copy data to Host

    copy_out_state = parent_sdfg.add_state("copy_to_host")

    parent_sdfg.add_array(
        "out_device", [N],
        dtype=dace.float32,
        transient=True,
        storage=dace.dtypes.StorageType.FPGA_Global)

    out_device = copy_out_state.add_read("out_device")

    parent_sdfg.add_array("out_data", [N], dtype=dace.float32)
    out_host = copy_out_state.add_write("out_data")

    copy_out_state.add_edge(out_device, None, out_host, None,
                            dace.memlet.Memlet.simple(out_host, "0:N"))

    ###########################################################################
    # FPGA: make fpga state, which will have two nested sdfg

    ##### AXPY 1 (A+B) part ######
    nested_axpy_1_sdfg = dace.SDFG('compute_axpy_1')
    nested_axpy_1_state = nested_axpy_1_sdfg.add_state("nested_axpy_1_state")

    nested_axpy_1_sdfg.add_array("mem_A", shape=[N], dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Global)
    in_read_A = nested_axpy_1_state.add_read("mem_A")
    nested_axpy_1_sdfg.add_array("mem_B", shape=[N], dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Global)
    in_read_B = nested_axpy_1_state.add_read("mem_B")
    nested_axpy_1_sdfg.add_stream('stream_out', dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Remote)
    stream_write = nested_axpy_1_state.add_write("stream_out")

    tasklet, map_entry, map_exit = nested_axpy_1_state.add_mapped_tasklet(
        'read',  # name
        dict(i='0:N'),  # map range
        dict(inp_A=dace.Memlet.simple(in_read_A.data, 'i'),  # input memlets
             inp_B=dace.Memlet.simple(in_read_B.data, 'i')),
        '''                                                 # code
out = inp_A + inp_B
        ''',
        dict(out=dace.Memlet.simple(stream_write.data, 'i')),  # output memlets,
        schedule=dace.dtypes.ScheduleType.FPGA_Device
    )

    # Add edges to map

    nested_axpy_1_state.add_edge(
        in_read_A, None,
        map_entry, None,
        memlet=dace.Memlet.simple(in_read_A.data, '0:N'))
    nested_axpy_1_state.add_edge(
        in_read_B, None,
        map_entry, None,
        memlet=dace.Memlet.simple(in_read_B.data, '0:N'))

    # Add output path (exit->dst)
    nested_axpy_1_state.add_edge(
        map_exit, None,
        stream_write, None,
        memlet=dace.Memlet.simple(stream_write.data, '0:N'))

    nested_axpy_1_sdfg.fill_scope_connectors()
    nested_axpy_1_sdfg.validate()

    ####### SAVE TO MEMORY ############
    #
    store_sdfg = dace.SDFG('store')
    store_state = store_sdfg.add_state("store_state")
    store_sdfg.add_stream("stream_in", dtype=dace.float32, storage=dace.dtypes.StorageType.FPGA_Remote)
    stream_read = store_state.add_read("stream_in")

    store_sdfg.add_array("mem", shape=[N], dtype=dace.float32,
                                 storage=dace.dtypes.StorageType.FPGA_Global)
    out_write = store_state.add_write("mem")

    store_state.add_edge(
        stream_read, None,
        out_write, None,
        memlet=dace.Memlet.simple(stream_read.data, '0:N'))

    store_sdfg.fill_scope_connectors()
    store_sdfg.validate()


    ##### make fpga state and nest SDFGs

    parent_nested_axpy = parent_sdfg.add_state("axpy_and_store")

    ### AXPY PART
    nested_axpy_1_node = parent_nested_axpy.add_nested_sdfg(nested_axpy_1_sdfg, parent_sdfg, {"mem_A", "mem_B"},
                                                            {"stream_out"})

    # parent_sdfg.add_array("in_device", shape=[N], dtype=dace.float32, transient=True, storage=dace.dtypes.StorageType.FPGA_Global)
    in_data_A = parent_nested_axpy.add_read("in_device_A")
    in_data_B = parent_nested_axpy.add_read("in_device_B")
    _, stream_node = parent_sdfg.add_stream("stream", dtype=dace.float32, transient=True,
                                            storage=dace.dtypes.StorageType.FPGA_Remote)
    #####################################################
    # set SMI properties
    stream_node.location["rcv_rank"] = "1"
    stream_node.location["port"] = "0"

    stream_wr = parent_nested_axpy.add_write("stream")
    parent_nested_axpy.add_memlet_path(in_data_A,
                                       nested_axpy_1_node,
                                       memlet=dace.Memlet.simple(in_data_A.data, '0:N'),
                                       dst_conn="mem_A")
    parent_nested_axpy.add_memlet_path(in_data_B,
                                       nested_axpy_1_node,
                                       memlet=dace.Memlet.simple(in_data_B.data, '0:N'),
                                       dst_conn="mem_B")
    parent_nested_axpy.add_memlet_path(nested_axpy_1_node,
                                       stream_wr,
                                       memlet=dace.Memlet.simple(stream_wr.data, '0:N'),
                                       src_conn="stream_out")


    #### STORE PART#############

    store_node = parent_nested_axpy.add_nested_sdfg(store_sdfg, parent_sdfg, {"stream_in"}, {"mem"})
    out_data = parent_nested_axpy.add_write("out_device")

    _, stream_node = parent_sdfg.add_stream("stream_rcv", dtype=dace.float32, transient=True,
                                            storage=dace.dtypes.StorageType.FPGA_Remote)
    #####################################################
    # set SMI properties
    stream_node.location["snd_rank"] = "1"
    stream_node.location["port"] = "1"

    stream_rd = parent_nested_axpy.add_read("stream_rcv")
    parent_nested_axpy.add_memlet_path(stream_rd,
                                       store_node,
                                       memlet=dace.Memlet.simple(stream_rd.data, '0:N'),
                                       dst_conn="stream_in")
    parent_nested_axpy.add_memlet_path(store_node,
                                       out_data,
                                       memlet=dace.Memlet.simple(out_data.data, '0:N'),
                                       src_conn="mem")

    parent_sdfg.add_edge(copy_in_state, parent_nested_axpy, dace.graph.edges.InterstateEdge())
    parent_sdfg.add_edge(parent_nested_axpy, copy_out_state, dace.graph.edges.InterstateEdge())

    parent_sdfg.validate()

    return parent_sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    # Do not show optimizer
    dace.config.Config.set("optimizer", "interface", value="")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)

    args = vars(parser.parse_args())

    N.set(args["N"])

    return_value = 0

    num_ranks = MPI.COMM_WORLD.Get_size()
    my_rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    if my_rank == 0:
        input_A = np.arange(N.get()).astype(dace.float32.type)
        input_B = np.arange(N.get()).astype(dace.float32.type)
        input_C = np.arange(N.get()).astype(dace.float32.type)

        out = np.random.uniform(-10, 0, N.get()).astype(dace.float32.type)

        sdfg = make_sender_sdfg()
        sdfg.specialize(dict(N=N))

        sdfg(in_A=input_A, in_B=input_B, out_data=out)

        diff = np.abs(out - (input_A + input_B + input_C))
        diff_total = np.sum(diff)



    else:

        # Initialize vector: X
        input_A = np.arange(N.get()).astype(dace.float32.type)
        input_B = np.arange(N.get()).astype(dace.float32.type)
        input_C =  np.arange(N.get()).astype(dace.float32.type)

        sdfg = make_receiver_sdfg()
        sdfg.specialize(dict(N=N))

        sdfg(in_C=input_C)

    MPI.COMM_WORLD.Barrier()

    print("==== Program end ====")

    if my_rank == 0:
        if diff_total >= 0.01:
            print("Verification failed!")
            return_value = 1
        else:
            print("Results verified successfully.")
    MPI.Finalize()

    exit(return_value)