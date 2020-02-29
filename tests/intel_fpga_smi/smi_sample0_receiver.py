# Computes vector addition between three vectors:
#   - the first addiction is computed on rank sender
#   - the second in rank_receiver who send the result back to rank sender

import argparse
import dace
import numpy as np
N = dace.symbol("N")

def make_sdfg():
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


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)

    args = vars(parser.parse_args())

    N.set(args["N"])

    print('Data copy: ' + str(N.get()))

    # Initialize vector: X
    input_A = np.arange(N.get()).astype(dace.float32.type)
    input_B = np.arange(N.get()).astype(dace.float32.type)
    input_C =  np.arange(N.get()).astype(dace.float32.type)

    sdfg = make_sdfg()
    sdfg.specialize(dict(N=N, smi_rank=1, smi_num_ranks=2))

    sdfg(in_C=input_C)


    print("==== Program end ====")

    exit(0)
