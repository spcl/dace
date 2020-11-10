# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various helper functions and classes for streaming BLAS operators on the FPGA

- FPGA state setup with copy to device and back to host
- streaming classes to stream data from/to memory arrays from/to FIFO queues
    using different access patterns
"""

from dace import InterstateEdge, vector
from dace.memlet import Memlet
from dace import dtypes
from dace import config

from dace.libraries.blas.utility import memory_operations as mem_ops

# ---------- ---------- ---------- ----------
# UTILITY
# ---------- ---------- ---------- ----------


def fpga_setup_states(sdfg, compute_state):
    """
    Add states for copying data from and to the FPGA
    """

    pre_state = sdfg.add_state('copy_to_FPGA')
    post_state = sdfg.add_state('copy_to_CPU')

    sdfg.add_edge(pre_state, compute_state, InterstateEdge())
    sdfg.add_edge(compute_state, post_state, InterstateEdge())

    return (pre_state, post_state)


def fpga_setup_connect_streamers(sdfg,
                                 state,
                                 lib_nodeIn,
                                 input_streams,
                                 lib_node_in_cons,
                                 lib_nodeOut,
                                 output_streams,
                                 lib_node_out_cons,
                                 input_memory_banks=None,
                                 output_memory_banks=None):
    """
    Add states for copying data from and to the FPGA
    and connect the given streaming classes with their
    access pattern and mem. location to the BLAS node
    """

    pre_state, post_state = fpga_setup_states(sdfg, state)

    for i, stream, libCon in zip(range(len(lib_node_in_cons)), input_streams,
                                 lib_node_in_cons):

        stream.copy_to_fpga(sdfg,
                            pre_state,
                            bank=(None if input_memory_banks is None else
                                  input_memory_banks[i]))

        stream.connect_to_lib(sdfg, state, lib_nodeIn, libCon)

    for i, stream, libCon in zip(range(len(lib_node_out_cons)), output_streams,
                                 lib_node_out_cons):

        stream.copy_to_cpu(sdfg,
                           post_state,
                           bank=(None if output_memory_banks is None else
                                 output_memory_banks[i]))

        stream.connect_to_lib(sdfg, state, lib_nodeOut, libCon)

    return pre_state, post_state


def fpga_setup_ConnectStreamersMultiNode(sdfg,
                                         state,
                                         lib_nodeIns,
                                         input_streams,
                                         lib_node_in_cons,
                                         lib_nodeOuts,
                                         output_streams,
                                         lib_node_out_cons,
                                         input_memory_banks=None,
                                         output_memory_banks=None):
    """
    Add states for copying data from and to the FPGA
    and connect the given streaming classes with their
    access pattern and mem. location to multiple diff. BLAS nodes
    """

    pre_state, post_state = fpga_setup_states(sdfg, state)

    for i, stream, libCon, lib_nodeIn in zip(range(len(lib_node_in_cons)),
                                             input_streams, lib_node_in_cons,
                                             lib_nodeIns):

        stream.copy_to_fpga(sdfg,
                            pre_state,
                            bank=(None if input_memory_banks is None else
                                  input_memory_banks[i]))

        stream.connect_to_lib(sdfg, state, lib_nodeIn, libCon)

    for i, stream, libCon, lib_nodeOut in zip(range(len(lib_node_out_cons)),
                                              output_streams, lib_node_out_cons,
                                              lib_nodeOuts):

        stream.copy_to_cpu(sdfg,
                           post_state,
                           bank=(None if output_memory_banks is None else
                                 output_memory_banks[i]))

        stream.connect_to_lib(sdfg, state, lib_nodeOut, libCon)

    return pre_state, post_state


# ---------- ---------- ---------- ----------
# READERS
# ---------- ---------- ---------- ----------
class StreamReadVector():
    """Configures a data streaming context for a DataNode. It is configured with a
    host memory data container and handles copying of data to the device and streams
    the data in a contiguous fashion with veclen sized chunks to the connected operator
    on the device.
    """
    def __init__(self,
                 source,
                 mem_size,
                 dtype,
                 buffer_size=None,
                 veclen=1,
                 unroll=False,
                 unroll_width=1,
                 repeat=1):

        if not (unroll == False and unroll_width == 1):
            raise NotImplementedError(
                "Unrolling on StreamReadVector not supported at the time")

        self.source = source
        self.mem_size = mem_size
        self.dtype = dtype

        self.buffer_size = buffer_size or config.Config.get(
            "library", "blas", "fpga", "default_stream_depth")
        self.veclen = veclen
        self.unroll = unroll
        self.unroll_width = unroll_width
        self.repeat = repeat

        self.fpga_data = None
        self.fpga_dataName = None
        self.fpga_stream = None

    def __eq__(self, other):

        if (self.source == other.source and self.mem_size == other.mem_size
                and self.dtype == other.dtype and self.veclen == other.veclen
                and self.repeat == other.repeat):

            return True
        else:
            return False

    def copy_to_fpga(self, sdfg, pre_state, bank=None):

        fpga_inputs, fpgaIn_names = mem_ops.fpga_copy_cpu_to_global(
            sdfg,
            pre_state, [self.source], [self.mem_size], [self.dtype],
            bank=bank,
            veclen=self.veclen)

        self.fpga_data = fpga_inputs[0]
        self.fpga_dataName = fpgaIn_names[0]

    def connect_to_lib(self,
                       sdfg,
                       state,
                       lib_node,
                       lib_connector,
                       access=False):

        vec_type = vector(self.dtype, self.veclen)

        in_mem, in_name = self.stream(state,
                                      self.fpga_data.data,
                                      self.mem_size,
                                      destName=lib_connector,
                                      access=access)

        stream_inp = state.add_stream(in_name,
                                      vec_type,
                                      buffer_size=self.buffer_size,
                                      transient=True,
                                      storage=dtypes.StorageType.FPGA_Local)
        self.fpga_stream = stream_inp

        state.add_memlet_path(stream_inp,
                              lib_node,
                              dst_conn=lib_connector,
                              memlet=Memlet.simple(stream_inp,
                                                   "0",
                                                   num_accesses=self.mem_size))

    def get_copy_size(self):

        return self.mem_size

    def stream(self, state, src, mem_size, destName='', access=False):

        dest = src + "_"
        if destName != '':
            dest += destName + "_"
        dest += "rS"

        data_in = None
        if access:
            data_in = self.fpga_data
        else:
            data_in = state.add_read(src)

        vec_type = vector(self.dtype, self.veclen)

        data_out = state.add_stream(dest,
                                    vec_type,
                                    buffer_size=self.buffer_size,
                                    transient=True,
                                    storage=dtypes.StorageType.FPGA_Local)

        repeat_map_entry = None
        repeat_map_exit = None

        if self.repeat != 1:
            repeat_map_entry, repeat_map_exit = state.add_map(
                'streamRepeat_{}_map'.format(dest),
                dict(r='0:{}'.format(self.repeat)),
                schedule=dtypes.ScheduleType.FPGA_Device)

        read_map_entry, read_map_exit = state.add_map(
            'streamRead_{}_map'.format(dest),
            dict(i='0:{0}/{1}'.format(mem_size, self.veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=self.unroll)

        read_tasklet = state.add_tasklet('sR_{}'.format(dest), ['inCon'],
                                         ['outCon'], 'outCon = inCon')

        if self.repeat != 1:

            state.add_memlet_path(data_in,
                                  repeat_map_entry,
                                  read_map_entry,
                                  read_tasklet,
                                  dst_conn='inCon',
                                  memlet=Memlet.simple(data_in.data, 'i'))

            state.add_memlet_path(read_tasklet,
                                  read_map_exit,
                                  repeat_map_exit,
                                  data_out,
                                  src_conn='outCon',
                                  memlet=Memlet.simple(data_out.data, '0'))

        else:

            state.add_memlet_path(data_in,
                                  read_map_entry,
                                  read_tasklet,
                                  dst_conn='inCon',
                                  memlet=Memlet.simple(data_in.data, 'i'))

            state.add_memlet_path(read_tasklet,
                                  read_map_exit,
                                  data_out,
                                  src_conn='outCon',
                                  memlet=Memlet.simple(data_out.data, '0'))

        return data_out, dest


# ---------- ---------- ---------- ----------
# WRITERS
# ---------- ---------- ---------- ----------


class StreamWriteVector():
    """Configures a data streaming context for a DataNode. It is configured with a
    host memory data container and handles copying of data from the device to this host container
    and streams the data in a contiguous fashion with veclen sized chunks from the connected
    operator to the on-device memory and handles copying back to the host.
    """
    def __init__(
        self,
        destination,
        mem_size,
        dtype,
        buffer_size=None,
        veclen=1,
        unroll=False,
        unroll_width=1,
    ):

        if not (unroll == False and unroll_width == 1):
            raise NotImplementedError(
                "Unrolling on StreamWriteVector not supported at the time")

        self.destination = destination
        self.mem_size = mem_size
        self.dtype = dtype

        self.buffer_size = buffer_size or config.Config.get(
            "library", "blas", "fpga", "default_stream_depth")
        self.veclen = veclen
        self.unroll = unroll
        self.unroll_width = unroll_width

        self.fpga_data = None
        self.fpga_dataName = None
        self.fpga_stream = None

    def copy_to_cpu(self, sdfg, post_state, bank=None):

        fpga_outputs, fpgaOut_names = mem_ops.fpga_copy_global_to_cpu(
            sdfg,
            post_state, [self.destination], [self.mem_size], [self.dtype],
            bank=bank,
            veclen=self.veclen)

        self.fpga_data = fpga_outputs[0]
        self.fpga_dataName = fpgaOut_names[0]

    def connect_to_lib(self,
                       sdfg,
                       state,
                       lib_node,
                       lib_connector,
                       access=False):

        vec_type = vector(self.dtype, self.veclen)

        out_mem, out_name = self.stream(sdfg,
                                        state,
                                        self.fpga_data.data,
                                        self.mem_size,
                                        self.dtype,
                                        src_name=lib_connector,
                                        access=access)

        stream_out = state.add_stream(out_name,
                                      vec_type,
                                      buffer_size=self.buffer_size,
                                      transient=True,
                                      storage=dtypes.StorageType.FPGA_Local)
        self.fpga_stream = stream_out

        state.add_memlet_path(lib_node,
                              stream_out,
                              src_conn=lib_connector,
                              memlet=Memlet.simple(stream_out,
                                                   "0",
                                                   num_accesses=-1))

    def get_copy_size(self):

        return self.mem_size

    def stream(self,
               sdfg,
               state,
               dest,
               mem_size,
               dtype,
               src_name='',
               access=False):

        src = dest + "_"
        if src_name != '':
            src += src_name + "_"
        src += "wS"

        vec_type = vector(self.dtype, self.veclen)

        data_in = state.add_stream(src,
                                   vec_type,
                                   buffer_size=self.buffer_size,
                                   transient=True,
                                   storage=dtypes.StorageType.FPGA_Local)

        data_out = None
        if access:
            data_out = self.fpga_data
        else:
            data_out = state.add_write(dest)

        write_map_entry, write_map_exit = state.add_map(
            'streamWrite_{}_map'.format(src),
            dict(i='0:{0}/{1}'.format(self.mem_size, self.veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=self.unroll)

        write_tasklet = state.add_tasklet('sW_{}'.format(dest), ['inCon'],
                                          ['outCon'], 'outCon = inCon')

        state.add_memlet_path(data_in,
                              write_map_entry,
                              write_tasklet,
                              dst_conn='inCon',
                              memlet=Memlet.simple(data_in.data, '0'))

        state.add_memlet_path(write_tasklet,
                              write_map_exit,
                              data_out,
                              src_conn='outCon',
                              memlet=Memlet.simple(data_out.data, 'i'))

        return data_in, src
