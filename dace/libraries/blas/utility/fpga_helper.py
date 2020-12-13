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
                                 output_memory_banks=None,
                                 entry_nodes=None,
                                 exit_nodes=None):
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

        stream.connect_to_lib(sdfg,
                              state,
                              lib_nodeIn,
                              libCon,
                              entry_nodes=entry_nodes)

    for i, stream, libCon in zip(range(len(lib_node_out_cons)), output_streams,
                                 lib_node_out_cons):

        stream.copy_to_cpu(sdfg,
                           post_state,
                           bank=(None if output_memory_banks is None else
                                 output_memory_banks[i]))

        stream.connect_to_lib(sdfg,
                              state,
                              lib_nodeOut,
                              libCon,
                              exit_nodes=exit_nodes)

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
                       access=False,
                       entry_nodes=None,
                       exit_nodes=None):

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
                              *(entry_nodes if entry_nodes else []),
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


class StreamReadMatrixFull():
    def __init__(
        self,
        source,
        rows,
        columns,
        rowTile,
        colTile,
        dtype,
        bufferSize=32,
        veclen=1,
        blockByRow=True,
        tileByRow=True,
        repeat=1,
        rowRepeat=1,
        increasedRowRepeat=False,
        rowPyramid=False,
        reverse=False,  # applies to increasedRowRepeat and rowPyramid
    ):

        self.source = source
        self.rows = rows
        self.columns = columns
        self.rowTile = rowTile
        self.colTile = colTile
        self.dtype = dtype

        self.bufferSize = bufferSize
        self.veclen = veclen
        self.blockByRow = blockByRow
        self.tileByRow = tileByRow
        self.repeat = repeat
        self.rowRepeat = rowRepeat
        self.increasedRowRepeat = increasedRowRepeat
        self.rowPyramid = rowPyramid  # only for blocks by Row
        self.reverse = reverse

        self.fpga_data = None
        self.fpga_dataName = None
        self.fpga_stream = None

    def __eq__(self, other):

        if (self.source == other.source and self.dtype == other.dtype
                and self.veclen == other.veclen and self.rows == other.rows
                and self.columns == other.columns
                and self.rowTile == other.rowTile
                and self.colTile == other.colTile
                and self.blockByRow == other.blockByRow
                and self.tileByRow == other.tileByRow
                and self.repeat == other.repeat
                and self.rowRepeat == other.rowRepeat
                and self.increasedRowRepeat == other.increasedRowRepeat
                and self.rowPyramid == other.rowPyramid
                and self.reverse == other.reverse):

            return True
        else:
            return False

    def copy_to_fpga(self, sdfg, preState, bank=None):

        fpga_inputs, fpgaIn_names = mem_ops.fpga_copy_cpu_to_global(
            sdfg,
            preState, [self.source], [self.rows * self.columns], [self.dtype],
            bank=bank)

        self.fpga_data = fpga_inputs[0]
        self.fpga_dataName = fpgaIn_names[0]

    def connect_to_lib(self,
                       sdfg,
                       state,
                       libNode,
                       libConnector,
                       access=False,
                       entry_nodes=None,
                       exit_nodes=None):

        in_mem, in_name = self.stream(state,
                                      self.fpga_data.data,
                                      self.rows * self.columns,
                                      destName=libConnector,
                                      access=access)

        vec_type = vector(self.dtype, self.veclen)
        stream_inp = state.add_stream(in_name,
                                      vec_type,
                                      buffer_size=self.bufferSize,
                                      transient=True,
                                      storage=dtypes.StorageType.FPGA_Local)
        self.fpga_stream = stream_inp

        state.add_memlet_path(
            stream_inp,
            *(entry_nodes if entry_nodes else []),
            libNode,
            dst_conn=libConnector,
            memlet=Memlet.simple(
                stream_inp,
                "0",
                num_accesses=self.rows * self.columns  #, veclen=self.veclen
            ))

    def getCopySize(self):

        return self.rows * self.columns

    def stream(self, state, src, memSize, destName='', access=False):

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
                                    buffer_size=self.bufferSize,
                                    transient=True,
                                    storage=dtypes.StorageType.FPGA_Local)

        read_tasklet = state.add_tasklet('sR_{}'.format(dest), ['inCon'],
                                         ['outCon'], 'outCon = inCon')

        firstDimMap_entry = None
        firstDimMap_exit = None
        secondDimMap_entry = None
        secondDimMap_exit = None

        repeatMap_entry = None
        repeatMap_exit = None

        if self.repeat != 1:
            repeatMap_entry, repeatMap_exit = state.add_map(
                'repeat_{}_map'.format(dest),
                dict(r='0:{0}'.format(self.repeat)),
                schedule=dtypes.ScheduleType.FPGA_Device)

        range = '0:{0}'.format(self.rowRepeat)
        if self.increasedRowRepeat:
            if self.reverse:
                range = 'i:{0}/{1}'.format(self.rows, self.rowTile)
            else:
                range = '0:i+1'

        rowRepeatMap_entry = None
        rowRepeatMap_exit = None
        if self.tileByRow and self.rowRepeat != 1:
            rowRepeatMap_entry, rowRepeatMap_exit = state.add_map(
                'rowRepeat_{}_map'.format(dest),
                dict(r_row=range),
                schedule=dtypes.ScheduleType.FPGA_Device)

        # Block ordering
        # ---------- ----------
        if self.blockByRow:

            range = '0:{0}'.format(self.rows / self.rowTile)
            if self.rowPyramid:
                if self.reverse:
                    range = 'r:{0}/{1}'.format(self.rows, self.rowTile)
                else:
                    range = '0:r+1'

            firstDimMap_entry, firstDimMap_exit = state.add_map(
                'streamfirstDimMap_{}_map'.format(dest),
                dict(i=range),
                schedule=dtypes.ScheduleType.FPGA_Device)

            secondDimMap_entry, secondDimMap_exit = state.add_map(
                'streamsecondDimMap_{}_map'.format(dest),
                dict(j='0:{0}'.format(self.columns / self.colTile)),
                schedule=dtypes.ScheduleType.FPGA_Device)

        else:

            secondDimMap_entry, secondDimMap_exit = state.add_map(
                'streamfirstDimMap_{}_map'.format(dest),
                dict(i='0:{0}'.format(self.rows / self.rowTile)),
                schedule=dtypes.ScheduleType.FPGA_Device)

            firstDimMap_entry, firstDimMap_exit = state.add_map(
                'streamsecondDimMap_{}_map'.format(dest),
                dict(j='0:{0}'.format(self.columns / self.colTile)),
                schedule=dtypes.ScheduleType.FPGA_Device)

        # Tile ordering
        # ---------- ----------
        if self.tileByRow:

            readRowTile_entry, readRowTile_exit = state.add_map(
                'streamReadRowTile_{}_map'.format(dest),
                dict(ii='0:{0}'.format(self.rowTile)),
                schedule=dtypes.ScheduleType.FPGA_Device,
                # unroll=(not self.tileByRow)
            )

            readColTile_entry, readColTile_exit = state.add_map(
                'streamReadColTile_{}_map'.format(dest),
                dict(jj='0:{0}/{1}'.format(self.colTile, self.veclen)),
                schedule=dtypes.ScheduleType.FPGA_Device,
                # unroll=self.tileByRow
            )

            if self.rowRepeat != 1:

                if self.repeat != 1:

                    state.add_memlet_path(
                        data_in,
                        repeatMap_entry,
                        firstDimMap_entry,
                        rowRepeatMap_entry,
                        secondDimMap_entry,
                        readRowTile_entry,
                        readColTile_entry,
                        read_tasklet,
                        dst_conn='inCon',
                        memlet=Memlet.simple(
                            data_in.data,
                            '(i *{0} + ii) * {1} + (j * {2} + jj * {3})'.format(
                                self.rowTile, self.columns, self.colTile,
                                self.veclen))  #, veclen=self.veclen)
                    )

                    state.add_memlet_path(
                        read_tasklet,
                        readColTile_exit,
                        readRowTile_exit,
                        secondDimMap_exit,
                        rowRepeatMap_exit,
                        firstDimMap_exit,
                        repeatMap_exit,
                        data_out,
                        src_conn='outCon',
                        memlet=Memlet.simple(data_out.data,
                                             '0')  #, veclen=self.veclen)
                    )

                else:

                    state.add_memlet_path(
                        data_in,
                        firstDimMap_entry,
                        rowRepeatMap_entry,
                        secondDimMap_entry,
                        readRowTile_entry,
                        readColTile_entry,
                        read_tasklet,
                        dst_conn='inCon',
                        memlet=Memlet.simple(
                            data_in.data,
                            '(i *{0} + ii) * {1} + (j * {2} + jj * {3})'.format(
                                self.rowTile, self.columns, self.colTile,
                                self.veclen))  #, veclen=self.veclen)
                    )

                    state.add_memlet_path(
                        read_tasklet,
                        readColTile_exit,
                        readRowTile_exit,
                        secondDimMap_exit,
                        rowRepeatMap_exit,
                        firstDimMap_exit,
                        data_out,
                        src_conn='outCon',
                        memlet=Memlet.simple(data_out.data,
                                             '0')  #, veclen=self.veclen)
                    )

            else:

                if self.repeat != 1:

                    state.add_memlet_path(
                        data_in,
                        repeatMap_entry,
                        firstDimMap_entry,
                        secondDimMap_entry,
                        readRowTile_entry,
                        readColTile_entry,
                        read_tasklet,
                        dst_conn='inCon',
                        memlet=Memlet.simple(
                            data_in.data,
                            '(i *{0} + ii) * {1} + (j * {2} + jj * {3})'.format(
                                self.rowTile, self.columns, self.colTile,
                                self.veclen))  #, veclen=self.veclen)
                    )

                    state.add_memlet_path(
                        read_tasklet,
                        readColTile_exit,
                        readRowTile_exit,
                        secondDimMap_exit,
                        firstDimMap_exit,
                        repeatMap_exit,
                        data_out,
                        src_conn='outCon',
                        memlet=Memlet.simple(data_out.data,
                                             '0')  #, veclen=self.veclen)
                    )

                else:

                    state.add_memlet_path(
                        data_in,
                        firstDimMap_entry,
                        secondDimMap_entry,
                        readRowTile_entry,
                        readColTile_entry,
                        read_tasklet,
                        dst_conn='inCon',
                        memlet=Memlet.simple(
                            data_in.data,
                            '(i *{0} + ii) * {1} + (j * {2} + jj * {3})'.format(
                                self.rowTile, self.columns, self.colTile,
                                self.veclen))  #, veclen=self.veclen)
                    )

                    state.add_memlet_path(
                        read_tasklet,
                        readColTile_exit,
                        readRowTile_exit,
                        secondDimMap_exit,
                        firstDimMap_exit,
                        data_out,
                        src_conn='outCon',
                        memlet=Memlet.simple(data_out.data,
                                             '0')  #, veclen=self.veclen)
                    )

        else:

            assert self.veclen == 1, "Vectorization not supported for streaming by columns, assume row-major storage"

            readRowTile_entry, readRowTile_exit = state.add_map(
                'streamReadRowTile_{}_map'.format(dest),
                dict(ii='0:{0}'.format(self.rowTile)),
                schedule=dtypes.ScheduleType.FPGA_Device,
                # unroll=(not self.tileByRow)
            )

            readColTile_entry, readColTile_exit = state.add_map(
                'streamReadColTile_{}_map'.format(dest),
                dict(jj='0:{0}'.format(self.colTile)),
                schedule=dtypes.ScheduleType.FPGA_Device,
                # unroll=self.tileByRow
            )

            if self.repeat > 1:

                state.add_memlet_path(
                    data_in,
                    repeatMap_entry,
                    firstDimMap_entry,
                    secondDimMap_entry,
                    readColTile_entry,
                    readRowTile_entry,
                    read_tasklet,
                    dst_conn='inCon',
                    memlet=Memlet.simple(
                        data_in.data,
                        '(i *{0} + ii) * {1} + (j * {2} +jj)'.format(
                            self.rowTile, self.columns, self.colTile)))

                state.add_memlet_path(read_tasklet,
                                      readRowTile_exit,
                                      readColTile_exit,
                                      firstDimMap_exit,
                                      secondDimMap_exit,
                                      repeatMap_exit,
                                      data_out,
                                      src_conn='outCon',
                                      memlet=Memlet.simple(data_out.data, '0'))

            else:

                state.add_memlet_path(
                    data_in,
                    firstDimMap_entry,
                    secondDimMap_entry,
                    readColTile_entry,
                    readRowTile_entry,
                    read_tasklet,
                    dst_conn='inCon',
                    memlet=Memlet.simple(
                        data_in.data,
                        '(i *{0} + ii) * {1} + (j * {2} +jj)'.format(
                            self.rowTile, self.columns, self.colTile)))

                state.add_memlet_path(read_tasklet,
                                      readRowTile_exit,
                                      readColTile_exit,
                                      firstDimMap_exit,
                                      secondDimMap_exit,
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
    def __init__(self,
                 destination,
                 mem_size,
                 dtype,
                 buffer_size=None,
                 veclen=1,
                 unroll=False,
                 unroll_width=1,
                 repeat=1):

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
        self.repeat = repeat

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
                       access=False,
                       entry_nodes=None,
                       exit_nodes=None):

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
                              *(exit_nodes if exit_nodes else []),
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

        repeat_map_entry = None
        repeat_map_exit = None

        if self.repeat != 1:
            repeat_map_entry, repeat_map_exit = state.add_map(
                'streamRepeat_{}_map'.format(dest),
                dict(r='0:{}'.format(self.repeat)),
                schedule=dtypes.ScheduleType.FPGA_Device)

        write_map_entry, write_map_exit = state.add_map(
            'streamWrite_{}_map'.format(src),
            dict(i='0:{0}/{1}'.format(self.mem_size, self.veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=self.unroll)

        write_tasklet = state.add_tasklet('sW_{}'.format(dest), ['inCon'],
                                          ['outCon'], 'outCon = inCon')

        if self.repeat != 1:
            state.add_memlet_path(data_in,
                                  repeat_map_entry,
                                  write_map_entry,
                                  write_tasklet,
                                  dst_conn='inCon',
                                  memlet=Memlet.simple(data_in.data, '0'))

            state.add_memlet_path(write_tasklet,
                                  write_map_exit,
                                  repeat_map_exit,
                                  data_out,
                                  src_conn='outCon',
                                  memlet=Memlet.simple(data_out.data, 'i'))
        else:
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


class StreamWriteMatrixFull():
    def __init__(
        self,
        destination,
        rows,
        columns,
        rowTile,
        colTile,
        dtype,
        bufferSize=32,
        veclen=1,
        blockByRow=True,  # TODO implement block by column
        tileByRow=True):

        self.destination = destination
        self.rows = rows
        self.columns = columns
        self.rowTile = rowTile
        self.colTile = colTile
        self.dtype = dtype
        self.bufferSize = bufferSize
        self.veclen = veclen

        self.blockByRow = blockByRow
        self.tileByRow = tileByRow

        self.fpga_data = None
        self.fpga_dataName = None
        self.fpga_stream = None

    def copy_to_cpu(self, sdfg, postState, bank=None):

        fpga_outputs, fpgaOut_names = mem_ops.fpga_copy_global_to_cpu(
            sdfg,
            postState, [self.destination], [self.rows * self.columns],
            [self.dtype],
            bank=bank)

        self.fpga_data = fpga_outputs[0]
        self.fpga_dataName = fpgaOut_names[0]

    def connect_to_lib(self, sdfg, state, libNode, libConnector, access=False,
                       entry_nodes=None,
                       exit_nodes=None):

        out_mem, out_name = self.stream(sdfg,
                                        state,
                                        self.fpga_data.data,
                                        srcName=libConnector,
                                        access=access)

        vec_type = vector(self.dtype, self.veclen)
        stream_out = state.add_stream(
            out_name,
            vec_type,
            #veclen=self.veclen,
            buffer_size=self.bufferSize,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local)
        self.fpga_stream = stream_out

        state.add_memlet_path(
            libNode,
            *(exit_nodes if exit_nodes else []),
            stream_out,
            src_conn=libConnector,
            memlet=Memlet.simple(
                stream_out,
                "0",
                num_accesses=self.rows * self.columns  #, veclen=self.veclen
            ))

    def get_copy_size(self):

        return self.rows * self.columns

    def stream(self, sdfg, state, dest, srcName='', access=False):

        src = dest + "_"
        if srcName != '':
            src += srcName + "_"
        src += "wS"

        vec_type = vector(self.dtype, self.veclen)
        data_in = state.add_stream(
            src,
            vec_type,
            #veclen=self.veclen,
            buffer_size=self.bufferSize,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local)

        data_out = None
        if access:
            data_out = self.fpga_data
        else:
            data_out = state.add_write(dest)

        readRows_entry, readRows_exit = state.add_map(
            'streamReadRows_{}_map'.format(dest),
            dict(i='0:{0}'.format(self.rows / self.rowTile)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        readCols_entry, readCols_exit = state.add_map(
            'streamReadCols_{}_map'.format(dest),
            dict(j='0:{0}'.format(self.columns / self.colTile)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        read_tasklet = state.add_tasklet('sW_{}'.format(dest), ['inCon'],
                                         ['outCon'], 'outCon = inCon')

        if self.tileByRow:

            readRowTile_entry, readRowTile_exit = state.add_map(
                'streamReadRowTile_{}_map'.format(dest),
                dict(ii='0:{0}'.format(self.rowTile)),
                schedule=dtypes.ScheduleType.FPGA_Device)

            readColTile_entry, readColTile_exit = state.add_map(
                'streamReadColTile_{}_map'.format(dest),
                dict(jj='0:{0}/{1}'.format(self.colTile, self.veclen)),
                schedule=dtypes.ScheduleType.FPGA_Device)

            state.add_memlet_path(
                data_in,
                readRows_entry,
                readCols_entry,
                readRowTile_entry,
                readColTile_entry,
                read_tasklet,
                dst_conn='inCon',
                memlet=Memlet.simple(data_in.data, '0')  #, veclen=self.veclen)
            )

            state.add_memlet_path(
                read_tasklet,
                readColTile_exit,
                readRowTile_exit,
                readCols_exit,
                readRows_exit,
                data_out,
                src_conn='outCon',
                memlet=Memlet.simple(
                    data_out.data,
                    '(i *{0} + ii) * {1} + (j * {2} + jj * {3})'.format(
                        self.rowTile, self.columns, self.colTile,
                        self.veclen)  #, veclen=self.veclen)
                ))

        else:

            assert self.veclen == 1, "Vectorization not supported for streaming by columns, assume row-major storage"

            readRowTile_entry, readRowTile_exit = state.add_map(
                'streamReadRowTile_{}_map'.format(dest),
                dict(ii='0:{0}'.format(self.rowTile)),
                schedule=dtypes.ScheduleType.FPGA_Device)

            readColTile_entry, readColTile_exit = state.add_map(
                'streamReadColTile_{}_map'.format(dest),
                dict(jj='0:{0}'.format(self.colTile)),
                schedule=dtypes.ScheduleType.FPGA_Device)

            state.add_memlet_path(data_in,
                                  readRows_entry,
                                  readCols_entry,
                                  readColTile_entry,
                                  readRowTile_entry,
                                  read_tasklet,
                                  dst_conn='inCon',
                                  memlet=Memlet.simple(data_in.data, '0'))

            state.add_memlet_path(
                read_tasklet,
                readRowTile_exit,
                readColTile_exit,
                readRows_exit,
                readCols_exit,
                data_out,
                src_conn='outCon',
                memlet=Memlet.simple(
                    data_out.data, '(i *{0} + ii) * {1} + (j * {2} +jj)'.format(
                        self.rowTile, self.columns, self.colTile)))

        return data_in, src
