"""
Various helper functions and classes for streaming BLAS operators on the FPGA

- FPGA state setup with copy to device and back to host
- streaming classes to stream data from/to memory arrays from/to FIFO queues
    using different access patterns
"""

import dace
from dace.memlet import Memlet
from dace import dtypes
from dace.libraries.blas.utility import memoryOperations as memOps



# ---------- ---------- ---------- ----------
# UTILITY
# ---------- ---------- ---------- ----------


def fpga_setup_states(sdfg, computeState):
    """
    Add states for copying data from and to the FPGA
    """

    preState = sdfg.add_state('copyToFPGA')
    postState = sdfg.add_state('copyToCPU')

    sdfg.add_edge(preState, computeState, dace.InterstateEdge(None))
    sdfg.add_edge(computeState, postState, dace.InterstateEdge(None))

    return (preState, postState)


def fpga_setup_connect_streamers(
        sdfg,
        state,
        libNodeIn,
        inputStreams,
        libNodeInCons,
        libNodeOut,
        outputStreams,
        libNodeOutCons,
        inputMemoryBanks=None,
        outputMemoryBanks=None
    ):
    """
    Add states for copying data from and to the FPGA
    and connect the given streaming classes with their
    access pattern and mem. location to the BLAS node
    """

    preState, postState = fpga_setup_states(sdfg, state)

    for i, stream, libCon in zip(range(len(libNodeInCons)), inputStreams, libNodeInCons):

        stream.copyToFPGA(
            sdfg, preState,
            bank=(None if inputMemoryBanks is None else inputMemoryBanks[i])
        )

        stream.connectToLib(
            sdfg,
            state,
            libNodeIn,
            libCon
        )

    for i, stream, libCon in zip(range(len(libNodeOutCons)), outputStreams, libNodeOutCons):

        stream.copyToCPU(
            sdfg, postState,
            bank=(None if outputMemoryBanks is None else outputMemoryBanks[i])
        )

        stream.connectToLib(
            sdfg,
            state,
            libNodeOut,
            libCon
        )

    return preState, postState



def fpga_setup_ConnectStreamersMultiNode(
        sdfg,
        state,
        libNodeIns,
        inputStreams,
        libNodeInCons,
        libNodeOuts,
        outputStreams,
        libNodeOutCons,
        inputMemoryBanks=None,
        outputMemoryBanks=None
    ):
    """
    Add states for copying data from and to the FPGA
    and connect the given streaming classes with their
    access pattern and mem. location to multiple diff. BLAS nodes
    """

    preState, postState = fpga_setup_states(sdfg, state)

    for i, stream, libCon, libNodeIn in zip(range(len(libNodeInCons)), inputStreams, libNodeInCons, libNodeIns):

        stream.copyToFPGA(
            sdfg, preState,
            bank=(None if inputMemoryBanks is None else inputMemoryBanks[i])
        )

        stream.connectToLib(
            sdfg,
            state,
            libNodeIn,
            libCon
        )

    for i, stream, libCon, libNodeOut in zip(range(len(libNodeOutCons)), outputStreams, libNodeOutCons, libNodeOuts):

        stream.copyToCPU(
            sdfg, postState,
            bank=(None if outputMemoryBanks is None else outputMemoryBanks[i])
        )

        stream.connectToLib(
            sdfg,
            state,
            libNodeOut,
            libCon
        )

    return preState, postState







# ---------- ---------- ---------- ----------
# BASE STREAMERS
# ---------- ---------- ---------- ----------
class streamReadBase():

    def connectToLib(self, sdfg, state, libNode, libConnector, access=False):
        print("WARNING, implement method 'connect' on child reader!")
        raise NotImplementedError

    def getCopySize(self):
        print("WARNING, implement method 'getCopySize' on child reader!")
        raise NotImplementedError

    def copyToFPGA(self, sdfg, preState, bank=None):
        print("WARNING, implement method 'copyToFPGA' on child reader!")
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError



class streamWriteBase():

    def connectToLib(self, sdfg, state, libNode, libConnector, access=False):
        print("WARNING, implement method 'connect' on child writer!")
        raise NotImplementedError

    def getCopySize(self):
        print("WARNING, implement method 'getCopySize' on child writer!")
        raise NotImplementedError

    def copyToCPU(self, sdfg, preState, bank=None):
        print("WARNING, implement method 'copyToCPU' on child writer!")
        raise NotImplementedError













# ---------- ---------- ---------- ----------
# READERS
# ---------- ---------- ---------- ----------
class streamReadVector(streamReadBase):

    def __init__(
            self,
            source,
            memSize,
            dtype,
            bufferSize=32,
            vecWidth=1,
            unroll=False,
            unrollWidth=1,
            repeat=1
        ):

        # if unrollWidth < vecWidth:
        #     unrollWidth = vecWidth

        assert unroll == False and unrollWidth == 1, "Unrolling not supported at the time"

        self.source = source
        self.memSize = memSize
        self.dtype = dtype

        self.bufferSize = bufferSize
        self.vecWidth = vecWidth
        self.unroll = unroll
        self.unrollWidth = unrollWidth
        self.repeat = repeat

        self.fpga_data = None
        self.fpga_dataName = None
        self.fpga_stream = None


    def __eq__(self, other):

        if (self.source == other.source and self.memSize == other.memSize and 
            self.dtype == other.dtype and self.vecWidth == other.vecWidth and
            self.repeat == other.repeat):

            return True
        else:
            return False
            



    def copyToFPGA(self, sdfg, preState, bank=None):


        fpga_inputs, fpgaIn_names = memOps.fpga_copyCPUToGlobal(
            sdfg,
            preState,
            [self.source],
            [self.memSize],
            [self.dtype],
            bank=bank
        )

        self.fpga_data = fpga_inputs[0]
        self.fpga_dataName = fpgaIn_names[0]


    def connectToLib(self, sdfg, state, libNode, libConnector, access=False):

        in_mem, in_name = self.stream(
            state,
            self.fpga_data.data,
            self.memSize,
            destName=libConnector,
            access=access
        )

        stream_inp = state.add_stream(
            in_name,
            self.dtype,
            veclen=self.vecWidth,
            buffer_size=self.bufferSize,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local
        )
        self.fpga_stream = stream_inp

        state.add_memlet_path(
            stream_inp, libNode,
            dst_conn=libConnector,
            memlet=Memlet.simple(
                stream_inp, "0", num_accesses=self.memSize, veclen=self.vecWidth
            )
        )


    def getCopySize(self):

        return self.memSize


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

        data_out = state.add_stream(
            dest,
            self.dtype,
            veclen=self.vecWidth,
            buffer_size=self.bufferSize,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local
        )

        repeatMap_entry = None
        repeatMap_exit = None

        if self.repeat != 1:
            repeatMap_entry, repeatMap_exit = state.add_map(
                'streamRepeat_{}_map'.format(dest),
                dict(r = '0:{}'.format(self.repeat)),
                schedule=dtypes.ScheduleType.FPGA_Device
            )

        readMap_entry, readMap_exit = state.add_map(
            'streamRead_{}_map'.format(dest),
            dict(i='0:{0}/{1}'.format(memSize, self.vecWidth)),
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=self.unroll
        )

        read_tasklet = state.add_tasklet(
            'sR_{}'.format(dest),
            ['inCon'],
            ['outCon'],
            'outCon = inCon'
        )

        if self.repeat != 1:

            state.add_memlet_path(
                data_in, repeatMap_entry, readMap_entry, read_tasklet,
                dst_conn='inCon',
                memlet=Memlet.simple(data_in.data, 'i * {}'.format(self.vecWidth), veclen=self.vecWidth)
            )

            state.add_memlet_path(
                read_tasklet, readMap_exit, repeatMap_exit, data_out,
                src_conn='outCon',
                memlet=Memlet.simple(data_out.data, '0', veclen=self.vecWidth)
            )

        else:

            state.add_memlet_path(
                data_in, readMap_entry, read_tasklet,
                dst_conn='inCon',
                memlet=Memlet.simple(data_in.data, 'i * {}'.format(self.vecWidth), veclen=self.vecWidth)
            )

            state.add_memlet_path(
                read_tasklet, readMap_exit, data_out,
                src_conn='outCon',
                memlet=Memlet.simple(data_out.data, '0', veclen=self.vecWidth)
            )

        return data_out, dest



# ---------- ---------- ---------- ----------
# WRITERS
# ---------- ---------- ---------- ----------


class streamWriteVector(streamWriteBase):

    def __init__(
            self,
            destination,
            memSize,
            dtype,
            bufferSize=32,
            vecWidth=1,
            unroll=False,
            unrollWidth=1,
        ):

        # if unrollWidth < vecWidth:
        #     unrollWidth = vecWidth

        assert unroll is False and unrollWidth == 1, "Unrolling not supported at the time"

        self.destination = destination
        self.memSize = memSize
        self.dtype = dtype

        self.bufferSize = bufferSize
        self.vecWidth = vecWidth
        self.unroll = unroll
        self.unrollWidth = unrollWidth

        self.fpga_data = None
        self.fpga_dataName = None
        self.fpga_stream = None


    def copyToCPU(self, sdfg, postState, bank=None):

        fpga_outputs, fpgaOut_names = memOps.fpga_copyGlobalToCPU(
            sdfg,
            postState,
            [self.destination],
            [self.memSize],
            [self.dtype],
            bank=bank
        )

        self.fpga_data = fpga_outputs[0]
        self.fpga_dataName = fpgaOut_names[0]


    def connectToLib(self, sdfg, state, libNode, libConnector, access=False):

        out_mem, out_name = self.stream(
            sdfg,
            state,
            self.fpga_data.data,
            self.memSize,
            self.dtype,
            srcName=libConnector,
            access=access
        )  

        stream_out = state.add_stream(
            out_name,
            self.dtype,
            veclen=self.vecWidth,
            buffer_size=self.bufferSize,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local
        )
        self.fpga_stream = stream_out

        state.add_memlet_path(
            libNode, stream_out,
            src_conn=libConnector,
            memlet=Memlet.simple(
                stream_out, "0", num_accesses=-1, veclen=self.vecWidth
            )
        )


    def getCopySize(self):

        return self.memSize


    def stream(self, sdfg, state, dest, memSize, dtype, srcName='', access=False):

        src = dest + "_"
        if srcName != '':
            src += srcName + "_" 
        src += "wS"

        data_in = state.add_stream(
            src,
            dtype,
            veclen=self.vecWidth,
            buffer_size=self.bufferSize,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local
        )

        data_out = None
        if access:
            data_out = self.fpga_data
        else:
            data_out = state.add_write(dest)

        writeMap_entry, writeMap_exit = state.add_map(
            'streamWrite_{}_map'.format(src),
            dict(i='0:{0}/{1}'.format(self.memSize, self.vecWidth)),
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=self.unroll
        )

        write_tasklet = state.add_tasklet(
            'sW_{}'.format(dest),
            ['inCon'],
            ['outCon'],
            'outCon = inCon'
        )

        state.add_memlet_path(
            data_in, writeMap_entry, write_tasklet,
            dst_conn='inCon',
            memlet=Memlet.simple(data_in.data, '0', veclen=self.vecWidth)
        )

        state.add_memlet_path(
            write_tasklet, writeMap_exit, data_out,
            src_conn='outCon',
            memlet=Memlet.simple(data_out.data, 'i * {}'.format(self.vecWidth), veclen=self.vecWidth)
        )

        return data_in, src
