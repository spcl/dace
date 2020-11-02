import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import dtypes

from dace.libraries.blas.utility.memory_operations import *
from dace.libraries.blas.utility.fpga_helper import *


@dace.library.expansion
class ExpandAxpyVectorized(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, vecWidth, n, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        vecType = dace.vector(dtype, vecWidth)

        vecAdd_sdfg = dace.SDFG('vecAdd_graph')
        vecAdd_state = vecAdd_sdfg.add_state()

        vecAdd_sdfg.add_symbol(a.name, dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        vecAdd_sdfg.add_array('_x', shape=[n/vecWidth], dtype=vecType)
        vecAdd_sdfg.add_array('_y', shape=[n/vecWidth], dtype=vecType)
        vecAdd_sdfg.add_array('_res', shape=[n/vecWidth], dtype=vecType)

        x_in = vecAdd_state.add_read('_x')
        y_in = vecAdd_state.add_read('_y')
        z_out = vecAdd_state.add_write('_res')

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        vecMap_entry, vecMap_exit = vecAdd_state.add_map(
            'vecAdd_map', dict(i='0:{0}/{1}'.format(n, vecWidth)))

        vecAdd_tasklet = vecAdd_state.add_tasklet(
            'vecAdd_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        vecAdd_state.add_memlet_path(x_in,
                                     vecMap_entry,
                                     vecAdd_tasklet,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet.simple(
                                         x_in.data,
                                         'i'))

        vecAdd_state.add_memlet_path(y_in,
                                     vecMap_entry,
                                     vecAdd_tasklet,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet.simple(
                                         y_in.data,
                                         'i'))

        vecAdd_state.add_memlet_path(vecAdd_tasklet,
                                     vecMap_exit,
                                     z_out,
                                     src_conn='z_con',
                                     memlet=dace.Memlet.simple(
                                         z_out.data,
                                         'i'))

        return vecAdd_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandAxpyVectorized.make_sdfg(node.dtype, node.vecWidth, node.n,
                                              node.a)


@dace.library.expansion
class ExpandAxpyFPGAStreaming(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, vecWidth, n, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        vecType = dace.vector(dtype, vecWidth)

        vecAdd_sdfg = dace.SDFG('vecAdd_graph')
        vecAdd_state = vecAdd_sdfg.add_state()

        vecAdd_sdfg.add_symbol(a.name, dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        # vecAdd_sdfg.add_scalar('_a', dtype=dtype, storage=dtypes.StorageType.FPGA_Global)

        x_in = vecAdd_state.add_stream('_x',
                                       vecType,
                                       buffer_size=32,
                                       storage=dtypes.StorageType.FPGA_Local)
        y_in = vecAdd_state.add_stream('_y',
                                       vecType,
                                       buffer_size=32,
                                       storage=dtypes.StorageType.FPGA_Local)
        z_out = vecAdd_state.add_stream('_res',
                                        vecType,
                                        buffer_size=32,
                                        storage=dtypes.StorageType.FPGA_Local)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        vecMap_entry, vecMap_exit = vecAdd_state.add_map(
            'vecAdd_map',
            dict(i='0:{0}/{1}'.format(n, vecWidth)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet = vecAdd_state.add_tasklet(
            'vecAdd_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        vecAdd_state.add_memlet_path(x_in,
                                     vecMap_entry,
                                     vecAdd_tasklet,
                                     dst_conn='x_con',
                                     memlet=dace.Memlet.simple(
                                         x_in.data, '0'))

        vecAdd_state.add_memlet_path(y_in,
                                     vecMap_entry,
                                     vecAdd_tasklet,
                                     dst_conn='y_con',
                                     memlet=dace.Memlet.simple(
                                         y_in.data, '0'))

        vecAdd_state.add_memlet_path(vecAdd_tasklet,
                                     vecMap_exit,
                                     z_out,
                                     src_conn='z_con',
                                     memlet=dace.Memlet.simple(
                                         z_out.data, '0'))

        return vecAdd_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandAxpyFPGAStreaming.make_sdfg(node.dtype, int(node.vecWidth),
                                                 node.n, node.a)



@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandAxpyVectorized,
        "fpga_stream": ExpandAxpyFPGAStreaming
    }
    default_implementation = 'pure'

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    vecWidth = dace.properties.SymbolicProperty(allow_none=False, default=1)
    n = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("n"))
    a = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("a"))

    def __init__(self,
                 name,
                 dtype=dace.float32,
                 vecWidth=1,
                 n=dace.symbolic.symbol("n"),
                 a=dace.symbolic.symbol("a"),
                 *args,
                 **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_res"},
                         **kwargs)
        self.dtype = dtype
        self.vecWidth = vecWidth
        self.n = n
        self.a = a

    def compare(self, other):

        if (self.dtype == other.dtype and self.vecWidth == other.vecWidth
                and self.implementation == other.implementation):

            return True
        else:
            return False

    def streamProductionLatency(self):
        return 0

    def streamConsumptionLatency(self):
        return {"_x": 0, "_y": 0}

    def validate(self, sdfg, state):

        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to axpy")

        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from axpy")

        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()

        if len(size) != 1:
            raise ValueError("axpy only supported on 1-dimensional arrays")

        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to axpy must have equal size")

        if size != out_memlet.subset.size():
            raise ValueError("Output of axpy must have same size as input")


        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None
                or out_memlet.wcr is not None):
            raise ValueError("WCR on axpy memlets not supported")

        if (self.implementation == "cublas" or self.implementation == "mkl"
                or self.implementation == "openblas"):

            memletNo = 0 if in_edges[0].dst_conn == "_y" else 1

            if str(in_edges[memletNo].src) != str(out_edges[0].dst):
                raise ValueError(
                    "y input and output of axpy must be same memory for " +
                    self.implementation)

        return True

    def make_stream_reader(self):
        return {
            "_x":
            streamReadVector('-',
                             self.n,
                             self.dtype,
                             vecWidth=int(self.vecWidth)),
            "_y":
            streamReadVector('-',
                             self.n,
                             self.dtype,
                             vecWidth=int(self.vecWidth))
        }

    def make_stream_writer(self):
        return {
            "_res":
            streamWriteVector('-',
                              self.n,
                              self.dtype,
                              vecWidth=int(self.vecWidth))
        }
