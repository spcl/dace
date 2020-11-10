# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import environments
from dace import dtypes
from dace import config

from dace.libraries.blas.utility.fpga_helper import StreamWriteVector, StreamReadVector


@dace.library.expansion
class ExpandAxpyVectorized(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, veclen, n, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        vec_type = dace.vector(dtype, veclen)

        vec_add_sdfg = dace.SDFG('vec_add_graph')
        vec_add_state = vec_add_sdfg.add_state()

        vec_add_sdfg.add_symbol(a.name, dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        vec_add_sdfg.add_array('_x', shape=[n / veclen], dtype=vec_type)
        vec_add_sdfg.add_array('_y', shape=[n / veclen], dtype=vec_type)
        vec_add_sdfg.add_array('_res', shape=[n / veclen], dtype=vec_type)

        x_in = vec_add_state.add_read('_x')
        y_in = vec_add_state.add_read('_y')
        z_out = vec_add_state.add_write('_res')

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        vec_map_entry, vec_map_exit = vec_add_state.add_map(
            'vecAdd_map', dict(i='0:{0}/{1}'.format(n, veclen)))

        vecAdd_tasklet = vec_add_state.add_tasklet(
            'vecAdd_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        vec_add_state.add_memlet_path(x_in,
                                      vec_map_entry,
                                      vecAdd_tasklet,
                                      dst_conn='x_con',
                                      memlet=dace.Memlet.simple(x_in.data, 'i'))

        vec_add_state.add_memlet_path(y_in,
                                      vec_map_entry,
                                      vecAdd_tasklet,
                                      dst_conn='y_con',
                                      memlet=dace.Memlet.simple(y_in.data, 'i'))

        vec_add_state.add_memlet_path(vecAdd_tasklet,
                                      vec_map_exit,
                                      z_out,
                                      src_conn='z_con',
                                      memlet=dace.Memlet.simple(
                                          z_out.data, 'i'))

        return vec_add_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandAxpyVectorized.make_sdfg(node.dtype, node.veclen, node.n,
                                              node.a)


@dace.library.expansion
class ExpandAxpyFPGAStreaming(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, veclen, n, a, buffer_size_x, buffer_size_y,
                  buffer_size_res):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        vec_type = dace.vector(dtype, veclen)

        vec_add_sdfg = dace.SDFG('vec_add_graph')
        vec_add_state = vec_add_sdfg.add_state()

        vec_add_sdfg.add_symbol(a.name, dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        # vec_add_sdfg.add_scalar('_a', dtype=dtype, storage=dtypes.StorageType.FPGA_Global)

        x_in = vec_add_state.add_stream('_x',
                                        vec_type,
                                        buffer_size=buffer_size_x,
                                        storage=dtypes.StorageType.FPGA_Local)
        y_in = vec_add_state.add_stream('_y',
                                        vec_type,
                                        buffer_size=buffer_size_y,
                                        storage=dtypes.StorageType.FPGA_Local)
        z_out = vec_add_state.add_stream('_res',
                                         vec_type,
                                         buffer_size=buffer_size_res,
                                         storage=dtypes.StorageType.FPGA_Local)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        vec_map_entry, vec_map_exit = vec_add_state.add_map(
            'vecAdd_map',
            dict(i='0:{0}/{1}'.format(n, veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        vecAdd_tasklet = vec_add_state.add_tasklet(
            'vecAdd_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        vec_add_state.add_memlet_path(x_in,
                                      vec_map_entry,
                                      vecAdd_tasklet,
                                      dst_conn='x_con',
                                      memlet=dace.Memlet.simple(x_in.data, '0'))

        vec_add_state.add_memlet_path(y_in,
                                      vec_map_entry,
                                      vecAdd_tasklet,
                                      dst_conn='y_con',
                                      memlet=dace.Memlet.simple(y_in.data, '0'))

        vec_add_state.add_memlet_path(vecAdd_tasklet,
                                      vec_map_exit,
                                      z_out,
                                      src_conn='z_con',
                                      memlet=dace.Memlet.simple(
                                          z_out.data, '0'))

        return vec_add_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        for e in state.in_edges(node):
            if e.dst_conn == "_x":
                buffer_size_x = sdfg.arrays[e.data.data].buffer_size
            elif e.dst_conn == "_y":
                buffer_size_y = sdfg.arrays[e.data.data].buffer_size
        for e in state.out_edges(node):
            if e.src_conn == "_res":
                buffer_size_res = sdfg.arrays[e.data.data].buffer_size
        return ExpandAxpyFPGAStreaming.make_sdfg(node.dtype, int(node.veclen),
                                                 node.n, node.a, buffer_size_x,
                                                 buffer_size_y, buffer_size_res)


@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):
    """Executes a * _x + _y. It implements the BLAS AXPY operation
        a vector-scalar product with a vector addition.

        Implementations:
        pure: dace primitive based implementation
        fpga_stream: FPGA implementation optimized for streaming
    """

    # Global properties
    implementations = {
        "pure": ExpandAxpyVectorized,
        "fpga_stream": ExpandAxpyFPGAStreaming
    }
    default_implementation = 'pure'

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)
    n = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("n"))
    a = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("a"))

    def __init__(self,
                 name,
                 dtype=dace.float32,
                 veclen=1,
                 n=None,
                 a=None,
                 *args,
                 **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_res"},
                         **kwargs)
        self.dtype = dtype
        self.veclen = veclen
        self.n = n or dace.symbolic.symbol("n")
        self.a = a or dace.symbolic.symbol("a")

    def compare(self, other):

        if (self.dtype == other.dtype and self.veclen == other.veclen
                and self.implementation == other.implementation):

            return True
        else:
            return False

    def stream_production_latency(self):
        return 0

    def stream_consumption_latency(self):
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

        return True

    def make_stream_reader(self):
        return {
            "_x": StreamReadVector('-',
                                   self.n,
                                   self.dtype,
                                   veclen=int(self.veclen)),
            "_y": StreamReadVector('-',
                                   self.n,
                                   self.dtype,
                                   veclen=int(self.veclen))
        }

    def make_stream_writer(self):
        return {
            "_res":
            StreamWriteVector('-', self.n, self.dtype, veclen=int(self.veclen))
        }
