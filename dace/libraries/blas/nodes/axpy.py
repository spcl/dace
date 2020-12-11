# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import environments
from dace import config, symbolic, SDFG, SDFGState, dtypes, memlet as mm
from dace.frontend.common import op_repository as oprepo

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

        vec_add_sdfg.add_symbol('a', dtype)

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
            'axpy_map', dict(i='0:{0}'.format(n)))

        axpy_tasklet = vec_add_state.add_tasklet(
            'axpy_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        vec_add_state.add_memlet_path(x_in,
                                      vec_map_entry,
                                      axpy_tasklet,
                                      dst_conn='x_con',
                                      memlet=dace.Memlet.simple(x_in.data, 'i'))

        vec_add_state.add_memlet_path(y_in,
                                      vec_map_entry,
                                      axpy_tasklet,
                                      dst_conn='y_con',
                                      memlet=dace.Memlet.simple(y_in.data, 'i'))

        vec_add_state.add_memlet_path(axpy_tasklet,
                                      vec_map_exit,
                                      z_out,
                                      src_conn='z_con',
                                      memlet=dace.Memlet.simple(
                                          z_out.data, 'i'))

        return vec_add_sdfg

    @staticmethod
    def expansion(node, state: SDFGState, sdfg, vec_width=1):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        n = state.out_edges(node)[0].data.subset.size()[0]
        return ExpandAxpyVectorized.make_sdfg(node.dtype, vec_width, n,
                                              node.a)


@dace.library.expansion
class ExpandAxpyFPGAStreaming(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, veclen, n, a, buffer_size_x, buffer_size_y,
                  buffer_size_res, streaming):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        vec_type = dace.vector(dtype, veclen)

        vec_add_sdfg = dace.SDFG('vec_add_graph')
        vec_add_state = vec_add_sdfg.add_state("axpy_compute_state")

        vec_add_sdfg.add_symbol('a', dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------

        if streaming:
            x_in = vec_add_state.add_stream(
                '_x',
                vec_type,
                buffer_size=buffer_size_x,
                storage=dtypes.StorageType.FPGA_Local)
            y_in = vec_add_state.add_stream(
                '_y',
                vec_type,
                buffer_size=buffer_size_y,
                storage=dtypes.StorageType.FPGA_Local)
            z_out = vec_add_state.add_stream(
                '_res',
                vec_type,
                buffer_size=buffer_size_res,
                storage=dtypes.StorageType.FPGA_Local)
        else:
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
            'axpy_map',
            dict(i='0:{0}/{1}'.format(n, veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        axpy_tasklet = vec_add_state.add_tasklet(
            'axpy_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        access_vol = '0' if streaming else 'i'
        vec_add_state.add_memlet_path(x_in,
                                      vec_map_entry,
                                      axpy_tasklet,
                                      dst_conn='x_con',
                                      memlet=dace.Memlet.simple(
                                          x_in.data, '{}'.format(access_vol)))

        vec_add_state.add_memlet_path(y_in,
                                      vec_map_entry,
                                      axpy_tasklet,
                                      dst_conn='y_con',
                                      memlet=dace.Memlet.simple(
                                          y_in.data, '{}'.format(access_vol)))

        vec_add_state.add_memlet_path(axpy_tasklet,
                                      vec_map_exit,
                                      z_out,
                                      src_conn='z_con',
                                      memlet=dace.Memlet.simple(
                                          z_out.data, '{}'.format(access_vol)))

        return vec_add_sdfg

    @staticmethod
    def expansion(node, state, sdfg, vec_width=1, n=symbolic.symbol('n')):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")

        buffer_size_x = 0
        buffer_size_y = 0
        buffer_size_res = 0

        streaming = False
        streaming_nodes = 0

        for e in state.in_edges(node):

            if isinstance(sdfg.arrays[e.data.data],
                          dace.data.Stream) and e.dst_conn == "_x":
                buffer_size_x = sdfg.arrays[e.data.data].buffer_size
                streaming = True
                streaming_nodes = streaming_nodes + 1

            elif isinstance(sdfg.arrays[e.data.data],
                            dace.data.Stream) and e.dst_conn == "_y":
                buffer_size_y = sdfg.arrays[e.data.data].buffer_size
                streaming = True
                streaming_nodes = streaming_nodes + 1

        for e in state.out_edges(node):
            if isinstance(sdfg.arrays[e.data.data],
                          dace.data.Stream) and e.src_conn == "_res":
                buffer_size_res = sdfg.arrays[e.data.data].buffer_size
                streaming = True
                streaming_nodes = streaming_nodes + 1

        if streaming and streaming_nodes < 3:
            raise ValueError(
                "All input and outputs must be of same type either Array or Stream"
            )

        return ExpandAxpyFPGAStreaming.make_sdfg(node.dtype, int(vec_width),
                                                 n, node.a, buffer_size_x,
                                                 buffer_size_y, buffer_size_res,
                                                 streaming)


@dace.library.expansion
class ExpandAxpyIntelFPGAVectorized(ExpandTransformation):

    # Intel FPGA expansion, inputs are Global FPGA Buffer
    # Plain data type, computation is internally unrolled

    environments = []

    @staticmethod
    def make_sdfg(dtype, vec_width, a):

        # --------------------
        # SETUP GRAPH
        # --------------------
        n = dace.symbol("n")

        axpy_sdfg = dace.SDFG('axpy_graph')
        axpy_state = axpy_sdfg.add_state()

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        axpy_sdfg.add_array('_x',
                            shape=[n],
                            dtype=dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
        axpy_sdfg.add_array('_y',
                            shape=[n],
                            dtype=dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
        axpy_sdfg.add_array('_res',
                            shape=[n],
                            dtype=dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)

        x_in = axpy_state.add_read('_x')
        y_in = axpy_state.add_read('_y')
        z_out = axpy_state.add_write('_res')

        # ---------- ---------------------
        # COMPUTE: we have two nested maps
        # ---------- --------------------
        #Strip mined loop
        outer_map_entry, outer_map_exit = axpy_state.add_map(
            'outer_map',
            dict(i='0:{0}/{1}'.format(n, vec_width)),
            schedule=dace.dtypes.ScheduleType.FPGA_Device)
        compute_map_entry, compute_map_exit = axpy_state.add_map(
            'compute_map',
            dict(j='0:{}'.format(vec_width)),
            schedule=dace.dtypes.ScheduleType.FPGA_Device,
            unroll=True)

        axpy_tasklet = axpy_state.add_tasklet(
            'axpy_task', ['x_con', 'y_con'], ['z_con'],
            'z_con = {} * x_con + y_con'.format(a))

        axpy_state.add_memlet_path(x_in,
                                   outer_map_entry,
                                   compute_map_entry,
                                   axpy_tasklet,
                                   dst_conn='x_con',
                                   memlet=dace.Memlet.simple(
                                       x_in.data, 'i*{}+j'.format(vec_width)))

        axpy_state.add_memlet_path(y_in,
                                   outer_map_entry,
                                   compute_map_entry,
                                   axpy_tasklet,
                                   dst_conn='y_con',
                                   memlet=dace.Memlet.simple(
                                       y_in.data, 'i*{}+j'.format(vec_width)))

        axpy_state.add_memlet_path(axpy_tasklet,
                                   compute_map_exit,
                                   outer_map_exit,
                                   z_out,
                                   src_conn='z_con',
                                   memlet=dace.Memlet.simple(
                                       z_out.data, 'i*{}+j'.format(vec_width)))

        return axpy_sdfg

    @staticmethod
    def expansion(node, state, sdfg, vec_width=1):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")

        node_sdfg = ExpandAxpyIntelFPGAVectorized.make_sdfg(
            node.dtype, int(vec_width), node.a)

        return node_sdfg


@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):
    """Executes a * _x + _y. It implements the BLAS AXPY operation
        a vector-scalar product with a vector addition.

        Implementations:
        pure: dace primitive based implementation
        fpga: FPGA implementation optimized for both streaming and array based inputs
    """

    # Global properties
    implementations = {
        "pure": ExpandAxpyVectorized,
        "fpga": ExpandAxpyFPGAStreaming,
        "Intel_FPGA_DRAM": ExpandAxpyIntelFPGAVectorized
    }
    default_implementation = 'pure'

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    a = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("a"))

    def __init__(self,
                 name,
                 dtype=dace.float32,
                 a=None,
                 *args,
                 **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_res"},
                         **kwargs)
        self.dtype = dtype
        self.a = a or dace.symbolic.symbol("a")

    def compare(self, other):

        if (self.dtype == other.dtype and self.veclen == other.veclen
                and self.implementation == other.implementation):

            return True
        else:
            return False

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

    # TODO: Do these methods belong here?
    def stream_production_latency(self):
        return 0

    def stream_consumption_latency(self):
        return {"_x": 0, "_y": 0}

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


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.axpy')
@oprepo.replaces('dace.libraries.blas.Axpy')
def axpy_libnode(sdfg: SDFG, state: SDFGState, a, x, y, result):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Axpy('axpy', dtype=sdfg.arrays[x].dtype, a=a)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_res', res, None, mm.Memlet(result))

    return []
