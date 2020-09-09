import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.pattern_matching import ExpandTransformation
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
        vecAdd_sdfg.add_array('_x', shape=[n], dtype=vecType)
        vecAdd_sdfg.add_array('_y', shape=[n], dtype=vecType)
        vecAdd_sdfg.add_array('_res', shape=[n], dtype=vecType)

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


@dace.library.expansion
class ExpandAxpyMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def make_sdfg(dtype, n, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        axpy_sdfg = dace.SDFG('axpy_graph')
        axpy_state = axpy_sdfg.add_state()
        # init_state = axpy_sdfg.add_state()

        axpy_sdfg.add_symbol(a.name, dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        axpy_sdfg.add_array('_x', shape=[n], dtype=dtype)
        axpy_sdfg.add_array('_y', shape=[n], dtype=dtype)
        axpy_sdfg.add_array('_res', shape=[n], dtype=dtype)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        x_in = axpy_state.add_read('_x')
        y_out = axpy_state.add_write('_res')

        if dtype == dace.float32:
            func = "saxpy"
        elif dtype == dace.float64:
            func = "daxpy"
        else:
            raise ValueError("Unsupported type for BLAS axpy product: " +
                             str(dtype))

        code = "cblas_{0}({1}, {2}, x, 1, y, 1);".format(func, n, a)

        task = axpy_state.add_tasklet('axpy_blas_task', ['x'], ['y'],
                                      code,
                                      language=dace.dtypes.Language.CPP)

        axpy_state.add_memlet_path(x_in,
                                   task,
                                   dst_conn='x',
                                   memlet=Memlet.simple(x_in.data,
                                                        "0:{}".format(n)))

        axpy_state.add_memlet_path(task,
                                   y_out,
                                   src_conn='y',
                                   memlet=Memlet.simple(y_out.data,
                                                        "0:{}".format(n)))

        return axpy_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        return ExpandAxpyMKL.make_sdfg(node.dtype, node.n, node.a)


@dace.library.expansion
class ExpandAxpyOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        return ExpandAxpyMKL.expansion(node, state, sdfg)


class ExpandAxpyCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def make_sdfg(dtype, n, a, node):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        axpy_sdfg = dace.SDFG('axpy_graph')
        axpy_state = axpy_sdfg.add_state()

        axpy_sdfg.add_symbol(a.name, dtype)

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        axpy_sdfg.add_array('_x', shape=[n], dtype=dtype)
        axpy_sdfg.add_array('_y', shape=[n], dtype=dtype)
        axpy_sdfg.add_array('_res', shape=[n], dtype=dtype)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        x_in = axpy_state.add_read('_x')
        res_out = axpy_state.add_write('_res')

        if dtype == dace.float32:
            func = "Saxpy"
        elif dtype == dace.float64:
            func = "Daxpy"
        else:
            raise ValueError("Unsupported type for BLAS axpy: " + str(dtype))

        code = (
            "const auto __dace_cuda_device = 0;\n" +
            "auto &__dace_cublas_handle = dace::blas::CublasHandle::Get(__dace_cuda_device);\n"
            "cublasSetStream(__dace_cublas_handle, dace::cuda::__streams[0]);\n"
            "cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_HOST);\n"
            "{dtype} alpha = {a};\n"
            "cublas{func}(__dace_cublas_handle, {n}, &alpha, __xDev.ptr<1>(), 1, "
            "__res_axpyDev.ptr<1>(), 1);".format(
                func=func, n=n, a=a, dtype=dtype))

        task = axpy_state.add_tasklet('axpy_blas_task', ['xDev'],
                                      ['res_axpyDev'],
                                      code,
                                      language=dace.dtypes.Language.CPP)

        axpy_state.add_memlet_path(x_in,
                                   task,
                                   dst_conn='xDev',
                                   memlet=Memlet.simple(x_in.data,
                                                        "0:{}".format(n)))

        axpy_state.add_memlet_path(task,
                                   res_out,
                                   src_conn='res_axpyDev',
                                   memlet=Memlet.simple(res_out.data,
                                                        "0:{}".format(n)))

        return axpy_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        return ExpandAxpyCuBLAS.make_sdfg(node.dtype, node.n, node.a, node)


@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandAxpyVectorized,
        "fpga_stream": ExpandAxpyFPGAStreaming,
        "MKL": ExpandAxpyMKL,
        "OpenBLAS": ExpandAxpyOpenBLAS,
        "cuBLAS": ExpandAxpyCuBLAS
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

        # TODO: check veclen with new native vector types
        # veclen = in_memlets[0].veclen

        if len(size) != 1:
            raise ValueError("axpy only supported on 1-dimensional arrays")

        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to axpy must have equal size")

        if size != out_memlet.subset.size():
            raise ValueError("Output of axpy must have same size as input")

        # TODO: check veclen with new native vector types
        # if veclen != in_memlets[1].veclen or veclen != out_memlet.veclen:
        #     raise ValueError(
        #         "Vector lengths of inputs/outputs to axpy must be identical")

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
