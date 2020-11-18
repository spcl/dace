# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from .. import environments
from dace import dtypes

from dace.libraries.blas.utility.initialization import fpga_init_array
from dace.libraries.blas.utility.reductions import fpga_binary_compute_partial_reduction, fpga_linear_result_reduction
from dace.libraries.blas.utility.memory_operations import fpga_map_singleton_to_stream


@dace.library.expansion
class ExpandDotPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_x, outer_array_x, shape_x, _), (edge_y, outer_array_y, shape_y,
                                               _),
         (_, outer_array_result, shape_result,
          _)) = _get_matmul_operands(node,
                                     parent_state,
                                     parent_sdfg,
                                     name_lhs="_x",
                                     name_rhs="_y",
                                     name_out="_result")

        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type
        dtype_result = outer_array_result.dtype.type

        if shape_x != shape_y or shape_result != [1]:
            raise SyntaxError("Invalid shapes to dot product.")

        N = shape_x[0]

        if outer_array_x.storage != outer_array_y.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_x.storage

        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, storage=storage)
        _, array_result = sdfg.add_array("_result", [1],
                                         dtype_result,
                                         storage=storage)

        mul_program = "__out = __x * __y"

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        mul_out, mul_out_array = "_result", array_result
        output_nodes = None

        # Initialization map
        init_write = init_state.add_write("_result")
        init_tasklet = init_state.add_tasklet("dot_init", {}, {"_out"},
                                              "_out = 0",
                                              location=node.location)
        init_state.add_memlet_path(init_tasklet,
                                   init_write,
                                   src_conn="_out",
                                   memlet=dace.Memlet.simple(init_write.data,
                                                             "0",
                                                             num_accesses=1))

        # Multiplication map
        state.add_mapped_tasklet(
            "_DOT_", {"__i": "0:{}".format(N)}, {
                "__x": dace.Memlet.simple("_x", "__i"),
                "__y": dace.Memlet.simple("_y", "__i")
            },
            mul_program, {
                "__out":
                dace.Memlet.simple(mul_out, "0", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandDotPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "sdot"
        elif dtype == dace.float64:
            func = "ddot"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        code = "_result = cblas_{}(n, _x, 1, _y, 1);".format(func)
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandDotMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        return ExpandDotOpenBLAS.expansion(node, state, sdfg)


@dace.library.expansion
class ExpandDotCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
        if dtype == dace.float32:
            func = "Sdot"
        elif dtype == dace.float64:
            func = "Ddot"
        else:
            raise ValueError("Unsupported type for cuBLAS dot product: " +
                             str(dtype))

        code = (environments.cublas.cuBLAS.handle_setup_code(node) +
                "cublas{func}(__dace_cublas_handle, n, ___x.ptr<1>(), 1, "
                "___y.ptr<1>(), 1, ___result.ptr<1>());".format(func=func))

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet




@dace.library.expansion
class ExpandDOTFPGAStreamingLinearReduction(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, partial_width, veclen, n, buffer_size_x, buffer_size_y):

        # --------------------------
        # Setup
        # --------------------------
        vec_type = dace.vector(dtype, veclen)

        dot_sdfg = dace.SDFG('dot_linearReduction')

        init_state = dot_sdfg.add_state('init_state')
        compute_state = dot_sdfg.add_state('compute_state')
        red_state = dot_sdfg.add_state('reduction_state')
        final_state = dot_sdfg.add_state('final_state')

        # --------------------------
        # Memory
        # --------------------------
        dot_sdfg.add_array(
            'red_buf',
            shape=[(max(partial_width, 2))],
            dtype=dtype,
            transient=True,
            storage=dtypes.StorageType.FPGA_Local if partial_width > 8 else dtypes.StorageType.FPGA_Registers
        )

        dot_sdfg.add_array(
            'res_buf',
            shape=[1],
            dtype=dtype,
            transient=True,
            storage=dtypes.StorageType.FPGA_Registers
        )

        # --------------------------
        # Init State
        # --------------------------
        fpga_init_array(init_state, 'red_buf', partial_width, 0)
        fpga_init_array(init_state, 'res_buf', 1, 0)

        # --------------------------
        # Compute State
        # --------------------------
        fpga_binary_compute_partial_reduction(
            dot_sdfg,
            compute_state,
            '_x',
            '_y',
            'red_buf',
            dtype,
            n,
            veclen,
            partial_width,
            'outCon = inCon1 * inCon2',
            vec_type=vec_type
        )

        # --------------------------
        # Reduction State
        # --------------------------
        fpga_linear_result_reduction(
            red_state,
            'red_buf',
            'res_buf',
            dtype,
            partial_width,
            toMem=True
        )

        fpga_map_singleton_to_stream(
            final_state,
            'res_buf',
            '_result',
            dtype
        )

        # --------------------------
        # Connect States
        # --------------------------
        dot_sdfg.add_edge(init_state, compute_state, dace.InterstateEdge())
        dot_sdfg.add_edge(compute_state, red_state, dace.InterstateEdge())
        dot_sdfg.add_edge(red_state, final_state, dace.InterstateEdge())


        dot_sdfg.fill_scope_connectors()

        return  dot_sdfg


    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")

        for e in state.in_edges(node):
            if e.dst_conn == "_x":
                buffer_size_x = sdfg.arrays[e.data.data].buffer_size
            elif e.dst_conn == "_y":
                buffer_size_y = sdfg.arrays[e.data.data].buffer_size

        return ExpandDOTFPGAStreamingLinearReduction.make_sdfg(
            node.dtype,
            node.partial_width,
            int(node.veclen),
            node.n,
            buffer_size_x,
            buffer_size_y
        )




@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
        "fpga_stream": ExpandDOTFPGAStreamingLinearReduction
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    partial_width = dace.properties.SymbolicProperty(allow_none=False, default=2)
    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))

    def __init__(self,
                name,
                dtype=None,
                partial_width=2,
                veclen=1,
                n=None,
                *args,
                **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_result"},
                         **kwargs)
        self.dtype = dtype
        self.veclen = veclen
        self.partial_width = partial_width
        self.n = n or dace.symbolic.symbol("n")


    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to dot product")
        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from dot product")
        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()
        if len(size) != 1:
            raise ValueError(
                "dot product only supported on 1-dimensional arrays")
        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to dot product must have equal size")
        if out_memlet.subset.num_elements() != 1:
            raise ValueError("Output of dot product must be a single element")
        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None
                or out_memlet.wcr is not None):
            raise ValueError("WCR on dot product memlets not supported")


    def compare(self, other):

        if (self.dtype == other.dtype and self.vecWidth == other.vecWidth
            and self.implementation == other.implementation):

            return True
        else:
            return False


    def streamProductionLatency(self):

        return self.n

    def streamConsumptionLatency(self):

        return {
            "_x": 0,
            "_y": 0
        }



    def getStreamReader(self):
        
        return {
            "_x" : streamReadVector(
                '-',
                self.n,
                self.dtype,
                vecWidth=int(self.vecWidth)
            ),
            "_y" : streamReadVector(
                '-',
                self.n,
                self.dtype,
                vecWidth=int(self.vecWidth)
            )
        }

    def getStreamWriter(self):
        
        return {
            "_res" : streamWriteVector(
                '-',
                1,
                self.dtype
            )
        }
