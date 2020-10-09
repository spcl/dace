# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from .. import environments


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


@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_result"},
                         **kwargs)
        self.dtype = dtype

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
