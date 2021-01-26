# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandDotPure(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):

        (outer_array_x, outer_array_y,
         outer_array_res) = node.validate(parent_sdfg, parent_state)

        shape_x = outer_array_x.shape
        shape_y = outer_array_y.shape
        shape_result = outer_array_res.shape

        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type
        dtype_result = outer_array_res.dtype.type
        sdfg = dace.SDFG(node.label + "_sdfg")

        if shape_x != shape_y or tuple(shape_result) != (1, ):
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


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg, n=None, **kwargs):
        desc_x, _, _ = node.validate(sdfg, state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen
        if dtype == dace.float32:
            func = "sdot"
        elif dtype == dace.float64:
            func = "ddot"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        n = n or node.n
        if veclen != 1:
            n /= veclen
        code = f"_result = cblas_{func}({n}, _x, 1, _y, 1);"
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
    def expansion(*args, **kwargs):
        return ExpandDotOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandDotCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg, n=None, **kwargs):

        desc_x, _, _ = node.validate(sdfg, state)

        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen
        n = n or node.n
        if veclen != 1:
            n /= veclen

        if dtype == dace.float32:
            func = "Sdot"
        elif dtype == dace.float64:
            func = "Ddot"
        else:
            raise ValueError("Unsupported type for cuBLAS dot product: " +
                             str(dtype))

        code = (environments.cublas.cuBLAS.handle_setup_code(node) +
                "cublas{func}(__dace_cublas_handle, {n}, ___x.ptr<1>(), 1, "
                "___y.ptr<1>(), 1, ___result.ptr<1>());".format(func=func, n=n))

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandDotFPGAPartialSums(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node,
                  parent_state,
                  parent_sdfg,
                  n=None,
                  partial_width=8,
                  **kwargs):
        """
        Expand Dot library node for FPGA with streams as inputs/outputs.
        :param n: Total size of buffer (can be symbolic).
        :param partial_width: Width of the inner reduction buffer.
        """
        desc_x, desc_y, desc_res = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG("dot")

        stream_state = sdfg.add_state("stream")

        dtype = desc_x.dtype.base_type
        veclen = desc_x.veclen
        vtype = dtypes.vector(dtype, veclen)
        n = n or node.n

        desc_x = desc_x.clone()
        desc_x.transient = False
        desc_y = desc_y.clone()
        desc_y.transient = False
        desc_res = desc_res.clone()
        desc_res.transient = False
        sdfg.add_datadesc("_x", desc_x)
        sdfg.add_datadesc("_y", desc_y)
        sdfg.add_datadesc("_result", desc_res)

        x_read = stream_state.add_read("_x")
        y_read = stream_state.add_read("_y")
        res_write = stream_state.add_write("_result")

        input_x_name = "input_x"
        sdfg.add_array(input_x_name, (1, ),
                       vtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        input_x_access = stream_state.add_access(input_x_name)

        input_y_name = "input_y"
        sdfg.add_array(input_y_name, (1, ),
                       vtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        input_y_access = stream_state.add_access(input_y_name)

        entry, exit = stream_state.add_map(
            "stream", {"i": f"0:{n}/{veclen}"},
            schedule=dtypes.ScheduleType.FPGA_Device)

        index_x = "0" if isinstance(desc_x, dt.Stream) else "i"
        index_y = "0" if isinstance(desc_y, dt.Stream) else "i"

        stream_state.add_memlet_path(x_read,
                                     entry,
                                     input_x_access,
                                     memlet=dace.Memlet(
                                         f"{x_read.data}[{index_x}]",
                                         other_subset="0",
                                         dynamic=False))
        stream_state.add_memlet_path(y_read,
                                     entry,
                                     input_y_access,
                                     memlet=dace.Memlet(
                                         f"{y_read.data}[{index_y}]",
                                         other_subset="0",
                                         dynamic=False))

        tasklet = stream_state.add_tasklet("multiply", {"__x", "__y"},
                                           {f"_product": vtype},
                                           f"_product = __x * __y")

        stream_state.add_memlet_path(input_x_access,
                                     tasklet,
                                     dst_conn="__x",
                                     memlet=dace.Memlet(f"{input_x_name}[0]"))
        stream_state.add_memlet_path(input_y_access,
                                     tasklet,
                                     dst_conn="__y",
                                     memlet=dace.Memlet(f"{input_y_name}[0]"))

        product_name = "product"
        sdfg.add_array(product_name, (veclen, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        product_access = stream_state.add_access(product_name)

        stream_state.add_memlet_path(
            tasklet,
            product_access,
            src_conn="_product",
            memlet=dace.Memlet(f"{product_name}[0:{veclen}]"))

        collapse_name = "reduce_vector"
        sdfg.add_array(collapse_name, (1, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        collapse_read = stream_state.add_read(collapse_name)
        collapse_access = stream_state.add_access(collapse_name)

        unroll_entry, unroll_exit = stream_state.add_map(
            "unroll", {"j": f"0:{veclen}"},
            unroll=True,
            schedule=dtypes.ScheduleType.FPGA_Device)

        collapse_tasklet = stream_state.add_tasklet(
            "reduce_vector", {"val_in", "reduce_in"}, {"reduce_out"}, """\
prev = reduce_in if j > 0 else 0
reduce_out = prev + val_in""")

        stream_state.add_memlet_path(collapse_read,
                                     unroll_entry,
                                     collapse_tasklet,
                                     dst_conn="reduce_in",
                                     memlet=dace.Memlet(f"{collapse_name}[0]"))
        stream_state.add_memlet_path(entry, collapse_read, memlet=dace.Memlet())
        stream_state.add_memlet_path(collapse_tasklet,
                                     unroll_exit,
                                     collapse_access,
                                     src_conn="reduce_out",
                                     memlet=dace.Memlet(f"{collapse_name}[0]"))
        stream_state.add_memlet_path(product_access,
                                     unroll_entry,
                                     collapse_tasklet,
                                     dst_conn="val_in",
                                     memlet=dace.Memlet(f"{product_name}[j]"))

        buffer_name = "partial_sums"
        sdfg.add_array(buffer_name, (partial_width, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        buffer_read = stream_state.add_read(buffer_name)
        buffer_write = stream_state.add_write(buffer_name)

        partial_sum_tasklet = stream_state.add_tasklet(
            "partial_sum", {"result_in", "buffer_in"}, {"buffer_out"}, f"""\
prev = buffer_in if i >= {partial_width} else 0
buffer_out = prev + result_in""")

        stream_state.add_memlet_path(
            collapse_access,
            partial_sum_tasklet,
            dst_conn="result_in",
            memlet=dace.Memlet(f"{collapse_access.data}[0]"))
        stream_state.add_memlet_path(
            buffer_read,
            entry,
            partial_sum_tasklet,
            dst_conn=f"buffer_in",
            memlet=dace.Memlet(f"{buffer_name}[i%{partial_width}]"))
        stream_state.add_memlet_path(
            partial_sum_tasklet,
            exit,
            buffer_write,
            src_conn=f"buffer_out",
            memlet=dace.Memlet(f"{buffer_name}[i%{partial_width}]"))

        reduce_entry, reduce_exit = stream_state.add_map(
            "reduce", {"i": f"0:{partial_width}"},
            schedule=dtypes.ScheduleType.FPGA_Device,
            unroll=True)

        reduce_tasklet = stream_state.add_tasklet(
            "reduce", {"reduce_in", "result_in"}, {"reduce_out"}, """\
prev = reduce_in if i > 0 else 0
reduce_out = prev + result_in""")

        stream_state.add_memlet_path(buffer_write,
                                     reduce_entry,
                                     reduce_tasklet,
                                     dst_conn="result_in",
                                     memlet=dace.Memlet(f"{buffer_name}[i]"))

        reduce_name = "reduce"
        sdfg.add_array(reduce_name, (1, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        reduce_read = stream_state.add_read(reduce_name)
        reduce_access = stream_state.add_access(reduce_name)

        stream_state.add_memlet_path(reduce_read,
                                     reduce_entry,
                                     reduce_tasklet,
                                     dst_conn="reduce_in",
                                     memlet=dace.Memlet(f"{reduce_name}[0]"))
        stream_state.add_memlet_path(reduce_tasklet,
                                     reduce_exit,
                                     reduce_access,
                                     src_conn="reduce_out",
                                     memlet=dace.Memlet(f"{reduce_name}[0]"))

        stream_state.add_memlet_path(reduce_access,
                                     res_write,
                                     memlet=dace.Memlet(f"{reduce_name}[0]",
                                                        other_subset="0"))

        return sdfg


@dace.library.expansion
class ExpandDotFPGAAccumulate(ExpandTransformation):
    """
    Version of DOT that assumes that native accumulation of the data type is
    (e.g., 32-bit floating point on Stratix 10).
    """

    environments = []

    @staticmethod
    def expansion(node,
                  parent_state,
                  parent_sdfg,
                  n=None,
                  partial_width=8,
                  **kwargs):

        desc_x, desc_y, desc_res = node.validate(parent_sdfg, parent_state)

        n = n or node.n

        sdfg = dace.SDFG("dot")

        state = sdfg.add_state("dot")

        dtype = desc_x.dtype.base_type
        veclen = desc_x.veclen
        vtype = dtypes.vector(dtype, veclen)

        desc_x = desc_x.clone()
        desc_x.transient = False
        desc_y = desc_y.clone()
        desc_y.transient = False
        desc_res = desc_res.clone()
        desc_res.transient = False
        sdfg.add_datadesc("_x", desc_x)
        sdfg.add_datadesc("_y", desc_y)
        sdfg.add_datadesc("_result", desc_res)

        x_read = state.add_read("_x")
        y_read = state.add_read("_y")
        res_write = state.add_write("_result")

        input_x_name = "input_x"
        sdfg.add_array(input_x_name, (1, ),
                       vtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        input_x_access = state.add_access(input_x_name)

        input_y_name = "input_y"
        sdfg.add_array(input_y_name, (1, ),
                       vtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        input_y_access = state.add_access(input_y_name)

        entry, exit = state.add_map("stream", {"i": f"0:{n}/{veclen}"},
                                    schedule=dtypes.ScheduleType.FPGA_Device)

        index_x = "0" if isinstance(desc_x, dt.Stream) else "i"
        index_y = "0" if isinstance(desc_y, dt.Stream) else "i"

        state.add_memlet_path(x_read,
                              entry,
                              input_x_access,
                              memlet=dace.Memlet(f"{x_read.data}[{index_x}]",
                                                 other_subset="0",
                                                 dynamic=False))
        state.add_memlet_path(y_read,
                              entry,
                              input_y_access,
                              memlet=dace.Memlet(f"{y_read.data}[{index_y}]",
                                                 other_subset="0",
                                                 dynamic=False))

        tasklet = state.add_tasklet("multiply", {"__x", "__y"},
                                    {f"_product": vtype},
                                    f"_product = __x * __y")

        state.add_memlet_path(input_x_access,
                              tasklet,
                              dst_conn="__x",
                              memlet=dace.Memlet(f"{input_x_name}[0]"))
        state.add_memlet_path(input_y_access,
                              tasklet,
                              dst_conn="__y",
                              memlet=dace.Memlet(f"{input_y_name}[0]"))

        product_name = "product"
        sdfg.add_array(product_name, (veclen, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        product_access = state.add_access(product_name)

        state.add_memlet_path(tasklet,
                              product_access,
                              src_conn="_product",
                              memlet=dace.Memlet(f"{product_name}[0:{veclen}]"))

        collapse_name = "reduce_vector"
        sdfg.add_array(collapse_name, (1, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        collapse_read = state.add_read(collapse_name)
        collapse_access = state.add_access(collapse_name)

        unroll_entry, unroll_exit = state.add_map(
            "unroll", {"j": f"0:{veclen}"},
            unroll=True,
            schedule=dtypes.ScheduleType.FPGA_Device)

        collapse_tasklet = state.add_tasklet(
            "reduce_vector", {"val_in", "reduce_in"}, {"reduce_out"}, """\
prev = reduce_in if j > 0 else 0
reduce_out = prev + val_in""")

        state.add_memlet_path(collapse_read,
                              unroll_entry,
                              collapse_tasklet,
                              dst_conn="reduce_in",
                              memlet=dace.Memlet(f"{collapse_name}[0]"))
        state.add_memlet_path(entry, collapse_read, memlet=dace.Memlet())
        state.add_memlet_path(collapse_tasklet,
                              unroll_exit,
                              collapse_access,
                              src_conn="reduce_out",
                              memlet=dace.Memlet(f"{collapse_name}[0]"))
        state.add_memlet_path(product_access,
                              unroll_entry,
                              collapse_tasklet,
                              dst_conn="val_in",
                              memlet=dace.Memlet(f"{product_name}[j]"))

        buffer_name = "reduce_buffer"
        sdfg.add_array(buffer_name, (1, ),
                       dtype,
                       transient=True,
                       storage=dtypes.StorageType.FPGA_Local)
        buffer_read = state.add_read(buffer_name)
        buffer_write = state.add_access(buffer_name)

        zero_tasklet = state.add_tasklet("zero", {}, {"buffer"}, "buffer = 0")
        state.add_memlet_path(zero_tasklet,
                              buffer_read,
                              src_conn="buffer",
                              memlet=dace.Memlet(f"{buffer_name}[0]"))

        reduce_tasklet = state.add_tasklet(
            "sum", {"buffer_in", "result_in"}, {"buffer_out"}, """\
prev = buffer_in if i > 0 else 0
buffer_out = prev + result_in""")

        state.add_memlet_path(collapse_access,
                              reduce_tasklet,
                              dst_conn="result_in",
                              memlet=dace.Memlet(f"{collapse_access.data}[0]"))
        state.add_memlet_path(buffer_read,
                              entry,
                              reduce_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet(f"{buffer_name}[0]"))
        state.add_memlet_path(reduce_tasklet,
                              exit,
                              buffer_write,
                              src_conn=f"buffer_out",
                              memlet=dace.Memlet(f"{buffer_name}[0]"))

        state.add_memlet_path(buffer_write,
                              res_write,
                              memlet=dace.Memlet(f"{buffer_name}[0]",
                                                 other_subset="0"))

        return sdfg


@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
        "FPGA_PartialSums": ExpandDotFPGAPartialSums,
        "FPGA_Accumulate": ExpandDotFPGAAccumulate,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("n"))

    def __init__(self, name, n=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_result"},
                         **kwargs)
        self.n = n or dace.symbolic.symbol("n")

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, y, res) of the three data descriptors in the
                 parent SDFG.
        """
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
        for e in state.in_edges(self):
            if e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]
        if desc_x.dtype != desc_y.dtype:
            raise TypeError("Data types of input operands must be equal: "
                            f"{desc_x.dtype}, {desc_y.dtype}")
        if desc_x.dtype.base_type != desc_res.dtype.base_type:
            raise TypeError("Data types of input and output must be equal: "
                            f"{desc_x.dtype}, {desc_res.dtype}")
        return desc_x, desc_y, desc_res


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.dot')
@oprepo.replaces('dace.libraries.blas.Dot')
def ger_libnode(sdfg: SDFG, state: SDFGState, x, y, result):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Dot('dot', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))

    return []
