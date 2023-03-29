# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import warnings
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandDotPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of DOT.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):

        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

        n = n or node.n or sz

        dtype_x = desc_x.dtype.type
        dtype_y = desc_y.dtype.type
        dtype_result = desc_res.dtype.type
        sdfg = dace.SDFG(node.label + "_sdfg")

        if desc_x.dtype.veclen > 1 or desc_y.dtype.veclen > 1:
            raise NotImplementedError("Pure expansion not implemented for vector types.")

        sdfg.add_array("_x", [n], dtype_x, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_y", [n], dtype_y, strides=[stride_y], storage=desc_y.storage)
        sdfg.add_array("_result", [1], dtype_result, storage=desc_res.storage)

        mul_program = "__out = __x * __y"

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        # Initialization map
        init_state.add_mapped_tasklet("_i_dotnit", {"__unused": "0:1"}, {},
                                      "_out = 0", {"_out": dace.Memlet("_result[0]")},
                                      external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("dot", {"__i": f"0:{n}"}, {
            "__x": dace.Memlet("_x[__i]"),
            "__y": dace.Memlet("_y[__i]")
        },
                                 mul_program, {"__out": dace.Memlet(f"_result[0]", wcr="lambda x, y: x + y")},
                                 external_edges=True,
                                 output_nodes=None)

        return sdfg


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        func = func.lower() + 'dot'

        n = n or node.n or sz
        if veclen != 1:
            n /= veclen
        code = f"_result = cblas_{func}({n}, _x, {stride_x}, _y, {stride_y});"
        # The return type is scalar in cblas_?dot signature
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtype},
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
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)
        func = func + 'dot'

        n = n or node.n or sz
        if veclen != 1:
            n /= veclen

        code = environments.cublas.cuBLAS.handle_setup_code(node)
        if node.accumulator_type is None:
            code += f"""cublas{func}(__dace_cublas_handle, {n}, _x, {stride_x}, _y,
                             {stride_y}, _result);"""
        else:
            code += f"""
            cublasDotEx(
                __dace_cublas_handle,
                {n},
                _x,
                {blas_helpers.dtype_to_cudadatatype(dtype)},
                {stride_x},
                _y,
                {blas_helpers.dtype_to_cudadatatype(desc_y.dtype)},
                {stride_y},
                _result,
                {blas_helpers.dtype_to_cudadatatype(desc_res.dtype)},
                {blas_helpers.dtype_to_cudadatatype(node.accumulator_type)});
            """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtypes.pointer(dtype)},
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandDotFpgaPartialSums(ExpandTransformation):
    """
    FPGA-expansion of DOT that does NOT assume that native accumulation of the
    data type is possible (e.g., floating point on Xilinx devices or float64
    on Stratix 10).

    To achieve II=1, accumulation is done into multiple partial sums, which are
    reduced at the end of the computation.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, partial_width=8):
        """
        :param node: The node to expand.
        :param parent_state: The state that the node is in.
        :param parent_sdfg: The SDFG that the node is in.
        :param n: Override the vector dimension. If this is not set, the value
                  specified in the node is used.
        :param partial_width: Width of the inner reduction buffer. Must be
                              larger than the latency of addition on the given
                              data type.
        """
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

        n = n or node.n or sz

        sdfg = dace.SDFG("dot")

        stream_state = sdfg.add_state("stream")

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

        x_read = stream_state.add_read("_x")
        y_read = stream_state.add_read("_y")
        res_write = stream_state.add_write("_result")

        input_x_name = "input_x"
        sdfg.add_array(input_x_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_x_access = stream_state.add_access(input_x_name)

        input_y_name = "input_y"
        sdfg.add_array(input_y_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_y_access = stream_state.add_access(input_y_name)

        entry, exit = stream_state.add_map("stream", {"_i_dot": f"0:{n}/{veclen}"},
                                           schedule=dtypes.ScheduleType.FPGA_Device)

        index_x = "0" if isinstance(desc_x, dt.Stream) else "_i_dot"
        index_y = "0" if isinstance(desc_y, dt.Stream) else "_i_dot"

        stream_state.add_memlet_path(x_read,
                                     entry,
                                     input_x_access,
                                     memlet=dace.Memlet(f"{x_read.data}[{index_x}]", other_subset="0", dynamic=False))
        stream_state.add_memlet_path(y_read,
                                     entry,
                                     input_y_access,
                                     memlet=dace.Memlet(f"{y_read.data}[{index_y}]", other_subset="0", dynamic=False))

        tasklet = stream_state.add_tasklet("multiply", {"__x", "__y"}, {f"_product": vtype}, f"_product = __x * __y")

        stream_state.add_memlet_path(input_x_access, tasklet, dst_conn="__x", memlet=dace.Memlet(f"{input_x_name}[0]"))
        stream_state.add_memlet_path(input_y_access, tasklet, dst_conn="__y", memlet=dace.Memlet(f"{input_y_name}[0]"))

        product_name = "product"
        sdfg.add_array(product_name, (veclen, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        product_access = stream_state.add_access(product_name)

        stream_state.add_memlet_path(tasklet,
                                     product_access,
                                     src_conn="_product",
                                     memlet=dace.Memlet(f"{product_name}[0:{veclen}]"))

        collapse_name = "reduce_vector"
        sdfg.add_array(collapse_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        collapse_read = stream_state.add_read(collapse_name)
        collapse_access = stream_state.add_access(collapse_name)

        unroll_entry, unroll_exit = stream_state.add_map("unroll", {"_j_dot": f"0:{veclen}"},
                                                         unroll=True,
                                                         schedule=dtypes.ScheduleType.FPGA_Device)

        collapse_tasklet = stream_state.add_tasklet(
            "reduce_vector", {"val_in", "reduce_in"}, {"reduce_out"}, """\
prev = reduce_in if _j_dot > 0 else 0
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
                                     memlet=dace.Memlet(f"{product_name}[_j_dot]"))

        buffer_name = "partial_sums"
        sdfg.add_array(buffer_name, (partial_width, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)

        # The partial result buffer must be initialized.
        init_tasklet = stream_state.add_tasklet("init_dummy_ps", {}, {"init_data"}, "init_data = 0")
        init_ps_entry, init_ps_exit = stream_state.add_map("init_unroll", {"_j_dot": f"0:{partial_width}"},
                                                           unroll=True,
                                                           schedule=dtypes.ScheduleType.FPGA_Device)
        buffer_read = stream_state.add_access(buffer_name)
        stream_state.add_memlet_path(init_ps_entry, init_tasklet, memlet=dace.Memlet())
        stream_state.add_memlet_path(init_tasklet,
                                     init_ps_exit,
                                     buffer_read,
                                     src_conn="init_data",
                                     memlet=dace.Memlet(f"{buffer_name}[_j_dot]"))

        buffer_write = stream_state.add_write(buffer_name)

        partial_sum_tasklet = stream_state.add_tasklet(
            "partial_sum", {"result_in", "buffer_in"}, {"buffer_out"}, f"""\
prev = buffer_in if _i_dot >= {partial_width} else 0
buffer_out = prev + result_in""")

        stream_state.add_memlet_path(collapse_access,
                                     partial_sum_tasklet,
                                     dst_conn="result_in",
                                     memlet=dace.Memlet(f"{collapse_access.data}[0]"))
        stream_state.add_memlet_path(buffer_read,
                                     entry,
                                     partial_sum_tasklet,
                                     dst_conn=f"buffer_in",
                                     memlet=dace.Memlet(f"{buffer_name}[_i_dot%{partial_width}]"))
        stream_state.add_memlet_path(partial_sum_tasklet,
                                     exit,
                                     buffer_write,
                                     src_conn=f"buffer_out",
                                     memlet=dace.Memlet(f"{buffer_name}[_i_dot%{partial_width}]"))

        reduce_entry, reduce_exit = stream_state.add_map("reduce", {"_i_dot": f"0:{partial_width}"},
                                                         schedule=dtypes.ScheduleType.FPGA_Device,
                                                         unroll=True)

        reduce_tasklet = stream_state.add_tasklet(
            "reduce", {"reduce_in", "result_in"}, {"reduce_out"}, """\
prev = reduce_in if _i_dot > 0 else 0
reduce_out = prev + result_in""")

        stream_state.add_memlet_path(buffer_write,
                                     reduce_entry,
                                     reduce_tasklet,
                                     dst_conn="result_in",
                                     memlet=dace.Memlet(f"{buffer_name}[_i_dot]"))

        reduce_name = "reduce"
        sdfg.add_array(reduce_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
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
                                     memlet=dace.Memlet(f"{reduce_name}[0]", other_subset="0"))

        return sdfg


@dace.library.expansion
class ExpandDotFpgaAccumulate(ExpandTransformation):
    """
    Version of DOT that assumes that native II=1 accumulation of the data type
    is possible on the target architecture (e.g., 32-bit floating point on
    Stratix 10).
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        """
        :param node: The node to expand.
        :param parent_state: The state that the node is in.
        :param parent_sdfg: The SDFG that the node is in.
        :param n: Override the vector dimension. If this is not set, the value
                  specified in the node is used.
        """

        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

        n = n or node.n or sz

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
        sdfg.add_array(input_x_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_x_access = state.add_access(input_x_name)

        input_y_name = "input_y"
        sdfg.add_array(input_y_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_y_access = state.add_access(input_y_name)

        entry, exit = state.add_map("stream", {"_i_dot": f"0:{n}/{veclen}"}, schedule=dtypes.ScheduleType.FPGA_Device)

        index_x = "0" if isinstance(desc_x, dt.Stream) else "_i_dot"
        index_y = "0" if isinstance(desc_y, dt.Stream) else "_i_dot"

        state.add_memlet_path(x_read,
                              entry,
                              input_x_access,
                              memlet=dace.Memlet(f"{x_read.data}[{index_x}]", other_subset="0", dynamic=False))
        state.add_memlet_path(y_read,
                              entry,
                              input_y_access,
                              memlet=dace.Memlet(f"{y_read.data}[{index_y}]", other_subset="0", dynamic=False))

        tasklet = state.add_tasklet("multiply", {"__x", "__y"}, {f"_product": vtype}, f"_product = __x * __y")

        state.add_memlet_path(input_x_access, tasklet, dst_conn="__x", memlet=dace.Memlet(f"{input_x_name}[0]"))
        state.add_memlet_path(input_y_access, tasklet, dst_conn="__y", memlet=dace.Memlet(f"{input_y_name}[0]"))

        product_name = "product"
        sdfg.add_array(product_name, (veclen, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        product_access = state.add_access(product_name)

        state.add_memlet_path(tasklet,
                              product_access,
                              src_conn="_product",
                              memlet=dace.Memlet(f"{product_name}[0:{veclen}]"))

        collapse_name = "reduce_vector"
        sdfg.add_array(collapse_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        collapse_read = state.add_read(collapse_name)
        collapse_access = state.add_access(collapse_name)

        unroll_entry, unroll_exit = state.add_map("unroll", {"_j_dot": f"0:{veclen}"},
                                                  unroll=True,
                                                  schedule=dtypes.ScheduleType.FPGA_Device)

        collapse_tasklet = state.add_tasklet("reduce_vector", {"val_in", "reduce_in"}, {"reduce_out"}, """\
prev = reduce_in if _j_dot > 0 else 0
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
                              memlet=dace.Memlet(f"{product_name}[_j_dot]"))

        buffer_name = "reduce_buffer"
        sdfg.add_array(buffer_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        buffer_read = state.add_read(buffer_name)
        buffer_write = state.add_access(buffer_name)

        zero_tasklet = state.add_tasklet("zero", {}, {"buffer"}, "buffer = 0")
        state.add_memlet_path(zero_tasklet, buffer_read, src_conn="buffer", memlet=dace.Memlet(f"{buffer_name}[0]"))

        reduce_tasklet = state.add_tasklet("sum", {"buffer_in", "result_in"}, {"buffer_out"}, """\
prev = buffer_in if _i_dot > 0 else 0
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

        state.add_memlet_path(buffer_write, res_write, memlet=dace.Memlet(f"{buffer_name}[0]", other_subset="0"))

        return sdfg


@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
        "FPGA_PartialSums": ExpandDotFpgaPartialSums,
        "FPGA_Accumulate": ExpandDotFpgaAccumulate,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)
    accumulator_type = dace.properties.TypeClassProperty(default=None,
                                                         choices=dtypes.Typeclasses,
                                                         allow_none=True,
                                                         desc="Accumulator or intermediate storage type")

    def __init__(self, name, n=None, accumulator_type=None, **kwargs):
        super().__init__(name, inputs={"_x", "_y"}, outputs={"_result"}, **kwargs)
        self.n = n
        self.accumulator_type = accumulator_type

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, y, res) of the three data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to dot product")
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from dot product")
        out_memlet = out_edges[0].data

        desc_x, desc_y, desc_res = None, None, None
        in_memlets = [None, None]
        for e in state.in_edges(self):
            if e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
                in_memlets[0] = e.data
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
                in_memlets[1] = e.data
        for e in state.out_edges(self):
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_x.dtype != desc_y.dtype:
            raise TypeError(f"Data types of input operands must be equal: {desc_x.dtype}, {desc_y.dtype}")
        if desc_x.dtype.base_type != desc_res.dtype.base_type:
            raise TypeError(f"Data types of input and output must be equal: {desc_x.dtype}, {desc_res.dtype}")

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        squeezed2 = copy.deepcopy(in_memlets[1].subset)
        sqdims1 = squeezed1.squeeze()
        sqdims2 = squeezed2.squeeze()

        if len(squeezed1.size()) != 1 or len(squeezed2.size()) != 1:
            raise ValueError("dot product only supported on 1-dimensional arrays")
        if out_memlet.subset.num_elements() != 1:
            raise ValueError("Output of dot product must be a single element")

        # We are guaranteed that there is only one non-squeezed dimension
        stride_x = desc_x.strides[sqdims1[0]]
        stride_y = desc_y.strides[sqdims2[0]]
        n = squeezed1.num_elements()
        if squeezed1.num_elements() != squeezed2.num_elements():
            raise ValueError('Size mismatch in inputs')

        return (desc_x, stride_x), (desc_y, stride_y), desc_res, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.dot')
@oprepo.replaces('dace.libraries.blas.Dot')
def dot_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, y, result, acctype=None):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Dot('dot', n=sdfg.arrays[x].shape[0], accumulator_type=acctype)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))

    return []
