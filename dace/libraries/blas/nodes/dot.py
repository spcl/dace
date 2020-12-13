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
    def expansion(node, state, sdfg, **kwargs):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandDotPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg, **kwargs):
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
    def expansion(node, state, sdfg, **kwargs):
        return ExpandDotOpenBLAS.expansion(node, state, sdfg)


@dace.library.expansion
class ExpandDotCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg, **kwargs):
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
class ExpandDotFPGA(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, partial_width, n, desc_x, desc_y, desc_res):

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
                                     memlet=dace.Memlet(f"{x_read.data}[{index_x}]",
                                                        other_subset="0",
                                                        dynamic=False))
        stream_state.add_memlet_path(y_read,
                                     entry,
                                     input_y_access,
                                     memlet=dace.Memlet(f"{y_read.data}[{index_y}]",
                                                        other_subset="0",
                                                        dynamic=False))

        tasklet = stream_state.add_tasklet("multiply", {"__x", "__y"},
                                           {f"_product": vtype}, f"_product = __x * __y")

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

        stream_state.add_memlet_path(tasklet,
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

        partial_sum_tasklet = stream_state.add_tasklet("partial_sum", {"result_in", "buffer_in"}, {"buffer_out"}, """\
prev = buffer_in if i > 0 else 0
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

        reduce_tasklet = stream_state.add_tasklet("reduce", {"reduce_in", "result_in"},
                                                  {"reduce_out"}, """\
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

    @staticmethod
    def expansion(node, state, sdfg, n=symbolic.symbol('n'), partial_width=8, **kwargs):
        """
        Expand Dot library node for FPGA with streams as inputs/outputs.
        :param n: Total size of buffer (can be symbolic).
        :param partial_width: Width of the inner reduction buffer.
        :param vec_width: Number of elements in vector type to use.
        """
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")

        for e in state.in_edges(node):
            if e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        return ExpandDotFPGA.make_sdfg(node.dtype, partial_width, n, desc_x,
                                       desc_y, desc_res)

@dace.library.expansion
class ExpandDotFPGAAccumulate(ExpandTransformation):
    """
    Version of DOT that assumes that native accumulation of the data type is
    (e.g., 32-bit floating point on Stratix 10).
    """

    environments = []

    @staticmethod
    def make_sdfg(dtype, partial_width, n, desc_x, desc_y, desc_res):

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

        entry, exit = state.add_map(
            "stream", {"i": f"0:{n}/{veclen}"},
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
                                           {f"_product": vtype}, f"_product = __x * __y")

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

    @staticmethod
    def expansion(node, state, sdfg, n=symbolic.symbol('n'), partial_width=8, **kwargs):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")

        for e in state.in_edges(node):
            if e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        return ExpandDotFPGAAccumulate.make_sdfg(node.dtype, partial_width, n,
                                                 desc_x, desc_y, desc_res)


@dace.library.expansion
class ExpandDOTIntelFPGAVectorized(ExpandTransformation):

    # Expansion targeting Intel FPGA
    environments = []

    @staticmethod
    def make_sdfg(dtype, vec_width, node, parent_state, parent_sdfg):

        # FPGA Nested SDFG:
        # - The dot product is represented by two nested maps: the innermost is a fully unrolled map,
        #       the outermost is obtained by strip mining the original loop (over n) to expose unrolling
        #       opportunity
        # - since we want to produce perfectly nested loop, the body of the map will be a nested SDFG
        #   Inside this, the computation is performed and, if we are on the last iteration of the outermost map,
        #   we compute the final result and we write it into memory
        # - Note: this expansion takes advantage of Intel single clock cycle accumulation
        # - TODO: deal with double precision

        #get input size
        n = parent_state.in_edges(node)[0].data.subset.size()[0]

        parent_sdfg = dace.SDFG('dot_graph')
        dot_state = parent_sdfg.add_state("dot_state")

        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------

        parent_sdfg.add_array('_x',
                              shape=[n],
                              dtype=dtype,
                              storage=dace.dtypes.StorageType.FPGA_Global)
        parent_sdfg.add_array('_y',
                              shape=[n],
                              dtype=dtype,
                              storage=dace.dtypes.StorageType.FPGA_Global)
        parent_sdfg.add_array('_result',
                              shape=[1],
                              dtype=dtype,
                              storage=dace.dtypes.StorageType.FPGA_Global)

        parent_sdfg.add_array('_accum',
                              dtype=dtype,
                              shape=[1],
                              transient=True,
                              storage=dace.dtypes.StorageType.FPGA_Registers)

        #--------------------
        # Create the nested Map body
        #--------------------

        nested_dot = dace.SDFG("dot_compute")
        nested_dot.add_symbol("i", dace.int32)
        nested_dot.add_symbol("n", dace.int32)

        nested_dot.add_array('nested_x',
                             shape=[vec_width],
                             dtype=dtype,
                             storage=dace.dtypes.StorageType.FPGA_Global)
        nested_dot.add_array('nested_y',
                             shape=[vec_width],
                             dtype=dtype,
                             storage=dace.dtypes.StorageType.FPGA_Global)
        nested_dot.add_array('nested_res',
                             shape=[1],
                             dtype=dtype,
                             storage=dace.dtypes.StorageType.FPGA_Global)

        nested_dot.add_array('nested_accum_in',
                             dtype=dtype,
                             shape=[1],
                             storage=dace.dtypes.StorageType.FPGA_Registers)
        nested_dot.add_array('nested_accum_out',
                             dtype=dtype,
                             shape=[1],
                             storage=dace.dtypes.StorageType.FPGA_Registers)

        dot_product = nested_dot.add_state("product")

        # Fully unrolled map
        dot_product_map_entry, dot_product_map_exit = dot_product.add_map(
            'product',
            dict(j='0:{}'.format(vec_width)),
            schedule=dace.dtypes.ScheduleType.FPGA_Device,
            unroll=True)

        dot_tasklet = dot_product.add_tasklet(
            'dot_task', ['x_con', 'y_con', 'red_con_in'], ['red_con_out'],
            'red_con_out = red_con_in + x_con * y_con')

        nested_x = dot_product.add_read("nested_x")
        nested_y = dot_product.add_read("nested_y")
        nested_accum_in = dot_product.add_read("nested_accum_in")
        nested_accum_out = dot_product.add_write("nested_accum_out")
        dot_product.add_memlet_path(nested_x,
                                    dot_product_map_entry,
                                    dot_tasklet,
                                    dst_conn='x_con',
                                    memlet=dace.Memlet.simple(
                                        nested_x.data, 'j'))
        dot_product.add_memlet_path(nested_y,
                                    dot_product_map_entry,
                                    dot_tasklet,
                                    dst_conn='y_con',
                                    memlet=dace.Memlet.simple(
                                        nested_y.data, 'j'))
        dot_product.add_memlet_path(nested_accum_in,
                                    dot_product_map_entry,
                                    dot_tasklet,
                                    dst_conn='red_con_in',
                                    memlet=dace.Memlet.simple(
                                        nested_accum_in.data, '0'))
        dot_product.add_memlet_path(dot_tasklet,
                                    dot_product_map_exit,
                                    nested_accum_out,
                                    src_conn='red_con_out',
                                    memlet=dace.Memlet.simple(
                                        nested_accum_out.data, '0'))

        # copy the result out
        dot_write_result = nested_dot.add_state("dot_write_result")
        nested_res = dot_write_result.add_read("nested_accum_out")
        res_out = dot_write_result.add_write('nested_res')

        write_tasklet = dot_write_result.add_tasklet('mapToStream_task',
                                                     ['inCon'], ['outCon'],
                                                     'outCon = inCon')

        dot_write_result.add_memlet_path(nested_res,
                                         write_tasklet,
                                         dst_conn='inCon',
                                         memlet=dace.Memlet.simple(
                                             nested_res.data, '0'))

        dot_write_result.add_memlet_path(write_tasklet,
                                         res_out,
                                         src_conn='outCon',
                                         memlet=dace.Memlet.simple(
                                             res_out.data, '0'))

        # Add interstate edges: copies out only if we are at the last iteration of the outermost map
        if_state = nested_dot.add_state_after(dot_product, "if_state")
        empty_state = nested_dot.add_state("empty_state")
        else_state = nested_dot.add_state("else_state")
        nested_dot.add_edge(
            if_state, dot_write_result,
            dace.sdfg.sdfg.InterstateEdge(
                condition=dace.properties.CodeProperty.from_string(
                    "i == {}/{} - 1".format(n, vec_width),
                    language=dace.dtypes.Language.Python)))
        nested_dot.add_edge(
            if_state, else_state,
            dace.sdfg.sdfg.InterstateEdge(
                condition=dace.properties.CodeProperty.from_string(
                    "i != {}/{} - 1".format(n, vec_width),
                    language=dace.dtypes.Language.Python)))
        nested_dot.add_edge(dot_write_result, empty_state,
                            dace.sdfg.sdfg.InterstateEdge())
        nested_dot.add_edge(else_state, empty_state,
                            dace.sdfg.sdfg.InterstateEdge())

        # --------------------
        # create the outermost map, nest the body
        # ---------------------

        accum_init = dot_state.add_access("_accum")
        init_tasklet = dot_state.add_tasklet('init_task', [], ['outCon'],
                                             'outCon = 0;',
                                             language=dace.dtypes.Language.CPP)

        dot_state.add_memlet_path(init_tasklet,
                                  accum_init,
                                  src_conn='outCon',
                                  memlet=dace.Memlet.simple(
                                      accum_init.data, '0'))

        dotMap_entry, dotMap_exit = dot_state.add_map(
            'dot_map',
            dict(i='0:{0}/{1}'.format(n, vec_width)),
            schedule=dace.dtypes.ScheduleType.FPGA_Device)

        # Nest the other SDFG
        nested_sdfg = dot_state.add_nested_sdfg(
            nested_dot,
            parent_sdfg, {"nested_x", "nested_y", "nested_accum_in"},
            {"nested_res", "nested_accum_out"},
            symbol_mapping={
                "i": "i",
                "n": n
            })

        x_read = dot_state.add_read("_x")
        y_read = dot_state.add_read("_y")

        accum_write = dot_state.add_write("_accum")
        res_write = dot_state.add_write("_result")

        dot_state.add_memlet_path(x_read,
                                  dotMap_entry,
                                  nested_sdfg,
                                  dst_conn="nested_x",
                                  memlet=dace.Memlet.simple(
                                      x_read,
                                      "i*{v}:i*{v}+{v}".format(v=vec_width),
                                      num_accesses=vec_width))
        dot_state.add_memlet_path(y_read,
                                  dotMap_entry,
                                  nested_sdfg,
                                  dst_conn="nested_y",
                                  memlet=dace.Memlet.simple(
                                      y_read,
                                      "i*{v}:i*{v}+{v}".format(v=vec_width),
                                      num_accesses=vec_width))
        dot_state.add_memlet_path(accum_init,
                                  dotMap_entry,
                                  nested_sdfg,
                                  dst_conn="nested_accum_in",
                                  memlet=dace.Memlet.simple(accum_init, "0"))
        dot_state.add_memlet_path(nested_sdfg,
                                  dotMap_exit,
                                  accum_write,
                                  src_conn='nested_accum_out',
                                  memlet=dace.Memlet.simple(
                                      accum_write.data, "0"))
        dot_state.add_memlet_path(nested_sdfg,
                                  dotMap_exit,
                                  res_write,
                                  src_conn='nested_res',
                                  memlet=dace.Memlet.simple(res_write.data,
                                                            "0",
                                                            dynamic=True))
        parent_sdfg.validate()
        return parent_sdfg

    @staticmethod
    def expansion(node, state, sdfg, vec_width=1, **kwargs):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        node_sdfg = ExpandDOTIntelFPGAVectorized.make_sdfg(
            node.dtype, int(vec_width), node, state, sdfg)

        # Modify internal schedules according to node schedule
        if node.schedule != dace.ScheduleType.Default:
            for nstate in node_sdfg.nodes():
                topnodes = nstate.scope_children()[None]
                for topnode in topnodes:
                    if isinstance(
                            topnode,
                        (dace.nodes.EntryNode, dace.nodes.LibraryNode)):
                        topnode.schedule = node.schedule
        # nest and map symbol
        symbol_mapping = {}
        expansion = state.add_nested_sdfg(node_sdfg,
                                          sdfg,
                                          node.in_connectors,
                                          node.out_connectors,
                                          name=node.name,
                                          debuginfo=node.debuginfo)
        return expansion


@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
        "FPGA": ExpandDotFPGA,
        "FPGAAccumulate": ExpandDotFPGAAccumulate,
        "IntelFPGA": ExpandDOTIntelFPGAVectorized
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


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.dot')
@oprepo.replaces('dace.libraries.blas.Dot')
def ger_libnode(sdfg: SDFG, state: SDFGState, x, y, result):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Dot('dot', dtype=sdfg.arrays[x].dtype)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))

    return []
