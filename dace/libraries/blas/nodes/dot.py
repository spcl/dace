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

        dot_sdfg.add_scalar(
            'res_buf',
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

        # -----
        # Write to stream State
        # -----
        fpga_map_singleton_to_stream(
            final_state,
            'res_buf',
            '_result',
            dace.vector(dtype, 1)
        )

        # --------------------------
        # Connect States
        # --------------------------
        dot_sdfg.add_edge(init_state, compute_state, dace.InterstateEdge())
        dot_sdfg.add_edge(init_state, red_state, dace.InterstateEdge())
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
                                      "i*{}".format(vec_width),
                                      num_accesses=vec_width))
        dot_state.add_memlet_path(y_read,
                                  dotMap_entry,
                                  nested_sdfg,
                                  dst_conn="nested_y",
                                  memlet=dace.Memlet.simple(
                                      y_read,
                                      "i*{}".format(vec_width),
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
                                  memlet=dace.Memlet.simple(
                                      res_write.data, "0"))
        parent_sdfg.validate()
        return parent_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        node_sdfg = ExpandDOTIntelFPGAVectorized.make_sdfg(
            node.dtype, int(node.vec_width), node, state, sdfg)

        # Modify internal schedules according to node schedule
        if node.schedule != dace.ScheduleType.Default:
            for nstate in node_sdfg.nodes():
                topnodes = nstate.scope_dict(node_to_children=True)[None]
                for topnode in topnodes:
                    if isinstance(
                            topnode,
                        (dace.nodes.EntryNode, dace.nodes.LibraryNode)):
                        topnode.schedule = node.schedule
        # nest and map symbol
        symbol_mapping = {}  #{"n": node.n}
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
        "fpga_stream": ExpandDOTFPGAStreamingLinearReduction,
        "IntelFPGA": ExpandDOTIntelFPGAVectorized
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    vec_width = dace.properties.SymbolicProperty(allow_none=False, default=1)

    partial_width = dace.properties.SymbolicProperty(allow_none=False, default=2)
    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))


    def __init__(self,
                name,
                dtype=None,
                partial_width=2,
                veclen=1,
                n=None,
                vec_width=1,
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
        self.vec_width = vec_width

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
