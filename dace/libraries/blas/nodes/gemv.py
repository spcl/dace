# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.sdfg import SDFG, SDFGState
from dace import memlet as mm
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from dace.frontend.common import op_repository as oprepo
from .. import environments
import numpy as np

from dace import dtypes
from dace.memlet import Memlet
from dace.libraries.blas.utility.initialization import fpga_init_array


@dace.library.expansion
class ExpandGemvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")
        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x,
                                                       shape_x, _),
         _) = _get_matmul_operands(node,
                                   parent_state,
                                   parent_sdfg,
                                   name_lhs="_A",
                                   name_rhs="_x",
                                   name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype_x = outer_array_x.dtype.type
        dtype_y = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_x).type]

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if trans_shape_a[1] != shape_x[0]:
            raise SyntaxError(
                "Matrix-vector product size mismatch: {} vs. {}".format(
                    trans_shape_a[1], shape_x[0]))

        N, M = trans_shape_a[0], trans_shape_a[1]
        shape_y = (N, )

        if outer_array_a.storage != outer_array_x.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_A",
                                    shape_a,
                                    dtype_a,
                                    strides=strides_a,
                                    storage=storage)
        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, storage=storage)

        mul_program = "__out = {} * __A * __x".format(node.alpha)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(
                shape_y, dtype_y, storage=storage)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        # Initialization map
        init_state.add_mapped_tasklet(
            "gemv_init",
            {"_o%d" % i: "0:%s" % symstr(d)
             for i, d in enumerate(shape_y)}, {},
            "out = 0", {
                "out":
                dace.Memlet.simple(
                    mul_out, ",".join(["_o%d" % i
                                       for i in range(len(shape_y))]))
            },
            external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet(
            "_GEMV_", {"__i%d" % i: "0:%s" % s
                       for i, s in enumerate([N, M])},
            {
                "__A":
                dace.Memlet.simple(
                    "_A", "__i1, __i0" if node.transA else "__i0, __i1"),
                "__x":
                dace.Memlet.simple("_x", "__i1")
            },
            mul_program, {
                "__out":
                dace.Memlet.simple(
                    mul_out, "__i0", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        add_program = "__y_out = ({} * __y_in) + __tmp".format(node.beta)

        memlet_idx = "__i"

        # addition map
        if node.beta != 0:
            state.add_mapped_tasklet(
                "_Add_", {"__i": "0:{}".format(N)}, {
                    "__y_in": dace.Memlet.simple("_y", memlet_idx),
                    "__tmp": dace.Memlet.simple(mul_out, "__i"),
                },
                add_program, {"__y_out": dace.Memlet.simple("_y", "__i")},
                external_edges=True,
                input_nodes={mul_out: access_tmp})

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGemvPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandGEMVIntelFPGAVectorized(ExpandTransformation):

    # Expansion targeting Intel FPGA
    # TODO: tiling

    environments = []

    @staticmethod
    def make_sdfg(node, dtype, parent_state, parent_sdfg, vec_width,
                  tile_m_size, tile_n_size):

        # This expansion accepts plain data type and it is internally vectorized
        # It supports both A non transposed and A transposed, with different expansion

        # This is organized as a sort of two-level map:
        # - the outermost goes over all the row of A - element of y
        # - the innermost is inside the dot product (which internally has another nested SDFG)

        # For the version with A non trasposed, the dot product must be implemented in such a way
        # that the loop is perfectly nested. We use the same approach used in the dot library node
        # (Acttualy we could have re-used it in here)

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------

        # get input sizes
        in_edge = next(parent_state.in_edges_by_connector(node, '_A'))
        n = in_edge.data.subset.size()[0]
        m = in_edge.data.subset.size()[1]
        alpha = node.alpha
        beta = node.beta
        transposed = node.transA
        gemv_sdfg = dace.SDFG("gemv{}_sdfg".format("_t_" if transposed else ""))

        gemv_state = gemv_sdfg.add_state(
            "gemv{}_state".format("_t" if transposed else ""))
        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        A_rows = n
        A_cols = m
        x_size = n if transposed else m
        y_size = m if transposed else n

        # Tiling in transposed GEMV for FPGA depends on vector length
        if transposed and (vec_width > tile_m_size) != False:
            tile_m_size = vec_width

        gemv_sdfg.add_symbol("alpha", dtype)
        gemv_sdfg.add_symbol("beta", dtype)

            
        gemv_sdfg.add_array('_A',
                            shape=[A_rows, A_cols],
                            dtype=dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
        gemv_sdfg.add_array('_x',
                            shape=[x_size],
                            dtype=dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
        gemv_sdfg.add_array('_y',
                            shape=[y_size],
                            dtype=dtype,
                            storage=dace.dtypes.StorageType.FPGA_Global)
        if transposed:
            gemv_sdfg.add_array('tile_y',
                                shape=[tile_m_size],
                                dtype=dtype,
                                storage=dace.dtypes.StorageType.FPGA_Local,
                                transient=True)

        ####################
        # Innermost map:
        ####################

        # this is a nested SDFG, that computes the result for y[i]

        # We need to change the accumulation with beta over y. Changes also the way in which accumulation is performed
        # (we don't have reduction here).
        # To have only a single state also for gemv transposed, we have to pass to the inntermost SDFG
        # the current outer loop iteration. If it is 0, it takes the previous value of y and multiplies it by beta
        # TODO: take beta as a symbol

        if not transposed:

            #########################################
            # Non transposed version: there is a nested SDFG that computes y[i]. We use an approach similar to the one
            # followed in the DOT product
            ########################################

            nested_gemv_sdfg = dace.SDFG("nested_gemv_graph")

            dot_state = nested_gemv_sdfg.add_state("compute_dot_state")

            nested_gemv_sdfg.add_array(
                'nested_A',
                shape=[A_cols],
                dtype=dtype,
                storage=dace.dtypes.StorageType.FPGA_Global)

            nested_gemv_sdfg.add_array(
                'nested_x',
                shape=[x_size],
                dtype=dtype,
                storage=dace.dtypes.StorageType.FPGA_Global)
            nested_gemv_sdfg.add_array(
                'nested_res',
                shape=[1],
                dtype=dtype,
                storage=dace.dtypes.StorageType.FPGA_Global)
            nested_gemv_sdfg.add_array(
                'prev_y',
                dtype=dtype,
                shape=[1],
                storage=dace.dtypes.StorageType.FPGA_Registers)

            # accumulation for the single dot product
            nested_gemv_sdfg.add_array(
                'accum',
                dtype=dtype,
                shape=[1],
                transient=True,
                storage=dace.dtypes.StorageType.FPGA_Registers)

            # ###################################################
            # DOT product to compute y[i]. The dot product is itself another nested SDFG (see DOT node expansion)
            # ###################################################

            nested_dot = dace.SDFG("dot_compute")
            nested_dot.add_symbol("j", dace.int32)
            nested_dot.add_symbol("alpha", dtype)
            nested_dot.add_symbol("beta", dtype)

            nested_dot.add_array('dot_x',
                                 shape=[vec_width],
                                 dtype=dtype,
                                 storage=dace.dtypes.StorageType.FPGA_Global)
            nested_dot.add_array('dot_y',
                                 shape=[vec_width],
                                 dtype=dtype,
                                 storage=dace.dtypes.StorageType.FPGA_Global)
            nested_dot.add_array('dot_res',
                                 shape=[1],
                                 dtype=dtype,
                                 storage=dace.dtypes.StorageType.FPGA_Global)
            nested_dot.add_array('dot_prev_y',
                                 shape=[1],
                                 dtype=dtype,
                                 storage=dace.dtypes.StorageType.FPGA_Registers)

            nested_dot.add_array('dot_accum_in',
                                 dtype=dtype,
                                 shape=[1],
                                 storage=dace.dtypes.StorageType.FPGA_Registers)
            nested_dot.add_array('dot_accum_out',
                                 dtype=dtype,
                                 shape=[1],
                                 storage=dace.dtypes.StorageType.FPGA_Registers)

            # Unrolled map
            dot_product = nested_dot.add_state("product")
            dot_product_map_entry, dot_product_map_exit = dot_product.add_map(
                'product',
                dict(i='0:{}'.format(vec_width)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device,
                unroll=True)

            dot_tasklet = dot_product.add_tasklet(
                'dot_task', ['x_con', 'y_con', 'red_con_in'], ['red_con_out'],
                'red_con_out = red_con_in + x_con * y_con')

            dot_x = dot_product.add_read("dot_x")
            dot_y = dot_product.add_read("dot_y")
            dot_accum_in = dot_product.add_access("dot_accum_in")
            dot_accum_out = dot_product.add_access("dot_accum_out")
            dot_product.add_memlet_path(dot_x,
                                        dot_product_map_entry,
                                        dot_tasklet,
                                        dst_conn='x_con',
                                        memlet=dace.Memlet.simple(
                                            dot_x.data, 'i'))

            dot_product.add_memlet_path(dot_y,
                                        dot_product_map_entry,
                                        dot_tasklet,
                                        dst_conn='y_con',
                                        memlet=dace.Memlet.simple(
                                            dot_y.data, 'i'))

            dot_product.add_memlet_path(dot_accum_in,
                                        dot_product_map_entry,
                                        dot_tasklet,
                                        dst_conn='red_con_in',
                                        memlet=dace.Memlet.simple(
                                            dot_accum_in.data, '0'))
            dot_product.add_memlet_path(dot_tasklet,
                                        dot_product_map_exit,
                                        dot_accum_out,
                                        src_conn='red_con_out',
                                        memlet=dace.Memlet.simple(
                                            dot_accum_out.data, '0'))

            dot_write_result = nested_dot.add_state("dot_write_result")
            res_out = dot_write_result.add_write('dot_res')
            nested_res = dot_write_result.add_write('dot_accum_out')
            dot_prev_y = dot_write_result.add_read('dot_prev_y')

            # copy the result out
            write_tasklet = dot_write_result.add_tasklet(
                'mapToStream_task', ['inCon', 'p_y'], ['outCon'],
                f'outCon = alpha * inCon + beta * p_y')

            dot_write_result.add_memlet_path(nested_res,
                                             write_tasklet,
                                             dst_conn='inCon',
                                             memlet=dace.Memlet.simple(
                                                 nested_res.data, '0'))
            dot_write_result.add_memlet_path(dot_prev_y,
                                             write_tasklet,
                                             dst_conn='p_y',
                                             memlet=dace.Memlet.simple(
                                                 dot_prev_y.data, '0'))

            dot_write_result.add_memlet_path(write_tasklet,
                                             res_out,
                                             src_conn='outCon',
                                             memlet=dace.Memlet.simple(
                                                 res_out.data, '0'))

            # Add interstate edges: copies out only if we are at the last iteration of the outer map
            if_state = nested_dot.add_state_after(dot_product, "if_state")
            empty_state = nested_dot.add_state("empty_state")
            else_state = nested_dot.add_state("else_state")
            nested_dot.add_edge(
                if_state, dot_write_result,
                dace.sdfg.sdfg.InterstateEdge(
                    condition=dace.properties.CodeProperty.from_string(
                        "j == {}/{} - 1".format(n, vec_width),
                        language=dace.dtypes.Language.Python)))
            nested_dot.add_edge(
                if_state, else_state,
                dace.sdfg.sdfg.InterstateEdge(
                    condition=dace.properties.CodeProperty.from_string(
                        "j != {}/{} - 1".format(n, vec_width),
                        language=dace.dtypes.Language.Python)))
            nested_dot.add_edge(dot_write_result, empty_state,
                                dace.sdfg.sdfg.InterstateEdge())
            nested_dot.add_edge(else_state, empty_state,
                                dace.sdfg.sdfg.InterstateEdge())

            # ------------------------------
            # Outer map for the dot product
            # ------------------------------

            accum_init = dot_state.add_access("accum")

            init_tasklet = dot_state.add_tasklet(
                'init_task', [], ['outCon'],
                'outCon = 0;',
                language=dace.dtypes.Language.CPP)

            dot_state.add_memlet_path(init_tasklet,
                                      accum_init,
                                      src_conn='outCon',
                                      memlet=dace.Memlet.simple(
                                          accum_init.data, '0'))

            # Nest the other SDFG

            nested_sdfg = dot_state.add_nested_sdfg(
                nested_dot, nested_gemv_sdfg,
                {"dot_x", "dot_y", "dot_accum_in", "dot_prev_y"},
                {"dot_res", "dot_accum_out"})

            # We need to update, and not simply overwrite, y if beta is different than zero
            # This would cause a loop-carried dependency on the dot product loop.
            # To resolve this, we need to first load y into a transient and then we pass it
            # to the dot product

            A_in = dot_state.add_read('nested_A')
            x_in = dot_state.add_write('nested_x')
            accum_write = dot_state.add_write("accum")

            res_write = dot_state.add_write("nested_res")
            prev_y = dot_state.add_read("prev_y")

            # vectorized Map for the single dot product (one row of A)
            dotMap_entry, dotMap_exit = dot_state.add_map(
                'dot_map',
                dict(j='0:{0}/{1}'.format(A_cols, vec_width)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device)

            dot_state.add_memlet_path(A_in,
                                      dotMap_entry,
                                      nested_sdfg,
                                      dst_conn="dot_x",
                                      memlet=dace.Memlet.simple(
                                          A_in.data, "j*{v}:j*{v}+{v}".format(v=vec_width)))
            dot_state.add_memlet_path(x_in,
                                      dotMap_entry,
                                      nested_sdfg,
                                      dst_conn="dot_y",
                                      memlet=dace.Memlet.simple(
                                          x_in.data, "j*{v}:j*{v}+{v}".format(v=vec_width)))
            dot_state.add_memlet_path(prev_y,
                                      dotMap_entry,
                                      nested_sdfg,
                                      dst_conn="dot_prev_y",
                                      memlet=dace.Memlet.simple(
                                          prev_y.data, "0"))

            dot_state.add_memlet_path(accum_init,
                                      dotMap_entry,
                                      nested_sdfg,
                                      dst_conn="dot_accum_in",
                                      memlet=dace.Memlet.simple(
                                          accum_init, "0"))
            dot_state.add_memlet_path(nested_sdfg,
                                      dotMap_exit,
                                      accum_write,
                                      src_conn='dot_accum_out',
                                      memlet=dace.Memlet.simple(
                                          accum_write.data, "0"))
            dot_state.add_memlet_path(nested_sdfg,
                                      dotMap_exit,
                                      res_write,
                                      src_conn='dot_res',
                                      memlet=dace.Memlet.simple(res_write.data,
                                                                "0",
                                                                dynamic=True))

            # ###################################################
            # END of DOT product
            # ###################################################

            # ###################################################
            # Outermost map: this will go over the row of A
            # ###################################################

            gemvMap_entry, gemvMap_exit = gemv_state.add_map(
                'gemv_row_map',
                dict(i='0:{}'.format(A_rows)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device)

            # Nest the other SDFG
            nested_sdfg = gemv_state.add_nested_sdfg(
                nested_gemv_sdfg, gemv_sdfg, {"nested_A", "nested_x", "prev_y"},
                {"nested_res"})

            A_read = gemv_state.add_read("_A")
            x_read = gemv_state.add_read("_x")
            y_read = gemv_state.add_read("_y")
            y_write = gemv_state.add_write("_y")

            gemv_state.add_memlet_path(A_read,
                                       gemvMap_entry,
                                       nested_sdfg,
                                       dst_conn="nested_A",
                                       memlet=dace.Memlet.simple(
                                           A_read,
                                           "i, 0:{}".format(m),
                                           num_accesses=m))

            gemv_sdfg.add_array('prev_y',
                                shape=[1],
                                dtype=dtype,
                                storage=dace.dtypes.StorageType.FPGA_Registers,
                                transient=True)
            outer_prev_y = gemv_state.add_access("prev_y")
            gemv_state.add_memlet_path(x_read,
                                       gemvMap_entry,
                                       nested_sdfg,
                                       dst_conn="nested_x",
                                       memlet=dace.Memlet.simple(
                                           x_read, "0:{}".format(m)))

            gemv_state.add_memlet_path(y_read,
                                       gemvMap_entry,
                                       outer_prev_y,
                                       dst_conn="prev_y",
                                       memlet=dace.Memlet.simple(y_read, "i"))

            gemv_state.add_memlet_path(outer_prev_y,
                                       nested_sdfg,
                                       dst_conn="prev_y",
                                       memlet=dace.Memlet.simple(
                                           outer_prev_y, "0"))

            gemv_state.add_memlet_path(nested_sdfg,
                                       gemvMap_exit,
                                       y_write,
                                       src_conn='nested_res',
                                       memlet=dace.Memlet.simple(
                                           y_write.data, "i", num_accesses=1))

        else:
            #######################################################
            # Transposed version: we keep the same access pattern for A (one entire row), but we change
            # the access patter for x (only one element is used) and for y (the entire y is updated)
            # A is transposed, the computation is described by three nested maps
            # One for the rows of A, one for the column (strip mined) and one for the computation (unrolled)
            # We support tiling over y (tile_m size) and therefore we have an additional outer map for it:
            # to compute one tile of y, we need to read one tile-column of A, the entire x and one tile of y
            #######################################################

            A_in = gemv_state.add_read('_A')
            x_in = gemv_state.add_read('_x')
            y_in = gemv_state.add_read('_y')
            y_out = gemv_state.add_write('_y')
            tile_y_read = gemv_state.add_access('tile_y')
            tile_y_write = gemv_state.add_write('tile_y')

            # Row Map

            col_tile_map_entry, col_tile_map_exit = gemv_state.add_map(
                'gemv_col_tile_map',
                dict(tj='0:{}/{}'.format(m, tile_m_size)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device)

            row_map_entry, row_map_exit = gemv_state.add_map(
                'gemv_row_map',
                dict(i='0:{}'.format(n)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device)

            # Column map
            col_map_entry, col_map_exit = gemv_state.add_map(
                'gemv_col_map',
                dict(j='0:{}/{}'.format(tile_m_size, vec_width)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device)

            # Unrolled computation map
            compute_map_entry, compute_map_exit = gemv_state.add_map(
                'gemv_compute_map',
                dict(jj='0:{}'.format(vec_width)),
                schedule=dace.dtypes.ScheduleType.FPGA_Device,
                unroll=True)

            compute_tasklet = gemv_state.add_tasklet(
                'gemv_tasklet', ['A_con', 'x_con', 'y_in', 'tile_y_in'],
                ['y_out', 'tile_y_out'], f'if i == 0: tile_y_in = beta * y_in \n'
                f'tile_y_out = tile_y_in + alpha * A_con * x_con\n'
                f'if i==({n}-1): y_out = tile_y_out')

            # Add memlets

            gemv_state.add_memlet_path(A_in,
                                       col_tile_map_entry,
                                       row_map_entry,
                                       col_map_entry,
                                       compute_map_entry,
                                       compute_tasklet,
                                       dst_conn='A_con',
                                       memlet=dace.Memlet.simple(
                                           A_in.data,
                                           "i, tj*{} + j*{} +jj".format(
                                               tile_m_size, vec_width)))

            gemv_state.add_memlet_path(x_in,
                                       col_tile_map_entry,
                                       row_map_entry,
                                       col_map_entry,
                                       compute_map_entry,
                                       compute_tasklet,
                                       dst_conn='x_con',
                                       memlet=dace.Memlet.simple(
                                           x_in.data, "i"))

            ######
            ## y

            # we have to fill tile_y properly: this happens on very outer most map iteration
            # The actual loading into tile_y will be done in the tasklet, where we can add `if` conditions
            gemv_state.add_memlet_path(col_tile_map_entry,
                                       tile_y_read,
                                       memlet=dace.Memlet())

            gemv_state.add_memlet_path(y_in,
                                       col_tile_map_entry,
                                       row_map_entry,
                                       col_map_entry,
                                       compute_map_entry,
                                       compute_tasklet,
                                       dst_conn='y_in',
                                       memlet=dace.Memlet(
                                           "_y[tj*{} + j*{} + jj]".format(
                                               tile_m_size, vec_width),
                                           dynamic=True))
            gemv_state.add_memlet_path(tile_y_read,
                                       row_map_entry,
                                       col_map_entry,
                                       compute_map_entry,
                                       compute_tasklet,
                                       dst_conn='tile_y_in',
                                       memlet=dace.Memlet.simple(
                                           tile_y_read.data,
                                           'j*{}+jj'.format(vec_width),
                                           dynamic=True))

            ## update tile_y
            gemv_state.add_memlet_path(compute_tasklet,
                                       compute_map_exit,
                                       col_map_exit,
                                       row_map_exit,
                                       tile_y_write,
                                       src_conn='tile_y_out',
                                       memlet=dace.Memlet.simple(
                                           tile_y_write.data,
                                           'j*{}+jj'.format(vec_width),
                                           dynamic=True))

            #add tile_y outgoing memlet
            gemv_state.add_memlet_path(tile_y_write,
                                       col_tile_map_exit,
                                       memlet=dace.Memlet())
            # when the tile is complete this will udpate the result in memory
            gemv_state.add_memlet_path(compute_tasklet,
                                       compute_map_exit,
                                       col_map_exit,
                                       row_map_exit,
                                       col_tile_map_exit,
                                       y_out,
                                       src_conn='y_out',
                                       memlet=dace.Memlet.simple(
                                           y_out.data,
                                           'tj*{}+j*{}+jj'.format(
                                               tile_m_size, vec_width),
                                           dynamic=True))

        gemv_sdfg.fill_scope_connectors()
        gemv_sdfg.validate()

        return gemv_sdfg

    @staticmethod
    def expansion(node, state, sdfg, vec_width=1, tile_m_size=1, tile_n_size=1):
        """
        Expand this node with an Intel FPGA vectorized version.
        :param vec_width: Vector width to use.
        :param tile_m_size: Tile size along A columns
        :param tile_n_size: Tile size along A rows
        """
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        nsdfg = ExpandGEMVIntelFPGAVectorized.make_sdfg(node, node.dtype, state,
                                                       sdfg, vec_width,
                                                       tile_m_size, tile_n_size)

        in_edge = next(state.in_edges_by_connector(node, '_A'))
        n = in_edge.data.subset.size()[0]
        m = in_edge.data.subset.size()[1]

        nsdfg_node = dace.sdfg.nodes.NestedSDFG(
            node.label, nsdfg, set(node.in_connectors.keys()),
            set(node.out_connectors.keys()),
            {'n': n, 'm': m, 'alpha': node.alpha, 'beta': node.beta})
        return nsdfg_node



@dace.library.expansion
class Expand_GEMV_FPGA_Streaming_RowTiles(ExpandTransformation):

    environments = []


    @staticmethod
    def make_sdfg(
            dtype, 
            nTile,
            mTile,
            partial_width,
            n, m,
            veclen,
            a, b
        ):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        # A: n rows, m columns, row-major (or transposed column-major)
        gemv_sdfg = dace.SDFG("gemv_fpga_stream_rowTiles")
        gemv_sdfg.add_symbol(a.name, a.dtype)

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        if b != 0:
            gemv_sdfg.add_symbol(b.name, b.dtype)

        gemv_state = gemv_sdfg.add_state()

        A_in = gemv_state.add_stream(
            '_A',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = gemv_state.add_stream(
                '_y',
                single_vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local,
                transient=(True if b == 0 else False)
            )

        # Must be received n/nTile times
        x_in = gemv_state.add_stream(
            '_x',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        res = gemv_state.add_stream(
            '_res',
            single_vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # COMPUTE
        # ---------- ----------

        nMap_entry, nMap_exit = gemv_state.add_map(
            'nTile_map',
            dict(i = '0:{0}/{1}'.format(n, nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        yTile_sdfg = Expand_GEMV_FPGA_Streaming_RowTiles.make_yTile(
            dtype,
            nTile,
            mTile,
            partial_width,
            n, m,
            veclen,
            a, b
        )
        nested_sdfg = gemv_state.add_nested_sdfg(
            yTile_sdfg,
            gemv_sdfg,
            {'_A_yTile', '_x_yTile', '_y_yTile'} if b != 0 else {'_A_yTile', '_x_yTile'},
            {'yTile'}
        )

        gemv_state.add_memlet_path(
            A_in, nMap_entry, nested_sdfg,
            dst_conn='_A_yTile',
            memlet=Memlet.simple(A_in.data, "0:{0}*{1}/{2}".format(n, m, veclen))
        )

        gemv_state.add_memlet_path(
            x_in, nMap_entry, nested_sdfg,
            dst_conn='_x_yTile',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m/veclen))
        )

        if b != 0:
            gemv_state.add_memlet_path(
                y_in, nMap_entry, nested_sdfg,
                dst_conn='_y_yTile',
                memlet=Memlet.simple(y_in.data,"0:{}".format(n))
            ) 

        gemv_state.add_memlet_path(
            nested_sdfg, nMap_exit, res,
            src_conn='yTile',
            memlet=Memlet.simple(res.data, "0:{}".format(n))
        )

        return gemv_sdfg

        



    @staticmethod
    def make_yTile(dtype, nTile, mTile, partial_width, n, m, veclen, a, b):

        yTile_sdfg = dace.SDFG("yTile_sdfg")

        init_state = yTile_sdfg.add_state('yTile_init')
        compute_state = yTile_sdfg.add_state('yTile_compute')

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        yTile_sdfg.add_symbol(a.name, a.dtype)
        if b != 0:
            yTile_sdfg.add_symbol(b.name, b.dtype)

        A_in = compute_state.add_stream(
            '_A_yTile',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = compute_state.add_stream(
                '_y_yTile',
                single_vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local
            )

        x_in = compute_state.add_stream(
            '_x_yTile',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        yTile_sdfg.add_array('y_tileRes', shape=[nTile], dtype=single_vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)

        data_out = compute_state.add_stream(
            'yTile',
            single_vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # INIT State
        # # ---------- ----------
        fpga_init_array(
            init_state,
            'y_tileRes',
            nTile,
            0,
            unroll=True
        )


        # ---------- ----------
        # Compute
        # ---------- ----------
        mMap_entry, mMap_exit = compute_state.add_map(
            'mTile_map',
            dict(j = '0:{0}/{1}'.format(m, mTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        outerComputeMap_entry, outerComputeMap_exit = compute_state.add_map(
            'outerCompute_map',
            dict(ii = '0:{}'.format(nTile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        y_out = compute_state.add_write('y_tileRes')
        
        reducedTile_sdfg = Expand_GEMV_FPGA_Streaming_RowTiles.fpga_make_matrixPartialReduction(
            dtype,
            nTile,
            mTile,
            partial_width,
            n, m,
            veclen,
            a, b
        )


        nested_sdfg = compute_state.add_nested_sdfg(
            reducedTile_sdfg,
            yTile_sdfg,
            {'_A_red', '_x_red', '_y_stream_red'} if b != 0 else {'_A_red', '_x_red'},
            {'_y_red', '_res_red'}
        )


        compute_state.add_memlet_path(
            A_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_A_red',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m))
        )

        compute_state.add_memlet_path(
            x_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_x_red',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m))
        )

        if b != 0:
            compute_state.add_memlet_path(
                y_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
                dst_conn='_y_stream_red',
                memlet=Memlet.simple(y_in.data, "0:{}".format(n))
            )

        compute_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, y_out,
            src_conn='_y_red',
            memlet=Memlet.simple(y_out.data, "0:{}".format(nTile))
        )

        compute_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, data_out,
            src_conn='_res_red',
            memlet=Memlet.simple(data_out.data, "0:{}".format(n))
        )



        yTile_sdfg.fill_scope_connectors()
        yTile_sdfg.add_edge(init_state, compute_state, dace.InterstateEdge(None))

        return yTile_sdfg


    @staticmethod
    def fpga_make_matrixPartialReduction(dtype, nTile, mTile, partial_width, n, m, veclen, a, b):

        # Assumes:
        # i: rowBlock index
        # j: colBlock index
        # ii: tileRow index
        # tile row streamed

        redTile_sdfg = dace.SDFG("redTile_sdfg")

        redTile_sdfg.add_symbol(a.name, a.dtype)

        if b != 0:
            redTile_sdfg.add_symbol(b.name, b.dtype)

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        init_state = redTile_sdfg.add_state('init_reduceTile')
        compute_state = redTile_sdfg.add_state('compute_reduceTile')
        red_state = redTile_sdfg.add_state('red_reduceTile')
        store_state = redTile_sdfg.add_state('store_reduceTile')

        read_y_state = redTile_sdfg.add_state('read_y_reduceTile')
        read_empty_state =  redTile_sdfg.add_state('read_empty_reduceTile')
        write_y_state = redTile_sdfg.add_state('write_y_reduceTile')
        write_empty_state =  redTile_sdfg.add_state('write_empty_reduceTile')
        end_state = redTile_sdfg.add_state('end_reduceTile')

        A_in = compute_state.add_stream(
            '_A_red',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = compute_state.add_stream(
            '_x_red',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = read_y_state.add_stream(
                '_y_stream_red',
                single_vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local
            )

        y_out_stream = write_y_state.add_stream(
            '_res_red',
            single_vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        redTile_sdfg.add_array('_y_red', shape=[nTile], dtype=single_vec_type, storage=dtypes.StorageType.FPGA_Local)

        redTile_sdfg.add_array('red_buf',
            shape=[max(2, partial_width)],
            dtype=dtype,
            storage=dtypes.StorageType.FPGA_Local if partial_width > 8 else dtypes.StorageType.FPGA_Registers,
            transient=True
        )
        redTile_sdfg.add_array('res_buf', shape=[1], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
        redTile_sdfg.add_array('x_buf', shape=[mTile/veclen], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)


        # ---------- ----------
        # Y READ
        # ---------- ----------
        y_out = read_y_state.add_write("_y_red")

        code = 'outCon = inCon * {}'.format(b)
        if b == 0:
            code = 'outCon = 0'

        y_read_tasklet = read_y_state.add_tasklet(
            'read_y_tasklet',
            (['inCon'] if b != 0 else []),
            ['outCon'],
            code
        )

        if b != 0:
            read_y_state.add_memlet_path(
                y_in, y_read_tasklet,
                dst_conn='inCon',
                memlet=Memlet.simple(y_in.data, "0", num_accesses=-1)
            )

        read_y_state.add_memlet_path(
            y_read_tasklet, y_out,
            src_conn='outCon',
            memlet=Memlet.simple(y_out.data, "ii")
        )


        redTile_sdfg.add_edge(init_state, read_y_state, dace.InterstateEdge("j == 0"))
        redTile_sdfg.add_edge(read_y_state, compute_state, dace.InterstateEdge(None))
        redTile_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("j != 0"))
        redTile_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))


        # ---------- ----------
        # INIT
        # ---------- ----------
        fpga_init_array(
            init_state,
            'red_buf',
            partial_width,
            0
        )

        fpga_init_array(
            init_state,
            'res_buf',
            1,
            0
        )


        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        buf_out = compute_state.add_write('red_buf')
        buf_in = compute_state.add_read('red_buf')
        x_buf = compute_state.add_read('x_buf')

        innerComputeMap_entry, innerComputeMap_exit = compute_state.add_map(
            'innerCompute_map',
            dict(jj = '0:{0}/{1}'.format(mTile, veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device,
        )


        red_sdfg = Expand_GEMV_FPGA_Streaming_RowTiles.make_unrolledCompute(
            dtype,
            nTile,
            mTile,
            veclen,
            partial_width,
            n, m, a
        )

        nested_sdfg = compute_state.add_nested_sdfg(
            red_sdfg,
            redTile_sdfg,
            {'_A_unroll', '_x_unroll', '_buf_in_unroll'},
            {'buf_out', '_x_buf'}
        )

        compute_state.add_memlet_path(
            A_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_A_unroll',
            memlet=Memlet.simple(A_in.data, "0:{}*{}/{}".format(n, m, veclen))
        )

        compute_state.add_memlet_path(
            x_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_x_unroll',
            memlet=Memlet.simple(x_in.data, "0:{}/{}".format(m, veclen))
        )

        compute_state.add_memlet_path(
            buf_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_buf_in_unroll',
            memlet=Memlet.simple(buf_in.data, "0:{}".format(max(2, partial_width)))
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, buf_out,
            src_conn='buf_out',
            memlet=Memlet.simple(buf_out.data, "0:{}".format(max(2, partial_width)))
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, x_buf,
            src_conn='_x_buf',
            memlet=Memlet.simple(x_buf.data, "0:{}/{}".format(mTile, veclen))
        )



        # ---------- ----------
        # REDUCE
        # ---------- ----------
        buf_in = red_state.add_read('red_buf')
        buf_res_in = red_state.add_read('res_buf')
        buf_res = red_state.add_write('res_buf')

        task, mapEn, mapEx = red_state.add_mapped_tasklet(
            'finalReduction',
            dict(k='0:{}'.format(partial_width)),
            dict(
                inCon=Memlet.simple(buf_in.data, 'k'),
                prevCon=Memlet.simple(buf_res_in.data, '0')
            ),
            'outCon = inCon + prevCon',
            dict(
                outCon=Memlet.simple(
                    buf_res.data, '0',
                    # wcr_str='lambda a, b: a + b'
                )
            )
        )

        red_state.add_edge(
            buf_in, None,
            mapEn, None,
            memlet=Memlet.simple(buf_in.data, "0:{}".format(partial_width))
        )

        red_state.add_edge(
            buf_res_in, None,
            mapEn, None,
            memlet=Memlet.simple(buf_res_in.data, "0")
        )

        red_state.add_edge(
            mapEx, None,
            buf_res, None,
            memlet=Memlet.simple(
                buf_res.data,
                "0",
                # wcr_str='lambda a, b: a + b'
            )
        )


        # ---------- ----------
        # STORE
        # ---------- ----------
        res = store_state.add_read('res_buf')
        out = store_state.add_write('_y_red')

        store_task = store_state.add_tasklet(
            'storeRed_task',
            ['inCon'],
            ['outCon'],
            'outCon = inCon'
        )

        store_state.add_memlet_path(
            res, store_task,
            dst_conn='inCon',
            memlet=Memlet.simple(res.data, '0')
        )

        store_state.add_memlet_path(
            store_task, out,
            src_conn='outCon',
            memlet=Memlet.simple(
                out.data, 
                "ii",
                wcr_str="lambda a, b: a + b"
            )
        )

        # ---------- ----------
        # Stream out
        # ---------- ----------
        y_in = write_y_state.add_read('_y_red')

        write_y_state.add_memlet_path(
            y_in, y_out_stream,
            memlet=Memlet.simple(y_in.data, "ii", other_subset_str='0')
        )


        redTile_sdfg.add_edge(store_state, write_y_state, dace.InterstateEdge("j == ({0}/{1}) - 1".format(m, mTile)))
        redTile_sdfg.add_edge(write_y_state, end_state, dace.InterstateEdge(None))
        redTile_sdfg.add_edge(store_state, write_empty_state, dace.InterstateEdge("j != ({0}/{1}) - 1".format(m, mTile)))
        redTile_sdfg.add_edge(write_empty_state, end_state, dace.InterstateEdge(None))

        redTile_sdfg.add_edge(compute_state, red_state, dace.InterstateEdge(None))
        redTile_sdfg.add_edge(red_state, store_state, dace.InterstateEdge(None))

        redTile_sdfg.fill_scope_connectors()

        return redTile_sdfg



    @staticmethod
    def make_unrolledCompute(dtype, nTile, mTile, veclen, partial_width, n, m, a):

        inner_sdfg = dace.SDFG("partial_reduction_inner")

        inner_sdfg.add_symbol(a.name, a.dtype)

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        init_state = inner_sdfg.add_state("init_state")
        compute_state = inner_sdfg.add_state('compute_state')
        write_state = inner_sdfg.add_state("write_state")

        read_x_state = inner_sdfg.add_state("readX_state")
        read_empty_state = inner_sdfg.add_state("readEmpty_state")

        A_in = init_state.add_stream(
            '_A_unroll',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = read_x_state.add_stream(
            '_x_unroll',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        inner_sdfg.add_array('_buf_in_unroll', shape=[max(2, partial_width)], dtype=dtype,
            storage=dtypes.StorageType.FPGA_Local if partial_width > 8 else dtypes.StorageType.FPGA_Registers
        )
        inner_sdfg.add_array('buf_out', shape=[max(2, partial_width)], dtype=dtype,
            storage=dtypes.StorageType.FPGA_Local if partial_width > 8 else dtypes.StorageType.FPGA_Registers
        )

        inner_sdfg.add_array('_x_buf', shape=[mTile/veclen], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local)
        inner_sdfg.add_array('vecBuf_x', shape=[1], dtype=vec_type, storage=dtypes.StorageType.FPGA_Registers, transient=True)
        inner_sdfg.add_array('memBuf_x', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)


        inner_sdfg.add_array('vecBuf_A', shape=[1], dtype=vec_type, storage=dtypes.StorageType.FPGA_Registers, transient=True)
        inner_sdfg.add_array('memBuf_A', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)

        

        inner_sdfg.add_array(
            'buf_reg',
            shape=[1],
            dtype=dtype,
            storage=dtypes.StorageType.FPGA_Registers,
            transient=True
        )


        # INIT Get A
        # -----------------------
        fpga_init_array(
            init_state,
            'buf_reg',
            1,
            0
        )

        vecBuf_A = init_state.add_access("vecBuf_A")
        memBuf_A = init_state.add_write("memBuf_A")

        init_state.add_memlet_path(
            A_in, vecBuf_A,
            memlet=Memlet.simple(vecBuf_A.data, "0", other_subset_str="0")
        )

        init_state.add_memlet_path(
            vecBuf_A, memBuf_A,
            memlet=Memlet.simple(vecBuf_A.data, "0")
        )


        # INIT Get x
        # -----------------------
        data_out = read_x_state.add_write('_x_buf')


        read_x_state.add_memlet_path(
            x_in, data_out,
            memlet=Memlet.simple(data_out.data, "jj", other_subset_str="0")
        )

        

        inner_sdfg.add_edge(init_state, read_x_state, dace.InterstateEdge("ii == 0"))
        inner_sdfg.add_edge(read_x_state, compute_state, dace.InterstateEdge(None))   

        inner_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("ii != 0"))
        inner_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))     



        # COMPUTE
        # -----------------------
        inner_buf_reg_read = compute_state.add_read('buf_reg')
        inner_buf_reg = compute_state.add_write('buf_reg')
        x_in = compute_state.add_read('_x_buf')

        memBuf = compute_state.add_access('memBuf_x')
        vecBuf = compute_state.add_access('vecBuf_x')

        compute_state.add_memlet_path(
            x_in, vecBuf,
            memlet=Memlet.simple(vecBuf.data, "0", other_subset_str="jj")
        )

        compute_state.add_memlet_path(
            vecBuf, memBuf,
            memlet=Memlet.simple(vecBuf.data, "0")
        )


        innerMap_entry, innerMap_exit = compute_state.add_map(
            'compRed_innerMap',
            dict(j_inner='0:{}'.format(veclen)),
            unroll=True,
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        innerTasklet = compute_state.add_tasklet(
            'compute_task',
            ['A_con', 'x_con', 'prev_con'],
            ['outCon'],
            'outCon = {} * A_con * x_con + prev_con'.format(a)
        )

        compute_state.add_memlet_path(
            memBuf_A, innerMap_entry, innerTasklet,
            dst_conn='A_con',
            memlet=Memlet.simple(
                memBuf_A.data, 'j_inner'
            )
        )

        compute_state.add_memlet_path(
            memBuf, innerMap_entry, innerTasklet,
            dst_conn='x_con',
            memlet=Memlet.simple(
                memBuf.data, 'j_inner'
            )
        )

        compute_state.add_memlet_path(
            inner_buf_reg_read, innerMap_entry, innerTasklet,
            dst_conn='prev_con',
            memlet=Memlet.simple(
                inner_buf_reg_read.data, '0'
            )
        )

        compute_state.add_memlet_path(
            innerTasklet, innerMap_exit, inner_buf_reg,
            src_conn='outCon',
            memlet=Memlet.simple(
                inner_buf_reg.data, '0',
                # wcr_str='lambda a, b: a + b',
            )
        )

        # WRITE
        # -----------------------
        inner_buf_reg = write_state.add_read('buf_reg')
        partial_out = write_state.add_write('buf_out')
        partial_in = write_state.add_read('_buf_in_unroll')

        write_task = write_state.add_tasklet(
            'write_out_task',
            ['inCon', 'prevCon'],
            ['outCon'],
            'outCon = prevCon + inCon'
        )

        write_state.add_memlet_path(
            inner_buf_reg, write_task,
            dst_conn='inCon',
            memlet=Memlet.simple(inner_buf_reg.data, '0')
        )

        write_state.add_memlet_path(
            partial_in, write_task,
            dst_conn='prevCon',
            memlet=Memlet.simple(partial_in.data, 'jj % {0}'.format(partial_width))
        )

        write_state.add_memlet_path(
            write_task, partial_out,
            src_conn='outCon',
            memlet=Memlet.simple(partial_out.data, 'jj % {0}'.format(partial_width))
        )

        inner_sdfg.fill_scope_connectors()
        inner_sdfg.add_edge(compute_state, write_state, dace.InterstateEdge(None))     

        return inner_sdfg



    @staticmethod
    def expansion(node, state, sdfg):

        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                                ".")
        return Expand_GEMV_FPGA_Streaming_RowTiles.make_sdfg(
            node.dtype,
            int(node.n_tile),
            int(node.m_tile),
            node.partial_width,
            node.n,
            node.m,
            int(node.veclen),
            node.alpha,
            node.beta
        )


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGemvPure,
        "IntelFPGA": ExpandGEMVIntelFPGAVectorized,
        "fpga_row": Expand_GEMV_FPGA_Streaming_RowTiles
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    alpha = dace.properties.SymbolicProperty(
        allow_none=False, default=1)
    beta = dace.properties.SymbolicProperty(
        allow_none=False, default=0)

    transA = Property(dtype=bool,
                      desc="Whether to transpose A before multiplying")

    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)
    n_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    m_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))

    # Size of partial buffer in dot reduction to shadow loop-carried dependency
    # latency of reduction operation on FPGA
    partial_width = dace.properties.SymbolicProperty(allow_none=False, default=16)

    def __init__(self,
                 name,
                 dtype=None,
                 location=None,
                 transA=False,
                 alpha=1,
                 beta=0,
                 veclen=1,
                 n_tile=1,
                 m_tile=1,
                 n=dace.symbolic.symbol("n"),
                 m=dace.symbolic.symbol("m"),
                 partial_width=1,
                 streaming=False):
        super().__init__(
            name,
            location=location,
            inputs={"_A", "_x", "_y"} if beta != 0 else {"_A", "_x"},
            outputs= {"_res"} if streaming else {"_y"})

        self.dtype = dtype
        self.transA = transA
        self.alpha = alpha
        self.beta = beta

        self.n = n
        self.m = m
        self.veclen = veclen
        self.n_tile = n_tile
        self.m_tile = m_tile
        self.partial_width = partial_width




    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to GEMV")
        size_y_in = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a = subset.size()
            if dst_conn == "_x":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
            if dst_conn == "_y":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y_in = subset.size()

        if self.transA:
            size_a = list(reversed(size_a))

        # TODO: test not working with streaming
        # if len(size_a) != 2 or len(size_x) != 1:
        #     raise ValueError(
        #         "Matrix-vector product only supported on matrix-vector input")

        # if size_a[1] != size_x[0]:
        #     raise ValueError("Inputs to matrix-matrix product "
        #                      "must agree in the k-dimension")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-vector product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_y_out = out_subset.size()
        if size_y_in is not None and size_y_in != size_y_out:
            raise ValueError("Input y-vector must match output y-vector.")
        if (len(size_y_out) != 1 or size_y_out[0] != size_a[0]):
            raise ValueError("Vector input to GEMV must match matrix rows.")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemv')
@oprepo.replaces('dace.libraries.blas.Gemv')
def gemv_libnode(sdfg: SDFG,
                 state: SDFGState,
                 A,
                 x,
                 y,
                 alpha,
                 beta,
                 trans=None):
    # Get properties
    if trans is None:
        trans = (sdfg.arrays[x].shape[0] == sdfg.arrays[A].shape[0])

    # Add nodes
    A_in, x_in = (state.add_read(name) for name in (A, x))
    y_out = state.add_write(y)

    libnode = Gemv('gemv',
                   dtype=sdfg.arrays[A].dtype,
                   transA=trans,
                   alpha=alpha,
                   beta=beta)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_y', y_out, None, mm.Memlet(y))

    if beta != 0:
        y_in = state.add_read(y)
        state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))

    return []
