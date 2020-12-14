# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.sdfg import SDFG, SDFGState
from dace import memlet as mm, data as dt
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from dace.frontend.common import op_repository as oprepo
from .. import environments
import numpy as np


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

            dot_state.add_memlet_path(
                A_in,
                dotMap_entry,
                nested_sdfg,
                dst_conn="dot_x",
                memlet=dace.Memlet.simple(
                    A_in.data, "j*{v}:j*{v}+{v}".format(v=vec_width)))
            dot_state.add_memlet_path(
                x_in,
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
                ['y_out', 'tile_y_out'],
                f'if i == 0: tile_y_in = beta * y_in \n'
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
                                                        tile_m_size,
                                                        tile_n_size)

        in_edge = next(state.in_edges_by_connector(node, '_A'))
        n = in_edge.data.subset.size()[0]
        m = in_edge.data.subset.size()[1]

        nsdfg_node = dace.sdfg.nodes.NestedSDFG(node.label, nsdfg,
                                                set(node.in_connectors.keys()),
                                                set(node.out_connectors.keys()),
                                                {
                                                    'n': n,
                                                    'm': m,
                                                    'alpha': node.alpha,
                                                    'beta': node.beta
                                                })
        return nsdfg_node


@dace.library.expansion
class ExpandGemvAccumulate(ExpandTransformation):
    # This corresponds to gemv_v1 in FBLAS

    environments = []

    @staticmethod
    def expansion(node,
                  state,
                  sdfg,
                  tile_size_x=None,
                  tile_size_y=None,
                  num_partial_sums=16,
                  **kwargs):

        node.validate(sdfg, state)

        for e in state.in_edges(node):
            if e.dst_conn == "_A":
                desc_a = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("gemv")
        state = sdfg.add_state("gemv")

        alpha = node.alpha
        beta = node.beta

        # Create local versions of input data nodes
        desc_a = desc_a.clone()
        desc_a.transient = False
        sdfg.add_datadesc("_A", desc_a)
        desc_x = desc_x.clone()
        desc_x.transient = False
        sdfg.add_datadesc("_x", desc_x)
        desc_y = desc_y.clone()
        desc_y.transient = False
        sdfg.add_datadesc("_y", desc_y)

        # Create accesses
        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        if beta != 0:
            read_y = state.add_read("_y")
        write_y = state.add_write("_y")

        size_x = desc_x.shape[0]
        size_y = desc_y.shape[0]
        if tile_size_x is None:
            tile_size_x = size_x
        if tile_size_y is None:
            tile_size_y = size_y
        num_tiles_y = f"{size_y}/{tile_size_y}"
        num_tiles_x = f"{size_x}/{tile_size_x}"

        veclen = desc_a.dtype.veclen

        # Create tile map
        y_tile_entry, y_tile_exit = state.add_map(
            "y_tiles", {"ty": f"0:{num_tiles_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        x_tile_entry, x_tile_exit = state.add_map(
            "x_tiles", {"tx": f"0:{num_tiles_x}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create y map
        y_entry, y_exit = state.add_map(
            "y", {"iy": f"0:{tile_size_y}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create x map
        x_entry, x_exit = state.add_map(
            "x", {"ix": f"0:{tile_size_x}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Local buffer of x
        sdfg.add_array("x_local", (tile_size_x, ),
                       desc_x.dtype,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)
        x_local_access = state.add_read("x_local")

        if beta != 0:
            raise NotImplementedError("Not yet implemented.")


        multiply_tasklet = state.add_tasklet("multiply", {"A_in", "x_in"},
                                             {f"product": desc_a.dtype},
                                             "product = A_in * x_in")

        if isinstance(desc_a, dt.Stream):
            subset = "0"
        elif node.transA:
            subset = f"tx * {tile_size_x} + ix, ty * {tile_size_y} + iy"
        else:
            subset = f"ty * {tile_size_y} + iy, tx * {tile_size_x} + ix"
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              y_entry,
                              x_entry,
                              multiply_tasklet,
                              dst_conn="A_in",
                              memlet=dace.Memlet(f"_A[{subset}]"))
        subset = ("0" if isinstance(desc_x, dt.Stream) else
                  f"tx*{tile_size_x}:(tx + 1)*{tile_size_x}")
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              x_local_access,
                              memlet=dace.Memlet(
                                  f"_x[{subset}]",
                                  other_subset=f"0:{tile_size_x}"))
        state.add_memlet_path(x_local_access,
                              y_entry,
                              x_entry,
                              multiply_tasklet,
                              dst_conn="x_in",
                              memlet=dace.Memlet(f"x_local[ix]"))

        # Write to buffer
        sdfg.add_array("product_vector", (1, ),
                       desc_a.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        product_vector = state.add_access("product_vector")
        state.add_memlet_path(multiply_tasklet,
                              product_vector,
                              src_conn="product",
                              memlet=dace.Memlet(f"product_vector[0]"))

        # Vector length conversion
        sdfg.add_array("product_scalar", (veclen, ),
                       desc_a.dtype.base_type,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        product_scalar = state.add_access("product_scalar")
        state.add_memlet_path(product_vector,
                              product_scalar,
                              memlet=dace.Memlet(
                                  f"product_vector[0]",
                                  other_subset=f"0:{veclen}"))

        # Now we need to collapse this
        reduce_vector_entry, reduce_vector_exit = state.add_map(
            "reduce_vector", {"u": f"0:{veclen}"},
            schedule=dace.ScheduleType.FPGA_Device,
            unroll=True)

        reduce_vector_tasklet = state.add_tasklet(
            "reduce_vector", {"product_in", "acc_in"}, {"acc_out"},
            "acc_out = product_in + acc_in")
        state.add_memlet_path(product_scalar,
                              reduce_vector_entry,
                              reduce_vector_tasklet,
                              dst_conn="product_in",
                              memlet=dace.Memlet(f"{product_scalar}[u]"))

        # Add accumulation register
        sdfg.add_array("accumulate_product", (1, ),
                       desc_a.dtype.base_type,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        accumulate_product_read = state.add_access("accumulate_product")
        accumulate_product_write = state.add_access("accumulate_product")

        # Initialize it to zero
        init_reduce_vector_tasklet = state.add_tasklet("init_reduce_vector", {},
                                                       {"acc_out"},
                                                       "acc_out = 0")
        state.add_memlet_path(x_entry,
                              init_reduce_vector_tasklet,
                              memlet=dace.Memlet())
        state.add_memlet_path(init_reduce_vector_tasklet,
                              accumulate_product_read,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"accumulate_product[0]"))

        # Connect it to the tasklet
        state.add_memlet_path(accumulate_product_read,
                              reduce_vector_entry,
                              reduce_vector_tasklet,
                              dst_conn="acc_in",
                              memlet=dace.Memlet(f"accumulate_product[0]"))
        state.add_memlet_path(reduce_vector_tasklet,
                              reduce_vector_exit,
                              accumulate_product_write,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"accumulate_product[0]"))

        # Partial sums
        sdfg.add_array("partial_sums", (num_partial_sums, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Registers,
                       transient=True)
        partial_sum_read = state.add_read("partial_sums")
        partial_sum_write = state.add_access("partial_sums")

        # Output array
        sdfg.add_array("y_local", (tile_size_y, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)

        # Now we need to actually accumulate into a local register of y
        y_local_read = state.add_read("y_local")
        y_local_write = state.add_read("y_local")
        update_y_tasklet = state.add_tasklet(
            "update_y", {"y_in", "acc_in"}, {"acc_out"}, f"""\
prev = acc_in if ix >= {num_partial_sums} else 0
acc_out = prev + y_in""")
        state.add_memlet_path(accumulate_product_write,
                              update_y_tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet(f"accumulate_product[0]"))
        state.add_memlet_path(partial_sum_read,
                              x_entry,
                              update_y_tasklet,
                              dst_conn="acc_in",
                              memlet=dace.Memlet(f"partial_sums[ix%{num_partial_sums}]"))
        state.add_memlet_path(y_tile_entry, y_local_read, memlet=dace.Memlet())
        state.add_memlet_path(y_entry, partial_sum_read, memlet=dace.Memlet())
        state.add_memlet_path(update_y_tasklet,
                              x_exit,
                              partial_sum_write,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"partial_sums[ix%{num_partial_sums}]"))

        # Reduce the partial sums
        reduce_sums_entry, reduce_sums_exit = state.add_map(
            "reduce_partial_sums", {"u": f"0:{num_partial_sums}"},
            schedule=dace.ScheduleType.FPGA_Device,
            unroll=True)
        reduce_sums_tasklet = state.add_tasklet("reduce_partial_sums", {"sum_in", "val_in"}, {"sum_out"}, """
prev = sum_in if u > 0 else 0
sum_out = prev + val_in""")
        sdfg.add_array("accumulate_sum", (1, ), desc_y.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        accumulate_sum_read = state.add_access("accumulate_sum")
        accumulate_sum_write = state.add_access("accumulate_sum")
        state.add_memlet_path(y_entry,
                              accumulate_sum_read,
                              memlet=dace.Memlet())
        state.add_memlet_path(accumulate_sum_read,
                              reduce_sums_entry,
                              reduce_sums_tasklet,
                              dst_conn="sum_in",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(reduce_sums_tasklet,
                              reduce_sums_exit,
                              accumulate_sum_write,
                              src_conn="sum_out",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(partial_sum_write,
                              reduce_sums_entry,
                              reduce_sums_tasklet,
                              dst_conn="val_in",
                              memlet=dace.Memlet("partial_sums[u]"))

        # Combine with y buffer
        combine_tasklet = state.add_tasklet("combine_y", {"val", "buffer_in"},
                                            {"buffer_out"},
                                            """\
prev = buffer_in if tx > 0 else 0
buffer_out = prev + val""")
        state.add_memlet_path(accumulate_sum_write,
                              combine_tasklet,
                              dst_conn="val",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(y_local_read,
                              x_tile_entry,
                              y_entry,
                              combine_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet("y_local[iy]"))


        state.add_memlet_path(combine_tasklet,
                              y_exit,
                              x_tile_exit,
                              y_local_write,
                              src_conn="buffer_out",
                              memlet=dace.Memlet(f"y_local[iy]"))


        subset = ("0" if isinstance(desc_y, dt.Stream) else
                  f"ty*{tile_size_y}:(ty + 1)*{tile_size_y}")
        state.add_memlet_path(y_local_write,
                              y_tile_exit,
                              write_y,
                              src_conn="y_out",
                              memlet=dace.Memlet(f"_y[{subset}]",
                                                 other_subset=f"0:{tile_size_y}",
                                                 dynamic=True))

        return sdfg


@dace.library.expansion
class ExpandGemvTilesByColumn(ExpandTransformation):
    # This corresponds to gemv_v2 in FBLAS

    environments = []

    @staticmethod
    def expansion(node,
                  state,
                  sdfg,
                  tile_size_x=None,
                  tile_size_y=None,
                  **kwargs):

        node.validate(sdfg, state)

        for e in state.in_edges(node):
            if e.dst_conn == "_A":
                desc_a = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("gemv")
        state = sdfg.add_state("gemv")

        alpha = node.alpha
        beta = node.beta

        # Create local versions of input data nodes
        desc_a = desc_a.clone()
        desc_a.transient = False
        sdfg.add_datadesc("_A", desc_a)
        desc_x = desc_x.clone()
        desc_x.transient = False
        sdfg.add_datadesc("_x", desc_x)
        desc_y = desc_y.clone()
        desc_y.transient = False
        sdfg.add_datadesc("_y", desc_y)

        # Create accesses
        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        if beta != 0:
            read_y = state.add_read("_y")
        write_y = state.add_write("_y")

        size_x = desc_x.shape[0]
        size_y = desc_y.shape[0]
        if tile_size_x is None:
            tile_size_x = size_x
        if tile_size_y is None:
            tile_size_y = size_y
        num_tiles_y = f"{size_y}/{tile_size_y}"
        num_tiles_x = f"{size_x}/{tile_size_x}"

        # Create y tile map
        y_tile_entry, y_tile_exit = state.add_map(
            "y_tiles", {"ty": f"0:{num_tiles_y}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create buffer
        sdfg.add_array("y_local", (tile_size_y, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)
        y_local = state.add_access("y_local")
        y_local_write = state.add_access("y_local")

        # Initialize buffer
        init_entry, init_exit = state.add_map(
            "init", {"iy": f"0:{tile_size_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        if beta != 0:
            if isinstance(desc_y, dt.Stream):
                subset = "0"
            else:
                subset = f"ty*{tile_size_y}+iy"
            init_tasklet = state.add_tasklet(
                "init", {"y_in"}, {"y_out"},
                f"y_out = {desc_y.dtype.base_type.ctype}({beta}) * y_in")
            state.add_memlet_path(read_y,
                                  y_tile_entry,
                                  init_entry,
                                  init_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet(f"_y[{subset}]"))
            state.add_memlet_path(init_tasklet,
                                  init_exit,
                                  y_local,
                                  src_conn="y_out",
                                  memlet=dace.Memlet(f"y_local[iy]"))
        else:
            state.add_memlet_path(y_tile_entry,
                                  init_entry,
                                  memlet=dace.Memlet())
            init_tasklet = state.add_tasklet("init", {}, {"y_out"}, "y_out = 0")
            state.add_memlet_path(init_entry,
                                  init_tasklet,
                                  memlet=dace.Memlet())
            state.add_memlet_path(init_tasklet,
                                  init_exit,
                                  y_local,
                                  src_conn="y_out",
                                  memlet=dace.Memlet("y_local[iy]"))

        # Create x tile map
        x_tile_entry, x_tile_exit = state.add_map(
            "x_tiles", {"tx": f"0:{num_tiles_x}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create loop over tile size in x
        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Buffer a scalar value of x
        sdfg.add_array("x_local", (1, ),
                       desc_x.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        x_local = state.add_access("x_local")
        subset = "0" if isinstance(desc_x,
                                   dt.Stream) else f"tx*{tile_size_x}+ix"
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              x_local,
                              memlet=dace.Memlet(f"_x[{subset}]"))

        # Create loop over tile size in y
        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Do computation
        tasklet = state.add_tasklet("gemv", {"A_in", "x_in", "y_in"}, {"y_out"},
                                    f"y_out = y_in + {alpha} * A_in * x_in")
        state.add_memlet_path(y_local,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet("y_local[iy]"))
        state.add_memlet_path(x_local,
                              y_entry,
                              tasklet,
                              dst_conn="x_in",
                              memlet=dace.Memlet("x_local[0]"))
        state.add_memlet_path(tasklet,
                              y_exit,
                              x_exit,
                              x_tile_exit,
                              y_local_write,
                              src_conn="y_out",
                              memlet=dace.Memlet("y_local[iy]"))
        if isinstance(desc_a, dt.Stream):
            subset = "0"
        elif node.transA:
            subset = f"tx * {tile_size_x} + ix, ty * {tile_size_y} + iy"
        else:
            subset = f"ty * {tile_size_y} + iy, tx * {tile_size_x} + ix"
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              tasklet,
                              dst_conn="A_in",
                              memlet=dace.Memlet(f"_A[{subset}]"))

        # Write out tile of y
        subset = ("0" if isinstance(desc_y, dt.Stream) else
                  f"ty * {tile_size_y}:(ty + 1) * {tile_size_y}")
        state.add_memlet_path(y_local_write,
                              y_tile_exit,
                              write_y,
                              memlet=dace.Memlet(f"_y[{subset}]"))

        return sdfg


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGemvPure,
        "IntelFPGA": ExpandGEMVIntelFPGAVectorized,
        "TilesByColumn": ExpandGemvTilesByColumn,
        "Accumulate": ExpandGemvAccumulate,
        "FPGA": ExpandGemvTilesByColumn
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    alpha = dace.properties.SymbolicProperty(allow_none=False, default=1)
    beta = dace.properties.SymbolicProperty(allow_none=False, default=0)

    transA = Property(dtype=bool,
                      desc="Whether to transpose A before multiplying")

    def __init__(self,
                 name,
                 dtype=None,
                 location=None,
                 transA=False,
                 alpha=1,
                 beta=0):
        super().__init__(
            name,
            location=location,
            inputs={"_A", "_x", "_y"} if beta != 0 else {"_A", "_x"},
            outputs={"_y"})
        self.dtype = dtype
        self.transA = transA
        self.alpha = alpha
        self.beta = beta

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

        if len(size_a) != 2 or len(size_x) != 1:
            raise ValueError(
                "Matrix-vector product only supported on matrix-vector input")

        if size_a[1] != size_x[0]:
            raise ValueError("Inputs to matrix-matrix product "
                             "must agree in the k-dimension")

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
