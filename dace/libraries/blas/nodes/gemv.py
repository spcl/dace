# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.symbolic import symstr
from dace.properties import Property
import dace.library
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from .. import environments
import numpy as np

from dace.libraries.blas.utility.initialization import fpga_init_array
from dace.libraries.blas.utility.fpga_helper import StreamWriteVector, StreamReadVector
from dace.libraries.blas.utility.fpga_helper import StreamReadMatrixFull
from dace.libraries.blas.utility.reductions import fpga_make_matrixPartialReduction


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
        state.add_mapped_tasklet("_Add_", {"__i": "0:{}".format(N)}, {
            "__y_in": dace.Memlet.simple("_y", memlet_idx),
            "__tmp": dace.Memlet.simple(mul_out, "__i"),
        },
                                 add_program,
                                 {"__y_out": dace.Memlet.simple("_y", "__i")},
                                 external_edges=True,
                                 input_nodes={mul_out: access_tmp})

        sdfg.save('/tmp/expansion.sdfg')
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGemvPure.make_sdfg(node, state, sdfg)




@dace.library.expansion
class ExpandGemvFPGAStreamingRowTiles(ExpandTransformation):

    environments = []


    @staticmethod
    def make_sdfg(
            dtype,
            nTile,
            mTile,
            partialWidth,
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

        if b != 0:
            gemv_sdfg.add_symbol(b.name, b.dtype)
        gemv_state = gemv_sdfg.add_state()
        vec_type=dace.dtypes.vector(dtype, 1)

        A_in = gemv_state.add_stream(
            '_A',
            vec_type,
            buffer_size=32,
            storage=dace.dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = gemv_state.add_stream(
                '_yi',
                vec_type,
                buffer_size=32,
                storage=dace.dtypes.StorageType.FPGA_Local,
                transient=(True if b == 0 else False)
            )

        # Must be received n/nTile times
        x_in = gemv_state.add_stream(
            '_x',
            vec_type,
            buffer_size=32,
            storage=dace.dtypes.StorageType.FPGA_Local
        )

        y_out = gemv_state.add_stream(
            '_yo',
            dtype=vec_type,
            buffer_size=32,
            storage=dace.dtypes.StorageType.FPGA_Local
        )

        # ---------- ----------
        # COMPUTE
        # ---------- ----------

        nMap_entry, nMap_exit = gemv_state.add_map(
            'nTile_map',
            dict(i = '0:{0}/{1}'.format(n, nTile)),
            schedule=dace.dtypes.ScheduleType.FPGA_Device
        )

        yTile_sdfg = ExpandGemvFPGAStreamingRowTiles.make_yTile(
            dtype,
            nTile,
            mTile,
            partialWidth,
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
            memlet=dace.Memlet.simple(A_in.data, "0:{0}*{1}".format(n, m))#, veclen=veclen)
        )

        gemv_state.add_memlet_path(
            x_in, nMap_entry, nested_sdfg,
            dst_conn='_x_yTile',
            memlet=dace.Memlet.simple(x_in.data, "0:{}".format(m))#, veclen=veclen)
        )

        if b != 0:
            gemv_state.add_memlet_path(
                y_in, nMap_entry, nested_sdfg,
                dst_conn='_y_yTile',
                memlet=dace.Memlet.simple(y_in.data,"0:{}".format(n))
            )

        gemv_state.add_memlet_path(
            nested_sdfg, nMap_exit, y_out,
            src_conn='yTile',
            memlet=dace.Memlet.simple(y_out.data, "0:{}".format(n)) #  num_accesses=-1) # num_accesses=int(nTile))
        )

        gemv_sdfg.fill_scope_connectors()

        return gemv_sdfg





    @staticmethod
    def make_yTile(dtype, nTile, mTile, partialWidth, n, m, veclen, a, b):

        yTile_sdfg = dace.SDFG("yTile_sdfg")

        init_state = yTile_sdfg.add_state('yTile_init')
        compute_state = yTile_sdfg.add_state('yTile_compute')

        yTile_sdfg.add_symbol(a.name, a.dtype)
        if b != 0:
            yTile_sdfg.add_symbol(b.name, b.dtype)

        vec_type = dace.dtypes.vector(dtype, veclen)
        singleton_vec = dace.dtypes.vector(dtype, 1)
        A_in = compute_state.add_stream(
            '_A_yTile',
            vec_type,
            buffer_size=32,
            storage=dace.dtypes.StorageType.FPGA_Local
        )

        y_in = None
        if b != 0:
            y_in = compute_state.add_stream(
                '_y_yTile',
                singleton_vec,
                buffer_size=32,
                storage=dace.dtypes.StorageType.FPGA_Local
            )

        x_in = compute_state.add_stream(
            '_x_yTile',
            vec_type,
            buffer_size=32,
            storage=dace.dtypes.StorageType.FPGA_Local
        )

        yTile_sdfg.add_array('y_tileRes', shape=[nTile], dtype=dtype, storage=dace.dtypes.StorageType.FPGA_Local, transient=True)

        data_out = compute_state.add_stream(
            'yTile',
            singleton_vec,
            buffer_size=32,
            storage=dace.dtypes.StorageType.FPGA_Local
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
            schedule=dace.dtypes.ScheduleType.FPGA_Device
        )

        outerComputeMap_entry, outerComputeMap_exit = compute_state.add_map(
            'outerCompute_map',
            dict(ii = '0:{}'.format(nTile)),
            schedule=dace.dtypes.ScheduleType.FPGA_Device
        )

        y_out = compute_state.add_write('y_tileRes')

        reducedTile_sdfg = fpga_make_matrixPartialReduction(
            dtype,
            nTile,
            mTile,
            partialWidth,
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
            memlet=dace.Memlet.simple(A_in.data, "0:{}*{}".format(n, m))#, veclen=veclen)
        )

        compute_state.add_memlet_path(
            x_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
            dst_conn='_x_red',
            memlet=dace.Memlet.simple(x_in.data, "0:{}".format(m))#, veclen=veclen)
        )

        if b != 0:
            compute_state.add_memlet_path(
                y_in, mMap_entry, outerComputeMap_entry, nested_sdfg,
                dst_conn='_y_stream_red',
                memlet=dace.Memlet.simple(y_in.data, "0:{}".format(n))
            )

        compute_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, y_out,
            src_conn='_y_red',
            memlet=dace.Memlet.simple(y_out.data, "0:{}".format(nTile))
        )

        compute_state.add_memlet_path(
            nested_sdfg, outerComputeMap_exit, mMap_exit, data_out,
            src_conn='_res_red',
            memlet=dace.Memlet.simple(data_out.data, "0:{}".format(n))
        )



        yTile_sdfg.fill_scope_connectors()
        yTile_sdfg.add_edge(init_state, compute_state, dace.InterstateEdge(None))

        return yTile_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGemvFPGAStreamingRowTiles.make_sdfg(
            node.dtype,
            int(node.nTile),
            int(node.mTile),
            node.partialWidth,
            node.n,
            node.m,
            int(node.veclen),
            node.alpha,
            node.beta
        )

@dace.library.expansion
class ExpandGEMVIntelFPGAVectorized(ExpandTransformation):

    # Expansion targeting Intel FPGA
    # TODO: tiling

    environments = []

    @staticmethod
    def make_sdfg(node, dtype, parent_state, parent_sdfg):

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
        n = parent_state.in_edges(node)[0].data.subset.size()[0]
        m = parent_state.in_edges(node)[0].data.subset.size()[1]
        alpha = node.alpha
        beta = node.beta
        vec_width = node.vec_width
        transposed = node.transA
        tile_m_size = node.tile_m_size
        gemv_sdfg = dace.SDFG(
            "gemv{}_sdfg".format("_t_" if transposed else ""))

        gemv_state = gemv_sdfg.add_state(
            "gemv{}_state".format("_t" if transposed else ""))
        # ---------- ----------
        # MEMORY LOCATIONS
        # ---------- ----------
        A_rows = n
        A_cols = m
        x_size = n if transposed else m
        y_size = m if transposed else n

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
                                shape=[node.tile_m_size],
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
            nested_dot.add_symbol("m", dace.int32)

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
                'outCon = inCon + p_y * {} '.format(beta))

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
                                          A_in.data, "j*{}".format(vec_width)))
            dot_state.add_memlet_path(x_in,
                                      dotMap_entry,
                                      nested_sdfg,
                                      dst_conn="dot_y",
                                      memlet=dace.Memlet.simple(
                                          x_in.data, "j*{}".format(vec_width)))
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
                                      memlet=dace.Memlet.simple(
                                          res_write.data, "0"))

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
                                       memlet=dace.Memlet.simple(y_write.data,
                                                            "i",
                                                            num_accesses=1))

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
                dict(i='0:n'),
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
                'if i == 0: tile_y_in[(({} * j) + jj)] = {}*y_in \n'
                'tile_y_out = tile_y_in + A_con * x_con\n'
                'if i==n-1: y_out = tile_y_out'.format(vec_width, beta, vec_width,
                                                       vec_width))

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

            gemv_state.add_memlet_path(
                y_in,
                col_tile_map_entry,
                row_map_entry,
                col_map_entry,
                compute_map_entry,
                compute_tasklet,
                dst_conn='y_in',
                memlet=dace.Memlet("_y[tj*{} + j*{} + jj]".format(
                    tile_m_size, vec_width),
                                   dynamic=True)
            )
            gemv_state.add_memlet_path(tile_y_read,
                                       row_map_entry,
                                       col_map_entry,
                                       compute_map_entry,
                                       compute_tasklet,
                                       dst_conn='tile_y_in',
                                       memlet=dace.Memlet.simple(
                                           tile_y_read.data,
                                           'j*{}+jj'.format(vec_width)))

            ## update tile_y
            gemv_state.add_memlet_path(compute_tasklet,
                                       compute_map_exit,
                                       col_map_exit,
                                       row_map_exit,
                                       tile_y_write,
                                       src_conn='tile_y_out',
                                       memlet=dace.Memlet.simple(
                                           tile_y_write.data,
                                           'j*{}+jj'.format(vec_width)))

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
                                           y_out.data, 'tj*{}+j*{}+jj'.format(
                                               tile_m_size, vec_width)))




        gemv_sdfg.fill_scope_connectors()
        gemv_sdfg.validate()

        return gemv_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGEMVIntelFPGAVectorized.make_sdfg(node, node.dtype, state, sdfg)


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGemvPure,
        "fpga_stream": ExpandGemvFPGAStreamingRowTiles,
        "IntelFPGA": ExpandGEMVIntelFPGAVectorized
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    alpha = dace.properties.SymbolicProperty(
        allow_none=False, default=dace.symbolic.symbol("alpha"))
    beta = dace.properties.SymbolicProperty(
        allow_none=False, default=dace.symbolic.symbol("beta"))
    vec_width = dace.properties.SymbolicProperty(allow_none=False, default=1)


    transA = Property(dtype=bool,
                      desc="Whether to transpose A before multiplying")

    tile_n_size = dace.properties.Property(
        dtype=int, desc="Tile size along A rows")
    tile_m_size = dace.properties.Property(
        dtype=int, desc="Tile size along A columns")


    # FPGA
    nTile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    mTile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))

    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)
    partialWidth = dace.properties.SymbolicProperty(default=1, allow_none=False)


    def __init__(self,
                 name,
                 dtype=None,
                 location=None,
                 transA=False,
                 #alpha=1,
                 #beta=0,
                 n_tile=1,
                 m_tile=1,
                 partial_width=4,
                 n=dace.symbolic.symbol("n"),
                 m=dace.symbolic.symbol("m"),
                 veclen=1,
                 vec_width = 1,
                 alpha=dace.symbolic.symbol("alpha"),
                 beta=dace.symbolic.symbol("beta"),
                 ):

        # TODO from manuel:
        # FPGA ???
        # input_cons = {'_A', '_x'}
        # if b != 0:
        #     input_cons = {"_A", "_x", "_y"}
        #
        #super().__init__(name,
        #                 location=location,
        #                 inputs={"_a", "_x"},
        #                 outputs={"_y"})

        super().__init__(
            name,
            location=location,
            inputs={"_A", "_x", "_yi"} if beta != 0 else {"_A", "_x"},
            outputs={"_yo"})
        self.dtype = dtype
        self.transA = transA
        self.alpha = alpha
        self.beta = beta

        # FPGA
        self.n = n
        self.m = m
        self.nTile = n_tile
        self.mTile = m_tile
        self.veclen = veclen
        self.partialWidth = partial_width



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
            if dst_conn == "_yi":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y_in = subset.size()

        if self.transA:
            size_a = list(reversed(size_a))

        #if len(size_a) != 2 or len(size_x) != 1:
        #    raise ValueError(
        #            f"Matrix-vector product only supported on matrix-vector input. Got A: {size_a} and x : {size_x}")

        #if size_a[1] != size_x[0]:
        #    raise ValueError("Inputs to matrix-matrix product "
        #                     "must agree in the k-dimension")

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

    def compare(self, other):

        if (self.dtype == other.dtype and self.veclen == other.veclen
            and self.implementation == other.implementation
            and self.nTile == other.nTile and self.mTile == other.mTile):

            return True
        else:
            return False

    # Implementation dependent, extend if more implementations added
    def streamProductionLatency(self):

        return (self.m / self.mTile - 1) * self.mTile * self.nTile + self.mTile

    def streamConsumptionLatency(self):

        return {
            "_x": self.nTile,
            "_y": (self.m / self.mTile - 1) * self.mTile * self.nTile + self.mTile,
            "_A": 0
        }

