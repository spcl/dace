# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from dace.symbolic import symstr
from dace.properties import Property, SymbolicProperty
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace.sdfg.nodes import LibraryNode
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
import dace.library as library
from dace.sdfg import SDFG, SDFGState, nodes
from dace import memlet as mm, subsets as sbs
import dace
import copy
import numpy as np

import dace.library
import dace.properties
import dace.sdfg.nodes

from dace import dtypes
from dace.memlet import Memlet

from dace.libraries.blas.utility.initialization import fpga_init_array
from dace.libraries.blas.utility.memory_operations import fpga_stream_to_local

from dace.libraries.blas.utility.fpga_helper import StreamReadVector
from dace.libraries.blas.utility.fpga_helper import StreamReadMatrixFull, StreamWriteMatrixFull


@library.expansion
class ExpandGerPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        inputs = ('_A', '_x', '_y')
        outputs = ('_res', )
        in_edges = [
            next(parent_state.in_edges_by_connector(node, conn))
            for conn in inputs
        ]
        out_edges = [
            next(parent_state.out_edges_by_connector(node, conn))
            for conn in outputs
        ]
        arrays = {}
        arrays.update({
            inp: parent_sdfg.arrays[e.data.data]
            for inp, e in zip(inputs, in_edges)
        })
        arrays.update({
            out: parent_sdfg.arrays[e.data.data]
            for out, e in zip(outputs, out_edges)
        })

        # TODO: Support memlet subsets
        if any(e.data.subset != sbs.Range.from_array(arrays[a])
               for a, e in zip(inputs, in_edges)):
            raise NotImplementedError
        if any(e.data.subset != sbs.Range.from_array(arrays[a])
               for a, e in zip(outputs, out_edges)):
            raise NotImplementedError

        sdfg = dace.SDFG(f'{node.label}_sdfg')
        sdfg.add_symbol('M', int)
        sdfg.add_symbol('N', int)
        sdfg.add_symbol('alpha', arrays['_A'].dtype)

        for name, desc in arrays.items():
            newdesc = copy.deepcopy(desc)
            newdesc.transient = False
            sdfg.add_datadesc(name, newdesc)


        state = sdfg.add_state()
        state.add_mapped_tasklet(
            'ger',
            {
                '_i': f'0:M',
                '_j': f'0:N'
            },
            {
                'a': mm.Memlet('_A[_i, _j]'),
                'xin': mm.Memlet('_x[_i]'),
                'yin': mm.Memlet(f'_y[_j]')
            },
            f'aout = alpha * xin * yin + a',
            {'aout': mm.Memlet('_res[_i, _j]')},
            external_edges=True,
        )

        outshape = arrays['_res'].shape
        nsdfg_node = nodes.NestedSDFG(node.label, sdfg, set(inputs),
                                      set(outputs), {
                                          'M': outshape[0],
                                          'N': outshape[1],
                                          'alpha': node.alpha
                                      })

        return nsdfg_node

    @staticmethod
    def expansion(node, state, sdfg, **kwargs):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGerPure.make_sdfg(node, state, sdfg)




# @dace.library.expansion
# class Expand_GER_FPGA_Streaming_Row_tiles(ExpandTransformation):

#     environments = []

#     @staticmethod
#     def make_sdfg(dtype, n_tile, m_tile, tileRowStreamed, n, m, veclen, a):

#         # ---------- ----------
#         # SETUP GRAPH
#         # ---------- ----------
#         # A: m rows, n columns, row-major (or transposed column-major)

#         ger_sdfg = dace.SDFG("ger_fpga_stream_rowTiles")

#         ger_sdfg.add_symbol(a.name, a.dtype)
#         ger_state = ger_sdfg.add_state('ger_compute')

#         A_in = ger_state.add_stream(
#             '_A',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         # Must be received m/m_tiles times
#         y_in = ger_state.add_stream(
#             '_y',
#             dtype,
#             veclen=1,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         x_in = ger_state.add_stream(
#             '_x',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         res = ger_state.add_stream(
#             '_RES',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )


#         ger_sdfg.add_array('x_buf_row', shape=[m_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

#         # ---------- ----------
#         # COMPUTE
#         # ---------- ----------
#         x_buf = ger_state.add_write('x_buf_row')

#         mMap_entry, mMap_exit = ger_state.add_map(
#             'm_block_map',
#             dict(j = '0:{}/{}'.format(m, m_tile)),
#             schedule=dtypes.ScheduleType.FPGA_Device
#         )

#         nMap_entry, nMap_exit = ger_state.add_map(
#             'n_tile_map',
#             dict(i = '0:{}/{}'.format(n, n_tile)),
#             schedule=dtypes.ScheduleType.FPGA_Device
#         )

#         outerComputeMap_entry, outerComputeMap_exit = ger_state.add_map(
#             'outerCompute_map',
#             dict(jj = '0:{}'.format(m_tile)),
#             schedule=dtypes.ScheduleType.FPGA_Device
#         )

#         tile_sdfg = None
#         tile_sdfg = Expand_GER_FPGA_Streaming_Row_tiles.make_computeTileRowStreamed(
#             dtype,
#             n_tile,
#             m_tile,
#             n, m,
#             veclen,
#             a
#         )

#         nested_sdfg = ger_state.add_nested_sdfg(
#             tile_sdfg,
#             ger_sdfg,
#             {'_A_tile', '_x_tile', '_y_tile'}, 
#             {'_A_out_tile', '_x_buf_tile'}
#         )

#         ger_state.add_memlet_path(
#             A_in, mMap_entry, nMap_entry, outerComputeMap_entry, nested_sdfg,
#             dst_conn='_A_tile',
#             memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m), veclen=veclen)
#         )

#         ger_state.add_memlet_path(
#             x_in, mMap_entry, nMap_entry, outerComputeMap_entry, nested_sdfg,
#             dst_conn='_x_tile',
#             memlet=Memlet.simple(x_in.data, "0:{}".format(m))
#         )

#         ger_state.add_memlet_path(
#             y_in, mMap_entry, nMap_entry, outerComputeMap_entry, nested_sdfg,
#             dst_conn='_y_tile',
#             memlet=Memlet.simple(y_in.data, "0:{}".format(n), veclen=veclen)
#         )

#         ger_state.add_memlet_path(
#             nested_sdfg, outerComputeMap_exit, mMap_exit, nMap_exit, res,
#             src_conn='_A_out_tile',
#             memlet=Memlet.simple(res.data, "0:{}*{}".format(n, m), veclen=veclen)
#         )

#         ger_state.add_memlet_path(
#             nested_sdfg, outerComputeMap_exit, mMap_exit, nMap_exit, x_buf,
#             src_conn='_x_buf_tile',
#             memlet=Memlet.simple(x_buf.data, "0:{}".format(m_tile))
#         )

#         return ger_sdfg



#     @staticmethod
#     def make_computeTileRowStreamed(dtype, n_tile, m_tile, n, m, veclen, a):

#         tile_sdfg = dace.SDFG("tile_sdfg")
#         tile_sdfg.add_symbol(a.name, a.dtype)


#         init_state = tile_sdfg.add_state('init_state_tile')
#         compute_state = tile_sdfg.add_state('copmute_state_tile')

#         read_x_state = tile_sdfg.add_state('read_x_reduceTile')
#         read_empty_state =  tile_sdfg.add_state('read_empty_reduceTile')

#         A_in = compute_state.add_stream(
#             '_A_tile',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         x_in = compute_state.add_stream(
#             '_x_tile',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         y_in = read_y_state.add_stream(
#             '_y_tile',
#             dtype,
#             veclen=1,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )
        
#         tile_sdfg.add_array('_x_buf_tile', shape=[m_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)
#         tile_sdfg.add_array('x_buf', shape=[1], dtype=dtype, storage=dtypes.StorageType.FPGA_Registers, transient=True)
#         tile_sdfg.add_array('y_buf', shape=[n_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)


#         A_out = compute_state.add_stream(
#             '_A_out_tile',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )


#         # ---------- ----------
#         # INIT State
#         # ---------- ----------
#         # fpga_streamToLocal(
#         #     init_state,
#         #     x_in,
#         #     'x_buf',
#         #     m_tile
#         # )

#         x_out = read_x_state.add_write("_x_buf_tile")
#         x_buf = read_x_state.add_access("x_buf")

#         read_x_state.add_memlet_path(
#             x_in, x_buf,
#             memlet=Memlet.simple(x_buf.data, "0")
#         )

#         read_x_state.add_memlet_path(
#             x_buf, x_out,
#             memlet=Memlet.simple(x_buf.data, "0", other_subset_str="jj")
#         )

#         tile_sdfg.add_edge(init_state, read_x_state, dace.InterstateEdge("i == 0"))
#         tile_sdfg.add_edge(read_x_state, compute_state, dace.InterstateEdge(None))
#         tile_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("i != 0"))
#         tile_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))



#         # ---------- ----------
#         # COMPUTE
#         # ---------- ----------
#         x_in = compute_state.add_read('_x_buf_tile')
#         y_buf = compute_state.add_write('y_buf')

#         innerComputeMap_entry, innerComputeMap_exit = compute_state.add_map(
#             'innerCompute_map',
#             dict(ii = '0:{}/{}'.format(n_tile, veclen)),
#             schedule=dtypes.ScheduleType.FPGA_Device,
#         )

#         red_sdfg = Expand_GER_FPGA_Streaming_Row_tiles.make_unrolledCompute(
#             dtype,
#             n_tile,
#             m_tile,
#             n, m,
#             veclen,
#             a
#         )

#         nested_sdfg = compute_state.add_nested_sdfg(
#             red_sdfg,
#             tile_sdfg,
#             {'_A_unroll', '_x_unroll', '_y_unroll'},
#             {'_A_out_unroll', '_y_buf_unroll'}
#         )

#         compute_state.add_memlet_path(
#             A_in, innerComputeMap_entry, nested_sdfg,
#             dst_conn='_A_unroll',
#             memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m), veclen=veclen)
#         )

#         compute_state.add_memlet_path(
#             y_in, innerComputeMap_entry, nested_sdfg,
#             dst_conn='_y_unroll',
#             memlet=Memlet.simple(y_in.data, "0:{}".format(n), veclen=veclen)
#         )

#         compute_state.add_memlet_path(
#             x_in, innerComputeMap_entry, nested_sdfg,
#             dst_conn='_x_unroll',
#             memlet=Memlet.simple(x_in.data, "0:{}".format(m_tile))
#         )

#         compute_state.add_memlet_path(
#             nested_sdfg, innerComputeMap_exit, A_out,
#             src_conn='_A_out_unroll',
#             memlet=Memlet.simple(A_out.data, "0:{}*{}".format(n ,m), veclen=veclen)
#         )

#         compute_state.add_memlet_path(
#             nested_sdfg, innerComputeMap_exit, y_buf,
#             src_conn='_y_buf_unroll',
#             memlet=Memlet.simple(y_buf.data, "0:{}".format(n_tile), veclen=veclen)
#         )


#         return tile_sdfg


    
#     @staticmethod
#     def make_unrolledCompute(dtype, n_tile, m_tile, n, m, veclen, a):

#         inner_sdfg = dace.SDFG("vectorize_inner_graph")
#         inner_sdfg.add_symbol(a.name, a.dtype)

#         init_state = inner_sdfg.add_state("init_state")
#         compute_state = inner_sdfg.add_state('compute_state')

#         read_y_state = inner_sdfg.add_state("read_y_state")
#         read_empty_state = inner_sdfg.add_state("readEmpty_state")

#         stream_out_state = inner_sdfg.add_state('streamOut_state')


#         # ---------- ----------
#         # DATA DEFN
#         # ---------- ----------
#         # A_in = init_state.add_stream(
#         #     '_A_unroll',
#         #     dtype,
#         #     veclen=veclen,
#         #     buffer_size=32,
#         #     storage=dtypes.StorageType.FPGA_Local
#         # )

#         A_in = compute_state.add_stream(
#             '_A_unroll',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         y_in = read_y_state.add_stream(
#             '_y_unroll',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         A_out = stream_out_state.add_stream(
#             '_A_out_unroll',
#             dtype,
#             veclen=veclen,
#             buffer_size=32,
#             storage=dtypes.StorageType.FPGA_Local
#         )

#         inner_sdfg.add_array('_x_unroll', shape=[m_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)
#         inner_sdfg.add_array('_y_buf_unroll', shape=[n_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)

#         inner_sdfg.add_array('A_out_buf', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
#         inner_sdfg.add_array('A_vec_buf', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
#         inner_sdfg.add_array('A_buf', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
#         inner_sdfg.add_array('y_vecbuf', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
#         inner_sdfg.add_array('y_membuf', shape=[veclen], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

#         # ---------- ----------
#         # GET DATA
#         # ---------- ---------
#         data_out = read_y_state.add_write('_y_buf_unroll')

#         copy_y_task = read_y_state.add_tasklet(
#             'streamToLocal_map',
#             ['inCon'],
#             ['outCon'],
#             'outCon = inCon'
#         )

#         read_y_state.add_memlet_path(
#             y_in, copy_y_task,
#             dst_conn="inCon",
#             memlet=Memlet.simple(y_in.data, "0", veclen=veclen)
#         )

#         read_y_state.add_memlet_path(
#             copy_y_task, data_out,
#             src_conn="outCon",
#             memlet=Memlet.simple(data_out.data, "ii * {}".format(veclen), veclen=veclen)
#         )

        

#         inner_sdfg.add_edge(init_state, read_y_state, dace.InterstateEdge("jj == 0"))
#         inner_sdfg.add_edge(read_y_state, compute_state, dace.InterstateEdge(None))   

#         inner_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("jj != 0"))
#         inner_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))     



#         # ---------- ----------
#         # COMPUTE
#         # ---------- ---------
#         y_in = compute_state.add_read("_y_buf_unroll")
#         x_in = compute_state.add_read("_x_unroll")
#         A_out_buf = compute_state.add_write("A_out_buf")


#         compute_task = compute_state.add_tasklet(
#             'compute_task',
#             ['A_con', 'x_con', 'y_con'],
#             ['outCon'],
#             'outCon = A_con + x_con * y_con * {}'.format(a)
#         )

#         compute_state.add_memlet_path(
#             A_in, compute_task,
#             dst_conn='A_con',
#             memlet=Memlet.simple(A_in.data, "0", veclen=veclen)
#         )

#         compute_state.add_memlet_path(
#             y_in, compute_task,
#             dst_conn='y_con',
#             memlet=Memlet.simple(y_in.data, "ii * {}".format(veclen), veclen=veclen)
#         )

#         compute_state.add_memlet_path(
#             x_in, compute_task,
#             dst_conn='x_con',
#             memlet=Memlet.simple(x_in.data, "jj")
#         )

#         compute_state.add_memlet_path(
#             compute_task, A_out_buf,
#             src_conn='outCon',
#             memlet=Memlet.simple(A_out_buf.data, "0", veclen=veclen)
#         )

#         # ---------- ----------
#         # STREAM RESULT
#         # ---------- ---------
#         A_out_buf = stream_out_state.add_read('A_out_buf')
        
#         stream_out_state.add_memlet_path(
#             A_out_buf, A_out,
#             memlet=Memlet.simple(A_out.data, "0", veclen=veclen)
#         )

#         inner_sdfg.add_edge(compute_state, stream_out_state, dace.InterstateEdge(None))

#         return inner_sdfg


#     @staticmethod
#     def expansion(node, state, sdfg):
#         node.validate(sdfg, state)
#         if node.dtype is None:
#             raise ValueError("Data type must be set to expand " + str(node) + ".")
#         return Expand_GER_FPGA_Streaming_Row_tiles.make_sdfg(
#             node.dtype,
#             node.n_tile,
#             node.m_tile,
#             node.tileRowStreamed,
#             node.n,
#             node.m,
#             int(node.veclen),
#             node.a
#         )











@dace.library.expansion
class Expand_GER_FPGA_Streaming_Col_tiles(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, n_tile, m_tile, n, m, veclen, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        # A: m rows, n columns, row-major
        ger_sdfg = dace.SDFG("ger_fpga_stream_rowTiles")

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        ger_sdfg.add_symbol(a.name, a.dtype)
        ger_state = ger_sdfg.add_state('ger_compute')

        A_in = ger_state.add_stream(
            '_A',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = ger_state.add_stream(
            '_y',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        # Must be received n/n_tiles times
        x_in = ger_state.add_stream(
            '_x',
            single_vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        res = ger_state.add_stream(
            '_res',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        ger_sdfg.add_array('y_buf_row', shape=[n_tile/veclen], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)
        ger_sdfg.add_array('x_buf', shape=[m_tile], dtype=single_vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        y_buf = ger_state.add_write('y_buf_row')
        x_buf = ger_state.add_write('x_buf')

        nMap_entry, nMap_exit = ger_state.add_map(
            'n_tile_map',
            dict(i = '0:{}/{}'.format(n, n_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        mMap_entry, mMap_exit = ger_state.add_map(
            'm_block_map',
            dict(j = '0:{}/{}'.format(m, m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        outerComputeMap_entry, outerComputeMap_exit = ger_state.add_map(
            'outerCompute_map',
            dict(jj = '0:{}'.format(m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        innerComputeMap_entry, innerComputeMap_exit = ger_state.add_map(
            'innerCompute_map',
            dict(ii = '0:{}/{}'.format(n_tile, veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device,
        )

        tile_sdfg = None
        tile_sdfg = Expand_GER_FPGA_Streaming_Col_tiles.make_unrolledCompute(
            dtype,
            n_tile,
            m_tile,
            n, m,
            veclen,
            a
        )

        nested_sdfg = ger_state.add_nested_sdfg(
            tile_sdfg,
            ger_sdfg,
            {'_A_unroll', '_x_stream', '_y_stream'}, 
            {'_A_out_unroll', '_y_unroll', '_x_buf_unroll'}
        )

        ger_state.add_memlet_path(
            A_in, nMap_entry, mMap_entry, outerComputeMap_entry, innerComputeMap_entry, nested_sdfg,
            dst_conn='_A_unroll',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m))
        )

        ger_state.add_memlet_path(
            x_in, nMap_entry, mMap_entry, outerComputeMap_entry, innerComputeMap_entry, nested_sdfg,
            dst_conn='_x_stream',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m))
        )

        ger_state.add_memlet_path(
            y_in, nMap_entry, mMap_entry, outerComputeMap_entry, innerComputeMap_entry, nested_sdfg,
            dst_conn='_y_stream',
            memlet=Memlet.simple(y_in.data, "0:{}".format(n))
        )

        ger_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, outerComputeMap_exit, mMap_exit, nMap_exit, res,
            src_conn='_A_out_unroll',
            memlet=Memlet.simple(res.data, "0:{}*{}".format(n, m))
        )

        ger_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, outerComputeMap_exit, mMap_exit, nMap_exit, y_buf,
            src_conn='_y_unroll',
            memlet=Memlet.simple(y_buf.data, "0:{}".format(n_tile/veclen))
        )

        ger_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, outerComputeMap_exit, mMap_exit, nMap_exit, x_buf,
            src_conn='_x_buf_unroll',
            memlet=Memlet.simple(x_buf.data, "0:{}".format(m_tile))
        )

        return ger_sdfg



    @staticmethod
    def make_computeTileRowStreamed(dtype, n_tile, m_tile, n, m, veclen, a):

        tile_sdfg = dace.SDFG("tile_sdfg")
        tile_sdfg.add_symbol(a.name, a.dtype)

        compute_state = tile_sdfg.add_state('copmute_state_tile')

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        A_in = compute_state.add_stream(
            '_A_tile',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = compute_state.add_stream(
            '_x_tile',
            single_vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = compute_state.add_stream(
            '_y_tile',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )
        
        tile_sdfg.add_array('_y_buf_tile', shape=[n_tile], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local)


        A_out = compute_state.add_stream(
            '_A_out_tile',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )


        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        y_buf = compute_state.add_read('_y_buf_tile')
        x_buf = compute_state.add_write('x_buf')



        red_sdfg = Expand_GER_FPGA_Streaming_Col_tiles.make_unrolledCompute(
            dtype,
            n_tile,
            m_tile,
            n, m,
            veclen,
            a
        )

        nested_sdfg = compute_state.add_nested_sdfg(
            red_sdfg,
            tile_sdfg,
            {'_A_unroll', '_x_unroll', '_y_stream'},
            {'_A_out_unroll', '_x_buf_unroll', '_y_unroll'}
        )

        compute_state.add_memlet_path(
            A_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_A_unroll',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m))
        )

        compute_state.add_memlet_path(
            x_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_x_unroll',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m))
        )

        compute_state.add_memlet_path(
            y_in, innerComputeMap_entry, nested_sdfg,
            dst_conn='_y_stream',
            memlet=Memlet.simple(y_in.data, "0:{}".format(n))
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_entry, y_buf,
            src_conn='_y_unroll',
            memlet=Memlet.simple(y_buf.data, "0:{}".format(n_tile))
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, A_out,
            src_conn='_A_out_unroll',
            memlet=Memlet.simple(A_out.data, "0:{}*{}".format(n ,m))
        )

        compute_state.add_memlet_path(
            nested_sdfg, innerComputeMap_exit, x_buf,
            src_conn='_x_buf_unroll',
            memlet=Memlet.simple(x_buf.data, "0:{}".format(m_tile))
        )


        return tile_sdfg


    
    @staticmethod
    def make_unrolledCompute(dtype, n_tile, m_tile, n, m, veclen, a):

        inner_sdfg = dace.SDFG("vectorize_inner_graph")
        inner_sdfg.add_symbol(a.name, a.dtype)

        init_state = inner_sdfg.add_state("init_state")
        compute_state = inner_sdfg.add_state('compute_state')

        read_x_state = inner_sdfg.add_state("read_x_state")
        read_empty_state = inner_sdfg.add_state("readEmpty_state")

        stream_out_state = inner_sdfg.add_state('streamOut_state')


        init_state_y = inner_sdfg.add_state('init_state_tile')

        read_y_state = inner_sdfg.add_state('read_y_reduceTile')
        read_empty_state_y =  inner_sdfg.add_state('read_empty_reduceTile')


        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)


        # ---------- ----------
        # DATA DEFN
        # ---------- ----------
        # A_in = init_state.add_stream(
        #     '_A_unroll',
        #     dtype,
        #     veclen=veclen,
        #     buffer_size=32,
        #     storage=dtypes.StorageType.FPGA_Local
        # )

        A_in = compute_state.add_stream(
            '_A_unroll',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = read_y_state.add_stream(
            '_y_stream',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = read_x_state.add_stream(
            '_x_stream',
            single_vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        A_out = stream_out_state.add_stream(
            '_A_out_unroll',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        inner_sdfg.add_array('_y_unroll', shape=[n_tile/veclen], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local)
        inner_sdfg.add_array('_x_buf_unroll', shape=[m_tile], dtype=single_vec_type, storage=dtypes.StorageType.FPGA_Local)

        inner_sdfg.add_array('A_out_buf', shape=[1], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)
        # inner_sdfg.add_array('A_buf', shape=[veclen], dtype=vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)
        inner_sdfg.add_array('x_vecbuf', shape=[veclen], dtype=single_vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)
        inner_sdfg.add_array('x_membuf', shape=[veclen], dtype=single_vec_type, storage=dtypes.StorageType.FPGA_Local, transient=True)

        inner_sdfg.add_array('y_buf', shape=[1], dtype=vec_type, storage=dtypes.StorageType.FPGA_Registers, transient=True)

        # ---------- ----------
        # GET DATA x
        # ---------- ---------
        data_out = read_x_state.add_write('_x_buf_unroll')

        copy_x_task = read_x_state.add_tasklet(
            'streamToLocal_map',
            ['inCon'],
            ['outCon'],
            'outCon = inCon'
        )

        read_x_state.add_memlet_path(
            x_in, copy_x_task,
            dst_conn="inCon",
            memlet=Memlet.simple(x_in.data, "0")
        )

        read_x_state.add_memlet_path(
            copy_x_task, data_out,
            src_conn="outCon",
            memlet=Memlet.simple(data_out.data, "jj")
        )

        

        inner_sdfg.add_edge(init_state, read_x_state, dace.InterstateEdge("ii == 0"))
        inner_sdfg.add_edge(read_x_state, init_state_y, dace.InterstateEdge(None))   

        inner_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge("ii != 0"))
        inner_sdfg.add_edge(read_empty_state, init_state_y, dace.InterstateEdge(None))     




        # ---------- ----------
        # Get data y
        # ---------- ----------
        y_out = read_y_state.add_write("_y_unroll")
        # y_buf = read_y_state.add_access("y_buf")

        # read_y_state.add_memlet_path(
        #     y_in, y_buf,
        #     memlet=Memlet.simple(y_buf.data, "0")
        # )

        # read_y_state.add_memlet_path(
        #     y_buf, y_out,
        #     memlet=Memlet.simple(y_buf.data, "0", other_subset_str="ii")
        # )

        read_y_state.add_memlet_path(
            y_in, y_out,
            memlet=Memlet.simple(y_in.data, "0", other_subset_str="ii")
        )

        inner_sdfg.add_edge(init_state_y, read_y_state, dace.InterstateEdge("j == 0 and jj == 0"))
        inner_sdfg.add_edge(read_y_state, compute_state, dace.InterstateEdge(None))
        inner_sdfg.add_edge(init_state_y, read_empty_state_y, dace.InterstateEdge("not (j == 0 and jj == 0)"))
        inner_sdfg.add_edge(read_empty_state_y, compute_state, dace.InterstateEdge(None))



        # ---------- ----------
        # COMPUTE
        # ---------- ---------
        x_in = compute_state.add_read("_x_buf_unroll")
        y_in = compute_state.add_read("_y_unroll")
        A_out_buf = compute_state.add_write("A_out_buf")


        compute_task = compute_state.add_tasklet(
            'compute_task',
            ['A_con', 'x_con', 'y_con'],
            ['outCon'],
            'outCon = A_con + x_con * y_con * {}'.format(a)
        )

        compute_state.add_memlet_path(
            A_in, compute_task,
            dst_conn='A_con',
            memlet=Memlet.simple(A_in.data, "0")
        )

        compute_state.add_memlet_path(
            y_in, compute_task,
            dst_conn='y_con',
            memlet=Memlet.simple(y_in.data, "ii")
        )

        compute_state.add_memlet_path(
            x_in, compute_task,
            dst_conn='x_con',
            memlet=Memlet.simple(x_in.data, "jj")
        )

        compute_state.add_memlet_path(
            compute_task, A_out_buf,
            src_conn='outCon',
            memlet=Memlet.simple(A_out_buf.data, "0")
        )

        # ---------- ----------
        # STREAM RESULT
        # ---------- ---------
        A_out_buf = stream_out_state.add_read('A_out_buf')
        
        stream_out_state.add_memlet_path(
            A_out_buf, A_out,
            memlet=Memlet.simple(A_out.data, "0", other_subset_str="0")
        )

        inner_sdfg.add_edge(compute_state, stream_out_state, dace.InterstateEdge(None))

        return inner_sdfg


    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return Expand_GER_FPGA_Streaming_Col_tiles.make_sdfg(
            node.dtype,
            node.n_tile,
            node.m_tile,
            node.n,
            node.m,
            int(node.veclen),
            node.a
        )





@library.node
class Ger(LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
        # "fpga_stream_row": Expand_GER_FPGA_Streaming_Row_tiles,
        "fpga_stream_column": Expand_GER_FPGA_Streaming_Col_tiles,
        "IntelFPGA": ExpandGerPure
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    n_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    m_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))
    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"))

    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)

    alpha = SymbolicProperty(
        default=1,
        desc=
        "A scalar which will be multiplied with the outer product x*yT before adding matrix A"
    )

    def __init__(self, name,
        dtype=dace.float32,
        n_tile=1, m_tile=1,
        n=dace.symbolic.symbol("n"),
        m=dace.symbolic.symbol("m"),
        a=dace.symbolic.symbol("a"),
        veclen=1,
        alpha=1,
        location=None,
        *args, **kwargs
        ):
        super().__init__(name,
                         location=location,
                         inputs={"_x", "_y", "_A"},
                         outputs={"_res"})
        self.dtype = dtype

        self.n_tile = n_tile
        self.m_tile = m_tile

        self.veclen = veclen

        self.n = n
        self.m = m
        self.a = a

        self.alpha = alpha

    def compare(self, other):

        if (self.dtype == other.dtype and self.veclen == other.veclen
            and self.implementation == other.implementation
            and self.n_tile == other.n_tile and self.m_tile == other.m_tile):

            return True
        else:
            return False

    def validate(self, sdfg, state):

        in_edges = state.in_edges(self)
        if len(in_edges) != 3:
            raise ValueError(
                "Expected exactly three inputs to the ger operation (vectors x, y and matrix A)"
            )

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from the ger operation")

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
                size_y = subset.size()

        # TODO these checks aren't working with streams
        #if len(size_a) != 2:
        #    raise ValueError("A must be a matrix")
        #if len(size_x) != 1:
        #    raise ValueError("x must be a vector")
        #if len(size_y) != 1:
        #    raise ValueError("y must be a vector")

        #if size_a[0] != size_x[0] or size_a[1] != size_y[0]:
        #    raise ValueError(
        #        "Input vectors x and y (outer product) must match with the matrix A dimensions."
        #    )

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from ger rank 1 operation.")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()

        # TODO: not working with streaming
        # if (len(size_out) != 2 or size_out[0] != size_a[0]
        #         or size_out[1] != size_a[1]):
        #     raise ValueError(
        #         "Output matrix must match input matrix a and outer product x*yT."
        #     )


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.ger')
@oprepo.replaces('dace.libraries.blas.Ger')
def ger_libnode(sdfg: SDFG, state: SDFGState, A, x, y, output, alpha):
    # Add nodes
    A_in, x_in, y_in = (state.add_read(name) for name in (A, x, y))
    out = state.add_write(output)

    libnode = Ger('ger', dtype=sdfg.arrays[A].dtype, alpha=alpha)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_res', out, None, mm.Memlet(output))

    return []
