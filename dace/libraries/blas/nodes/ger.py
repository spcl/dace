# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from dace.symbolic import symstr
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from dace.sdfg.nodes import LibraryNode
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
import dace.sdfg.nodes
import dace.library as library
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

        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_x, outer_array_x, shape_x, strides_x), (edge_y, outer_array_y,
                                                       shape_y, strides_y),
         cdata) = _get_matmul_operands(node,
                                       parent_state,
                                       parent_sdfg,
                                       name_lhs="_x",
                                       name_rhs="_y",
                                       name_out="_res")

        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_x, dtype_y).type]

        if (len(shape_x) != 1 or len(shape_y) != 1):
            raise SyntaxError("Vectors must have single dimension")

        M, N = shape_x[0], shape_y[0]
        shape_c = (M, N)

        if outer_array_x.storage != outer_array_y.storage:
            raise ValueError("Input vectors must have same storage.")
        storage = outer_array_x.storage

        _, array_x = sdfg.add_array("_x",
                                    shape_x,
                                    dtype_x,
                                    strides=strides_x,
                                    storage=storage)
        _, array_y = sdfg.add_array("_y",
                                    shape_y,
                                    dtype_y,
                                    strides=strides_y,
                                    storage=storage)
        _, array_a = sdfg.add_array("_A", shape_c, dtype_c, storage=storage)
        _, array_res = sdfg.add_array("_res", shape_c, dtype_c, storage=storage)

        if node.alpha == 1.0:
            mul_program = "__out = __x * __y"
        else:
            mul_program = "__out = {} * __x * __y".format(node.alpha)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        # prepare beta*C
        mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(
            shape_c, dtype_c, storage=storage)
        access_tmp = state.add_read(tmp)
        output_nodes = {mul_out: access_tmp}

        # init map
        init_state.add_mapped_tasklet(
            '_GER_init',
            {'_o%d' % i: '0:%s' % symstr(d)
             for i, d in enumerate(shape_c)}, {},
            'out = 0', {
                'out':
                dace.Memlet.simple(
                    mul_out, ','.join(['_o%d' % i
                                       for i in range(len(shape_c))]))
            },
            external_edges=True)

        # outer product map
        state.add_mapped_tasklet(
            "_GER_outer_",
            {"__i%d" % i: "0:%s" % s
             for i, s in enumerate([M, N])}, {
                 "__x": dace.Memlet.simple("_x", "__i0"),
                 "__y": dace.Memlet.simple("_y", "__i1")
             },
            mul_program, {
                "__out":
                dace.Memlet.simple(
                    mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        add_program = "__res = __A + __tmp"

        # manually broadcasting C to [M, N]
        if list(shape_c) == [M, N]:
            memlet_idx = '__i0, __i1'
        elif list(shape_c) == [1, N]:
            memlet_idx = '0, __i1'
        elif list(shape_c) == [M, 1]:
            memlet_idx = '__i0, 0'
        elif list(shape_c) == [N]:
            memlet_idx = '__i1'
        else:
            raise ValueError(
                "Could not broadcast input _res to ({}, {})".format(M, N))

        # addition map
        state.add_mapped_tasklet(
            "_GER_add_",
            {"__i%d" % i: "0:%s" % s
             for i, s in enumerate([M, N])}, {
                 "__A": dace.Memlet.simple("_A", memlet_idx),
                 "__tmp": dace.Memlet.simple(mul_out, "__i0, __i1"),
             },
            add_program, {"__res": dace.Memlet.simple("_res", "__i0, __i1")},
            external_edges=True,
            input_nodes={mul_out: access_tmp})

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGerPure.make_sdfg(node, state, sdfg)




@dace.library.expansion
class Expand_GER_FPGA_Streaming_RowTiles(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, n_tile, m_tile, n, m, veclen, a):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        # A: n rows, m columns, row-major (or transposed column-major)

        ger_sdfg = dace.SDFG("ger_fpga_stream_rowTiles")

        ger_sdfg.add_symbol(a.name, a.dtype)
        ger_state = ger_sdfg.add_state('ger_compute')

        vec_type = dtypes.vector(dtype, veclen)
        singleton_vec = dtypes.vector(dtype, 1)
        A_in = ger_state.add_stream(
            '_A',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = ger_state.add_stream(
            '_y',
            singleton_vec,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        # Must be received n/n_tiles times
        x_in = ger_state.add_stream(
            '_x',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        res = ger_state.add_stream(
            '_res',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        # ---------- ----------
        # Read X for every N block
        # ---------- ----------
        n_map_entry, n_map_exit = ger_state.add_map(
            'n_block_map',
            dict(ti = '0:{}/{}'.format(n, n_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        #tile_sdfg = None
        #if tileRowStreamed:
        tile_sdfg = Expand_GER_FPGA_Streaming_RowTiles.make_computeTileRowStreamed(
            dtype,
            n_tile,
            m_tile,
            n, m,
            veclen,
            a
        )
        #else:

        #    raise ValueError("Not supported a.t.m")

        #    tile_sdfg = Expand_GER_FPGA_Streaming_RowTiles.make_computeTileColStreamed(
        #        dtype,
        #        n_tile,
        #        m_tile,
        #        n, m
        #    )

        nested_sdfg = ger_state.add_nested_sdfg(
            tile_sdfg,
            ger_sdfg,
            {'_A', '_x', '_y'},
            {'_A_out'}
        )

        ger_state.add_memlet_path(
            A_in, n_map_entry, nested_sdfg,
            dst_conn='_A',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(m, n))
        )

        ger_state.add_memlet_path(
            x_in, n_map_entry, nested_sdfg,
            dst_conn='_x',
            memlet=Memlet.simple(x_in.data, "0:{}".format(m))
        )

        ger_state.add_memlet_path(
            y_in, n_map_entry, nested_sdfg,
            dst_conn='_y',
            memlet=Memlet.simple(y_in.data, "0:{}".format(n))
        )

        ger_state.add_memlet_path(
            nested_sdfg, n_map_exit, res,
            src_conn='_A_out',
            memlet=Memlet.simple(res.data, "0:{}*{}".format(m, n))
        )

        return ger_sdfg



    @staticmethod
    def make_computeTileRowStreamed(dtype, n_tile, m_tile, n, m, veclen, a):
        tile_sdfg = dace.SDFG("tile_sdfg")
        tile_sdfg.add_symbol(a.name, a.dtype)

        read_x_state = tile_sdfg.add_state('read_x')
        compute_state = tile_sdfg.add_state('compute_state_tile')

        vec_type = dtypes.vector(dtype, veclen)
        singleton_vec = dtypes.vector(dtype, 1)
        A_in = compute_state.add_stream(
            '_A',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = compute_state.add_stream(
            '_y',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        x_in = read_x_state.add_stream(
            '_x',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        A_out = compute_state.add_stream(
            '_A_out',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        tile_sdfg.add_array('y_buf', shape=[m_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        y_buf = compute_state.add_write('y_buf')

        #
        # Read X
        #
        tile_sdfg.add_array('x_buf', shape=[n_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)
        read_x_entry, read_x_exit = read_x_state.add_map(
            'read_x_map',
            dict(i = '0:{}'.format(n_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )
        read_x_buf = read_x_state.add_write('x_buf')
        read_x_task = read_x_state.add_tasklet('read_x_task',
            ['in_con'],
            ['out_con'],
            'out_con = in_con'
        )
        read_x_state.add_memlet_path(
            x_in, read_x_entry, read_x_task,
            dst_conn='in_con',
            memlet=Memlet.simple(x_in.data, '0')
        )
        read_x_state.add_memlet_path(
            read_x_task, read_x_exit, read_x_buf,
            src_conn='out_con',
            memlet=Memlet.simple(read_x_buf.data, 'i')
        )

        tile_sdfg.add_edge(read_x_state, compute_state, dace.InterstateEdge(None))

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        x_in = compute_state.add_read('x_buf')

        m_map_entry, m_map_exit = compute_state.add_map(
            'm_block_map',
            dict(tj = '0:{}/{}'.format(m, m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        tile_n_entry, tile_n_exit = compute_state.add_map(
            'n_tile_map',
            dict(i = '0:{}'.format(n_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )

        red_sdfg = Expand_GER_FPGA_Streaming_RowTiles.make_unrolledCompute(
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
            {'_A', 'x_buf', '_y'},
            {'_A_out', 'y_buf'}
        )

        compute_state.add_memlet_path(
            A_in, m_map_entry, tile_n_entry, nested_sdfg,
            dst_conn='_A',
            memlet=Memlet.simple(A_in.data, "0:{}*{}".format(n, m))
        )

        compute_state.add_memlet_path(
            x_in, m_map_entry, tile_n_entry, nested_sdfg,
            dst_conn='x_buf',
            memlet=Memlet.simple(x_in.data, "0:{}".format(n_tile))
        )

        compute_state.add_memlet_path(
            y_in, m_map_entry, tile_n_entry, nested_sdfg,
            dst_conn='_y',
            memlet=Memlet.simple(y_in.data, "0:{}".format(n))
        )

        compute_state.add_memlet_path(
            nested_sdfg, tile_n_exit, m_map_exit, A_out,
            src_conn='_A_out',
            memlet=Memlet.simple(A_out.data, "0:{}*{}".format(n, m))
        )

        compute_state.add_memlet_path(
            nested_sdfg, tile_n_exit, y_buf,
            src_conn='y_buf',
            memlet=Memlet.simple(y_buf.data, "0:{}".format(m_tile))
        )

        return tile_sdfg



    @staticmethod
    def make_unrolledCompute(dtype, n_tile, m_tile, n, m, veclen, a):
        inner_sdfg = dace.SDFG("vectorize_inner_graph")
        inner_sdfg.add_symbol(a.name, a.dtype)

        init_state = inner_sdfg.add_state("init_state")
        compute_state = inner_sdfg.add_state('compute_state')
        read_y_state = inner_sdfg.add_state("read_y_state")
        read_empty_state = inner_sdfg.add_state("read_empty_state")

        stream_out_state = inner_sdfg.add_state('stream_out_state')

        vec_type = dtypes.vector(dtype, veclen)
        singleton_vec = dtypes.vector(dtype, 1)
        A_in = compute_state.add_stream(
            '_A',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        y_in = read_y_state.add_stream(
            '_y',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        A_out = stream_out_state.add_stream(
            '_A_out',
            vec_type,
            buffer_size=32,
            storage=dtypes.StorageType.FPGA_Local
        )

        inner_sdfg.add_array('x_buf', shape=[n_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local)
        inner_sdfg.add_array('A_buf', shape=[m_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=True)

        #
        # Read y
        #
        inner_sdfg.add_array('y_buf', shape=[m_tile], dtype=dtype, storage=dtypes.StorageType.FPGA_Local, transient=False)
        read_y_entry, read_y_exit = read_y_state.add_map(
            'read_y_map',
            dict(k = '0:{}'.format(m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device
        )
        read_y_buf = read_y_state.add_write('y_buf')
        read_y_task = read_y_state.add_tasklet('read_y_task',
            ['in_con'],
            ['out_con'],
            'out_con = in_con'
        )
        read_y_state.add_memlet_path(
            y_in, read_y_entry, read_y_task,
            dst_conn='in_con',
            memlet=Memlet.simple(y_in.data, '0')
        )
        read_y_state.add_memlet_path(
            read_y_task, read_y_exit, read_y_buf,
            src_conn='out_con',
            memlet=Memlet.simple(read_y_buf.data, 'k')
        )

        inner_sdfg.add_edge(init_state, read_y_state, dace.InterstateEdge('i == 0'))
        inner_sdfg.add_edge(read_y_state, compute_state, dace.InterstateEdge(None))
        inner_sdfg.add_edge(init_state, read_empty_state, dace.InterstateEdge('i != 0'))
        inner_sdfg.add_edge(read_empty_state, compute_state, dace.InterstateEdge(None))

        #
        # Compute
        #

        m_tile_entry, m_tile_exit = compute_state.add_map(
            'm_tile_map',
            dict(j = '0:{}'.format(m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device,
        )

        x_in = compute_state.add_read("x_buf")
        y_in = compute_state.add_read("y_buf")
        A_buf = compute_state.add_write("A_buf")

        compute_task = compute_state.add_tasklet(
            'compute_task',
            ['A_con', 'x_con', 'y_con'],
            ['out_con'],
            'out_con = {} * x_con * y_con + A_con'.format(a)
        )

        compute_state.add_memlet_path(
            A_in, m_tile_entry, compute_task,
            dst_conn='A_con',
            memlet=Memlet.simple(A_in.data, "0")
        )

        compute_state.add_memlet_path(
            x_in, m_tile_entry, compute_task,
            dst_conn='x_con',
            memlet=Memlet.simple(x_in.data, "i")
        )

        compute_state.add_memlet_path(
            y_in, m_tile_entry, compute_task,
            dst_conn='y_con',
            memlet=Memlet.simple(y_in.data, "j")
        )

        compute_state.add_memlet_path(
            compute_task, m_tile_exit, A_buf,
            src_conn='out_con',
            memlet=Memlet.simple(A_buf.data, "j")
        )

        # ---------- ----------
        # STREAM RESULT
        # ---------- ---------
        A_buf = stream_out_state.add_read('A_buf')

        stream_out_entry, stream_out_exit = stream_out_state.add_map(
            'stream_out_map',
            dict(j='0:{}'.format(m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device,
        )
        stream_out_task = stream_out_state.add_tasklet(
            'stream_out_task',
            ['in_con'],
            ['out_con'],
            'out_con = in_con'
        )

        stream_out_state.add_memlet_path(
            A_buf, stream_out_entry, stream_out_task,
            dst_conn='in_con',
            memlet=Memlet.simple(A_buf.data, "j")
        )

        stream_out_state.add_memlet_path(
            stream_out_task, stream_out_exit, A_out,
            src_conn='out_con',
            memlet=Memlet.simple(A_out.data, '0')
        )

        inner_sdfg.add_edge(compute_state, stream_out_state, dace.InterstateEdge(None))

        return inner_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return Expand_GER_FPGA_Streaming_RowTiles.make_sdfg(
            node.dtype,
            node.n_tile,
            node.m_tile,
            node.n,
            node.m,
            int(node.veclen),
            node.a
        )


@dace.library.node
class Ger(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
        "fpga_stream": Expand_GER_FPGA_Streaming_RowTiles
    }
    default_implementation = 'pure'

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    n_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    m_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))
    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"))

    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)

    alpha = Property(
        dtype=tuple(dace.dtypes._CONSTANT_TYPES),
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
        #if (len(size_out) != 2 or size_out[0] != size_a[0]
        #        or size_out[1] != size_a[1]):
        #    raise ValueError(
        #        "Output matrix must match input matrix a and outer product x*yT."
        #    )

