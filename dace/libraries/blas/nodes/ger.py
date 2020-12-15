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


@dace.library.expansion
class Expand_GER_FPGA_Streaming_Col_tiles(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype, n_tile, m_tile, n, m, veclen, a, streaming):

        # ---------- ----------
        # SETUP GRAPH
        # ---------- ----------
        # A: m rows, n columns, row-major
        ger_sdfg = dace.SDFG("ger_fpga_stream_rowTiles")

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        ger_sdfg.add_symbol(a.name, a.dtype)
        ger_state = ger_sdfg.add_state('ger_compute')

        if streaming:
            A_in = ger_state.add_stream('_A',
                                        vec_type,
                                        buffer_size=32,
                                        storage=dtypes.StorageType.FPGA_Local)

            y_in = ger_state.add_stream('_y',
                                        vec_type,
                                        buffer_size=32,
                                        storage=dtypes.StorageType.FPGA_Local)

            # Must be received n/n_tiles times
            x_in = ger_state.add_stream('_x',
                                        single_vec_type,
                                        buffer_size=32,
                                        storage=dtypes.StorageType.FPGA_Local)

            res = ger_state.add_stream('_res',
                                       vec_type,
                                       buffer_size=32,
                                       storage=dtypes.StorageType.FPGA_Local)

        else:
            ger_sdfg.add_array("_A", shape=[n * m / veclen], dtype=vec_type)
            ger_sdfg.add_array("_y", shape=[n / veclen], dtype=vec_type)
            ger_sdfg.add_array("_x", shape=[m], dtype=single_vec_type)
            ger_sdfg.add_array("_res", shape=[n * m / veclen], dtype=vec_type)

            A_in = ger_state.add_read("_A")
            y_in = ger_state.add_read("_y")
            x_in = ger_state.add_read("_x")
            res = ger_state.add_write("_res")

        ger_sdfg.add_array('y_buf_row',
                           shape=[n_tile / veclen],
                           dtype=vec_type,
                           storage=dtypes.StorageType.FPGA_Local,
                           transient=True)
        ger_sdfg.add_array('x_buf',
                           shape=[m_tile],
                           dtype=single_vec_type,
                           storage=dtypes.StorageType.FPGA_Local,
                           transient=True)

        # ---------- ----------
        # COMPUTE
        # ---------- ----------
        y_buf = ger_state.add_write('y_buf_row')
        x_buf = ger_state.add_write('x_buf')

        nMap_entry, nMap_exit = ger_state.add_map(
            'n_tile_map',
            dict(i='0:{}/{}'.format(n, n_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        mMap_entry, mMap_exit = ger_state.add_map(
            'm_block_map',
            dict(j='0:{}/{}'.format(m, m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        outerComputeMap_entry, outerComputeMap_exit = ger_state.add_map(
            'outerCompute_map',
            dict(jj='0:{}'.format(m_tile)),
            schedule=dtypes.ScheduleType.FPGA_Device)

        innerComputeMap_entry, innerComputeMap_exit = ger_state.add_map(
            'innerCompute_map',
            dict(ii='0:{}/{}'.format(n_tile, veclen)),
            schedule=dtypes.ScheduleType.FPGA_Device,
        )

        tile_sdfg = None
        tile_sdfg = Expand_GER_FPGA_Streaming_Col_tiles.make_unrolledCompute(
            dtype, n_tile, m_tile, n, m, veclen, a, streaming)

        nested_sdfg = ger_state.add_nested_sdfg(
            tile_sdfg, ger_sdfg, {'_A_unroll', '_x_stream', '_y_stream'},
            {'_A_out_unroll', '_y_unroll', '_x_buf_unroll'})

        ger_state.add_memlet_path(A_in,
                                  nMap_entry,
                                  mMap_entry,
                                  outerComputeMap_entry,
                                  innerComputeMap_entry,
                                  nested_sdfg,
                                  dst_conn='_A_unroll',
                                  memlet=Memlet.simple(A_in.data,
                                                       "0:{}*{}".format(n, m)))

        ger_state.add_memlet_path(x_in,
                                  nMap_entry,
                                  mMap_entry,
                                  outerComputeMap_entry,
                                  innerComputeMap_entry,
                                  nested_sdfg,
                                  dst_conn='_x_stream',
                                  memlet=Memlet.simple(x_in.data,
                                                       "0:{}".format(m)))

        ger_state.add_memlet_path(y_in,
                                  nMap_entry,
                                  mMap_entry,
                                  outerComputeMap_entry,
                                  innerComputeMap_entry,
                                  nested_sdfg,
                                  dst_conn='_y_stream',
                                  memlet=Memlet.simple(y_in.data,
                                                       "0:{}".format(n)))

        ger_state.add_memlet_path(nested_sdfg,
                                  innerComputeMap_exit,
                                  outerComputeMap_exit,
                                  mMap_exit,
                                  nMap_exit,
                                  res,
                                  src_conn='_A_out_unroll',
                                  memlet=Memlet.simple(res.data,
                                                       "0:{}*{}".format(n, m)))

        ger_state.add_memlet_path(nested_sdfg,
                                  innerComputeMap_exit,
                                  outerComputeMap_exit,
                                  mMap_exit,
                                  nMap_exit,
                                  y_buf,
                                  src_conn='_y_unroll',
                                  memlet=Memlet.simple(
                                      y_buf.data,
                                      "0:{}".format(n_tile / veclen)))

        ger_state.add_memlet_path(nested_sdfg,
                                  innerComputeMap_exit,
                                  outerComputeMap_exit,
                                  mMap_exit,
                                  nMap_exit,
                                  x_buf,
                                  src_conn='_x_buf_unroll',
                                  memlet=Memlet.simple(x_buf.data,
                                                       "0:{}".format(m_tile)))

        return ger_sdfg

    @staticmethod
    def make_unrolledCompute(dtype, n_tile, m_tile, n, m, veclen, a, streaming):

        inner_sdfg = dace.SDFG("vectorize_inner_graph")
        inner_sdfg.add_symbol(a.name, a.dtype)

        init_state = inner_sdfg.add_state("init_state")
        compute_state = inner_sdfg.add_state('compute_state')

        read_x_state = inner_sdfg.add_state("read_x_state")
        read_empty_state = inner_sdfg.add_state("readEmpty_state")

        stream_out_state = inner_sdfg.add_state('streamOut_state')

        init_state_y = inner_sdfg.add_state('init_state_tile')

        read_y_state = inner_sdfg.add_state('read_y_reduceTile')
        read_empty_state_y = inner_sdfg.add_state('read_empty_reduceTile')

        vec_type = dace.vector(dtype, veclen)
        single_vec_type = dace.vector(dtype, 1)

        # ---------- ----------
        # DATA DEFN
        # ---------- ----------
        if streaming:
            A_in = compute_state.add_stream(
                '_A_unroll',
                vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local)

            y_in = read_y_state.add_stream(
                '_y_stream',
                vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local)

            x_in = read_x_state.add_stream(
                '_x_stream',
                single_vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local)

            A_out = stream_out_state.add_stream(
                '_A_out_unroll',
                vec_type,
                buffer_size=32,
                storage=dtypes.StorageType.FPGA_Local)

        else:

            inner_sdfg.add_array("_A_unroll",
                                 shape=[n * m / veclen],
                                 dtype=vec_type)
            inner_sdfg.add_array("_y_stream",
                                 shape=[n / veclen],
                                 dtype=vec_type)
            inner_sdfg.add_array("_x_stream", shape=[m], dtype=single_vec_type)
            inner_sdfg.add_array("_A_out_unroll",
                                 shape=[n * m / veclen],
                                 dtype=vec_type)

            A_in = compute_state.add_read("_A_unroll")
            y_in = read_y_state.add_read("_y_stream")
            x_in = read_x_state.add_read("_x_stream")
            A_out = stream_out_state.add_write("_A_out_unroll")

        inner_sdfg.add_array('_y_unroll',
                             shape=[n_tile / veclen],
                             dtype=vec_type,
                             storage=dtypes.StorageType.FPGA_Local)
        inner_sdfg.add_array('_x_buf_unroll',
                             shape=[m_tile],
                             dtype=single_vec_type,
                             storage=dtypes.StorageType.FPGA_Local)

        inner_sdfg.add_array('A_out_buf',
                             shape=[1],
                             dtype=vec_type,
                             storage=dtypes.StorageType.FPGA_Local,
                             transient=True)
        inner_sdfg.add_array('x_vecbuf',
                             shape=[veclen],
                             dtype=single_vec_type,
                             storage=dtypes.StorageType.FPGA_Local,
                             transient=True)
        inner_sdfg.add_array('x_membuf',
                             shape=[veclen],
                             dtype=single_vec_type,
                             storage=dtypes.StorageType.FPGA_Local,
                             transient=True)

        inner_sdfg.add_array('y_buf',
                             shape=[1],
                             dtype=vec_type,
                             storage=dtypes.StorageType.FPGA_Registers,
                             transient=True)

        # ---------- ----------
        # GET DATA x
        # ---------- ---------
        data_out = read_x_state.add_write('_x_buf_unroll')

        copy_x_task = read_x_state.add_tasklet('streamToLocal_map', ['inCon'],
                                               ['outCon'], 'outCon = inCon')

        read_x_state.add_memlet_path(
            x_in,
            copy_x_task,
            dst_conn="inCon",
            memlet=Memlet.simple(
                x_in.data, "0" if streaming else "j * {} + jj".format(m_tile)))

        read_x_state.add_memlet_path(copy_x_task,
                                     data_out,
                                     src_conn="outCon",
                                     memlet=Memlet.simple(data_out.data, "jj"))

        inner_sdfg.add_edge(init_state, read_x_state,
                            dace.InterstateEdge("ii == 0"))
        inner_sdfg.add_edge(read_x_state, init_state_y,
                            dace.InterstateEdge(None))

        inner_sdfg.add_edge(init_state, read_empty_state,
                            dace.InterstateEdge("ii != 0"))
        inner_sdfg.add_edge(read_empty_state, init_state_y,
                            dace.InterstateEdge(None))

        # ---------- ----------
        # Get data y
        # ---------- ----------
        y_out = read_y_state.add_write("_y_unroll")

        read_y_state.add_memlet_path(
            y_in,
            y_out,
            memlet=Memlet.simple(y_in.data,
                                 "0" if streaming else
                                 "(i * {} / {} + ii)".format(n_tile, veclen),
                                 other_subset_str="ii"))

        inner_sdfg.add_edge(init_state_y, read_y_state,
                            dace.InterstateEdge("j == 0 and jj == 0"))
        inner_sdfg.add_edge(read_y_state, compute_state,
                            dace.InterstateEdge(None))
        inner_sdfg.add_edge(init_state_y, read_empty_state_y,
                            dace.InterstateEdge("not (j == 0 and jj == 0)"))
        inner_sdfg.add_edge(read_empty_state_y, compute_state,
                            dace.InterstateEdge(None))

        # ---------- ----------
        # COMPUTE
        # ---------- ---------
        x_in = compute_state.add_read("_x_buf_unroll")
        y_in = compute_state.add_read("_y_unroll")
        A_out_buf = compute_state.add_write("A_out_buf")

        compute_task = compute_state.add_tasklet(
            'compute_task', ['A_con', 'x_con', 'y_con'], ['outCon'],
            'outCon = A_con + x_con * y_con * {}'.format(a))

        compute_state.add_memlet_path(
            A_in,
            compute_task,
            dst_conn='A_con',
            memlet=Memlet.simple(
                A_in.data, "0" if streaming else
                '(j *{0} + jj) * {1} / {3} + (i * {2} + ii * {3}) / {3}'.format(
                    m_tile, n, n_tile, veclen)))

        compute_state.add_memlet_path(y_in,
                                      compute_task,
                                      dst_conn='y_con',
                                      memlet=Memlet.simple(y_in.data, "ii"))

        compute_state.add_memlet_path(x_in,
                                      compute_task,
                                      dst_conn='x_con',
                                      memlet=Memlet.simple(x_in.data, "jj"))

        compute_state.add_memlet_path(compute_task,
                                      A_out_buf,
                                      src_conn='outCon',
                                      memlet=Memlet.simple(A_out_buf.data, "0"))

        # ---------- ----------
        # STREAM RESULT
        # ---------- ---------
        A_out_buf = stream_out_state.add_read('A_out_buf')

        stream_out_state.add_memlet_path(
            A_out_buf,
            A_out,
            memlet=Memlet.simple(
                A_out.data,
                "0" if streaming else
                '(j *{0} + jj) * {1} / {3} + (i * {2} + ii * {3}) / {3}'.format(
                    m_tile, n, n_tile, veclen),
                other_subset_str="0"))

        inner_sdfg.add_edge(compute_state, stream_out_state,
                            dace.InterstateEdge(None))

        return inner_sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")

        streaming = False
        streaming_nodes = 0

        for e in state.in_edges(node):

            if isinstance(
                    sdfg.arrays[e.data.data],
                    dace.data.Stream) and e.dst_conn in ["_x", "_y", "_A"]:
                streaming = True
                streaming_nodes = streaming_nodes + 1

        for e in state.out_edges(node):
            if isinstance(sdfg.arrays[e.data.data],
                          dace.data.Stream) and e.src_conn == "_res":
                streaming = True
                streaming_nodes = streaming_nodes + 1

        if streaming and streaming_nodes < 4:
            raise ValueError(
                "All inputs and outputs must be of same type either Array or Stream"
            )

        return Expand_GER_FPGA_Streaming_Col_tiles.make_sdfg(
            node.dtype, node.n_tile, node.m_tile, node.n, node.m,
            int(node.veclen), node.alpha, streaming)


@dace.library.expansion
class ExpandGerFpga(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg, tile_size_x=None, tile_size_y=None, **kwargs):
        node.validate(sdfg, state)

        for e in state.in_edges(node):
            if e.dst_conn == "_A":
                desc_a_in = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_res":
                desc_a_out = sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("ger")
        state = sdfg.add_state("ger")

        desc_a_in = desc_a_in.clone()
        desc_x = desc_x.clone()
        desc_y = desc_y.clone()
        desc_a_out = desc_a_out.clone()
        desc_a_in.transient = False
        desc_a_out.transient = False
        desc_x.transient = False
        desc_y.transient = False
        sdfg.add_datadesc("_A", desc_a_in)
        sdfg.add_datadesc("_res", desc_a_out)
        sdfg.add_datadesc("_x", desc_x)
        sdfg.add_datadesc("_y", desc_y)

        m = desc_a_in.shape[0]
        n = desc_a_in.shape[1]
        alpha = node.alpha

        size_x = m
        size_y = n

        if tile_size_x is None:
            tile_size_x = node.m_tile
        if tile_size_y is None:
            tile_size_y = node.n_tile

        num_tiles_x = f"{size_x} / {tile_size_x}"
        num_tiles_y = f"{size_y} / {tile_size_y}"

        y_tile_entry, y_tile_exit = state.add_map(
            "y_tiles", {"ty": f"0:{num_tiles_y}"},
            schedule=dace.ScheduleType.FPGA_Device)

        sdfg.add_array("y_local", (tile_size_y, ),
                       desc_y.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        y_local = state.add_access("y_local")

        # Load y buffer
        read_y = state.add_read("_y")
        subset = ("0" if isinstance(desc_y, dace.data.Stream) else
                  f"ty*{tile_size_y}+iy")
        read_y_entry, read_y_exit = state.add_map(
            "read_y", {"iy": f"0:{tile_size_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        read_y_tasklet = state.add_tasklet("read_y", {"y_memory"}, {"y_buffer"},
                                           "y_buffer = y_memory")
        state.add_memlet_path(read_y,
                              y_tile_entry,
                              read_y_entry,
                              read_y_tasklet,
                              dst_conn="y_memory",
                              memlet=dace.Memlet(f"_y[{subset}]"))
        state.add_memlet_path(read_y_tasklet,
                              read_y_exit,
                              y_local,
                              src_conn="y_buffer",
                              memlet=dace.Memlet(f"y_local[iy]"))

        x_tile_entry, x_tile_exit = state.add_map(
            "x_tiles", {"tx": f"0:{num_tiles_x}"},
            schedule=dace.ScheduleType.FPGA_Device)

        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Load x
        read_x = state.add_read("_x")
        sdfg.add_array("x_local", (1, ),
                       desc_x.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        x_local = state.add_access("x_local")
        subset = ("0" if isinstance(desc_x, dace.data.Stream) else
                  f"tx*{tile_size_x} + ix")
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              x_local,
                              memlet=dace.Memlet(f"_x[{subset}]",
                                                 other_subset="0"))

        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Actual computation
        compute_tasklet = state.add_tasklet(
            "ger", {"a_in", "x_in", "y_in"}, {"a_out"},
            f"a_out = {alpha} * x_in * y_in + a_in")

        # Stream in A
        read_a = state.add_read("_A")
        subset_a = ("0" if isinstance(desc_a_in, dace.data.Stream) else
                    f"tx*{tile_size_x} + ix, ty*{tile_size_y} + iy")
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              compute_tasklet,
                              dst_conn="a_in",
                              memlet=dace.Memlet(f"_A[{subset_a}]"))

        # Load buffered x and y
        state.add_memlet_path(x_local,
                              y_entry,
                              compute_tasklet,
                              dst_conn="x_in",
                              memlet=dace.Memlet("x_local[0]"))
        state.add_memlet_path(
            y_local,
            x_tile_entry,
            x_entry,
            y_entry,
            compute_tasklet,
            dst_conn="y_in",
            memlet=dace.Memlet(f"y_local[iy]"))

        # Store result
        write_a = state.add_write("_res")
        state.add_memlet_path(compute_tasklet,
                              y_exit,
                              x_exit,
                              x_tile_exit,
                              y_tile_exit,
                              write_a,
                              src_conn="a_out",
                              memlet=dace.Memlet(f"_res[{subset_a}]"))

        return sdfg


@library.node
class Ger(LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
        "fpga_column": Expand_GER_FPGA_Streaming_Col_tiles,
        "IntelFPGA": ExpandGerPure,
        "FPGA": ExpandGerFpga
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    n_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)
    m_tile = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("m"))

    veclen = dace.properties.SymbolicProperty(allow_none=False, default=1)

    alpha = SymbolicProperty(
        default=1,
        desc=
        "A scalar which will be multiplied with the outer product x*yT before adding matrix A"
    )

    def __init__(self,
                 name,
                 dtype=dace.float32,
                 n_tile=1,
                 m_tile=1,
                 n=dace.symbolic.symbol("n"),
                 m=dace.symbolic.symbol("m"),
                 veclen=1,
                 alpha=1,
                 location=None):
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

        self.alpha = alpha

    def compare(self, other):

        if (self.dtype == other.dtype and self.veclen == other.veclen
                and self.implementation == other.implementation
                and self.n_tile == other.n_tile
                and self.m_tile == other.m_tile):

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
