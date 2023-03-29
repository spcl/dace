# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.properties import Property, SymbolicProperty
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace.sdfg.nodes import LibraryNode
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
import dace.library as library
from dace.sdfg import SDFG, SDFGState, nodes
from dace import data as dt, memlet as mm, subsets as sbs
import dace
import copy
import numpy as np

import dace.library
import dace.properties
import dace.sdfg.nodes

from dace import dtypes
from dace.memlet import Memlet


@library.expansion
class ExpandGerPure(ExpandTransformation):
    """
    Generic expansion of GER.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        inputs = ('_A', '_x', '_y')
        outputs = ('_res', )
        in_edges = [next(parent_state.in_edges_by_connector(node, conn)) for conn in inputs]
        out_edges = [next(parent_state.out_edges_by_connector(node, conn)) for conn in outputs]
        arrays = {}
        arrays.update({inp: parent_sdfg.arrays[e.data.data] for inp, e in zip(inputs, in_edges)})
        arrays.update({out: parent_sdfg.arrays[e.data.data] for out, e in zip(outputs, out_edges)})

        # TODO: Support memlet subsets
        if any(e.data.subset != sbs.Range.from_array(arrays[a]) for a, e in zip(inputs, in_edges)):
            raise NotImplementedError
        if any(e.data.subset != sbs.Range.from_array(arrays[a]) for a, e in zip(outputs, out_edges)):
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
        nsdfg_node = nodes.NestedSDFG(node.label, sdfg, set(inputs), set(outputs), {
            'M': outshape[0],
            'N': outshape[1],
            'alpha': node.alpha
        })

        return nsdfg_node


@dace.library.expansion
class ExpandGerFpga(ExpandTransformation):
    """
    FPGA-specific expansion of GER with support for vectorization and tiling
    in both dimensions.
    """

    environments = []

    @staticmethod
    def expansion(node, state, sdfg, m=None, n=None, tile_size_x=None, tile_size_y=None):
        """
        :param node: Node to expand.
        :param state: State that the node is in.
        :param sdfg: SDFG that the node is in.
        :param m: Override the number of rows.
        :param n: Override the number of columns.
        :param tile_size_x: Tile size along the M-dimension (rows of A, size of
                            vector x).
        :param tile_size_x: Tile size along the N-dimension (columns of A,
                            size of vector y).
        """

        desc_a_in, desc_x, desc_y = node.validate(sdfg, state)
        desc_a_out = None
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

        m = m or node.m
        n = n or node.n
        alpha = node.alpha
        veclen = desc_y.dtype.veclen

        size_x = m
        size_y = n / veclen

        num_tiles_x = f"{size_x} / {tile_size_x}"
        num_tiles_y = f"{size_y} / {tile_size_y}"

        y_tile_entry, y_tile_exit = state.add_map("y_tiles", {"ty": f"0:{num_tiles_y}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        sdfg.add_array("y_local", (tile_size_y, ), desc_y.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        y_local = state.add_access("y_local")

        # Load y buffer
        read_y = state.add_read("_y")
        subset = ("0" if isinstance(desc_y, dace.data.Stream) else f"ty*{tile_size_y}+iy")
        read_y_entry, read_y_exit = state.add_map("read_y", {"iy": f"0:{tile_size_y}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)
        read_y_tasklet = state.add_tasklet("read_y", {"y_memory"}, {"y_buffer"}, "y_buffer = y_memory")
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

        x_tile_entry, x_tile_exit = state.add_map("x_tiles", {"tx": f"0:{num_tiles_x}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Load x
        read_x = state.add_read("_x")
        sdfg.add_array("x_local", (1, ), desc_x.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        x_local = state.add_access("x_local")
        subset = ("0" if isinstance(desc_x, dace.data.Stream) else f"tx*{tile_size_x} + ix")
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              x_local,
                              memlet=dace.Memlet(f"_x[{subset}]", other_subset="0"))

        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Actual computation
        compute_tasklet = state.add_tasklet("ger", {"a_in", "x_in", "y_in"}, {"a_out"},
                                            f"a_out = {alpha} * x_in * y_in + a_in")

        # Stream in A
        read_a = state.add_read("_A")
        subset_a = ("0" if isinstance(desc_a_in, dace.data.Stream) else f"tx*{tile_size_x} + ix, ty*{tile_size_y} + iy")
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              compute_tasklet,
                              dst_conn="a_in",
                              memlet=dace.Memlet(f"_A[{subset_a}]"))

        # Load buffered x and y
        state.add_memlet_path(x_local, y_entry, compute_tasklet, dst_conn="x_in", memlet=dace.Memlet("x_local[0]"))
        state.add_memlet_path(y_local,
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
    """
    Implements the BLAS operation GER, which computes alpha*x*y^T + A (i.e.,
    the outer product of two vectors x and y added to a matrix A).

    The node expects input connectors "_x", "_y", and "_A", and output
    connector "_res".
    """

    # Global properties
    implementations = {"pure": ExpandGerPure, "FPGA": ExpandGerFpga}
    default_implementation = None

    # Object fields
    n_tile_size = dace.properties.SymbolicProperty(allow_none=False, default=1)
    m_tile_size = dace.properties.SymbolicProperty(allow_none=False, default=1)

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"))

    alpha = SymbolicProperty(
        default=1, desc="A scalar which will be multiplied with the outer product x*yT before adding matrix A")

    def __init__(self, name, n=dace.symbolic.symbol("n"), m=dace.symbolic.symbol("m"), alpha=1, location=None):
        super().__init__(name, location=location, inputs={"_x", "_y", "_A"}, outputs={"_res"})

        self.n = n
        self.m = m

        self.alpha = alpha

    def compare(self, other):

        if (self.implementation == other.implementation and self.n_tile_size == other.n_tile
                and self.m_tile_size == other.m_tile):

            return True
        else:
            return False

    def validate(self, sdfg, state):

        desc_a = None
        desc_x = None
        desc_y = None
        size_a = None
        size_x = None
        size_y = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a = subset.size()
                desc_a = sdfg.arrays[memlet.data]
            if dst_conn == "_x":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
                desc_x = sdfg.arrays[memlet.data]
            if dst_conn == "_y":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y = subset.size()
                desc_y = sdfg.arrays[memlet.data]

        if size_a is None or size_x is None:
            raise ValueError("Expected at least two inputs to Ger (matrix A and vector x)")

        if size_y is None:
            raise ValueError("Expected exactly one output from Ger (vector y).")

        # The following checks don't work for streams
        if (not isinstance(desc_x, dt.Array) or not isinstance(desc_y, dt.Array) or not isinstance(desc_a, dt.Array)):
            return

        if len(size_a) != 2:
            raise ValueError("A must be a matrix")
        if len(size_x) != 1:
            raise ValueError("x must be a vector")
        if len(size_y) != 1:
            raise ValueError("y must be a vector")

        if size_a[0] != size_x[0] or size_a[1] != size_y[0]:
            raise ValueError("Input vectors x and y (outer product) must match with the matrix A dimensions.")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from ger rank 1 operation.")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()

        if (len(size_out) != 2 or size_out[0] != size_a[0] or size_out[1] != size_a[1]):
            raise ValueError("Output matrix must match input matrix a and outer product x*yT.")

        return desc_a, desc_x, desc_y


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.ger')
@oprepo.replaces('dace.libraries.blas.Ger')
def ger_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, A, x, y, output, alpha):
    # Add nodes
    A_in, x_in, y_in = (state.add_read(name) for name in (A, x, y))
    out = state.add_write(output)

    libnode = Ger('ger', alpha=alpha)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_res', out, None, mm.Memlet(output))

    return []
