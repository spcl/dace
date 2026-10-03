# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import warnings

from dace.properties import SymbolicProperty
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace.sdfg.nodes import LibraryNode
import dace.library as library
from dace.sdfg import SDFG, SDFGState, nodes
from dace import data as dt, memlet as mm, subsets as sbs
import dace
import copy

import dace.library
import dace.properties
import dace.sdfg.nodes

from dace.libraries.blas import blas_helpers
from .. import environments


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


def _ger_strides(node, parent_state, parent_sdfg):
    """Extract leading-dim strides from ``_A``, ``_x`` and ``_y`` memlets."""
    sx = sy = lda = None
    for e in parent_state.in_edges(node):
        sq = copy.deepcopy(e.data.subset)
        dims = sq.squeeze()
        desc = parent_sdfg.arrays[e.data.data]
        if e.dst_conn == '_A':
            lda = desc.strides[dims[0]]
        elif e.dst_conn == '_x':
            sx = desc.strides[dims[0]]
        elif e.dst_conn == '_y':
            sy = desc.strides[dims[0]]
    return lda, sx, sy


@library.expansion
class ExpandGerOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        a_desc = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, '_A')).data.data]
        dtype = a_desc.dtype.base_type
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandGerPure.expansion(node, parent_state, parent_sdfg, **kwargs)
        lda, sx, sy = _ger_strides(node, parent_state, parent_sdfg)
        prefix = func.lower()
        m, n = node.m, node.n
        alpha = node.alpha
        # cBLAS ger updates A in place; copy _A into _res first, then call.
        if dtype in (dace.complex64, dace.complex128):
            cfunc = prefix + 'gerc'
        else:
            cfunc = prefix + 'ger'
        code = f"""
        for (int __i = 0; __i < ({m}); ++__i)
          cblas_{prefix}copy({n}, _A + __i * ({lda}), 1, _res + __i * ({lda}), 1);
        """
        if dtype in (dace.complex64, dace.complex128):
            code += f"""
            {dtype.ctype} __alpha = {dtype.ctype}({alpha});
            cblas_{cfunc}(CblasRowMajor, {m}, {n}, &__alpha, _x, {sx}, _y, {sy}, _res, {lda});
            """
        else:
            code += f"cblas_{cfunc}(CblasRowMajor, {m}, {n}, ({dtype.ctype})({alpha}), _x, {sx}, _y, {sy}, _res, {lda});"
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@library.expansion
class ExpandGerMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGerOpenBLAS.expansion(*args, **kwargs)


@library.expansion
class ExpandGerCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        a_desc = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, '_A')).data.data]
        dtype = a_desc.dtype.base_type
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandGerPure.expansion(node, parent_state, parent_sdfg, **kwargs)
        lda, sx, sy = _ger_strides(node, parent_state, parent_sdfg)
        m, n = node.m, node.n
        alpha = node.alpha
        cfunc = func + 'ger' if dtype not in (dace.complex64, dace.complex128) else func + 'gerc'
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        # cuBLAS ger updates A in place; copy _A into _res first, then call.
        code += f"""
        {dtype.ctype} __alpha = {dtype.ctype}({alpha});
        cudaMemcpyAsync(_res, _A, sizeof({dtype.ctype}) * ({m}) * ({lda}),
                        cudaMemcpyDeviceToDevice, __dace_current_stream);
        cublas{cfunc}(__dace_cublas_handle, {m}, {n}, &__alpha, _x, {sx}, _y, {sy}, _res, {lda});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@library.node
class Ger(LibraryNode):
    """
    Implements the BLAS operation GER, which computes alpha*x*y^T + A (i.e.,
    the outer product of two vectors x and y added to a matrix A).

    The node expects input connectors "_x", "_y", and "_A", and output
    connector "_res".
    """

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
        "OpenBLAS": ExpandGerOpenBLAS,
        "MKL": ExpandGerMKL,
        "cuBLAS": ExpandGerCuBLAS,
    }
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
