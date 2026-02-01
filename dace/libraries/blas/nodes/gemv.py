# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace import properties, symbolic
import dace.library
import dace.sdfg.nodes
from dace.sdfg import SDFG, SDFGState
from dace import memlet as mm, data as dt
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from dace.libraries.blas import blas_helpers
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import environments
import numpy as np
import warnings


@dace.library.expansion
class ExpandGemvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        ((edge_a, outer_array_a, _, _, shape_a, strides_a), (edge_x, outer_array_x, _, _, shape_x, strides_x),
         (edge_y, outer_array_y, _, _, shape_y, strides_y)) = _get_matmul_operands(node,
                                                                                   parent_state,
                                                                                   parent_sdfg,
                                                                                   name_lhs="_A",
                                                                                   name_rhs="_x",
                                                                                   name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type

        if outer_array_a.dtype.veclen > 1 or outer_array_x.dtype.veclen > 1:
            raise NotImplementedError("Vectorization for pure GEMV NYI.")

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if trans_shape_a[1] != shape_x[0]:
            raise SyntaxError("Matrix-vector product size mismatch: {} vs. {}".format(trans_shape_a[1], shape_x[0]))

        N, M = trans_shape_a[0], trans_shape_a[1]

        if outer_array_a.storage != outer_array_x.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_A", shape_a, dtype_a, strides=strides_a, storage=storage)
        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, strides=strides_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, strides=strides_y, storage=storage)

        mul_program = "__out = {} * __A * __x".format(node.alpha)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_transient('gemv_tmp',
                                                                         shape_y,
                                                                         dtype_y,
                                                                         storage=storage,
                                                                         find_new_name=True)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        # Initialization map
        init_state.add_mapped_tasklet(
            "gemv_init", {
                "_o%d" % i: "0:%s" % symbolic.symstr(d)
                for i, d in enumerate(shape_y)
            }, {},
            "out = 0",
            {"out": dace.Memlet("{}[{}]".format(mul_out, ",".join(["_o%d" % i for i in range(len(shape_y))])))},
            external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("_GEMV_", {
            "__i%d" % i: "0:%s" % s
            for i, s in enumerate([N, M])
        }, {
            "__A": dace.Memlet("_A[{}]".format("__i1, __i0" if node.transA else "__i0, __i1")),
            "__x": dace.Memlet("_x[__i1]")
        },
                                 mul_program, {"__out": dace.Memlet(f"{mul_out}[__i0]", wcr="lambda x, y: x + y")},
                                 external_edges=True,
                                 output_nodes=output_nodes)

        add_program = "__y_out = ({} * __y_in) + __tmp".format(node.beta)

        memlet_idx = "__i"

        # addition map
        if node.beta != 0:
            state.add_mapped_tasklet("_Add_", {"__i": "0:{}".format(N)}, {
                "__y_in": dace.Memlet(f"_y[{memlet_idx}]"),
                "__tmp": dace.Memlet(f"{mul_out}[__i]"),
            },
                                     add_program, {"__y_out": dace.Memlet("_y[__i]")},
                                     external_edges=True,
                                     input_nodes={mul_out: access_tmp})

        return sdfg


@dace.library.expansion
class ExpandGemvCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node: 'Gemv', state, sdfg, m=None, n=None, **kwargs):
        node.validate(sdfg, state)

        ((edge_a, outer_array_a, _, _, shape_a, strides_a), (edge_x, outer_array_x, _, _, shape_x, strides_x),
         (edge_y, outer_array_y, _, _, shape_y, strides_y)) = _get_matmul_operands(node,
                                                                                   state,
                                                                                   sdfg,
                                                                                   name_lhs="_A",
                                                                                   name_rhs="_x",
                                                                                   name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype = outer_array_x.dtype.base_type
        veclen = outer_array_x.dtype.veclen
        m = m or node.m
        n = n or node.n
        if m is None:
            m = shape_y[0]
        if n is None:
            n = shape_x[0]

        transA = node.transA
        if strides_a[0] == 1:
            transA = not transA
            lda = strides_a[1]
        elif strides_a[1] == 1:
            lda = strides_a[0]
        else:
            warnings.warn('Matrix must be contiguous in at least '
                          'one dimension. Falling back to pure expansion.')
            return ExpandGemvPure.expansion(node, state, sdfg, m=m, n=n, **kwargs)

        trans = 'CUBLAS_OP_N' if transA else 'CUBLAS_OP_T'
        if not node.transA:
            m, n = n, m

        if veclen != 1:
            warnings.warn('Vector GEMV not supported, falling back to pure')
            return ExpandGemvPure.expansion(node, state, sdfg, m=m, n=n, **kwargs)

        try:
            func, ctype, runtimetype = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandGemvPure.expansion(node, state, sdfg, m=m, n=n, **kwargs)
        func += 'gemv'
        call_prefix = environments.cublas.cuBLAS.handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        constants = {
            1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{runtimetype}Pone()",
            0.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{runtimetype}Zero()",
        }
        if node.alpha not in constants or node.beta not in constants:
            # Deal with complex input constants
            if isinstance(node.alpha, complex):
                alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
            else:
                alpha = f'{dtype.ctype}({node.alpha})'
            if isinstance(node.beta, complex):
                beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
            else:
                beta = f'{dtype.ctype}({node.beta})'

            # Set pointer mode to host
            call_prefix += f'''cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_HOST);
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            call_suffix += '''
cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
            '''
            alpha = f'({ctype} *)&alpha'
            beta = f'({ctype} *)&beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        code = (call_prefix + f"""
cublas{func}(__dace_cublas_handle, {trans}, {m}, {n}, {alpha}, _A, {lda},
             _x, {strides_x[0]}, {beta}, _y, {strides_y[0]});
                """ + call_suffix)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandGemvOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node: 'Gemv', state, sdfg, m=None, n=None, **kwargs):
        from dace.sdfg.scope import is_devicelevel_gpu
        if is_devicelevel_gpu(sdfg, state, node):
            return ExpandGemvPure.expansion(node, state, sdfg)

        node.validate(sdfg, state)

        ((edge_a, outer_array_a, _, _, shape_a, strides_a), (edge_x, outer_array_x, _, _, shape_x, strides_x),
         (edge_y, outer_array_y, _, _, shape_y, strides_y)) = _get_matmul_operands(node,
                                                                                   state,
                                                                                   sdfg,
                                                                                   name_lhs="_A",
                                                                                   name_rhs="_x",
                                                                                   name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype = outer_array_x.dtype.base_type
        veclen = outer_array_x.dtype.veclen
        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'

        m = m or node.m
        n = n or node.n
        if m is None:
            m = shape_y[0]
        if n is None:
            n = shape_x[0]

        transA = node.transA
        if strides_a[0] == 1:
            transA = not transA
            lda = strides_a[1]
        elif strides_a[1] == 1:
            lda = strides_a[0]
        else:
            warnings.warn('Matrix must be contiguous in at least '
                          'one dimension. Falling back to pure expansion.')
            return ExpandGemvPure.expansion(node, state, sdfg, m=m, n=n, **kwargs)

        layout = 'CblasColMajor'
        trans = 'CblasNoTrans' if transA else 'CblasTrans'
        if not node.transA:
            m, n = n, m

        if veclen != 1:
            warnings.warn('Vector GEMV not supported, falling back to pure.')
            return ExpandGemvPure.expansion(node, state, sdfg, m=m, n=n, **kwargs)

        try:
            func, ctype, runtimetype = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandGemvPure.expansion(node, state, sdfg, m=m, n=n, **kwargs)

        func = func.lower() + 'gemv'

        code = ''
        if dtype in (dace.complex64, dace.complex128):
            code = f'''
            {dtype.ctype} __alpha = {alpha};
            {dtype.ctype} __beta = {beta};
            '''
            alpha = '&__alpha'
            beta = '&__beta'

        code += f"""cblas_{func}({layout}, {trans}, {m}, {n}, {alpha}, _A, {lda},
                                _x, {strides_x[0]}, {beta}, _y, {strides_y[0]});"""

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandGemvMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGemvOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandGemvPBLAS(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node: 'Gemv', state, sdfg, m=None, n=None, **kwargs):
        node.validate(sdfg, state)
        ((edge_a, outer_array_a, _, _, shape_a, strides_a), (edge_x, outer_array_x, _, _, shape_x, strides_x),
         (edge_y, outer_array_y, _, _, shape_y, strides_y)) = _get_matmul_operands(node,
                                                                                   state,
                                                                                   sdfg,
                                                                                   name_lhs="_A",
                                                                                   name_rhs="_x",
                                                                                   name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype = outer_array_x.dtype.base_type
        veclen = outer_array_x.dtype.veclen
        m = m or node.m
        n = n or node.n
        if m is None:
            m = shape_y[0]
        if n is None:
            n = shape_x[0]

        transA = node.transA

        Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
        Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
        try:
            sdfg.add_symbol('Px', dace.int32)
            sdfg.add_symbol('Py', dace.int32)
        except FileExistsError:
            pass

        @dace.program
        def _gemNv_pblas(_A: dtype[m, n], _x: dtype[n], _y: dtype[m]):
            lA = np.empty((m // Px, n // Py), dtype=_A.dtype)
            lx = np.empty((n // Px, ), dtype=_x.dtype)
            dace.comm.BCScatter(_A, lA, (m // Px, n // Py))
            dace.comm.BCScatter(_x, lx, (n // Px, 1))
            ly = distr.MatMult(lA, lx, (m, n))
            dace.comm.BCGather(ly, _y, (m // Px, 1))

        @dace.program
        def _gemTv_pblas(_A: dtype[m, n], _x: dtype[m], _y: dtype[n]):
            lA = np.empty((m // Px, n // Py), dtype=_A.dtype)
            lx = np.empty((m // Px, ), dtype=_x.dtype)
            dace.comm.BCScatter(_A, lA, (m // Px, n // Py))
            dace.comm.BCScatter(_x, lx, (m // Px, 1))
            ly = distr.MatMult(lx, lA, (m, n))
            dace.comm.BCGather(ly, _y, (n // Px, 1))

        # NOTE: The following is done to avoid scalar promotion, which results
        # in ValueError: Node type "BlockCyclicScatter" not supported for
        # promotion
        if transA:
            sdfg = _gemTv_pblas.to_sdfg(simplify=False)
        else:
            sdfg = _gemNv_pblas.to_sdfg(simplify=False)
        sdfg.simplify()
        return sdfg


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGemvPure,
        "OpenBLAS": ExpandGemvOpenBLAS,
        "MKL": ExpandGemvMKL,
        "cuBLAS": ExpandGemvCuBLAS,
        "PBLAS": ExpandGemvPBLAS
    }
    default_implementation = None

    # Object fields
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")

    n = properties.SymbolicProperty(allow_none=True, default=None)
    m = properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, location=None, transA=False, alpha=1, beta=0):
        super().__init__(name,
                         location=location,
                         inputs={"_A", "_x", "_y"} if beta != 0 else {"_A", "_x"},
                         outputs={"_y"})
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

        if len(size_a) != 2 or len(size_x) != 1:
            raise ValueError("Matrix-vector product only supported on matrix-vector input")

        a_cols = size_a[1] if not self.transA else size_a[0]
        a_rows = size_a[0] if not self.transA else size_a[1]

        if a_cols != size_x[0]:
            raise ValueError(f"Columns of A ({a_cols}) don't match "
                             f"size of x ({size_x[0]}).")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-vector product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_y_out = out_subset.size()
        if size_y_in is not None and size_y_in != size_y_out:
            raise ValueError("Input y-vector must match output y-vector.")
        if (len(size_y_out) != 1 or size_y_out[0] != a_rows):
            raise ValueError("Vector input to GEMV must match matrix rows.")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemv')
@oprepo.replaces('dace.libraries.blas.Gemv')
def gemv_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, A, x, y, alpha, beta, trans=None):
    # Get properties
    if trans is None:
        trans = (sdfg.arrays[x].shape[0] == sdfg.arrays[A].shape[0])

    # Add nodes
    A_in, x_in = (state.add_read(name) for name in (A, x))
    y_out = state.add_write(y)

    libnode = Gemv('gemv', transA=trans, alpha=alpha, beta=beta)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_y', y_out, None, mm.Memlet(y))

    if beta != 0:
        y_in = state.add_read(y)
        state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))

    return []
