# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, Dict, Optional
from dace.data import Array
from dace import dtypes, memlet as mm, properties
from dace.symbolic import symstr
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, get_gemm_opts, check_access
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands,
                                              _get_codegen_gemm_opts)
from .. import environments
import numpy as np
from numbers import Number


def _is_complex(dtype):
    if hasattr(dtype, "is_complex") and callable(dtype.is_complex):
        return dtype.is_complex()
    else:
        return dtype in [np.complex64, np.complex128]


def _cast_to_dtype_str(value, dtype: dace.dtypes.typeclass) -> str:
    if _is_complex(dtype) and _is_complex(type(value)):
        raise ValueError("Cannot use complex beta with non-complex array")

    if _is_complex(dtype):
        cast_value = complex(value)

        return "dace.{type}({real}, {imag})".format(
            type=dace.DTYPE_TO_TYPECLASS[dtype].to_string(),
            real=cast_value.real,
            imag=cast_value.imag,
        )
    else:
        return "dace.{}({})".format(dace.DTYPE_TO_TYPECLASS[dtype].to_string(),
                                    value)


@dace.library.expansion
class ExpandGemmPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b,
                                                       shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_b).type]

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        if (len(trans_shape_a) != 2 or len(trans_shape_b) != 2
                or trans_shape_a[1] != trans_shape_b[0]):
            raise SyntaxError("Matrix sizes must match")
        M, K, N = trans_shape_a[0], trans_shape_a[1], trans_shape_b[1]
        shape_c = (M, N)

        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a",
                                    shape_a,
                                    dtype_a,
                                    strides=strides_a,
                                    storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b",
                                    shape_b,
                                    dtype_b,
                                    strides=strides_b,
                                    storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c",
                                    shape_c,
                                    dtype_c,
                                    strides=cdata[-1],
                                    storage=cdata[1].storage)

        if node.alpha == 1.0:
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(
                _cast_to_dtype_str(node.alpha, dtype_a))

        if node.beta == 1:
            state = sdfg.add_state(node.label + "_state")
        else:
            init_state = sdfg.add_state(node.label + "_initstate")
            state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta != 0:
            sdfg.add_array("_cin",
                           shape_c,
                           dtype_c,
                           strides=cdata[-1],
                           storage=cdata[1].storage)

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        # Initialization / beta map
        if node.beta == 0:
            init_state.add_mapped_tasklet(
                'gemm_init',
                {'_o%d' % i: '0:%s' % symstr(d)
                 for i, d in enumerate(shape_c)}, {},
                'out = 0', {
                    'out':
                    dace.Memlet.simple(
                        mul_out, ','.join(
                            ['_o%d' % i for i in range(len(shape_c))]))
                },
                external_edges=True)
        elif node.beta == 1:
            # Do nothing for initialization, only update the values
            pass
        else:
            # Beta map
            add_program = "__y = ({} * __c)".format(
                _cast_to_dtype_str(node.beta, dtype_a))

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
                    "Could not broadcast input _c to ({}, {})".format(M, N))

            init_state.add_mapped_tasklet(
                "gemm_init",
                {"__i%d" % i: "0:%s" % s
                 for i, s in enumerate([M, N])}, {
                     "__c": dace.Memlet.simple("_cin", memlet_idx),
                 },
                add_program, {"__y": dace.Memlet.simple("_c", "__i0, __i1")},
                external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet(
            "gemm", {"__i%d" % i: "0:%s" % s
                     for i, s in enumerate([M, N, K])},
            {
                "__a":
                dace.Memlet.simple(
                    "_a", "__i2, __i0" if node.transA else "__i0, __i2"),
                "__b":
                dace.Memlet.simple(
                    "_b", "__i1, __i2" if node.transB else "__i2, __i1")
            },
            mul_program, {
                "__out":
                dace.Memlet.simple(
                    mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandGemmPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandGemmOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape,
         astrides), (_, bdesc, bshape,
                     bstrides), _ = _get_matmul_operands(node, state, sdfg)
        dtype = adesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + 'gemm'
        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'

        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)

        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc,
                                     alpha, beta, dtype.ctype, func)

        # Adaptations for BLAS API
        opt['ta'] = 'CblasNoTrans' if opt['ta'] == 'N' else 'CblasTrans'
        opt['tb'] = 'CblasNoTrans' if opt['tb'] == 'N' else 'CblasTrans'

        code = ''
        if dtype in (dace.complex64, dace.complex128):
            code = f'''
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            opt['alpha'] = '&alpha'
            opt['beta'] = '&beta'

        code += ("cblas_{func}(CblasColMajor, {ta}, {tb}, "
                 "{M}, {N}, {K}, {alpha}, {x}, {lda}, {y}, {ldb}, {beta}, "
                 "_c, {ldc});").format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandGemmMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGemmOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandGemmCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == '_a':
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        check_access(dtypes.ScheduleType.GPU_Default, adesc, bdesc, cdesc)

        dtype = adesc.dtype.base_type
        func = '%sgemm' % to_blastype(dtype.type)
        if dtype == dace.float16:
            cdtype = '__half'
            factort = 'Half'
        elif dtype == dace.float32:
            cdtype = 'float'
            factort = 'Float'
        elif dtype == dace.float64:
            cdtype = 'double'
            factort = 'Double'
        elif dtype == dace.complex64:
            cdtype = 'cuComplex'
            factort = 'Complex64'
        elif dtype == dace.complex128:
            cdtype = 'cuDoubleComplex'
            factort = 'Complex128'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        call_prefix = environments.cublas.cuBLAS.handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        constants = {
            1.0:
            f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Pone()",
            #-1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Mone()",
            0.0:
            f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Zero()",
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
            alpha = f'({cdtype} *)&alpha'
            beta = f'({cdtype} *)&beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc,
                                     alpha, beta, cdtype, func)

        # Matrix multiplication
        call = '''cublas{func}(__dace_cublas_handle,
            CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
            {M}, {N}, {K},
            {alpha},
            ({dtype}*){x}, {lda},
            ({dtype}*){y}, {ldb},
            {beta},
            ({dtype}*)_c, {ldc});'''

        code = (call_prefix + call.format_map(opt) + call_suffix)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        # If buffers are not on the GPU, copy them
        if any(desc.storage not in
               [dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned]
               for desc in [adesc, bdesc, cdesc]):
            nsdfg = dace.SDFG('nested_gemm')
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                dcopy = dc(desc)
                dcopy.transient = False
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu = dc(desc)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
            nstate = nsdfg.add_state()
            a = nstate.add_read('_a')
            ga = nstate.add_access('_a_gpu')
            b = nstate.add_read('_b')
            gb = nstate.add_access('_b_gpu')
            c = nstate.add_write('_c')
            gc = nstate.add_access('_c_gpu')

            # Reset code and connectors
            tasklet.in_connectors = {
                "_conn" + k: None
                for k in tasklet.in_connectors
            }
            tasklet.out_connectors = {
                "_conn" + k: None
                for k in tasklet.out_connectors
            }

            call = '''cublas{func}(__dace_cublas_handle,
                CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
                {M}, {N}, {K},
                {alpha},
                ({dtype}*){x}, {lda},
                ({dtype}*){y}, {ldb},
                {beta},
                ({dtype}*)_conn_c, {ldc});'''
            opt['x'] = '_conn' + opt['x']
            opt['y'] = '_conn' + opt['y']
            tasklet.code.as_string = (call_prefix + call.format_map(opt) +
                                      call_suffix)

            nstate.add_node(tasklet)
            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(ga, None, tasklet, '_conn_a',
                            dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, tasklet, '_conn_b',
                            dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '_conn_c', gc, None,
                            dace.Memlet.from_array('_c_gpu', cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                rc = nstate.add_read('_c')
                rgc = nstate.add_access('_c_gpu')
                tasklet.add_in_connector('_conn_cin')
                nstate.add_nedge(rc, rgc, dace.Memlet('_c'))
                nstate.add_edge(rgc, None, tasklet, '_conn_cin',
                                dace.Memlet('_c_gpu'))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.node
class Gemm(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ B) + beta * C. C should be unidirectionally
       broadcastable (ONNX terminology) to A @ B.
    """

    # Global properties
    implementations = {
        "pure": ExpandGemmPure,
        "MKL": ExpandGemmMKL,
        "OpenBLAS": ExpandGemmOpenBLAS,
        "cuBLAS": ExpandGemmCuBLAS
    }
    default_implementation = None

    # Object fields
    transA = properties.Property(
        dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(
        dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(
        allow_none=False,
        default=1,
        desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(
        allow_none=False,
        default=0,
        desc="A scalar which will be multiplied with C before adding C")

    def __init__(self,
                 name,
                 location=None,
                 transA=False,
                 transB=False,
                 alpha=1,
                 beta=0):
        super().__init__(
            name,
            location=location,
            inputs=({"_a", "_b", "_cin"} if beta != 0 else {"_a", "_b"}),
            outputs={"_c"})
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to gemm")
        size2 = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
            if dst_conn == '_c':
                subset = dc(memlet.subset)
                subset.squeeze()
                size2 = subset.size()

        if self.transA:
            size0 = list(reversed(size0))
        if self.transB:
            size1 = list(reversed(size1))

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if len(size0) != 2 or len(size1) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if size0[1] != size1[0]:
            raise ValueError("Inputs to matrix-matrix product "
                             "must agree in the k-dimension")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size3 = out_subset.size()
        if size2 is not None and size2 != size3:
            raise ValueError("Input C matrix must match output matrix.")
        if len(size3) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if len(size3) == 2 and list(size3) != [size0[-2], size1[-1]]:
            raise ValueError(
                "Output to matrix-matrix product must agree in the m and n "
                "dimensions")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemm')
@oprepo.replaces('dace.libraries.blas.Gemm')
def gemv_libnode(sdfg: SDFG,
                 state: SDFGState,
                 A,
                 B,
                 C,
                 alpha,
                 beta,
                 trans_a=False,
                 trans_b=False):
    # Add nodes
    A_in, B_in = (state.add_read(name) for name in (A, B))
    C_out = state.add_write(C)

    libnode = Gemm('gemm',
                   transA=trans_a,
                   transB=trans_b,
                   alpha=alpha,
                   beta=beta)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_a', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_b', mm.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, mm.Memlet(C))

    if beta != 0:
        C_in = state.add_read(C)
        state.add_edge(C_in, None, libnode, '_cin', mm.Memlet(C))

    return []
