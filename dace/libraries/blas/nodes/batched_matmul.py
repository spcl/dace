# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from dace import dtypes, memlet as mm, properties, data as dt
from typing import Any, Dict, Optional
from dace.symbolic import symstr
import dace.library
import dace.properties
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import (to_blastype, get_gemm_opts, check_access, dtype_to_cudadatatype,
                                              to_cublas_computetype)
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_batchmm_opts, _get_codegen_gemm_opts)
from .. import environments


@dace.library.expansion
class ExpandBatchedMatMulPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        # Get metadata from parent SDFG
        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b, shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)
        outedge = parent_state.out_edges(node)[0]
        cdesc = parent_sdfg.arrays[outedge.data.data]
        bopt = _get_batchmm_opts(shape_a, strides_a, shape_b, strides_b, cdesc.shape, cdesc.strides)

        if shape_a[-1] != shape_b[-2]:
            raise SyntaxError('Matrix sizes must match')
        if bopt:
            shape_c = (bopt['b'], shape_a[-2], shape_b[-1])
        else:
            shape_c = (shape_a[-2], shape_b[-1])

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = cdesc.dtype.type

        if outer_array_a.storage != outer_array_b.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        # Create replacement SDFG
        sdfg = dace.SDFG(node.label + "_sdfg")

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-1], storage=storage)

        # Add an initialization state
        init_state = sdfg.add_state()
        init_state.add_mapped_tasklet(
            'batched_matmul_init', {'_o%d' % i: '0:%s' % symstr(d)
                                    for i, d in enumerate(shape_c)}, {},
            'out = 0', {'out': dace.Memlet.simple('_c', ','.join(['_o%d' % i for i in range(len(shape_c))]))},
            external_edges=True)

        state = sdfg.add_state_after(init_state, node.label + "_state")

        state.add_mapped_tasklet(
            '_BatchedBatchedMatMult_', {
                '__i%d' % i: '0:%s' % s
                for i, s in enumerate([bopt['b'], array_a.shape[-2], array_b.shape[-1], array_a.shape[-1]])
            }, {
                '__a': dace.Memlet.simple("_a", ('__i1, __i3' if len(array_a.shape) == 2 else '__i0, __i1, __i3')),
                '__b': dace.Memlet.simple("_b", ('__i3, __i2' if len(array_b.shape) == 2 else '__i0, __i3, __i2'))
            },
            '__c = __a * __b', {'__c': dace.Memlet.simple("_c", '__i0, __i1, __i2', wcr_str='lambda x, y: x + y')},
            external_edges=True)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandBatchedMatMulPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandBatchedMatMulMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, astrides), (_, bdesc, bshape, bstrides), _ = _get_matmul_operands(node, state, sdfg)
        cdesc: dt.Array = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        dtype = cdesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + 'gemm'
        if dtype == dace.float32:
            alpha = "1.0f"
            beta = "0.0f"
            prefix = "s"
        elif dtype == dace.float64:
            alpha = "1.0"
            beta = "0.0"
            prefix = "d"
        elif dtype == dace.complex64:
            alpha = "dace::blas::BlasConstants::Get().Complex64Pone()"
            beta = "dace::blas::BlasConstants::Get().Complex64Zero()"
            prefix = "c"
        elif dtype == dace.complex128:
            alpha = "dace::blas::BlasConstants::Get().Complex128Pone()"
            beta = "dace::blas::BlasConstants::Get().Complex128Zero()"
            prefix = "z"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " + str(dtype))
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdesc.dtype.ctype, func)

        opt['prefix'] = prefix
        opt['dtype'] = cdesc.dtype.ctype

        code = '''
        const MKL_INT __group_count = 1;
        MKL_INT __group_sizes[__group_count] = {{ {BATCH} }};
        MKL_INT __m_array[__group_count] = {{ {M} }};
        MKL_INT __n_array[__group_count] = {{ {N} }};
        MKL_INT __k_array[__group_count] = {{ {K} }};
        char __transa[__group_count] = {{ '{ta}' }};
        char __transb[__group_count] = {{ '{tb}' }};
        {dtype} __alpha_array[__group_count] = {{ {alpha} }};
        {dtype} __beta_array[__group_count] = {{ {beta} }};
        MKL_INT __lda_array[__group_count] = {{ {lda} }};
        MKL_INT __ldb_array[__group_count] = {{ {ldb} }};
        MKL_INT __ldc_array[__group_count] = {{ {ldc} }};

        const {dtype}** __A = new const {dtype}*[{BATCH}];
        const {dtype}** __B = new const {dtype}*[{BATCH}];
        {dtype}** __C = new {dtype}*[{BATCH}];
        for (int __ib = 0; __ib < {BATCH}; __ib++) {{
            __A[__ib] = (({dtype}*){x}) + __ib*{stride_a};
            __B[__ib] = (({dtype}*){y}) + __ib*{stride_b};
            __C[__ib] = (({dtype}*)_c) + __ib*{stride_c};
        }}

        {prefix}gemm_batch(__transa, __transb, __m_array, __n_array, __k_array, __alpha_array, __A, __lda_array, __B, __ldb_array, __beta_array, __C, __ldc_array, &__group_count, __group_sizes);'''.format_map(
            opt)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandBatchedMatMulOpenBLAS(ExpandTransformation):
    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, astrides), (_, bdesc, bshape, bstrides), _ = _get_matmul_operands(node, state, sdfg)
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        dtype = cdesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + 'gemm'
        if dtype == dace.float32:
            alpha = "1.0f"
            beta = "0.0f"
        elif dtype == dace.float64:
            alpha = "1.0"
            beta = "0.0"
        elif dtype == dace.complex64:
            alpha = "dace::blas::BlasConstants::Get().Complex64Pone()"
            beta = "dace::blas::BlasConstants::Get().Complex64Zero()"
        elif dtype == dace.complex128:
            alpha = "dace::blas::BlasConstants::Get().Complex128Pone()"
            beta = "dace::blas::BlasConstants::Get().Complex128Zero()"
        else:
            raise ValueError("Unsupported type for BLAS dot product: " + str(dtype))
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdesc.dtype.ctype, func)

        # Adaptations for MKL/BLAS API
        opt['ta'] = 'CblasNoTrans' if opt['ta'] == 'N' else 'CblasTrans'
        opt['tb'] = 'CblasNoTrans' if opt['tb'] == 'N' else 'CblasTrans'

        code = '''
        for (int __ib = 0; __ib < {BATCH}; ++__ib) {{
            cblas_{func}(CblasColMajor, {ta}, {tb}, {M}, {N}, {K}, {alpha},
                         (({dtype}*){x}) + __ib*{stride_a}, {lda},
                         (({dtype}*){y}) + __ib*{stride_b}, {ldb},
                         {beta},
                         (({dtype}*)_c) + __ib*{stride_c}, {ldc});
        }}'''.format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandBatchedMatMulCuBLAS(ExpandTransformation):

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
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))

        dtype = cdesc.dtype.base_type
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
            1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Pone()",
            0.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Zero()",
        }
        if node.alpha not in constants:
            # Deal with complex input constants
            if isinstance(node.alpha, complex):
                alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
            else:
                alpha = f'{dtype.ctype}({node.alpha})'

            # Set pointer mode to host
            call_prefix += f'''cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_HOST);
                {dtype.ctype} alpha = {alpha};
                {dtype.ctype} beta = 0;
                '''
            call_suffix += '''
    cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
                '''
            beta = f'({cdtype} *)&beta'
            alpha = f'({cdtype} *)&alpha'
        else:
            alpha = constants[node.alpha]
            beta = "__state->cublas_handle.Constants(__dace_cuda_device).%sZero()" % factort

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['array_prefix'] = '_' if needs_copy else ''

        # Matrix multiplication
        if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
            call = '''cublas{func}StridedBatched(__dace_cublas_handle,
                CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
                {M}, {N}, {K},
                {alpha},
                ({dtype}*){array_prefix}{x}, {lda}, {stride_a},
                ({dtype}*){array_prefix}{y}, {ldb}, {stride_b},
                {beta},
                ({dtype}*){array_prefix}_c, {ldc}, {stride_c},
                {BATCH});'''.format_map(opt)
        else:
            if node.compute_type is not None:
                acctype = node.compute_type
            elif node.accumulator_type is not None:
                acc_dtype: dtypes.typeclass = node.accumulator_type
                acctype = f'CUBLAS_COMPUTE_{to_cublas_computetype(acc_dtype)}'
            else:
                acctype = f'CUBLAS_COMPUTE_{to_cublas_computetype(dtype)}'

            algorithm = 'CUBLAS_GEMM_DEFAULT_TENSOR_OP'
            if node.algorithm is not None:
                algorithm = node.algorithm

            call = f'''
            cublasGemmStridedBatchedEx(__dace_cublas_handle,
                CUBLAS_OP_{opt['ta']}, CUBLAS_OP_{opt['tb']},
                {opt['M']}, {opt['N']}, {opt['K']},
                {alpha},
                {opt['array_prefix']}{opt['x']},
                {dtype_to_cudadatatype(opt['xdtype'])},
                {opt['lda']}, {opt['stride_a']},
                {opt['array_prefix']}{opt['y']},
                {dtype_to_cudadatatype(opt['ydtype'])},
                {opt['ldb']}, {opt['stride_b']},
                {beta},
                {opt['array_prefix']}_c,
                {dtype_to_cudadatatype(opt['cdtype'])},
                {opt['ldc']}, {opt['stride_c']},
                {opt['BATCH']},
                {acctype}, {algorithm});
            '''

        code = call_prefix + call + call_suffix
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        # If buffers are not on the GPU, copy them
        if needs_copy:
            nsdfg = dace.SDFG('nested_batched_matmul')
            tasklet = dace.sdfg.nodes.Tasklet(node.name, {
                '__a': dtypes.pointer(adesc.dtype),
                '__b': dtypes.pointer(bdesc.dtype)
            }, {'__c': dtypes.pointer(cdesc.dtype)},
                                              code,
                                              language=dace.dtypes.Language.CPP)

            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.transient = False
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                nsdfg.add_datadesc(name, dcopy)
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
            nstate.add_node(tasklet)
            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))
            nstate.add_edge(ga, None, tasklet, '__a', dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, tasklet, '__b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '__c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.node
class BatchedMatMul(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandBatchedMatMulPure,
        "MKL": ExpandBatchedMatMulMKL,
        "OpenBLAS": ExpandBatchedMatMulOpenBLAS,
        "cuBLAS": ExpandBatchedMatMulCuBLAS
    }
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")
    algorithm = properties.Property(dtype=str,
                                    allow_none=True,
                                    default=None,
                                    desc="If applicable, chooses the vendor-provided implementation "
                                    "(algorithm) for the multiplication")
    accumulator_type = properties.TypeClassProperty(
        default=None,
        choices=dtypes.Typeclasses,
        allow_none=True,
        desc="Accumulator or intermediate storage type used in multiplication")
    compute_type = properties.Property(default=None,
                                       dtype=str,
                                       allow_none=True,
                                       desc="If applicable, overrides computation type (CUBLAS-specific, see "
                                       "``cublasComputeType_t``)")

    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs={'_a', '_b'}, outputs={'_c'})

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to batched matrix-matrix product")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from "
                             "batched matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if len(size0) not in [2, 3]:
            raise ValueError("Batched matrix-matrix product only supported on matrices")
        if len(size1) != 3:
            raise ValueError("Batched matrix-matrix product only supported on matrices")
        if size0[-1] != size1[-2]:
            raise ValueError("Inputs to matrix-matrix product "
                             "must agree in the k-dimension")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size2 = out_subset.size()
        if len(size2) != 3:
            raise ValueError("batched matrix-matrix product only supported on matrices")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.bmm')
def bmmnode(pv, sdfg: dace.SDFG, state: dace.SDFGState, A, B, C, alpha=1, beta=0, trans_a=False, trans_b=False):
    # Add nodes
    A_in, B_in = (state.add_read(name) for name in (A, B))
    C_out = state.add_write(C)

    libnode = BatchedMatMul('bmm')
    libnode.alpha = alpha
    libnode.beta = beta
    libnode.transA = trans_a
    libnode.transB = trans_b
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_a', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_b', mm.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, mm.Memlet(C))

    return []
