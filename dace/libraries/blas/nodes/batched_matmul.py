# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from math import prod
from dace import dtypes, memlet as mm, properties, data as dt
from dace.symbolic import symstr, equal
import dace.library
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, check_access, dtype_to_cudadatatype, to_cublas_computetype
from dace.libraries.blas.nodes.matmul import _get_matmul_operands, _get_batchmm_opts, _get_codegen_gemm_opts
from .. import environments
import warnings


@dace.library.expansion
class ExpandBatchedMatMulPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        # Get metadata from parent SDFG
        ((edge_a, outer_array_a, shape_a, strides_a, _, _), (edge_b, outer_array_b, shape_b, strides_b, _, _),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)
        outedge = parent_state.out_edges(node)[0]
        cdesc = parent_sdfg.arrays[outedge.data.data]
        bopt = _get_batchmm_opts(shape_a, strides_a, shape_b, strides_b, cdesc.shape, cdesc.strides)

        # Handle 1D inputs - determine dimensions
        is_a_1d = len(shape_a) == 1
        is_b_1d = len(shape_b) == 1

        if is_a_1d:
            # [k] treated as row vector for matmul
            m_dim = 1
            k_dim = shape_a[0]
        else:
            m_dim = shape_a[-2]
            k_dim = shape_a[-1]

        if is_b_1d:
            # [k] treated as column vector for matmul
            k_dim_b = shape_b[0]
            n_dim = 1
        else:
            k_dim_b = shape_b[-2]
            n_dim = shape_b[-1]

        res = equal(k_dim, k_dim_b)
        if res is None:
            warnings.warn(f"K-dimensions {k_dim} may not match {k_dim_b}", UserWarning)
        elif not res:
            raise SyntaxError(f"K-dimensions must match: {k_dim} vs {k_dim_b}")

        # Determine output shape - the actual output shape from cdesc
        shape_c = cdesc.shape

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
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-3], storage=storage)

        # Add an initialization state
        init_state = sdfg.add_state()
        init_state.add_mapped_tasklet(
            'batched_matmul_init', {
                '_o%d' % i: '0:%s' % symstr(d)
                for i, d in enumerate(shape_c)
            }, {},
            'out = 0', {'out': dace.Memlet.simple('_c', ','.join(['_o%d' % i for i in range(len(shape_c))]))},
            external_edges=True)

        state = sdfg.add_state_after(init_state, node.label + "_state")

        # Calculate number of batch dimensions in output
        # For 1D cases, output may have fewer dimensions
        # e.g., [3, 32, 64] @ [64] = [3, 32]
        if is_a_1d and is_b_1d:
            # [k] @ [k] = scalar, this shouldn't happen in batched context
            num_batch_dims = len(shape_c)
        elif is_a_1d:
            # [k] @ [batch..., k, n] = [batch..., n]
            num_batch_dims = len(shape_c) - 1  # All dims except N
        elif is_b_1d:
            # [batch..., m, k] @ [k] = [batch..., m]
            num_batch_dims = len(shape_c) - 1  # All dims except M
        else:
            # Regular case: [batch..., m, k] @ [batch..., k, n] = [batch..., m, n]
            num_batch_dims = len(shape_c) - 2

        # Build map parameters: batch dimensions + M, N, K
        map_params = {}
        for i in range(num_batch_dims):
            map_params['__i%d' % i] = '0:%s' % symstr(shape_c[i])

        # M, N, K dimensions - always create map parameters
        map_params['__im'] = '0:%s' % symstr(m_dim)
        map_params['__in'] = '0:%s' % symstr(n_dim)
        map_params['__ik'] = '0:%s' % symstr(k_dim)

        # Build memlet access patterns
        # Handle 1D inputs specially - they only have __ik dimension
        if is_a_1d:
            # [k] input - just use __ik
            memlet_a = '__ik'
        elif len(array_a.shape) == 2:
            # 2D input [M, K]
            memlet_a = '__im, __ik'
        else:
            # 3D+ input [batch..., M, K]
            # Align input batch dims to output batch dims
            num_a_batch = len(array_a.shape) - 2
            # Start from the rightmost batch dimension of output and work backwards
            offset = num_batch_dims - num_a_batch
            a_batch_indices = ', '.join(['__i%d' % (offset + i) for i in range(num_a_batch)])
            memlet_a = f'{a_batch_indices}, __im, __ik'

        # For B: if 1D, use [K]; if 2D, use [K, N]; if 3D+, use [batch_indices..., K, N]
        if is_b_1d:
            # [k] input - just use __ik
            memlet_b = '__ik'
        elif len(array_b.shape) == 2:
            # 2D input [K, N]
            memlet_b = '__ik, __in'
        else:
            # 3D+ input [batch..., K, N]
            # Align input batch dims to output batch dims
            num_b_batch = len(array_b.shape) - 2
            # Start from the rightmost batch dimension of output and work backwards
            offset = num_batch_dims - num_b_batch
            b_batch_indices = ', '.join(['__i%d' % (offset + i) for i in range(num_b_batch)])
            memlet_b = f'{b_batch_indices}, __ik, __in'

        # For C: build indices matching the output shape
        c_indices_parts = []
        if is_a_1d and is_b_1d:
            # Scalar output - all batch dims
            for i in range(num_batch_dims):
                c_indices_parts.append(f'__i{i}')
        elif is_a_1d:
            # [k] @ [batch..., k, n] = [batch..., n]
            # Output has batch dims + n
            for i in range(num_batch_dims):
                c_indices_parts.append(f'__i{i}')
            c_indices_parts.append('__in')
        elif is_b_1d:
            # [batch..., m, k] @ [k] = [batch..., m]
            # Output has batch dims + m
            for i in range(num_batch_dims):
                c_indices_parts.append(f'__i{i}')
            c_indices_parts.append('__im')
        else:
            # Regular: [batch..., m, k] @ [batch..., k, n] = [batch..., m, n]
            for i in range(num_batch_dims):
                c_indices_parts.append(f'__i{i}')
            c_indices_parts.append('__im')
            c_indices_parts.append('__in')
        c_indices = ', '.join(c_indices_parts)

        state.add_mapped_tasklet('_BatchedMatMult_',
                                 map_params, {
                                     '__a': dace.Memlet.simple("_a", memlet_a),
                                     '__b': dace.Memlet.simple("_b", memlet_b)
                                 },
                                 '__c = __a * __b',
                                 {'__c': dace.Memlet.simple("_c", c_indices, wcr_str='lambda x, y: x + y')},
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
    def _expand_gemv_loop(node, state, sdfg, adesc, bdesc, cdesc, ashape, bshape, astrides, bstrides, dtype, is_a_1d,
                          is_b_1d):
        """Expand batched matrix-vector or vector-matrix multiplication using GEMV loops."""
        from dace.codegen.common import sym2cpp

        prefix = to_blastype(dtype.type).lower()
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
            raise ValueError("Unsupported type for BLAS: " + str(dtype))

        # Determine batch size and strides
        cshape = cdesc.shape

        if is_a_1d and is_b_1d:
            # Both 1D - shouldn't happen in batched context
            raise ValueError("Both inputs are 1D - use dot product instead")
        elif is_a_1d:
            # [k] @ [batch..., k, n] = [batch..., n]
            batch_size = prod(cshape[:-1])
            k = ashape[0]
            n = bshape[-1]

            # Detect storage order from B's strides
            # B has shape [batch..., k, n] with strides [..., stride_k, stride_n]
            # Row-major: stride_k > stride_n (elements in same row are contiguous)
            # Column-major: stride_k < stride_n (elements in same column are contiguous)
            stride_k = bstrides[-2]
            stride_n = bstrides[-1]

            if stride_n == 1:  # Row-major: rightmost dimension has stride 1
                layout = 'CblasRowMajor'
                trans = 'CblasTrans'  # For a @ B[i], compute B[i]^T @ a
                ldb = n  # Leading dimension in row-major
                stride_b = k * n  # Stride between batches
            else:  # Column-major: leftmost matrix dimension has stride 1
                layout = 'CblasColMajor'
                trans = 'CblasTrans'
                ldb = k
                stride_b = k * n

            stride_c = n  # Output stride

        elif is_b_1d:
            # [batch..., m, k] @ [k] = [batch..., m]
            batch_size = prod(cshape[:-1])
            m = ashape[-2]
            k = ashape[-1]

            # Detect storage order from A's strides
            stride_k = astrides[-1]

            if stride_k == 1:  # Row-major
                layout = 'CblasRowMajor'
                trans = 'CblasNoTrans'  # For A[i] @ b, no transpose needed
                lda = k  # Leading dimension in row-major
                stride_a = m * k
            else:  # Column-major
                layout = 'CblasColMajor'
                trans = 'CblasNoTrans'
                lda = m
                stride_a = m * k

            stride_c = m  # Output stride
        else:
            raise ValueError("Unexpected case - neither input is 1D")

        # Generate code
        if is_a_1d:
            # [k] @ [batch..., k, n]: loop over batch, each time: c[i] = B[i]^T @ a
            code = f'''
            for (int __ib = 0; __ib < {sym2cpp(batch_size)}; ++__ib) {{
                cblas_{prefix}gemv({layout}, {trans}, {sym2cpp(k)}, {sym2cpp(n)},
                         {alpha},
                         (({dtype.ctype}*)_b) + __ib*{sym2cpp(stride_b)}, {sym2cpp(ldb)},
                         ({dtype.ctype}*)_a, 1,
                         {beta},
                         (({dtype.ctype}*)_c) + __ib*{sym2cpp(stride_c)}, 1);
            }}'''
        else:  # is_b_1d
            # [batch..., m, k] @ [k]: loop over batch, each time: c[i] = A[i] @ b
            code = f'''
            for (int __ib = 0; __ib < {sym2cpp(batch_size)}; ++__ib) {{
                cblas_{prefix}gemv({layout}, {trans}, {sym2cpp(m)}, {sym2cpp(k)},
                         {alpha},
                         (({dtype.ctype}*)_a) + __ib*{sym2cpp(stride_a)}, {sym2cpp(lda)},
                         ({dtype.ctype}*)_b, 1,
                         {beta},
                         (({dtype.ctype}*)_c) + __ib*{sym2cpp(stride_c)}, 1);
            }}'''

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, astrides, _, _), (_, bdesc, bshape, bstrides, _,
                                             _), _ = _get_matmul_operands(node, state, sdfg)
        cdesc: dt.Array = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        dtype = cdesc.dtype.base_type

        # Check if we have 1D inputs (vector operations)
        is_a_1d = len(ashape) == 1
        is_b_1d = len(bshape) == 1

        # For 1D cases, use GEMV instead of batched GEMM
        if is_a_1d or is_b_1d:
            return ExpandBatchedMatMulMKL._expand_gemv_loop(node, state, sdfg, adesc, bdesc, cdesc, ashape, bshape,
                                                            astrides, bstrides, dtype, is_a_1d, is_b_1d)

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
        const MKL_INT group_count = 1;
        MKL_INT group_sizes[group_count] = {{ {BATCH} }};
        MKL_INT m_array[group_count] = {{ {M} }};
        MKL_INT n_array[group_count] = {{ {N} }};
        MKL_INT k_array[group_count] = {{ {K} }};
        char transa[group_count] = {{ '{ta}' }};
        char transb[group_count] = {{ '{tb}' }};
        {dtype} alpha_array[group_count] = {{ {alpha} }};
        {dtype} beta_array[group_count] = {{ {beta} }};
        MKL_INT lda_array[group_count] = {{ {lda} }};
        MKL_INT ldb_array[group_count] = {{ {ldb} }};
        MKL_INT ldc_array[group_count] = {{ {ldc} }};

        const {dtype}** __mkl_BMM_A = new const {dtype}*[{BATCH}];
        const {dtype}** __mkl_BMM_B = new const {dtype}*[{BATCH}];
        {dtype}** __mkl_BMM_C = new {dtype}*[{BATCH}];
        for (int __ib = 0; __ib < {BATCH}; __ib++) {{
            // Handle broadcasting - compute correct index for inputs with fewer batch dimensions
            int __a_idx = ({stride_a} > 0) ? (({a_batch_size} < {BATCH}) ? (__ib % {a_batch_size}) : __ib) : 0;
            int __b_idx = ({stride_b} > 0) ? (({b_batch_size} < {BATCH}) ? (__ib % {b_batch_size}) : __ib) : 0;
            __mkl_BMM_A[__ib] = (({dtype}*){x}) + __a_idx*{stride_a};
            __mkl_BMM_B[__ib] = (({dtype}*){y}) + __b_idx*{stride_b};
            __mkl_BMM_C[__ib] = (({dtype}*)_c) + __ib*{stride_c};
        }}

        {prefix}gemm_batch(transa, transb, m_array, n_array, k_array, alpha_array, __mkl_BMM_A, lda_array, __mkl_BMM_B, ldb_array, beta_array, __mkl_BMM_C, ldc_array, &group_count, group_sizes);

        delete[] __mkl_BMM_A;
        delete[] __mkl_BMM_B;
        delete[] __mkl_BMM_C;
        '''.format_map(opt)

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
    def _expand_gemv_loop(node, state, sdfg, adesc, bdesc, cdesc, ashape, bshape, astrides, bstrides, dtype, is_a_1d,
                          is_b_1d):
        """Expand batched matrix-vector or vector-matrix multiplication using GEMV loops."""
        from dace.codegen.common import sym2cpp

        prefix = to_blastype(dtype.type).lower()
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
            raise ValueError("Unsupported type for BLAS: " + str(dtype))

        # Determine batch size and strides
        cshape = cdesc.shape

        if is_a_1d and is_b_1d:
            # Both 1D - shouldn't happen in batched context
            raise ValueError("Both inputs are 1D - use dot product instead")
        elif is_a_1d:
            # [k] @ [batch..., k, n] = [batch..., n]
            batch_size = prod(cshape[:-1])
            k = ashape[0]
            n = bshape[-1]

            # Detect storage order from B's strides
            # B has shape [batch..., k, n] with strides [..., stride_k, stride_n]
            # Row-major: stride_k > stride_n (elements in same row are contiguous)
            # Column-major: stride_k < stride_n (elements in same column are contiguous)
            stride_k = bstrides[-2]
            stride_n = bstrides[-1]

            if stride_n == 1:  # Row-major: rightmost dimension has stride 1
                layout = 'CblasRowMajor'
                trans = 'CblasTrans'  # For a @ B[i], compute B[i]^T @ a
                ldb = n  # Leading dimension in row-major
                stride_b = k * n  # Stride between batches
            else:  # Column-major: leftmost matrix dimension has stride 1
                layout = 'CblasColMajor'
                trans = 'CblasTrans'
                ldb = k
                stride_b = k * n

            stride_c = n  # Output stride

        elif is_b_1d:
            # [batch..., m, k] @ [k] = [batch..., m]
            batch_size = prod(cshape[:-1])
            m = ashape[-2]
            k = ashape[-1]

            # Detect storage order from A's strides
            stride_k = astrides[-1]

            if stride_k == 1:  # Row-major
                layout = 'CblasRowMajor'
                trans = 'CblasNoTrans'  # For A[i] @ b, no transpose needed
                lda = k  # Leading dimension in row-major
                stride_a = m * k
            else:  # Column-major
                layout = 'CblasColMajor'
                trans = 'CblasNoTrans'
                lda = m
                stride_a = m * k

            stride_c = m  # Output stride
        else:
            raise ValueError("Unexpected case - neither input is 1D")

        # Generate code
        if is_a_1d:
            # [k] @ [batch..., k, n]: loop over batch, each time: c[i] = B[i]^T @ a
            code = f'''
            for (int __ib = 0; __ib < {sym2cpp(batch_size)}; ++__ib) {{
                cblas_{prefix}gemv({layout}, {trans}, {sym2cpp(k)}, {sym2cpp(n)},
                         {alpha},
                         (({dtype.ctype}*)_b) + __ib*{sym2cpp(stride_b)}, {sym2cpp(ldb)},
                         ({dtype.ctype}*)_a, 1,
                         {beta},
                         (({dtype.ctype}*)_c) + __ib*{sym2cpp(stride_c)}, 1);
            }}'''
        else:
            # [batch..., m, k] @ [k]: loop over batch, each time: c[i] = A[i] @ b
            code = f'''
            for (int __ib = 0; __ib < {sym2cpp(batch_size)}; ++__ib) {{
                cblas_{prefix}gemv({layout}, {trans}, {sym2cpp(m)}, {sym2cpp(k)},
                         {alpha},
                         (({dtype.ctype}*)_a) + __ib*{sym2cpp(stride_a)}, {sym2cpp(lda)},
                         ({dtype.ctype}*)_b, 1,
                         {beta},
                         (({dtype.ctype}*)_c) + __ib*{sym2cpp(stride_c)}, 1);
            }}'''

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, astrides, _, _), (_, bdesc, bshape, bstrides, _,
                                             _), _ = _get_matmul_operands(node, state, sdfg)
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)
        dtype = cdesc.dtype.base_type

        # Check if we have 1D inputs (vector operations)
        is_a_1d = len(ashape) == 1
        is_b_1d = len(bshape) == 1

        # For 1D cases, use GEMV instead of batched GEMM
        if is_a_1d or is_b_1d:
            return ExpandBatchedMatMulOpenBLAS._expand_gemv_loop(node, state, sdfg, adesc, bdesc, cdesc, ashape, bshape,
                                                                 astrides, bstrides, dtype, is_a_1d, is_b_1d)

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
            // Handle broadcasting - compute correct index for inputs with fewer batch dimensions
            int __a_idx = ({stride_a} > 0) ? (({a_batch_size} < {BATCH}) ? (__ib % {a_batch_size}) : __ib) : 0;
            int __b_idx = ({stride_b} > 0) ? (({b_batch_size} < {BATCH}) ? (__ib % {b_batch_size}) : __ib) : 0;
            cblas_{func}(CblasColMajor, {ta}, {tb}, {M}, {N}, {K}, {alpha},
                         (({dtype}*){x}) + __a_idx*{stride_a}, {lda},
                         (({dtype}*){y}) + __b_idx*{stride_b}, {ldb},
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

        # Check if we need broadcasting (non-uniform strides)
        needs_broadcasting = (opt.get('a_batch_size') and opt.get('b_batch_size')
                              and (opt['a_batch_size'] != opt['BATCH'] or opt['b_batch_size'] != opt['BATCH']))

        # Matrix multiplication
        if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
            if needs_broadcasting:
                # Use manual loop for broadcasting cases
                call = '''
                for (int __ib = 0; __ib < {BATCH}; ++__ib) {{
                    int __a_idx = ({stride_a} > 0) ? (({a_batch_size} < {BATCH}) ? (__ib % {a_batch_size}) : __ib) : 0;
                    int __b_idx = ({stride_b} > 0) ? (({b_batch_size} < {BATCH}) ? (__ib % {b_batch_size}) : __ib) : 0;
                    cublas{func}(__dace_cublas_handle,
                        CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
                        {M}, {N}, {K},
                        {alpha},
                        ({dtype}*){array_prefix}{x} + __a_idx*{stride_a}, {lda},
                        ({dtype}*){array_prefix}{y} + __b_idx*{stride_b}, {ldb},
                        {beta},
                        ({dtype}*){array_prefix}_c + __ib*{stride_c}, {ldc});
                }}'''.format_map(opt)
            else:
                # Use StridedBatched for uniform case
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

            if needs_broadcasting:
                # Use manual loop for broadcasting cases with GemmEx
                call = f'''
                for (int __ib = 0; __ib < {opt['BATCH']}; ++__ib) {{{{
                    int __a_idx = ({opt['stride_a']} > 0) ? (({opt['a_batch_size']} < {opt['BATCH']}) ? (__ib % {opt['a_batch_size']}) : __ib) : 0;
                    int __b_idx = ({opt['stride_b']} > 0) ? (({opt['b_batch_size']} < {opt['BATCH']}) ? (__ib % {opt['b_batch_size']}) : __ib) : 0;
                    cublasGemmEx(__dace_cublas_handle,
                        CUBLAS_OP_{opt['ta']}, CUBLAS_OP_{opt['tb']},
                        {opt['M']}, {opt['N']}, {opt['K']},
                        {alpha},
                        {opt['array_prefix']}{opt['x']} + __a_idx*{opt['stride_a']},
                        {dtype_to_cudadatatype(opt['xdtype'])},
                        {opt['lda']},
                        {opt['array_prefix']}{opt['y']} + __b_idx*{opt['stride_b']},
                        {dtype_to_cudadatatype(opt['ydtype'])},
                        {opt['ldb']},
                        {beta},
                        {opt['array_prefix']}_c + __ib*{opt['stride_c']},
                        {dtype_to_cudadatatype(opt['cdtype'])},
                        {opt['ldc']},
                        {acctype}, {algorithm});
                }}}}
                '''
            else:
                # Use StridedBatchedEx for uniform case
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
                # TODO: Make a seprate PR to remove the squeezing
                # subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                # TODO: Make a seprate PR to remove the squeezing
                # subset.squeeze()
                size1 = subset.size()
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from "
                             "batched matrix-matrix product")
        out_memlet = out_edges[0].data

        # Valid cases: 1D@ND (N>=3), ND@1D (N>=3), ND@MD (N or M >=3)
        if len(size0) < 1 or len(size1) < 1:
            raise ValueError("Inputs must be at least 1D")

        # For batched operations, we need at least one operand to be 3D+ or one to be 1D
        has_1d = (len(size0) == 1 or len(size1) == 1)
        has_batch = (len(size0) >= 3 or len(size1) >= 3)

        if not has_1d and not has_batch and not (len(size0) == 2 and len(size1) == 2):
            # This would be just regular 2D@2D which isn't batched
            raise ValueError(
                "Batched operation requires at least one input to be 1D or have batch dimensions (3D or higher)")

        # Validate K-dimension compatibility
        # For 1D inputs, the single dimension is the k-dimension
        # For 2D+ inputs with matrix structure [..., M, K] or [..., K, N]
        k_dim_a = size0[0] if len(size0) == 1 else size0[-1]
        k_dim_b = size1[0] if len(size1) == 1 else size1[-2]

        res = equal(k_dim_a, k_dim_b)
        if res is None:
            warnings.warn(
                f'K-dimension of first operand {k_dim_a} and k-dimension of second operand {k_dim_b} '
                f'may not match', UserWarning)
        elif not res:
            raise ValueError(f"Inputs must agree in the k-dimension: {k_dim_a} vs {k_dim_b}")


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
