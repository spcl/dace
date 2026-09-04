# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from dace import dtypes, memlet as mm, properties, data as dt
from dace.symbolic import symstr, equal
import dace.library
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, check_access, dtype_to_cudadatatype, to_cublas_computetype
from dace.libraries.blas.nodes.matmul import _get_matmul_operands, _get_batchmm_opts, _get_codegen_gemm_opts
from .. import environments
from dace.libraries.blas import gpu_dialect
import warnings
from dace.ordered import OrderedSet


def refuse_broadcast_batches(node, state, sdfg) -> None:
    """Refuse a broadcast batch dimension for the fixed-stride BLAS lowerings.

    ``cublas*StridedBatched`` and ``*gemm_batch`` advance each operand by one matrix per batch
    element. A dimension of extent 1 on one side must instead be re-read for every element, which
    no single stride expresses, so such a call would read past the end of the smaller operand.
    The pure expansion handles it by indexing that dimension at 0.

    Refused on the dimensions where broadcasting is what the pairing MEANS: one side provably 1,
    or the two provably different. Two dimensions whose equality is merely unprovable -- distinct
    symbols, ``a: [B, M, K] @ b: [L, K, N]`` -- are not a broadcast. numpy pairs those only when
    they are equal at run time, the same thing the frontend assumed when it gave the result its
    batch shape; refusing them sends every symbolically-batched matmul to a failed expansion.

    :param node: The library node being expanded.
    :param state: The state the node lives in.
    :param sdfg: The SDFG the state belongs to.
    :raise ValueError: If either operand carries a batch dimension that must broadcast.
    """
    (_, _, shape_a, _, _, _), (_, _, shape_b, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
    batch_a, batch_b = shape_a[:-2], shape_b[:-2]
    if not batch_a or not batch_b:
        return
    offset = len(batch_a) - len(batch_b)
    paired = zip(batch_a[offset:], batch_b) if offset >= 0 else zip(batch_a, batch_b[-offset:])
    for d0, d1 in paired:
        same = equal(d0, d1)
        if same is True:
            continue
        if same is False or equal(d0, 1) is True or equal(d1, 1) is True:
            raise ValueError(f'{type(node).__name__} cannot broadcast batch dimensions {batch_a} against {batch_b}: '
                             'this expansion walks a fixed batch stride. Use the "pure" implementation, or '
                             'materialize both operands at the same batch shape.')


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

        res = equal(shape_a[-1], shape_b[-2])
        if res is None:
            warnings.warn(f"First matrix columns {shape_a[-1]} may not match second matrix rows {shape_b[-2]}",
                          UserWarning)
        elif not res:
            raise SyntaxError("Matrix sizes must match")

        # Determine output shape based on batch options
        if bopt:
            # Use batch dimensions from bopt (may be multi-dimensional)
            batch_dims = bopt.get('batch_dims', [bopt['b']])
            shape_c = tuple(batch_dims) + (shape_a[-2], shape_b[-1])
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
        num_batch_dims = len(shape_c) - 2

        # Build map parameters: batch dimensions + M, N, K
        map_params = {}
        for i in range(num_batch_dims):
            map_params['__i%d' % i] = '0:%s' % symstr(shape_c[i])

        # M, N, K dimensions
        map_params['__im'] = '0:%s' % symstr(shape_a[-2])
        map_params['__in'] = '0:%s' % symstr(shape_b[-1])
        map_params['__ik'] = '0:%s' % symstr(shape_a[-1])

        def batch_indices(operand_shape) -> str:
            """The operand's batch subscripts, following NumPy's broadcasting rules.

            Batch dimensions align to the RIGHT against the output's, and a dimension of extent 1
            broadcasts, so it is read at 0 for every output batch element. Taking the output index
            for such a dimension -- or aligning from the left -- reads past the end of the operand.
            """
            num_operand_batch = len(operand_shape) - 2
            offset = num_batch_dims - num_operand_batch
            subscripts = []
            for j, dim in enumerate(operand_shape[:-2]):
                broadcast = equal(dim, 1) is True
                subscripts.append('0' if broadcast else '__i%d' % (offset + j))
            return ', '.join(subscripts)

        # For A: if 2D, use [M, K]; if 3D+, use [batch_indices..., M, K]
        if len(array_a.shape) == 2:
            memlet_a = '__im, __ik'
        else:
            memlet_a = f'{batch_indices(array_a.shape)}, __im, __ik'

        # For B: if 2D, use [K, N]; if 3D+, use [batch_indices..., K, N]
        if len(array_b.shape) == 2:
            memlet_b = '__ik, __in'
        else:
            memlet_b = f'{batch_indices(array_b.shape)}, __ik, __in'

        # For C: always has batch dimensions
        c_indices = ', '.join(['__i%d' % i for i in range(num_batch_dims)]) + ', __im, __in'

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
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        refuse_broadcast_batches(node, state, sdfg)
        (_, adesc, ashape, astrides, _, _), (_, bdesc, bshape, bstrides, _,
                                             _), _ = _get_matmul_operands(node, state, sdfg)
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
            __mkl_BMM_A[__ib] = (({dtype}*){x}) + __ib*{stride_a};
            __mkl_BMM_B[__ib] = (({dtype}*){y}) + __ib*{stride_b};
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
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        refuse_broadcast_batches(node, state, sdfg)
        (_, adesc, ashape, astrides, _, _), (_, bdesc, bshape, bstrides, _,
                                             _), _ = _get_matmul_operands(node, state, sdfg)
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
class ExpandBatchedMatMulGPUBLAS(ExpandTransformation):

    environments = []

    @classmethod
    def expansion(cls, node, state, sdfg):
        node.validate(sdfg, state)
        refuse_broadcast_batches(node, state, sdfg)

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

        call_prefix = cls.environments[0].handle_setup_code(node)
        call_suffix = ''
        # Handle alpha / beta
        constants = {
            1.0: f"__state->{cls.dialect.handle_field}.Constants().{factort}Pone()",
            0.0: f"__state->{cls.dialect.handle_field}.Constants().{factort}Zero()",
        }
        if node.alpha not in constants:
            # Deal with complex input constants
            if isinstance(node.alpha, complex):
                alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
            else:
                alpha = f'{dtype.ctype}({node.alpha})'

            # Set pointer mode to host
            call_prefix += f'''{cls.dialect.check_error}(
                {cls.dialect.set_pointer_mode}({cls.dialect.handle}, {cls.dialect.pointer_host}));
                {dtype.ctype} alpha = {alpha};
                {dtype.ctype} beta = 0;
                '''
            call_suffix += f'''
    {cls.dialect.check_error}({cls.dialect.set_pointer_mode}({cls.dialect.handle}, {cls.dialect.pointer_device}));
                '''
            beta = f'({cdtype} *)&beta'
            alpha = f'({cdtype} *)&alpha'
        else:
            alpha = constants[node.alpha]
            beta = f"__state->{cls.dialect.handle_field}.Constants().{factort}Zero()"

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['array_prefix'] = '_' if needs_copy else ''

        # Matrix multiplication
        if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
            call = f'''{cls.dialect.check_error}({cls.dialect.strided_batched(func)}({cls.dialect.handle},
                {cls.dialect.op(opt['ta'])}, {cls.dialect.op(opt['tb'])},
                {opt['M']}, {opt['N']}, {opt['K']},
                {opt['alpha']},
                ({opt['dtype']}*){opt['array_prefix']}{opt['x']}, {opt['lda']}, {opt['stride_a']},
                ({opt['dtype']}*){opt['array_prefix']}{opt['y']}, {opt['ldb']}, {opt['stride_b']},
                {opt['beta']},
                ({opt['dtype']}*){opt['array_prefix']}_c, {opt['ldc']}, {opt['stride_c']},
                {opt['BATCH']}));'''
        else:
            # The mixed-precision path names cuBLAS COMPUTE and ALGO enums and the CUDA-wide
            # `cudaDataType`. rocBLAS spells all three differently (`rocblas_datatype_*`), and this
            # module has no mapping for them, so a rocBLAS build must REFUSE here rather than emit
            # cuBLAS identifiers that no ROCm toolchain can resolve.
            if cls.dialect is not gpu_dialect.CUBLAS:
                raise NotImplementedError(
                    f"{cls.dialect.name} has no mixed-precision batched GEMM in this expansion: the "
                    "compute type, the algorithm and the operand data types are all named with cuBLAS "
                    "enums here. Use a uniform dtype, or the 'pure' expansion.")
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
            {cls.dialect.check_error}(cublasGemmStridedBatchedEx({cls.dialect.handle},
                {cls.dialect.op(opt['ta'])}, {cls.dialect.op(opt['tb'])},
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
                {acctype}, {algorithm}));
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


@dace.library.expansion
class ExpandBatchedMatMulCuBLAS(ExpandBatchedMatMulGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    dialect = gpu_dialect.CUBLAS


@dace.library.expansion
class ExpandBatchedMatMulRocBLAS(ExpandBatchedMatMulGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    dialect = gpu_dialect.ROCBLAS


@dace.library.node
class BatchedMatMul(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandBatchedMatMulPure,
        "MKL": ExpandBatchedMatMulMKL,
        "OpenBLAS": ExpandBatchedMatMulOpenBLAS,
        "cuBLAS": ExpandBatchedMatMulCuBLAS,
        "rocBLAS": ExpandBatchedMatMulRocBLAS
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
        default=None, allow_none=True, desc="Accumulator or intermediate storage type used in multiplication")
    compute_type = properties.Property(default=None,
                                       dtype=str,
                                       allow_none=True,
                                       desc="If applicable, overrides computation type (CUBLAS-specific, see "
                                       "``cublasComputeType_t``)")

    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs=OrderedSet(('_a', '_b')), outputs={'_c'})

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to batched matrix-matrix product")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
                full0 = memlet.subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
                full1 = memlet.subset.size()
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from "
                             "batched matrix-matrix product")
        out_memlet = out_edges[0].data

        # Both inputs must be at least 2D
        if len(size0) < 2:
            raise ValueError(f"First input must be at least 2D, got shape with {len(size0)} dimensions")
        if len(size1) < 2:
            raise ValueError(f"Second input must be at least 2D, got shape with {len(size1)} dimensions")

        # At least one input must have batch dimensions (3D or higher) for batched operation
        if len(size0) <= 2 and len(size1) <= 2:
            raise ValueError(
                "Batched matrix-matrix product requires at least one input to have batch dimensions (3D or higher)")

        # Batch dimensions follow NumPy's broadcasting rules: they align to the RIGHT, and a
        # dimension of extent 1 stretches. Only a genuine mismatch is rejected here -- whether a
        # given expansion can express a broadcast is the expansion's business, not the node's
        # (the pure lowering indexes a broadcast dimension at 0; the BLAS ones walk a fixed batch
        # stride and refuse, see ``refuse_broadcast_batches``).
        # ``squeeze()`` drops EVERY extent-1 dimension and cannot tell an index singleton from a
        # slice singleton, so a broadcast batch axis disappears from ``size0``/``size1`` before it
        # can be examined -- ``(2,1,M,K) @ (1,3,K,N)`` arrives here looking like batch 2 vs batch 3.
        # The batch analysis therefore reads the UNSQUEEZED extents.
        batch0, batch1 = full0[:-2], full1[:-2]
        if batch0 and batch1:
            offset = len(batch0) - len(batch1)
            paired = zip(batch0[offset:], batch1) if offset >= 0 else zip(batch0, batch1[-offset:])
            for d0, d1 in paired:
                if equal(d0, d1) is False and equal(d0, 1) is not True and equal(d1, 1) is not True:
                    raise ValueError(f'Batch dimensions of the two inputs do not broadcast, got {batch0} and '
                                     f'{batch1}: {d0} and {d1} differ and neither is 1')

        # Validate K-dimension compatibility
        res = equal(size0[-1], size1[-2])
        if res is None:
            warnings.warn(
                f'First tensor\'s last mode {size0[-1]} and second tensor\'s second-last mode {size1[-2]} '
                f'may not match', UserWarning)
        elif not res:
            raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")

        # Output must have batch dimensions
        if len(out_memlet.subset) < 3:
            raise ValueError(
                f"Batched matrix-matrix product output must be at least 3D, got {len(out_memlet.subset)} dimensions")


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
