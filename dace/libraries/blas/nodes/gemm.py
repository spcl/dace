# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from dace import dtypes, memlet as mm, properties, data as dt
from dace.symbolic import symstr, equal, equal_valued
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, check_access, dtype_to_cudadatatype, to_cublas_computetype
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from .. import environments
import numpy as np
import warnings


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
            type=dace.dtype_to_typeclass(dtype).to_string(),
            real=cast_value.real,
            imag=cast_value.imag,
        )
    else:
        return "dace.{}({})".format(dace.dtype_to_typeclass(dtype).to_string(), value)


@dace.library.expansion
class ExpandGemmPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a, _, _), (edge_b, outer_array_b, shape_b, strides_b, _, _),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.dtype_to_typeclass(np.result_type(dtype_a, dtype_b).type)

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        if len(trans_shape_a) != 2 or len(trans_shape_b) != 2:
            raise SyntaxError("Matrix sizes must match")
        res = equal(trans_shape_a[1], trans_shape_b[0])
        if res is None:
            warnings.warn(
                f"First matrix columns {trans_shape_a[1]} may not match "
                f"second matrix rows {trans_shape_b[0]}", UserWarning)
        elif not res:
            raise SyntaxError("Matrix sizes must match")
        M, K, N = trans_shape_a[0], trans_shape_a[1], trans_shape_b[1]
        shape_c = (M, N)

        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-3], storage=cdata[1].storage)

        if equal_valued(1, node.alpha):
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))

        if equal_valued(1, node.beta):
            state = sdfg.add_state(node.label + "_state")
        else:
            init_state = sdfg.add_state(node.label + "_initstate")
            state = sdfg.add_state_after(init_state, node.label + "_state")

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        # Initialization / beta map
        if equal_valued(0, node.beta):
            init_state.add_mapped_tasklet(
                'gemm_init', {
                    '_o%d' % i: '0:%s' % symstr(d)
                    for i, d in enumerate(shape_c)
                }, {},
                'out = 0', {'out': dace.Memlet.simple(mul_out, ','.join(['_o%d' % i for i in range(len(shape_c))]))},
                external_edges=True)
        elif equal_valued(1, node.beta):
            # Do nothing for initialization, only update the values
            pass
        else:
            # Beta map
            add_program = "__y = ({} * __c)".format(_cast_to_dtype_str(node.beta, dtype_a))

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
                raise ValueError("Could not broadcast input _c to ({}, {})".format(M, N))

            init_state.add_mapped_tasklet("gemm_init", {
                "__i%d" % i: "0:%s" % s
                for i, s in enumerate([M, N])
            }, {
                "__c": dace.Memlet.simple("_c", memlet_idx),
            },
                                          add_program, {"__y": dace.Memlet.simple("_c", "__i0, __i1")},
                                          external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("gemm", {
            "__i%d" % i: "0:%s" % s
            for i, s in enumerate([M, N, K])
        }, {
            "__a": dace.Memlet.simple("_a", "__i2, __i0" if node.transA else "__i0, __i2"),
            "__b": dace.Memlet.simple("_b", "__i1, __i2" if node.transB else "__i2, __i1")
        },
                                 mul_program,
                                 {"__out": dace.Memlet.simple(mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")},
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
        (_, adesc, _, _, _, _), (_, bdesc, _, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
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

        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, dtype.ctype, func)
        opt['cast'] = "(float *)" if dtype == dace.float32sr else ""

        # Adaptations for BLAS API
        opt['ta'] = 'CblasNoTrans' if opt['ta'] == 'N' else 'CblasTrans'
        opt['tb'] = 'CblasNoTrans' if opt['tb'] == 'N' else 'CblasTrans'

        code = ''
        if dtype in (dace.complex64, dace.complex128):
            code = f'''
            {dtype.ctype} __alpha = {alpha};
            {dtype.ctype} __beta = {beta};
            '''
            opt['alpha'] = '&__alpha'
            opt['beta'] = '&__beta'

        code += ("cblas_{func}(CblasColMajor, {ta}, {tb}, "
                 "{M}, {N}, {K}, {alpha},{cast} {x}, {lda}, {cast} {y}, {ldb}, {beta}, "
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
class ExpandGemmGPUBLAS(ExpandTransformation):

    environments = []

    @classmethod
    def expansion(cls, node, state, sdfg):
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

        # If buffers are not on the GPU, copy them
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))

        dtype = adesc.dtype.base_type
        func = cls.funcname(to_blastype(dtype.type))
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
            cdtype = f'{cls.dtype_backend}Complex'
            factort = 'Complex64'
        elif dtype == dace.complex128:
            cdtype = f'{cls.dtype_backend}DoubleComplex'
            factort = 'Complex128'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        call_prefix = cls.environments[0].handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        constants = {
            1.0: f"__state->{cls.backend}blas_handle.Constants(__dace_cuda_device).{factort}Pone()",
            #-1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Mone()",
            0.0: f"__state->{cls.backend}blas_handle.Constants(__dace_cuda_device).{factort}Zero()",
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
            call_prefix += f'''{cls.set_pointer_mode}(__dace_{cls.backend}blas_handle, {cls.pointer_host});
            {dtype.ctype} __alpha = {alpha};
            {dtype.ctype} __beta = {beta};
            '''
            call_suffix += f'''{cls.set_pointer_mode}(__dace_{cls.backend}blas_handle, {cls.pointer_device});'''
            alpha = f'({cdtype} *)&__alpha'
            beta = f'({cdtype} *)&__beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['arr_prefix'] = arr_prefix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'

        # Matrix multiplication
        if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
            opt['backend'] = cls.backend
            opt['backend_op_ta'] = cls.backend_op(opt['ta'])
            opt['backend_op_tb'] = cls.backend_op(opt['tb'])

            call = '''{backend}blas{func}(__dace_{backend}blas_handle,
                {backend_op_ta}, {backend_op_tb},
                {M}, {N}, {K},
                {alpha},
                ({dtype}*){arr_prefix}{x}, {lda},
                ({dtype}*){arr_prefix}{y}, {ldb},
                {beta},
                ({dtype}*){arr_prefix}_c, {ldc});'''.format_map(opt)
        else:
            if node.compute_type is not None:
                acctype = node.compute_type
            elif node.accumulator_type is not None:
                acc_dtype: dtypes.typeclass = node.accumulator_type
                acctype = f'{cls.backend.upper()}BLAS_COMPUTE_{to_cublas_computetype(acc_dtype)}'
            else:
                acctype = f'{cls.backend.upper()}BLAS_COMPUTE_{to_cublas_computetype(dtype)}'

            algorithm = f'{cls.backend.upper()}BLAS_GEMM_DEFAULT_TENSOR_OP'
            if node.algorithm is not None:
                algorithm = node.algorithm

            call = f'''
            {cls.backend}blas{cls.ex_suffix}(__dace_{cls.backend}blas_handle,
                {cls.backend_op(opt['ta'])},
                {cls.backend_op(opt['tb'])},
                {opt['M']}, {opt['N']}, {opt['K']},
                {alpha},
                {arr_prefix}{opt['x']},
                {dtype_to_cudadatatype(opt['xdtype'])},
                {opt['lda']},
                {arr_prefix}{opt['y']},
                {dtype_to_cudadatatype(opt['ydtype'])},
                {opt['ldb']},
                {beta},
                {arr_prefix}_c,
                {dtype_to_cudadatatype(opt['cdtype'])},
                {opt['ldc']},
                {acctype},
                {algorithm});
            '''

        code = (call_prefix + call + call_suffix)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        # If buffers are not on the GPU, copy them
        if needs_copy:
            nsdfg = dace.SDFG('nested_gemm')
            for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                dcopy.transient = False
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

            # Reset code and connectors
            tasklet.in_connectors = {"_conn" + k: None for k in tasklet.in_connectors}
            tasklet.out_connectors = {"_conn" + k: None for k in tasklet.out_connectors}

            # Remove _conn_c from in connectors if it exists
            if "_conn_c" in tasklet.in_connectors:
                tasklet.remove_in_connector("_conn_c")

            nstate.add_node(tasklet)
            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(ga, None, tasklet, '_conn_a', dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, tasklet, '_conn_b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '_conn_c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            if not equal_valued(0, node.beta):
                rc = nstate.add_read('_c')
                rgc = nstate.add_access('_c_gpu')
                tasklet.add_in_connector('_conn_cin')
                nstate.add_nedge(rc, rgc, dace.Memlet('_c'))
                nstate.add_edge(rgc, None, tasklet, '_conn_cin', dace.Memlet('_c_gpu'))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.expansion
class ExpandGemmCuBLAS(ExpandGemmGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    backend = 'cu'
    dtype_backend = 'cu'
    set_pointer_mode = 'cublasSetPointerMode'
    pointer_host = 'CUBLAS_POINTER_MODE_HOST'
    pointer_device = 'CUBLAS_POINTER_MODE_DEVICE'
    ex_suffix = 'GemmEx'

    @classmethod
    def backend_op(cls, mode: str) -> str:
        return f'CUBLAS_OP_{mode}'

    @classmethod
    def funcname(cls, dtype: str) -> str:
        return f'{dtype}gemm'


@dace.library.expansion
class ExpandGemmRocBLAS(ExpandGemmGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    backend = 'roc'
    dtype_backend = 'hip'
    set_pointer_mode = 'rocblas_set_pointer_mode'
    pointer_host = 'rocblas_pointer_mode_host'
    pointer_device = 'rocblas_pointer_mode_device'
    ex_suffix = '_gemm_ex'

    @classmethod
    def backend_op(cls, mode: str) -> str:
        if mode == 'N':
            return 'rocblas_operation_none'
        elif mode == 'T':
            return 'rocblas_operation_transpose'
        raise ValueError(f'Invalid gemm matrix operation {mode}')

    @classmethod
    def funcname(cls, dtype: str) -> str:
        return f'_{dtype.lower()}gemm'


@dace.library.expansion
class ExpandGemmPBLAS(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, _, _, _), (_, bdesc, bshape, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
        dtype = adesc.dtype.base_type

        if not equal_valued(0, node.beta):
            raise NotImplementedError

        M = ashape[0]
        K = ashape[1]
        N = bshape[1]
        Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
        Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
        try:
            sdfg.add_symbol('Px', dace.int32)
            sdfg.add_symbol('Py', dace.int32)
        except FileExistsError:
            pass

        @dace.program
        def _gemm_pblas(_a: dtype[M, K], _b: dtype[K, N], _c: dtype[M, N]):
            lA = np.empty((M // Px, K // Py), dtype=_a.dtype)
            lB = np.empty((K // Px, N // Py), dtype=_b.dtype)
            dace.comm.BCScatter(_a, lA, (M // Px, K // Py))
            dace.comm.BCScatter(_b, lB, (K // Px, N // Py))
            lC = distr.MatMult(lA, lB, (M, N, K))
            dace.comm.BCGather(lC, _c, (M // Px, N // Py))

        return _gemm_pblas.to_sdfg()


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
        "cuBLAS": ExpandGemmCuBLAS,
        "rocBLAS": ExpandGemmRocBLAS,
        "PBLAS": ExpandGemmPBLAS,
    }
    default_implementation = None

    # Object fields
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")
    cin = properties.Property(dtype=bool, default=True, desc="Whether to have a _c in connector when beta != 0")
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

    def __init__(self, name, location=None, transA=False, transB=False, alpha=1, beta=0, cin=True):
        super().__init__(name,
                         location=location,
                         inputs=({"_a", "_b", "_c"} if not equal_valued(0, beta) and cin else {"_a", "_b"}),
                         outputs={"_c"})
        self.transA = True if transA else False
        self.transB = True if transB else False
        self.alpha = alpha
        self.beta = beta
        self.cin = cin

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to gemm")
        size2 = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                size0 = memlet.subset.size()
            if dst_conn == '_b':
                size1 = memlet.subset.size()
            if dst_conn == '_c':
                size2 = memlet.subset.size()

        if self.transA:
            size0 = list(reversed(size0))
        if self.transB:
            size1 = list(reversed(size1))

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if len(size0) != 2 or len(size1) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        res = equal(size0[1], size1[0])
        if res is None:
            warnings.warn(f'First matrix columns {size0[1]} and second matrix rows {size1[0]} may not match',
                          UserWarning)
        elif not res:
            raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")
        size3 = out_memlet.subset.size()
        if size2 is not None:
            res = [equal(s0, s1) for s0, s1 in zip(size2, size3)]
            fail = any([r is False for r in res])
            success = all([r is True for r in res])
            if fail:
                raise ValueError("Input C matrix must match output matrix.")
            elif not success:
                warnings.warn(f"Size of input C matrix {size2} may not match output matrix size {size3}", UserWarning)
        if len(size3) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if len(size3) == 2:
            res = [equal(s0, s1) for s0, s1 in zip(size3, [size0[-2], size1[-1]])]
            fail = any([r is False for r in res])
            success = all([r is True for r in res])
            if fail:
                raise ValueError("Output to matrix-matrix product must agree in the m and n dimensions")
            elif not success:
                warnings.warn(f'Size of output {size3} may not match input {size0} @ {size1}', UserWarning)


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemm')
@oprepo.replaces('dace.libraries.blas.Gemm')
def gemm_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
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

    libnode = Gemm('gemm', transA=trans_a, transB=trans_b, alpha=alpha, beta=beta)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_a', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_b', mm.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, mm.Memlet(C))

    if not equal_valued(0, beta):
        C_in = state.add_read(C)
        state.add_edge(C_in, None, libnode, '_c', mm.Memlet(C))

    return []
