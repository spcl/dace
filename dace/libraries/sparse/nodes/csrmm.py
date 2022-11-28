# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from dace import dtypes, memlet as mm, properties, data as dt, propagate_memlets_sdfg
from dace.symbolic import symstr
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import (to_blastype, get_gemm_opts, check_access, dtype_to_cudadatatype,
                                              to_cublas_computetype)
from dace.libraries.sparse import environments
import numpy as np


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
        return "dace.{}({})".format(dace.DTYPE_TO_TYPECLASS[dtype].to_string(), value)


def _get_csrmm_operands(node,
                        state,
                        sdfg,
                        name_lhs_rows="_a_rows",
                        name_lhs_cols="_a_cols",
                        name_lhs_vals="_a_vals",
                        name_rhs="_b",
                        name_out="_c"):
    """Returns the CSRMM input edges, arrays, and shape."""

    result = {}
    result[name_lhs_rows] = None
    result[name_lhs_cols] = None
    result[name_lhs_vals] = None
    result[name_rhs] = None
    result[name_out] = None

    for edge in state.all_edges(node):
        if edge.dst_conn in result.keys():
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed]
            res = edge, outer_array, size, strides
            result[edge.dst_conn] = res
        elif edge.src_conn == name_out:
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_output_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed]
            result[edge.src_conn] = (edge, outer_array, size, strides)
    for name, res in result.items():
        if res is None:
            raise ValueError("Matrix multiplication connector "
                             "\"{}\" not found.".format(name))
    return result


@dace.library.expansion
class ExpandCSRMMPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        if node.opA != 0:
            raise NotImplementedError

        nsdfg = SDFG(node.label + "_nsdfg")

        operands = _get_csrmm_operands(node, state, sdfg)
        nstate = nsdfg.add_state("state", is_start_state=True)
        for name, desc in operands.items():
            desc = desc[1]

            if isinstance(desc, dt.View):
                ndesc = desc.as_array()
            else:
                ndesc = dc(desc)
            ndesc.lifetime = dtypes.AllocationLifetime.Scope
            ndesc.transient = False
            nsdfg.add_datadesc(name, ndesc)

        array_a_vals = nsdfg.arrays['_a_vals']
        array_a_rows = nsdfg.arrays['_a_rows']
        array_a_cols = nsdfg.arrays['_a_cols']
        array_b = nsdfg.arrays['_b']
        array_c = nsdfg.arrays['_c']

        a_val_node = nstate.add_access('_a_vals')
        a_row_node = nstate.add_access('_a_rows')
        a_col_node = nstate.add_access('_a_cols')
        b_node = nstate.add_access('_b')
        c_node = nstate.add_access('_c')

        if node.beta == 0.0:
            shape_c = operands['_c'][1].shape

            init_state = nsdfg.add_state_before(nstate, node.label + "_initstate")
            init_state.add_mapped_tasklet(
                'csrmm_init', {'_o%d' % i: '0:%s' % symstr(d)
                               for i, d in enumerate(shape_c)}, {},
                'out = 0', {'out': dace.Memlet.simple('_c', ','.join(['_o%d' % i for i in range(len(shape_c))]))},
                external_edges=True)
        elif node.beta == 1.0:
            # Simplify computation
            edges = state.edges_by_connector(node, "_cin")
            for edge in edges:
                state.remove_edge(edge)

                if state.in_degree(edge.src) == 0 and state.out_degree(edge.src) == 0:
                    state.remove_node(edge.src)

            node.remove_in_connector("_cin")
        else:
            init_state = nsdfg.add_state_before(nstate, node.label + "_initstate")

            cdesc = operands['_c'][1]
            cin_desc = dc(cdesc)
            nsdfg.add_datadesc('_cin', cin_desc)

            init_state.add_mapped_tasklet(
                'csrmm_init', {'_o%d' % i: '0:%s' % symstr(d)
                               for i, d in enumerate(cdesc.shape)},
                {'_in': dace.Memlet.simple('_cin', ','.join(['_o%d' % i for i in range(len(cdesc.shape))]))},
                f'_out = {node.beta} * _in',
                {'_out': dace.Memlet.simple('_c', ','.join(['_o%d' % i for i in range(len(cdesc.shape))]))},
                external_edges=True)

        # Multiplication map
        outer_map_entry, outer_map_exit = nstate.add_map("spmm_1", dict(i='0:' + str(array_a_rows.shape[0] - 1)))
        outer_map_entry.add_in_connector("IN__a_vals")
        outer_map_entry.add_in_connector("IN__a_cols")
        outer_map_entry.add_in_connector("IN__a_rows")
        outer_map_entry.add_in_connector("IN__b")
        outer_map_exit.add_out_connector("OUT__c")

        nstate.add_edge(a_val_node, None, outer_map_entry, "IN__a_vals", mm.Memlet.from_array("_a_vals", array_a_vals))
        nstate.add_edge(a_col_node, None, outer_map_entry, "IN__a_cols", mm.Memlet.from_array("_a_cols", array_a_cols))
        nstate.add_edge(a_row_node, None, outer_map_entry, "IN__a_rows", mm.Memlet.from_array("_a_rows", array_a_rows))
        nstate.add_edge(b_node, None, outer_map_entry, "IN__b", mm.Memlet.from_array("_b", array_b))
        nstate.add_edge(outer_map_exit, "OUT__c", c_node, None, mm.Memlet.from_array("_c", array_c))

        outer_map_entry.add_out_connector("OUT__a_vals")
        outer_map_entry.add_out_connector("OUT__a_cols")
        outer_map_entry.add_out_connector("OUT__a_rows")
        outer_map_entry.add_out_connector("OUT__b")

        inner_map_entry, inner_map_exit = nstate.add_map("spmm_2", dict(j="__map_19_b0:__map_19_e1"))
        inner_map_entry.add_in_connector("__map_19_b0")
        inner_map_entry.add_in_connector("__map_19_e1")
        nstate.add_edge(outer_map_entry, "OUT__a_rows", inner_map_entry, "__map_19_b0",
                        mm.Memlet("_a_rows[i]", data="_a_rows"))
        nstate.add_edge(outer_map_entry, "OUT__a_rows", inner_map_entry, "__map_19_e1",
                        mm.Memlet("_a_rows[i + 1]", data="_a_rows"))

        inner_map_entry.add_in_connector("IN_tmp_a_vals")
        nstate.add_edge(outer_map_entry, "OUT__a_vals", inner_map_entry, "IN_tmp_a_vals",
                        mm.Memlet.from_array("_a_vals", array_a_vals))

        inner_map_entry.add_in_connector("IN_tmp_a_cols")
        nstate.add_edge(outer_map_entry, "OUT__a_cols", inner_map_entry, "IN_tmp_a_cols",
                        mm.Memlet.from_array("_a_cols", array_a_cols))

        inner_map_entry.add_in_connector("IN_tmp_b")
        nstate.add_edge(outer_map_entry, "OUT__b", inner_map_entry, "IN_tmp_b", mm.Memlet.from_array("_b", array_b))

        inner_map_exit.add_out_connector("OUT__c_1")
        outer_map_exit.add_in_connector("IN__c")
        nstate.add_edge(inner_map_exit, "OUT__c_1", outer_map_exit, "IN__c",
                        mm.Memlet(expr=f"_c[i, 0:{str(array_c.shape[1])}]"))

        inner_map_entry.add_out_connector("OUT_tmp_a_vals")
        inner_map_entry.add_out_connector("OUT_tmp_a_cols")
        inner_map_entry.add_out_connector("OUT_tmp_b")

        k_map_entry, k_map_exit = nstate.add_map("spmm_3", dict(k=f"0:{str(array_b.shape[1])}"))
        k_map_entry.add_in_connector("IN_tmp_a_vals_1")
        nstate.add_edge(inner_map_entry, "OUT_tmp_a_vals", k_map_entry, "IN_tmp_a_vals_1",
                        mm.Memlet.simple("_a_vals", "j"))

        k_map_entry.add_in_connector("IN_tmp_a_cols_1")
        nstate.add_edge(inner_map_entry, "OUT_tmp_a_cols", k_map_entry, "IN_tmp_a_cols_1",
                        mm.Memlet.simple("_a_cols", "j"))

        k_map_entry.add_in_connector("IN_tmp_b_1")
        nstate.add_edge(inner_map_entry, "OUT_tmp_b", k_map_entry, "IN_tmp_b_1", mm.Memlet.from_array("_b", array_b))

        k_map_exit.add_out_connector("OUT__c_1")
        inner_map_exit.add_in_connector("IN__c_1")
        nstate.add_edge(k_map_exit, "OUT__c_1", inner_map_exit, "IN__c_1",
                        mm.Memlet(expr=f"_c[i, 0:{str(array_c.shape[1])}]"))

        k_map_entry.add_out_connector("OUT_tmp_a_cols_1")
        k_map_entry.add_out_connector("OUT_tmp_a_vals_1")
        k_map_entry.add_out_connector("OUT_tmp_b_1")

        tasklet_ind = nstate.add_tasklet("Indirection",
                                         inputs={
                                             "__ind_b": None,
                                             "index_a_cols_0": None
                                         },
                                         outputs={'lookup': None},
                                         code="lookup = __ind_b[index_a_cols_0]")
        nsdfg.add_scalar("_b_value", dtype=array_b.dtype, transient=True)
        nstate.add_edge(k_map_entry, "OUT_tmp_a_cols_1", tasklet_ind, "index_a_cols_0",
                        mm.Memlet.simple("_a_cols", "j"))
        nstate.add_edge(k_map_entry, "OUT_tmp_b_1", tasklet_ind, "__ind_b",
                        mm.Memlet.simple("_b", f"0:{array_b.shape[0]}, k"))

        tasklet_mult = nstate.add_tasklet("spmm", {
            "__a": None,
            "__b": None
        }, {"__o": None},
                                          code=f"__o = {node.alpha} * (__a * __b)")
        nstate.add_edge(k_map_entry, "OUT_tmp_a_vals_1", tasklet_mult, "__a", mm.Memlet.simple("_a_vals", "j"))
        nstate.add_edge(tasklet_ind, "lookup", tasklet_mult, "__b", mm.Memlet.simple("_b_value", "0"))

        k_map_exit.add_in_connector("IN__c_1")
        nstate.add_edge(tasklet_mult, "__o", k_map_exit, "IN__c_1",
                        mm.Memlet.simple("_c", subset_str="i, k", wcr_str="lambda x, y: (x + y)"))

        nsdfg.validate()
        propagate_memlets_sdfg(nsdfg)

        return nsdfg


@dace.library.expansion
class ExpandCSRMMMKL(ExpandTransformation):
    environments = [environments.IntelMKLSparse]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        operands = _get_csrmm_operands(node, state, sdfg)
        arows = operands['_a_rows'][1]
        acols = operands['_a_cols'][1]
        avals = operands['_a_vals'][1]
        bdesc = operands['_b'][1]

        dtype = avals.dtype.base_type
        func = f"mkl_sparse_{to_blastype(dtype.type).lower()}"
        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'

        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        check_access(dtypes.ScheduleType.CPU_Multicore, arows, acols, avals, bdesc, cdesc)

        opt = {}

        opt['func'] = func

        if node.opA == 1:
            opt['opA'] = 'SPARSE_OPERATION_TRANSPOSE'
        elif node.opA == 2:
            opt['opA'] = 'SPARSE_OPERATION_CONJUGATE_TRANSPOSE'
        else:
            opt['opA'] = 'SPARSE_OPERATION_NON_TRANSPOSE'

        opt['layout'] = 'SPARSE_LAYOUT_ROW_MAJOR'

        code = ''
        if dtype in (dace.complex64, dace.complex128):
            code = f'''
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            opt['alpha'] = '&alpha'
            opt['beta'] = '&beta'
        else:
            opt['alpha'] = alpha
            opt['beta'] = beta

        opt['nrows'] = cdesc.shape[0]
        opt['ncols'] = cdesc.shape[1]
        opt['arows'] = cdesc.shape[0]
        opt['acols'] = bdesc.shape[0]
        if node.opA != 0:
            opt['arows'], opt['acols'] = opt['acols'], opt['arows']

        opt['ldb'] = opt['ncols']
        opt['ldc'] = opt['ncols']

        code += """
            sparse_matrix_t __csrA;
            {func}_create_csr(&__csrA, SPARSE_INDEX_BASE_ZERO, {arows}, {acols}, _a_rows, _a_rows + 1, _a_cols, _a_vals);
            struct matrix_descr __descrA;
            __descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
            __descrA.mode = SPARSE_FILL_MODE_UPPER;
            __descrA.diag = SPARSE_DIAG_NON_UNIT;

            {func}_mm({opA}, {alpha}, __csrA, __descrA, {layout}, _b, {ncols}, {ldb}, {beta}, _c, {ldc});
        """.format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandCSRMMCuSPARSE(ExpandTransformation):

    environments = [environments.cuSPARSE]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_csrmm_operands(node, state, sdfg)
        arows = operands['_a_rows'][1]
        acols = operands['_a_cols'][1]
        avals = operands['_a_vals'][1]
        bdesc = operands['_b'][1]
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        # If buffers are not on the GPU, copy them
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (arows, acols, avals, bdesc, cdesc))

        dtype = avals.dtype.base_type
        func = "cusparseSpMM"
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

        call_prefix = environments.cuSPARSE.handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        # TODO: Maybe fix this later
        # constants = {
        #     1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Pone()",
        #     #-1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Mone()",
        #     0.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Zero()",
        # }
        # if node.alpha not in constants or node.beta not in constants:
        if True:
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
            call_prefix += f'''cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            call_suffix += '''cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);'''
            alpha = f'({cdtype} *)&alpha'
            beta = f'({cdtype} *)&beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        # Set up options for code formatting
        # opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)

        opt = {}

        opt['arr_prefix'] = arr_prefix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'

        opt['func'] = func

        if node.opA == 1:
            opt['opA'] = 'CUSPARSE_OPERATION_TRANSPOSE'
        elif node.opA == 2:
            opt['opA'] = 'CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE'
        else:
            opt['opA'] = 'CUSPARSE_OPERATION_NON_TRANSPOSE'

        opt['opB'] = 'CUSPARSE_OPERATION_NON_TRANSPOSE'

        opt['layout'] = 'CUSPARSE_ORDER_ROW'

        opt['compute'] = f'CUDA_R_{to_cublas_computetype(dtype)}'
        opt['handle'] = '__dace_cusparse_handle'

        opt['alpha'] = alpha
        opt['beta'] = beta

        opt['nrows'] = cdesc.shape[0]
        opt['ncols'] = cdesc.shape[1]
        opt['arows'] = cdesc.shape[0]
        opt['acols'] = bdesc.shape[0]
        opt['bcols'] = bdesc.shape[1]
        opt['annz'] = avals.shape[0]
        if node.opA != 0:
            opt['arows'], opt['acols'] = opt['acols'], opt['arows']

        opt['ldb'] = opt['ncols']
        opt['ldc'] = opt['ncols']

        call = """
            cusparseSpMatDescr_t matA;
            cusparseDnMatDescr_t matB, matC;
            void*                dBuffer    = NULL;
            size_t               bufferSize = 0;
            // Create sparse matrix A in CSR format
            dace::sparse::CheckCusparseError( cusparseCreateCsr(&matA, {arows}, {acols}, {annz},
                                                {arr_prefix}_a_rows, {arr_prefix}_a_cols, {arr_prefix}_a_vals,
                                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                CUSPARSE_INDEX_BASE_ZERO, {compute}) );
            // Create dense matrix B
            dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matB, {acols}, {bcols}, {ldb}, {arr_prefix}_b,
                                                {compute}, {layout}) );
            // Create dense matrix C
            dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matC, {arows}, {bcols}, {ldc}, {arr_prefix}_c,
                                                {compute}, {layout}) );
            // allocate an external buffer if needed
            dace::sparse::CheckCusparseError( cusparseSpMM_bufferSize(
                                            {handle},
                                            {opA},
                                            {opB},
                                            {alpha}, matA, matB, {beta}, matC, {compute},
                                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
            cudaMalloc(&dBuffer, bufferSize);

            // execute SpMM
            dace::sparse::CheckCusparseError( cusparseSpMM({handle},
                                            {opA},
                                            {opB},
                                            {alpha}, matA, matB, {beta}, matC, {compute},
                                            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );

            // destroy matrix/vector descriptors
            dace::sparse::CheckCusparseError( cusparseDestroySpMat(matA) );
            dace::sparse::CheckCusparseError( cusparseDestroyDnMat(matB) );
            dace::sparse::CheckCusparseError( cusparseDestroyDnMat(matC) );
            cudaFree(dBuffer);
        """.format_map(opt)

        # # Matrix multiplication
        # if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
        #     call = '''cublas{func}(__dace_cublas_handle,
        #         CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
        #         {M}, {N}, {K},
        #         {alpha},
        #         ({dtype}*){arr_prefix}{x}, {lda},
        #         ({dtype}*){arr_prefix}{y}, {ldb},
        #         {beta},
        #         ({dtype}*){arr_prefix}_c, {ldc});'''.format_map(opt)
        # else:
        #     if node.compute_type is not None:
        #         acctype = node.compute_type
        #     elif node.accumulator_type is not None:
        #         acc_dtype: dtypes.typeclass = node.accumulator_type
        #         acctype = f'CUBLAS_COMPUTE_{to_cublas_computetype(acc_dtype)}'
        #     else:
        #         acctype = f'CUBLAS_COMPUTE_{to_cublas_computetype(dtype)}'

        #     algorithm = 'CUBLAS_GEMM_DEFAULT_TENSOR_OP'
        #     if node.algorithm is not None:
        #         algorithm = node.algorithm

        #     call = f'''
        #     cublasGemmEx(__dace_cublas_handle,
        #         CUBLAS_OP_{opt['ta']}, CUBLAS_OP_{opt['tb']},
        #         {opt['M']}, {opt['N']}, {opt['K']},
        #         {alpha},
        #         {arr_prefix}{opt['x']},
        #         {dtype_to_cudadatatype(opt['xdtype'])},
        #         {opt['lda']},
        #         {arr_prefix}{opt['y']},
        #         {dtype_to_cudadatatype(opt['ydtype'])},
        #         {opt['ldb']},
        #         {beta},
        #         {arr_prefix}_c,
        #         {dtype_to_cudadatatype(opt['cdtype'])},
        #         {opt['ldc']},
        #         {acctype},
        #         {algorithm});
        #     '''

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
            for name, desc in [('_a_rows', arows), ('_a_cols', acols), ('_a_vals', avals), ('_b', bdesc),
                               ('_c', cdesc)]:
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
            ar = nstate.add_read('_a_rows')
            gar = nstate.add_access('_a_rows_gpu')
            ac = nstate.add_read('_a_cols')
            gac = nstate.add_access('_a_cols_gpu')
            av = nstate.add_read('_a_vals')
            gav = nstate.add_access('_a_vals_gpu')
            b = nstate.add_read('_b')
            gb = nstate.add_access('_b_gpu')
            c = nstate.add_write('_c')
            gc = nstate.add_access('_c_gpu')

            # Reset code and connectors
            tasklet.in_connectors = {"_conn" + k: None for k in tasklet.in_connectors}
            tasklet.out_connectors = {"_conn" + k: None for k in tasklet.out_connectors}

            nstate.add_node(tasklet)
            nstate.add_nedge(ar, gar, dace.Memlet.from_array('_a_rows', arows))
            nstate.add_nedge(ac, gac, dace.Memlet.from_array('_a_cols', acols))
            nstate.add_nedge(av, gav, dace.Memlet.from_array('_a_vals', avals))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(gar, None, tasklet, '_conn_a_rows', dace.Memlet.from_array('_a_rows_gpu', arows))
            nstate.add_edge(gac, None, tasklet, '_conn_a_cols', dace.Memlet.from_array('_a_cols_gpu', arows))
            nstate.add_edge(gav, None, tasklet, '_conn_a_vals', dace.Memlet.from_array('_a_vals_gpu', arows))
            nstate.add_edge(gb, None, tasklet, '_conn_b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '_conn_c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            if node.beta != 0.0:
                rc = nstate.add_read('_c')
                rgc = nstate.add_access('_c_gpu')
                tasklet.add_in_connector('_conn_cin')
                nstate.add_nedge(rc, rgc, dace.Memlet('_c'))
                nstate.add_edge(rgc, None, tasklet, '_conn_cin', dace.Memlet('_c_gpu'))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.node
class CSRMM(dace.sdfg.nodes.LibraryNode):
    """
    Executes alpha * (A @ B) + beta * C. C should be unidirectionally broadcastable (ONNX terminology) to A @ B.
    A is a sparse matrix in CSR format, while B is dense.
    """

    # Global properties
    implementations = {"pure": ExpandCSRMMPure, "MKL": ExpandCSRMMMKL, "cuSPARSE": ExpandCSRMMCuSPARSE}
    default_implementation = None

    # Object fields
    opA = properties.Property(dtype=int, default=0, desc="0: non-transpose, 1: trasponse, 2: conjugate transpose")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")
    cin = properties.Property(dtype=bool, default=True, desc="Whether to have a _cin connector when beta != 0")
    algorithm = properties.Property(dtype=str,
                                    allow_none=True,
                                    default=None,
                                    desc="If applicable, chooses the vendor-provided implementation "
                                    "(algorithm) for the multiplication")
    compute_type = properties.Property(default=None,
                                       dtype=str,
                                       allow_none=True,
                                       desc="If applicable, overrides computation type (CUBLAS-specific, see "
                                       "``cublasComputeType_t``)")

    def __init__(self, name, location=None, opA=0, alpha=1, beta=0, cin=True):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_rows", "_a_cols", "_a_vals", "_b", "_cin"}
                                 if beta != 0 and cin else {"_a_rows", "_a_cols", "_a_vals", "_b"}),
                         outputs={"_c"})
        self.opA = opA
        self.alpha = alpha
        self.beta = beta
        self.cin = cin

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [4, 5]:
            raise ValueError("Expected 4 or 5 inputs to CSRMM")
        size4 = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a_rows':
                subset = dc(memlet.subset)
                subset.squeeze()
                size0 = subset.size()
            if dst_conn == '_a_cols':
                subset = dc(memlet.subset)
                subset.squeeze()
                size1 = subset.size()
            if dst_conn == '_a_vals':
                subset = dc(memlet.subset)
                subset.squeeze()
                size2 = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                size3 = subset.size()
            if dst_conn == '_cin':
                subset = dc(memlet.subset)
                subset.squeeze()
                size4 = subset.size()

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if len(size3) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")

        A_rows = size0[0] - 1
        A_cols = size3[0]
        if self.opA != 0:
            A_rows, A_cols = A_cols, A_rows
        B_cols = size3[1]

        # if size0[1] != size1[0]:
        #     raise ValueError("Inputs to matrix-matrix product " "must agree in the k-dimension")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size5 = out_subset.size()
        if size4 is not None and size4 != size5:
            raise ValueError("Input C matrix must match output matrix.")
        if len(size5) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if len(size5) == 2 and list(size5) != [A_rows, B_cols]:
            raise ValueError("Output to matrix-matrix product must agree in the m and n "
                             "dimensions")
