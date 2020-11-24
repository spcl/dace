# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, Dict, Optional
from dace.data import Array
from dace.symbolic import symstr
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import (to_blastype, get_gemm_opts)
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands,
                                              _get_batchmm_opts,
                                              _get_codegen_gemm_opts)
from .. import environments


@dace.library.expansion
class ExpandBatchedMatMulPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        # Get metadata from parent SDFG
        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b,
                                                       shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)
        outedge = parent_state.out_edges(node)[0]
        cdesc = parent_sdfg.arrays[outedge.data.data]
        bopt = _get_batchmm_opts(shape_a, strides_a, shape_b, strides_b,
                                 cdesc.shape, cdesc.strides)

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

        _, array_a = sdfg.add_array("_a",
                                    shape_a,
                                    dtype_a,
                                    strides=strides_a,
                                    storage=storage)
        _, array_b = sdfg.add_array("_b",
                                    shape_b,
                                    dtype_b,
                                    strides=strides_b,
                                    storage=storage)
        _, array_c = sdfg.add_array("_c",
                                    shape_c,
                                    dtype_c,
                                    strides=cdata[-1],
                                    storage=storage)

        # Add an initialization state
        init_state = sdfg.add_state()
        init_state.add_mapped_tasklet(
            'batched_matmul_init',
            {'_o%d' % i: '0:%s' % symstr(d)
             for i, d in enumerate(shape_c)}, {},
            'out = 0', {
                'out':
                dace.Memlet.simple(
                    '_c', ','.join(['_o%d' % i for i in range(len(shape_c))]))
            },
            external_edges=True)

        state = sdfg.add_state_after(init_state, node.label + "_state")

        state.add_mapped_tasklet(
            '_BatchedBatchedMatMult_', {
                '__i%d' % i: '0:%s' % s
                for i, s in enumerate([
                    bopt['b'], array_a.shape[-2], array_b.shape[-1],
                    array_a.shape[-1]
                ])
            }, {
                '__a':
                dace.Memlet.simple("_a", ('__i1, __i3' if len(
                    array_a.shape) == 2 else '__i0, __i1, __i3')),
                '__b':
                dace.Memlet.simple("_b", ('__i3, __i2' if len(
                    array_b.shape) == 2 else '__i0, __i3, __i2'))
            },
            '__c = __a * __b', {
                '__c':
                dace.Memlet.simple(
                    "_c", '__i0, __i1, __i2', wcr_str='lambda x, y: x + y')
            },
            external_edges=True)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandBatchedMatMulPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandBatchedMatMulMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        dtype = node.dtype
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
            raise ValueError("Unsupported type for BLAS dot product: " +
                             str(dtype))
        (_, adesc, ashape,
         astrides), (_, bdesc, bshape,
                     bstrides), _ = _get_matmul_operands(node, state, sdfg)
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc,
                                     alpha, beta, cdesc.dtype.ctype, func)

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
        dtype = node.dtype
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

        alpha = "dace::blas::CublasConstants::Get(__dace_cuda_device).%sPone()" % factort
        beta = "dace::blas::CublasConstants::Get(__dace_cuda_device).%sZero()" % factort

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

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc,
                                     alpha, beta, cdtype, func)

        # Matrix multiplication
        call = '''cublas{func}StridedBatched(__dace_cublas_handle,
            CUBLAS_OP_{ta}, CUBLAS_OP_{tb},
            {M}, {N}, {K},
            {alpha},
            ({dtype}*){x}, {lda}, {stride_a},
            ({dtype}*){y}, {ldb}, {stride_b},
            {beta},
            ({dtype}*)_c, {ldc}, {stride_c},
            {BATCH});'''

        code = (environments.cublas.cuBLAS.handle_setup_code(node) +
                call.format_map(opt))
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        # If buffers are not on the GPU, copy them
        # TODO: This creates variable shadowing
        if any(desc.storage not in
               [dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned]
               for desc in [adesc, bdesc, cdesc]):
            nsdfg = dace.SDFG('nested_batched_matmul')
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
            nstate.add_node(tasklet)
            nstate.add_nedge(a, ga, dace.Memlet.from_array('_a', adesc))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))
            nstate.add_edge(ga, None, tasklet, '_a',
                            dace.Memlet.from_array('_a_gpu', adesc))
            nstate.add_edge(gb, None, tasklet, '_b',
                            dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '_c', gc, None,
                            dace.Memlet.from_array('_c_gpu', cdesc))
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
        "cuBLAS": ExpandBatchedMatMulCuBLAS
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name,
                         location=location,
                         inputs={'_a', '_b'},
                         outputs={'_c'})
        self.dtype = dtype

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError(
                "Expected exactly two inputs to batched matrix-matrix product")
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
            raise ValueError(
                "Batched matrix-matrix product only supported on matrices")
        if len(size1) != 3:
            raise ValueError(
                "Batched matrix-matrix product only supported on matrices")
        if size0[-1] != size1[-2]:
            raise ValueError("Inputs to matrix-matrix product "
                             "must agree in the k-dimension")
        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size2 = out_subset.size()
        if len(size2) != 3:
            raise ValueError(
                "batched matrix-matrix product only supported on matrices")
