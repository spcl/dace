from dace import data, dtypes, library, properties, Memlet
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers
from dace.libraries.sparse import environments
from dace.sdfg import nodes, SDFGState, SDFG


@library.node
class SPMMD(nodes.LibraryNode):
    """
    Sparse Matrix-Matrix Multiplication with Dense output.

    Executes ``C = alpha * (opA(A) @ opB(B)) + beta * C``. ``A`` and ``B`` are sparse matrices and ``C`` is Dense.
    ``opX()`` is one of the following:
     - Identity (e.g., opA(A) = A)
     - Transpose (e.g., opA(A) = A^T)
    alpha and beta are scalars.
    """

    implementations = {}
    default_implementation = None

    transA = properties.Property(
        dtype=bool, desc='Whether to transpose ``A`` before multiplying', default=False
    )
    transB = properties.Property(
        dtype=bool, desc='Whether to transpose ``B`` before multiplying', default=False
    )
    alpha = properties.Property(
        dtype=float,
        allow_none=False,
        default=1.0,
        desc='Scalar multiplied with ``(A @ B)`` before adding ``C``',
    )
    beta = properties.Property(
        dtype=float,
        allow_none=False,
        default=0.0,
        desc='Scalar multiplied with ``C`` before adding',
    )

    def __init__(
        self, name, location=None, transA=False, transB=False, alpha=1.0, beta=0.0
    ):
        """
        :param name: Name of the library node.
        :param location: ??
        :param transA: Whether to transpose ``A`` before multiplying
        :param transB: Whether to transpose ``B`` before multiplying
        :param alpha: Scalar multiplied with ``(A @ B)`` before adding ``C``
        :param beta: Scalar multiplied with ``C`` before adding
        """
        super().__init__(
            name,
            location=location,
            inputs={'_A', '_B'} if beta == 0.0 else {'_A', '_B', '_C'},
            outputs={'_C'},
        )
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta
    
    def validate(self, sdfg: SDFG, state: SDFGState):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError('Expected 2 or 3 input edges')
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError('Expected 1 output edge')

        for e in in_edges:
            desc = sdfg.arrays[e.data.data]

            if e.dst_conn in ['_A', '_B']:
                if not isinstance(desc, data.Tensor):
                    raise ValueError('Expected A and B inputs to be a tensor')
                if not desc.is_CSR() and not desc.is_CSC():
                    raise ValueError('Expected A and B inputs to be in CSR or CSC format')
            elif e.dst_conn == '_C':
                if not isinstance(desc, data.Array):
                    raise ValueError('Expected C input to be an array')
                if not len(desc.shape) != 2:
                    raise ValueError('Expected C input to be a 2D array')
            else:
                raise ValueError('Unexpected input edge')
            
        for e in out_edges:
            if e.src_conn != '_C':
                raise ValueError('Unexpected output edge')
            
            desc = sdfg.arrays[e.data.data]
            if not isinstance(desc, data.Array):
                raise ValueError('Expected C output to be an array')
            
        # dimension check


        # TODO dtype check


@library.register_expansion(SPMMD, 'MKL')
class MKLSPMMDExpansion(library.ExpandTransformation):
    environments = [environments.IntelMKLSparse]

    @staticmethod
    def expansion(
        node: SPMMD, parent_state: SDFGState, parent_sdfg: SDFG
    ) -> nodes.Tasklet:

        descs = {e.dst_conn: parent_sdfg.arrays[e.data.data] for e in parent_state.in_edges(node)}
        descs['_C'] = parent_sdfg.arrays[next(iter(parent_state.out_edges(node))).data.data]

        A = descs['_A']
        B = descs['_B']
        C = descs['_C']

        dtype = A.value_dtype
        blas_type = blas_helpers.to_blastype(dtype.type)
        func_pref = f'mkl_sparse_{blas_type.lower()}'

        code = ''

        a_rows = A.tensor_shape[0]
        a_cols = A.tensor_shape[1]
        a_op = 'SPARSE_OPERATION_TRANSPOSE' if node.transA else 'SPARSE_OPERATION_NON_TRANSPOSE'
        a_format = 'csr' if A.is_CSR() else 'csc'

        code += f"""
            sparse_matrix_t __A;
            {func_pref}_create_{a_format}(&__A, SPARSE_INDEX_BASE_ZERO, {a_rows}, {a_cols}, _A->idx1_pos, _A->idx1_pos + 1, _A->idx1_crd, _A->values);
            struct matrix_descr __A_descr;
            __A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            __A_descr.mode = SPARSE_FILL_MODE_LOWER;
            __A_descr.diag = SPARSE_DIAG_NON_UNIT;
            sparse_operation_t __A_op = {a_op};
        """

        b_rows = B.tensor_shape[0]
        b_cols = B.tensor_shape[1]
        b_op = 'SPARSE_OPERATION_TRANSPOSE' if node.transB else 'SPARSE_OPERATION_NON_TRANSPOSE'
        b_format = 'csr' if B.is_CSR() else 'csc'

        code += f"""
            sparse_matrix_t __B;
            {func_pref}_create_{b_format}(&__B, SPARSE_INDEX_BASE_ZERO, {b_rows}, {b_cols}, _B->idx1_pos, _B->idx1_pos + 1, _B->idx1_crd, _B->values);
            struct matrix_descr __B_descr;
            __B_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            __B_descr.mode = SPARSE_FILL_MODE_LOWER;
            __B_descr.diag = SPARSE_DIAG_NON_UNIT;
            sparse_operation_t __B_op = {b_op};
        """

        # TODO support column major
        layout = 'SPARSE_LAYOUT_ROW_MAJOR'

        code += f"""
            sparse_status_t __status = {func_pref}_sp2md(__A_op, __A_descr, __A, __B_op, __B_descr, __B, {node.alpha}, {node.beta}, _C, {layout}, {C.shape[1]});
            if (__status != SPARSE_STATUS_SUCCESS) {{
                exit((int) __status);
            }}
        """

        # TODO do we need to destroy the matrices?

        tasklet = nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dtypes.Language.CPP,
        )

        return tasklet


@library.register_expansion(SPMMD, 'taco')
class TacoSPMMDExpansion(library.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(
        node: SPMMD, parent_state: SDFGState, parent_sdfg: SDFG
    ) -> nodes.LibraryNode:
        from dace.libraries.sparse.nodes import TensorIndexNotation

        if node.beta != 0.0:
            raise NotImplementedError('TACO SPMMD does not support beta != 0.0')
        
        a_access = 'A(i, k)' if not node.transA else 'A(k, i)'
        b_access = 'B(k, j)' if not node.transB else 'B(j, k)'
        
        tin_node = TensorIndexNotation(
            'taco_spmmd',
            f'C(i, j) = {node.alpha} * {a_access} * {b_access}',
            [],  # TODO improve loop ordering
        )

        tin_node.add_in_connector('tin_A')
        tin_node.add_in_connector('tin_B')
        tin_node.add_out_connector('tin_C')

        # rewrite memlet destinations because TIN node expects them to be of the form `tin_{tensor_name}`
        for e in parent_state.in_edges(node):
            e.dst_conn = f'tin{e.dst_conn}'
        for e in parent_state.out_edges(node):
            e.src_conn = f'tin{e.src_conn}'
        
        return tin_node


@oprepo.replaces('dace.libraries.sparse.SPMMD')
def spdmm_frontend(
    pv: 'ProgramVisitor',
    sdfg: SDFG,
    state: SDFGState,
    name,
    transA: bool = False,
    transB: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    **kwargs,
):
    name = name.value

    data_nodes = {name: state.add_access(data) for name, data in kwargs.items()}

    spmmd_node = SPMMD(name, transA=transA, transB=transB, alpha=alpha, beta=beta)
    state.add_node(spmmd_node)

    # input memlets
    state.add_edge(data_nodes['A'], None, spmmd_node, '_A', Memlet(data=kwargs['A']))
    state.add_edge(data_nodes['B'], None, spmmd_node, '_B', Memlet(data=kwargs['B']))
    if beta != 0.0:  # add C input memlet if neccessary
        state.add_edge(data_nodes['C'], None, spmmd_node, '_C', Memlet(data=kwargs['C']))

    # output memlet
    state.add_edge(spmmd_node, '_C', data_nodes['C'], None, Memlet(data=kwargs['C']))

    return []