''' DaCe SDFG linear algebra operation library. '''

import copy
import dace
import dace.sdfg as sd
import dace.subsets as sbs
from dace import symbolic
import typing

State = dace.sdfg.SDFGState
Shape = typing.List[typing.Union[int, dace.symbol]]
Index = typing.List[typing.Union[int, str, dace.symbol]]
Node = dace.graph.nodes.Node
DNode = dace.graph.nodes.AccessNode

# TODO: Most of the external operations here emit Z (complex double) ops, fix


# TODO: Refactor to use GPUTransformLocalStorage?
def gpu_transform_tasklet(sdfg, graph, tasklet_node):
    """ Transforms a tasklet to run on the GPU. Adapted from 
        `GPUTransformLocalStorage`.
        @see: dace.transformation.dataflow.GPUTransformLocalStorage
    """
    cnode = tasklet_node
    exit_nodes = [tasklet_node]

    gpu_storage_types = [
        dace.types.StorageType.GPU_Global, dace.types.StorageType.GPU_Shared,
        dace.types.StorageType.GPU_Stack
    ]

    #######################################################
    # Add GPU copies of CPU arrays (i.e., not already on GPU)

    # First, understand which arrays to clone
    all_out_edges = []
    for enode in exit_nodes:
        all_out_edges.extend(list(graph.out_edges(enode)))
    in_arrays_to_clone = set()
    out_arrays_to_clone = set()
    for e in graph.in_edges(cnode):
        data_node = sd.find_input_arraynode(graph, e)
        if data_node.desc(sdfg).storage not in gpu_storage_types:
            in_arrays_to_clone.add((data_node, e.data))
    for e in all_out_edges:
        data_node = sd.find_output_arraynode(graph, e)
        if data_node.desc(sdfg).storage not in gpu_storage_types:
            out_arrays_to_clone.add((data_node, e.data))

    # Second, create a GPU clone of each array
    # TODO: Overapproximate union of memlets
    cloned_arrays = {}
    in_cloned_arraynodes = {}
    out_cloned_arraynodes = {}
    for array_node, memlet in in_arrays_to_clone:
        array = array_node.desc(sdfg)
        cloned_name = 'gpu_' + array_node.data
        for i, r in enumerate(memlet.bounding_box_size()):
            size = symbolic.overapproximate(r)
            try:
                if int(size) == 1:
                    suffix = []
                    for c in str(memlet.subset[i][0]):
                        if c.isalpha() or c.isdigit() or c == '_':
                            suffix.append(c)
                        elif c == '+':
                            suffix.append('p')
                        elif c == '-':
                            suffix.append('m')
                        elif c == '*':
                            suffix.append('t')
                        elif c == '/':
                            suffix.append('d')
                    cloned_name += '_' + ''.join(suffix)
            except:
                continue
        if cloned_name in sdfg.arrays.keys():
            cloned_array = sdfg.arrays[cloned_name]
        elif array_node.data in cloned_arrays:
            cloned_array = cloned_arrays[array_node.data]
        else:
            full_shape = []
            for r in memlet.bounding_box_size():
                size = symbolic.overapproximate(r)
                try:
                    full_shape.append(int(size))
                except:
                    full_shape.append(size)
            actual_dims = [
                idx for idx, r in enumerate(full_shape)
                if not (isinstance(r, int) and r == 1)
            ]
            if len(actual_dims) == 0:  # abort
                actual_dims = [len(full_shape) - 1]
            if isinstance(array, dace.data.Scalar):
                cloned_array = sdfg.add_array(
                    name=cloned_name,
                    shape=[1],
                    dtype=array.dtype,
                    transient=True,
                    storage=dace.types.StorageType.GPU_Global)
            else:
                cloned_array = sdfg.add_array(
                    name=cloned_name,
                    shape=[full_shape[d] for d in actual_dims],
                    dtype=array.dtype,
                    materialize_func=array.materialize_func,
                    transient=True,
                    storage=dace.types.StorageType.GPU_Global,
                    allow_conflicts=array.allow_conflicts,
                    access_order=tuple(
                        [array.access_order[d] for d in actual_dims]),
                    strides=[array.strides[d] for d in actual_dims],
                    offset=[array.offset[d] for d in actual_dims])
            cloned_arrays[array_node.data] = cloned_name
        cloned_node = type(array_node)(cloned_name)

        in_cloned_arraynodes[array_node.data] = cloned_node
    for array_node, memlet in out_arrays_to_clone:
        array = array_node.desc(sdfg)
        cloned_name = 'gpu_' + array_node.data
        for i, r in enumerate(memlet.bounding_box_size()):
            size = symbolic.overapproximate(r)
            try:
                if int(size) == 1:
                    suffix = []
                    for c in str(memlet.subset[i][0]):
                        if c.isalpha() or c.isdigit() or c == '_':
                            suffix.append(c)
                        elif c == '+':
                            suffix.append('p')
                        elif c == '-':
                            suffix.append('m')
                        elif c == '*':
                            suffix.append('t')
                        elif c == '/':
                            suffix.append('d')
                    cloned_name += '_' + ''.join(suffix)
            except:
                continue
        if cloned_name in sdfg.arrays.keys():
            cloned_array = sdfg.arrays[cloned_name]
        elif array_node.data in cloned_arrays:
            cloned_array = cloned_arrays[array_node.data]
        else:
            full_shape = []
            for r in memlet.bounding_box_size():
                size = symbolic.overapproximate(r)
                try:
                    full_shape.append(int(size))
                except:
                    full_shape.append(size)
            actual_dims = [
                idx for idx, r in enumerate(full_shape)
                if not (isinstance(r, int) and r == 1)
            ]
            if len(actual_dims) == 0:  # abort
                actual_dims = [len(full_shape) - 1]
            if isinstance(array, dace.data.Scalar):
                cloned_array = sdfg.add_array(
                    name=cloned_name,
                    shape=[1],
                    dtype=array.dtype,
                    transient=True,
                    storage=dace.types.StorageType.GPU_Global)
            else:
                cloned_array = sdfg.add_array(
                    name=cloned_name,
                    shape=[full_shape[d] for d in actual_dims],
                    dtype=array.dtype,
                    materialize_func=array.materialize_func,
                    transient=True,
                    storage=dace.types.StorageType.GPU_Global,
                    allow_conflicts=array.allow_conflicts,
                    access_order=tuple(
                        [array.access_order[d] for d in actual_dims]),
                    strides=[array.strides[d] for d in actual_dims],
                    offset=[array.offset[d] for d in actual_dims])
            cloned_arrays[array_node.data] = cloned_name
        cloned_node = type(array_node)(cloned_name)
        cloned_node.setzero = True

        out_cloned_arraynodes[array_node.data] = cloned_node

    # Third, connect the cloned arrays to the originals
    for array_name, node in in_cloned_arraynodes.items():
        graph.add_node(node)
        is_scalar = isinstance(sdfg.arrays[array_name], dace.data.Scalar)
        for edge in graph.in_edges(cnode):
            if edge.data.data == array_name:
                graph.remove_edge(edge)
                newmemlet = copy.deepcopy(edge.data)
                newmemlet.data = node.data

                if is_scalar:
                    newmemlet.subset = sbs.Indices([0])
                else:
                    offset = []
                    lost_dims = []
                    lost_ranges = []
                    newsubset = [None] * len(edge.data.subset)
                    for ind, r in enumerate(edge.data.subset):
                        offset.append(r[0])
                        if isinstance(edge.data.subset[ind], tuple):
                            begin = edge.data.subset[ind][0] - r[0]
                            end = edge.data.subset[ind][1] - r[0]
                            step = edge.data.subset[ind][2]
                            if begin == end:
                                lost_dims.append(ind)
                                lost_ranges.append((begin, end, step))
                            else:
                                newsubset[ind] = (begin, end, step)
                        else:
                            newsubset[ind] -= r[0]
                    if len(lost_dims) == len(edge.data.subset):
                        newmemlet.subset = type(
                            edge.data.subset)([lost_ranges[-1]])
                    else:
                        newmemlet.subset = type(edge.data.subset)(
                            [r for r in newsubset if r is not None])

                graph.add_edge(node, edge.src_conn, edge.dst, edge.dst_conn,
                               newmemlet)

                edge.data.other_subset = newmemlet.subset
                graph.add_edge(edge.src, None, node, None, edge.data)
    for array_name, node in out_cloned_arraynodes.items():
        graph.add_node(node)
        is_scalar = isinstance(sdfg.arrays[array_name], dace.data.Scalar)
        for edge in all_out_edges:
            if edge.data.data == array_name:
                graph.remove_edge(edge)
                newmemlet = copy.deepcopy(edge.data)
                newmemlet.data = node.data

                if is_scalar:
                    newmemlet.subset = sbs.Indices([0])
                else:
                    offset = []
                    lost_dims = []
                    lost_ranges = []
                    newsubset = [None] * len(edge.data.subset)
                    for ind, r in enumerate(edge.data.subset):
                        offset.append(r[0])
                        if isinstance(edge.data.subset[ind], tuple):
                            begin = edge.data.subset[ind][0] - r[0]
                            end = edge.data.subset[ind][1] - r[0]
                            step = edge.data.subset[ind][2]
                            if begin == end:
                                lost_dims.append(ind)
                                lost_ranges.append((begin, end, step))
                            else:
                                newsubset[ind] = (begin, end, step)
                        else:
                            newsubset[ind] -= r[0]
                    if len(lost_dims) == len(edge.data.subset):
                        newmemlet.subset = type(
                            edge.data.subset)([lost_ranges[-1]])
                    else:
                        newmemlet.subset = type(edge.data.subset)(
                            [r for r in newsubset if r is not None])

                graph.add_edge(edge.src, edge.src_conn, node, edge.dst_conn,
                               newmemlet)

                edge.data.data = node.data
                edge.data.other_subset = edge.data.subset
                edge.data.subset = newmemlet.subset
                graph.add_edge(node, None, edge.dst, None, edge.data)


class ValidationError(Exception):
    """ An exception raised when inputs are not validated in SDFG library 
        calls. """

    def __init__(self, message):
        super().__init__(message)


def validate_matrix_multiplication(
        A_shape: Shape,
        B_shape: Shape,
        C_shape: Shape,
        A_index: Index = None,
        B_index: Index = None,
        C_index: Index = None
) -> ((str, str, str), (str, str, str), (str, str, str), (str, str, str)):
    """ Validates a matrix multiplication operation, based on the shapes and
        indices of the arrays involved. Returns the ranges of the maps and
        memlets at all levels as strings.
    """

    # Validate input
    if len(A_shape) < 2:
        raise ValidationError(
            'Array A has less than 2 dimensions: {}'.format(A_shape))
    A_mm_shape = A_shape[-2:]
    if len(B_shape) < 2:
        raise ValidationError(
            'Array B has less than 2 dimensions: {}'.format(B_shape))
    B_mm_shape = B_shape[-2:]
    if A_mm_shape[-1] != B_mm_shape[0]:
        raise ValidationError(
            'N-dimension mismatch between arrays A and B: {} != {}'.format(
                A_mm_shape[-1], B_mm_shape[0]))

    # Dimension sizes and ranges
    M = A_mm_shape[0]
    N = A_mm_shape[-1]
    K = B_mm_shape[-1]
    M_range = '0:{}'.format(M)
    N_range = '0:{}'.format(N)
    K_range = '0:{}'.format(K)

    # Validate slices and set input array access ranges
    A_outer_range = '{}, {}'.format(M_range, N_range)
    A_middle_range = '{}, ik'.format(M_range)
    A_inner_range = 'ii, ik'
    if len(A_shape) > 2:
        if A_index is None or len(A_index) != len(A_shape) - 2:
            raise ValidationError(
                'Invalid slice {} for array A with dimensions {}'.format(
                    A_index, A_shape))
        A_index = [str(idx) for idx in A_index]
        A_outer_range = '{}, {}'.format(', '.join(A_index), A_outer_range)
        A_middle_range = '{}, {}'.format(', '.join(A_index), A_middle_range)
        A_inner_range = '{}, {}'.format(', '.join(A_index), A_inner_range)
    B_outer_range = '{}, {}'.format(N_range, K_range)
    B_middle_range = 'ik, {}'.format(K_range)
    B_inner_range = 'ik, ij'
    if len(B_shape) > 2:
        if B_index is None or len(B_index) != len(B_shape) - 2:
            raise ValidationError(
                'Invalid slice {} for array B with dimensions {}'.format(
                    B_index, B_shape))
        B_index = [str(idx) for idx in B_index]
        B_outer_range = '{}, {}'.format(', '.join(B_index), B_outer_range)
        B_middle_range = '{}, {}'.format(', '.join(B_index), B_middle_range)
        B_inner_range = '{}, {}'.format(', '.join(B_index), B_inner_range)

    # Validate output
    C_mm_shape = [M, K]
    if len(C_shape) < 2:
        raise ValidationError(
            'Array C has less than 2 dimensions: {}'.format(C_shape))
    if list(C_shape[-2:]) != C_mm_shape:
        raise ValidationError(
            'Shape mismatch in array C: expected {}, but got {}'.format(
                C_mm_shape, C_shape[-2:]))
    C_outer_range = '{}, {}'.format(M_range, K_range)
    C_middle_range = '{}, {}'.format(M_range, K_range)
    C_inner_range = 'ii, ij'
    if len(C_shape) > 2:
        if C_index is None or len(C_index) != len(C_shape) - 2:
            raise ValidationError(
                'Invalid slice {} for array C with dimensions {}'.format(
                    C_index, C_shape))
        C_index = [str(idx) for idx in C_index]
        C_outer_range = '{}, {}'.format(', '.join(C_index), C_outer_range)
        C_middle_range = '{}, {}'.format(', '.join(C_index), C_middle_range)
        C_inner_range = '{}, {}'.format(', '.join(C_index), C_inner_range)

    return ((M_range, N_range, K_range), (A_outer_range, A_middle_range,
                                          A_inner_range),
            (B_outer_range, B_middle_range,
             B_inner_range), (C_outer_range, C_middle_range, C_inner_range))


def matrix_multiplication(state: State,
                          A_src: Node,
                          A_node: DNode,
                          B_src: Node,
                          B_node: DNode,
                          C_dst: Node,
                          C_node: DNode,
                          accumulate: bool = False,
                          interchange: bool = True,
                          A_index: Index = None,
                          B_index: Index = None,
                          C_index: Index = None,
                          label: str = None):
    """ Adds a matrix multiplication operation to an existing SDFG state.
        @param A_src: The source node from which the memlet of matrix A is
                      connected.
        @param A_node: The Access Node for matrix A.
        @param B_src: The source node from which the memlet of matrix B is
                      connected.
        @param B_node: The Access Node for matrix B.
        @param C_dst: The destination node to which the memlet of matrix C is
                      connected.
        @param C_node: The Access Node for matrix C.
        @param accumulate: Whether to accumulate to C or store to it.
        @param interchange: If True, interchanges the multiplication maps for
                            performance (in some cases).
        @param A_index: Slice of matrix A to use for multiplication.
        @param B_index: Slice of matrix B to use for multiplication.
        @param C_index: Slice of matrix C to use for multiplication.
        @param label: Optional label for the maps and tasklet.
    """

    # Validate input
    sdfg = state.parent
    map_ranges, A_ranges, B_ranges, C_ranges = validate_matrix_multiplication(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape,
        C_node.desc(sdfg).shape, A_index, B_index, C_index)

    # Extract ranges
    M_range, N_range, K_range = map_ranges
    A_outer_range, A_middle_range, A_inner_range = A_ranges
    B_outer_range, B_middle_range, B_inner_range = B_ranges
    C_outer_range, C_middle_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = state.label

    # Create maps/tasklet
    k_entry, k_exit = state.add_map(
        name=label + '_' + 'k_map',
        ndrange=dict(ik=N_range),
        schedule=dace.types.ScheduleType.Sequential)
    k_entry.in_connectors = {'IN_1', 'IN_2'}
    k_entry.out_connectors = {'OUT_1', 'OUT_2'}
    k_exit.in_connectors = {'IN_1'}
    k_exit.out_connectors = {'OUT_1'}
    ij_entry, ij_exit = state.add_map(
        name=label + '_' + 'ij_map', ndrange=dict(ii=M_range, ij=K_range))
    tasklet = state.add_tasklet(
        name=label + '_' + 'tasklet',
        inputs={'a', 'b'},
        outputs={'c'},
        code='c = a * b')
    ij_entry.in_connectors = {'IN_1', 'IN_2'}
    ij_entry.out_connectors = {'OUT_1', 'OUT_2'}
    ij_exit.in_connectors = {'IN_1'}
    ij_exit.out_connectors = {'OUT_1'}

    # Add edges
    if interchange:
        state.add_edge(A_src, None, k_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_outer_range))
        state.add_edge(B_src, None, k_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_outer_range))
        state.add_edge(k_entry, 'OUT_1', ij_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_middle_range))
        state.add_edge(k_entry, 'OUT_2', ij_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_middle_range))
        state.add_edge(ij_entry, 'OUT_1', tasklet, 'a',
                       dace.Memlet.simple(A_node, A_inner_range))
        state.add_edge(ij_entry, 'OUT_2', tasklet, 'b',
                       dace.Memlet.simple(B_node, B_inner_range))
        wcr = 0
        if accumulate:
            wcr = None
        state.add_edge(
            tasklet, 'c', ij_exit, 'IN_1',
            dace.Memlet.simple(
                C_node,
                C_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=wcr,
                wcr_conflict=False))
        state.add_edge(ij_exit, 'OUT_1', k_exit, 'IN_1',
                       dace.Memlet.simple(C_node, C_middle_range))
        state.add_edge(k_exit, 'OUT_1', C_dst, None,
                       dace.Memlet.simple(C_node, C_outer_range))
    else:
        state.add_edge(A_src, None, ij_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_outer_range))
        state.add_edge(B_src, None, ij_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_outer_range))
        state.add_edge(ij_entry, 'OUT_1', k_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_middle_range))
        state.add_edge(ij_entry, 'OUT_2', k_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_middle_range))
        state.add_edge(k_entry, 'OUT_1', tasklet, 'a',
                       dace.Memlet.simple(A_node, A_inner_range))
        state.add_edge(k_entry, 'OUT_2', tasklet, 'b',
                       dace.Memlet.simple(B_node, B_inner_range))
        wcr = 0
        if accumulate:
            wcr = None
        state.add_edge(
            tasklet, 'c', k_exit, 'IN_1',
            dace.Memlet.simple(
                C_node,
                C_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=wcr,
                wcr_conflict=False))
        state.add_edge(k_exit, 'OUT_1', ij_exit, 'IN_1',
                       dace.Memlet.simple(C_node, C_middle_range))
        state.add_edge(ij_exit, 'OUT_1', C_dst, None,
                       dace.Memlet.simple(C_node, C_outer_range))


def matrix_multiplication_cublas(state: State,
                                 A_src: Node,
                                 A_node: DNode,
                                 B_src: Node,
                                 B_node: DNode,
                                 C_dst: Node,
                                 C_node: DNode,
                                 accumulate: bool = False,
                                 interchange: bool = True,
                                 alpha: str = 'const_pone',
                                 beta: str = 'const_zero',
                                 A_index: Index = None,
                                 B_index: Index = None,
                                 C_index: Index = None,
                                 label: str = None):
    """ Adds a matrix multiplication operation to an existing SDFG state,
        using CUBLAS as the implementation.
        @param A_src: The source node from which the memlet of matrix A is
                      connected.
        @param A_node: The Access Node for matrix A.
        @param B_src: The source node from which the memlet of matrix B is
                      connected.
        @param B_node: The Access Node for matrix B.
        @param C_dst: The destination node to which the memlet of matrix C is
                      connected.
        @param C_node: The Access Node for matrix C.
        @param accumulate: Whether to accumulate to C or store to it.
        @param interchange: If True, interchanges the multiplication maps for
                            performance (in some cases).
        @param alpha: Alpha value for GEMM.
        @param beta: Beta value for GEMM.
        @param A_index: Slice of matrix A to use for multiplication.
        @param B_index: Slice of matrix B to use for multiplication.
        @param C_index: Slice of matrix C to use for multiplication.
        @param label: Optional label for the maps and tasklet.
    """

    # Validate input
    sdfg = state.parent
    map_ranges, A_ranges, B_ranges, C_ranges = validate_matrix_multiplication(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape,
        C_node.desc(sdfg).shape, A_index, B_index, C_index)

    # Extract ranges
    M_range, N_range, K_range = map_ranges
    A_outer_range, A_middle_range, A_inner_range = A_ranges
    B_outer_range, B_middle_range, B_inner_range = B_ranges
    C_outer_range, C_middle_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = state.label

    # Create tasklet
    tasklet = state.add_tasklet(
        name=label + '_' + 'tasklet',
        inputs={'a', 'b'},
        outputs={'c'},
        code='''
        //cuDoubleComplex alpha = make_cuDoubleComplex(1, 0);
        //cuDoubleComplex beta = make_cuDoubleComplex(0, 0);
        cublasSetStream(handle, __dace_current_stream);
        cublasStatus_t status = cublasZgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            bsize, bsize, bsize,
            const_pone,
            (cuDoubleComplex*)b, bsize,
            (cuDoubleComplex*)a, bsize,
            const_zero,
            (cuDoubleComplex*)c, bsize
        );
        ''',  # cuBLAS is column-major, so we switch the arguments
        language=dace.types.Language.CPP)

    state.add_edge(A_src, None, tasklet, 'a',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(B_src, None, tasklet, 'b',
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(tasklet, 'c', C_dst, None,
                   dace.Memlet.simple(C_node, C_outer_range))

    gpu_transform_tasklet(sdfg, state, tasklet)


def matrix_multiplication_cublas_v2(state: State,
                                    A_src: Node,
                                    A_node: DNode,
                                    B_src: Node,
                                    B_node: DNode,
                                    C_src: Node,
                                    C_src_node: DNode,
                                    C_dst: Node,
                                    C_dst_node: DNode,
                                    accumulate: bool = False,
                                    interchange: bool = True,
                                    alpha: str = 'const_pone',
                                    beta: str = 'const_zero',
                                    A_index: Index = None,
                                    B_index: Index = None,
                                    C_index: Index = None,
                                    label: str = None):
    """ Adds a matrix multiplication operation to an existing SDFG state,
        using CUBLAS as the implementation, and providing a separate source
        and destination nodes for the output matrix.
        @param A_src: The source node from which the memlet of matrix A is
                      connected.
        @param A_node: The Access Node for matrix A.
        @param B_src: The source node from which the memlet of matrix B is
                      connected.
        @param B_node: The Access Node for matrix B.
        @param C_src: The node from which the memlet of matrix C is
                      connected into the multiplication.
        @param C_src_node: The input Access Node for matrix C.
        @param C_dst: The node to which the memlet of matrix C is
                      connected out of the multiplication.
        @param C_dst_node: The output Access Node for matrix C.
        @param accumulate: Whether to accumulate to C or store to it.
        @param interchange: If True, interchanges the multiplication maps for
                            performance (in some cases).
        @param alpha: Alpha value for GEMM.
        @param beta: Beta value for GEMM.
        @param A_index: Slice of matrix A to use for multiplication.
        @param B_index: Slice of matrix B to use for multiplication.
        @param C_index: Slice of matrix C to use for multiplication.
        @param label: Optional label for the maps and tasklet.
    """

    # Validate input
    sdfg = state.parent
    map_ranges, A_ranges, B_ranges, C_ranges = validate_matrix_multiplication(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape,
        C_src_node.desc(sdfg).shape, A_index, B_index, C_index)

    # Extract ranges
    M_range, N_range, K_range = map_ranges
    A_outer_range, A_middle_range, A_inner_range = A_ranges
    B_outer_range, B_middle_range, B_inner_range = B_ranges
    C_outer_range, C_middle_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = state.label

    # Create tasklet
    tasklet = state.add_tasklet(
        name=label + '_' + 'tasklet',
        inputs={'a', 'b', 'cin'},
        outputs={'c'},
        code='''
        //cuDoubleComplex alpha = make_cuDoubleComplex(1, 0);
        //cuDoubleComplex beta = make_cuDoubleComplex(0, 0);
        cublasSetStream(handle, __dace_current_stream);
        cublasStatus_t status = cublasZgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            bsize, bsize, bsize,
            {alpha},
            (cuDoubleComplex*)b, bsize,
            (cuDoubleComplex*)a, bsize,
            {beta},
            (cuDoubleComplex*)c, bsize
        );
        '''.format(
            alpha=alpha,
            beta=beta),  # cuBLAS is column-major, so we switch the arguments
        language=dace.types.Language.CPP)

    state.add_edge(A_src, None, tasklet, 'a',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(B_src, None, tasklet, 'b',
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(C_src, None, tasklet, 'cin',
                   dace.Memlet.simple(C_src_node, C_outer_range))
    state.add_edge(tasklet, 'c', C_dst, None,
                   dace.Memlet.simple(C_dst_node, C_outer_range))

    gpu_transform_tasklet(sdfg, state, tasklet)


def matrix_multiplication_mkl(state: State,
                              A_src: Node,
                              A_node: DNode,
                              B_src: Node,
                              B_node: DNode,
                              C_dst: Node,
                              C_node: DNode,
                              accumulate: bool = False,
                              interchange: bool = True,
                              A_index: Index = None,
                              B_index: Index = None,
                              C_index: Index = None,
                              label: str = None):
    """ Adds a matrix multiplication operation to an existing SDFG state,
        using MKL as the implementation.
        @param A_src: The source node from which the memlet of matrix A is
                      connected.
        @param A_node: The Access Node for matrix A.
        @param B_src: The source node from which the memlet of matrix B is
                      connected.
        @param B_node: The Access Node for matrix B.
        @param C_dst: The destination node to which the memlet of matrix C is
                      connected.
        @param C_node: The Access Node for matrix C.
        @param accumulate: Whether to accumulate to C or store to it.
        @param interchange: If True, interchanges the multiplication maps for
                            performance (in some cases).
        @param A_index: Slice of matrix A to use for multiplication.
        @param B_index: Slice of matrix B to use for multiplication.
        @param C_index: Slice of matrix C to use for multiplication.
        @param label: Optional label for the maps and tasklet.
    """

    # Validate input
    sdfg = state.parent
    map_ranges, A_ranges, B_ranges, C_ranges = validate_matrix_multiplication(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape,
        C_node.desc(sdfg).shape, A_index, B_index, C_index)

    # Extract ranges
    M = A_node.desc(sdfg).shape[-2]
    N = A_node.desc(sdfg).shape[-1]
    K = B_node.desc(sdfg).shape[-1]
    M_range, N_range, K_range = map_ranges
    A_outer_range, A_middle_range, A_inner_range = A_ranges
    B_outer_range, B_middle_range, B_inner_range = B_ranges
    C_outer_range, C_middle_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = state.label

    # Create tasklet
    tasklet = state.add_tasklet(
        name=label + '_' + 'tasklet',
        inputs={'a', 'b'},
        outputs={'c'},
        code='''
        std::complex<double> alpha(1, 0);
        std::complex<double> beta(0, 0);
        char opa = 'N';
        char opb = 'N';
        zgemm(
            &opa, &opb,
            &{m}, &{n}, &{k},
            (MKL_Complex16*)&alpha,
            (MKL_Complex16*)a, &{m},
            (MKL_Complex16*)b, &{n},
            (MKL_Complex16*)&beta,
            (MKL_Complex16*)c, &{m}
        );
        '''.format(m=M, n=N, k=K),
        language=dace.types.Language.CPP)

    state.add_edge(A_src, None, tasklet, 'a',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(B_src, None, tasklet, 'b',
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(tasklet, 'c', C_dst, None,
                   dace.Memlet.simple(C_node, C_outer_range))


def matrix_multiplication_s(A_label: str,
                            A_shape: Shape,
                            A_type: dace.types.typeclass,
                            B_label: str,
                            B_shape: Shape,
                            B_type: dace.types.typeclass,
                            create_C: bool = True,
                            C_label: str = None,
                            C_shape: Shape = None,
                            C_type: dace.types.typeclass = None,
                            is_A_transient: bool = False,
                            is_B_transient: bool = False,
                            is_C_transient: bool = False,
                            accumulate: bool = False,
                            interchange: bool = True,
                            A_index: Index = None,
                            B_index: Index = None,
                            C_index: Index = None,
                            label: str = None) -> State:
    """ Creates a new state with a matrix multiplication operation. """

    # Set output attributes
    if create_C:
        if C_label is None:
            C_label = A_label + B_label
        if C_type is None:
            C_type = A_type
        C_shape = [A_shape[-2], B_shape[-1]]
    else:
        if C_shape is None:
            raise ValidationError(
                'Array C is not transient, but its shape is not set')

    # Validate input
    map_ranges, A_ranges, B_ranges, C_ranges = validate_matrix_multiplication(
        A_shape, B_shape, C_shape, A_index, B_index, C_index)

    # Extract ranges
    M_range, N_range, K_range = map_ranges
    A_outer_range, A_middle_range, A_inner_range = A_ranges
    B_outer_range, B_middle_range, B_inner_range = B_ranges
    C_outer_range, C_middle_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = A_label + B_label

    # Create state
    state = State(label=label)

    # Create data nodes
    A_node = state.add_array(
        A_label, A_shape, A_type, transient=is_A_transient)
    B_node = state.add_array(
        B_label, B_shape, B_type, transient=is_B_transient)
    C_node = state.add_array(
        C_label, C_shape, C_type, transient=is_C_transient or create_C)

    # Create maps/tasklet
    k_entry, k_exit = state.add_map(
        name=label + '_' + 'k_map',
        ndrange=dict(ik=N_range),
        schedule=dace.types.ScheduleType.Sequential)
    k_entry.in_connectors = {'IN_1', 'IN_2'}
    k_entry.out_connectors = {'OUT_1', 'OUT_2'}
    k_exit.in_connectors = {'IN_1'}
    k_exit.out_connectors = {'OUT_1'}
    ij_entry, ij_exit = state.add_map(
        name=label + '_' + 'ij_map', ndrange=dict(ii=M_range, ij=K_range))
    tasklet = state.add_tasklet(
        name=label + '_' + 'tasklet',
        inputs={'a', 'b'},
        outputs={'c'},
        code='c = a * b')
    ij_entry.in_connectors = {'IN_1', 'IN_2'}
    ij_entry.out_connectors = {'OUT_1', 'OUT_2'}
    ij_exit.in_connectors = {'IN_1'}
    ij_exit.out_connectors = {'OUT_1'}

    # Add edges
    if interchange:
        state.add_edge(A_node, None, k_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_outer_range))
        state.add_edge(B_node, None, k_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_outer_range))
        state.add_edge(k_entry, 'OUT_1', ij_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_middle_range))
        state.add_edge(k_entry, 'OUT_2', ij_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_middle_range))
        state.add_edge(ij_entry, 'OUT_1', tasklet, 'a',
                       dace.Memlet.simple(A_node, A_inner_range))
        state.add_edge(ij_entry, 'OUT_2', tasklet, 'b',
                       dace.Memlet.simple(B_node, B_inner_range))
        wcr = 0
        if accumulate:
            wcr = None
        state.add_edge(
            tasklet, 'c', ij_exit, 'IN_1',
            dace.Memlet.simple(
                C_node,
                C_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=wcr,
                wcr_conflict=False))
        state.add_edge(ij_exit, 'OUT_1', k_exit, 'IN_1',
                       dace.Memlet.simple(C_node, C_middle_range))
        state.add_edge(k_exit, 'OUT_1', C_node, None,
                       dace.Memlet.simple(C_node, C_outer_range))
    else:
        state.add_edge(A_node, None, ij_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_outer_range))
        state.add_edge(B_node, None, ij_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_outer_range))
        state.add_edge(ij_entry, 'OUT_1', k_entry, 'IN_1',
                       dace.Memlet.simple(A_node, A_middle_range))
        state.add_edge(ij_entry, 'OUT_2', k_entry, 'IN_2',
                       dace.Memlet.simple(B_node, B_middle_range))
        state.add_edge(k_entry, 'OUT_1', tasklet, 'a',
                       dace.Memlet.simple(A_node, A_inner_range))
        state.add_edge(k_entry, 'OUT_2', tasklet, 'b',
                       dace.Memlet.simple(B_node, B_inner_range))
        wcr = 0
        if accumulate:
            wcr = None
        state.add_edge(
            tasklet, 'c', k_exit, 'IN_1',
            dace.Memlet.simple(
                C_node,
                C_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=wcr,
                wcr_conflict=False))
        state.add_edge(k_exit, 'OUT_1', ij_exit, 'IN_1',
                       dace.Memlet.simple(C_node, C_middle_range))
        state.add_edge(ij_exit, 'OUT_1', C_node, None,
                       dace.Memlet.simple(C_node, C_outer_range))

    return state


def validate_scalar_array_multiplication(
        alpha_shape: Shape,
        A_shape: Shape,
        B_shape: Shape,
        alpha_index: Index = None,
        A_index: Index = None,
        B_index: Index = None
) -> (typing.Dict[str, str], (str, str), (str, str), (str, str)):
    """ Validates a scalar-array multiplication operation, based on the shapes 
        and indices of the arrays involved. Returns the ranges of the maps and
        memlets at all levels as strings. """

    # Validate data
    if alpha_shape != [1]:
        if alpha_index is None or len(alpha_shape) != len(alpha_index):
            raise ValidationError(
                'Slice of alpha is not a scalar: {}, {}'.format(
                    alpha_shape, alpha_index))
    if A_index is not None:
        true_A_shape = A_shape[len(A_index):]
    else:
        true_A_shape = A_shape
    if B_index is not None:
        true_B_shape = B_shape[len(B_index):]
    else:
        true_B_shape = B_shape
    if true_A_shape != true_B_shape:
        raise ValidationError('Dimension mismatch between arrays A and B: '
                              '{}({}) != {}({})'.format(
                                  true_A_shape, A_shape, true_B_shape,
                                  B_shape))

    # Map ranges
    map_ranges = dict()
    for i, dim in enumerate(true_A_shape):
        map_ranges['i{}'.format(i)] = '0:{}'.format(dim)

    # Memlet ranges
    alpha_outer_range = '0'
    alpha_inner_range = '0'
    if alpha_index is not None:
        alpha_index = [str(idx) for idx in alpha_index]
        alpha_outer_range = ', '.join(alpha_index)
        alpha_inner_range = ', '.join(alpha_index)
    A_outer_range = ', '.join(map_ranges.values())
    A_inner_range = ', '.join(map_ranges.keys())
    if A_index is not None:
        A_index = [str(idx) for idx in A_index]
        A_outer_range = '{}, {}'.format(', '.join(A_index), A_outer_range)
        A_inner_range = '{}, {}'.format(', '.join(A_index), A_inner_range)
    B_outer_range = ', '.join(map_ranges.values())
    B_inner_range = ', '.join(map_ranges.keys())
    if B_index is not None:
        B_index = [str(idx) for idx in B_index]
        B_outer_range = '{}, {}'.format(', '.join(B_index), B_outer_range)
        B_inner_range = '{}, {}'.format(', '.join(B_index), B_inner_range)

    return (map_ranges, (alpha_outer_range, alpha_inner_range),
            (A_outer_range, A_inner_range), (B_outer_range, B_inner_range))


def scalar_array_multiplication(state: State,
                                alpha_src: Node,
                                alpha_node: DNode,
                                A_src: Node,
                                A_node: DNode,
                                B_dst: Node,
                                B_node: DNode,
                                accumulate: bool = False,
                                wcr_conflict: bool = False,
                                alpha_index: Index = None,
                                A_index: Index = None,
                                B_index: Index = None,
                                label: str = None):
    """ Adds a scalar-array multiplication operation to an exisiting state. """

    # Validate data
    sdfg = state.parent
    alpha_shape = [1]
    if hasattr(alpha_node, 'shape'):
        alpha_shape = alpha_node.shape
    ranges = validate_scalar_array_multiplication(
        alpha_shape,
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape, alpha_index, A_index, B_index)
    map_ranges, alpha_ranges, A_ranges, B_ranges = ranges
    alpha_outer_range, alpha_inner_range = alpha_ranges
    A_outer_range, A_inner_range = A_ranges
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges

    # Set label
    if label is None:
        label = state.label

    # Create map/tasklet
    map_entry, map_exit = state.add_map(
        name=label + '_map', ndrange=map_ranges)
    map_entry.in_connectors = {'IN_1', 'IN_2'}
    map_entry.out_connectors = {'OUT_1', 'OUT_2'}
    map_exit.in_connectors = {'IN_1'}
    map_exit.out_connectors = {'OUT_1'}
    tasklet = state.add_tasklet(
        name=label + '_tasklet',
        inputs={'scalar', 'a'},
        outputs={'b'},
        code='b = scalar * a')

    # Add edges
    state.add_edge(alpha_src, None, map_entry, 'IN_1',
                   dace.Memlet.simple(alpha_node, alpha_outer_range))
    state.add_edge(A_src, None, map_entry, 'IN_2',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(map_exit, 'OUT_1', B_dst, None,
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(map_entry, 'OUT_1', tasklet, 'scalar',
                   dace.Memlet.simple(alpha_node, alpha_inner_range))
    state.add_edge(map_entry, 'OUT_2', tasklet, 'a',
                   dace.Memlet.simple(A_node, A_inner_range))
    if accumulate:
        state.add_edge(
            tasklet, 'b', map_exit, 'IN_1',
            dace.Memlet.simple(
                B_node,
                B_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=None,
                wcr_conflict=wcr_conflict))
    else:
        state.add_edge(tasklet, 'b', map_exit, 'IN_1',
                       dace.Memlet.simple(B_node, B_inner_range))


def scalar_array_multiplication_s(alpha_label: str,
                                  alpha_shape: Shape,
                                  alpha_type: dace.types.typeclass,
                                  A_label: str,
                                  A_shape: Shape,
                                  A_type: dace.types.typeclass,
                                  create_B: bool = True,
                                  B_label: str = None,
                                  B_shape: Shape = None,
                                  B_type: dace.types.typeclass = None,
                                  is_alpha_transient: bool = False,
                                  is_A_transient: bool = False,
                                  is_B_transient: bool = False,
                                  accumulate: bool = False,
                                  wcr_conflict: bool = False,
                                  alpha_index: Index = None,
                                  A_index: Index = None,
                                  B_index: Index = None,
                                  label: str = None) -> State:
    """ Creates a new state with a scalar-array multiplication operation. """

    # Set output attributes
    if create_B:
        if B_label is None:
            B_label = alpha_label + A_label
        if B_type is None:
            B_type = A_type
        B_shape = A_shape
    else:
        if B_shape is None:
            raise ValidationError(
                'Array B is not transient, but its shape is not set')

    # Validate data
    ranges = validate_scalar_array_multiplication(
        alpha_shape, A_shape, B_shape, alpha_index, A_index, B_index)
    map_ranges, alpha_ranges, A_ranges, B_ranges = ranges
    alpha_outer_range, alpha_inner_range = alpha_ranges
    A_outer_range, A_inner_range = A_ranges
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges

    # Set label
    if label is None:
        label = alpha_label + A_label

    # Create state
    state = State(label=label)

    # Create data nodes
    alpha_node = state.add_array(
        alpha_label, alpha_shape, alpha_type, transient=is_alpha_transient)
    A_node = state.add_array(
        A_label, A_shape, A_type, transient=is_A_transient)
    B_node = state.add_array(
        B_label, B_shape, B_type, transient=is_B_transient or create_B)

    # Create map/tasklet
    map_entry, map_exit = state.add_map(
        name=label + '_map', ndrange=map_ranges)
    map_entry.in_connectors = {'IN_1', 'IN_2'}
    map_entry.out_connectors = {'OUT_1', 'OUT_2'}
    map_exit.in_connectors = {'IN_1'}
    map_exit.out_connectors = {'OUT_1'}
    tasklet = state.add_tasklet(
        name=label + '_tasklet',
        inputs={'scalar', 'a'},
        outputs={'b'},
        code='b = scalar * a')

    # Add edges
    state.add_edge(alpha_node, None, map_entry, 'IN_1',
                   dace.Memlet.simple(alpha_node, alpha_outer_range))
    state.add_edge(A_node, None, map_entry, 'IN_2',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(map_exit, 'OUT_1', B_node, None,
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(map_entry, 'OUT_1', tasklet, 'scalar',
                   dace.Memlet.simple(alpha_node, alpha_inner_range))
    state.add_edge(map_entry, 'OUT_2', tasklet, 'a',
                   dace.Memlet.simple(A_node, A_inner_range))
    if accumulate:
        state.add_edge(
            tasklet, 'b', map_exit, 'IN_1',
            dace.Memlet.simple(
                B_node,
                B_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=None,
                wcr_conflict=wcr_conflict))
    else:
        state.add_edge(tasklet, 'b', map_exit, 'IN_1',
                       dace.Memlet.simple(B_node, B_inner_range))

    return state


def constant_array_multiplication(state: State,
                                  constant,
                                  A_src: Node,
                                  A_node: DNode,
                                  B_dst: Node,
                                  B_node: DNode,
                                  accumulate: bool = False,
                                  A_index: Index = None,
                                  B_index: Index = None,
                                  label: str = None):
    """ Adds a scalar-array multiplication operation to an exisiting state. """

    # Validate data
    # ranges = validate_scalar_array_multiplication(
    #     [1], A_node.shape, B_node.shape,
    #     None, A_index, B_index
    # )
    sdfg = state.parent
    ranges = validate_scalar_array_multiplication([1],
                                                  A_node.desc(sdfg).shape,
                                                  B_node.desc(sdfg).shape,
                                                  None, A_index, B_index)
    map_ranges, _, A_ranges, B_ranges = ranges
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges

    # Set label
    if label is None:
        label = state.label

    # Create map/tasklet
    map_entry, map_exit = state.add_map(
        name=label + '_map', ndrange=map_ranges)
    map_entry.in_connectors = {'IN_1'}
    map_entry.out_connectors = {'OUT_1'}
    map_exit.in_connectors = {'IN_1'}
    map_exit.out_connectors = {'OUT_1'}
    tasklet = state.add_tasklet(
        name=label + '_tasklet',
        inputs={'a'},
        outputs={'b'},
        code='b = {} * a'.format(constant))

    # Add edges
    state.add_edge(A_src, None, map_entry, 'IN_1',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(map_exit, 'OUT_1', B_dst, None,
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(map_entry, 'OUT_1', tasklet, 'a',
                   dace.Memlet.simple(A_node, A_inner_range))
    if accumulate:
        state.add_edge(
            tasklet, 'b', map_exit, 'IN_1',
            dace.Memlet.simple(
                B_node,
                B_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=None,
                wcr_conflict=False))
    else:
        state.add_edge(tasklet, 'b', map_exit, 'IN_1',
                       dace.Memlet.simple(B_node, B_inner_range))


def unary_array_op(state: State,
                   A_src: Node,
                   A_node: DNode,
                   B_dst: Node,
                   B_node: DNode,
                   code: str,
                   lang=dace.types.Language.Python,
                   accumulate: bool = False,
                   A_index: Index = None,
                   B_index: Index = None,
                   label: str = None):
    """ Adds a unary array operation to an exisiting state. """

    # Validate data
    sdfg = state.parent
    ranges = validate_scalar_array_multiplication([1],
                                                  A_node.desc(sdfg).shape,
                                                  B_node.desc(sdfg).shape,
                                                  None, A_index, B_index)
    map_ranges, _, A_ranges, B_ranges = ranges
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges

    # Set label
    if label is None:
        label = state.label

    # Create map/tasklet
    map_entry, map_exit = state.add_map(
        name=label + '_map', ndrange=map_ranges)
    map_entry.in_connectors = {'IN_1'}
    map_entry.out_connectors = {'OUT_1'}
    map_exit.in_connectors = {'IN_1'}
    map_exit.out_connectors = {'OUT_1'}
    tasklet = state.add_tasklet(
        name=label + '_tasklet',
        inputs={'a'},
        outputs={'b'},
        code=code,
        language=lang)

    # Add edges
    state.add_edge(A_src, None, map_entry, 'IN_1',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(map_exit, 'OUT_1', B_dst, None,
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(map_entry, 'OUT_1', tasklet, 'a',
                   dace.Memlet.simple(A_node, A_inner_range))
    if accumulate:
        state.add_edge(
            tasklet, 'b', map_exit, 'IN_1',
            dace.Memlet.simple(
                B_node,
                B_inner_range,
                wcr_str='lambda x, y: x + y',
                wcr_identity=None,
                wcr_conflict=False))
    else:
        state.add_edge(tasklet, 'b', map_exit, 'IN_1',
                       dace.Memlet.simple(B_node, B_inner_range))


def validate_matrix_transpose(
        A_shape: Shape,
        B_shape: Shape,
        A_index: Index = None,
        B_index: Index = None
) -> (typing.Dict[str, str], (str, str), (str, str)):
    """ Validates a matrix transpose operation, based on the shapes and indices
        of the arrays involved. Returns the ranges of the maps and memlets at 
        all levels as strings. """

    # Validate data
    if len(A_shape) < 2:
        raise ValidationError(
            'Array A has less than 2 dimensions: {}'.format(A_shape))
    A_tr_shape = A_shape[-2:]
    if len(B_shape) < 2:
        raise ValidationError(
            'Array B has less than 2 dimensions: {}'.format(B_shape))
    B_tr_shape = B_shape[-2:]
    if A_tr_shape[0] != B_tr_shape[-1] or A_tr_shape[-1] != B_tr_shape[0]:
        raise ValidationError(
            'Dimension mismatch between arrays A and B: {} != {}'.format(
                A_tr_shape, B_tr_shape))

    # Map ranges
    map_ranges = dict(
        ii='0:{}'.format(A_tr_shape[0]), ij='0:{}'.format(A_tr_shape[-1]))

    # Validate slices and set array access ranges
    A_outer_range = '0:{}, 0:{}'.format(A_tr_shape[0], A_tr_shape[-1])
    A_inner_range = 'ii, ij'
    if len(A_shape) > 2:
        if A_index is None or len(A_index) != len(A_shape) - 2:
            raise ValidationError(
                'Invalid slice {} for array A with dimensions {}'.format(
                    A_index, A_shape))
        A_index = [str(idx) for idx in A_index]
        A_outer_range = '{}, {}'.format(', '.join(A_index), A_outer_range)
        A_inner_range = '{}, {}'.format(', '.join(A_index), A_inner_range)
    B_outer_range = '0:{}, 0:{}'.format(A_tr_shape[-1], A_tr_shape[0])
    B_inner_range = 'ij, ii'
    if len(B_shape) > 2:
        if B_index is None or len(B_index) != len(B_shape) - 2:
            raise ValidationError(
                'Invalid slice {} for array B with dimensions {}'.format(
                    B_index, B_shape))
        B_index = [str(idx) for idx in B_index]
        B_outer_range = '{}, {}'.format(', '.join(B_index), B_outer_range)
        B_inner_range = '{}, {}'.format(', '.join(B_index), B_inner_range)

    return (map_ranges, (A_outer_range, A_inner_range), (B_outer_range,
                                                         B_inner_range))


def matrix_transpose(state: State,
                     A_src: Node,
                     A_node: DNode,
                     B_dst: Node,
                     B_node: DNode,
                     A_index: Index = None,
                     B_index: Index = None,
                     code: str = None,
                     lang=dace.types.Language.Python,
                     label: str = None):
    """ Adds a matrix transpose operation to an existing state. """

    # Validate data
    sdfg = state.parent
    map_ranges, A_ranges, B_ranges = validate_matrix_transpose(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape, A_index, B_index)
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges

    # Set label
    if label is None:
        label = state.label

    # Create map/tasklet
    if code is None:
        code = 'b = a'
    _, map_entry, map_exit = state.add_mapped_tasklet(
        name=label,
        map_ranges=map_ranges,
        inputs=dict(a=dace.Memlet.simple(A_node, A_inner_range)),
        outputs=dict(b=dace.Memlet.simple(B_node, B_inner_range)),
        code=code,
        language=lang)

    # Add edges
    state.add_nedge(A_src, map_entry, dace.Memlet.simple(
        A_node, A_outer_range))
    state.add_nedge(map_exit, B_dst, dace.Memlet.simple(B_node, B_outer_range))

    return state


def matrix_transpose_double(state: State,
                            A_src: Node,
                            A_node: DNode,
                            B_dst: Node,
                            B_node: DNode,
                            C_dst: Node,
                            C_node: DNode,
                            A_index: Index = None,
                            B_index: Index = None,
                            C_index: Index = None,
                            code: str = None,
                            lang=dace.types.Language.Python,
                            label: str = None):
    """ Adds a matrix transpose operation, which transposes to two different
        matrices, to an existing state. """

    # Validate data
    sdfg = state.parent
    map_ranges, A_ranges, B_ranges = validate_matrix_transpose(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape, A_index, B_index)
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges
    _, _, C_ranges = validate_matrix_transpose(
        A_node.desc(sdfg).shape,
        C_node.desc(sdfg).shape, A_index, C_index)
    C_outer_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = state.label

    # Create map/tasklet
    if code is None:
        code = '''
b = a
c = a
        '''
    _, map_entry, map_exit = state.add_mapped_tasklet(
        name=label,
        map_ranges=map_ranges,
        inputs=dict(a=dace.Memlet.simple(A_node, A_inner_range)),
        outputs=dict(
            b=dace.Memlet.simple(B_node, B_inner_range),
            c=dace.Memlet.simple(C_node, C_inner_range),
        ),
        code=code,
        language=lang)

    # Add edges
    state.add_nedge(A_src, map_entry, dace.Memlet.simple(
        A_node, A_outer_range))
    state.add_nedge(map_exit, B_dst, dace.Memlet.simple(B_node, B_outer_range))
    state.add_nedge(map_exit, C_dst, dace.Memlet.simple(C_node, C_outer_range))

    return state


def matrix_transpose_s(A_label: str,
                       A_shape: Shape,
                       A_type: dace.types.typeclass,
                       create_B: bool = True,
                       B_label: str = None,
                       B_shape: Shape = None,
                       B_type: dace.types.typeclass = None,
                       is_alpha_transient: bool = False,
                       is_A_transient: bool = False,
                       is_B_transient: bool = False,
                       A_index: Index = None,
                       B_index: Index = None,
                       label: str = None) -> State:
    """ Creates a new state with a matrix transpose operation. """

    # Set output attributes
    if create_B:
        if B_label is None:
            B_label = A_label + '^T'
        if B_type is None:
            B_type = A_type
        B_shape = list(A_shape).reverse()
    else:
        if B_shape is None:
            raise ValidationError(
                'Array B is not transient, but its shape is not set')

    # Validate data
    map_ranges, A_ranges, B_ranges = validate_matrix_transpose(
        A_shape, B_shape, A_index, B_index)
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges

    # Set label
    if label is None:
        label = A_label + '^T'

    # Create state
    state = State(label=label)

    # Create datanodes
    A_node = state.add_array(
        A_label, A_shape, A_type, transient=is_A_transient)
    B_node = state.add_array(
        B_label, B_shape, B_type, transient=is_B_transient or create_B)

    # Create map/tasklet
    _, map_entry, map_exit = state.add_mapped_tasklet(
        name=label,
        map_ranges=map_ranges,
        inputs=dict(a=dace.Memlet.simple(A_node, A_inner_range)),
        outputs=dict(b=dace.Memlet.simple(B_node, B_inner_range)),
        code='b = a')

    # Add edges
    state.add_nedge(A_node, map_entry, dace.Memlet.simple(
        A_node, A_outer_range))
    state.add_nedge(map_exit, B_node, dace.Memlet.simple(
        B_node, B_outer_range))

    return state


def validate_matrix_pointwise_op(
        A_shape: Shape,
        B_shape: Shape,
        C_shape: Shape,
        reduce: bool = False,
        A_index: Index = None,
        B_index: Index = None,
        C_index: Index = None
) -> (typing.Dict[str, str], (str, str), (str, str), (str, str)):
    """ Validates a point-wise matrix operation. """

    # Validate data
    if A_index is not None:
        true_A_shape = A_shape[len(A_index):]
    else:
        true_A_shape = A_shape
    if B_index is not None:
        true_B_shape = B_shape[len(B_index):]
    else:
        true_B_shape = B_shape
    if true_A_shape != true_B_shape:
        raise ValidationError('Dimension mismatch between arrays A and B: '
                              '{}({}) != {}({})'.format(
                                  true_A_shape, A_shape, true_B_shape,
                                  B_shape))
    if reduce:
        if C_index is None or len(C_shape) != len(C_index):
            raise ValidationError(
                'Point-wise matrix operation result cannot be reduced: '
                '{}({})'.format(C_shape, C_index))
    else:
        if C_index is not None:
            true_C_shape = C_shape[len(C_index):]
        else:
            true_C_shape = C_shape
        if true_A_shape != true_B_shape:
            raise ValidationError('Dimension mismatch between arrays A and C: '
                                  '{}({}) != {}({})'.format(
                                      true_A_shape, A_shape, true_C_shape,
                                      C_shape))

    # Map ranges
    map_ranges = dict()
    for i, dim in enumerate(true_A_shape):
        map_ranges['i{}'.format(i)] = '0:{}'.format(dim)

    # Memlet ranges
    A_outer_range = ', '.join(map_ranges.values())
    A_inner_range = ', '.join(map_ranges.keys())
    if A_index is not None:
        A_index = [str(idx) for idx in A_index]
        A_outer_range = '{}, {}'.format(', '.join(A_index), A_outer_range)
        A_inner_range = '{}, {}'.format(', '.join(A_index), A_inner_range)
    B_outer_range = ', '.join(map_ranges.values())
    B_inner_range = ', '.join(map_ranges.keys())
    if B_index is not None:
        B_index = [str(idx) for idx in B_index]
        B_outer_range = '{}, {}'.format(', '.join(B_index), B_outer_range)
        B_inner_range = '{}, {}'.format(', '.join(B_index), B_inner_range)
    if reduce:
        C_index = [str(idx) for idx in C_index]
        C_outer_range = ', '.join(C_index)
        C_inner_range = ', '.join(C_index)
    else:
        C_outer_range = ', '.join(map_ranges.values())
        C_inner_range = ', '.join(map_ranges.keys())
        if C_index is not None:
            C_index = [str(idx) for idx in C_index]
            C_outer_range = '{}, {}'.format(', '.join(C_index), C_outer_range)
            C_inner_range = '{}, {}'.format(', '.join(C_index), C_inner_range)

    return (map_ranges, (A_outer_range, A_inner_range),
            (B_outer_range, B_inner_range), (C_outer_range, C_inner_range))


def matrix_pointwise_op(state: State,
                        A_src: Node,
                        A_node: DNode,
                        B_src: Node,
                        B_node: DNode,
                        C_dst: Node,
                        C_node: DNode,
                        op: str,
                        reduce: bool = False,
                        reduce_op: str = None,
                        accumulate: bool = False,
                        A_index: Index = None,
                        B_index: Index = None,
                        C_index: Index = None,
                        label: str = None):
    """ Adds a matrix point-wise operation to an existing state. """

    # Validate data
    sdfg = state.parent
    C_shape = None
    if reduce and not hasattr(C_node.desc(sdfg), 'shape'):
        C_shape = [1]
    else:
        C_shape = C_node.desc(sdfg).shape
    map_ranges, A_ranges, B_ranges, C_ranges = validate_matrix_pointwise_op(
        A_node.desc(sdfg).shape,
        B_node.desc(sdfg).shape, C_shape, reduce, A_index, B_index, C_index)
    A_outer_range, A_inner_range = A_ranges
    B_outer_range, B_inner_range = B_ranges
    C_outer_range, C_inner_range = C_ranges

    # Set label
    if label is None:
        label = state.label

    # Create map/tasklet
    if reduce:
        schedule = dace.types.ScheduleType.Sequential
    else:
        schedule = dace.types.ScheduleType.Default
    map_entry, map_exit = state.add_map(
        name=label + '_map', ndrange=map_ranges, schedule=schedule)
    map_entry.in_connectors = {'IN_1', 'IN_2'}
    map_entry.out_connectors = {'OUT_1', 'OUT_2'}
    map_exit.in_connectors = {'IN_1'}
    map_exit.out_connectors = {'OUT_1'}
    tasklet = state.add_tasklet(
        name=label + '_tasklet',
        inputs={'a', 'b'},
        outputs={'c'},
        code='c = a ' + op + ' b')

    # Add edges
    state.add_edge(A_src, None, map_entry, 'IN_1',
                   dace.Memlet.simple(A_node, A_outer_range))
    state.add_edge(B_src, None, map_entry, 'IN_2',
                   dace.Memlet.simple(B_node, B_outer_range))
    state.add_edge(map_exit, 'OUT_1', C_dst, None,
                   dace.Memlet.simple(C_node, C_outer_range))
    state.add_edge(map_entry, 'OUT_1', tasklet, 'a',
                   dace.Memlet.simple(A_node, A_inner_range))
    state.add_edge(map_entry, 'OUT_2', tasklet, 'b',
                   dace.Memlet.simple(B_node, B_inner_range))
    if reduce:
        wcr = 0
        if accumulate:
            wcr = None
        state.add_edge(
            tasklet, 'c', map_exit, 'IN_1',
            dace.Memlet.simple(
                C_node,
                C_inner_range,
                wcr_str='lambda x, y: x ' + reduce_op + ' y',
                wcr_identity=wcr,
                wcr_conflict=False))
    else:
        state.add_edge(tasklet, 'c', map_exit, 'IN_1',
                       dace.Memlet.simple(C_node, C_inner_range))


def csr2dense_cusparse(state: State, val: DNode, rowptr: DNode, colind: DNode,
                       dense: DNode):
    """ Adds a CSR->Dense data layout transformation to a state, using 
        CUSPARSE for the implementation. """
    sdfg = state.parent
    dense_array = dense.desc(sdfg)
    d_shape = dense_array.shape
    d_dtype = dense_array.dtype
    T = state.add_transient(dense.data + 'T', d_shape, d_dtype)

    tasklet = state.add_tasklet(
        name=dense.data + '_csr2dense',
        inputs={'val', 'rowptr', 'colind'},
        outputs={'dense'},
        code='''
    cusparseSetStream(sparse_handle, __dace_current_stream);
    cusparseZcsr2dense(
        sparse_handle,
        {m}, {n},
        sparse_mat_descr,
        (cuDoubleComplex*)val,
        rowptr,
        colind,
        (cuDoubleComplex*)dense,
        {m}
    );
        '''.format(m=str(d_shape[0]), n=str(d_shape[1])),
        language=dace.types.Language.CPP)
    state.add_edge(val, None, tasklet, 'val',
                   dace.Memlet.from_array(val.data, val.desc(sdfg)))
    state.add_edge(rowptr, None, tasklet, 'rowptr',
                   dace.Memlet.from_array(rowptr.data, rowptr.desc(sdfg)))
    state.add_edge(colind, None, tasklet, 'colind',
                   dace.Memlet.from_array(colind.data, colind.desc(sdfg)))
    state.add_edge(tasklet, 'dense', T, None,
                   dace.Memlet.from_array(T.data, T.desc(sdfg)))
    gpu_transform_tasklet(sdfg, state, tasklet)
    matrix_transpose(state, T, T, dense, dense, label=T.data)


def matrix_inversion_cusolver(state, arg, mat_inv, mat_index, label):
    """ Adds a matrix inverse operation to a state, using CUSOLVER
        for the implementation. """

    sdfg = state.parent
    m_shape = mat_inv.desc(sdfg).shape
    inv_range = '0 : {sz}, 0 : {sz}'.format(sz=m_shape[-1])
    if mat_index is not None:
        index = [str(idx) for idx in mat_index]
        inv_range = '{}, {}'.format(', '.join(index), inv_range)
    inv_task = state.add_tasklet(
        name=label,
        inputs={'a'},
        outputs={'b'},
        code='''
        cusolverDnSetStream(solver_handle, __dace_current_stream);
        int new_lwork = 0;
        cusolverDnZgetrf_bufferSize(
            solver_handle,
            {n}, {n},
            (cuDoubleComplex*)a,
            {n},
            &new_lwork
        );
        //cudaDeviceSynchronize();
        if (new_lwork > lwork) {{
            lwork = new_lwork;
            cudaFree(dwork);
            cudaMalloc<cuDoubleComplex>(&dwork, sizeof(cuDoubleComplex) * lwork);
        }}
        cusolverDnZgetrf(
            solver_handle,
            {n}, {n},
            (cuDoubleComplex*)a,
            {n},
            dwork, ipiv, info
        );
        //cudaDeviceSynchronize();
        cudaMemcpyAsync(b, dev_I, sizeof(cuDoubleComplex) * {n} * {n}, cudaMemcpyDeviceToDevice, __dace_current_stream);
        cusolverDnZgetrs(
            solver_handle,
            CUBLAS_OP_N,
            {n},
            {n}, /* nrhs */
            (cuDoubleComplex*)a,
            {n},
            ipiv,
            (cuDoubleComplex*)b,
            {n},
            info
        );
        //cudaDeviceSynchronize();
        '''.format(n=m_shape[-1]),
        language=dace.types.Language.CPP)
    state.add_edge(arg, None, inv_task, 'a',
                   dace.Memlet.from_array(arg.data, arg.desc(sdfg)))
    state.add_edge(inv_task, 'b', mat_inv, None,
                   dace.Memlet.simple(mat_inv, inv_range))
    gpu_transform_tasklet(sdfg, state, inv_task)
