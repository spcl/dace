import dace
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dutils, infer_types
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.standard import Transpose
import copy
from dace.autodiff.backward_pass_generator import BackwardPassGenerator
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.dtypes import DeviceType
from dace.libraries.blas.blas_helpers import to_blastype
import dace.sdfg.nodes as nodes
                  
def autooptimize_sdfgs_for_ad(bwd_generator: BackwardPassGenerator):
    """
    A pass that will be applied after getting the backward SDFG to optimize the SDFG.
    """
    forward_sdfg: SDFG = bwd_generator.sdfg
    backward_sdfg: SDFG = bwd_generator.backward_sdfg

    # Tranformations to be applied after the backward pass has been generated
    # 1- Revert MatMul back to a library node and clbals calls
    forward_gemm_to_library_node(forward_sdfg)
    backward_gemm_to_library_node(backward_sdfg)
    
    # # 2- We make all the arrays in the SDFG transient except for the gradient computations
    fwd_modified = []
    for state in forward_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if "gradient" not in node.data and not forward_sdfg.arrays[
                        node.data].transient:
                    if node.data not in fwd_modified:
                        fwd_modified.append(
                            (node.data, forward_sdfg.arrays[node.data]))
                    forward_sdfg.arrays[node.data].transient = True
    bwd_modified = []
    for state in backward_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if "gradient" not in node.data:
                    if node.data not in bwd_modified and not backward_sdfg.arrays[
                            node.data].transient:
                        bwd_modified.append(
                            (node.data, backward_sdfg.arrays[node.data]))
                    backward_sdfg.arrays[node.data].transient = True

    # 3- Call autoopt for CPU
    backward_sdfg_opt = backward_sdfg
    forward_sdfg_opt = forward_sdfg
    forward_sdfg_opt.simplify()
    forward_sdfg_opt = auto_optimize(forward_sdfg, DeviceType.CPU)
    if bwd_generator.separate_sdfgs:
        backward_sdfg.simplify()
        backward_sdfg_opt = auto_optimize(backward_sdfg, DeviceType.CPU)

    # 4- Reset the non-transient arrays to avoid modifying the signature of the function
    for data, desc in fwd_modified:
        if data not in forward_sdfg_opt.arrays and data in forward_sdfg.arrays:
            # The desciptor has been removed by auto-opt
            # We add it back again to avoid modifying the signature of the function
            forward_sdfg_opt.add_datadesc(data, desc)
        desc.transient = False
    for data, desc in bwd_modified:
        if data not in backward_sdfg_opt.arrays and data in backward_sdfg.arrays:
            # The desciptor has been removed by auto-opt
            # We add it back again to avoid modifying the signature of the function
            backward_sdfg_opt.add_datadesc(data, desc)
        desc.transient = False

    post_processing_pass(forward_sdfg_opt)
    post_processing_pass(backward_sdfg_opt)
    
    # Change the generator to use the optimized SDFG
    bwd_generator.backward_sdfg = backward_sdfg_opt
    bwd_generator.sdfg = forward_sdfg_opt

def post_processing_pass(sdfg: SDFG):
    """
    A pass that will be applied after getting the optimized backward SDFG.
    At the moment this only avoids a specific bug with boolean assignements.
    If the output is infered as a scalar, 
    the output is not initalized and if the code is not true we assign random values
    """
    for node, parent in sdfg.all_nodes_recursive():
        # This only applies to Tasklets
        if not isinstance(node, dace.nodes.Tasklet):
            continue
        
        # Boolean tasklets specifically
        if not (parent.in_degree(node) == 1 and parent.out_degree(node) == 1):
            continue
        
        in_edge = parent.in_edges(node)[0]
        out_edge = parent.out_edges(node)[0]
        
        if in_edge.dst_conn is None or node.in_connectors[in_edge.dst_conn] != dace.dtypes.bool:
            continue    
        
        if out_edge.src_conn is None or node.out_connectors[out_edge.src_conn] != dace.dtypes.float64:
            continue
        
        assert False
        node.out_connectors[out_edge.src_conn] = dace.dtypes.pointer(dace.dtypes.float64)
    

def scale_matrix_to_cblas_call(sdfg: SDFG):
    """
    Pattern match matrix scaling like B = alpha * A and convert it to a cblas call.
    """

    # First pattern match the scaling, this should be a map nest with two inputs and one output

    for node, parent in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.MapEntry) or len(
                node.in_connectors) != 2 or len(
                    parent.exit_node(node).out_connectors) != 1:
            continue

        # The inputs should be the matrix A and the scalar alpha
        in_edges = parent.in_edges(node)
        input_1, input_2 = in_edges[0].src, in_edges[1].src
        if not isinstance(input_1, dace.nodes.AccessNode) or not isinstance(
                input_2, dace.nodes.AccessNode):
            continue
        scalar = None
        matrix = None
        if sdfg.arrays[input_1.data].shape == (1, ):
            scalar = input_1
            scalar_memlet = in_edges[0].data
            matrix = input_2
            matrix_memlet = in_edges[1].data
        if sdfg.arrays[input_2.data].shape == (1, ):
            scalar = input_2
            scalar_memlet = in_edges[1].data
            matrix = input_1
            matrix_memlet = in_edges[0].data

        if not scalar or len(sdfg.arrays[matrix.data].shape) != 2:
            continue

        # The output should be the matrix B which has the same shape as the input matrix
        map_exit = parent.exit_node(node)
        out_edge = parent.out_edges(map_exit)[0]
        output = out_edge.dst
        if not isinstance(output, dace.nodes.AccessNode) or sdfg.arrays[
                output.data].shape != sdfg.arrays[matrix.data].shape:
            continue

        # The ranges of the map should iterate over all of the elements of the matrix
        map_ranges = node.map.range
        if map_ranges != matrix_memlet.subset:
            continue

        # Additionally, We should check the tasklet to make sure that it a multiplication operation
        tasklet = parent.out_edges(node)[0].dst
        if not isinstance(tasklet, dace.nodes.Tasklet):
            continue

        if len(tasklet.in_connectors) != 2 or len(tasklet.out_connectors) != 1:
            continue

        output_conn = list(tasklet.out_connectors.keys())[0]
        input_1_conn, input_2_conn = list(
            tasklet.in_connectors.keys())[0], list(
                tasklet.in_connectors.keys())[1]
        exepcted_multiplication1 = f"{output_conn} = ({input_1_conn} * {input_2_conn})"
        exepcted_multiplication2 = f"{output_conn} = ({input_2_conn} * {input_1_conn})"
        if exepcted_multiplication1 not in tasklet.code.as_string and exepcted_multiplication2 not in tasklet.code.as_string:
            continue

        # Found the pattern, now we need to convert it to a cblas call
        # Add import of cblas header
        dtype = sdfg.arrays[output.data].dtype
        total_size = f"{sdfg.arrays[output.data].shape[0]} * {sdfg.arrays[output.data].shape[1]}"
        func = to_blastype(dtype.type).lower() + 'scal'
        clads_code = f"cblas_{func}({total_size}, _b, _a, 1);"
        # First we need to create the tasklet which will contain the code
        taskelt = dace.nodes.Tasklet(
            "cblas_scal",
            inputs={"_a", "_b"},
            outputs={"_c"},
            code=clads_code,
            language=dace.Language.CPP,
        )
        taskelt.environments = [
            'dace.libraries.blas.environments.intel_mkl.IntelMKL'
        ]
        parent.add_node(taskelt)

        # Replicate the output becuase cblas is performing the operation in place
        replicated_output = parent.add_access(output.data)
        copy_memlet = sdfg.make_array_memlet(matrix.data)
        copy_memlet.wcr = "(lambda x, y: (x + y))"
        parent.add_edge(matrix, None, replicated_output, None, copy_memlet)

        # Now we need to connect the tasklet to the map
        matrix_memlet.data = replicated_output.data
        parent.add_edge(replicated_output, None, taskelt, "_a", matrix_memlet)
        parent.add_edge(scalar, None, taskelt, "_b", scalar_memlet)

        # The output memlet is empty becuase the operation is inplace
        parent.add_edge(taskelt, "_c", output, None,
                        dace.Memlet(data=output.data))

        # Remove the old nodes
        parent.remove_node(node)
        parent.remove_node(map_exit)
        parent.remove_node(tasklet)


def can_be_applied_forward_gemm_to_library_node(state: SDFGState,
                                                nsdfg: dace.nodes.NestedSDFG):
    """
    Check if the GEMM node can be converted to a library node.
    """
    if not isinstance(nsdfg, dace.nodes.NestedSDFG):
        return None
    nsdfg_in_edges = state.in_edges(nsdfg)
    if len(nsdfg_in_edges) != 2:
        # There should be two edges to connectors
        # This is to take into considiration the synchronization edges with empty memlets
        with_conn = 0
        for e in nsdfg_in_edges:
            if e.dst_conn is not None:
                with_conn += 1
        if with_conn != 2:
            return None
    nsdfg_out_edges = state.out_edges(nsdfg)
    if len(nsdfg_out_edges) != 1:
        return None
    # TODO: patterm matching
    for state in nsdfg.sdfg.states():
        if ("_MatMult_gemm_initstate"
                in state.label) and "reversed" not in state.label:
            return state.label
        if ("_MatMult_gemv" in state.label) and "reversed" not in state.label:
            return state.label
        if ("_MatMult_dot" in state.label) and "reversed" not in state.label:
            return state.label
        if ("_Dot_" in state.label) and "reversed" not in state.label:
            return state.label
    return None


def forward_gemm_to_library_node(sdfg: SDFG):
    """
    A pass that will be applied after getting the backward SDFG to convert the GEMM node to a library node.
    The GEMM node will be expanded at the begenning of the backward pass generator.
    Without additional transformations, this will result in bad performance so we revert back to the library node for now.
    """
    # Iterate through all sdfg states
    for state in sdfg.all_states():
        # Iterate through all nodes in the state
        for node in state.nodes():
            matmul = can_be_applied_forward_gemm_to_library_node(state, node)
            # Check if the node is a GEMM node
            if matmul is not None:
                if "_MatMult_gemm_initstate" in matmul:
                    # Create the new library node
                    libnode = MatMul('_MatMult_1_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    for e in in_edges:
                        state.add_edge(e.src, e.src_conn, libnode, e.dst_conn,
                                       e.data)
                        state.remove_edge(e)

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    assert len(out_edges) == 1
                    for e in out_edges:
                        state.add_edge(libnode, e.src_conn, e.dst, e.dst_conn,
                                       e.data)
                        state.remove_edge(e)

                if "_MatMult_dot" in matmul:
                    # Create the new library node
                    libnode = MatMul('_MatMult_2_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    assert len(in_edges) == 2
                    for e in in_edges:
                        if e.dst_conn == "_x":
                            state.add_edge(e.src, e.src_conn, libnode, "_a",
                                           e.data)
                        elif e.dst_conn == "_y":
                            state.add_edge(e.src, e.src_conn, libnode, "_b",
                                           e.data)
                        else:
                            raise ValueError("Inputs don't match for DOT")
                        state.remove_edge(e)

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    assert len(out_edges) == 1
                    for e in out_edges:
                        state.add_edge(libnode, "_c", e.dst, e.dst_conn,
                                       e.data)
                        state.remove_edge(e)

                if "_MatMult_gemv" in matmul:
                    # Create the new library node
                    libnode = MatMul('_MatMult_3_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    assert len(in_edges) == 2
                    for e in in_edges:
                        if e.dst_conn == "_x":
                            # Get the shape of the input array
                            assert e.src.data == e.data.data
                            shape = copy.deepcopy(e.data.subset)
                            shape.squeeze()
                            assert len(shape) == 1
                            x_input = shape
                        if e.dst_conn == "_A":
                            # Get the shape of the input array
                            assert e.src.data == e.data.data
                            shape = copy.deepcopy(e.data.subset)
                            shape.squeeze()
                            assert len(shape) == 2
                            A_input = shape

                    if x_input[0] == A_input[0]:
                        new_A_conn = "_b"
                        new_x_conn = "_a"
                    elif x_input[0] == A_input[1]:
                        new_A_conn = "_a"
                        new_x_conn = "_b"
                    else:
                        raise ValueError("Inputs don't match for GEMV")

                    for e in in_edges:
                        if e.dst_conn == "_A":
                            # Get the shape of the input array
                            assert e.src.data == e.data.data
                            shape = copy.deepcopy(e.data.subset)
                            shape.squeeze()
                            assert len(shape) == 2

                            state.add_edge(e.src, e.src_conn, libnode,
                                           new_A_conn, e.data)
                            state.remove_edge(e)
                        elif e.dst_conn == "_x":
                            state.add_edge(e.src, e.src_conn, libnode,
                                           new_x_conn, e.data)
                            state.remove_edge(e)
                        else:
                            assert False

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    assert len(out_edges) == 1
                    for e in out_edges:
                        state.add_edge(libnode, "_c", e.dst, e.dst_conn,
                                       e.data)
                        state.remove_edge(e)
                if "_Dot_" in matmul:
                    # Create the new library node
                    libnode = Dot('_Dot_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    for e in in_edges:
                        state.add_edge(e.src, e.src_conn, libnode, e.dst_conn,
                                       e.data)
                        state.remove_edge(e)

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    assert len(out_edges) == 1
                    for e in out_edges:
                        state.add_edge(libnode, e.src_conn, e.dst, e.dst_conn,
                                       e.data)
                        state.remove_edge(e)
                # Remove the nested SDFG node
                state.remove_node(node)


def can_be_applied_backward_gemm_to_library_node(state: SDFGState,
                                                 nsdfg: dace.nodes.NestedSDFG):
    """
    Check if the GEMM node can be converted to a library node.
    """
    if not isinstance(nsdfg, dace.nodes.NestedSDFG):
        return None
    nsdfg_in_edges = state.in_edges(nsdfg)
    nsdfg_out_edges = state.out_edges(nsdfg)
    if len(nsdfg_in_edges) == 2 and len(nsdfg_out_edges) != 1:
        return None
    elif len(nsdfg_in_edges) == 3 and len(nsdfg_out_edges) != 2:
        return None

    for state in nsdfg.sdfg.states():
        if "_MatMult_gemm_initstate" in state.label and "reversed" in state.label:
            return state.label
        if "_MatMult_gemv" in state.label and "reversed" in state.label:
            # TODO: need to fix squeezing for this to work
            return state.label
    return None


def backward_gemm_to_library_node(sdfg: SDFG):
    """
    A pass that will be applied after getting the backward SDFG to convert the GEMM node to a library node.
    The GEMM node will be expanded at the begenning of the backward pass generator.
    Without additional transformations, this will result in bad performance so we revert back to the library node for now.
    """
    # Iterate through all sdfg states
    for state in sdfg.all_states():
        # Iterate through all nodes in the state
        for node in state.nodes():
            # Check if the node is a GEMM node
            matmul = can_be_applied_backward_gemm_to_library_node(state, node)
            if matmul is not None:
                if "_MatMult_gemm_initstate" in matmul:
                    # Create the new library node
                    matmul_libnode = MatMul('_MatMult_4_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    # Create another matmul node
                    matmul_libnode2 = MatMul('_MatMult_5_')
                    for e in in_edges:
                        if e.dst_conn == "_a" or e.dst_conn == "_b":
                            # We need to transpose the input matrix
                            transpose_libnode = Transpose(
                                "_Transpose_",
                                dtype=sdfg.arrays[e.src.data].dtype)
                            state.add_edge(e.src, e.src_conn,
                                           transpose_libnode, "_inp", e.data)
                            # Transpose the memlet data
                            assert e.data.subset is not None
                            new_memlet = copy.deepcopy(e.data)
                            if e.data.subset.dims() == 2:
                                new_memlet.subset = dace.subsets.Range([
                                    new_memlet.subset[1], new_memlet.subset[0]
                                ])
                                if e.dst_conn == "_a":

                                    # Create a new array to contain the transposed matrix
                                    name, _ = sdfg.add_array(
                                        "_transposed_1_" + e.src.data, [
                                            sdfg.arrays[e.src.data].shape[1],
                                            sdfg.arrays[e.src.data].shape[0]
                                        ],
                                        sdfg.arrays[e.src.data].dtype,
                                        transient=True)
                                    transposed_node = state.add_access(name)
                                    state.add_edge(
                                        transpose_libnode, "_out",
                                        transposed_node, None,
                                        sdfg.make_array_memlet(name))
                                    new_memlet.data = name
                                    state.add_edge(transposed_node, None,
                                                   matmul_libnode, "_a",
                                                   new_memlet)
                                else:
                                    # Create a new array to contain the transposed matrix
                                    name, _ = sdfg.add_array(
                                        "_transposed_2_" + e.src.data, [
                                            sdfg.arrays[e.src.data].shape[1],
                                            sdfg.arrays[e.src.data].shape[0]
                                        ],
                                        sdfg.arrays[e.src.data].dtype,
                                        transient=True)
                                    transposed_node = state.add_access(name)
                                    new_memlet.data = name
                                    state.add_edge(
                                        transpose_libnode, "_out",
                                        transposed_node, None,
                                        sdfg.make_array_memlet(name))
                                    state.add_edge(transposed_node, None,
                                                   matmul_libnode2, "_b",
                                                   new_memlet)
                            else:
                                assert e.data.subset.dims() > 2
                                new_memlet.subset = dace.subsets.Range(
                                    new_memlet.subset[:-2] + [
                                        new_memlet.subset[-1],
                                        new_memlet.subset[-2]
                                    ])
                                if e.dst_conn == "_a":

                                    # Create a new array to contain the transposed matrix
                                    name, _ = sdfg.add_array(
                                        "_transposed_3_" + e.src.data,
                                        list(
                                            sdfg.arrays[e.src.data].shape[:-2])
                                        + [
                                            sdfg.arrays[e.src.data].shape[-1],
                                            sdfg.arrays[e.src.data].shape[-2]
                                        ],
                                        sdfg.arrays[e.src.data].dtype,
                                        transient=True)
                                    transposed_node = state.add_access(name)
                                    transposition_memlet = copy.deepcopy(
                                        new_memlet)
                                    transposition_memlet.data = name
                                    state.add_edge(transpose_libnode, "_out",
                                                   transposed_node, None,
                                                   transposition_memlet)
                                    new_memlet.data = name
                                    if new_memlet.subset is not None and new_memlet.subset.dims(
                                    ) == 2:
                                        state.add_edge(transposed_node, None,
                                                       matmul_libnode, "_a",
                                                       new_memlet)
                                    elif new_memlet.subset is not None and new_memlet.subset.dims(
                                    ) > 2:
                                        original_array = e.src.data
                                        original_shape = sdfg.arrays[
                                            original_array].shape
                                        oriringal_dtype = sdfg.arrays[
                                            original_array].dtype
                                        # We need to create a view with the last two dimensions to remove the loop values
                                        view_name, desc = sdfg.add_view(
                                            name + "_view",
                                            original_shape[-2:],
                                            oriringal_dtype)
                                        view_node = state.add_access(view_name)
                                        view_to_libnode_memlet = copy.deepcopy(
                                            new_memlet)
                                        new_memlet.other_subset = dace.subsets.Range(
                                            new_memlet.subset[-2:])
                                        state.add_edge(transposed_node, None,
                                                       view_node, "views",
                                                       new_memlet)
                                        view_to_libnode_memlet.data = view_name
                                        view_to_libnode_memlet.subset = dace.subsets.Range(
                                            view_to_libnode_memlet.subset[-2:])
                                        state.add_edge(view_node, None,
                                                       matmul_libnode, "_a",
                                                       view_to_libnode_memlet)

                                    else:
                                        raise ValueError(
                                            "Subset is not correct")
                                else:
                                    # Create a new array to contain the transposed matrix
                                    name, _ = sdfg.add_array(
                                        "_transposed_4_" + e.src.data,
                                        list(
                                            sdfg.arrays[e.src.data].shape[:-2])
                                        + [
                                            sdfg.arrays[e.src.data].shape[-1],
                                            sdfg.arrays[e.src.data].shape[-2]
                                        ],
                                        sdfg.arrays[e.src.data].dtype,
                                        transient=True)
                                    transposed_node = state.add_access(name)
                                    new_memlet.data = name
                                    transposition_memlet = copy.deepcopy(
                                        new_memlet)
                                    state.add_edge(transpose_libnode, "_out",
                                                   transposed_node, None,
                                                   transposition_memlet)
                                    state.add_edge(transposed_node, None,
                                                   matmul_libnode2, "_b",
                                                   new_memlet)
                        else:
                            if e.data.is_empty():
                                state.add_edge(e.src, e.src_conn,
                                               matmul_libnode, None,
                                               copy.deepcopy(e.data))
                                state.add_edge(e.src, e.src_conn,
                                               matmul_libnode2, None,
                                               copy.deepcopy(e.data))
                            else:
                                state.add_edge(e.src, e.src_conn,
                                               matmul_libnode, "_b",
                                               copy.deepcopy(e.data))
                                state.add_edge(e.src, e.src_conn,
                                               matmul_libnode2, "_a",
                                               copy.deepcopy(e.data))

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    for e in out_edges:
                        if e.src_conn == "gradient__a":
                            state.add_edge(matmul_libnode2, "_c", e.dst,
                                           e.dst_conn, copy.deepcopy(e.data))
                        else:
                            assert e.src_conn == "gradient__b"
                            state.add_edge(matmul_libnode, "_c", e.dst,
                                           e.dst_conn, copy.deepcopy(e.data))
                        state.remove_edge(e)
                if "_MatMult_gemv" in matmul:
                    # Create the new library node
                    matmul_libnode = MatMul('_MatMult_6_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    # Create another matmul node
                    matmul_libnode2 = MatMul('_MatMult_7_')

                    # We need to decide which way we will padd the views
                    # Get the output shape
                    out_edges = state.out_edges(node)
                    for e in out_edges:
                        if e.src_conn == "gradient__A":
                            assert e.dst.data == e.data.data
                            shape = sdfg.arrays[e.dst.data].shape
                            # shape.squeeze()
                            assert len(shape) == 2
                            matrix_output_shape = shape

                    in_edges = state.in_edges(node)
                    A_input_shape = None
                    x_input_shape = None
                    for e in in_edges:
                        shape = sdfg.arrays[e.src.data].shape
                        if e.dst_conn == "gradient__y":
                            assert len(shape) == 1
                            gradient_input_shape = shape
                            gradient_input_e = e
                        elif e.dst_conn == "_x":
                            assert len(shape) == 1
                            x_input_shape = shape
                            x_input_e = e
                        elif e.dst_conn == "_A":
                            if len(shape) > 2:
                                # In case this is stored data
                                shape = shape[-2:]
                            A_input_shape = shape
                            A_input_e = e
                    if A_input_shape is None and x_input_shape is None:
                        if x_input_shape[0] == A_input_shape[1]:
                            new_A_conn = "_b"
                            new_x_conn = "_a"
                        elif x_input_shape[0] == A_input_shape[0]:
                            new_A_conn = "_a"
                            new_x_conn = "_b"
                        else:
                            raise ValueError("Inputs don't match for GEMV")
                    else:
                        new_A_conn = "_a"
                        new_x_conn = "_b"

                    # In case the vector shape matches both dims of the matrix we rely on the label to decide if this is
                    # B= y @ A or B = A @ y
                    if gradient_input_shape[0] == matrix_output_shape[
                            0] and "gemvt" not in matmul:
                        gy_shape = list(sdfg.arrays[
                            gradient_input_e.src.data].shape) + [1]
                        gy_subset = dace.subsets.Range(
                            [gradient_input_e.data.subset[0], (0, 0, 1)])
                        gy_other_subset = dace.subsets.Range(
                            [gradient_input_e.data.subset[0], (0, 0, 1)])
                        gy_conn = "_a"
                        x_shape = [1] + list(
                            sdfg.arrays[x_input_e.src.data].shape)
                        x_subset = dace.subsets.Range([
                            (0, 0, 1), x_input_e.data.subset[0]
                        ])
                        x_other_subset = dace.subsets.Range([
                            (0, 0, 1), x_input_e.data.subset[0]
                        ])
                        x_conn = "_b"
                    else:
                        assert x_input_shape[0] == matrix_output_shape[0]
                        gy_shape = [1] + list(
                            sdfg.arrays[gradient_input_e.src.data].shape)
                        gy_subset = dace.subsets.Range([
                            (0, 0, 1), gradient_input_e.data.subset[0]
                        ])
                        gy_other_subset = dace.subsets.Range([
                            (0, 0, 1), gradient_input_e.data.subset[0]
                        ])
                        gy_conn = "_b"
                        x_shape = list(
                            sdfg.arrays[x_input_e.src.data].shape) + [1]
                        x_subset = dace.subsets.Range(
                            [x_input_e.data.subset[0], (0, 0, 1)])
                        x_other_subset = dace.subsets.Range(
                            [x_input_e.data.subset[0], (0, 0, 1)])
                        x_conn = "_a"

                    # If this is a tarnposed gemvt we need to switch the two connectors only
                    for e in in_edges:
                        if e.dst_conn == "_A":
                            state.add_edge(e.src, e.src_conn, matmul_libnode,
                                           new_A_conn, copy.deepcopy(e.data))
                        elif e.dst_conn == "_x":
                            # Create a view of the source node which pads the shape with 1 at the benegning
                            an = e.src
                            assert len(sdfg.arrays[an.data].shape) == 1
                            view_name, desc = sdfg.add_view(
                                "_view_" + an.data,
                                x_shape,
                                sdfg.arrays[an.data].dtype,
                                find_new_name=True)
                            
                            # We need this in case we are expanding a view vector into a matrix
                            # If this is a view of a view
                            if isinstance(sdfg.arrays[an.data], (dace.data.View, dace.data.ArrayView)):
                                desc.strides = (sdfg.arrays[an.data].strides[0], desc.strides[1])
                            view = state.add_access(view_name)
                            view_memlet = copy.deepcopy(e.data)
                            view_memlet.data = view_name
                            # Change the memlet subset to add 1 at the benegning
                            view_memlet.subset = x_subset
                            state.add_edge(view, None, matmul_libnode2, x_conn,
                                           view_memlet)
                            an_to_view_memlet = copy.deepcopy(e.data)
                            an_to_view_memlet.other_subset = x_other_subset
                            state.add_edge(e.src, None, view, "views",
                                           an_to_view_memlet)
                        else:
                            if e.data.is_empty():
                                # Add empty synchronization edges
                                state.add_edge(e.src, e.src_conn,
                                               matmul_libnode, None,
                                               copy.deepcopy(e.data))
                                state.add_edge(e.src, e.src_conn,
                                               matmul_libnode2, None,
                                               copy.deepcopy(e.data))
                            else:
                                # Create a view of the source node which pads the shape with 1 at the benegning
                                an = e.src
                                assert len(sdfg.arrays[an.data].shape) == 1
                                # if "_A" in e.dst.in_connectors:
                                # TODO
                                # Only add the connection if this lib node is necessary
                                state.add_edge(e.src, None,
                                               matmul_libnode, new_x_conn,
                                               copy.deepcopy(e.data))

                                # Create the second view
                                view_name_2, desc = sdfg.add_view(
                                    "_view_" + an.data,
                                    gy_shape,
                                    sdfg.arrays[an.data].dtype,
                                    find_new_name=True)
                                view_2 = state.add_access(view_name_2)
                                
                                # We need this in case we are expanding a view vector into a matrix
                                # If this is a view of a view
                                if isinstance(sdfg.arrays[an.data], (dace.data.View, dace.data.ArrayView)):
                                    desc.strides = (sdfg.arrays[an.data].strides[0], desc.strides[1])
                                view_memlet_2 = copy.deepcopy(e.data)
                                view_memlet_2.data = view_name_2
                                # Change the memlet subset to add 1 at the benegning
                                view_memlet_2.subset = gy_subset
                                state.add_edge(view_2, None, matmul_libnode2,
                                               gy_conn, view_memlet_2)

                                # Connect the node to the view to preserve the order of the program
                                an_to_view_memlet_2 = copy.deepcopy(e.data)
                                an_to_view_memlet_2.other_subset = gy_other_subset
                                state.add_edge(e.src, None, view_2, "views",
                                               an_to_view_memlet_2)

                        state.remove_edge(e)
                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    for e in out_edges:
                        if e.src_conn == "gradient__x":
                            state.add_edge(matmul_libnode, "_c", e.dst,
                                           e.dst_conn, copy.deepcopy(e.data))
                        else:
                            assert e.src_conn == "gradient__A"
                            state.add_edge(matmul_libnode2, "_c", e.dst,
                                           e.dst_conn, copy.deepcopy(e.data))
                        state.remove_edge(e)
                # Remove the nested SDFG node
                state.remove_node(node)
