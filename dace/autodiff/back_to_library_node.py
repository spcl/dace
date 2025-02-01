import dace
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dutils, infer_types
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.standard import Transpose
import copy


def can_be_applied_forward_gemm_to_library_node(state: SDFGState, nsdfg: dace.nodes.NestedSDFG):
    """
    Check if the GEMM node can be converted to a library node.
    """
    if not isinstance(nsdfg, dace.nodes.NestedSDFG):
        return None
    nsdfg_in_edges = state.in_edges(nsdfg)
    if len(nsdfg_in_edges) != 2:
        return None
    nsdfg_out_edges = state.out_edges(nsdfg)
    if len(nsdfg_out_edges) != 1:
        return None
    # TODO: patterm matching
    for state in nsdfg.sdfg.states():
        if ("_MatMult_gemm_initstate" in state.label
                or "_MatMult_gemv" in state.label) and "reversed" not in state.label:
            return state.label
    return None


def forward_gemm_to_library_node(sdfg: SDFG):
    """
    A pass that will be applied after getting the backward SDFG to convert the GEMM node to a library node.
    The GEMM node will be expanded at the begenning of the backward pass generator.
    Without additional transformations, this will result in bad performance so we revert back to the library node for now.
    """
    # Iterate through all sdfg states
    for state in sdfg.nodes():
        # Iterate through all nodes in the state
        for node in state.nodes():
            matmul = can_be_applied_forward_gemm_to_library_node(state, node)
            # Check if the node is a GEMM node
            if matmul is not None:
                if "_MatMult_gemm_initstate" in matmul:
                    # Create the new library node
                    libnode = MatMul('_MatMult_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    assert len(in_edges) == 2
                    for e in in_edges:
                        state.add_edge(e.src, e.src_conn, libnode, e.dst_conn, e.data)
                        state.remove_edge(e)

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    assert len(out_edges) == 1
                    for e in out_edges:
                        state.add_edge(libnode, e.src_conn, e.dst, e.dst_conn, e.data)
                        state.remove_edge(e)

                if "_MatMult_gemv" in matmul:
                    # Create the new library node
                    libnode = MatMul('_MatMult_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    assert len(in_edges) == 2
                    for e in in_edges:
                        if e.dst_conn == "_x":
                            # Get the shape of the input array
                            shape = sdfg.arrays[e.src.data].shape
                            assert len(shape) == 1
                            x_input = shape
                        if e.dst_conn == "_A":
                            # Get the shape of the input array
                            shape = sdfg.arrays[e.src.data].shape
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
                            shape = sdfg.arrays[e.src.data].shape
                            assert len(shape) == 2

                            state.add_edge(e.src, e.src_conn, libnode, new_A_conn, e.data)
                            state.remove_edge(e)
                        elif e.dst_conn == "_x":
                            state.add_edge(e.src, e.src_conn, libnode, new_x_conn, e.data)
                            state.remove_edge(e)
                        else:
                            assert False

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    assert len(out_edges) == 1
                    for e in out_edges:
                        state.add_edge(libnode, "_c", e.dst, e.dst_conn, e.data)
                        state.remove_edge(e)
                # Remove the nested SDFG node
                state.remove_node(node)


def can_be_applied_backward_gemm_to_library_node(state: SDFGState, nsdfg: dace.nodes.NestedSDFG):
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
        if ("_MatMult_gemm_initstate" in state.label or "_MatMult_gemv" in state.label) and "reversed" in state.label:
            return state.label
    return None


def backward_gemm_to_library_node(sdfg: SDFG):
    """
    A pass that will be applied after getting the backward SDFG to convert the GEMM node to a library node.
    The GEMM node will be expanded at the begenning of the backward pass generator.
    Without additional transformations, this will result in bad performance so we revert back to the library node for now.
    """
    # Iterate through all sdfg states
    for state in sdfg.nodes():
        # Iterate through all nodes in the state
        for node in state.nodes():
            # Check if the node is a GEMM node
            matmul = can_be_applied_backward_gemm_to_library_node(state, node)
            if matmul is not None:
                if "_MatMult_gemm_initstate" in matmul:
                    # Create the new library node
                    matmul_libnode = MatMul('_MatMult_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    # Create another matmul node
                    matmul_libnode2 = MatMul('_MatMult_')
                    for e in in_edges:
                        if e.dst_conn == "_a" or e.dst_conn == "_b":
                            # We need to transpose the input matrix
                            transpose_libnode = Transpose("_Transpose_", dtype=sdfg.arrays[e.src.data].dtype)
                            state.add_edge(e.src, e.src_conn, transpose_libnode, "_inp", e.data)
                            # Transpose the memlet data
                            assert e.data.subset is not None and e.data.subset.dims() == 2
                            new_memlet = copy.deepcopy(e.data)
                            new_memlet.subset = dace.subsets.Range([new_memlet.subset[1], new_memlet.subset[0]])
                            if e.dst_conn == "_a":

                                # Create a new array to contain the transposed matrix
                                name, _ = sdfg.add_array(
                                    "_transposed_" + e.src.data,
                                    [sdfg.arrays[e.src.data].shape[1], sdfg.arrays[e.src.data].shape[0]],
                                    sdfg.arrays[e.src.data].dtype,
                                    transient=True)
                                transposed_node = state.add_access(name)
                                state.add_edge(transpose_libnode, "_out", transposed_node, None,
                                               sdfg.make_array_memlet(name))
                                new_memlet.data = name
                                state.add_edge(transposed_node, None, matmul_libnode, "_a", new_memlet)
                            else:
                                # Create a new array to contain the transposed matrix
                                name, _ = sdfg.add_array(
                                    "_transposed_" + e.src.data,
                                    [sdfg.arrays[e.src.data].shape[1], sdfg.arrays[e.src.data].shape[0]],
                                    sdfg.arrays[e.src.data].dtype,
                                    transient=True)
                                transposed_node = state.add_access(name)
                                new_memlet.data = name
                                state.add_edge(transpose_libnode, "_out", transposed_node, None,
                                               sdfg.make_array_memlet(name))
                                state.add_edge(transposed_node, None, matmul_libnode2, "_b", new_memlet)
                        else:
                            state.add_edge(e.src, e.src_conn, matmul_libnode, "_b", copy.deepcopy(e.data))
                            state.add_edge(e.src, e.src_conn, matmul_libnode2, "_a", copy.deepcopy(e.data))
                            # put the transposed matrix to the matmul node

                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    for e in out_edges:
                        if e.src_conn == "gradient__a":
                            state.add_edge(matmul_libnode2, "_c", e.dst, e.dst_conn, copy.deepcopy(e.data))
                        else:
                            assert e.src_conn == "gradient__b"
                            state.add_edge(matmul_libnode, "_c", e.dst, e.dst_conn, copy.deepcopy(e.data))
                        state.remove_edge(e)
                if "_MatMult_gemv" in matmul:
                    # Create the new library node
                    matmul_libnode = MatMul('_MatMult_')

                    # Get the two inputs to the nested SDFG
                    in_edges = state.in_edges(node)
                    # Create another matmul node
                    matmul_libnode2 = MatMul('_MatMult_')

                    # We need to decide which way we will padd the views
                    # Get the output shape
                    out_edges = state.out_edges(node)
                    for e in out_edges:
                        if e.src_conn == "gradient__A":
                            shape = sdfg.arrays[e.data.data].shape
                            assert len(shape) == 2
                            matrix_output_shape = shape

                    in_edges = state.in_edges(node)
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
                            assert len(shape) == 2
                            A_input_shape = shape
                            A_input_e = e

                    if x_input_shape[0] == A_input_shape[1]:
                        new_A_conn = "_b"
                        new_x_conn = "_a"
                    elif x_input_shape[0] == A_input_shape[0]:
                        new_A_conn = "_a"
                        new_x_conn = "_b"
                    else:
                        raise ValueError("Inputs don't match for GEMV")

                    if gradient_input_shape[0] == matrix_output_shape[0]:
                        gy_shape = list(sdfg.arrays[gradient_input_e.src.data].shape) + [1]
                        gy_subset = dace.subsets.Range([gradient_input_e.data.subset[0], (0, 0, 1)])
                        gy_other_subset = dace.subsets.Range([gradient_input_e.data.subset[0], (0, 0, 1)])
                        gy_conn = "_a"
                        x_shape = [1] + list(sdfg.arrays[x_input_e.src.data].shape)
                        x_subset = dace.subsets.Range([(0, 0, 1), x_input_e.data.subset[0]])
                        x_other_subset = dace.subsets.Range([(0, 0, 1), x_input_e.data.subset[0]])
                        x_conn = "_b"
                    else:
                        assert x_input_shape[0] == matrix_output_shape[0]
                        gy_shape = [1] + list(sdfg.arrays[gradient_input_e.src.data].shape)
                        gy_subset = dace.subsets.Range([(0, 0, 1), gradient_input_e.data.subset[0]])
                        gy_other_subset = dace.subsets.Range([(0, 0, 1), gradient_input_e.data.subset[0]])
                        gy_conn = "_b"
                        x_shape = list(sdfg.arrays[x_input_e.src.data].shape) + [1]
                        x_subset = dace.subsets.Range([x_input_e.data.subset[0], (0, 0, 1)])
                        x_other_subset = dace.subsets.Range([x_input_e.data.subset[0], (0, 0, 1)])
                        x_conn = "_a"

                    for e in in_edges:
                        if e.dst_conn == "_A":
                            state.add_edge(e.src, e.src_conn, matmul_libnode, new_A_conn, copy.deepcopy(e.data))
                        elif e.dst_conn == "_x":
                            # Create a view of the source node which pads the shape with 1 at the benegning
                            an = e.src
                            assert len(sdfg.arrays[an.data].shape) == 1
                            view_name, desc = sdfg.add_view("_view_" + an.data,
                                                            x_shape,
                                                            sdfg.arrays[an.data].dtype,
                                                            find_new_name=True)
                            view = state.add_access(view_name)
                            view_memlet = copy.deepcopy(e.data)
                            view_memlet.data = view_name
                            # Change the memlet subset to add 1 at the benegning
                            view_memlet.subset = x_subset
                            state.add_edge(view, None, matmul_libnode2, x_conn, view_memlet)
                            an_to_view_memlet = copy.deepcopy(e.data)
                            an_to_view_memlet.other_subset = x_other_subset
                            state.add_edge(e.src, None, view, "views", an_to_view_memlet)
                        else:
                            # Create a view of the source node which pads the shape with 1 at the benegning
                            an = e.src
                            assert len(sdfg.arrays[an.data].shape) == 1
                            state.add_edge(e.src, None, matmul_libnode, new_x_conn, copy.deepcopy(e.data))

                            # Create the second view
                            view_name_2, desc = sdfg.add_view("_view_" + an.data,
                                                              gy_shape,
                                                              sdfg.arrays[an.data].dtype,
                                                              find_new_name=True)
                            view_2 = state.add_access(view_name_2)
                            view_memlet_2 = copy.deepcopy(e.data)
                            view_memlet_2.data = view_name_2
                            # Change the memlet subset to add 1 at the benegning
                            view_memlet_2.subset = gy_subset
                            state.add_edge(view_2, None, matmul_libnode2, gy_conn, view_memlet_2)

                            # Connect the node to the view to preserve the order of the program
                            an_to_view_memlet_2 = copy.deepcopy(e.data)
                            an_to_view_memlet_2.other_subset = gy_other_subset
                            state.add_edge(e.src, None, view_2, "views", an_to_view_memlet_2)

                        state.remove_edge(e)
                    # Get the output of the nested SDFG
                    out_edges = state.out_edges(node)
                    for e in out_edges:
                        if e.src_conn == "gradient__x":
                            state.add_edge(matmul_libnode, "_c", e.dst, e.dst_conn, copy.deepcopy(e.data))
                        else:
                            assert e.src_conn == "gradient__A"
                            state.add_edge(matmul_libnode2, "_c", e.dst, e.dst_conn, copy.deepcopy(e.data))
                        state.remove_edge(e)
                # Remove the nested SDFG node
                state.remove_node(node)
