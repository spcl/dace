import dace
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dutils, infer_types
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.standard import Transpose
import copy
import networkx as nx
from dace.autodiff.backward_pass_generator import BackwardPassGenerator
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.dtypes import DeviceType
from dace.libraries.blas.blas_helpers import to_blastype
import dace.sdfg.nodes as nodes
from dace.sdfg.utils import inline_control_flow_regions
from dace.sdfg.state import LoopRegion
from dace.sdfg import utils as sdutil
from dace.transformation.interstate import StateFusion
from dace import data as dt, dtypes, registry, sdfg, subsets


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
    cavity_flow_opt(backward_sdfg)
    fuse_states_cav(backward_sdfg)

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

    if "go_fast" in forward_sdfg.name or "seidel" in forward_sdfg.name:
        print("Preprocessing the SDFG")
        # manually remove the forward pass states if there is no storing
        remove_forward_pass_if_no_Store(bwd_generator)

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


def cavity_flow_opt(sdfg: SDFG):
    """
    """
    for state in sdfg.all_states():
        if state.label == "cavity_flow_pressure_poisson_62_call_37_reversed":
            # Remove all nodes from this state
            for node in state.nodes():
                state.remove_node(node)

        if state.label == "call_40_reversed":
            # Remove all nodes from this state
            for node in state.nodes():
                if isinstance(
                        node, nodes.AccessNode
                ) and "gradient_pn" == node.data and state.in_degree(
                        node) == 2:
                    target_node = node
                    # Do a bfs to see what to remove
                    nodes_to_remove = state.bfs_nodes(target_node)
                    nodes_to_remove = [
                        n for n in list(nodes_to_remove) if n != target_node
                    ]
                    state.remove_nodes_from(nodes_to_remove)
                    target_node.data = "gradient_p"
                    state.in_edges(target_node)[0].data.data = "gradient_p"
                    first_map = state.in_edges(target_node)[0].src
                    state.in_edges(first_map)[0].data.data = "gradient_p"
                    state.in_edges(first_map)[1].data.data = "gradient_p"
                    state.in_edges(target_node)[1].data.data = "gradient_p"
                    second_map = state.in_edges(target_node)[1].src
                    state.in_edges(second_map)[0].data.data = "gradient_p"
                    state.in_edges(second_map)[1].data.data = "gradient_p"

                    break


def remove_forward_pass_if_no_Store(backward_gen: BackwardPassGenerator):
    """
    Remove the forward pass if there is no storing of the output
    """
    for node, parent in backward_gen.backward_sdfg.all_nodes_recursive():
        if isinstance(
                node,
                SDFGState) and node in backward_gen.reversed_states_map.keys():
            for snode in node.nodes():
                node.remove_node(snode)

    # Remove empty loops
    for node, parent in backward_gen.backward_sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion) and len(node.nodes()) == 0:
            print(f"Removing empty loop {node} in {backward_gen.sdfg.name}")
            parent.remove_node(node)

        if isinstance(node,
                      LoopRegion) and len(node.nodes()) == 1 and isinstance(
                          node.nodes()[0], SDFGState) and len(
                              node.nodes()[0].nodes()) == 0:
            in_edges = parent.in_edges(node)
            out_edges = parent.out_edges(node)
            if not len(in_edges) == 1 or not len(out_edges) == 1:
                continue
            in_edge = in_edges[0]
            out_edge = out_edges[0]
            parent.add_edge(in_edge.src, out_edge.dst, in_edge.data)
            print(f"Removing {node}")
            parent.remove_node(node)

    # Also remove wcr edge if unncecessary
    for node, parent in backward_gen.backward_sdfg.all_nodes_recursive():
        if "BinOp_23_reversed" == parent.label and isinstance(
                node, nodes.AccessNode) and parent.in_degree(
                    node) == 1 and parent.out_degree(node) == 0:
            in_edge = parent.in_edges(node)[0]
            if "gradient_a" not in in_edge.dst.label:
                continue
            if isinstance(in_edge.src,
                          nodes.MapExit) and in_edge.data.wcr is not None:
                print(
                    f"Resstign unnecessary wcr edge {in_edge} in {backward_gen.sdfg.name}"
                )
                for tree_edge in parent.memlet_tree(in_edge):
                    tree_edge.data.wcr = None


def preprocess_fwd_sdfg(forward_sdfg: SDFG):
    """
    Some preprocessing steps to make AD easier. Mainly pattern detect softmax to remove the max reduction
    """

    # Remove Softmax max reduction
    # for node, parent in forward_sdfg.all_nodes_recursive():
    #     # Chec if the node is still in the SDFG in case the pattern has been removed
    #     if node not in parent.nodes():
    #         continue
    #     is_soft_max, pattern_nodes = _is_softmax_reduction(node, parent)
    #     if is_soft_max:
    #         _softmax_reduction_to_lib_node(pattern_nodes, parent)

    # Remove unnecessary temporary Scalars
    for node, parent in forward_sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and "tmp1" in node.data:
            in_edges = parent.in_edges(node)
            out_edges = parent.out_edges(node)
            if len(in_edges) == 1 and len(out_edges) == 1:
                in_edge = in_edges[0]
                out_edge = out_edges[0]
                if isinstance(in_edge.src, nodes.AccessNode):
                    node_desc = forward_sdfg.arrays[node.data]
                    in_desc = forward_sdfg.arrays[in_edge.src.data]
                    if node_desc.shape == (
                            1,
                    ) and node_desc.transient and not in_desc.transient:
                        # skip connection to out edge directly
                        new_memlet = copy.deepcopy(in_edge.data)
                        new_memlet.other_subset = None
                        parent.add_edge(in_edge.src, in_edge.src_conn,
                                        out_edge.dst, out_edge.dst_conn,
                                        new_memlet)
                        parent.remove_node(node)


def fuse_states_cav(sdfg_bwd_ao: SDFG):
    """
    """
    if sdfg_bwd_ao.name != "cavity_flow":
        return
    for node, parent in sdfg_bwd_ao.all_nodes_recursive():
        # if isinstance(node, SDFGState) and "assign_80_8_reversed" == node.label:
        #     state2 = node
        # if isinstance(node, SDFGState) and "assign_81_8_reversed" == node.label:
        #     state3 = node
        # if isinstance(node, SDFGState) and "assign_83_8_reversed" == node.label:
        #     state4 = node
        # if isinstance(node, SDFGState) and "assign_84_8_reversed" == node.label:
        #     state5 = node
        # if isinstance(node, SDFGState) and "assign_85_8_reversed" == node.label:
        #     state6 = node

        if isinstance(node, SDFGState) and "assign_79_8" == node.label:
            state20 = node
        if isinstance(node, SDFGState) and "assign_80_8" == node.label:
            state30 = node
        if isinstance(node, SDFGState) and "assign_81_8" == node.label:
            state40 = node
        if isinstance(node, SDFGState) and "assign_83_8" == node.label:
            state50 = node
        if isinstance(node, SDFGState) and "assign_84_8" == node.label:
            state60 = node
        if isinstance(node, SDFGState) and "assign_85_8" == node.label:
            state70 = node

    # cavity_flow_fuse_initial_states(sdfg_bwd_ao, first_state=state2, second_state=state3)
    # cavity_flow_fuse_initial_states(sdfg_bwd_ao, first_state=state2, second_state=state4)
    # cavity_flow_fuse_initial_states(sdfg_bwd_ao, first_state=state2, second_state=state5)
    # cavity_flow_fuse_initial_states(sdfg_bwd_ao, first_state=state2, second_state=state6)
    # cycle_edge = state2.parent_graph.in_edges(state2)[0]
    # state2.parent_graph.remove_edge(cycle_edge)

    cavity_flow_fuse_initial_states(sdfg_bwd_ao,
                                    first_state=state20,
                                    second_state=state30)
    cavity_flow_fuse_initial_states(sdfg_bwd_ao,
                                    first_state=state20,
                                    second_state=state40)
    cavity_flow_fuse_initial_states(sdfg_bwd_ao,
                                    first_state=state20,
                                    second_state=state50)
    cavity_flow_fuse_initial_states(sdfg_bwd_ao,
                                    first_state=state20,
                                    second_state=state60)
    cavity_flow_fuse_initial_states(sdfg_bwd_ao,
                                    first_state=state20,
                                    second_state=state70)


def top_level_nodes(state: SDFGState):
    return state.scope_children()[None]


def cavity_flow_fuse_initial_states(sdfg: SDFG, first_state: SDFGState,
                                    second_state: SDFGState):
    """
    Fuse the initial states of the cavity flow SDFG
    """

    graph = first_state.parent_graph

    # Remove interstate edge(s)
    edges = graph.edges_between(first_state, second_state)
    for edge in edges:
        if edge.data.assignments:
            for src, dst, other_data in graph.in_edges(first_state):
                other_data.assignments.update(edge.data.assignments)
        graph.remove_edge(edge)

    # Special case 1: first state is empty
    if first_state.is_empty():
        sdutil.change_edge_dest(graph, first_state, second_state)
        graph.remove_node(first_state)
        if graph.start_block == first_state:
            graph.start_block = graph.node_id(second_state)
        return

    # Special case 2: second state is empty
    if second_state.is_empty():
        sdutil.change_edge_src(graph, second_state, first_state)
        sdutil.change_edge_dest(graph, second_state, first_state)
        graph.remove_node(second_state)
        if graph.start_block == second_state:
            graph.start_block = graph.node_id(first_state)
        return

    # Normal case: both states are not empty

    # Find source/sink (data) nodes
    first_input = [
        node for node in first_state.source_nodes()
        if isinstance(node, nodes.AccessNode)
    ]
    first_output = [
        node for node in first_state.sink_nodes()
        if isinstance(node, nodes.AccessNode)
    ]
    second_input = [
        node for node in second_state.source_nodes()
        if isinstance(node, nodes.AccessNode)
    ]

    top2 = top_level_nodes(second_state)

    # first input = first input - first output
    first_input = [
        node for node in first_input
        if next((x for x in first_output if x.data == node.data), None) is None
    ]

    # NOTE: We exclude Views from the process of merging common data nodes because it may lead to double edges.
    second_mid = [
        x for x in list(nx.topological_sort(second_state._nx))
        if isinstance(x, nodes.AccessNode) and second_state.out_degree(x) > 0
        and not isinstance(sdfg.arrays[x.data], dt.View)
    ]

    # Merge second state to first state
    # First keep a backup of the topological sorted order of the nodes
    sdict = first_state.scope_dict()
    order = [
        x for x in reversed(list(nx.topological_sort(first_state._nx)))
        if isinstance(x, nodes.AccessNode) and sdict[x] is None
    ]
    for node in second_state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            # update parent information
            node.sdfg.parent = first_state
        first_state.add_node(node)
    for src, src_conn, dst, dst_conn, data in second_state.edges():
        first_state.add_edge(src, src_conn, dst, dst_conn, data)

    top = top_level_nodes(first_state)

    # Merge common (data) nodes
    merged_nodes = set()
    removed_nodes = set()
    for node in second_mid:

        # merge only top level nodes, skip everything else
        if node not in top2:
            continue

        candidates = [
            x for x in order if x.data == node.data and x in top
            and x not in merged_nodes and x not in removed_nodes
        ]
        source_node = first_state.in_degree(node) == 0

        # If not source node, try to connect every memlet-intersecting candidate
        if not source_node:
            for cand in candidates:
                if StateFusion.memlets_intersect(first_state, [cand], False,
                                                 second_state, [node], True):
                    if nx.has_path(first_state._nx, cand,
                                   node):  # Do not create cycles
                        continue
                    sdutil.change_edge_src(first_state, cand, node)
                    sdutil.change_edge_dest(first_state, cand, node)
                    first_state.remove_node(cand)
                    removed_nodes.add(cand)
            continue

        if len(candidates) == 0:
            continue
        elif len(candidates) == 1:
            n = candidates[0]
        else:
            # Choose first candidate that intersects memlets
            for cand in candidates:
                if StateFusion.memlets_intersect(first_state, [cand], False,
                                                 second_state, [node], True):
                    n = cand
                    break
            else:
                # No node intersects, use topologically-last node
                n = candidates[0]

        sdutil.change_edge_src(first_state, node, n)
        sdutil.change_edge_dest(first_state, node, n)
        first_state.remove_node(node)
        removed_nodes.add(node)
        merged_nodes.add(n)

    # Redirect edges and remove second state
    sdutil.change_edge_src(graph, second_state, first_state)
    graph.remove_node(second_state)
    if graph.start_block == second_state:
        graph.start_block = graph.node_id(first_state)


def _is_softmax_reduction(node: nodes.NestedSDFG, parent: SDFGState):
    """
        Change the forward pass for the softmax function to avoid having a max reduction
        """
    if isinstance(node, nodes.NestedSDFG) and "softmax" in node.label.lower():
        return True, node

    # Pattern match the softmax start node
    if not isinstance(node, nodes.AccessNode) or parent.out_degree(node) != 2:
        return False, None
    out_edges = parent.out_edges(node)

    reduction = out_edges[0].dst if isinstance(
        out_edges[0].dst, nodes.LibraryNode) else out_edges[1].dst
    map_entry = out_edges[0].dst if isinstance(
        out_edges[0].dst, dace.nodes.MapEntry) else out_edges[1].dst
    if not isinstance(reduction, nodes.LibraryNode) or not isinstance(
            map_entry, dace.nodes.MapEntry):
        return False, None

    if not "Reduce" in reduction.label:
        return False, None

    # Get the map exit node
    map_exit = parent.exit_node(map_entry)

    # out edges of exit
    out_edges = parent.out_edges(map_exit)
    if len(out_edges) != 1:
        return False, None
    first_exit_out_edge = out_edges[0]

    # should point to an
    if not isinstance(first_exit_out_edge.dst, nodes.AccessNode):
        return False, None

    edges = parent.out_edges(first_exit_out_edge.dst)
    if len(edges) != 1:
        return False, None
    second_map = edges[0].dst

    second_exit = parent.exit_node(second_map)
    edges = parent.out_edges(second_exit)
    if len(edges) != 1:
        return False, None
    second_exit_out_edge = edges[0]

    # should be an
    if not isinstance(second_exit_out_edge.dst, nodes.AccessNode):
        return False, None

    # shoudl have two edges out
    edges = parent.out_edges(second_exit_out_edge.dst)
    if len(edges) != 2:
        return False, None

    second_reduction = edges[0].dst if isinstance(
        edges[0].dst, nodes.LibraryNode) else edges[1].dst
    second_map_entry = edges[0].dst if isinstance(
        edges[0].dst, dace.nodes.MapEntry) else edges[1].dst
    if not isinstance(second_reduction, nodes.LibraryNode) or not isinstance(
            second_map_entry, dace.nodes.MapEntry):
        return False, None

    # final map exit
    final_map_exit = parent.exit_node(second_map_entry)
    edges = parent.out_edges(final_map_exit)
    if len(edges) != 1:
        return False, None
    final_exit_out_edge = edges[0]

    output_node = final_exit_out_edge.dst
    if not isinstance(output_node, nodes.AccessNode):
        return False, None

    nodes_list = parent.all_nodes_between(node, output_node)
    nodes_list = list(nodes_list)
    nodes_list.insert(0, node)
    nodes_list.append(output_node)
    return True, nodes_list


def _softmax_reduction_to_lib_node(pattern_nodes, parent: SDFGState):
    """
        Change the forward pass for the softmax function to avoid having a max reduction
        """
    import dace.libraries.onnx as donnx
    # Get the equivelent library node
    lib_node = donnx.ONNXSoftmax("Softmax")
    parent.add_node(lib_node)

    if isinstance(pattern_nodes, nodes.NestedSDFG):
        node = pattern_nodes

        # in edges
        in_edges = parent.in_edges(node)
        assert len(in_edges) == 1
        in_edge = in_edges[0]

        # out edges
        out_edges = parent.out_edges(node)
        assert len(out_edges) == 1
        out_edge = out_edges[0]

        # Connect it to the input and output
        parent.add_edge(in_edge.src, in_edge.src_conn, lib_node, "input",
                        in_edge.data)
        parent.add_edge(lib_node, "output", out_edge.dst, out_edge.dst_conn,
                        out_edge.data)

        # Remove the nested SDFG
        parent.remove_node(node)
    else:
        input = pattern_nodes[0]
        in_edge = parent.out_edges(input)
        assert len(in_edge) == 2
        assert in_edge[0].data == in_edge[1].data
        in_edge = in_edge[0]
        parent.add_edge(in_edge.src, in_edge.src_conn, lib_node, "input",
                        in_edge.data)

        output = pattern_nodes[-1]
        out_edge = parent.in_edges(output)
        assert len(out_edge) == 1
        out_edge = out_edge[0]
        parent.add_edge(lib_node, "output", out_edge.dst, out_edge.dst_conn,
                        out_edge.data)

        pattern_nodes = pattern_nodes[1:-1]
        # Remove the list of nodes from the parent
        parent.remove_nodes_from(pattern_nodes)


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

        if in_edge.dst_conn is None or node.in_connectors[
                in_edge.dst_conn] != dace.dtypes.bool:
            continue

        if out_edge.src_conn is None or node.out_connectors[
                out_edge.src_conn] != dace.dtypes.float64:
            continue

        node.out_connectors[out_edge.src_conn] = dace.dtypes.pointer(
            dace.dtypes.float64)

    # Avoid wcr on the first sum reduction
    # for node, parent in sdfg.all_nodes_recursive():
    #     if isinstance(node, dace.nodes.AccessNode) and "gradient_A" in node.data and parent.in_degree(node) == 1 and parent.out_degree(node) == 0:
    #         in_edge = parent.in_edges(node)[0]
    #         if isinstance(in_edge.src, nodes.MapExit) and in_edge.data.wcr is not None:
    #             for tree_edge in parent.memlet_tree(in_edge):
    #                 tree_edge.data.wcr = None


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
                            if isinstance(
                                    sdfg.arrays[an.data],
                                (dace.data.View, dace.data.ArrayView)):
                                desc.strides = (
                                    sdfg.arrays[an.data].strides[0],
                                    desc.strides[1])
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
                                if isinstance(
                                        sdfg.arrays[an.data],
                                    (dace.data.View, dace.data.ArrayView)):
                                    desc.strides = (
                                        sdfg.arrays[an.data].strides[0],
                                        desc.strides[1])
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
