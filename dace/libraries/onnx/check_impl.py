""" Checkers for ONNXRuntime kernel op implementations. Currently the implementations
    just try to build and load the nodes individually.
"""
from copy import deepcopy as dc

import dace
from dace.libraries.onnx.nodes.onnx_op import ONNXOp

def check_impl(sdfg, state, node: ONNXOp):
    """ Check whether a ONNXOp node has an implementation in ORT """

    n_sdfg = dace.SDFG("checker")
    n_state = n_sdfg.add_state()

    # copy over the node to the new sdfg
    n_node = dc(node)
    n_state.add_node(n_node)
    params = {}
    for edge, is_input in node.iter_edges(state):
        memlet = edge.data
        arr = sdfg.arrays[memlet.data]

        n_sdfg.add_array(memlet.data, arr.shape, arr.dtype)
        access = n_state.add_access(memlet.data)
        if is_input:
            n_state.add_edge(access, None, n_node, edge.dst_conn, n_sdfg.get_array_memlet(memlet.data))
        else:
            n_state.add_edge(n_node, edge.src_conn, access, None, n_sdfg.get_array_memlet(memlet.data))

    n_sdfg.validate()

    # expand the node. This will add the initalization code to the sdfg
    n_sdfg.expand_library_nodes()

    # remove the state
    n_sdfg.remove_node(n_state)
    arrs = list(n_sdfg.arrays.keys())

    for arr in arrs:
        n_sdfg.remove_data(arr, validate=True)

    start_state = n_sdfg.add_state()

    n_sdfg()

