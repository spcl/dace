# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data, dtypes
from dace.codegen.tools import type_inference
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import nodes
from dace.sdfg.utils import dfs_topological_sort
from typing import Dict, List

#############################################################################
# Connector type inference
#
# The modification in this vectorization is to infer the input connectors,
# then let the node infer the outputs _before_ inferring the output connectors
# (basically moved this line up), because type_inference is now able to infer indirection


def infer_connector_types(sdfg: SDFG):
    """ 
    Infers connector types throughout an SDFG and its nested SDFGs in-place.
    :param sdfg: The SDFG to infer.
    """
    # Loop over states, and in a topological sort over each state's nodes
    for state in sdfg.nodes():
        for node in dfs_topological_sort(state):
            # Try to infer input connector type from node type or previous edges
            for e in state.in_edges(node):
                cname = e.dst_conn
                if cname is None:
                    continue
                scalar = (e.data.subset and e.data.subset.num_elements() == 1)
                if e.data.data is not None:
                    allocated_as_scalar = (sdfg.arrays[e.data.data].storage is
                                           not dtypes.StorageType.GPU_Global)
                else:
                    allocated_as_scalar = True

                if node.in_connectors[cname].type is None:
                    # If nested SDFG, try to use internal array type
                    if isinstance(node, nodes.NestedSDFG):
                        scalar = (isinstance(node.sdfg.arrays[cname],
                                             data.Scalar)
                                  and allocated_as_scalar)
                        dtype = node.sdfg.arrays[cname].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    elif e.data.data is not None:  # Obtain type from memlet
                        src_edge = state.memlet_path(e)[0]
                        if src_edge.src_conn is not None:
                            ctype = src_edge.src.out_connectors[src_edge.src_conn]
                        else:
                            scalar |= isinstance(sdfg.arrays[e.data.data],
                                                data.Scalar)
                            if isinstance(node, nodes.LibraryNode):
                                scalar &= allocated_as_scalar
                            dtype = sdfg.arrays[e.data.data].dtype
                            ctype = (dtype if scalar else dtypes.pointer(dtype))
                    else:  # Code->Code
                        src_edge = state.memlet_path(e)[0]
                        sconn = src_edge.src.out_connectors[src_edge.src_conn]
                        if sconn.type is None:
                            raise TypeError('Ambiguous or uninferable type in'
                                            ' connector "%s" of node "%s"' %
                                            (sconn, src_edge.src))
                        ctype = sconn
                    node.in_connectors[cname] = ctype

            # Let the node infer other output types on its own
            node.infer_connector_types(sdfg, state)

            # Try to infer outputs from output edges
            for e in state.out_edges(node):
                cname = e.src_conn
                if cname is None:
                    continue
                scalar = (e.data.subset and e.data.subset.num_elements() == 1
                          and (not e.data.dynamic or
                               (e.data.dynamic and e.data.wcr is not None)))
                if e.data.data is not None:
                    allocated_as_scalar = (sdfg.arrays[e.data.data].storage is
                                           not dtypes.StorageType.GPU_Global)
                else:
                    allocated_as_scalar = True
                
                if node.out_connectors[cname].type is None:
                    # If nested SDFG, try to use internal array type
                    if isinstance(node, nodes.NestedSDFG):
                        scalar = (isinstance(node.sdfg.arrays[cname],
                                             data.Scalar)
                                  and allocated_as_scalar)
                        dtype = node.sdfg.arrays[cname].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    elif e.data.data is not None:  # Obtain type from memlet
                        scalar |= isinstance(sdfg.arrays[e.data.data],
                                             data.Scalar)
                        if isinstance(node, nodes.LibraryNode):
                            scalar &= allocated_as_scalar
                        dtype = sdfg.arrays[e.data.data].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    else:
                        continue
                    node.out_connectors[cname] = ctype

            # If there are any remaining uninferable connectors, fail
            for e in state.out_edges(node):
                cname = e.src_conn
                if cname and node.out_connectors[cname].type is None:
                    raise TypeError('Ambiguous or uninferable type in'
                                    ' connector "%s" of node "%s"' %
                                    (cname, node))

