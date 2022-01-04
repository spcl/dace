# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Infer Types: This module is responsible for inferring connector types in the SDFG.
"""
from typing import *
from dace.sdfg.graph import MultiConnectorEdge, Graph, SubgraphView
from dace.sdfg.state import SDFGState
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.nodes import Node, Tasklet
import dace.dtypes
import dace.sdfg.infer_types
import dace.transformation.dataflow
import dace.transformation.helpers
import dace.codegen.targets.sve.util
import dace.frontend.operations
import dace.data as data
import dace.dtypes as dtypes
from collections import defaultdict
from dace.sdfg.utils import dfs_topological_sort


class TypeInferenceDict(DefaultDict[Tuple[Tasklet, str, bool], dtypes.typeclass]):
    def __init__(self):
        super().__init__(lambda: dtypes.typeclass(None))


def infer_tasklet_connectors(sdfg: SDFG, state: SDFGState, node: Tasklet, inferred: TypeInferenceDict):
    """ Infers the connectors in a Tasklet using its code and type inference. """

    if node.code.language != dtypes.Language.Python:
        raise NotImplementedError('Tasklet inference for other languages than Python not supported')

    if any(inferred[(node, conn, True)].type is None for conn in node.in_connectors):
        raise TypeError('Cannot infer output connectors of tasklet "%s", '
                        'not all input connectors have types' % str(node))

    # Avoid import loop
    from dace.codegen.tools.type_inference import infer_types

    # Get symbols defined at beginning of node
    syms = state.symbols_defined_at(node)

    in_syms = {}
    for conn in node.in_connectors:
        if inferred[(node, conn, True)].type is not None:
            # Let the inferred dictionary win if it has some type information
            in_syms[conn] = inferred[(node, conn, True)]
        else:
            in_syms[conn] = node.in_connectors[conn]

    syms.update(in_syms)

    # Infer all types in tasklet
    new_syms = infer_types(node.code.code, syms)

    for cname in node.out_connectors:
        if inferred[(node, cname, False)].type is None:
            if cname not in new_syms:
                raise TypeError('Cannot infer type of tasklet %s output '
                                '"%s", please specify manually.' % (node.label, cname))
            inferred[(node, cname, False)] = new_syms[cname]


def infer_node_connectors(sdfg: SDFG, state: SDFGState, node: nodes.Node, inferred: TypeInferenceDict):
    """ Infers the connectors of a node and updates `inferred` accordingly. """

    # Try to infer input connector type from node type or previous edges
    for e in state.in_edges(node):
        cname = e.dst_conn
        if cname is None:
            continue

        scalar = (e.data.subset and e.data.subset.num_elements() == 1)
        if e.data.data is not None:
            allocated_as_scalar = (sdfg.arrays[e.data.data].storage is not dtypes.StorageType.GPU_Global)
        else:
            allocated_as_scalar = True

        if inferred[(node, cname, True)].type is None:
            # If nested SDFG, try to use internal array type
            if isinstance(node, nodes.NestedSDFG):
                scalar = (isinstance(node.sdfg.arrays[cname], data.Scalar) and allocated_as_scalar)
                dtype = node.sdfg.arrays[cname].dtype
                ctype = (dtype if scalar else dtypes.pointer(dtype))
            elif e.data.data is not None:  # Obtain type from memlet
                scalar |= isinstance(sdfg.arrays[e.data.data], data.Scalar)
                if isinstance(node, nodes.LibraryNode):
                    scalar &= allocated_as_scalar
                dtype = sdfg.arrays[e.data.data].dtype
                ctype = (dtype if scalar else dtypes.pointer(dtype))
            else:  # Code->Code
                src_edge = state.memlet_path(e)[0]
                sconn = src_edge.src.out_connectors[src_edge.src_conn]
                if sconn.type is None:
                    raise TypeError('Ambiguous or uninferable type in'
                                    ' connector "%s" of node "%s"' % (sconn, src_edge.src))
                ctype = sconn
            inferred[(node, cname, True)] = ctype

    # Try to infer outputs from output edges
    for e in state.out_edges(node):
        cname = e.src_conn
        if cname is None:
            continue

        scalar = (e.data.subset and e.data.subset.num_elements() == 1
                  and (not e.data.dynamic or (e.data.dynamic and e.data.wcr is not None)))
        if e.data.data is not None:
            allocated_as_scalar = (sdfg.arrays[e.data.data].storage is not dtypes.StorageType.GPU_Global)
        else:
            allocated_as_scalar = True

        if inferred[(node, cname, False)].type is None:
            # If nested SDFG, try to use internal array type
            if isinstance(node, nodes.NestedSDFG):
                scalar = (isinstance(node.sdfg.arrays[cname], data.Scalar) and allocated_as_scalar)
                dtype = node.sdfg.arrays[cname].dtype
                ctype = (dtype if scalar else dtypes.pointer(dtype))
            elif e.data.data is not None:  # Obtain type from memlet
                scalar |= isinstance(sdfg.arrays[e.data.data], data.Scalar)
                if isinstance(node, nodes.LibraryNode):
                    scalar &= allocated_as_scalar
                dtype = sdfg.arrays[e.data.data].dtype
                ctype = (dtype if scalar else dtypes.pointer(dtype))
            else:
                continue
            inferred[(node, cname, False)] = ctype

    # Let the node infer other output types on its own
    if isinstance(node, nodes.Tasklet):
        infer_tasklet_connectors(sdfg, state, node, inferred)
    elif isinstance(node, nodes.NestedSDFG):
        infer_connector_types(node.sdfg, inferred=inferred)

    # If there are any remaining uninferable connectors, fail
    for e in state.out_edges(node):
        cname = e.src_conn
        if cname and inferred[(node, cname, False)].type is None:
            raise TypeError('Ambiguous or uninferable type in' ' connector "%s" of node "%s"' % (cname, node))


def infer_connector_types(sdfg: SDFG,
                          state: SDFGState = None,
                          graph: SubgraphView = None,
                          inferred: TypeInferenceDict = None):
    """
        Infers the connector types of an SDFG, state or subgraph and returns them in a dictionary
        consisting of tuples with node, name and a bool whether it is an input connector
        (`True` for input, `False` for output).

        This method does not modify the connectors, meaning it is read-only.
        To apply the changes, use `apply_connector_types`.

        It can be executed in different modes, depending on the provided arguments:
        * on an SDFG by only providing `sdfg`
        * on a state by providing `sdfg` and `state`
        * on a subgraph by providing `sdfg`, `state` and `graph`

        :param sdfg: The SDFG to infer.
        :param state: The state to infer.
        :param graph: The graph to infer.
        :param inferred: The dictionary of already inferred types.
    """

    if inferred is None:
        inferred = TypeInferenceDict()

    if sdfg is None:
        raise ValueError('No SDFG was provided')

    if state is None and graph is None:
        for state in sdfg.nodes():
            for node in dfs_topological_sort(state):
                infer_node_connectors(sdfg, state, node, inferred)

    elif state is not None and graph is None:
        for node in dfs_topological_sort(state):
            infer_node_connectors(sdfg, state, node, inferred)
    elif state is not None and graph is not None:
        for node in dfs_topological_sort(graph):
            infer_node_connectors(sdfg, state, node, inferred)
    else:
        raise ValueError('Missing some arguments')

    return inferred


def apply_connector_types(inferred: TypeInferenceDict):
    """ Applies the inferred connector types on the SDFG. """

    for (node, conn, is_in), dtype in inferred.items():
        if dtype.type is None:
            continue
        if is_in:
            node.in_connectors[conn] = dtype
        else:
            node.out_connectors[conn] = dtype
