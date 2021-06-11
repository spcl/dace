# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Vectorization: This module allows to force vectorization of an SDFG.
"""

import dace
import dace.codegen.targets.sve.util as util
import dace.codegen.targets.sve.codegen
from dace.transformation.optimizer import Optimizer
import dace.dtypes
from dace.codegen.targets.sve.infer_types import infer_connector_types
import dace.sdfg.infer_types
import dace.sdfg.graph as graph
import dace.sdfg.nodes as nodes
import dace.data as data
import dace.symbolic as symbolic
import dace.codegen.targets.common


def get_connector_edges(dfg: dace.SDFGState, node: nodes.Node, conn: str, is_in_conn: bool) -> graph.Edge:
    edges = []
    for e in dfg.all_edges(node):
        if (is_in_conn and e.dst == node and e.dst_conn == conn) or (not is_in_conn and e.src == node and e.src_conn == conn):
            edges.append(e)
    return edges


def is_single_element_subset(subset) -> bool:
    for s in subset:
        if isinstance(s, tuple):
            print(s)
            if s[0] != s[1]:
                return False
    return True


def vectorize_connector(sdfg: dace.SDFG, dfg: dace.SDFGState, node: dace.nodes.Node, par: str, conn: str, is_input: bool):
    edges = get_connector_edges(dfg, node, conn, is_input)
    connectors = node.in_connectors if is_input else node.out_connectors

    for edge in edges:
        if edge.data.data is None:
            # Empty memlets
            return

        desc = sdfg.arrays[edge.data.data]

        if isinstance(desc, data.Stream):
            # Streams are treated differently in SVE, instead of pointers they become vectors of unknown size
            connectors[conn] = dace.dtypes.vector(
                connectors[conn].base_type, -1)
            return

        if isinstance(connectors[conn], (dace.dtypes.vector, dace.dtypes.pointer)):
            # No need for vectorization
            return

        subset = edge.data.subset

        sve_dim = None
        for i, rng in enumerate(subset):
            for expr in rng:
                if symbolic.symbol(par) in symbolic.pystr_to_symbolic(expr).free_symbols:
                    if sve_dim is not None and sve_dim != i:
                        raise util.NotSupportedError(
                            'Non-vectorizable memlet (loop param occurs in more than one dimension)')
                    sve_dim = i

        if sve_dim is None and edge.data.wcr is None:
            # Should stay scalar
            return

        if sve_dim is not None:
            sve_subset = subset[sve_dim]
            edge.data.subset[sve_dim] = (
                sve_subset[0], sve_subset[0] + util.SVE_LEN, sve_subset[2])

        connectors[conn] = dace.dtypes.vector(
            connectors[conn].type or desc.dtype, util.SVE_LEN)


def vectorize(sdfg: dace.SDFG, par: str, ignored_conns: list = []):
    input_bits = set([sdfg.arrays[a].dtype.bytes * 8 for a in sdfg.arrays])
    if len(input_bits) > 1:
        raise NotImplementedError('Different data type sizes as inputs')
    input_bit_width = list(input_bits)[0]

    sdfg.apply_strict_transformations()

    # FIXME: Hardcoded for the demo machine (512 bits)
    util.SVE_LEN.set(512 / input_bit_width)

    for node, dfg in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if node.params[-1] == par:
                node.schedule = dace.ScheduleType.SVE_Map
                for c in node.out_connectors:
                    edges = get_connector_edges(dfg, node, c, False)
                    vectorize_connector(sdfg, dfg,
                                        node, par, c, False)
                    for e in edges:
                        vectorize_connector(sdfg, dfg,
                                            e.dst, par, e.dst_conn, True)

    for edge, dfg in sdfg.all_edges_recursive():
        if not isinstance(dfg, dace.SDFGState):
            continue
        # Force every output connector within the graph to be a vector
        #if edge.data.wcr is None:
        #    continue
        scope = util.get_sve_scope(sdfg, dfg, edge.src)
        if scope is not None:
            vectorize_connector(sdfg, dfg, edge.src, par, edge.src_conn, False)

    # Then use a tweaked (but incorrect) version of infer_connector_types
    infer_connector_types(sdfg)

    return sdfg
