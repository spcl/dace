# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Vectorization: This module allows to force vectorization of an SDFG.
"""

import dace
import dace.codegen.targets.sve.util as util
import dace.codegen.targets.sve.codegen
from dace.transformation.optimizer import Optimizer
import dace.dtypes
import dace.sdfg.infer_types
import dace.sdfg.graph as graph
import dace.sdfg.nodes as nodes
import dace.data as data
import dace.symbolic as symbolic


def get_connector_edge(dfg: dace.SDFGState, node: nodes.Node, conn: str, is_in_conn: bool) -> graph.Edge:
    for e in dfg.all_edges(node):
        if (is_in_conn and e.dst == node and e.dst_conn == conn) or (not is_in_conn and e.src == node and e.src_conn == conn):
            return e
    return None


def is_single_element_subset(subset) -> bool:
    for s in subset:
        if isinstance(s, tuple):
            print(s)
            if s[0] != s[1]:
                return False
    return True


def vectorize_connector(sdfg: dace.SDFG, dfg: dace.SDFGState, node: dace.nodes.Node, par: str, conn: str, is_input: bool):
    edge = get_connector_edge(dfg, node, conn, is_input)
    connectors = node.in_connectors if is_input else node.out_connectors

    if edge.data.data is None:
        # Empty memlets
        return

    desc = sdfg.arrays[edge.data.data]

    if isinstance(desc, data.Stream):
        # Streams are treated differently in SVE, instead of pointers they become vectors of unknown size
        connectors[conn] = dace.dtypes.vector(connectors[conn].base_type, -1)
        return

    if isinstance(connectors[conn], (dace.dtypes.vector, dace.dtypes.pointer)):
        # No need for vectorization
        return

    subset = edge.data.subset
    
    if is_single_element_subset(subset):
        sve_axis = -1
        for i, axis in enumerate(subset):
            for expr in axis:
                sym = symbolic.pystr_to_symbolic(expr)
                if par in sym.free_symbols:
                    # TODO: What if it occurs multiple times?
                    sve_axis = i
        sve_subset = subset[sve_axis]
        # TODO: Set proper stride
        edge.data.subset[sve_axis] = (
            sve_subset[0], sve_subset[0] + util.SVE_LEN - sve_subset[2], sve_subset[2])
   
    connectors[conn] = dace.dtypes.vector(connectors[conn], util.SVE_LEN)


def vectorize(sdfg: dace.SDFG, par: str, ignored_conns: list=[]):
    input_bits = set([sdfg.arrays[a].dtype.bytes * 8 for a in sdfg.arrays])
    if len(input_bits) > 1:
        raise NotImplementedError('Different data type sizes as inputs')
    input_bit_width = list(input_bits)[0]

    dace.ScheduleType.register('SVE_Map')

    # FIXME: Hardcoded for the demo machine (512 bits)
    util.SVE_LEN.set(512 / input_bit_width)

    dace.sdfg.infer_types.infer_connector_types(sdfg)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if node.params[-1] == par:
                node.schedule = dace.ScheduleType.SVE_Map
        elif isinstance(node, dace.nodes.Tasklet):
            for c in node.in_connectors:
                if c in ignored_conns:
                    continue
                vectorize_connector(sdfg, sdfg.start_state,
                                    node, par, c, True)
            for c in node.out_connectors:
                if c in ignored_conns:
                    continue
                vectorize_connector(sdfg, sdfg.start_state,
                                    node, par, c, False)

    dace.SCOPEDEFAULT_SCHEDULE[
        dace.ScheduleType.SVE_Map] = dace.ScheduleType.Sequential
    dace.SCOPEDEFAULT_STORAGE[
        dace.ScheduleType.SVE_Map] = dace.StorageType.CPU_Heap

    return sdfg
