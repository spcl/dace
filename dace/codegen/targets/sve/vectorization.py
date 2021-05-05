# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Vectorization: This module is horrible, but allows for heavy vectorization (which the current Vectorization transform can't do yet).
    It is only needed for Milestone II and will be replaced by a nicer version in Milestone III.
"""

import dace
import dace.codegen.targets.sve.util
import dace.codegen.targets.sve.codegen
from dace.transformation.optimizer import Optimizer
import numpy as np
import dace.dtypes
from dace.sdfg import graph, state, find_input_arraynode, find_output_arraynode


def vectorize(sdfg, par, weak=False, special=False):
    # HACK: This function is extremely hacky and only needed for demonstration purposes in Milestone II.
    # I am fully aware that this is horrible!

    if weak:
        for xform in Optimizer(sdfg).get_pattern_matches(patterns=[
                dace.transformation.dataflow.vectorization.Vectorization
        ]):
            xform.apply(sdfg)
        return

    if len(set([sdfg.arrays[a].dtype.bytes for a in sdfg.arrays])) > 1:
        raise NotImplementedError('Different data type sizes as inputs')

    dace.ScheduleType.register('SVE_Map')

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if node.params[-1] == par:
                node.schedule = dace.ScheduleType.SVE_Map
        elif isinstance(node, dace.nodes.Tasklet):
            for c in node.in_connectors:
                if isinstance(node.in_connectors[c],
                              (dace.dtypes.vector, dace.dtypes.pointer)):
                    continue
                edges = sdfg.start_state.in_edges_by_connector(node, c)
                for e in edges:
                    data = e.data.data
                    if data is None:
                        continue
                    desc = sdfg.arrays[data]
                    if e.dst_conn == c:
                        if e.data.subset.ranges[0][0] != e.data.subset.ranges[
                                0][1]:
                            node.in_connectors[c] = dace.dtypes.pointer(
                                desc.dtype)
                        elif len(e.data.subset.free_symbols) == 0 and special:
                            node.in_connectors[c] = desc.dtype
                        else:
                            node.in_connectors[c] = dace.dtypes.vector(
                                desc.dtype, -1)

            for c in node.out_connectors:
                if isinstance(node.out_connectors[c],
                              (dace.dtypes.vector, dace.dtypes.pointer)):
                    continue
                edges = sdfg.start_state.out_edges_by_connector(node, c)
                for e in edges:
                    data = e.data.data
                    if data is None:
                        continue
                    desc = sdfg.arrays[data]
                    if e.src_conn == c:
                        if e.data.subset.ranges[0][0] != e.data.subset.ranges[
                                0][1]:
                            node.out_connectors[c] = dace.dtypes.pointer(
                                desc.dtype)
                        else:
                            node.out_connectors[c] = dace.dtypes.vector(
                                desc.dtype, -1)

    dace.SCOPEDEFAULT_SCHEDULE[
        dace.ScheduleType.SVE_Map] = dace.ScheduleType.Sequential
    dace.SCOPEDEFAULT_STORAGE[
        dace.ScheduleType.SVE_Map] = dace.StorageType.CPU_Heap

    return sdfg
