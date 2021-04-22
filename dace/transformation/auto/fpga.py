# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA-Oriented Automatic optimization routines for SDFGs. """

from dace.sdfg import SDFG, SDFGState, trace_nested_access
from dace import config, data as dt, dtypes, Memlet, symbolic
from dace.sdfg import SDFG, nodes, graph as gr


def fpga_global_to_local(sdfg: SDFG) -> None:
    """ Takes an entire  SDFG and changes the storage type of a global FPGA data container
        to Local in the following situation:
           - the data is transient,
           - the data is not a transient shared with other states, and
           - the data has a compile-time known size.
    """

    count = 0

    for name, desc in sdfg.arrays.items():
        if desc.transient and name not in sdfg.shared_transients(
        ) and desc.storage == dtypes.StorageType.FPGA_Global:

            # Get the total size, trying to resolve it to constant if it is a symbol
            total_size = symbolic.resolve_symbol_to_constant(
                desc.total_size, sdfg)

            if total_size is not None:
                desc.storage = dtypes.StorageType.FPGA_Local
                count = count + 1

                # update all access nodes that refer to this container
                for node, graph in sdfg.all_nodes_recursive():
                    if isinstance(node, nodes.AccessNode):
                        trace = trace_nested_access(node, graph, graph.parent)

                        for (_, candidate
                             ), memlet_trace, state_trace, sdfg_trace in trace:
                            if candidate is not None and candidate.data == name:
                                nodedesc = node.desc(graph)
                                nodedesc.storage = dtypes.StorageType.FPGA_Local

    if config.Config.get_bool('debugprint'):
        print(f'Applied {count} Global-To-Local.')
