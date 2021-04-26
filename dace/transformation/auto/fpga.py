# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA-Oriented Automatic optimization routines for SDFGs. """

from dace.sdfg import SDFG, SDFGState, trace_nested_access
from dace import config, data as dt, dtypes, Memlet, symbolic
from dace.sdfg import SDFG, nodes, graph as gr


def fpga_global_to_local(sdfg: SDFG, max_size: int = 1048576) -> None:
    """ Takes an entire  SDFG and changes the storage type of a global FPGA data container
        to Local in the following situation:
           - the data is transient,
           - the data is not a transient shared with other states, and
           - the data has a compile-time known size.
        :param: sdfg: The SDFG to operate on. It must be a top-level SDFG.
        :param: max_size: maximum size (in bytes) that a container can have to be considered for
            storage type change
        :note: Operates in-place on the SDFG.
    """

    if sdfg.parent is not None:
        if config.Config.get_bool('debugprint'):
            print('Global-To-Local can be applied only to a top-level SDFG.')
        return  # Not the top-most SDFG. Do nothing.

    converted = []

    for name, desc in sdfg.arrays.items():
        if desc.transient and name not in sdfg.shared_transients(
        ) and desc.storage == dtypes.StorageType.FPGA_Global:

            # Get the total size, trying to resolve it to constant if it is a symbol
            total_size = symbolic.resolve_symbol_to_constant(
                desc.total_size, sdfg)

            if total_size is not None and total_size * desc.dtype.bytes <= max_size:
                desc.storage = dtypes.StorageType.FPGA_Local
                converted.append(name)

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
        print(
            f'Applied {len(converted)} Global-To-Local{": " if len(converted)>0 else "."} {", ".join(converted)}'
        )


def fpga_rr_interleave_containers_to_banks(sdfg: SDFG, num_banks: int = 4):
    '''
    Allocates the (global) arrays to FPGA off-chip memory banks, interleaving them in a
    Round-Robin (RR) fashion, following the order of appearance in the SDFG.
    :param sdfg: The SDFG to operate on. It must be a top-level SDFG.
    :param: num_banks: number of off-chip memory banks to consider
    :returns: a list containing  the number of arrays allocated to each bank
    :note: Operates in-place on the SDFG.
    '''

    if sdfg.parent is not None:
        if config.Config.get_bool('debugprint'):
            print(
                'Interleave Containers can be applied only to a top-level SDFG.'
            )
        return []  # Not the top-most SDFG. Do nothing.

    # keep track of memory allocated to each bank
    num_allocated = [0 for i in range(num_banks)]

    for i, (name, desc) in enumerate(sdfg.arrays.items()):
        if desc.storage == dtypes.StorageType.FPGA_Global:
            desc.location["bank"] = i % num_banks
            num_allocated[i % num_banks] = num_allocated[i % num_banks] + 1

    return num_allocated
