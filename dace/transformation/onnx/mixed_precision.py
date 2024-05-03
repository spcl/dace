import dace
from dace import nodes, dtypes


def enlarge_reduction_accumulators(target_sdfg: dace.SDFG):
    """
    Enlarge fp16 accumulators to fp32.
    """
    # dtype of accumulator should have at least single precision
    for n, s in target_sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.AccessNode):
            if any(e.data.wcr for e in s.in_edges(n)):
                dst_array = s.parent.arrays[n.data]
                if dst_array.dtype == dace.dtypes.float16:
                    dst_array.dtype = dace.dtypes.float32

    # propagate datatype to outer sdfgs

    for state, sdfg in target_sdfg.all_nodes_recursive():
        if not isinstance(sdfg, dace.SDFG):
            continue
        if sdfg.parent is None:
            continue
        for data, arr in sdfg.arrays.items():
            pnode = sdfg.parent_nsdfg_node
            pstate = sdfg.parent
            psdfg = sdfg.parent_sdfg
            if data not in pnode.out_connectors:
                continue
            for edge in pstate.out_edges(pnode):
                if not isinstance(edge.dst, dace.nodes.AccessNode):
                    continue
                if edge.src_conn != data:
                    continue
                psdfg.arrays[edge.dst.data].dtype = arr.dtype
