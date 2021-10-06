import dace
import aenum
from dace import dtypes, nodes
import warnings
from dace.registry import extensible_enum, undefined_safe_enum


@undefined_safe_enum
@extensible_enum
class NcclReductionType(aenum.AutoNumberEnum):
    """ Reduction types supported by NCCL. """
    ncclSum = ()  #: Sum
    ncclProd = ()  #: Product
    ncclMin = ()  #: Minimum value
    ncclMax = ()  #: Maximum value


NCCL_SUPPORTED_OPERATIONS = {
    None: NcclReductionType.ncclSum,
    dtypes.ReductionType.Sum: NcclReductionType.ncclSum,
    dtypes.ReductionType.Product: NcclReductionType.ncclProd,
    dtypes.ReductionType.Min: NcclReductionType.ncclMin,
    dtypes.ReductionType.Max: NcclReductionType.ncclMax
}


def aggregate_calls(sdfg: dace.SDFG, state: dace.SDFGState,
                    lib_node: nodes.LibraryNode, code: str):
    group_handle_conn = '_group_handle'
    sync = True
    if group_handle_conn in lib_node.in_connectors:
        sync = False
        for edge in state.in_edges(lib_node):
            if edge.dst_conn == group_handle_conn:
                in_gh_edge = edge
                in_gh_node = edge.src
        if not state.predecessors(in_gh_node):
            code = """ncclGroupStart();\n""" + code
        else:
            predecessor_node = state.predecessors(in_gh_node)[0]
            state.add_edge(predecessor_node, None, lib_node, None,
                           dace.Memlet())
            state.remove_edge_and_connectors(state.in_edges(in_gh_node)[0])
        state.remove_edge_and_connectors(in_gh_edge)
        lib_node.remove_in_connector(group_handle_conn)
        state.remove_node(in_gh_node)

    if group_handle_conn in lib_node.out_connectors:
        for edge in state.out_edges(lib_node):
            if edge.src_conn == group_handle_conn:
                out_gh_edge = edge
                out_gh_node = edge.dst
        if not state.successors(out_gh_node):
            code += """ncclGroupEnd();"""
            sync = True
            out_gh_data = out_gh_node.data
            state.remove_edge_and_connectors(out_gh_edge)
            state.remove_node(out_gh_node)
            try:
                sdfg.remove_data(out_gh_data)
            except ValueError as ex:
                warnings.warn(str(ex))
        lib_node.remove_out_connector(group_handle_conn)
    # if sync:
    #     code += """\ncudaStreamSynchronize(__dace_current_stream);"""
    return code


def Nccl_dtypes(dtype):
    nccl_dtype_str = ""
    if dtype == dace.dtypes.float16:
        nccl_dtype_str = "ncclFloat16"
    elif dtype == dace.dtypes.float32:
        nccl_dtype_str = "ncclFloat32"
    elif dtype == dace.dtypes.float64:
        nccl_dtype_str = "ncclFloat64"
    elif dtype == dace.dtypes.int8:
        nccl_dtype_str = "ncclInt8"
    elif dtype == dace.dtypes.int32:
        nccl_dtype_str = "ncclInt32"
    elif dtype == dace.dtypes.int32:
        nccl_dtype_str = "ncclInt32"
    elif dtype == dace.dtypes.int64:
        nccl_dtype_str = "ncclInt64"
    elif dtype == dace.dtypes.uint8:
        nccl_dtype_str = "ncclUint8"
    elif dtype == dace.dtypes.uint32:
        nccl_dtype_str = "ncclUint32"
    elif dtype == dace.dtypes.uint64:
        nccl_dtype_str = "ncclUint64"
    else:
        raise ValueError("DDT of " + str(dtype) + " not supported yet.")
    return nccl_dtype_str
