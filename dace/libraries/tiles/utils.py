import copy
import dace

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_connector_info(node, state, sdfg, conn_name: str, is_input: bool):
    """
    Return *(edge, outer_array, shape_2d, strides_2d)* for the named connector.

    Raises
    ------
    ValueError
        If the connector is not found or the memlet does not describe a 2-D
        region after squeezing singleton dimensions.
    """
    edges = state.in_edges(node) if is_input else state.out_edges(node)
    conn_attr = "dst_conn" if is_input else "src_conn"

    for edge in edges:
        if getattr(edge, conn_attr) == conn_name:
            subset = copy.deepcopy(edge.data.subset)
            idx = subset.squeeze()          # removes size-1 dims, returns kept indices
            size = subset.size()

            if len(size) != 2:
                raise ValueError(
                    f"Connector '{conn_name}' must reference a 2-D region "
                    f"(got {len(size)}-D after squeezing singleton dimensions)."
                )

            find_fn = (dace.sdfg.find_input_arraynode if is_input
                       else dace.sdfg.find_output_arraynode)
            outer_array = sdfg.data(find_fn(state, edge).data)

            strides = (outer_array.strides[idx[0]], outer_array.strides[idx[1]])
            return edge, outer_array, (size[0], size[1]), strides

    direction = "input" if is_input else "output"
    raise ValueError(f"{direction.capitalize()} connector '{conn_name}' not found.")

