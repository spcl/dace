# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import dace
from dace.data import Property
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import ListProperty, make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.auto_tile.auto_tile_util import order_tiling_params
from dace.transformation.dataflow.tiling import MapTiling
from dace import dtypes
from dace import subsets


@make_properties
class ThreadCoarsening(transformation.SingleStateTransformation):
    """
    Thread coarsening means for GPU code-gen one thread does not comute 1 cell of the output, but
    a tile_sizes (dimensions go as x,y,z...) sub domain of the output.
    """

    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_group_map_entry = transformation.PatternNode(nodes.MapEntry)

    tile_sizes = ListProperty(
        element_type=int, desc="Computational domain per core / thread"
    )

    unroll = Property(dtype=bool, default=True, desc="Unroll the thread group map")
    unroll_mask = ListProperty(element_type=bool, default=None, desc="Which dimensions to unroll")

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.thread_group_map_entry, cls.device_map_entry)
        ]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        # Tmp variables used in the sequential map will have their dimensions multiplied by the "tile_sizes" dimensions
        inner_sequential_map_entry = None
        for out_edge in state.out_edges(self.thread_group_map_entry):
            u, _, v, _, _ = out_edge
            if isinstance(v, nodes.MapEntry):
                assert v.map.schedule == dtypes.ScheduleType.Sequential
                inner_sequential_map_entry = v
                break
        """
        if inner_sequential_map_entry is None:
            return False
        if inner_sequential_map_entry is not None:
            for in_edge in state.in_edges(inner_sequential_map_entry):
                u, _, _, _, _ = in_edge
                if not isinstance(u, nodes.AccessNode) and not isinstance(
                    u, nodes.Tasklet
                ):
                    return False
                if isinstance(u, nodes.Tasklet):
                    return False # Todo
                    # only assignment type of tasklets are allowed
                    if len(u.in_connectors) > 0:
                        return False
        """
        return MapTiling.can_be_applied(
            self, state, expr_index=expr_index, sdfg=sdfg, permissive=permissive
        )

    def apply(self, state: SDFGState, sdfg: SDFG):
        # When the ThreadBlock scheduled loop is tiled, then beg:end:1 becomes beg:end:tile_size
        # For GPU scheduling the thread block scheduled map needs to be scaled according to the tile_sizes
        # Furthermore the step of the device scheduled map needs to be increase too.
        # This can be handled by changing the range and the step of the thread block scheduled loop and increasing the step size of the parent

        dev_entry: nodes.MapEntry = self.device_map_entry
        thread_group_entry: nodes.MapEntry = self.thread_group_map_entry

        # If there are transient access nodes leading to the first inner sequential map the sizes of the arrays and the
        # subsets / volumes of the corresponding memlets need to be adapted

        used_dimensions = min(len(self.tile_sizes), len(thread_group_entry.map.params))
        tile_sizes = order_tiling_params(thread_group_entry.map.range, self.tile_sizes)

        MapTiling.apply_to(
            sdfg=sdfg,
            options=dict(
                prefix="d",
                skew=True,
                tile_sizes=tile_sizes,
                divides_evenly=True,
                tile_trivial=True,
            ),
            map_entry=thread_group_entry,
        )

        sequential_map_entry = thread_group_entry
        sequential_map_entry.map.schedule = dtypes.ScheduleType.Sequential
        sequential_map_entry.map.unroll = self.unroll
        sequential_map_entry.map.unroll_mask = self.unroll_mask
        sequential_map_entry.map.label = "ThreadCoarsenedMap"

        # Find the thread block map encapsulating the sequential map
        # Entry node of a map should be the map scope avoce
        thread_group_entry = state.entry_node(thread_group_entry)

        inner_sequential_map_entry = None
        for out_edge in state.out_edges(sequential_map_entry):
            u, _, v, _, _ = out_edge
            if isinstance(v, nodes.MapEntry):
                assert v.map.schedule == dtypes.ScheduleType.Sequential
                inner_sequential_map_entry = v
                break

        # Update the range if the inner sequential map
        sequential_map: nodes.Map = sequential_map_entry.map

        # Rm unnecessary copy-over edges
        """
        edges_to_remove = []
        for edge in state.out_edges(thread_group_entry):
            u, u_conn, v, v_conn, memlet = edge
            if u_conn == None and v_conn == None and memlet.data == None:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            state.remove_edge(edge)
        """

        # Move the access above the outer sequential map and update memlets for the map entry
        if inner_sequential_map_entry is not None:
            updated_arr_names = set()

            for in_edge in state.in_edges(inner_sequential_map_entry):
                u, _, _, _, _ = in_edge
                if isinstance(u, nodes.AccessNode):
                    access_node = u
                    # Current assumption outer seq map -> access node -> inner seq map
                    assert len(state.in_edges(access_node)) == 1
                    assert len(state.out_edges(access_node)) == 1
                    to_access_node_edge = state.in_edges(access_node)[0]
                    in_u, in_u_conn, in_v, in_v_conn, in_memlet = to_access_node_edge
                    access_node_to_inner_seq_edge = in_edge
                    out_u, out_u_conn, out_v, out_v_conn, out_memlet = (
                        access_node_to_inner_seq_edge
                    )
                    # The u can be an access node, in_u should be either a tasklet (assignment for the tmp storage)
                    # or or the sequential_map_entry
                    if isinstance(in_u, dace.nodes.Tasklet):
                        if len(u.in_connectors) > 0:
                            raise Exception(
                                "Case where thread coarsening is not applicable detected"
                            )
                        else:
                            raise Exception("Assignments to the temporary storage currently not supported")
                    if (not isinstance(in_u, dace.nodes.Tasklet)
                        and not isinstance(in_u, dace.nodes.AccessNode)
                        and in_u != sequential_map_entry):
                        raise Exception(
                            "Only access nodes and tasklets between the thread group and sequential map are allowed"
                        )

                    data = sdfg.arrays[u.data]
                    data_is_scalar = isinstance(data, dace.data.Scalar)
                    if not isinstance(data, dace.data.Scalar):
                        raise Exception(
                            f"Currently unsupported data type for thread coarsening: {type(data)}"
                            f"Array support will be implemented"
                        )

                    assert out_v == inner_sequential_map_entry

                    intermediate_node = None
                    intermediate_edge = None
                    # TODO: support for tasklet assignments to the temporary storage
                    #if isinstance(in_u, dace.nodes.Tasklet):
                    #    intermediate_node = in_u
                    #    assert len(state.in_edges(in_u)) == 1
                    #    intermediate_edge = state.in_edges(in_u)[0]

                    state.remove_edge(to_access_node_edge)
                    state.remove_edge(access_node_to_inner_seq_edge)

                    if intermediate_edge is not None:
                        state.remove_edge(intermediate_edge)

                    if data_is_scalar:
                        new_memlet_list = [
                            (0, end - 1, 1) for end in tile_sizes[-used_dimensions:]
                        ]
                        inner_offset_memlet_list = [
                            (dace.symbol(param) - beg, dace.symbol(param) - beg, 1)
                            for param, (beg, _, _) in zip(
                                sequential_map.params, sequential_map.range
                            )
                        ]
                    else:
                        raise Exception("TODO")

                    if in_memlet.data == None:
                        new_in_memlet = Memlet(subset=None, data=in_memlet.data)
                    else:
                        new_in_memlet = Memlet(
                            subset=subsets.Range(new_memlet_list),
                            data=in_memlet.data,
                            wcr=in_memlet.wcr,
                            wcr_nonatomic=in_memlet.wcr_nonatomic,
                            allow_oob=in_memlet.allow_oob,
                            debuginfo=in_memlet.debuginfo,
                        )
                    new_out_memlet = Memlet(
                        subset=subsets.Range(new_memlet_list),
                        data=out_memlet.data,
                        wcr=out_memlet.wcr,
                        wcr_nonatomic=out_memlet.wcr_nonatomic,
                        allow_oob=out_memlet.allow_oob,
                        debuginfo=out_memlet.debuginfo,
                    )
                    offseted_memlet = Memlet(
                        subset=subsets.Range(inner_offset_memlet_list),
                        data=out_memlet.data,
                        wcr=out_memlet.wcr,
                        wcr_nonatomic=out_memlet.wcr_nonatomic,
                        allow_oob=out_memlet.allow_oob,
                        debuginfo=out_memlet.debuginfo,
                    )


                    state.add_edge(
                        thread_group_entry,
                        in_u_conn,
                        access_node,
                        in_v_conn,
                        new_in_memlet,
                    )
                    state.add_edge(
                        access_node,
                        out_u_conn,
                        sequential_map_entry,
                        out_v_conn,
                        new_out_memlet,
                    )

                    # If data was first created it will be be (outconn) None -> access node (inconn) smth.
                    # In that case we copy as it is to the map above, but the new map needs both connectors
                    if intermediate_node is None and in_u_conn is not None:
                        thread_group_entry.add_out_connector(in_u_conn)
                    # If there is a tasklet before the connector needs to be None
                    assert out_v_conn != None
                    sequential_map_entry.add_in_connector(out_v_conn)

                    in_conn_for_inner_seq = out_v_conn[:]
                    out_conn_for_seq = "OUT_" + out_v_conn[3:]

                    sequential_map_entry.add_out_connector(out_conn_for_seq)
                    inner_sequential_map_entry.add_in_connector(in_conn_for_inner_seq)

                    state.add_edge(
                        sequential_map_entry,
                        out_conn_for_seq,
                        inner_sequential_map_entry,
                        in_conn_for_inner_seq,
                        offseted_memlet,
                    )
                    updated_arr_names.add(out_memlet.data)

                    # It was scalar before convert to array
                    data_type = sdfg.arrays[out_memlet.data].dtype
                    assert (
                        sdfg.arrays[out_memlet.data].storage
                        == dtypes.StorageType.Register or
                        sdfg.arrays[out_memlet.data].storage
                        == dtypes.StorageType.Default
                    )
                    sdfg.remove_data(out_memlet.data, validate=False)
                    sdfg.add_array(
                        name=out_memlet.data,
                        shape=tile_sizes[-used_dimensions:],
                        storage=dtypes.StorageType.Register,
                        dtype=data_type,
                        transient=True,
                        alignment=16,
                    )

            # Now update remaining memlets, accessing temporary scalars
            data_to_check = set(updated_arr_names)
            edges_to_check = set(state.out_edges(inner_sequential_map_entry))
            while len(edges_to_check) > 0:
                edge = edges_to_check.pop()
                u, u_conn, v, v_conn, memlet = edge
                if memlet.data != None and memlet.data in data_to_check:
                    offseted_memlet = Memlet(
                        subset=subsets.Range(inner_offset_memlet_list),
                        data=memlet.data,
                        wcr=memlet.wcr,
                        wcr_nonatomic=memlet.wcr_nonatomic,
                        allow_oob=memlet.allow_oob,
                        debuginfo=memlet.debuginfo,
                    )
                    state.remove_edge(edge)
                    state.add_edge(u, u_conn, v, v_conn, offseted_memlet)
                if not (
                    isinstance(v, nodes.MapExit)
                    and v == state.exit_node(sequential_map_entry)
                ):
                    edges_to_check = edges_to_check.union(state.out_edges(v))

        # Map Tiling does not update the range as we need them
        # Update the threadblock and device ranges
        dev_updated_range_list = [
            (beg, end, step * tstep)
            for (beg, end, step), (_, _, tstep) in zip(
                dev_entry.map.range, thread_group_entry.map.range
            )
        ]
        dev_entry.map.range = subsets.Range(dev_updated_range_list)
        thread_block_updated_range_list = [
            (beg, (end + 1) * step - 1, step)
            for (beg, end, step) in thread_group_entry.map.range
        ]
        thread_group_entry.map.range = subsets.Range(thread_block_updated_range_list)

        for m in [dev_entry.map, thread_group_entry.map, sequential_map_entry.map]:
            d = dict()
            for param in m.params:
                d[param] = dtypes.typeclass("intc")
            m.param_types = d

    @staticmethod
    def annotates_memlets():
        return False
