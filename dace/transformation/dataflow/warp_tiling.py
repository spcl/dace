# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np
from dace import properties, nodes, dtypes, subsets, symbolic
from dace import Memlet, SDFG, SDFGState
from dace.frontend.operations import detect_reduction_type
from dace.transformation import transformation as xf, helpers as xfh
from dace.sdfg import utils as sdutil
from dace.libraries.standard.block_reduce import block_allreduce_code, block_redop


def lane_identity_literal(dtype: dtypes.typeclass, identity) -> str:
    """``identity`` as a C++ literal of ``dtype``; integers stay integers (a 64-bit extreme through
    ``float`` rounds out of range)."""
    if np.issubdtype(dtype.type, np.integer):
        return f"{dtype.ctype}({int(identity)})"
    return f"{dtype.ctype}({float(identity)!r})"


def bind_lane_symbol(inner: SDFG, lane_sdfg: SDFG) -> None:
    """Map ``__tid`` into ``inner`` and every nested SDFG between it and ``lane_sdfg``, which holds the lane map."""
    while inner is not lane_sdfg and inner.parent_nsdfg_node is not None:
        inner.parent_nsdfg_node.symbol_mapping["__tid"] = symbolic.pystr_to_symbolic("__tid")
        if "__tid" not in inner.symbols:
            inner.add_symbol("__tid", dtypes.int32)
        inner = inner.parent_sdfg


def seed_lane_partial(state: SDFGState, inner_map: nodes.MapEntry, name: str, literal: str) -> None:
    """Set ``name`` to ``literal`` in the scope of ``inner_map``, ordered before it."""
    seed = state.add_tasklet("lane_partial_seed", {}, {"__out"}, f"__out = {literal};", dtypes.Language.CPP)
    parent = state.entry_node(inner_map)
    if parent is not None:
        state.add_nedge(parent, seed, Memlet())
    write = state.add_write(name)
    state.add_edge(seed, "__out", write, None, Memlet(name))
    state.add_nedge(write, inner_map, Memlet())


def accumulator_source(state: SDFGState, inner_map: nodes.MapEntry, data: str) -> nodes.AccessNode:
    """The access node holding ``data`` as ``inner_map`` starts: the one feeding the map, else a fresh read
    (the value an earlier state left)."""
    for edge in state.in_edges(inner_map):
        if isinstance(edge.src, nodes.AccessNode) and edge.src.data == data:
            return edge.src
    return state.add_read(data)


@properties.make_properties
class WarpTiling(xf.SingleStateTransformation):
    """
    Implements a GPU specialization tiling that takes a GPU kernel map (with
    nested maps, but without explicit block sizes) and divides its work across
    a warp. Specifically, it tiles its contents by a configurable warp size
    (default: 32), and optionally preferring recomputation (map replication)
    over local storage within the kernel. If write-conflicted reductions happen
    within the given map, the transformation adds warp reductions to the tiles.
    """

    warp_size = properties.Property(dtype=int, default=32, category="Scheduling", desc="Hardware warp size")
    replicate_maps = properties.Property(
        dtype=bool,
        default=True,
        category="Parameters",
        desc="Replicate tiled maps that lead to multiple other tiled maps",
    )

    mapentry = xf.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.mapentry)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg: SDFG, permissive) -> bool:
        me = self.mapentry

        if len(xfh.get_internal_scopes(graph, me, immediate=True)) == 0:
            return False

        # GPU map that has no predefined thread-block maps
        return me.schedule == dtypes.ScheduleType.GPU_Device and not xfh.gpu_map_has_explicit_threadblocks(graph, me)

    def apply(self, graph: SDFGState, sdfg: SDFG) -> nodes.MapEntry:
        me = self.mapentry

        # Add new map within map
        mx = graph.exit_node(me)
        new_me, new_mx = graph.add_map(
            "warp_tile", dict(__tid=f"0:{self.warp_size}"), dtypes.ScheduleType.GPU_ThreadBlock
        )
        __tid = symbolic.pystr_to_symbolic("__tid")
        for e in graph.out_edges(me):
            xfh.reconnect_edge_through_map(graph, e, new_me, True)
        for e in graph.in_edges(mx):
            xfh.reconnect_edge_through_map(graph, e, new_mx, False)

        # Stride and offset all internal maps
        maps_to_stride = xfh.get_internal_scopes(graph, new_me, immediate=True)
        for nstate, nmap in maps_to_stride:
            # Skip sequential maps
            if nmap.schedule == dtypes.ScheduleType.Sequential:
                continue

            nsdfg = nstate.parent
            nsdfg_node = nsdfg.parent_nsdfg_node

            # Map cannot be partitioned across a warp
            if (nmap.range.size()[-1] < self.warp_size) == True:
                continue

            bind_lane_symbol(nsdfg, sdfg)
            # Lane ``__tid`` starts ``__tid`` steps in. Shifting the start, not the end, keeps an unsigned
            # bound from underflowing (``stop - start - __tid``).
            begin, end, step = nmap.range[-1]
            nmap.range[-1] = (begin + step * __tid, end, step * self.warp_size)
            subgraph = nstate.scope_subgraph(nmap)
            inner_map_exit = nstate.exit_node(nmap)
            # If requested, replicate maps with multiple dependent maps
            if self.replicate_maps:
                destinations = [nstate.memlet_path(edge)[-1].dst for edge in nstate.out_edges(inner_map_exit)]

                for dst in destinations:
                    # Transformation will not replicate map with more than one
                    # output
                    if len(destinations) != 1:
                        break
                    if not isinstance(dst, nodes.AccessNode):
                        continue  # Not leading to access node
                    if not xfh.contained_in(nstate, dst, new_me):
                        continue  # Memlet path goes out of map
                    if not nsdfg.arrays[dst.data].transient:
                        continue  # Cannot modify non-transients
                    for edge in nstate.out_edges(dst)[1:]:
                        rep_subgraph = xfh.replicate_scope(nsdfg, nstate, subgraph)
                        rep_edge = nstate.out_edges(rep_subgraph.sink_nodes()[0])[0]
                        # Add copy of data
                        newdesc = copy.deepcopy(sdfg.arrays[dst.data])
                        newname = nsdfg.add_datadesc(dst.data, newdesc, find_new_name=True)
                        newaccess = nstate.add_access(newname)
                        # Redirect edges
                        xfh.redirect_edge(nstate, rep_edge, new_dst=newaccess, new_data=newname)
                        xfh.redirect_edge(nstate, edge, new_src=newaccess, new_data=newname)

            # If has WCR, add warp-collaborative reduction on outputs
            for out_edge in nstate.out_edges(inner_map_exit):
                dst = nstate.memlet_path(out_edge)[-1].dst
                if not xfh.contained_in(nstate, dst, new_me):
                    # Skip edges going out of map
                    continue
                if dst.desc(nsdfg).storage == dtypes.StorageType.GPU_Global:
                    # Skip shared memory
                    continue
                if out_edge.data.wcr is not None:
                    ctype = nsdfg.arrays[out_edge.data.data].dtype.ctype
                    redtype = detect_reduction_type(out_edge.data.wcr)
                    if redtype == dtypes.ReductionType.Custom:
                        raise NotImplementedError
                    credtype = "dace::ReductionType::" + str(redtype)[str(redtype).find(".") + 1 :]

                    # One element: each lane folds its strided share into a private partial that
                    # starts at the op's IDENTITY (starting it at the accumulator's value would count
                    # that value once per lane), then every lane folds the lanes' total into its copy.
                    if out_edge.data.subset.num_elements() == 1:
                        acc_desc = nsdfg.arrays[out_edge.data.data]
                        identity = dtypes.reduction_identity(acc_desc.dtype, redtype)
                        if identity is None:
                            continue
                        name = nsdfg._find_new_name(out_edge.data.data)
                        nsdfg.add_scalar(name, acc_desc.dtype, transient=True)

                        seed_lane_partial(nstate, nmap, name, lane_identity_literal(acc_desc.dtype, identity))

                        newnode = nstate.add_access(name)
                        nstate.remove_edge(out_edge)
                        edge = nstate.add_edge(
                            out_edge.src, out_edge.src_conn, newnode, None, copy.deepcopy(out_edge.data)
                        )
                        for e in nstate.memlet_path(edge):
                            e.data.data = name
                            e.data.subset = subsets.Range([(0, 0, 1)])

                        functor = block_redop(redtype, ctype)
                        if self.warp_size == 32:
                            code = f"__out = {functor}(__acc, dace::warpReduce<{credtype}, {ctype}>::reduce(__a));"
                        else:
                            total = f"__lanes_{name}"
                            code = (
                                f"{ctype} {total};\n"
                                + block_allreduce_code(name, ctype, self.warp_size, "__a", functor, total)
                                + f"\n__out = {functor}(__acc, {total});"
                            )
                        wrt = nstate.add_tasklet("lanereduce", {"__a", "__acc"}, {"__out"}, code, dtypes.Language.CPP)
                        nstate.add_edge(newnode, None, wrt, "__a", Memlet(name))
                        acc_memlet = copy.deepcopy(out_edge.data)
                        acc_memlet.wcr = None
                        nstate.add_edge(
                            accumulator_source(nstate, nmap, out_edge.data.data), None, wrt, "__acc", acc_memlet
                        )
                        out_edge.data.wcr = None
                        nstate.add_edge(wrt, "__out", out_edge.dst, None, out_edge.data)
                    else:  # More than one element: mapped tasklet
                        # Could be a parallel summation
                        # TODO(later): Check if reduction
                        continue
            # End of WCR to warp reduction

        # Make nested SDFG out of new scope
        xfh.nest_state_subgraph(sdfg, graph, graph.scope_subgraph(new_me, False, False))

        return new_me
