# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse a Map into the partial Reduce that consumes it as an outer Map over the kept dimensions and an inner reduction.

``MapReduceFusion`` turns ``Map[i, j, k] -> X -> Reduce(axis k) -> T`` into one Map whose WCR accumulates into ``T``,
initialized to the identity beforehand. ``T`` then stays a full-size array even when the next Map reads it elementwise
over ``i, j``: a WCR intermediate does not fuse vertically. This transformation produces instead ::

    Map[i, j] { acc = identity; Map[k] { acc (+)= ... }; T[i, j] = acc }

``T`` is now written once per ``(i, j)``, so the consumer fuses and ``T`` shrinks to a scalar. The kept dimensions stay
parallel and the reduced ones stay a WCR, so the nest is as parallel as the consumer it is about to fuse with. It
applies only when such a consumer exists; otherwise ``MapReduceFusion`` keeps the reduction dimensions parallel too.
"""

import copy

from dace import dtypes, nodes, subsets
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.transformation import helpers
from dace.transformation.dataflow.mapreduce import MapReduceFusion


class MapPartialReduceFusion(MapReduceFusion):
    """Fuse ``Map -> X -> Reduce(some axes) -> T -> Map over the kept axes`` into a nest the consumer fuses with."""

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        if expr_index != 0 or not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        reduce_node, in_array, out_array, tmap_exit = self.reduce, self.in_array, self.out_array, self.tmap_exit
        ndim = len(sdfg.arrays[in_array.data].shape)
        axes = reduce_node.axes
        if reduce_node.identity is None or axes is None or not 0 < len(axes) < ndim:
            return False
        # The Map may write other data too: the split leaves those writes in the inner Map, unchanged. Not another
        # reduction (a WCR write, or data a Reduce reads): ``MapReduceFusion`` keeps those in the full Map instead.
        if len(graph.edges_between(tmap_exit, in_array)) != 1 or not sdfg.arrays[out_array.data].transient:
            return False
        if any(
            e.data.wcr is not None or any(isinstance(oe.dst, type(reduce_node)) for oe in graph.out_edges(e.dst))
            for e in graph.out_edges(tmap_exit)
            if e.dst is not in_array
        ):
            return False
        params = self.dim_params(graph)
        if params is None:
            return False
        kept_ranges = sorted(str(r) for d, r in enumerate(graph.in_edges(reduce_node)[0].data.subset) if d not in axes)
        return any(
            isinstance(e.dst, nodes.MapEntry) and sorted(str(r) for r in e.dst.map.range) == kept_ranges
            for e in graph.out_edges(out_array)
        )

    def dim_params(self, graph: SDFGState) -> list | None:
        """The map parameter each dimension of the intermediate is written at, if every one is exactly one parameter."""
        tmap_exit, in_array = self.tmap_exit, self.in_array
        (write,) = [e for e in graph.in_edges(tmap_exit) if e.data.data == in_array.data] or [None]
        if write is None or write.data.wcr is not None:
            return None
        params = tmap_exit.map.params
        dims = []
        for begin, end, step in write.data.subset:
            if begin != end or step != 1 or str(begin) not in params:
                return None
            dims.append(begin)
        return dims if sorted(map(str, dims)) == sorted(params) else None

    def apply(self, graph: SDFGState, sdfg: SDFG):
        tmap_exit, in_array, reduce_node, out_array = self.tmap_exit, self.in_array, self.reduce, self.out_array
        axes = reduce_node.axes
        dims = self.dim_params(graph)
        params = list(tmap_exit.map.params)
        kept = [params.index(str(p)) for d, p in enumerate(dims) if d not in axes]
        out_edge = graph.out_edges(reduce_node)[0]
        reduce_in = graph.in_edges(reduce_node)[0].data.subset
        in_desc = sdfg.arrays[in_array.data]

        outer_entry, inner_entry = helpers.extract_map_dims(sdfg, graph.entry_node(tmap_exit), kept)
        outer_exit, inner_exit = graph.exit_node(outer_entry), graph.exit_node(inner_entry)
        inner_entry.map.schedule = dtypes.ScheduleType.Sequential

        acc, acc_desc = sdfg.add_scalar(
            sdfg.find_new_name_avoiding_connectors(f"{in_array.data}_acc"),
            in_desc.dtype,
            transient=True,
            storage=dtypes.StorageType.Register,
            find_new_name=True,
        )
        (write,) = [e for e in graph.in_edges(inner_exit) if e.data.data == in_array.data]
        write.data = Memlet(data=acc, subset="0", wcr=reduce_node.wcr)
        (passed,) = [e for e in graph.in_edges(outer_exit) if e.src is inner_exit and e.data.data == in_array.data]
        accumulated = graph.add_access(acc)
        graph.remove_edge(passed)
        graph.add_edge(
            inner_exit, passed.src_conn, accumulated, None, Memlet(data=acc, subset="0", wcr=reduce_node.wcr)
        )

        # The kept dimensions index the intermediate; the output may write them at an offset (``T[4*b + i]``).
        kept_dims = [d for d in range(len(dims)) if d not in axes]
        index = [
            (o[0] + dims[d] - reduce_in[d][0],) * 2 + (1,) for o, d in zip(out_edge.data.subset, kept_dims, strict=True)
        ]
        graph.add_edge(
            accumulated,
            None,
            outer_exit,
            passed.dst_conn,
            Memlet(data=out_edge.data.data, subset=subsets.Range(index), other_subset="0"),
        )
        (leaving,) = [e for e in graph.out_edges(outer_exit) if e.src_conn == "OUT_" + passed.dst_conn[3:]]
        graph.add_edge(outer_exit, leaving.src_conn, out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
        graph.remove_nodes_from([in_array, reduce_node])
        if not any(isinstance(n, nodes.AccessNode) and n.data == in_array.data for n in graph.nodes()):
            sdfg.remove_data(in_array.data, validate=False)

        init = graph.add_tasklet("partial_reduce_init", {}, {"__out": None}, f"__out = {reduce_node.identity}")
        seeded = graph.add_access(acc)
        graph.add_nedge(outer_entry, init, Memlet())
        graph.add_edge(init, "__out", seeded, None, Memlet(data=acc, subset="0"))
        graph.add_nedge(seeded, inner_entry, Memlet())


class MapReduceFusionUnlessPartial(MapReduceFusion):
    """``MapReduceFusion`` that leaves every reduction ``MapPartialReduceFusion`` takes, so the Maps producing its operand
    can fuse into the Map feeding it first: split earlier, that Map no longer matches their iteration space."""

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        partial = MapPartialReduceFusion()
        partial.setup_match(sdfg, self.cfg_id, self.state_id, self.subgraph, expr_index, override=True)
        return not partial.can_be_applied(graph, expr_index, sdfg, permissive)
