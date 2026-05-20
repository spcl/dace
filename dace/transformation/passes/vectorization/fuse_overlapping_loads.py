# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that fuses overlapping per-lane loads into one shared union access."""
import copy
import dace
from typing import Dict, List, Optional, Tuple
from dace import SDFG, properties
from dace.memlet import Memlet
from dace.sdfg.graph import Edge
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class FuseOverlappingLoads(ppl.Pass):
    """Fuse overlapping per-lane loads of the same array into one shared union access."""

    # This pass is tested as part of the vectorization pipeline.
    CATEGORY: str = 'Vectorization'

    fusion_threshold = properties.Property(
        dtype=int,
        default=1,
        desc="Minimum overlap count *above* which a per-array load fan is fused into a shared union "
        "window. A source array with ``> fusion_threshold`` overlapping incoming access nodes on a "
        "map entry is fused; ``<= fusion_threshold`` is left as the per-subset shape. Default ``1`` "
        "preserves the original behaviour (fuse iff ``>= 2`` overlapping reads). Raise to suppress "
        "fusion on small overlaps (e.g. a 2-point pair); lower to ``0`` to attempt fusion on every "
        "non-empty group (rarely useful — a single read has no overlap to fuse).")

    def __init__(self, fusion_threshold: int = 1):
        super().__init__()
        self.fusion_threshold = fusion_threshold
        # Instrumentation counters set during ``apply_pass``: how many
        # per-array load fans the pass GATED (skipped because
        # ``len(v) <= fusion_threshold``) vs FUSED (collapsed into a
        # shared union window). These expose what the pass actually did,
        # which is the non-fragile way to integration-test the threshold
        # (counting fusion-product nodes post-hoc is implementation-
        # fragile because the pass refactors topology). Reset on each
        # ``apply_pass``.
        self._last_groups_gated = 0
        self._last_groups_fused = 0

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _collect_loads(
            self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry
    ) -> Dict[str, List[Tuple[Edge[Memlet], dace.nodes.AccessNode, Edge[Memlet]]]]:
        """Group the incoming access nodes of ``map_entry`` by their source array.

        :param state: State containing ``map_entry``.
        :param map_entry: Inner map entry whose load fan-out is inspected.
        :returns: Map from source array name to ``(src_in_edge, access_node,
            map_in_edge)`` tuples.
        """
        # The supported pattern is:
        # Map Entry -> [AccessNodes...] -> MapEntry
        # Where we have:
        # Src -(Arr1)> AccessNode -(Arr1_0)> MapEntry
        # Src -(Arr1)> AccessNode -(Arr1_0)> MapEntry
        # Src -(Arr1)> AccessNode -(Arr1_0)> MapEntry
        # Src -(Arr1)> AccessNode -(Arr1_0)> MapEntry
        # Src -(Arr1)> AccessNode -(Arr1_0)> MapEntry
        # We will connect access nodes through:
        # Src -(Arr1)(UnionSubset)-> AccessNode0 -> -(Arr1_0)> AccessNode -> MapEntry
        #                                        -> -(Arr1_0)> AccessNode -> MapEntry
        #                                        -> -(Arr1_0)> AccessNode -> MapEntry
        #                                        -> -(Arr1_0)> AccessNode -> MapEntry
        #                                        -> -(Arr1_0)> AccessNode -> MapEntry

        in_edges = state.in_edges(map_entry)
        # Get incoming access nodes
        in_src_edges = {ie for ie in in_edges if isinstance(ie.src, dace.nodes.AccessNode)}

        # Generate map of access node and data
        in_data_and_srcs = dict()
        for in_edge in in_src_edges:
            in_src = in_edge.src
            # If other subset is not None continue
            # This could be done by handling src and dst subsets
            # while also calling memlet path but I do not want to
            # implement it
            if in_edge.data.other_subset is not None:
                continue
            src_node_in_edges = state.in_edges(in_src)
            # If length is not 1 continue
            if len(src_node_in_edges) != 1:
                continue
            for src_in_edge in src_node_in_edges:
                # If other subset is not None continue
                # This could be done by handling src and dst subsets
                # while also calling memlet path but I do not want to
                # implement it
                if src_in_edge.data.other_subset is not None:
                    continue
                if src_in_edge.data.data not in in_data_and_srcs:
                    in_data_and_srcs[src_in_edge.data.data] = list()
                in_data_and_srcs[src_in_edge.data.data].append((src_in_edge, in_src, in_edge))

        return in_data_and_srcs

    def compute_union_and_offsets(
        self, in_data_and_srcs: Dict[str, List[Tuple[Edge[Memlet], dace.nodes.AccessNode, Edge[Memlet]]]]
    ) -> Dict[str, dace.subsets.Range]:
        """Compute the union subset and shape for each grouped source array.

        :param in_data_and_srcs: Output of :meth:`_collect_loads`.
        :returns: Map from array name to ``(union_subset, union_shape)``.
        """
        union_subsets = dict()
        for k, v in in_data_and_srcs.items():
            union_subset: dace.subsets.Range = v[0][0].data.subset
            for in_edge, an, out_edge in v[1:]:
                union_subset = union_subset.union(in_edge.data.subset)
            # int_floor (NOT sympy //) so a strided shape (s>1) emits a
            # correct C++ integer extent, not the broken floor(x-1/s).
            union_shape = [dace.symbolic.int_floor((e + 1) - b, s) for (b, e, s) in union_subset]
            union_subsets[k] = (union_subset, union_shape)
        return union_subsets

    def _apply(self, sdfg: dace.SDFG):
        """Fuse overlapping loads in ``sdfg`` and recurse into nested SDFGs.

        :param sdfg: SDFG to transform in place.
        """
        for state in sdfg.all_states():
            sdict = state.scope_dict()
            for node in state.nodes():
                parent_node = sdict[node]
                if (isinstance(node, dace.nodes.MapEntry) and parent_node is not None
                        and isinstance(parent_node, dace.nodes.MapEntry)):
                    # Load overlapping nodes
                    in_data_and_srcs = self._collect_loads(state, node)
                    union_subsets = self.compute_union_and_offsets(in_data_and_srcs)

                    # 1. Remove all in edges to the src node
                    # 2. Add an intermediate access node
                    # 3. Add new edges (need to offset the union subset from individual)
                    for k, v in in_data_and_srcs.items():
                        # Skip groups at or below the fusion threshold:
                        # the default (1) preserves "fuse iff >= 2 reads";
                        # raise the threshold to suppress fusion on small
                        # overlaps without disabling the pass entirely.
                        if len(v) <= self.fusion_threshold:
                            self._last_groups_gated += 1
                            continue
                        self._last_groups_fused += 1

                        # Collect out connectors to remove
                        out_connectors = dict()
                        for vitem in v:
                            out_connectors[vitem[0].src_conn] = vitem[0].src

                        # Keep on for the new edge
                        out_conn_to_use, src_node = out_connectors.popitem()
                        for outc, src in out_connectors.items():
                            # Remove the memlet path going up until the parent for all the other paths
                            # we have decided to remove
                            in_edges = set(state.in_edges_by_connector(src, outc.replace("OUT_", "IN_")))
                            assert len(in_edges) == 1
                            in_edge = in_edges.pop()
                            edges = state.memlet_path(in_edge)
                            src.remove_out_connector(outc)
                            ssrc = edges[0].src
                            sdst = edges[0].dst
                            for e in edges:
                                state.remove_edge(e)
                                if e.src_conn is not None:
                                    e.src.remove_out_connector(e.src_conn)
                                if e.dst_conn is not None:
                                    e.dst.remove_in_connector(e.dst_conn)
                                if e == in_edge:
                                    break
                            if state.degree(ssrc) == 0:
                                state.remove_node(ssrc)
                            elif state.out_degree(ssrc) == 0:
                                for src_in_edge in state.in_edges(ssrc):
                                    state.add_edge(src_in_edge.src, None, sdst, None, dace.memlet.Memlet(None))
                                state.remove_node(ssrc)

                        for vitem in v:
                            state.remove_edge(vitem[0])

                        # Create new array for union subset
                        # Get the first access node, properties except the
                        # needed shape should be similar to needed
                        src_desc = state.sdfg.arrays[v[0][1].data]
                        copy_src_desc: dace.data.Array = copy.deepcopy(src_desc)
                        collapsed_shape = [dim for dim in union_subsets[k][1] if dim != 1]
                        copy_src_desc.set_shape(new_shape=collapsed_shape)
                        arr_name = state.sdfg.add_datadesc(name=f"{k}_union",
                                                           datadesc=copy_src_desc,
                                                           find_new_name=True)
                        new_access_node = state.add_access(arr_name)

                        state.add_edge(src_node, out_conn_to_use, new_access_node, None,
                                       dace.memlet.Memlet(data=k, subset=union_subsets[k][0]))

                        # Add from this access node to all other access node
                        for vitem in v:
                            original_copy_in_subset: dace.subsets.Range = vitem[0].data.subset
                            offset_subset_range: dace.subsets.Range = union_subsets[k][0]
                            offset_subset: dace.subsets.Range = dace.subsets.Range([
                                (b, b, 1) for (b, e, s) in offset_subset_range
                            ])
                            union_set = union_subsets[k][1]
                            offsetted_range = original_copy_in_subset.offset_new(offset_subset, True)
                            # Need to not include if dimension of the parent thing is 1
                            assert len(offsetted_range) == len(union_set)
                            collapsed_offsetted_range = dace.subsets.Range([
                                (b, e, s) for (b, e, s), dim in zip(offsetted_range, union_set)
                                if not (b == 0 and e == 0 and s == 1 and dim == 1)
                            ])
                            state.add_edge(new_access_node, None, vitem[1], None,
                                           dace.memlet.Memlet(
                                               data=arr_name,
                                               subset=collapsed_offsetted_range,
                                           ))

        state.sdfg.validate()

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._apply(node.sdfg)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Run the pass over ``sdfg``.

        :param sdfg: SDFG to transform in place.
        """
        # Reset instrumentation counters at the entry — they accumulate
        # only what THIS apply did, not what prior runs did. Inspected
        # by integration tests that gate ``fusion_threshold`` behavior.
        self._last_groups_gated = 0
        self._last_groups_fused = 0
        self._apply(sdfg)
