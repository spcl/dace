# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the access deduplication transformation. """

from collections import defaultdict
import copy
import itertools
from typing import List, Set

from dace import data, dtypes, sdfg as sd, subsets, symbolic, registry
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf
import dace.transformation.helpers as helpers

import warnings


@registry.autoregister_params(singlestate=True)
class DeduplicateAccess(xf.Transformation):
    """ 
    This transformation takes a node that is connected to multiple destinations
    with overlapping memlets, and consolidates those accesses through a 
    transient array or scalar.
    """

    _map_entry = nodes.MapEntry(nodes.Map('_', [], []))
    _node1 = nodes.Node()
    _node2 = nodes.Node()

    @staticmethod
    def expressions():
        state = sd.SDFGState()
        state.add_nedge(DeduplicateAccess._map_entry, DeduplicateAccess._node1,
                        Memlet())
        state.add_nedge(DeduplicateAccess._map_entry, DeduplicateAccess._node2,
                        Memlet())
        return [state]

    @staticmethod
    def can_be_applied(graph: sd.SDFGState,
                       candidate,
                       expr_index,
                       sdfg,
                       permissive=False):
        map_entry = graph.node(candidate[DeduplicateAccess._map_entry])
        nid1 = candidate[DeduplicateAccess._node1]
        node1 = graph.node(nid1)
        nid2 = candidate[DeduplicateAccess._node2]
        node2 = graph.node(nid2)

        # Two nodes must be ordered (avoid duplicates/nondeterminism)
        if nid1 >= nid2:
            return False

        # Two nodes must belong to same connector
        edges1 = set(e.src_conn for e in graph.edges_between(map_entry, node1))
        edges2 = set(e.src_conn for e in graph.edges_between(map_entry, node2))
        if len(edges1 & edges2) == 0:
            return False

        # For each common connector
        for conn in (edges1 & edges2):
            # Deduplication: Only apply to first pair of edges
            node_ids = [
                graph.node_id(e.dst) for e in graph.out_edges(map_entry)
                if e.src_conn == conn
            ]
            if any(nid < nid1 for nid in node_ids):
                return False
            if any(nid < nid2 for nid in node_ids if nid != nid1):
                return False

            # Matching condition: Bounding box union of subsets is smaller than
            # adding the subset sizes
            memlets: List[Memlet] = [
                e.data for e in graph.out_edges(map_entry) if e.src_conn == conn
            ]
            union_subset = memlets[0].subset
            for memlet in memlets[1:]:
                union_subset = subsets.bounding_box_union(
                    union_subset, memlet.subset)

            # TODO: Enhance me!
            # NOTE: This does not always result in correct behaviour for certain
            # ranges whose volume is not comparable by "<",
            # e.g "2*K" >? "K+1" > "K-1" >? "1"

            if permissive:
                try:
                    if union_subset.num_elements() < sum(
                            m.subset.num_elements() for m in memlets):
                        return True
                except TypeError:
                    pass

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return str(graph.node(candidate[DeduplicateAccess._map_entry]))

    def apply(self, sdfg: sd.SDFG):
        graph: sd.SDFGState = sdfg.nodes()[self.state_id]
        map_entry = graph.node(self.subgraph[DeduplicateAccess._map_entry])
        node1 = graph.node(self.subgraph[DeduplicateAccess._node1])
        node2 = graph.node(self.subgraph[DeduplicateAccess._node2])

        # Steps:
        # 1. Find unique subsets
        # 2. Find sets of contiguous subsets
        # 3. Create transients for subsets
        # 4. Redirect edges through new transients

        edges1 = set(e.src_conn for e in graph.edges_between(map_entry, node1))
        edges2 = set(e.src_conn for e in graph.edges_between(map_entry, node2))

        # Only apply to first connector (determinism)
        conn = sorted(edges1 & edges2)[0]

        edges = [e for e in graph.out_edges(map_entry) if e.src_conn == conn]

        # Get original data descriptor
        dname = edges[0].data.data
        desc = sdfg.arrays[edges[0].data.data]
        if isinstance(edges[0].dst,
                      nodes.AccessNode) and '15' in edges[0].dst.data:
            sdfg.save('faulty_dedup.sdfg')

        # Get unique subsets
        unique_subsets = set(e.data.subset for e in edges)

        # Find largest contiguous subsets
        try:
            # Start from stride-1 dimension
            contiguous_subsets = helpers.find_contiguous_subsets(
                unique_subsets,
                dim=next(i for i, s in enumerate(desc.strides) if s == 1))
        except (StopIteration, NotImplementedError):
            warnings.warn(
                "DeduplicateAcces::Not operating on Stride One Dimension!")
            contiguous_subsets = unique_subsets
        # Then find subsets for rest of the dimensions
        contiguous_subsets = helpers.find_contiguous_subsets(contiguous_subsets)
        # Map original edges to subsets
        edge_mapping = defaultdict(list)
        for e in edges:
            for ind, subset in enumerate(contiguous_subsets):
                if subset.covers(e.data.subset):
                    edge_mapping[ind].append(e)
                    break
            else:
                raise ValueError(
                    "Failed to find contiguous subset for edge %s" % e.data)

        # Create transients for subsets and redirect edges
        for ind, subset in enumerate(contiguous_subsets):
            name, _ = sdfg.add_temp_transient(subset.size(), desc.dtype)
            anode = graph.add_access(name)
            graph.add_edge(map_entry, conn, anode, None,
                           Memlet(data=dname, subset=subset))
            for e in edge_mapping[ind]:
                graph.remove_edge(e)
                new_memlet = copy.deepcopy(e.data)
                new_edge = graph.add_edge(anode, None, e.dst, e.dst_conn,
                                          new_memlet)
                for pe in graph.memlet_tree(new_edge):
                    # Rename data on memlet
                    pe.data.data = name
                    # Offset memlets to match new transient
                    pe.data.subset.offset(subset, True)
