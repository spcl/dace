# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains the access deduplication transformation. """

from collections import defaultdict
import copy
import itertools
from typing import List, Set

from dace import data, dtypes, sdfg as sd, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf
import dace.transformation.helpers as helpers

import warnings


class DeduplicateAccess(xf.SingleStateTransformation):
    """ 
    This transformation takes a node that is connected to multiple destinations
    with overlapping memlets, and consolidates those accesses through a 
    transient array or scalar.
    """

    map_entry = xf.PatternNode(nodes.MapEntry)
    node1 = xf.PatternNode(nodes.Node)
    node2 = xf.PatternNode(nodes.Node)

    @classmethod
    def expressions(cls):
        state = sd.SDFGState()
        state.add_nedge(cls.map_entry, cls.node1, Memlet())
        state.add_nedge(cls.map_entry, cls.node2, Memlet())
        return [state]

    def can_be_applied(self, graph: sd.SDFGState, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        nid1 = graph.node_id(self.node1)
        node1 = self.node1
        nid2 = graph.node_id(self.node2)
        node2 = self.node2

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
            node_ids = [graph.node_id(e.dst) for e in graph.out_edges(map_entry) if e.src_conn == conn]
            if any(nid < nid1 for nid in node_ids):
                return False
            if any(nid < nid2 for nid in node_ids if nid != nid1):
                return False

            # Matching condition: Bounding box union of subsets is smaller than
            # adding the subset sizes
            memlets: List[Memlet] = [e.data for e in graph.out_edges(map_entry) if e.src_conn == conn]
            union_subset = memlets[0].subset
            for memlet in memlets[1:]:
                union_subset = subsets.bounding_box_union(union_subset, memlet.subset)

            # TODO: Enhance me!
            # NOTE: This does not always result in correct behaviour for certain
            # ranges whose volume is not comparable by "<",
            # e.g "2*K" >? "K+1" > "K-1" >? "1"

            if permissive:
                try:
                    if union_subset.num_elements() < sum(m.subset.num_elements() for m in memlets):
                        return True
                except TypeError:
                    pass

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        node1 = self.node1
        node2 = self.node2

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

        # Get unique subsets
        unique_subsets = set(e.data.subset for e in edges)

        # Find largest contiguous subsets
        try:
            # Start from stride-1 dimension
            contiguous_subsets = helpers.find_contiguous_subsets(unique_subsets,
                                                                 dim=next(i for i, s in enumerate(desc.strides)
                                                                          if s == 1))
        except (StopIteration, NotImplementedError):
            warnings.warn("DeduplicateAcces::Not operating on Stride One Dimension!")
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
                raise ValueError("Failed to find contiguous subset for edge %s" % e.data)

        # Create transients for subsets and redirect edges
        for ind, subset in enumerate(contiguous_subsets):
            name, _ = sdfg.add_temp_transient(subset.size(), desc.dtype)
            anode = graph.add_access(name)
            graph.add_edge(map_entry, conn, anode, None, Memlet(data=dname, subset=subset))
            for e in edge_mapping[ind]:
                graph.remove_edge(e)
                new_memlet = copy.deepcopy(e.data)
                new_edge = graph.add_edge(anode, None, e.dst, e.dst_conn, new_memlet)
                for pe in graph.memlet_tree(new_edge):
                    # Rename data on memlet
                    pe.data.data = name
                    # Offset memlets to match new transient
                    pe.data.subset.offset(subset, True)
