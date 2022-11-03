# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This file implements the GreedyEnumerator class """

from dace.transformation.estimator.enumeration import Enumerator

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools

import heapq


class QueuedEntry:

    def __init__(self, map_entry, index, reverse=False):
        self.map_entry = map_entry
        self.index = index
        self.reverse = reverse

    def __lt__(self, other):
        if not self.reverse:
            return self.index < other.index
        else:
            return self.index > other.index


@make_properties
class GreedyEnumerator(Enumerator):
    """
    Enumerates all maximally fusible subgraphs in a greedy manner, 
    each of the corresponding map sets from an iteration being disjoint
    """

    mode = Property(desc="Data type the Iterator should return. "
                    "Choice between Subgraph and List of Map Entries.",
                    default="map_entries",
                    choices=["subgraph", "map_entries"],
                    dtype=str)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = CompositeFusion.can_be_applied,
                 reverse=False):

        super().__init__(sdfg, graph, subgraph, condition_function)

        # need topology information
        self.calculate_topology(subgraph)
        self._reverse = reverse

    def iterator(self):
        # iterate through adjacency list starting with map with lowest label.
        # then greedily explore neighbors with next lowest label and see whether set is fusible
        # if not fusible, cancel and create a new set

        if len(self._adjacency_list) == 0:
            return
        first_map = next(me for me in self._adjacency_list if self._labels[me] == 0)

        # define queue / visited set which helps us find starting points
        # for the next inner iterations
        added = set()
        outer_queued = set(self._source_maps)
        outer_queue = [QueuedEntry(me, self._labels[me], reverse=self._reverse) for me in self._source_maps]
        while len(outer_queue) > 0:

            # current iteration: define queue / set with which we are going
            # to find current components

            while len(outer_queue) > 0:
                next_iterate = heapq.heappop(outer_queue)
                if next_iterate.map_entry not in added:
                    break
                elif len(outer_queue) == 0:
                    next_iterate = None
                    break

            if not next_iterate:
                break

            current_set = set()
            inner_queue = [next_iterate]
            inner_queued = {next_iterate.map_entry}

            while len(inner_queue) > 0:

                # select starting map
                current = heapq.heappop(inner_queue)
                current_map = current.map_entry

                # check whether current | current_set can be fused
                add_current_map = False
                if len(current_set) == 0:
                    add_current_map = True
                else:
                    subgraph = helpers.subgraph_from_maps(self._sdfg, self._graph, current_set | {current_map})
                    if self._condition_function(self._sdfg, subgraph):
                        add_current_map = True

                if add_current_map:
                    # add it to current set and continue BFS
                    added.add(current_map)
                    current_set.add(current_map)
                    # recurse further
                    for current_neighbor_map in self._adjacency_list[current_map]:
                        # add to outer queue and set
                        if current_neighbor_map not in added:
                            if current_neighbor_map not in outer_queued:
                                heapq.heappush(
                                    outer_queue,
                                    QueuedEntry(current_neighbor_map,
                                                self._labels[current_neighbor_map],
                                                reverse=self._reverse))
                                outer_queued.add(current_neighbor_map)
                            # add to inner queue and set
                            if current_neighbor_map not in inner_queued:
                                heapq.heappush(
                                    inner_queue,
                                    QueuedEntry(current_neighbor_map,
                                                self._labels[current_neighbor_map],
                                                reverse=self._reverse))
                                inner_queued.add(current_neighbor_map)

            # yield
            if self.mode == 'map_entries':
                yield tuple(current_set)
            else:
                yield helpers.subgraph_from_maps(self._sdfg, self._graph, current_set)
