# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This file implements the BruteForceEnumerator class """

from dace.transformation.estimator.enumeration import MapScoringEnumerator

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools


@make_properties
class BruteForceEnumerator(MapScoringEnumerator):
    """ 
    Enumerates all fusible map combinations in a 
    brute-force way. Returns subgraphs or map sets 
    and their respective scores, if a scoring function 
    is defined.
    """

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = None,
                 scoring_function=None):
        # initialize base class
        super().__init__(sdfg,
                         graph,
                         subgraph=subgraph,
                         condition_function=condition_function,
                         scoring_function=scoring_function)

    def brute_force(self):
        """
        Iterate over self.map_entries and yield all fusible 
        combinations along with their score.
        """
        for i in range(2, len(self._map_entries) + 1):
            for sg in itertools.combinations(self._map_entries, i):

                current_subgraph = helpers.subgraph_from_maps(self._sdfg, self._graph, sg)

                # evaluate condition if specified
                conditional_eval = True
                if self._condition_function:
                    conditional_eval = self._condition_function(self._sdfg, current_subgraph)

                # evaluate score if possible
                score = 0
                if conditional_eval and self._scoring_function:
                    score = self._scoring_function(current_subgraph)

                # yield element if condition is True
                if conditional_eval:
                    yield (tuple(sg), score) if self.mode == 'map_entries' else (current_subgraph, score)

    def iterator(self):
        yield from self.brute_force()
