""" This file implements the ConnectedEnumerator class """

from dace.transformation.estimator.enumeration import Enumerator
from dace.transformation.estimator.enumeration import ScoringEnumerator

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools

@make_properties
class ConnectedEnumerator(ScoringEnumerator):
    '''
    Enumerates all subgraphs that are connected through Access Nodes
    '''

    prune = Property(desc="Perform Greedy Pruning during Enumeration",
                     default=True,
                     dtype=bool)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = None,
                 scoring_function=None,
                 **kwargs):

        # initialize base class
        super().__init__(sdfg, graph, subgraph, condition_function,
                         scoring_function)
        self._local_maxima = []
        self._function_args = kwargs

        self.calculate_topology(subgraph)
       
        
    def traverse(self, current: List, forbidden: Set):
        if len(current) > 0:
            # get current subgraph we are inspecting
            #print("*******")
            #print(current)
            current_subgraph = helpers.subgraph_from_maps(
                self._sdfg, self._graph, current, self._scope_children)
            #print(current_subgraph.nodes())

            # evaluate condition if specified
            conditional_eval = True
            if self._condition_function:
                conditional_eval = self._condition_function(
                    self._sdfg, current_subgraph)
                #print("EVALUATED TO", conditional_eval)
            # evaluate score if possible
            score = 0
            if conditional_eval and self._scoring_function:
                score = self._scoring_function(current_subgraph)

            # calculate where to backtrack next if not prune
            '''
            go_next = set()
            if conditional_eval or not self._prune or len(current) == 1:
                go_next = set(m for c in current
                              for m in self._adjacency_list[c]
                              if m not in current and m not in forbidden)
                if self.debug:
                    go_next = list(go_next)
                    go_next.sort(key=lambda e: e.map.label)
            '''

            go_next = list()
            if conditional_eval or self.prune == False or len(current) == 1:
                go_next = list(set(m for c in current
                                   for m in self._adjacency_list[c]
                                   if m not in current and m not in forbidden))

                # for determinism and correctness during pruning
                go_next.sort(key = lambda me: self._labels[me])
             

            # yield element if condition is True
            if conditional_eval:
                self._histogram[len(current)] += 1
                yield (tuple(current),
                       score) if self.mode == 'map_entries' else (
                           current_subgraph, score)

        else:
            # special case at very beginning: explore every node
            go_next = list(set(m for m in self._adjacency_list.keys()))
            go_next.sort(key = lambda me: self._labels[me])
            
        if len(go_next) > 0:
            # recurse further
            forbidden_current = set()
            for child in go_next:
                current.append(child)
                yield from self.traverse(current, forbidden | forbidden_current)
                current.pop()
                forbidden_current.add(child)

       
    def iterator(self):
        '''
        returns an iterator that iterates over
        search space yielding tuples (subgraph, score)
        '''
        self._histogram = defaultdict(int)
        yield from self.traverse([], set())
