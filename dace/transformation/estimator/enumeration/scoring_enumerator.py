""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.transformation.estimator import ScoringFunction
from dace.transformation.estimator.enumeration import Enumerator
import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools
import warnings 


@make_properties
class ScoringEnumerator(Enumerator):
    '''
    Abstract Enumerator class that is used by enumerators 
    which rely on a scoring function 
    '''

    def __init__(self, 
                 sdfg, 
                 graph,
                 subgraph,
                 condition_function,
                 scoring_function):
        
        super().__init__(sdfg, graph, subgraph, condition_function)
        self._scoring_function = scoring_function
        if self._condition_function is None and self._scoring_function is not None:
            warnings.warn('Initialized with no condition function but scoring'
                          'function. Will try to score all subgraphs!')
    
    def list(self):
        return list(e[0] for e in self.iterator())

    def scores(self):
        return list(e for e in self.iterator())