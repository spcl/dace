# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.transformation.estimator.enumeration import Enumerator
import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools
import warnings


@make_properties
class MapScoringEnumerator(Enumerator):
    '''
    Abstract Enumerator class that is used by enumerators 
    which rely on a scoring function 
    '''

    mode = Property(desc="Data type the Iterator should return. "
                         "Choice between Subgraph and List of Map Entries.",
                    default="map_entries",
                    choices=["subgraph", "map_entries"],
                    dtype=str)

    def __init__(self, sdfg, graph, subgraph, condition_function,
                 scoring_function):

        super().__init__(sdfg, graph, subgraph, condition_function)

        # used to attach a score to each enumerated subgraph
        self._scoring_function = scoring_function

    def list(self):
        return list(e[0] for e in self.iterator())

    def scores(self):
        return list(e for e in self.iterator())
