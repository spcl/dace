""" This file implements the Scoring Function class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes
import dace.dtypes as dtypes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable, Type, Dict

import json


@make_properties
class ScoringFunction:
    '''
    Class used to Score Subgraphs in order to
    rank them for their fusion applicability
    '''
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 io: Dict = None,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 transformation_function: Type = CompositeFusion,
                 scope_dict=None,
                 **kwargs):

        # set sdfg-related variables
        self._sdfg = sdfg
        self._sdfg_id = sdfg.sdfg_id
        self._graph = graph
        self._state_id = sdfg.nodes().index(graph)
        self._subgraph = subgraph
        self._scope_dict = scope_dict
        self._transformation = transformation_function

        # set gpu attribute
        if gpu is None:
            # detect whether the state is assigned to GPU
            map_entries = [n for n in subgraph.nodes() if isinstance(n, nodes.MapEntry)]
            schedule = next(iter(map_entries)).schedule
            if any([m.schedule != schedule for m in map_entries]):
                raise RuntimeError(
                    "Schedules in maps to analyze should be the same")
            self._gpu = True if schedule in [
                dtypes.ScheduleType.GPU_Device,
                dtypes.ScheduleType.GPU_ThreadBlock
            ] else False
        else:
            self._gpu = gpu

        # search for outermost map entries
        self._map_entries = helpers.get_outermost_scope_maps(
            sdfg, graph, subgraph)

        # set IO if defined
        if io:
            self._inputs, self._outputs, self._symbols = io

        # kwargs define current part of search space to explore
        self._kwargs = kwargs

    def score(self, subgraph: SubgraphView, **kwargs):
        # NOTE: self._subgraph and subgraph are not the same!
        # subgraph has to be a subgraph of self._subgraph
        raise NotImplementedError

    def __call__(self, subgraph: SubgraphView, **kwargs):
        return self.score(subgraph, **kwargs)

    @staticmethod
    def name():
        return NotImplementedError
