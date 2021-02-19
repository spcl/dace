""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.dataflow import DeduplicateAccess
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.perf.movement_counter import count_moved_data_state
from dace.perf.movement_counter import count_moved_data_state_composite
from dace.perf.movement_counter import count_moved_data_subgraph

import dace.sdfg.propagation as propagation
import dace.symbolic as symbolic
import sympy

from typing import Set, Union, List, Callable, Dict, Type

from dace.transformation.estimator.scoring import ScoringFunction

import json
import warnings
import os
import numpy as np
import sys


@make_properties
class MemletScore(ScoringFunction):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''
    debug = Property(desc="Debug Mode", dtype=bool, default=True)

    propagate_all = Property(desc="Do a final propagation step "
                             "before evaluation "
                             "using propagate_memlets_state()",
                             dtype=bool,
                             default=True)

    deduplicate = Property(desc="Do a Deduplication step"
                           "after Subgraph Fusion ",
                           dtype=bool,
                           default=False)

    exit_on_error = Property(
        desc="Exit program if error occurs, else return -1",
        dtype=bool,
        default=True)
    view_on_error = Property(desc="View program if faulty",
                             dtype=bool,
                             default=False)
    save_on_error = Property(desc="Save SDFG if faulty",
                             dtype=bool,
                             default=False)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 io: Dict,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 transformation_function: Type = CompositeFusion,
                 **kwargs):
        super().__init__(sdfg=sdfg,
                         graph=graph,
                         subgraph=subgraph,
                         io=io,
                         gpu=gpu,
                         transformation_function=transformation_function,
                         **kwargs)

        # inputs and outputs not needed
        self._outputs, self._inputs = None, None

        # estimate traffic on base sdfg
        # also apply deduplication on baseline graph in order to get a more realistic estimate
        if self.deduplicate:
            sdfg_copy = SDFG.from_json(self._sdfg.to_json())
            graph_copy = sdfg_copy.nodes()[self._state_id]
            sdfg_copy.apply_transformations_repeated(DeduplicateAccess,
                                                     states=[graph_copy])
            self._base_traffic = self.estimate_traffic(sdfg_copy, graph_copy)
        else:
            # can just use the graph directly
            self._base_traffic = self.estimate_traffic(sdfg, graph)

        self._i = 0

    def symbolic_evaluation(self, term):

        if isinstance(term, (int, float)):
            return term 
        # take care of special functions appearing in term and resolve those
        x, y = sympy.symbols('x y')
        sym_locals = {
            sympy.Function('int_floor'):
            sympy.Lambda((x, y),
                         sympy.functions.elementary.integers.floor(x / y)),
            sympy.Function('int_ceil'):
            sympy.Lambda((x, y),
                         sympy.functions.elementary.integers.ceiling(x / y)),
            sympy.Function('floor'):
            sympy.Lambda((x), sympy.functions.elementary.integers.floor(x)),
            sympy.Function('ceiling'):
            sympy.Lambda((x), sympy.functions.elementary.integers.ceiling(x)),
        }
        for fun, lam in sym_locals.items():
            term.replace(fun, lam)
        try:
            result = symbolic.evaluate(term, self._symbols)
        except TypeError:
            print(f"Error: Cannot evaluate {term}")
            raise TypeError()
        result = int(result)
        return result

    def estimate_traffic(self, sdfg, graph):
        try:
            # get traffic count
            traffic_symbolic = count_moved_data_state_composite(graph)
            # evaluate w.r.t. symbols
            traffic = self.symbolic_evaluation(traffic_symbolic)
            if traffic == 0:
                raise RuntimeError("Traffic is Zero")
        except Exception as e:
            print("ERROR in score_memlet:")
            print(e)
            traffic = 0
            if self.view_on_error:
                sdfg.view()
            if self.save_on_error:
                i = 0
                while (os.path.exists(f"error{i}.sdfg")):
                    i += 1
                sdfg.save('error.sdfg')
            if self.exit_on_error:
                sys.exit(0)
        return traffic

    def score(self, subgraph: SubgraphView):
        '''
        Applies CompositeFusion to the Graph and compares Memlet Volumes
        with the untransformed SDFG passed to __init__().
        '''

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        if self.debug:
            print("ScoreMemlet::Debug::Subgraph to Score:",
                  subgraph_copy.nodes())

        transformation_function = self._transformation(subgraph_copy)
        # assign properties to transformation
        for (arg, val) in self._kwargs.items():
            try:
                setattr(transformation_function, arg, val)
            except AttributeError:
                warnings.warn(f"Attribute {arg} not found in transformation")
            except (TypeError, ValueError):
                warnings.warn(f"Attribute {arg} has invalid value {val}")

        transformation_function.apply(sdfg_copy)
        if self.deduplicate:
            sdfg_copy.apply_transformations_repeated(DeduplicateAccess,
                                                     states=[graph_copy])
        if self.propagate_all or self.deduplicate:
            propagation.propagate_memlets_scope(sdfg_copy, graph_copy,
                                                graph_copy.scope_leaves())

        self._i += 1
        sdfg_copy.save(f"inspect_{self._i}.sdfg")
        current_traffic = self.estimate_traffic(sdfg_copy, graph_copy)
        return current_traffic / self._base_traffic

    @staticmethod
    def name():
        return "Estimated Memlet Traffic"
