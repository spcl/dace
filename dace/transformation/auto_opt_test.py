# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
from dace.transformation.subgraph import SubgraphFusion, MultiExpansion, helpers 
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.estimator.enumeration import GreedyEnumerator
from dace.transformation.estimator.programs import factory
from dace.sdfg.utils import dfs_topological_sort
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView

import sys

from dace.transformation.subgraph import SubgraphFusion

from dace.transformation.auto_optimize import greedy_fuse, auto_optimize

def get_sdfg(name):
    return factory.get_program(name)

def get_args(name):
    return factory.get_args(name)

def run_scenario(name, 
                 gpu = False,
                 run = False, 
                 view = False,
                 validate_all = True):
    
    sdfg = get_sdfg(name)
    args = get_args(name) 

    if gpu:
        sdfg.apply_gpu_transformations()
    
    if run:
        result1 = sdfg(args)

    greedy_fuse(sdfg, validate_all = True)

    if run:
        result2 = sdfg(args)
    
    if view:
        sdfg.view() 
    



run_scenario("greedy")


    

