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

from copy import deepcopy as dcpy

def get_sdfg(name):
    return factory.get_program(name)

def get_args(name):
    return factory.get_args(name)

def run_scenario(name, 
                 gpu = False,
                 run = True, 
                 view = False,
                 validate_all = True):
    
    sdfg = get_sdfg(name)
    args = get_args(name) 

    arg1 = dcpy(args)
    arg2 = dcpy(args)

    if gpu:
        sdfg.apply_gpu_transformations()
    
    if run:
        result1 = sdfg(**{**arg1[0], **arg1[1], **arg1[2]})

    greedy_fuse(sdfg, validate_all = True)

    if run:
        result2 = sdfg(**{**arg2[0], **arg2[1], **arg2[2]})
    
    if view:
        sdfg.view() 

    if run:
        if result1 is not None:
            print("__return:\t", np.linalg.norm(result1))
            print("__return:\t", np.linalg.norm(result2))
        for aname, array in args[1].items():
            print(f"{aname}:\t", np.linalg.norm(arg1[1][aname]))
            print(f"{aname}:\t", np.linalg.norm(arg2[1][aname]))



#run_scenario("greedy", view = True)  # [OK]
run_scenario("vadv") # [OK]
#run_scenario("hdiff_mini") 
#run_scenario("hdiff")

