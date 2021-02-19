# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
from dace.transformation.subgraph import SubgraphFusion, MultiExpansion, helpers 
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.estimator.enumeration import GreedyEnumerator
from dace.sdfg.utils import dfs_topological_sort
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView

import sys

from dace.transformation.subgraph import SubgraphFusion

N, M, O, P, Q = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q']]
N.set(5)
M.set(6)
O.set(7)
P.set(8)
Q.set(9)


A = np.random.rand(N.get()).astype(np.float64)
B = np.random.rand(M.get()).astype(np.float64)
C = np.random.rand(O.get()).astype(np.float64)

out1 = np.ndarray((N.get(), M.get()), np.float64)
out2 = np.ndarray((1), np.float64)
out3 = np.ndarray((N.get(), M.get(), O.get()), np.float64)


@dace.program
def greedy(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O]):
    aa = A*2 + 1
    bb = B*3 + 4
    tmp = np.ndarray((N,M), dtype = np.float32)

    for i,j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            inp1 << aa[i]
            inp2 << bb[j]
            out >> tmp[i,j]

            out = inp1 * 2 + inp2 * 2 

    cc = C* 5 + 2
    ccc = cc + 3 + C 

    result = np.ndarray((N,M,O), dtype = np.float32)
    for i,j,k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            inp1 << tmp[i,j]
            inp2 << ccc[k]
            out >> result

            out = inp1 + inp2
    
    return result 

@dace.program 
def greedy2(A: dace.float64[N]):
    aa = A*2
    bb = aa * 2
    cc = bb * 3
    tmp = A*4
    dd = tmp + cc 
    return dd 


def enumerate_greedy(sdfg, graph, subgraph):
    greedy_enumerator = GreedyEnumerator(sdfg = sdfg,
                                         graph = graph,
                                         subgraph = subgraph)
    map_sets = list()
    print("---------TEST----------")
    for subgraph in greedy_enumerator:
        print("Current Subgraph = ", subgraph)
        map_sets.append(subgraph)
        print(subgraph)

    for map_set in map_sets:
        for other_set in map_sets:
            if other_set != map_set:
                assert len(set(map_set) & set(other_set)) == 0
    print("-----------------------")


def case_1(sdfg):
    # Test Case 1: Whole graph 
    graph = sdfg.nodes()[0]
    subgraph = None 
    enumerate_greedy(sdfg, graph, subgraph)

def case_2(sdfg):
    graph = sdfg.nodes()[0]
    index = 0
    map_entries = []
    for node in dfs_topological_sort(graph):
        if isinstance(node, nodes.MapEntry):
            index += 1
            map_entries.append(node)
        
        if index > 6:
            break 
    
    subgraph = helpers.subgraph_from_maps(sdfg, graph, map_entries)
    enumerate_greedy(sdfg, graph, subgraph)


greedy_sdfg = greedy.to_sdfg()
# Test Case 1: Whole graph 
case_1(greedy_sdfg)
case_2(greedy_sdfg)

