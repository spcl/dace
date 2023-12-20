# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" 
Intra-state code generation scheduling functionality.

These functions govern the order in which nodes are traversed during code generation.
Takes care of branches within the state dataflow graph.
"""
from typing import Iterator, Set
from dace.sdfg import nodes, SDFG, utils as sdutil
from dace.sdfg.state import StateSubgraphView


def data_aware_topological_sort(sdfg: SDFG, subgraph: StateSubgraphView,
                                start_nodes: Set[nodes.Node]) -> Iterator[nodes.Node]:
    yield from sdutil.dfs_topological_sort(subgraph, start_nodes)
