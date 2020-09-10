# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import sys

from dace.transformation.subgraph import ReduceExpansion, SubgraphFusion, MultiExpansion
import dace.transformation.subgraph.helpers as helpers

import dace.dtypes as dtypes
from dace.sdfg.graph import SubgraphView
import dace.libraries.standard as stdlib
import dace.sdfg.nodes as nodes
from typing import Union, List
from util import expand_maps, expand_reduce, fusion

dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity=0)

    tmp_out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


H.set(10)
B.set(10)
SN.set(20)
SM.set(20)


def get_partition(sdfg, graph):
    subgraph1 = SubgraphView(graph, [])
    subgraph2 = SubgraphView(graph, [])

    cnt1 = 0
    for node in dace.sdfg.utils.dfs_topological_sort(graph):
        if isinstance(node, stdlib.nodes.reduce.Reduce):
            if cnt1 < 2:
                subgraph1._subgraph_nodes.append(node)
                cnt1 += 1
            else:
                subgraph2._subgraph_nodes.append(node)

        if isinstance(node, nodes.MapEntry):
            if cnt1 < 2:
                subgraph1._subgraph_nodes.append(node)
                cnt1 += 1
            else:
                subgraph2._subgraph_nodes.append(node)

    return [subgraph1, subgraph2]


def test_2fuse():
    sdfg = softmax.to_sdfg()
    sdfg._name = 'softmax_2part'
    sdfg.apply_strict_transformations()
    X_in = np.random.rand(H.get(), B.get(), SN.get(),
                          SM.get()).astype(np.float32)

    csdfg = sdfg.compile()
    res1 = csdfg(X_in=X_in, H=H, B=B, SN=SN, SM=SM)

    subgraph = get_partition(sdfg, sdfg.nodes()[0])
    expand_reduce(sdfg, sdfg.nodes()[0], subgraph)
    expand_maps(sdfg, sdfg.nodes()[0], subgraph)
    fusion(sdfg, sdfg.nodes()[0], subgraph)

    csdfg = sdfg.compile()
    res2 = csdfg(X_in=X_in, H=H, B=B, SN=SN, SM=SM)

    assert np.allclose(res1, res2)
    print("PASS")
    return


def test_1fuse():
    sdfg = softmax.to_sdfg()
    sdfg._name = 'softmax_fused'
    sdfg.apply_strict_transformations()
    X_in = np.random.rand(H.get(), B.get(), SN.get(),
                          SM.get()).astype(np.float32)

    csdfg = sdfg.compile()
    res1 = csdfg(X_in=X_in, H=H, B=B, SN=SN, SM=SM)

    expand_reduce(sdfg, sdfg.nodes()[0])
    expand_maps(sdfg, sdfg.nodes()[0])
    fusion(sdfg, sdfg.nodes()[0])

    #sdfg.specialize({'SM':SM})
    csdfg = sdfg.compile()
    res2 = csdfg(X_in=X_in, H=H, B=B, SN=SN, SM=SM)

    print(np.linalg.norm(res1))
    print(np.linalg.norm(res2))
    assert np.allclose(res1, res2)
    print("PASS")
    return


if __name__ == "__main__":
    test_2fuse()
    test_1fuse()
