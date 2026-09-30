# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Fusing two maps renames the second map's parameter at the dtype the map declares for it. """
import dace
from dace import symbolic
from dace.transformation.dataflow import MapFusionVertical

M = dace.symbol('M', dace.int64)


def two_maps_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('renamed_parameter')
    sdfg.add_symbol('M', dace.int64)
    for name in ('a', 'c'):
        sdfg.add_array(name, [M], dace.float64)
    sdfg.add_transient('tmp', [M], dace.float64)
    state = sdfg.add_state()
    i = symbolic.symbol('i', dace.int64)
    j = symbolic.symbol('j', dace.int64)
    tmp = state.add_access('tmp')
    state.add_mapped_tasklet('first', {'i': '0:M'}, {'x': dace.Memlet(data='a', subset=dace.subsets.Range([(i, i, 1)]))},
                             'y = x + 1', {'y': dace.Memlet(data='tmp', subset=dace.subsets.Range([(i, i, 1)]))},
                             output_nodes={'tmp': tmp},
                             external_edges=True)
    state.add_mapped_tasklet('second', {'j': '0:M'},
                             {'x': dace.Memlet(data='tmp', subset=dace.subsets.Range([(j, j, 1)]))},
                             'y = x * 2', {'y': dace.Memlet(data='c', subset=dace.subsets.Range([(j, j, 1)]))},
                             input_nodes={'tmp': tmp},
                             external_edges=True)
    return sdfg


def test_renamed_parameter_keeps_the_declared_dtype():
    sdfg = two_maps_sdfg()
    sdfg.validate()
    assert sdfg.apply_transformations(MapFusionVertical) == 1
    sdfg.validate()
    state = sdfg.states()[0]
    iterators = {
        sym
        for edge in state.edges() if not edge.data.is_empty() for rng in edge.data.subset.ndrange() for expr in rng
        if symbolic.issymbolic(expr) for sym in expr.free_symbols if sym.name == 'i'
    }
    assert {sym.dtype for sym in iterators} == {dace.int64}


if __name__ == '__main__':
    test_renamed_parameter_keeps_the_declared_dtype()
