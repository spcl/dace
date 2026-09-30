# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A loop iterator is typed like the map parameter it becomes, so LoopToMap keeps one dtype per name. """
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg.type_inference import infer_iteration_symbol_type
from dace.transformation.interstate import LoopToMap


def counting_loop_sdfg(bound_dtype: dace.typeclass) -> dace.SDFG:
    sdfg = dace.SDFG('counting_loop')
    sdfg.add_symbol('M', bound_dtype)
    sdfg.add_array('a', [dace.symbol('M', bound_dtype)], dace.float64)
    loop = LoopRegion('walk', 'i < M', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in + 1.0')
    state.add_edge(state.add_read('a'), None, tasklet, '__in', dace.Memlet('a[i]'))
    state.add_edge(tasklet, '__out', state.add_write('a'), None, dace.Memlet('a[i]'))
    return sdfg


@pytest.mark.parametrize('bound_dtype', [dace.int32, dace.int64])
def test_loop_iterator_takes_the_map_parameter_dtype(bound_dtype):
    sdfg = counting_loop_sdfg(bound_dtype)
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    map_dtype = infer_iteration_symbol_type(0, dace.symbol('M', bound_dtype) - 1, symbols=dict(sdfg.symbols))
    assert map_dtype == bound_dtype
    assert loop.new_symbols(dict(sdfg.symbols)) == {'i': map_dtype}


@pytest.mark.parametrize('bound_dtype', [dace.int32, dace.int64])
def test_loop_to_map_declares_the_nested_iterator_at_the_map_parameter_dtype(bound_dtype):
    sdfg = counting_loop_sdfg(bound_dtype)
    assert sdfg.apply_transformations(LoopToMap) == 1
    declared = {s.name: s.symbols['i'] for s in sdfg.all_sdfgs_recursive() if 'i' in s.symbols}
    assert declared and all(dtype == bound_dtype for dtype in declared.values()), declared
    sdfg.validate()


if __name__ == '__main__':
    for dtype in (dace.int32, dace.int64):
        test_loop_iterator_takes_the_map_parameter_dtype(dtype)
        test_loop_to_map_declares_the_nested_iterator_at_the_map_parameter_dtype(dtype)
