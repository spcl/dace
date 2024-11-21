import numpy as np

import dace
from dace.sdfg import SDFG
from dace.memlet import Memlet

import pytest


ll = ((np.uint64(np.iinfo(np.int32).max) + np.uint64(4))//np.uint64(4))*np.uint64(4)

@pytest.mark.parametrize("set_explicit", [
    False,
    True
])
def test_explicit_param(set_explicit):
    N = dace.symbol('N', dtype=dace.dtypes.typeclass(np.uint64))

    @dace.program
    def sdfg_internal(input: dace.float32[4], output: dace.float32[4]):
        @dace.tasklet
        def init():
            out >> output
            out = input

        for k in range(4):

            @dace.tasklet
            def do():
                oin << output[k]
                out >> output[k]
                out = oin * input[k]

    # Construct SDFG
    mysdfg = SDFG('outer_sdfg')
    state = mysdfg.add_state()
    Aarr = mysdfg.add_array('A', [N], dace.float32)
    A = state.add_access(Aarr[0])
    Barr = mysdfg.add_array('B', [N], dace.float32)
    B = state.add_access(Barr[0])

    map_entry, map_exit = state.add_map(
        'elements', [('i', f'0:N:1')])
    if set_explicit:
        map_entry.map.param_types = {'i': dace.dtypes.typeclass(np.uint64)}
    nsdfg = state.add_nested_sdfg(
        sdfg_internal.to_sdfg(), mysdfg, {'input'}, {'output'})

    # Add edges
    state.add_memlet_path(A, map_entry, nsdfg,
                          dst_conn='input', memlet=Memlet.simple(A, 'i%250'))
    state.add_memlet_path(nsdfg, map_exit, B,
                          src_conn='output', memlet=Memlet.simple(B, 'i%250'))

    N = np.uint64(ll)

    input = dace.ndarray([250], dace.float32)

    mysdfg.validate()
    mysdfg(A=input, B=input, N=N)

@pytest.mark.parametrize("set_explicit", [
    False,
    True
])
def test_explicit_param_with_nested_sdfg_map(set_explicit):
    N = dace.symbol('N')

    @dace.program
    def sdfg_internal_map(input: dace.float32[4], output: dace.float32[4]):
        for k1 in dace.map[0:4:1]:
            @dace.tasklet
            def do():
                oin << output[k1]
                out >> output[k1]
                out = oin * input[k1]

    # Construct SDFG
    mysdfg = SDFG('outer_sdfg')
    state = mysdfg.add_state()
    Aarr = mysdfg.add_array('A', [N], dace.float32)
    A = state.add_access(Aarr[0])
    Barr = mysdfg.add_array('B', [N], dace.float32)
    B = state.add_access(Barr[0])

    map_entry, map_exit = state.add_map(
        'elements', [('i', f'0:N:4')])
    if set_explicit:
        map_entry.map.param_types = {'i': dace.dtypes.typeclass(np.uint64)}
    if set_explicit:
        nsdfg = state.add_nested_sdfg(
            sdfg_internal_map.to_sdfg(), mysdfg, {'input'}, {'output'},
            symbol_type_mapping={'i': dace.dtypes.typeclass(np.uint64)})
    else:
        nsdfg = state.add_nested_sdfg(
            sdfg_internal_map.to_sdfg(), mysdfg, {'input'}, {'output'})

    # Add edges
    state.add_memlet_path(A, map_entry, nsdfg,
                          dst_conn='input', memlet=Memlet.simple(A, 'i%64:(i+4)%64:1'))
    state.add_memlet_path(nsdfg, map_exit, B,
                          src_conn='output', memlet=Memlet.simple(B, 'i%64:(i+4)%64:1'))

    N = np.uint64(ll)

    input = dace.ndarray([64], dace.float32)

    mysdfg.validate()
    mysdfg(A=input, B=input, N=N)


if __name__ == "__main__":
    test_explicit_param(False)
    test_explicit_param(True)
    test_explicit_param_with_nested_sdfg_map(False)
    test_explicit_param_with_nested_sdfg_map(True)
