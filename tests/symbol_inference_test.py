# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def symbol_inference(A: dace.float64[N, N], B: dace.float64[M + 1, M * 2]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a >> A[i, j]
            a = N

    for i, j in dace.map[0:M + 1, 0:M * 2]:
        with dace.tasklet:
            b >> B[i, j]
            b = M


@dace.program
def symbol_inference_joint(A: dace.float64[N + M], B: dace.float64[N + 2 * M]):
    for i in dace.map[0:N + M]:
        with dace.tasklet:
            a >> A[i]
            a = N

    for i in dace.map[0:N + 2 * M]:
        with dace.tasklet:
            b >> B[i]
            b = M


def test_symbol_inference():
    real_N = 5
    real_M = 7
    A = np.random.rand(real_N, real_N)
    B = np.random.rand(real_M + 1, real_M * 2)
    symbol_inference(A, B)
    assert np.allclose(A, np.full_like(A, real_N))
    assert np.allclose(B, np.full_like(B, real_M))


def test_symbol_inference_joint():
    real_N = 3
    real_M = 2
    A = np.random.rand(real_N + real_M)
    B = np.random.rand(real_N + real_M * 2)
    symbol_inference_joint(A, B)
    assert np.allclose(A, np.full_like(A, real_N))
    assert np.allclose(B, np.full_like(B, real_M))


def test_dynamic_range_bound_types_the_map_parameter():
    """A dynamic-range connector is the declared type of the bound that names it.

    The parameter's type must come from that connector and not from the dtype the bound's symbol
    instance happens to carry: symbol identity is by name, so an expression rebuilt from a string
    (what every ``replace_dict`` substitution does) mints its names untyped, while deserialization
    rebuilds them from the declared table. Reading the instance makes the same map report two
    different parameter types across a save/load round trip.
    """
    sdfg = dace.SDFG('dynamic_range_bound')
    sdfg.add_array('A', [128], dace.float64)
    sdfg.add_scalar('lo', dace.uint64, transient=False)
    state = sdfg.add_state()

    # Parsed from a string, so the bound carries an untyped ``lo`` -- the unstable dtype.
    bound = dace.symbolic.pystr_to_symbolic('lo')
    assert all(s.dtype is not dace.uint64 for s in bound.free_symbols)

    entry, exit_node = state.add_map('m', {'i': dace.subsets.Range([(bound, 127, 1)])})
    entry.add_in_connector('lo', dace.uint64)
    state.add_edge(state.add_read('lo'), None, entry, 'lo', dace.Memlet('lo[0]'))

    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_memlet_path(entry, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet, exit_node, state.add_write('A'), src_conn='o', memlet=dace.Memlet('A[i]'))

    new_symbols = entry.new_symbols(sdfg, state, dict(sdfg.symbols))
    assert new_symbols['lo'] == dace.uint64
    assert new_symbols['i'] == dace.uint64


if __name__ == '__main__':
    test_symbol_inference()
    test_symbol_inference_joint()
    test_dynamic_range_bound_types_the_map_parameter()
