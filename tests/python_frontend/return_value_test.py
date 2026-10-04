# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_return_scalar():

    @dace.program
    def return_scalar():
        return 5

    res = return_scalar()
    assert res == 5

    # The return value above is actually an array. If you would
    # add the return value annotation to the program, i.e. `-> dace.int32`, you would
    # get a validation error.
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, )
    assert res.dtype == np.int64


def test_return_scalar_in_nested_function():

    @dace.program
    def nested_function() -> dace.int32:
        return 5

    @dace.program
    def return_scalar():
        return nested_function()

    res = return_scalar()
    assert res == 5

    # The return value above is actually an array. If you would
    # add the return value annotation to the program, i.e. `-> dace.int32`, you would
    # get a validation error.
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, )
    assert res.dtype == np.int32


def test_return_array():

    @dace.program
    def return_array():
        return 5 * np.ones(5)

    res = return_array()
    assert np.allclose(res, 5 * np.ones(5))


def test_return_tuple():

    @dace.program
    def return_tuple():
        return 5, 6

    res = return_tuple()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res == (5, 6)


def test_return_tuple_multi():

    @dace.program
    def return_tuple_2():
        return 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

    res = return_tuple_2()
    assert isinstance(res, tuple)
    assert len(res) == 12
    assert res == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


def test_return_array_tuple():

    @dace.program
    def return_array_tuple():
        return 5 * np.ones(5), 6 * np.ones(6)

    res = return_array_tuple()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert np.allclose(res[0], 5 * np.ones(5))
    assert np.allclose(res[1], 6 * np.ones(6))


def test_return_void():

    @dace.program
    def return_void(a: dace.float64[20]):
        a[:] += 1
        return
        a[:] = 5

    a = np.random.rand(20)
    ref = a + 1
    res = return_void(a)
    assert res is None
    assert np.allclose(a, ref)


def test_return_tuple_1_element():

    @dace.program
    def return_one_element_tuple(a: dace.float64[20]):
        return (a + 3.5, )

    a = np.random.rand(20)
    ref = a + 3.5
    res = return_one_element_tuple(a)
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert np.allclose(res[0], ref)


def test_return_void_in_if():

    @dace.program
    def return_void(a: dace.float64[20]):
        if a[0] < 0:
            return
        a[:] = 5

    a = np.random.rand(20)
    return_void(a)
    assert np.allclose(a, 5)
    a[:] = np.random.rand(20)
    a[0] = -1
    ref = a.copy()
    return_void(a)
    assert np.allclose(a, ref)


def test_return_void_in_for():

    @dace.program
    def return_void(a: dace.float64[20]):
        for _ in range(20):
            return
        a[:] = 5

    a = np.random.rand(20)
    ref = a.copy()
    return_void(a)
    assert np.allclose(a, ref)


N, M, K = (dace.symbol(name, dtype=dace.int64) for name in 'NMK')


@dace.program
def _relu(x: dace.float32[M, K]):
    return np.maximum(x, 0)


def _check_nested_returns(sdfg: dace.SDFG):
    """ Checks that the return values of the nested SDFGs are the containers they are returned into. """
    sdfg.validate()
    nested = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.NestedSDFG)]
    assert nested
    for node in nested:
        assert not node.sdfg.arrays['__return'].transient


def test_return_from_nested_call_with_typed_symbols():
    """ The nested program's sizes are typed symbols that the call maps to the caller's. """

    @dace.program
    def nested_return_typed_symbols(a: dace.float32[N, N]):
        y = _relu(a - 1)
        return y + 1

    _check_nested_returns(nested_return_typed_symbols.to_sdfg(simplify=False))
    a = np.random.rand(5, 5).astype(np.float32)
    assert np.allclose(nested_return_typed_symbols(a), np.maximum(a - 1, 0) + 1)


def test_return_from_nested_call_with_constant_sizes():
    """ The nested program's sizes are typed symbols that the call maps to constants. """

    @dace.program
    def nested_return_constant_sizes(a: dace.float32[4, 3]):
        y = _relu(a - 1)
        return y + 1

    _check_nested_returns(nested_return_constant_sizes.to_sdfg(simplify=False))
    a = np.random.rand(4, 3).astype(np.float32)
    assert np.allclose(nested_return_constant_sizes(a), np.maximum(a - 1, 0) + 1)


if __name__ == '__main__':
    test_return_scalar()
    test_return_scalar_in_nested_function()
    test_return_array()
    test_return_tuple()
    test_return_array_tuple()
    test_return_void()
    test_return_void_in_if()
    test_return_void_in_for()
    test_return_from_nested_call_with_typed_symbols()
    test_return_from_nested_call_with_constant_sizes()
