# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests preprocessing of SDFG call tree and (nested) closure. """
import dace
import numpy as np
import os
import tempfile


def test_nested_objects_same_name():
    class ObjA:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)

        @dace.method
        def __call__(self, A):
            return A + self.q

    class ObjB:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)
            self.obja = ObjA(q * 2)

        @dace.method
        def outer(self, A):
            return A + self.q + self.obja(A)

    obj = ObjB(5)
    unusedA = np.random.rand(20)

    # Get closure first
    closure = obj.outer.closure_resolver(None, None)

    # Save SDFG
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    obj.outer.to_sdfg(unusedA).save(temp_path)

    # Get closure
    # Verify obj's closure is `self.q`: obj.q
    assert len(obj.outer.resolver.closure_arrays) == len(closure.closure_arrays)
    assert closure.closure_arrays['__g_self_q'][2]() is obj.q
    # Verify obj.obja.__call__'s closure is `self.q`: obj.obja.q
    assert closure.closure_arrays['__g_self_q_0'][2]() is obj.obja.q

    # Load SDFG
    A = np.random.rand(20)
    obj = ObjB(6)
    obj.outer.load_sdfg(temp_path, A)

    # Verify that cache contains new SDFG
    assert len(obj.outer._cache.cache) == 1

    # Verify that program works
    expected = A + obj.q + A + (obj.q * 2)
    result = obj.outer(A)
    assert np.allclose(expected, result)

    # Verify that no new entries were added to the cache
    assert len(obj.outer._cache.cache) == 1

    os.remove(temp_path)  # Clean up


def test_calltree():
    class ObjA:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)

        @dace.method
        def __call__(self, A):
            return A + self.q

    class ObjB:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)
            self.obja = ObjA(q * 2)

        @dace.method
        def outer(self, A):
            return A + self.q + self.obja(A)

    obj = ObjB(5)
    res = obj.outer.closure_resolver(None, None)
    assert res.call_tree_length() == 2


def test_same_function_different_closure():
    arrx = np.full([20], 1)
    arry = np.full([20], 2)

    @dace.program
    def nested(A: dace.float64[20], dir: dace.constant):
        if dir == 'x':
            return A + arrx
        elif dir == 'y':
            return A * arry
        return A + 3

    @dace.program
    def mainprog(A: dace.float64[20]):
        B = nested(A, 'x')
        return nested(B, 'y')

    closure = mainprog.closure_resolver(None, None)
    assert closure.call_tree_length() == 3
    assert len(closure.closure_arrays) == 2  # arrx and arry should appear once

    A = np.random.rand(20)
    expected = (A + 1) * 2
    res = mainprog(A)
    assert np.allclose(res, expected)
    assert len(mainprog.resolver.closure_arrays) == 2


if __name__ == '__main__':
    test_nested_objects_same_name()
    test_calltree()
    test_same_function_different_closure()
