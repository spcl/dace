# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for compile-time value tracking in the next-generation frontend:
symbolic values of ANF scalar temporaries (computed shape expressions),
constant bindings for non-representable compile-time objects (enum classes),
descriptor-attribute inference, and the local-array creation call family
(``dace.ndarray``, ``dace.define_local``, ``numpy.ndarray``).
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

S0 = dace.symbol('S0', dtype=dace.int64)
S1 = dace.symbol('S1', dtype=dace.int64)
S4 = dace.symbol('S4', dtype=dace.int64)


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_symbolic_shape_creation():
    """Computed symbolic shapes reach creation calls through ANF temps."""

    @dace.program
    def prog(inp: dace.float32[S0, S1], weights: dace.float32[S4, S4]):
        output = np.zeros((S0, S1 - S4 + 1), dtype=np.float32)
        output[0, 0] = inp[0, 0] + weights[0, 0]
        return output

    tree = nextgen.parse_program(prog)
    assert not _callbacks(tree)
    assert tuple(str(s) for s in tree.containers['output'].shape) == ('S0', 'S1 - S4 + 1')


def test_symbolic_shape_execution():

    @dace.program
    def prog(inp: dace.float64[S0]):
        out = np.zeros((S0 - 2, ), dtype=np.float64)
        for i in dace.map[0:S0 - 2]:
            out[i] = inp[i] + inp[i + 1] + inp[i + 2]
        return out

    tree = nextgen.parse_program(prog)
    assert not _callbacks(tree)
    func = tree.as_sdfg().compile()
    inp = np.random.rand(10)
    result = func(inp=inp, S0=10)
    assert np.allclose(result, inp[:-2] + inp[1:-1] + inp[2:])


def test_ndarray_with_lifetime():
    """dace.ndarray with an enum-valued lifetime keyword lowers with zero
    callbacks and the descriptor carries the lifetime."""

    @dace.program
    def prog(output: dace.int32[1]):
        tmp = dace.ndarray([1], output.dtype, lifetime=dace.AllocationLifetime.Persistent)
        tmp[0] = 5
        output[0] = tmp[0]

    tree = nextgen.parse_program(prog)
    assert not _callbacks(tree)
    descriptor = tree.containers['tmp']
    assert descriptor.dtype == dace.int32
    assert descriptor.lifetime == dace.AllocationLifetime.Persistent

    value = np.zeros([1], dtype=np.int32)
    tree.as_sdfg().compile()(output=value)
    assert value[0] == 5


def test_define_local_storage():

    @dace.program
    def prog(output: dace.float64[4]):
        tmp = dace.define_local([4], dace.float64, storage=dace.StorageType.CPU_Heap)
        for i in dace.map[0:4]:
            tmp[i] = i * 2.0
        output[:] = tmp

    tree = nextgen.parse_program(prog)
    assert not _callbacks(tree)
    assert tree.containers['tmp'].storage == dace.StorageType.CPU_Heap

    value = np.zeros([4])
    tree.as_sdfg().compile()(output=value)
    assert np.allclose(value, [0.0, 2.0, 4.0, 6.0])


def test_descriptor_attribute_arguments():
    """A.dtype and A.shape resolve as compile-time call arguments."""

    @dace.program
    def prog(inp: dace.float32[S0, S1]):
        out = np.zeros(inp.shape, dtype=inp.dtype)
        out[0, 0] = inp[0, 0]
        return out

    tree = nextgen.parse_program(prog)
    assert not _callbacks(tree)
    descriptor = tree.containers['out']
    assert descriptor.dtype == dace.float32
    assert tuple(str(s) for s in descriptor.shape) == ('S0', 'S1')


def test_mixed_symbol_dtype_promotion():
    """Symbol dtypes resolve by name through the context, so expressions over
    same-named cached sympy symbols do not crash type inference."""

    @dace.program
    def prog(inp: dace.float64[S1, S4]):
        out = np.zeros((S1 - S4, ), dtype=np.float64)
        out[0] = inp[0, 0]
        return out

    tree = nextgen.parse_program(prog)
    assert not _callbacks(tree)


if __name__ == '__main__':
    test_symbolic_shape_creation()
    test_symbolic_shape_execution()
    test_ndarray_with_lifetime()
    test_define_local_storage()
    test_descriptor_attribute_arguments()
    test_mixed_symbol_dtype_promotion()
