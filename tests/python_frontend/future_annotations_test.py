# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for the @dace.program frontend under PEP 563 (``from __future__ import
annotations``): every argument annotation in a module that carries the future import arrives at
the frontend as a plain string, and it must be resolved back into the real dace type using the
program's globals and closure before an SDFG is built.
"""
from __future__ import annotations

import numpy as np

import dace

N = dace.symbol('N')


@dace.program
def store_one(a: dace.float64[N], b: dace.int32[N]):
    a[0] = 1.0
    b[1] = 7


@dace.program
def add_scalar(a: dace.float64[N], c: dace.float64, out: dace.float64[N]):
    out[:] = a[:] + c


@dace.program
def inner_add(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    out[:] = a[:] + b[:]


@dace.program
def outer_add(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    inner_add(a, b, out)


def make_closure_program():
    """ Builds a @dace.program whose symbol and dtype live only in an enclosing function. """
    M = dace.symbol('M')
    elemtype = dace.float64

    @dace.program
    def closure_scale(x: elemtype[M], y: elemtype[M]):
        for i in dace.map[0:M]:
            y[i] = x[i] * elemtype(2)

    return closure_scale


def test_array_argument_with_symbolic_shape_resolves_under_future_annotations():
    """ The reproducer: a stringized Array[N] annotation must not collapse to a scalar pointer. """
    sdfg = store_one.to_sdfg(simplify=False)
    arglist = sdfg.arglist()
    assert isinstance(arglist['a'], dace.data.Array), arglist['a']
    assert isinstance(arglist['b'], dace.data.Array), arglist['b']

    a = np.zeros(4, dtype=np.float64)
    b = np.zeros(4, dtype=np.int32)
    store_one(a, b)
    assert a.tolist() == [1.0, 0.0, 0.0, 0.0], a.tolist()
    assert b.tolist() == [0, 7, 0, 0], b.tolist()


def test_scalar_argument_resolves_under_future_annotations():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = np.zeros(4, dtype=np.float64)
    add_scalar(a, 5.0, out)
    expected = a + 5.0
    assert out.tolist() == expected.tolist(), out.tolist()


def test_module_level_symbol_in_annotation_resolves_under_future_annotations():
    """ N is a module-level dace.symbol; the annotation string must resolve it from globals, not
    mint a fresh, disconnected symbol of the same name. """
    sdfg = store_one.to_sdfg(simplify=False)
    arglist = sdfg.arglist()
    assert str(arglist['a'].shape[0]) == 'N', arglist['a'].shape


def test_closure_local_symbol_and_dtype_resolve_under_future_annotations():
    """ A symbol and a dtype alias defined inside an enclosing function, not at module scope, must
    resolve via the program's closure -- get_type_hints alone only sees __globals__. """
    closure_scale = make_closure_program()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.zeros(5, dtype=np.float64)
    closure_scale(x, y)
    expected = x * 2.0
    assert y.tolist() == expected.tolist(), y.tolist()


def test_nested_dace_program_call_resolves_annotations_under_future_annotations():
    """ A @dace.program calling another @dace.program parses the callee's own stringized
    annotations too. """
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    outer_add(a, b, out)
    expected = a + b
    assert out.tolist() == expected.tolist(), out.tolist()
