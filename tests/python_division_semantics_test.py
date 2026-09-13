# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``//`` and ``%`` in a tasklet mean what they mean in Python, hence in numpy.

Both round the quotient toward NEGATIVE INFINITY, so the remainder takes the divisor's sign:
``-32 // 7`` is ``-5`` and ``-32 % 7`` is ``3``. C rounds toward zero and gives the remainder the
dividend's sign (``-4`` and ``-4``), and C has no ``%`` for floating point at all. Neither operator
can therefore be written infix, and both were: ``//`` came out as ``ifloor(a / b)``, where the
integer division has already truncated and flooring an integer changes nothing, and ``%`` came out
as a bare ``%``, which answered ``-4`` on integers and failed to compile on floats.

The device half of the table is not redundant with the host half. The correction term is a BRANCH,
and the branch is what once held a call to host-only ``std::div``: nvcc answers a host call from
device code with warning #20011 rather than an error and deletes the region around it, so the kernel
launched, returned success, and stored nothing (tsvc ``s315``). Only running it on the device says
the branch survived.

Both are halves of ``py_divmod``, which answers a zero divisor as numpy does (``0`` for integers,
``inf``/``nan`` for floats); that and the other special values are tabled in
``tests/numpy/ufunc_multi_output_test.py``.
"""
import itertools

import numpy as np
import pytest
import sympy

import dace

N = dace.symbol('N', dtype=dace.int64)

#: Both signs, a divisor that divides evenly and ones that do not: the disagreement needs a nonzero
#: remainder and operands of opposite sign, and every such combination is here.
VALUES = (-32, -7, -3, -1, 1, 3, 7, 32)

DTYPES = (('int32', dace.int32, np.int32), ('int64', dace.int64, np.int64), ('float32', dace.float32, np.float32),
          ('float64', dace.float64, np.float64))

TARGETS = (pytest.param('host'), pytest.param('device', marks=pytest.mark.gpu))


def division_program(op: str, dtype) -> dace.frontend.python.parser.DaceProgram:
    """``out = a <op> b`` elementwise, as a map so the device lowering is a real kernel."""
    if op == '//':

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            for i in dace.map[0:N]:
                out[i] = a[i] // b[i]
    else:

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            for i in dace.map[0:N]:
                out[i] = a[i] % b[i]

    return prog


def operand_pairs(nptype):
    pairs = list(itertools.product(VALUES, VALUES))
    return (np.array([x for x, _ in pairs], dtype=nptype), np.array([y for _, y in pairs], dtype=nptype))


@pytest.mark.parametrize('op', ('//', '%'))
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
@pytest.mark.parametrize('target', TARGETS)
def test_floor_division_and_modulo_agree_with_numpy(op, name, dtype, nptype, target):
    """Every sign combination, every dtype, both targets, against numpy itself."""
    a, b = operand_pairs(nptype)
    expected = (a // b) if op == '//' else (a % b)

    sdfg = division_program(op, dtype).to_sdfg()
    if target == 'device':
        sdfg.apply_gpu_transformations()
    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])

    disagreements = [(a[i], b[i], out[i], expected[i]) for i in range(a.shape[0]) if out[i] != expected[i]]
    assert not disagreements, f'{op} on {name}: {disagreements[:4]}'


@pytest.mark.parametrize('op,call', (('//', 'py_floor('), ('%', 'py_mod(')))
def test_neither_operator_is_emitted_infix(op, call):
    """The emitted text is the product here: infix is what C means by these, not what Python does.

    The numeric table above would catch an infix ``%`` on integers but not on floats, where the
    emitted line does not compile at all and there is no number to compare.
    """
    code = division_program(op, dace.int64).to_sdfg().generate_code()[0].clean_code
    assert call in code, f'{op} lowered without {call}, so it lowered infix'


@dace.program
def shifted_start(A: dace.int64[N], B: dace.int64[N, N]):
    for i in range(N):
        for j in range((i - 2) % N, N):
            B[i, j] = A[j] + i


@dace.program
def wrapped_read(A: dace.int64[N], B: dace.int64[N]):
    for i in dace.map[0:N]:
        B[i] = A[(i - 3) % N]


@pytest.mark.parametrize('n', (1, 2, 5, 7))
def test_a_negative_modulo_in_a_loop_bound_is_floored(n):
    """``% N`` in a range reaches C as sympy's ``Mod``, not as a tasklet ``%``. Truncated, ``(0 - 2) % 7``
    starts the loop at -2 and writes before the row."""
    A = np.arange(1, n + 1, dtype=np.int64)
    expected = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range((i - 2) % n, n):
            expected[i, j] = A[j] + i

    out = np.zeros((n, n), dtype=np.int64)
    shifted_start(A, out, N=n)

    assert np.array_equal(out, expected), (out, expected)


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('n', (1, 2, 5, 7))
def test_a_negative_modulo_in_a_subscript_is_floored(n, target):
    """A subscript ``(i - 3) % N`` is printed from the memlet's symbolic ``Mod``; truncated, it reads
    ``A[-3]``."""
    A = np.arange(1, n + 1, dtype=np.int64)
    expected = np.array([A[(i - 3) % n] for i in range(n)], dtype=np.int64)

    sdfg = wrapped_read.to_sdfg()
    if target == 'device':
        sdfg.apply_gpu_transformations()
    out = np.zeros(n, dtype=np.int64)
    sdfg(A=A, B=out, N=n)

    assert np.array_equal(out, expected), (out, expected)


I = dace.symbol('I')
K = dace.symbol('K', nonnegative=True)
M = dace.symbol('M', positive=True)


@pytest.mark.parametrize('dividend,divisor,infix', (
    (I - 2, N, False),
    (K, N, False),
    (I, M, False),
    (K, M, True),
))
def test_sympy_mod_prints_the_c_operator_only_where_no_sign_can_differ(dividend, divisor, infix):
    """Floored and truncated remainders agree exactly on a nonnegative dividend and a positive
    divisor; anywhere else the infix ``%`` is a different operation."""
    printed = dace.symbolic.symstr(sympy.Mod(dividend, divisor, evaluate=False), cpp_mode=True)
    assert ('%' in printed and 'py_mod' not in printed) == infix, printed
    assert ('py_mod(' in printed) == (not infix), printed
