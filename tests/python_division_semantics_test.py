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

Division by zero is deliberately absent from the table and is a KNOWN divergence: numpy answers
``0`` for integers and ``inf``/``nan`` for floats, where the emitted C traps or is undefined.
Matching it would put a branch on every division, which is not a trade this makes silently.
"""
import itertools

import numpy as np
import pytest

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
