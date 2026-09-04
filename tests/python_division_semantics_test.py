# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that ``//`` and ``%`` in a tasklet follow Python (hence NumPy) semantics. """
import itertools

import numpy as np
import pytest

import dace

N = dace.symbol('N', dtype=dace.int64)

# Both signs, and divisors that do and do not divide evenly: the disagreement between Python and C
# needs a nonzero remainder and operands of opposite sign.
VALUES = (-32, -7, -3, -1, 1, 3, 7, 32)

DTYPES = (('int32', dace.int32, np.int32), ('int64', dace.int64, np.int64), ('float32', dace.float32, np.float32),
          ('float64', dace.float64, np.float64))

TARGETS = (pytest.param('host'), pytest.param('device', marks=pytest.mark.gpu))


def division_program(op: str, dtype):
    """ ``out = a <op> b`` elementwise, as a map so that the GPU lowering is a real kernel. """
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
    """ Python rounds the quotient toward negative infinity, so the remainder takes the divisor's
        sign: ``-32 // 7 == -5`` and ``-32 % 7 == 3``. C rounds toward zero and answers ``-4`` for
        both.

        The GPU half of the table is not redundant: the correction term is a branch, and that branch
        used to hold a call to host-only ``std::div``. nvcc reports such a call with a warning rather
        than an error and then removes the region around it, so the kernel launched, reported
        success, and stored nothing.
    """
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
    """ The numeric table above catches an infix ``%`` on integers, but not on floats, where the
        emitted line does not compile and there is no number left to compare.
    """
    code = division_program(op, dace.int64).to_sdfg().generate_code()[0].clean_code
    assert call in code, f'{op} lowered without {call}, so it lowered infix'


if __name__ == '__main__':
    for operator in ('//', '%'):
        for dtype_name, dace_type, numpy_type in DTYPES:
            test_floor_division_and_modulo_agree_with_numpy(operator, dtype_name, dace_type, numpy_type, 'host')
    test_neither_operator_is_emitted_infix('//', 'py_floor(')
    test_neither_operator_is_emitted_infix('%', 'py_mod(')
