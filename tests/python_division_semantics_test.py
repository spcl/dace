# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Division and modulo semantics (doc/sdfg/ir.rst, "Division and Modulo Semantics"): a Python program's ``//`` and
``%`` follow Python (hence NumPy), a bare ``%`` in an SDFG is C's, and ``PyMod`` / ``FtnMod`` / ``FtnModulo`` /
``CMod`` name the others. """
import itertools

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic

N = dace.symbol('N', dtype=dace.int64)
I = dace.symbol('I', dtype=dace.int64)

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
    sdfg.name = f'division_{"floordiv" if op == "//" else "mod"}_{name}_{target}'  # one build folder per case
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
    sdfg = division_program(op, dace.int64).to_sdfg()
    sdfg.name = f'division_code_{"floordiv" if op == "//" else "mod"}'
    code = sdfg.generate_code()[0].clean_code
    assert call in code, f'{op} lowered without {call}, so it lowered infix'


@pytest.mark.parametrize('text,value',
                         (('(-7) % 3', -1), ('7 % (-3)', 1), ('CMod(-7, 3)', -1), ('FtnMod(-7, 3)', -1),
                          ('Mod(-7, 3)', 2), ('PyMod(-7, 3)', 2), ('PyMod(7, -3)', -2), ('FtnModulo(-7, 3)', 2)))
def test_a_constant_modulo_folds_with_the_rounding_its_spelling_names(text, value):
    """ ``%``, ``CMod`` and ``FtnMod`` truncate; ``Mod``, ``PyMod`` and ``FtnModulo`` floor. """
    assert symbolic.pystr_to_symbolic(text) == value


def test_a_symbolic_modulo_reads_back_from_its_string_with_its_own_rounding():
    """ SymPy prints its floored ``Mod`` as ``Mod(a, b)`` and a bare ``%`` parses to C's modulo, so the two must not
        print alike: a saved SDFG would otherwise load with the other rounding. """
    floored = symbolic.pystr_to_symbolic('PyMod(I - 2, N)')
    truncating = symbolic.pystr_to_symbolic('(I - 2) % N')

    floored_back = symbolic.pystr_to_symbolic(str(floored))
    truncating_back = symbolic.pystr_to_symbolic(str(truncating))

    assert floored_back == floored and truncating_back == truncating
    assert floored_back.subs({I: 0, N: 5}) == 3
    assert truncating_back.subs({I: 0, N: 5}) == -2
    assert 'py_mod(' in symbolic.symstr(floored, cpp_mode=True)
    assert '%' in symbolic.symstr(truncating, cpp_mode=True)


def test_the_c_modulo_of_a_nonnegative_dividend_and_a_positive_divisor_is_sympy_mod():
    """ The two roundings agree there, so simplification sees a single form. """
    k = dace.symbol('k', nonnegative=True)
    m = dace.symbol('m', positive=True)

    assert isinstance(symbolic.pystr_to_symbolic('CMod(k, m)', symbol_map={'k': k, 'm': m}), sympy.Mod)


def elementwise_tasklet_sdfg(name: str, code: str, dtype) -> dace.SDFG:
    """ ``z = <code>`` over ``a`` and ``b``, built with the SDFG API rather than a frontend. """
    sdfg = dace.SDFG(name)
    for array in ('a', 'b', 'out'):
        sdfg.add_array(array, [N], dtype)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('elements', dict(i='0:N'))
    tasklet = state.add_tasklet('remainder', {'x': dtype, 'y': dtype}, {'z': dtype}, code)
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='x', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(state.add_read('b'), entry, tasklet, dst_conn='y', memlet=dace.Memlet('b[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('out'), src_conn='z', memlet=dace.Memlet('out[i]'))
    return sdfg


INT64 = ('int64', dace.int64, np.int64)
FLOAT64 = ('float64', dace.float64, np.float64)
TASKLET_CASES = (('percent', 'z = x % y', INT64, np.fmod), ('cmod', 'z = CMod(x, y)', INT64, np.fmod),
                 ('cmod', 'z = CMod(x, y)', FLOAT64, np.fmod), ('ftnmod', 'z = FtnMod(x, y)', INT64, np.fmod),
                 ('ftnmod', 'z = FtnMod(x, y)', FLOAT64, np.fmod), ('pymod', 'z = PyMod(x, y)', INT64, np.mod),
                 ('pymod', 'z = PyMod(x, y)', FLOAT64, np.mod), ('ftnmodulo', 'z = FtnModulo(x, y)', INT64, np.mod),
                 ('ftnmodulo', 'z = FtnModulo(x, y)', FLOAT64, np.mod))


@pytest.mark.parametrize('label,code,types,reference', TASKLET_CASES)
def test_a_tasklet_modulo_computes_what_its_spelling_names(label, code, types, reference):
    """ A tasklet's ``%`` is C's, whatever language its code is written in; the named functions pick the others. """
    name, dtype, nptype = types
    a, b = operand_pairs(nptype)
    sdfg = elementwise_tasklet_sdfg(f'tasklet_modulo_{label}_{name}', code, dtype)

    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])

    assert np.array_equal(out, reference(a, b))


def interstate_modulo_sdfg(name: str, assignment: str) -> dace.SDFG:
    """ Assigns ``k = <assignment>`` on an inter-state edge and stores ``k``. """
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('I', dace.int64)
    sdfg.add_array('out', [1], dace.int64)
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'k': assignment}))
    tasklet = second.add_tasklet('store', {}, {'o': dace.int64}, 'o = k')
    second.add_edge(tasklet, 'o', second.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


@pytest.mark.parametrize('label,assignment,value', (('percent', '(I - 7) % 3', -1), ('pymod', 'PyMod(I - 7, 3)', 2)))
def test_an_interstate_edge_modulo_computes_what_its_spelling_names(label, assignment, value):
    """ A symbolic ``%`` is C's too; ``PyMod`` floors. """
    sdfg = interstate_modulo_sdfg(f'interstate_modulo_{label}', assignment)

    out = np.zeros(1, dtype=np.int64)
    sdfg(out=out, I=0)

    assert out[0] == value


@dace.program
def floored_subscript(a: dace.int64[N], out: dace.int64[1]):
    out[0] = a[(I - 3) % N]


@dace.program
def floored_condition(out: dace.int64[1]):
    if (I - 3) % 4 == 1:
        out[0] = 1
    else:
        out[0] = 0


def test_a_python_program_modulo_in_a_subscript_floors():
    """ ``(0 - 3) % 8`` is ``5`` in Python and ``-3`` in C, an out-of-bounds read. """
    a = np.arange(8, dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)

    floored_subscript(a=a, out=out, I=0, N=8)

    assert out[0] == 5


def test_a_python_program_modulo_in_a_condition_floors():
    """ ``(0 - 3) % 4`` is ``1`` in Python and ``-3`` in C. """
    out = np.zeros(1, dtype=np.int64)

    floored_condition(out=out, I=0)

    assert out[0] == 1


if __name__ == '__main__':
    for operator in ('//', '%'):
        for dtype_name, dace_type, numpy_type in DTYPES:
            test_floor_division_and_modulo_agree_with_numpy(operator, dtype_name, dace_type, numpy_type, 'host')
    test_neither_operator_is_emitted_infix('//', 'py_floor(')
    test_neither_operator_is_emitted_infix('%', 'py_mod(')
    test_a_symbolic_modulo_reads_back_from_its_string_with_its_own_rounding()
    test_the_c_modulo_of_a_nonnegative_dividend_and_a_positive_divisor_is_sympy_mod()
    for case in TASKLET_CASES:
        test_a_tasklet_modulo_computes_what_its_spelling_names(*case)
    test_a_python_program_modulo_in_a_subscript_floors()
    test_a_python_program_modulo_in_a_condition_floors()
