# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Division and modulo semantics of the Python frontend, tasklets, symbolic expressions and inter-state edges. """
import itertools

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic

N = dace.symbol('N', dtype=dace.int64)
I = dace.symbol('I', dtype=dace.int64)
K = dace.symbol('K', nonnegative=True)
M = dace.symbol('M', positive=True)

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
    """ ``-32 // 7 == -5`` and ``-32 % 7 == 3``, where C gives ``-4`` for both. """
    a, b = operand_pairs(nptype)
    expected = (a // b) if op == '//' else (a % b)

    label = 'floordiv' if op == '//' else 'mod'
    sdfg = division_program(op, dtype).to_sdfg()
    sdfg.name = f'division_{label}_{name}_{target}'
    if target == 'device':
        sdfg.apply_gpu_transformations()
    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])

    disagreements = [(a[i], b[i], out[i], expected[i]) for i in range(a.shape[0]) if out[i] != expected[i]]
    assert not disagreements, f'{op} on {name}: {disagreements[:4]}'


@pytest.mark.parametrize('op,call', (('//', 'py_floor('), ('%', 'py_mod(')))
def test_neither_operator_is_emitted_infix(op, call):
    """ An infix ``%`` on floats does not compile, so the numeric table cannot catch it. """
    sdfg = division_program(op, dace.int64).to_sdfg()
    sdfg.name = 'division_code_floordiv' if op == '//' else 'division_code_mod'
    code = sdfg.generate_code()[0].clean_code
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


@pytest.mark.parametrize('text,value',
                         (('(-7) % 3', -1), ('7 % (-3)', 1), ('CMod(-7, 3)', -1), ('FtnMod(-7, 3)', -1),
                          ('Mod(-7, 3)', 2), ('PyMod(-7, 3)', 2), ('PyMod(7, -3)', -2), ('FtnModulo(-7, 3)', 2)))
def test_a_constant_modulo_folds_with_the_rounding_its_spelling_names(text, value):
    """``%``, ``CMod`` and ``FtnMod`` truncate; ``Mod``, ``PyMod`` and ``FtnModulo`` floor."""
    assert symbolic.pystr_to_symbolic(text) == value


def test_a_symbolic_modulo_reads_back_from_its_string_with_its_own_rounding():
    """ A saved SDFG must load with the rounding it was saved with. """
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
    """The two roundings agree there, so simplification sees a single form."""
    assert isinstance(symbolic.pystr_to_symbolic('CMod(K, M)', symbol_map={'K': K, 'M': M}), sympy.Mod)


def elementwise_tasklet_sdfg(name: str, code: str, dtype) -> dace.SDFG:
    """``z = <code>`` over ``a`` and ``b``, built with the SDFG API rather than a frontend."""
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
    """A tasklet's ``%`` is C's, whatever language its code is written in; the named functions pick the others."""
    name, dtype, nptype = types
    a, b = operand_pairs(nptype)
    sdfg = elementwise_tasklet_sdfg(f'tasklet_modulo_{label}_{name}', code, dtype)

    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])

    assert np.array_equal(out, reference(a, b))


def interstate_modulo_sdfg(name: str, assignment: str) -> dace.SDFG:
    """Assigns ``k = <assignment>`` on an inter-state edge and stores ``k``."""
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
    """A symbolic ``%`` is C's too; ``PyMod`` floors."""
    sdfg = interstate_modulo_sdfg(f'interstate_modulo_{label}', assignment)

    out = np.zeros(1, dtype=np.int64)
    sdfg(out=out, I=0)

    assert out[0] == value


@dace.program
def floored_condition(out: dace.int64[1]):
    if (I - 3) % 4 == 1:
        out[0] = 1
    else:
        out[0] = 0


def test_a_python_program_modulo_in_a_condition_floors():
    """``(0 - 3) % 4`` is ``1`` in Python and ``-3`` in C."""
    out = np.zeros(1, dtype=np.int64)

    floored_condition(out=out, I=0)

    assert out[0] == 1


if __name__ == '__main__':
    for operator in ('//', '%'):
        for dtype_name, dace_type, numpy_type in DTYPES:
            test_floor_division_and_modulo_agree_with_numpy(operator, dtype_name, dace_type, numpy_type, 'host')
    test_neither_operator_is_emitted_infix('//', 'py_floor(')
    test_neither_operator_is_emitted_infix('%', 'py_mod(')
    for size in (1, 2, 5, 7):
        test_a_negative_modulo_in_a_loop_bound_is_floored(size)
        test_a_negative_modulo_in_a_subscript_is_floored(size, 'host')
    test_a_symbolic_modulo_reads_back_from_its_string_with_its_own_rounding()
    test_the_c_modulo_of_a_nonnegative_dividend_and_a_positive_divisor_is_sympy_mod()
    for case in TASKLET_CASES:
        test_a_tasklet_modulo_computes_what_its_spelling_names(*case)
    test_a_python_program_modulo_in_a_condition_floors()
