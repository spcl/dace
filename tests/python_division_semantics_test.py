# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" ``%`` and ``//`` in the Python frontend, tasklets, inter-state edges, memlets and map ranges, against NumPy. """
import itertools

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic
from dace.transformation.passes.nonnegative_map_symbols import NonnegativeMapSymbols

N = dace.symbol('N', dtype=dace.int64)

VALUES = (-32, -7, -3, -1, 1, 3, 7, 32)

DTYPES = (('int32', dace.int32, np.int32), ('int64', dace.int64, np.int64), ('float32', dace.float32, np.float32),
          ('float64', dace.float64, np.float64))
INTEGER_DTYPES = DTYPES[:2]

TARGETS = (pytest.param('host'), pytest.param('device', marks=pytest.mark.gpu))

# (label, spelling of ``a`` and ``b`` in Python code or a symbolic expression, NumPy function with the same rounding)
SPELLINGS = (('percent', '{a} % {b}', np.mod), ('slashes', '{a} // {b}', np.floor_divide),
             ('PyMod', 'PyMod({a}, {b})', np.mod), ('PyFloor', 'PyFloor({a}, {b})',
                                                    np.floor_divide), ('CMod', 'CMod({a}, {b})', np.fmod))
OPERATORS = (('%', np.mod, 'mod'), ('//', np.floor_divide, 'floordiv'))

SYMBOLIC_VALUES = tuple(itertools.product((-7, 7), (-3, 3)))


def operand_pairs(nptype):
    pairs = list(itertools.product(VALUES, VALUES))
    return np.array([x for x, _ in pairs], dtype=nptype), np.array([y for _, y in pairs], dtype=nptype)


def operator_program(op: str, dtype, whole_arrays: bool):
    """ ``out = a <op> b`` on whole arrays, or elementwise in a map so that the GPU lowering is a real kernel. """
    if whole_arrays and op == '//':

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            out[:] = a // b
    elif whole_arrays:

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            out[:] = a % b
    elif op == '//':

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


def run_elementwise(sdfg: dace.SDFG, nptype, target: str = 'host'):
    a, b = operand_pairs(nptype)
    if target == 'device':
        sdfg.apply_gpu_transformations()
    out = np.zeros(a.shape, dtype=nptype)
    sdfg(a=a, b=b, out=out, N=a.shape[0])
    return a, b, out


def assert_matches(a, b, out, expected, what: str):
    disagreements = [(a[i], b[i], out[i], expected[i]) for i in range(a.shape[0]) if out[i] != expected[i]]
    assert not disagreements, f'{what}: (a, b, got, want) {disagreements[:4]}'


@pytest.mark.parametrize('whole_arrays', (True, False))
@pytest.mark.parametrize('op,reference,label', OPERATORS)
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
@pytest.mark.parametrize('target', TARGETS)
def test_the_frontend_floors_modulo_and_division_like_numpy(whole_arrays, op, reference, label, name, dtype, nptype,
                                                            target):
    """ ``-32 // 7 == -5`` and ``-32 % 7 == 3``, where C gives ``-4`` for both. """
    sdfg = operator_program(op, dtype, whole_arrays).to_sdfg()
    sdfg.name = f'frontend_{label}_{name}_{target}_{"arrays" if whole_arrays else "map"}'

    a, b, out = run_elementwise(sdfg, nptype, target)

    assert_matches(a, b, out, reference(a, b), f'{op} on {name}')


@pytest.mark.parametrize('op,reference,label', OPERATORS)
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
def test_the_frontend_floors_augmented_modulo_and_division_like_numpy(op, reference, label, name, dtype, nptype):
    if op == '//':

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            out[:] = a
            out //= b
    else:

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            out[:] = a
            out %= b

    sdfg = prog.to_sdfg()
    sdfg.name = f'frontend_augmented_{label}_{name}'

    a, b, out = run_elementwise(sdfg, nptype)

    assert_matches(a, b, out, reference(a, b), f'{op}= on {name}')


@pytest.mark.parametrize('op,call', (('//', 'py_floor('), ('%', 'py_mod(')))
def test_the_frontend_does_not_emit_an_infix_operator(op, call):
    """ An infix ``%`` on floats does not compile, so the numeric table cannot catch it. """
    sdfg = operator_program(op, dace.int64, whole_arrays=False).to_sdfg()
    sdfg.name = 'frontend_infix_floordiv' if op == '//' else 'frontend_infix_mod'

    assert call in sdfg.generate_code()[0].clean_code


def tasklet_sdfg(name: str, code: str, dtype) -> dace.SDFG:
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


@pytest.mark.parametrize('label,spelling,reference', SPELLINGS)
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
def test_a_python_tasklet_names_the_rounding_of_its_modulo_and_division(label, spelling, reference, name, dtype,
                                                                        nptype):
    sdfg = tasklet_sdfg(f'tasklet_{label}_{name}', 'z = ' + spelling.format(a='x', b='y'), dtype)

    a, b, out = run_elementwise(sdfg, nptype)

    assert_matches(a, b, out, reference(a, b), f'{spelling} on {name}')


@pytest.mark.parametrize('name,dtype,nptype', INTEGER_DTYPES)
def test_a_cpp_tasklet_keeps_cs_percent(name, dtype, nptype):
    sdfg = tasklet_sdfg(f'tasklet_cpp_percent_{name}', 'z = x % y;', dtype)
    tasklet = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.Tasklet))
    tasklet.code = dace.properties.CodeBlock('z = x % y;', dace.Language.CPP)

    a, b, out = run_elementwise(sdfg, nptype)

    assert_matches(a, b, out, np.fmod(a, b), f'% in a C++ tasklet on {name}')


@pytest.mark.parametrize('op,reference,label', OPERATORS)
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
def test_a_tasklet_of_a_python_program_has_pythons_semantics(op, reference, label, name, dtype, nptype):
    if op == '//':

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            for i in dace.map[0:N]:
                with dace.tasklet:
                    x << a[i]
                    y << b[i]
                    z >> out[i]
                    z = x // y
    else:

        @dace.program
        def prog(a: dtype[N], b: dtype[N], out: dtype[N]):
            for i in dace.map[0:N]:
                with dace.tasklet:
                    x << a[i]
                    y << b[i]
                    z >> out[i]
                    z = x % y

    sdfg = prog.to_sdfg()
    sdfg.name = f'program_tasklet_{label}_{name}'

    a, b, out = run_elementwise(sdfg, nptype)

    assert_matches(a, b, out, reference(a, b), f'{op} in a tasklet on {name}')


def edge_condition_sdfg(name: str, condition: str, dtype) -> dace.SDFG:
    """ Stores 1 if ``condition`` holds on the symbols ``a``, ``b`` and ``want``, else 0. """
    sdfg = dace.SDFG(name)
    for symbol in ('a', 'b', 'want'):
        sdfg.add_symbol(symbol, dtype)
    sdfg.add_array('out', [1], dace.int32)
    start = sdfg.add_state('start', is_start_block=True)
    for label, taken, value in (('yes', condition, 1), ('no', f'not ({condition})', 0)):
        state = sdfg.add_state(label)
        tasklet = state.add_tasklet('store', {}, {'o': dace.int32}, f'o = {value}')
        state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))
        sdfg.add_edge(start, state, dace.InterstateEdge(condition=taken))
    return sdfg


def edge_holds(sdfg: dace.SDFG, nptype, a, b, want) -> bool:
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out, a=nptype(a), b=nptype(b), want=nptype(want))
    return bool(out[0])


@pytest.mark.parametrize('label,spelling,reference', SPELLINGS)
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
def test_an_interstate_edge_names_the_rounding_of_its_modulo_and_division(label, spelling, reference, name, dtype,
                                                                          nptype):
    sdfg = edge_condition_sdfg(f'edge_{label}_{name}', spelling.format(a='a', b='b') + ' == want', dtype)

    for a, b in SYMBOLIC_VALUES:
        want = reference(nptype(a), nptype(b))
        assert edge_holds(sdfg, nptype, a, b, want), f'{spelling} with ({a}, {b}) on {name}'
        assert not edge_holds(sdfg, nptype, a, b, want + 1), f'{spelling} with ({a}, {b}) on {name}'


@pytest.mark.parametrize('op,reference,label', OPERATORS)
@pytest.mark.parametrize('name,dtype,nptype', DTYPES)
def test_a_python_program_condition_has_pythons_semantics(op, reference, label, name, dtype, nptype):
    if op == '//':

        @dace.program
        def prog(a: dtype, b: dtype, want: dtype, out: dace.int32[1]):
            if a // b == want:
                out[0] = 1
            else:
                out[0] = 0
    else:

        @dace.program
        def prog(a: dtype, b: dtype, want: dtype, out: dace.int32[1]):
            if a % b == want:
                out[0] = 1
            else:
                out[0] = 0

    sdfg = prog.to_sdfg()
    sdfg.name = f'program_condition_{label}_{name}'

    for a, b in SYMBOLIC_VALUES:
        out = np.zeros(1, dtype=np.int32)
        sdfg(a=nptype(a), b=nptype(b), want=reference(nptype(a), nptype(b)), out=out)
        assert out[0] == 1, f'{op} on {name}: ({a}, {b})'


def subset_sdfg(name: str, subscript: str, dtype) -> dace.SDFG:
    """ Reads ``A[<subscript> + 32]``, with the subscript a function of the symbols ``a`` and ``b``. """
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('a', dtype)
    sdfg.add_symbol('b', dtype)
    sdfg.add_array('A', [64], dace.int64)
    sdfg.add_array('out', [1], dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('copy', {'x'}, {'z'}, 'z = x')
    state.add_edge(state.add_read('A'), None, tasklet, 'x', dace.Memlet(f'A[{subscript} + 32]'))
    state.add_edge(tasklet, 'z', state.add_write('out'), None, dace.Memlet('out[0]'))
    return sdfg


def range_sdfg(name: str, bound: str, dtype) -> dace.SDFG:
    """ Sets ``out[i] = 1`` for ``i`` in ``0:<bound> + 8``, with the bound a function of ``a`` and ``b``. """
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('a', dtype)
    sdfg.add_symbol('b', dtype)
    sdfg.add_array('out', [64], dace.int64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('fill',
                             dict(i=f'0:{bound} + 8'), {},
                             'z = 1', {'z': dace.Memlet('out[i]')},
                             external_edges=True)
    return sdfg


@pytest.mark.parametrize('label,spelling,reference', SPELLINGS)
@pytest.mark.parametrize('name,dtype,nptype', INTEGER_DTYPES)
def test_a_memlet_and_a_map_range_name_the_rounding_of_their_modulo_and_division(label, spelling, reference, name,
                                                                                 dtype, nptype):
    expression = spelling.format(a='a', b='b')
    subset = subset_sdfg(f'subset_{label}_{name}', expression, dtype)
    extent = range_sdfg(f'range_{label}_{name}', expression, dtype)

    for a, b in SYMBOLIC_VALUES:
        want = int(reference(nptype(a), nptype(b)))
        out = np.zeros(1, dtype=np.int64)
        subset(A=np.arange(64, dtype=np.int64), out=out, a=nptype(a), b=nptype(b))
        assert out[0] == want + 32, f'{expression} with ({a}, {b}) as a subscript on {name}'
        filled = np.zeros(64, dtype=np.int64)
        extent(out=filled, a=nptype(a), b=nptype(b))
        assert filled.sum() == want + 8, f'{expression} with ({a}, {b}) as a range bound on {name}'


@pytest.mark.parametrize('name,dtype,nptype', INTEGER_DTYPES)
def test_a_python_program_subscript_and_range_have_pythons_semantics(name, dtype, nptype):

    @dace.program
    def subscript(A: dace.int64[64], out: dace.int64[1], a: dtype, b: dtype):
        out[0] = A[a % b + 32] + A[a // b + 32]

    @dace.program
    def extent(out: dace.int64[64], a: dtype, b: dtype):
        for i in dace.map[0:a % b + 8]:
            out[i] = 1
        for i in dace.map[0:a // b + 8]:
            out[i] += 1

    subscript_sdfg = subscript.to_sdfg()
    subscript_sdfg.name = f'program_subscript_{name}'
    extent_sdfg = extent.to_sdfg()
    extent_sdfg.name = f'program_range_{name}'

    for a, b in SYMBOLIC_VALUES:
        want = int(np.mod(nptype(a), nptype(b))) + int(np.floor_divide(nptype(a), nptype(b)))
        out = np.zeros(1, dtype=np.int64)
        subscript_sdfg(A=np.arange(64, dtype=np.int64), out=out, a=nptype(a), b=nptype(b))
        assert out[0] == want + 64, f'subscript with ({a}, {b}) on {name}'
        filled = np.zeros(64, dtype=np.int64)
        extent_sdfg(out=filled, a=nptype(a), b=nptype(b))
        assert filled.sum() == want + 16, f'range with ({a}, {b}) on {name}'


CONSTANTS = (('(-7) % 3', 2), ('7 % (-3)', -2), ('CMod(-7, 3)', -1), ('FtnMod(-7, 3)', -1), ('Mod(-7, 3)', 2),
             ('PyMod(-7, 3)', 2), ('PyMod(7, -3)', -2), ('FtnModulo(-7, 3)', 2), ('(-7) // 3', -3), ('7 // (-3)', -3),
             ('PyFloor(-7, 3)', -3), ('int_floor(7, -3)', -3))
READ_BACK = ('PyMod(I - 2, N)', '(I - 2) % N', 'PyFloor(I - 2, N)', '(I - 2) // N', 'CMod(I - 2, N)')


@pytest.mark.parametrize('text,value', CONSTANTS)
def test_a_constant_folds_with_the_rounding_its_spelling_names(text, value):
    assert symbolic.pystr_to_symbolic(text) == value


@pytest.mark.parametrize('function,dividend,divisor,value', (('CMod', -7.5, 2, -1.5), ('PyMod', -7.5, 2, 0.5)))
def test_a_floating_point_constant_folds_with_the_rounding_its_spelling_names(function, dividend, divisor, value):
    assert symbolic.pystr_to_symbolic(f'{function}({dividend}, {divisor})') == value


@pytest.mark.parametrize('text', READ_BACK)
def test_a_symbolic_expression_reads_back_from_its_string_with_its_own_rounding(text):
    """ A saved SDFG must load with the rounding it was saved with. """
    i, n = dace.symbol('I', dtype=dace.int64), dace.symbol('N', dtype=dace.int64)
    expression = symbolic.pystr_to_symbolic(text)

    for rendered, parse in ((str(expression), symbolic.pystr_to_symbolic), (symbolic.symstr(expression),
                                                                            symbolic.pystr_to_symbolic),
                            (symbolic.serialize_symbolic(expression), symbolic.deserialize_symbolic)):
        back = parse(rendered)
        assert back == expression, rendered
        assert back.subs({i: 0, n: 5}) == expression.subs({i: 0, n: 5}), rendered


def test_the_sympy_mod_is_floored_and_prints_as_such():
    mod = sympy.Mod(dace.symbol('I', dtype=dace.int64) - 2, dace.symbol('N', dtype=dace.int64))

    assert 'py_mod(' in symbolic.symstr(mod, cpp_mode=True)
    assert symbolic.deserialize_symbolic(symbolic.serialize_symbolic(mod)) == mod


def test_a_provably_nonnegative_pair_prints_as_the_c_operator():
    k = dace.symbol('k', dtype=dace.uint32)
    m = dace.symbol('m', dtype=dace.int64)
    p = dace.symbol('p', dtype=dace.int64, positive=True)

    assert '(p) % (4)' in symbolic.symstr(symbolic.PyMod(p, 4), cpp_mode=True)
    assert '(p) / (4)' in symbolic.symstr(symbolic.PyFloor(p, 4), cpp_mode=True)

    assert '(k) % (4)' in symbolic.symstr(symbolic.PyMod(k, 4), cpp_mode=True)
    assert '(k) / (4)' in symbolic.symstr(symbolic.PyFloor(k, 4), cpp_mode=True)
    assert 'py_mod(m, 4)' in symbolic.symstr(symbolic.PyMod(m, 4), cpp_mode=True)
    assert 'py_floor(m, 4)' in symbolic.symstr(symbolic.PyFloor(m, 4), cpp_mode=True)
    assert 'py_floor(m, 4)' in symbolic.symstr(symbolic.pystr_to_symbolic('m // 4'), cpp_mode=True)


def test_the_c_modulo_of_a_nonnegative_dividend_and_a_positive_divisor_is_the_floored_one():
    k = dace.symbol('k', nonnegative=True, integer=True)
    m = dace.symbol('m', positive=True, integer=True)

    assert symbolic.CMod(k, m) == symbolic.PyMod(k, m)


@dace.program
def periodic_read(A: dace.int64[N], B: dace.int64[N]):
    for i in dace.map[0:N]:
        B[i] = A[(i + 1) % N]


@dace.program
def shifted_read(A: dace.int64[N], B: dace.int64[1], s: dace.int64):
    B[0] = A[(s + 1) % N]


def test_a_map_parameter_and_the_extent_of_its_range_keep_cs_percent():
    sdfg = periodic_read.to_sdfg()
    sdfg.name = 'periodic_read'

    assert 'py_mod(' not in sdfg.generate_code()[0].clean_code
    a = np.arange(8, dtype=np.int64)
    b = np.zeros(8, dtype=np.int64)
    sdfg(A=a, B=b, N=8)
    assert np.array_equal(b, np.roll(a, -1))


def test_the_pass_declares_the_parameter_and_the_extent_of_a_map_nonnegative():
    sdfg = periodic_read.to_sdfg()
    sdfg.name = 'periodic_read_pass'

    assert NonnegativeMapSymbols().apply_pass(sdfg, {}) == 1

    inner = next(edge.data for state in sdfg.states() for edge in state.edges()
                 if edge.data.data == 'A' and edge.data.subset.free_symbols == {'i', 'N'})
    assert all(symbol.is_nonnegative for symbol in inner.subset.ranges[0][0].free_symbols)


def test_the_pass_leaves_a_map_starting_at_a_symbol_alone():

    @dace.program
    def shifted_map(A: dace.int64[N], s: dace.int64):
        for i in dace.map[s:N]:
            A[i % N] = 1

    sdfg = shifted_map.to_sdfg()
    sdfg.name = 'shifted_map'

    assert NonnegativeMapSymbols().apply_pass(sdfg, {}) is None


def test_a_free_symbol_is_not_assumed_nonnegative():
    sdfg = shifted_read.to_sdfg()
    sdfg.name = 'shifted_read'

    assert 'py_mod(' in sdfg.generate_code()[0].clean_code


def test_the_modulo_functions_have_one_implementation_per_rounding():
    assert symbolic.FtnMod is symbolic.CMod
    assert symbolic.FtnModulo is symbolic.PyMod
    assert symbolic.PyFloor is symbolic.int_floor


if __name__ == '__main__':
    for whole_arrays in (True, False):
        for op, reference, label in OPERATORS:
            for name, dtype, nptype in DTYPES:
                test_the_frontend_floors_modulo_and_division_like_numpy(whole_arrays, op, reference, label, name, dtype,
                                                                        nptype, 'host')
    for op, reference, label in OPERATORS:
        for name, dtype, nptype in DTYPES:
            test_the_frontend_floors_augmented_modulo_and_division_like_numpy(op, reference, label, name, dtype, nptype)
            test_a_tasklet_of_a_python_program_has_pythons_semantics(op, reference, label, name, dtype, nptype)
            test_a_python_program_condition_has_pythons_semantics(op, reference, label, name, dtype, nptype)
    test_the_frontend_does_not_emit_an_infix_operator('//', 'py_floor(')
    test_the_frontend_does_not_emit_an_infix_operator('%', 'py_mod(')
    for label, spelling, reference in SPELLINGS:
        for name, dtype, nptype in DTYPES:
            test_a_python_tasklet_names_the_rounding_of_its_modulo_and_division(label, spelling, reference, name, dtype,
                                                                                nptype)
            test_an_interstate_edge_names_the_rounding_of_its_modulo_and_division(label, spelling, reference, name,
                                                                                  dtype, nptype)
        for name, dtype, nptype in INTEGER_DTYPES:
            test_a_memlet_and_a_map_range_name_the_rounding_of_their_modulo_and_division(
                label, spelling, reference, name, dtype, nptype)
    for name, dtype, nptype in INTEGER_DTYPES:
        test_a_cpp_tasklet_keeps_cs_percent(name, dtype, nptype)
        test_a_python_program_subscript_and_range_have_pythons_semantics(name, dtype, nptype)
    for text, value in CONSTANTS:
        test_a_constant_folds_with_the_rounding_its_spelling_names(text, value)
    for function, dividend, divisor, value in (('CMod', -7.5, 2, -1.5), ('PyMod', -7.5, 2, 0.5)):
        test_a_floating_point_constant_folds_with_the_rounding_its_spelling_names(function, dividend, divisor, value)
    for text in READ_BACK:
        test_a_symbolic_expression_reads_back_from_its_string_with_its_own_rounding(text)
    test_the_sympy_mod_is_floored_and_prints_as_such()
    test_a_provably_nonnegative_pair_prints_as_the_c_operator()
    test_the_c_modulo_of_a_nonnegative_dividend_and_a_positive_divisor_is_the_floored_one()
    test_the_modulo_functions_have_one_implementation_per_rounding()
    test_a_map_parameter_and_the_extent_of_its_range_keep_cs_percent()
    test_the_pass_declares_the_parameter_and_the_extent_of_a_map_nonnegative()
    test_the_pass_leaves_a_map_starting_at_a_symbol_alone()
    test_a_free_symbol_is_not_assumed_nonnegative()
