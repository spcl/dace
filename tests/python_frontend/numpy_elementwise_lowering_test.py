# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Native lowering of the element-wise NumPy predicates and repairs.

Every call covered here used to reach a ``numpy_<name>`` callback: a tasklet that re-enters the
Python interpreter, holds the GIL, cannot be scheduled on a device and blocks fusion across it.
A callback returns the right numbers, so a numeric assertion alone says nothing about whether the
call was lowered -- each test therefore pins the SDFG structure first and the values second.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes as nd


def callback_nodes(sdfg: dace.SDFG) -> list[str]:
    """Every trace of a Python callback in an SDFG: the interpreter state container it needs, and
    the tasklets that call back into it."""
    found = []
    for nested in sdfg.all_sdfgs_recursive():
        found += [f'{nested.label}:__pystate' for name in nested.arrays if name == '__pystate']
        for state in nested.states():
            for node in state.nodes():
                if isinstance(node, nd.Tasklet) and ('numpy_' in node.code.as_string
                                                     or node.label.startswith('callback')):
                    found.append(f'{state.label}:{node.label}')
    return found


def assert_native(sdfg: dace.SDFG) -> None:
    assert callback_nodes(sdfg) == []


def tasklet_code(sdfg: dace.SDFG) -> str:
    return '\n'.join(node.code.as_string for state in sdfg.states() for node in state.nodes()
                     if isinstance(node, nd.Tasklet))


def map_entries(sdfg: dace.SDFG) -> list[nd.MapEntry]:
    return [node for state in sdfg.states() for node in state.nodes() if isinstance(node, nd.MapEntry)]


def test_isclose_is_native():

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8], out: dace.bool[8]):
        out[:] = np.isclose(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([1.0, 2.0, np.inf, -np.inf, np.nan, 3.0, np.inf, 1e-9])
    b = np.array([1.0 + 1e-12, 2.5, np.inf, np.inf, np.nan, 3.0 + 1e-9, 1.0, 0.0])
    out = np.zeros(8, dtype=np.bool_)
    prog(a, b, out)
    assert np.array_equal(out, np.isclose(a, b))


def test_isclose_equal_nan_is_native():

    @dace.program
    def prog(a: dace.float64[4], b: dace.float64[4], out: dace.bool[4]):
        out[:] = np.isclose(a, b, equal_nan=True)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([np.nan, np.nan, 1.0, np.inf])
    b = np.array([np.nan, 1.0, 1.0, np.inf])
    out = np.zeros(4, dtype=np.bool_)
    prog(a, b, out)
    assert np.array_equal(out, np.isclose(a, b, equal_nan=True))


def test_isclose_against_a_constant_is_native():

    @dace.program
    def prog(a: dace.float64[5], out: dace.bool[5]):
        out[:] = np.isclose(a, 2.0)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([2.0, 2.0 + 1e-12, 2.1, np.inf, np.nan])
    out = np.zeros(5, dtype=np.bool_)
    prog(a, out)
    assert np.array_equal(out, np.isclose(a, 2.0))


def test_isclose_complex_is_native():

    @dace.program
    def prog(a: dace.complex128[4], b: dace.complex128[4], out: dace.bool[4]):
        out[:] = np.isclose(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([1 + 1j, 2 + 0j, np.inf + 0j, 1 + 2j])
    b = np.array([1 + 1j, 2.5 + 0j, np.inf + 0j, 1 + 2.0000000001j])
    out = np.zeros(4, dtype=np.bool_)
    prog(a, b, out)
    assert np.array_equal(out, np.isclose(a, b))


def test_isclose_broadcasts_like_numpy():

    @dace.program
    def prog(a: dace.float64[3, 4], b: dace.float64[4], out: dace.bool[3, 4]):
        out[:] = np.isclose(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.array([0.0, 1.0, 6.0, 7.0])
    out = np.zeros((3, 4), dtype=np.bool_)
    prog(a, b, out)
    assert np.array_equal(out, np.isclose(a, b))


def test_allclose_is_native_and_reduces():

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8], out: dace.bool[1]):
        out[0] = np.allclose(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    # The whole-array verdict must come from the reduction path, not from a per-element write race.
    assert any(isinstance(node, Reduce) for state in sdfg.states() for node in state.nodes())

    a = np.random.default_rng(7).random(8)
    out = np.zeros(1, dtype=np.bool_)
    prog(a, a.copy(), out)
    assert bool(out[0]) is np.allclose(a, a)

    b = a.copy()
    b[5] += 0.5
    prog(a, b, out)
    assert bool(out[0]) is np.allclose(a, b)


def test_array_equal_is_native_and_reduces():

    @dace.program
    def prog(a: dace.int64[6], b: dace.int64[6], out: dace.bool[1]):
        out[0] = np.array_equal(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert any(isinstance(node, Reduce) for state in sdfg.states() for node in state.nodes())

    a = np.arange(6)
    out = np.zeros(1, dtype=np.bool_)
    prog(a, a.copy(), out)
    assert bool(out[0]) is np.array_equal(a, a)

    b = a.copy()
    b[2] = 99
    prog(a, b, out)
    assert bool(out[0]) is np.array_equal(a, b)


def test_array_equal_equal_nan_is_native():

    @dace.program
    def prog(a: dace.float64[4], b: dace.float64[4], out: dace.bool[1]):
        out[0] = np.array_equal(a, b, equal_nan=True)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([1.0, np.nan, 3.0, 4.0])
    out = np.zeros(1, dtype=np.bool_)
    prog(a, a.copy(), out)
    assert bool(out[0]) is np.array_equal(a, a, equal_nan=True)


def test_array_equal_mismatched_shapes_decided_while_parsing():
    """Shapes live on the descriptors, so the False is settled before any dataflow is built."""

    @dace.program
    def prog(a: dace.float64[6], b: dace.float64[5], out: dace.bool[1]):
        out[0] = np.array_equal(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert map_entries(sdfg) == []

    out = np.ones(1, dtype=np.bool_)
    prog(np.zeros(6), np.zeros(5), out)
    assert bool(out[0]) is np.array_equal(np.zeros(6), np.zeros(5))


def test_array_equal_refuses_undecidable_symbolic_shapes():
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[M], out: dace.bool[1]):
        out[0] = np.array_equal(a, b)

    with pytest.raises(ValueError, match='cannot decide whether the symbolic shapes'):
        prog.to_sdfg(simplify=False)


def test_nan_to_num_is_native():

    @dace.program
    def prog(a: dace.float64[6], out: dace.float64[6]):
        out[:] = np.nan_to_num(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([1.0, np.nan, np.inf, -np.inf, -2.0, 0.0])
    out = np.zeros(6)
    prog(a, out)
    assert np.array_equal(out, np.nan_to_num(a))


def test_nan_to_num_with_explicit_fills_is_native():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4]):
        out[:] = np.nan_to_num(a, nan=-1.0, posinf=1000.0, neginf=-1000.0)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([np.nan, np.inf, -np.inf, 2.0])
    out = np.zeros(4)
    prog(a, out)
    assert np.array_equal(out, np.nan_to_num(a, nan=-1.0, posinf=1000.0, neginf=-1000.0))


def test_nan_to_num_keeps_float32():
    """A float32 operand must not widen: the fills carry the input dtype into the tasklet."""

    @dace.program
    def prog(a: dace.float32[4], out: dace.float32[4]):
        out[:] = np.nan_to_num(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert all(desc.dtype != dace.float64 for desc in sdfg.arrays.values() if desc.transient)

    a = np.array([np.nan, np.inf, -np.inf, 2.5], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    prog(a, out)
    assert np.array_equal(out, np.nan_to_num(a))


def test_nan_to_num_on_integers_is_a_copy():
    """Integers carry neither NaN nor an infinity, so NumPy hands back an untouched copy."""

    @dace.program
    def prog(a: dace.int64[4], out: dace.int64[4]):
        out[:] = np.nan_to_num(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert map_entries(sdfg) == []

    a = np.array([1, -2, 3, 4], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    prog(a, out)
    assert np.array_equal(out, np.nan_to_num(a))


def test_nan_to_num_refuses_complex():

    @dace.program
    def prog(a: dace.complex128[4], out: dace.complex128[4]):
        out[:] = np.nan_to_num(a)

    with pytest.raises(ValueError, match='no native path back into a complex value'):
        prog.to_sdfg(simplify=False)


def test_i0_is_native():

    @dace.program
    def prog(a: dace.float64[9], out: dace.float64[9]):
        out[:] = np.i0(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    # Both Chebyshev branches must be in the emitted body, not just the small-argument one.
    body = tasklet_code(sdfg)
    assert '__i0_s_res' in body and '__i0_l_res' in body

    a = np.array([0.0, 0.5, 1.0, -1.0, 4.0, 8.0, 8.5, 20.0, -30.0])
    out = np.zeros(9)
    prog(a, out)
    np.testing.assert_allclose(out, np.i0(a), rtol=1e-14, atol=0.0)


def test_i0_on_integers_widens_like_numpy():

    @dace.program
    def prog(a: dace.int64[4], out: dace.float64[4]):
        out[:] = np.i0(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([0, 1, 5, 12], dtype=np.int64)
    out = np.zeros(4)
    prog(a, out)
    np.testing.assert_allclose(out, np.i0(a), rtol=1e-14, atol=0.0)


def test_i0_refuses_complex():

    @dace.program
    def prog(a: dace.complex128[4], out: dace.complex128[4]):
        out[:] = np.i0(a)

    with pytest.raises(ValueError, match='not defined for complex input'):
        prog.to_sdfg(simplify=False)


def test_sinc_is_native():

    @dace.program
    def prog(a: dace.float64[7], out: dace.float64[7]):
        out[:] = np.sinc(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([0.0, 0.5, 1.0, -1.5, 2.0, 0.25, -0.75])
    out = np.zeros(7)
    prog(a, out)
    assert np.array_equal(out, np.sinc(a))


def test_sinc_keeps_float32():

    @dace.program
    def prog(a: dace.float32[4], out: dace.float32[4]):
        out[:] = np.sinc(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert all(desc.dtype != dace.float64 for desc in sdfg.arrays.values() if desc.transient)

    a = np.array([0.0, 0.5, 1.0, -1.5], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    prog(a, out)
    assert np.array_equal(out, np.sinc(a))


def test_angle_is_native():

    @dace.program
    def prog(a: dace.complex128[5], out: dace.float64[5]):
        out[:] = np.angle(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert 'atan2' in tasklet_code(sdfg)

    a = np.array([1 + 0j, 1 + 1j, -1 + 0j, -1 - 1j, 0 + 2j])
    out = np.zeros(5)
    prog(a, out)
    assert np.array_equal(out, np.angle(a))


def test_angle_in_degrees_is_native():

    @dace.program
    def prog(a: dace.complex128[4], out: dace.float64[4]):
        out[:] = np.angle(a, deg=True)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([1 + 0j, 1 + 1j, -1 + 0j, 0 + 2j])
    out = np.zeros(4)
    prog(a, out)
    assert np.array_equal(out, np.angle(a, deg=True))


def test_angle_of_real_input_is_native():
    """A real operand still has an argument: 0 where it is non-negative, pi where it is negative."""

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4]):
        out[:] = np.angle(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)

    a = np.array([1.0, -1.0, 0.0, -3.5])
    out = np.zeros(4)
    prog(a, out)
    assert np.array_equal(out, np.angle(a))


def test_iscomplexobj_resolves_while_parsing():
    """A predicate on the descriptor, not on the values: no map, and the operands are never read."""

    @dace.program
    def prog(a: dace.complex128[4], b: dace.float64[4], out: dace.bool[2]):
        out[0] = np.iscomplexobj(a)
        out[1] = np.iscomplexobj(b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert map_entries(sdfg) == []
    read = {node.data for state in sdfg.states() for node in state.nodes() if isinstance(node, nd.AccessNode)}
    assert 'a' not in read and 'b' not in read

    out = np.zeros(2, dtype=np.bool_)
    prog(np.zeros(4, dtype=np.complex128), np.zeros(4), out)
    assert bool(out[0]) is np.iscomplexobj(np.zeros(4, dtype=np.complex128))
    assert bool(out[1]) is np.iscomplexobj(np.zeros(4))


def test_real_if_close_on_real_input_is_a_passthrough():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4]):
        out[:] = np.real_if_close(a)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    assert map_entries(sdfg) == []

    a = np.random.default_rng(11).random(4)
    out = np.zeros(4)
    prog(a, out)
    assert np.array_equal(out, np.real_if_close(a))


def test_real_if_close_refuses_complex():
    """The result is real or complex depending on the values, which no descriptor can express."""

    @dace.program
    def prog(a: dace.complex128[4], out: dace.complex128[4]):
        out[:] = np.real_if_close(a)

    with pytest.raises(ValueError, match='result dtype is data-dependent'):
        prog.to_sdfg(simplify=False)


def test_generated_code_never_enters_the_interpreter():
    """The end of the pipeline, not just the SDFG: no GIL, no callback table, no interpreter state."""

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8], out: dace.float64[8]):
        out[:] = np.sinc(np.nan_to_num(a)) + np.i0(b)

    sdfg = prog.to_sdfg(simplify=False)
    assert_native(sdfg)
    code = '\n'.join(obj.code for obj in sdfg.generate_code())
    # The call form, not the bare name: the native ufunc tasklets are labelled ``_numpy_<name>_``.
    for marker in ('__pystate', 'PyGILState', 'PyObject', 'numpy_sinc(', 'numpy_i0(', 'numpy_nan_to_num('):
        assert marker not in code


if __name__ == '__main__':
    test_isclose_is_native()
    test_isclose_equal_nan_is_native()
    test_isclose_against_a_constant_is_native()
    test_isclose_complex_is_native()
    test_isclose_broadcasts_like_numpy()
    test_allclose_is_native_and_reduces()
    test_array_equal_is_native_and_reduces()
    test_array_equal_equal_nan_is_native()
    test_array_equal_mismatched_shapes_decided_while_parsing()
    test_array_equal_refuses_undecidable_symbolic_shapes()
    test_nan_to_num_is_native()
    test_nan_to_num_with_explicit_fills_is_native()
    test_nan_to_num_keeps_float32()
    test_nan_to_num_on_integers_is_a_copy()
    test_nan_to_num_refuses_complex()
    test_i0_is_native()
    test_i0_on_integers_widens_like_numpy()
    test_i0_refuses_complex()
    test_sinc_is_native()
    test_sinc_keeps_float32()
    test_angle_is_native()
    test_angle_in_degrees_is_native()
    test_angle_of_real_input_is_native()
    test_iscomplexobj_resolves_while_parsing()
    test_real_if_close_on_real_input_is_a_passthrough()
    test_real_if_close_refuses_complex()
    test_generated_code_never_enters_the_interpreter()
