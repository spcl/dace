# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``np.outer`` / ``np.inner`` / ``np.kron`` / ``np.cross`` must lower to SDFG nodes, not callbacks.

Every case asserts the STRUCTURE first: a ``numpy_<name>`` callback returns the right numbers, so a
numeric check alone passes through one unchanged and proves nothing about the lowering.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd


def assert_callback_free(prog: dace.frontend.python.parser.DaceProgram) -> dace.SDFG:
    """Fails if the parsed SDFG carries a Python-callback container or tasklet."""
    sdfg = prog.to_sdfg(simplify=False)
    for nested in sdfg.all_sdfgs_recursive():
        pystate = [name for name in nested.arrays if '__pystate' in name]
        assert not pystate, f'{nested.label} holds callback state {pystate}'
        for state in nested.states():
            for node in state.nodes():
                if not isinstance(node, nd.Tasklet):
                    continue
                assert 'numpy_' not in node.code.as_string, f'callback tasklet {node.label} in {nested.label}'
                assert not node.label.startswith('callback'), f'callback tasklet {node.label} in {nested.label}'
    return sdfg


def test_outer_vectors():

    @dace.program
    def prog(a: dace.float64[4], b: dace.float64[3]):
        return np.outer(a, b)

    assert_callback_free(prog)
    a, b = np.arange(1.0, 5.0), np.arange(5.0, 8.0)
    result = prog(a, b)
    assert result.shape == (4, 3)
    assert np.array_equal(result, np.outer(a, b))


def test_outer_flattens_operands():
    """``np.outer`` flattens first, so a 2-D operand contributes its SIZE, not its rank."""

    @dace.program
    def prog(a: dace.float64[2, 3], b: dace.float64[4]):
        return np.outer(a, b)

    assert_callback_free(prog)
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(10.0, 14.0)
    result = prog(a, b)
    assert result.shape == (6, 4)
    assert np.array_equal(result, np.outer(a, b))


def test_inner_matrices():
    """Contraction over the LAST mode of BOTH operands -- a non-square pair catches a transpose."""

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[5, 3]):
        return np.inner(a, b)

    assert_callback_free(prog)
    a = np.arange(12.0).reshape(4, 3)
    b = np.arange(100.0, 115.0).reshape(5, 3)
    result = prog(a, b)
    assert result.shape == (4, 5)
    assert np.array_equal(result, np.inner(a, b))


def test_inner_is_not_dot():
    """``np.inner`` contracts ``b``'s LAST mode where ``np.dot`` contracts its second-to-last."""

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[5, 3]):
        return np.inner(a, b)

    a = np.arange(12.0).reshape(4, 3)
    b = np.arange(100.0, 115.0).reshape(5, 3)
    assert np.array_equal(prog(a, b), a @ b.T)


def test_inner_matrix_vector():

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[3]):
        return np.inner(a, b)

    assert_callback_free(prog)
    a = np.arange(12.0).reshape(4, 3)
    b = np.arange(2.0, 5.0)
    result = prog(a, b)
    assert result.shape == (4, )
    assert np.array_equal(result, np.inner(a, b))


def test_inner_rank3():
    """A rank-3 operand keeps both of its leading modes ahead of ``b``'s."""

    @dace.program
    def prog(a: dace.float64[2, 4, 3], b: dace.float64[5, 3]):
        return np.inner(a, b)

    assert_callback_free(prog)
    a = np.arange(24.0).reshape(2, 4, 3)
    b = np.arange(50.0, 65.0).reshape(5, 3)
    result = prog(a, b)
    assert result.shape == (2, 4, 5)
    assert np.array_equal(result, np.inner(a, b))


def test_inner_vectors():

    @dace.program
    def prog(a: dace.float64[3], b: dace.float64[3], out: dace.float64[1]):
        out[0] = np.inner(a, b)

    assert_callback_free(prog)
    a, b = np.arange(1.0, 4.0), np.arange(4.0, 7.0)
    out = np.zeros(1)
    prog(a, b, out)
    assert np.array_equal(out[0], np.inner(a, b))


def test_kron_matrices():

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[5, 2]):
        return np.kron(a, b)

    assert_callback_free(prog)
    a = np.arange(12.0).reshape(4, 3)
    b = np.arange(20.0, 30.0).reshape(5, 2)
    result = prog(a, b)
    assert result.shape == (20, 6)
    assert np.array_equal(result, np.kron(a, b))


def test_kron_vectors():

    @dace.program
    def prog(a: dace.float64[4], b: dace.float64[3]):
        return np.kron(a, b)

    assert_callback_free(prog)
    a, b = np.arange(1.0, 5.0), np.arange(7.0, 10.0)
    result = prog(a, b)
    assert result.shape == (12, )
    assert np.array_equal(result, np.kron(a, b))


def test_kron_unequal_rank():
    """NumPy right-aligns unequal ranks by PREPENDING length-1 modes; the result stays rank-2."""

    @dace.program
    def prog(a: dace.float64[3], b: dace.float64[2, 4]):
        return np.kron(a, b)

    assert_callback_free(prog)
    a = np.arange(1.0, 4.0)
    b = np.arange(10.0, 18.0).reshape(2, 4)
    result = prog(a, b)
    assert result.shape == (2, 12)
    assert np.array_equal(result, np.kron(a, b))


def test_cross_vectors():

    @dace.program
    def prog(a: dace.float64[3], b: dace.float64[3]):
        return np.cross(a, b)

    assert_callback_free(prog)
    a, b = np.array([1.0, 2.0, 3.0]), np.array([-4.0, 5.0, 6.0])
    result = prog(a, b)
    assert result.shape == (3, )
    assert np.array_equal(result, np.cross(a, b))


def test_cross_stack_keeps_leading_modes():

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[4, 3]):
        return np.cross(a, b)

    assert_callback_free(prog)
    a = np.arange(1.0, 13.0).reshape(4, 3)
    b = np.arange(30.0, 42.0).reshape(4, 3)[::-1].copy()
    result = prog(a, b)
    assert result.shape == (4, 3)
    assert np.array_equal(result, np.cross(a, b))


def test_cross_broadcasts_one_vector():

    @dace.program
    def prog(a: dace.float64[4, 3], b: dace.float64[3]):
        return np.cross(a, b)

    assert_callback_free(prog)
    a = np.arange(1.0, 13.0).reshape(4, 3)
    b = np.array([2.0, -3.0, 5.0])
    result = prog(a, b)
    assert result.shape == (4, 3)
    assert np.array_equal(result, np.cross(a, b))


def test_cross_2d_operand_implies_zero_z():
    """A 2-vector operand has an implied zero third component; the result still has three."""

    @dace.program
    def prog(a: dace.float64[4, 2], b: dace.float64[4, 3]):
        return np.cross(a, b)

    assert_callback_free(prog)
    a = np.arange(1.0, 9.0).reshape(4, 2)
    b = np.arange(20.0, 32.0).reshape(4, 3)
    reference = np.cross(np.concatenate([a, np.zeros((4, 1))], axis=1), b)
    result = prog(a, b)
    assert result.shape == (4, 3)
    assert np.array_equal(result, reference)


def test_cross_2d_pair_drops_the_last_mode():
    """Two 2-vectors leave only ``z``: that mode is INDEXED away, so it is absent, not length-1."""

    @dace.program
    def prog(a: dace.float64[4, 2], b: dace.float64[4, 2]):
        return np.cross(a, b)

    assert_callback_free(prog)
    a = np.arange(1.0, 9.0).reshape(4, 2)
    b = np.arange(20.0, 28.0).reshape(4, 2)
    result = prog(a, b)
    assert result.shape == (4, )
    assert np.array_equal(result, a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def test_cross_refuses_a_four_component_vector():

    @dace.program
    def prog(a: dace.float64[4], b: dace.float64[4]):
        return np.cross(a, b)

    with pytest.raises(Exception, match='last mode of 2 or 3'):
        prog.to_sdfg(simplify=False)


def test_kron_refuses_a_scalar_operand():

    @dace.program
    def prog(a: dace.float64[4], b: dace.float64):
        return np.kron(a, b)

    with pytest.raises(Exception, match='0-D operand'):
        prog.to_sdfg(simplify=False)


if __name__ == '__main__':
    test_outer_vectors()
    test_outer_flattens_operands()
    test_inner_matrices()
    test_inner_is_not_dot()
    test_inner_matrix_vector()
    test_inner_rank3()
    test_inner_vectors()
    test_kron_matrices()
    test_kron_vectors()
    test_kron_unequal_rank()
    test_cross_vectors()
    test_cross_stack_keeps_leading_modes()
    test_cross_broadcasts_one_vector()
    test_cross_2d_operand_implies_zero_z()
    test_cross_2d_pair_drops_the_last_mode()
    test_cross_refuses_a_four_component_vector()
    test_kron_refuses_a_scalar_operand()
