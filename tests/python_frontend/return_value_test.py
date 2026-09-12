# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.sdfg import nodes
from dace.sdfg.state import ReturnBlock
from dace.transformation.interstate import InlineMultistateSDFG


def test_return_scalar():

    @dace.program
    def return_scalar():
        return 5

    res = return_scalar()
    assert res == 5

    # The return value above is actually an array. If you would
    # add the return value annotation to the program, i.e. `-> dace.int32`, you would
    # get a validation error.
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, )
    assert res.dtype == np.int64


def test_return_scalar_in_nested_function():

    @dace.program
    def nested_function() -> dace.int32:
        return 5

    @dace.program
    def return_scalar():
        return nested_function()

    res = return_scalar()
    assert res == 5

    # The return value above is actually an array. If you would
    # add the return value annotation to the program, i.e. `-> dace.int32`, you would
    # get a validation error.
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, )
    assert res.dtype == np.int32


def test_return_array():

    @dace.program
    def return_array():
        return 5 * np.ones(5)

    res = return_array()
    assert np.allclose(res, 5 * np.ones(5))


def test_return_tuple():

    @dace.program
    def return_tuple():
        return 5, 6

    res = return_tuple()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res == (5, 6)


def test_return_array_tuple():

    @dace.program
    def return_array_tuple():
        return 5 * np.ones(5), 6 * np.ones(6)

    res = return_array_tuple()
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert np.allclose(res[0], 5 * np.ones(5))
    assert np.allclose(res[1], 6 * np.ones(6))


def test_return_void():

    @dace.program
    def return_void(a: dace.float64[20]):
        a[:] += 1
        return
        a[:] = 5

    a = np.random.rand(20)
    ref = a + 1
    res = return_void(a)
    assert res is None
    assert np.allclose(a, ref)


def test_return_tuple_1_element():

    @dace.program
    def return_one_element_tuple(a: dace.float64[20]):
        return (a + 3.5, )

    a = np.random.rand(20)
    ref = a + 3.5
    res = return_one_element_tuple(a)
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert np.allclose(res[0], ref)


def test_return_void_in_if():

    @dace.program
    def return_void(a: dace.float64[20]):
        if a[0] < 0:
            return
        a[:] = 5

    a = np.random.rand(20)
    return_void(a)
    assert np.allclose(a, 5)
    a[:] = np.random.rand(20)
    a[0] = -1
    ref = a.copy()
    return_void(a)
    assert np.allclose(a, ref)


def test_return_void_in_for():

    @dace.program
    def return_void(a: dace.float64[20]):
        for _ in range(20):
            return
        a[:] = 5

    a = np.random.rand(20)
    ref = a.copy()
    return_void(a)
    assert np.allclose(a, ref)


def test_a_trailing_return_in_a_nested_program_does_not_end_the_caller():

    @dace.program
    def callee(x: dace.float64[20], o: dace.float64[20]):
        o[:] = x * 2.0
        return

    @dace.program
    def caller(x: dace.float64[20], o: dace.float64[20], marker: dace.float64[20]):
        callee(x, o)
        marker[:] = 7.0

    x = np.random.rand(20)
    o = np.zeros(20)
    marker = np.zeros(20)
    caller(x, o, marker)
    assert np.allclose(o, x * 2.0, rtol=0, atol=1e-14)
    assert np.allclose(marker, 7.0, rtol=0, atol=0)


def test_an_early_return_in_a_nested_program_does_not_end_the_caller():

    @dace.program
    def callee(a: dace.float64[20]):
        if a[0] > 0.0:
            a[1] = 1.0
            return
        a[2] = 2.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[0] = 1.0
    ref = np.zeros(20)
    ref[0] = 1.0
    ref[1] = 1.0
    ref[5] = 5.0
    caller(a)
    assert np.allclose(a, ref, rtol=0, atol=0)

    a = np.zeros(20)
    ref = np.zeros(20)
    ref[2] = 2.0
    ref[5] = 5.0
    caller(a)
    assert np.allclose(a, ref, rtol=0, atol=0)


def test_a_return_inside_a_loop_in_a_nested_program_does_not_end_the_caller():

    @dace.program
    def callee(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 0.0:
                a[0] = 7.0
                return
        a[1] = 1.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[3] = 1.0
    ref = np.zeros(20)
    ref[0] = 7.0
    ref[3] = 1.0
    ref[5] = 5.0
    caller(a)
    assert np.allclose(a, ref, rtol=0, atol=0)

    a = np.zeros(20)
    ref = np.zeros(20)
    ref[1] = 1.0
    ref[5] = 5.0
    caller(a)
    assert np.allclose(a, ref, rtol=0, atol=0)


def test_a_return_two_call_levels_down_does_not_end_the_outermost_caller():

    @dace.program
    def innermost(a: dace.float64[20]):
        if a[0] > 0.0:
            a[1] = 1.0
            return
        a[2] = 2.0

    @dace.program
    def middle(a: dace.float64[20]):
        innermost(a)
        a[3] = 3.0

    @dace.program
    def outermost(a: dace.float64[20]):
        middle(a)
        a[4] = 4.0

    a = np.zeros(20)
    a[0] = 1.0
    ref = np.zeros(20)
    ref[0] = 1.0
    ref[1] = 1.0
    ref[3] = 3.0
    ref[4] = 4.0
    outermost(a)
    assert np.allclose(a, ref, rtol=0, atol=0)


def test_a_return_beside_a_break_in_a_nested_program_does_not_end_the_caller():

    @dace.program
    def callee(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 1.5:
                a[0] = 3.0
                return
            if a[i] > 0.5:
                break
        a[1] = 1.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[2] = 1.0
    ref = np.zeros(20)
    ref[1] = 1.0
    ref[2] = 1.0
    ref[5] = 5.0
    caller(a)
    assert np.allclose(a, ref, rtol=0, atol=0)

    a = np.zeros(20)
    a[2] = 2.0
    ref = np.zeros(20)
    ref[0] = 3.0
    ref[2] = 2.0
    ref[5] = 5.0
    caller(a)
    assert np.allclose(a, ref, rtol=0, atol=0)


def test_a_trailing_return_does_not_keep_the_nested_program_nested():

    @dace.program
    def callee(x: dace.float64[20], o: dace.float64[20]):
        o[:] = x * 2.0
        return

    @dace.program
    def caller(x: dace.float64[20], o: dace.float64[20], marker: dace.float64[20]):
        callee(x, o)
        marker[:] = 7.0

    sdfg = caller.to_sdfg(simplify=True)
    nested = [n for state in sdfg.all_states() for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
    assert not nested


def count_return_blocks(sdfg):
    return sum(1 for blk in sdfg.all_control_flow_blocks() if isinstance(blk, ReturnBlock))


def build_a_trailing_return():

    @dace.program
    def callee(x: dace.float64[20], o: dace.float64[20]):
        o[:] = x * 2.0
        return

    @dace.program
    def caller(x: dace.float64[20], o: dace.float64[20], marker: dace.float64[20]):
        callee(x, o)
        marker[:] = 7.0

    x = np.random.rand(20)
    args = dict(x=x, o=np.zeros(20), marker=np.zeros(20))
    ref = dict(o=x * 2.0, marker=np.full(20, 7.0))
    return caller, args, ref


def build_a_loop_then_a_trailing_return():

    @dace.program
    def callee(x: dace.float64[20], o: dace.float64[20]):
        for i in range(20):
            o[i] = x[i] * 2.0
        return

    @dace.program
    def caller(x: dace.float64[20], o: dace.float64[20], marker: dace.float64[20]):
        callee(x, o)
        marker[:] = 7.0

    x = np.random.rand(20)
    args = dict(x=x, o=np.zeros(20), marker=np.zeros(20))
    ref = dict(o=x * 2.0, marker=np.full(20, 7.0))
    return caller, args, ref


def build_a_branch_then_a_trailing_return():

    @dace.program
    def callee(x: dace.float64[20], o: dace.float64[20]):
        if x[0] > 0.0:
            o[:] = x * 2.0
        else:
            o[:] = x * 3.0
        return

    @dace.program
    def caller(x: dace.float64[20], o: dace.float64[20], marker: dace.float64[20]):
        callee(x, o)
        marker[:] = 7.0

    x = np.random.rand(20)
    args = dict(x=x, o=np.zeros(20), marker=np.zeros(20))
    ref = dict(o=x * 2.0, marker=np.full(20, 7.0))
    return caller, args, ref


TRAILING_RETURN_CASES = [
    pytest.param(build_a_trailing_return, id='a_trailing_return'),
    pytest.param(build_a_loop_then_a_trailing_return, id='a_loop_then_a_trailing_return'),
    pytest.param(build_a_branch_then_a_trailing_return, id='a_branch_then_a_trailing_return'),
]


@pytest.mark.parametrize('build', TRAILING_RETURN_CASES)
def test_a_trailing_return_leaves_the_callers_store_intact_after_inlining(build):
    """InlineMultistateSDFG used to splice the callee's own end-of-program return into the caller,
    where a lowered return means halt the caller: the store after the call site went dead with no
    diagnostic. The transformation must run (this shape is meant to inline) and the return must
    leave no trace in the caller's control-flow graph."""
    caller, args, ref = build()
    sdfg = caller.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(InlineMultistateSDFG)
    assert applied == 1, applied
    assert count_return_blocks(sdfg) == 0, sdfg.to_json()
    csdfg = sdfg.compile()
    csdfg(**args)
    for name, want in ref.items():
        assert np.allclose(args[name], want, rtol=0, atol=0), (name, args[name], want)


def build_an_early_return_in_a_branch():

    @dace.program
    def callee(a: dace.float64[20]):
        if a[0] > 0.0:
            a[1] = 1.0
            return
        a[2] = 2.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[0] = 1.0
    ref = np.zeros(20)
    ref[0] = 1.0
    ref[1] = 1.0
    ref[5] = 5.0
    return caller, dict(a=a), dict(a=ref)


def build_a_return_in_a_loop():

    @dace.program
    def callee(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 0.0:
                a[0] = 7.0
                return
        a[1] = 1.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[3] = 1.0
    ref = np.zeros(20)
    ref[0] = 7.0
    ref[3] = 1.0
    ref[5] = 5.0
    return caller, dict(a=a), dict(a=ref)


def build_a_return_beside_a_break():

    @dace.program
    def callee(a: dace.float64[20]):
        for i in range(20):
            if a[i] > 1.5:
                a[0] = 3.0
                return
            if a[i] > 0.5:
                break
        a[1] = 1.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[2] = 2.0
    ref = np.zeros(20)
    ref[0] = 3.0
    ref[2] = 2.0
    ref[5] = 5.0
    return caller, dict(a=a), dict(a=ref)


def build_two_returns_in_a_conditional():

    @dace.program
    def callee(a: dace.float64[20]):
        if a[0] > 0.0:
            a[1] = 1.0
            return
        elif a[0] < 0.0:
            a[2] = 2.0
            return
        a[3] = 3.0

    @dace.program
    def caller(a: dace.float64[20]):
        callee(a)
        a[5] = 5.0

    a = np.zeros(20)
    a[0] = 1.0
    ref = np.zeros(20)
    ref[0] = 1.0
    ref[1] = 1.0
    ref[5] = 5.0
    return caller, dict(a=a), dict(a=ref)


REFUSAL_CASES = [
    pytest.param(build_an_early_return_in_a_branch, id='an_early_return_in_a_branch'),
    pytest.param(build_a_return_in_a_loop, id='a_return_in_a_loop'),
    pytest.param(build_a_return_beside_a_break, id='a_return_beside_a_break'),
    pytest.param(build_two_returns_in_a_conditional, id='two_returns_in_a_conditional'),
]


@pytest.mark.parametrize('build', REFUSAL_CASES)
def test_a_non_trailing_return_is_refused_and_still_computes_the_correct_answer(build):
    """A refusal that leaves the wrong answer standing is worse than the miscompile it replaces:
    InlineMultistateSDFG must decline to inline a callee whose return is not a trailing sink, and
    code generation must still produce the right result through the callee's own exit."""
    caller, args, ref = build()
    sdfg = caller.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(InlineMultistateSDFG)
    assert applied == 0, applied
    csdfg = sdfg.compile()
    csdfg(**args)
    for name, want in ref.items():
        assert np.allclose(args[name], want, rtol=0, atol=0), (name, args[name], want)


if __name__ == '__main__':
    test_return_scalar()
    test_return_scalar_in_nested_function()
    test_return_array()
    test_return_tuple()
    test_return_array_tuple()
    test_return_void()
    test_return_void_in_if()
    test_return_void_in_for()
    test_a_trailing_return_in_a_nested_program_does_not_end_the_caller()
    test_an_early_return_in_a_nested_program_does_not_end_the_caller()
    test_a_return_inside_a_loop_in_a_nested_program_does_not_end_the_caller()
    test_a_return_two_call_levels_down_does_not_end_the_outermost_caller()
    test_a_return_beside_a_break_in_a_nested_program_does_not_end_the_caller()
    test_a_trailing_return_does_not_keep_the_nested_program_nested()

    for build in TRAILING_RETURN_CASES:
        test_a_trailing_return_leaves_the_callers_store_intact_after_inlining(build.values[0])
    for build in REFUSAL_CASES:
        test_a_non_trailing_return_is_refused_and_still_computes_the_correct_answer(build.values[0])
