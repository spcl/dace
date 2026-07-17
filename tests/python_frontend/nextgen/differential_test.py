# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Differential tests: both the staged schedule-tree frontend and the next-gen
frontend must build a verified tree for every corpus program (build parity),
the next-gen tree's structural signature must stay within the recorded
callback budget while exposing the same semantic surface (arguments/returns),
and — for trees tree-to-SDFG conversion supports — the compiled SDFG must
produce the same results as plain NumPy execution.

Programs whose trees contain conversion gaps (callbacks, SDFG calls, early
returns, ...) xfail the execution level with the gap named. Callback budgets
record current coverage — lower them as coverage grows.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python.nextgen.verify import verify_tree

from tests.python_frontend.nextgen import interop

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def prog_copy(A: dace.float64[N], B: dace.float64[N]):
    B[:] = A


@dace.program
def prog_elementwise(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    C[:] = A + B


@dace.program
def prog_broadcast(A: dace.float64[N, M], b: dace.float64[M], C: dace.float64[N, M]):
    C[:] = A * b


@dace.program
def prog_loop_accumulate(A: dace.float64[N]):
    result = 0.0
    for i in range(N):
        result = result + A[i]
    return result


@dace.program
def prog_branch(A: dace.float64[N], flag: dace.int32):
    if flag > 0:
        A[0] = 1.0
    else:
        A[0] = 2.0


@dace.program
def prog_while(A: dace.float64[N]):
    i = 0
    while i < 10:
        A[0] = A[0] + 1.0
        i = i + 1


@dace.program
def _callee_scale(X: dace.float64[N], Y: dace.float64[N]):
    Y[:] = X * 2.0


@dace.program
def prog_nested_call(A: dace.float64[N], B: dace.float64[N]):
    _callee_scale(A, B)


@dace.program
def prog_explicit_tasklet(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            b = a + 1.0
            b >> B[i]


@dace.program
def prog_numpy_creation(A: dace.float64[10]):
    z = np.zeros(10)
    A[:] = z


@dace.program
def prog_callback(A: dace.float64[N]):
    print(A)
    A[0] = 1.0


@dace.program
def prog_elif(A: dace.float64[N], flag: dace.int32):
    if flag > 0:
        A[0] = 1.0
    elif flag < 0:
        A[0] = 2.0
    else:
        A[0] = 3.0


@dace.program
def prog_break(A: dace.float64[N]):
    i = 0
    while i < 100:
        A[0] = A[0] + 1.0
        i = i + 1
        if i >= 5:
            break


@dace.program
def prog_early_return(A: dace.float64[N]):
    if A[0] > 0.5:
        return 1.0
    return 2.0


@dace.program
def prog_view_read(A: dace.float64[N], B: dace.float64[4]):
    b = A[1:5]
    B[:] = b + 1.0


@dace.program
def prog_view_write(A: dace.float64[N]):
    b = A[1:5]
    b[:] = 7.0


def _read_box(box):
    return box['value']


@dace.program
def prog_detected_callable(A: dace.float64[N]):
    # A detected Python callable executed through a callback; the callable is
    # resolved from the tree's constants at execution time.
    box = {'value': 41.5}
    y: dace.float64 = _read_box(box)
    A[0] = y


@dace.program
def prog_pyobject_multi(A: dace.float64[N]):
    # A batched callback with multiple outputs, one of them an opaque Python
    # object (box) and one typed by annotation (y): exercises the pyobject
    # out-parameter slot of the callback ABI.
    box = {'value': 41.5}
    y: dace.float64 = box['value']
    A[0] = y


#: (program, nextgen callback budget). The budget is a hard ceiling recording
#: current coverage; a failing budget means a lowering regression, a
#: too-generous budget hides progress.
CORPUS = [
    pytest.param(prog_copy, 0, id='copy'),
    pytest.param(prog_elementwise, 0, id='elementwise'),
    pytest.param(prog_broadcast, 0, id='broadcast'),
    pytest.param(prog_loop_accumulate, 0, id='loop_accumulate'),
    pytest.param(prog_branch, 0, id='branch'),
    pytest.param(prog_while, 0, id='while'),
    pytest.param(prog_nested_call, 0, id='nested_call'),
    pytest.param(prog_explicit_tasklet, 0, id='explicit_tasklet'),
    pytest.param(prog_numpy_creation, 0, id='numpy_creation'),
    pytest.param(prog_callback, 1, id='callback'),
    pytest.param(prog_elif, 0, id='elif'),
    pytest.param(prog_break, 0, id='break'),
    pytest.param(prog_early_return, 0, id='early_return'),
    pytest.param(prog_pyobject_multi, 1, id='pyobject_multi'),
    pytest.param(prog_detected_callable, 1, id='detected_callable'),
    pytest.param(prog_view_read, 0, id='view_read'),
    pytest.param(prog_view_write, 0, id='view_write'),
]


@pytest.mark.parametrize('program, callback_budget', CORPUS)
def test_build_parity(program, callback_budget):
    """Both frontends produce a tree; the nextgen tree passes verification."""
    old_root, new_root = interop.build_both(program)
    assert old_root is not None
    assert new_root is not None
    verify_tree(new_root)


@pytest.mark.parametrize('program, callback_budget', CORPUS)
def test_signature_comparison(program, callback_budget):
    old_root, new_root = interop.build_both(program)
    old_signature = interop.signature(old_root)
    new_signature = interop.signature(new_root)

    # Semantic surface: same arguments with the same descriptors
    assert new_signature.arg_names == old_signature.arg_names
    assert new_signature.argument_types == old_signature.argument_types
    # Same return surface (arity; the frontends name return containers differently)
    assert new_signature.return_arity == old_signature.return_arity

    # Coverage: the nextgen tree stays within its recorded callback budget
    assert new_signature.callback_count <= callback_budget, \
        f'callback budget exceeded: {new_signature.callback_count} > {callback_budget}'

    # Sanity: if the old frontend lowered dataflow, the new one did too
    if interop.has_compute(old_signature):
        assert interop.has_compute(new_signature) or new_signature.callback_count > 0


def _copy_inputs():
    A = np.random.rand(12)
    return {'A': A, 'B': np.zeros(12)}, {'N': 12}


def _elementwise_inputs():
    return {'A': np.random.rand(12), 'B': np.random.rand(12), 'C': np.zeros(12)}, {'N': 12}


def _broadcast_inputs():
    return {'A': np.random.rand(6, 4), 'b': np.random.rand(4), 'C': np.zeros((6, 4))}, {'N': 6, 'M': 4}


def _tasklet_inputs():
    return {'A': np.random.rand(12), 'B': np.zeros(12)}, {'N': 12}


def _creation_inputs():
    return {'A': np.random.rand(10)}, {}


def _accumulate_inputs():
    return {'A': np.random.rand(12)}, {'N': 12}


def _nested_inputs():
    return {'A': np.random.rand(12), 'B': np.zeros(12)}, {'N': 12}


def _branch_inputs():
    return {'A': np.random.rand(12), 'flag': np.int32(1)}, {'N': 12}


def _while_inputs():
    return {'A': np.random.rand(12)}, {'N': 12}


#: (program, input factory, reference function computing the expected outputs
#: from copies of the input arrays; '__return' holds an expected return value).
EXECUTION_CORPUS = [
    pytest.param(prog_copy, _copy_inputs, lambda a: {'B': a['A']}, id='copy'),
    pytest.param(prog_elementwise, _elementwise_inputs, lambda a: {'C': a['A'] + a['B']}, id='elementwise'),
    pytest.param(prog_broadcast, _broadcast_inputs, lambda a: {'C': a['A'] * a['b']}, id='broadcast'),
    pytest.param(prog_explicit_tasklet, _tasklet_inputs, lambda a: {'B': a['A'] + 1.0}, id='explicit_tasklet'),
    pytest.param(prog_numpy_creation, _creation_inputs, lambda a: {'A': np.zeros(10)}, id='numpy_creation'),
    pytest.param(prog_loop_accumulate, _accumulate_inputs, lambda a: {'__return': a['A'].sum()}, id='loop_accumulate'),
    pytest.param(prog_nested_call, _nested_inputs, lambda a: {'B': a['A'] * 2.0}, id='nested_call'),
    pytest.param(prog_branch, _branch_inputs, lambda a: {'A': np.concatenate(([1.0], a['A'][1:]))}, id='branch'),
    pytest.param(prog_while,
                 _while_inputs,
                 lambda a: {'A': np.concatenate(([a['A'][0] + 10.0], a['A'][1:]))},
                 id='while'),
    pytest.param(prog_callback, _tasklet_inputs, lambda a: {'A': np.concatenate(([1.0], a['A'][1:]))}, id='callback'),
    pytest.param(prog_elif,
                 lambda: ({
                     'A': np.random.rand(12),
                     'flag': np.int32(-1)
                 }, {
                     'N': 12
                 }),
                 lambda a: {'A': np.concatenate(([2.0], a['A'][1:]))},
                 id='elif'),
    pytest.param(prog_break,
                 _while_inputs,
                 lambda a: {'A': np.concatenate(([a['A'][0] + 5.0], a['A'][1:]))},
                 id='break'),
    pytest.param(prog_early_return,
                 lambda: ({
                     'A': np.full(12, 0.9)
                 }, {
                     'N': 12
                 }),
                 lambda a: {'__return': 1.0},
                 id='early_return'),
    pytest.param(prog_pyobject_multi,
                 _while_inputs,
                 lambda a: {'A': np.concatenate(([41.5], a['A'][1:]))},
                 id='pyobject_multi'),
    pytest.param(prog_detected_callable,
                 _while_inputs,
                 lambda a: {'A': np.concatenate(([41.5], a['A'][1:]))},
                 id='detected_callable'),
    pytest.param(prog_view_read,
                 lambda: ({
                     'A': np.random.rand(12),
                     'B': np.zeros(4)
                 }, {
                     'N': 12
                 }),
                 lambda a: {'B': a['A'][1:5] + 1.0},
                 id='view_read'),
    pytest.param(prog_view_write,
                 _while_inputs,
                 lambda a: {'A': np.concatenate((a['A'][:1], np.full(4, 7.0), a['A'][5:]))},
                 id='view_write'),
]


@pytest.mark.parametrize('program, make_inputs, reference', EXECUTION_CORPUS)
def test_execution_comparison(program, make_inputs, reference):
    from dace.frontend.python import nextgen

    new_root = nextgen.parse_program(program)
    gap = interop.execution_gap(new_root)
    if gap:
        pytest.xfail(f'tree_to_sdfg gap: {gap}')

    arrays, symbols = make_inputs()
    expected = reference({
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in arrays.items()
    })
    try:
        sdfg = new_root.as_sdfg()
    except NotImplementedError as gap_error:
        pytest.xfail(f'tree_to_sdfg gap: {gap_error}')

    result = sdfg(**arrays, **symbols, **interop.callback_arguments(new_root))

    for name, expected_value in expected.items():
        if name == '__return':
            np.testing.assert_allclose(np.asarray(result).ravel(), np.ravel(expected_value))
        else:
            np.testing.assert_allclose(arrays[name], expected_value, err_msg=f'output "{name}" mismatch')


if __name__ == '__main__':
    for parameter in CORPUS:
        program, callback_budget = parameter.values
        test_build_parity(program, callback_budget)
        test_signature_comparison(program, callback_budget)
    for parameter in EXECUTION_CORPUS:
        program, make_inputs, reference = parameter.values
        test_execution_comparison(program, make_inputs, reference)
