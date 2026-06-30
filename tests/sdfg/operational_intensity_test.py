# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the operational intensity analysis. """
import contextlib
import io
from typing import Dict, Tuple
from unittest import mock

import pytest
import dace
import sympy as sp
import numpy as np
from dace.sdfg.performance_evaluation.operational_intensity import analyze_sdfg_op_in
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.sdfg.utils import inline_control_flow_regions
from dace.symbolic import pystr_to_symbolic, SymbolicType
from dace.frontend.python.parser import DaceProgram

from math import isclose

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K')

TILE_SIZE = dace.symbol('TILE_SIZE')


@dace.program
def single_map64(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    z[:] = x + y
    # does N work, loads 3*N elements of 8 bytes
    # --> op_in should be N / 3*8*N = 1/24 (no reuse) assuming L divides N


@dace.program
def single_map16(x: dace.float16[N], y: dace.float16[N], z: dace.float16[N]):
    z[:] = x + y
    # does N work, loads 3*N elements of 2 bytes
    # --> op_in should be N / 3*2*N = 1/6 (no reuse) assuming L divides N


@dace.program
def single_for_loop(x: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        x[i] += y[i]
    # N work, 2*N*8 bytes loaded
    # --> 1/16 op in


@dace.program
def if_else(x: dace.int64[100], sum: dace.int64[1]):
    if x[10] > 50:
        for i in range(100):
            sum += x[i]
    if x[0] > 3:
        for i in range(100):
            sum += x[i]
    # no else --> simply analyze the ifs. if cache big enough, everything is reused;


@dace.program
def unaligned_for_loop(x: dace.float32[100], sum: dace.int64[1]):
    for i in range(17, 53):
        sum += x[i]
    # 36 = 144byte array elemets accessed 1 = 4byte scalar accessed 36 ops -> 64byte line size=> 3 lines + 1 line scalar => op in = 9/64


@dace.program
def sequential_maps(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    z[:] = x + y
    z[:] *= 2
    z[:] += x
    # does N work, loads 3*N elements of 8 bytes
    # --> op_in should be N / 3*8*N = 1/24 (no reuse) assuming L divides N


@dace.program
def nested_reuse(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N], result: dace.float64[1]):
    # load x, y and z
    z[:] = x + y
    result[0] = np.sum(z)
    # tests whether the access to z from the nested SDFG correspond with the prior accesses
    # to z outside of the nested SDFG.


@dace.program
def mmm(x: dace.float64[N, N], y: dace.float64[N, N], z: dace.float64[N, N]):
    for n, k, m in dace.map[0:N, 0:N, 0:N]:
        z[n, k] += x[n, m] * y[m, k]


@dace.program
def tiled_mmm(x: dace.float64[N, N], y: dace.float64[N, N], z: dace.float64[N, N]):
    for n_TILE, k_TILE, m_TILE in dace.map[0:N:TILE_SIZE, 0:N:TILE_SIZE, 0:N:TILE_SIZE]:
        for n, k, m in dace.map[n_TILE:n_TILE + TILE_SIZE, k_TILE:k_TILE + TILE_SIZE, m_TILE:m_TILE + TILE_SIZE]:
            z[n, k] += x[n, m] * y[m, k]


@dace.program
def tiled_mmm_32(x: dace.float32[N, N], y: dace.float32[N, N], z: dace.float32[N, N]):
    for n_TILE, k_TILE, m_TILE in dace.map[0:N:TILE_SIZE, 0:N:TILE_SIZE, 0:N:TILE_SIZE]:
        for n, k, m in dace.map[n_TILE:n_TILE + TILE_SIZE, k_TILE:k_TILE + TILE_SIZE, m_TILE:m_TILE + TILE_SIZE]:
            z[n, k] += x[n, m] * y[m, k]


@dace.program
def reduction_library_node(x: dace.float64[N]):
    return np.sum(x)


#(sdfg, c, l, assumptions, expected_result)
test_cases: Dict[str, Tuple[DaceProgram, int, int, Dict[str, int], SymbolicType]] = {
    'single_map64_even': (single_map64, 64 * 64, 64, {
        'N': 512
    }, 1 / 24),
    'single_map16_even': (single_map16, 64 * 64, 64, {
        'N': 512
    }, 1 / 6),
    'single_for_loop': (single_for_loop, 64 * 64, 64, {
        'N': 512
    }, 1 / 16),
    'if_else': (if_else, 64 * 64, 64, {
        'N': 512
    }, 200 / (14 * 64))
    # 14 cache misses, because DaCe introduces intermediate variable
    ,
    'unaligned_for_loop': (unaligned_for_loop, 64 * 64, 64, {}, 9 / 64),
    # now num_elements_on_single_cache_line does not divie N anymore
    # -->513 work, 520 elements loaded --> 513 / (520*8*3)
    'single_map64_uneven': (single_map64, 64 * 64, 64, {
        'N': 513
    }, 513 / (3 * 8 * 520)),
    'sequential_maps': (sequential_maps, 1024, 3 * 8, {
        'N': 29
    }, 87 / (90 * 8)),
    # smaller cache --> only two arrays fit --> x loaded twice now
    'sequential_maps_small': (sequential_maps, 6, 3 * 8, {
        'N': 7
    }, 21 / (13 * 3 * 8)),
    'nested_reuse': (nested_reuse, 1024, 64, {
        'N': 1024
    }, 2048 / (3 * 1024 * 8 + 128)),
    'mmm': (mmm, 20, 16, {
        'N': 24
    }, (2 * 24**3) / ((36 * 24**2 + 24 * 12) * 16)),
    'tiled_mmm': (tiled_mmm, 20, 16, {
        'N': 24,
        'TILE_SIZE': 4
    }, (2 * 24**3) / (16 * 24 * 6**3)),
    'tiled_mmm_32': (tiled_mmm_32, 10, 16, {
        'N': 24,
        'TILE_SIZE': 4
    }, (2 * 24**3) / (16 * 12 * 6**3)),
    'reduction_library_node': (reduction_library_node, 1024, 64, {
        'N': 128
    }, 128.0 / (dace.symbol('Reduce_misses', positive=True) * 64.0 + 64.0)),
}


@pytest.mark.parametrize('test_name', list(test_cases.keys()))
def test_operational_intensity(test_name: str):
    test, c, l, assumptions, correct = test_cases[test_name]
    op_in_map: Dict[str, sp.Expr] = {}
    sdfg = test.to_sdfg()
    if test_name == 'nested_reuse':
        sdfg.expand_library_nodes()
    if test_name in ['sequential_maps', 'sequential_maps_small', 'nested_reuse', 'mmm', 'tiled_mmm', 'tiled_mmm_32']:
        sdfg.simplify()
    analyze_sdfg_op_in(sdfg, op_in_map, c * l, l, assumptions)
    res = (op_in_map[get_uuid(sdfg)])
    if test_name == 'reduction_library_node':
        # Symbolic result (depends on the opaque Reduce_misses symbol); compare expressions directly.
        assert pystr_to_symbolic(correct) == res
    else:
        assert isclose(correct, res)


_ASK_USER_LOOP_ITERS = 8


@dace.program
def ask_user_branch(x: dace.float64[64], y: dace.float64[64]):
    # Data-dependent branches doing different amounts of work, so the chosen branch changes the result.
    if x[0] > 0:
        y[:] = x + 1.0
    else:
        y[:] = x * x * x * x


@dace.program
def ask_user_branch_in_loop(x: dace.float64[64], y: dace.float64[64]):
    for _ in range(_ASK_USER_LOOP_ITERS):
        if x[0] > 0:
            y[:] = x + 1.0
        else:
            y[:] = x * x * x * x


def _op_in_with_choice(program: DaceProgram, choice: int) -> Tuple[int, float]:
    """ Run the analysis in ``ask_user`` mode, answering every branch prompt with ``choice``.

    :param program: The DaCe program to analyze.
    :param choice: The branch index to feed to every prompt.
    :returns: The number of prompts raised and the resulting operational intensity.
    """
    sdfg = program.to_sdfg()
    sdfg.simplify()
    prompts = []

    def fake_input(*_):
        prompts.append(choice)
        return str(choice)

    op_in_map: Dict[str, sp.Expr] = {}
    with mock.patch('builtins.input', fake_input), contextlib.redirect_stdout(io.StringIO()):
        analyze_sdfg_op_in(sdfg, op_in_map, 1024, 64, {}, ask_user=True)
    return len(prompts), float(op_in_map[get_uuid(sdfg)])


def test_operational_intensity_ask_user_branch_selection():
    """ ``ask_user`` picks which branch of a data-dependent conditional to analyze. Both choices
    must complete without error and, since the branches differ in work, yield different results. """
    prompts_true, op_in_true = _op_in_with_choice(ask_user_branch, 0)
    prompts_else, op_in_else = _op_in_with_choice(ask_user_branch, 1)
    assert prompts_true == 1 and prompts_else == 1
    assert not isclose(op_in_true, op_in_else)


def test_operational_intensity_ask_user_decision_reused_in_loop():
    """ A branch chosen once is reused on later visits, so a conditional inside a loop prompts only
    once and the loop intensity is the single-iteration intensity scaled by the trip count. """
    single_prompts, single_op_in = _op_in_with_choice(ask_user_branch, 0)
    loop_prompts, loop_op_in = _op_in_with_choice(ask_user_branch_in_loop, 0)
    assert single_prompts == 1 and loop_prompts == 1
    assert isclose(loop_op_in, _ASK_USER_LOOP_ITERS * single_op_in)


def test_operational_intensity_range_simulation():
    """ Smoke-test the simulation path: a symbol given a range ``'start,stop,step'`` is sampled, its
    cache misses simulated per sample, and the operational intensity fitted as a function of it.
    Streaming ``single_map64`` has no reuse, so the fit is the constant ``1 / 24``. """
    op_in_map: Dict[str, sp.Expr] = {}
    sdfg = single_map64.to_sdfg()
    # Sampling at multiples of 8 keeps the 64-byte cache lines (8 doubles) evenly divided.
    analyze_sdfg_op_in(sdfg, op_in_map, 64 * 64, 64, {'N': '64,576,64'}, test_set_size=2)

    # The fitted result is a string expression in N; parse it (``pystr_to_symbolic`` avoids the
    # collision between the symbol ``N`` and SymPy's numeric-evaluation function ``N``).
    op_in = pystr_to_symbolic(op_in_map[get_uuid(sdfg)])
    for n in (64, 256, 512):
        assert isclose(float(op_in.subs(N, n)), 1 / 24, rel_tol=1e-6)


def test_operational_intensity_bails_on_unstructured_control_flow():
    """ The analysis only models structured control flow. On an inlined SDFG (LoopRegions and
    ConditionalBlocks flattened to a legacy state machine) it must warn and produce a zero result
    rather than a wrong one. """
    sdfg = ask_user_branch.to_sdfg()
    sdfg.simplify()
    inline_control_flow_regions(sdfg)
    op_in_map: Dict[str, sp.Expr] = {}
    with pytest.warns(UserWarning, match='structured control flow'):
        analyze_sdfg_op_in(sdfg, op_in_map, 1024, 64, {})
    assert op_in_map[get_uuid(sdfg)] == 0


if __name__ == '__main__':
    for test_name in test_cases.keys():
        test_operational_intensity(test_name)
    test_operational_intensity_ask_user_branch_selection()
    test_operational_intensity_ask_user_decision_reused_in_loop()
    test_operational_intensity_range_simulation()
    test_operational_intensity_bails_on_unstructured_control_flow()
