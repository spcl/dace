# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The counter scan and the first/last-match lifts as CPF renders them: the class line of each lifted loop says
``parallel`` with its reason (a scan, a min or max reduction), no loop of the kernel stays ``sequential`` or ``unknown``,
and the C and CUDA units compute what NumPy computes."""

import ctypes
import pathlib
import tempfile
from collections.abc import Callable

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import (
    build_standalone,
    call_standalone,
    entry_argtypes,
    render_gpu,
    require_gpu_backend,
)
from tests.codegen.cpf.test_emission_cuda import nvcc_library

N = dace.symbol("N")

Arrays = dict[str, np.ndarray]


@dace.program
def offsets(k: dace.int64[N], m: dace.float64[N], start: dace.int64[N], cnt: dace.int64[1]):
    n = 0
    for i in range(N):
        start[i] = n
        if m[i] > 0.0:
            n = n + k[i] * (k[i] + 1) // 2
    cnt[0] = n


@dace.program
def last_match(d: dace.float64[N], pos: dace.int64[1], val: dace.float64[1]):
    p = -1
    v = 0.0
    for i in range(N):
        if d[i] < 0.0:
            p = i
            v = d[i]
    pos[0] = p
    val[0] = v


@dace.program
def first_match(d: dace.float64[N], pos: dace.int64[1]):
    p = -1
    for i in range(N):
        if d[i] < 0.0 and p < 0:
            p = i
    pos[0] = p


def offsets_inputs(rng: np.random.Generator, size: int) -> Arrays:
    return {
        "k": rng.integers(0, 9, size),
        "m": rng.uniform(-1.0, 1.0, size),
        "start": np.zeros(size, np.int64),
        "cnt": np.zeros(1, np.int64),
    }


def match_inputs(rng: np.random.Generator, size: int) -> Arrays:
    d = rng.uniform(0.0, 1.0, size)
    d[rng.choice(size, min(size, 5), replace=False)] *= -1.0
    return {"d": d, "pos": np.zeros(1, np.int64), "val": np.zeros(1)}


def offsets_in_numpy(host: Arrays) -> Arrays:
    increments = np.where(host["m"] > 0.0, host["k"] * (host["k"] + 1) // 2, 0)
    running = np.cumsum(increments)
    return {"start": running - increments, "cnt": running[-1:]}


def last_match_in_numpy(host: Arrays) -> Arrays:
    hit = np.flatnonzero(host["d"] < 0.0)[-1]
    return {"pos": np.array([hit]), "val": host["d"][hit : hit + 1]}


def first_match_in_numpy(host: Arrays) -> Arrays:
    return {"pos": np.flatnonzero(host["d"] < 0.0)[:1]}


#: program, its inputs, its NumPy reference, the class line its lifted loop must carry
CASES: dict[str, tuple[dace.program, Callable, Callable, str]] = {
    "offsets": (offsets, offsets_inputs, offsets_in_numpy, "// parallel -- scan: "),
    "last_match": (last_match, match_inputs, last_match_in_numpy, "// parallel -- reduction: max over the indices"),
    "first_match": (first_match, match_inputs, first_match_in_numpy, "// parallel -- reduction: min over the indices"),
}


def inputs_of(case: str, size: int) -> Arrays:
    program, inputs = CASES[case][:2]
    host = inputs(np.random.default_rng(size), size)
    names = set(program.to_sdfg(simplify=False).arglist())
    return {name: value for name, value in host.items() if name in names}


def assert_class_lines(code: str, case: str) -> None:
    lines = [line.strip() for line in code.splitlines()]
    assert CASES[case][3] in "\n".join(lines), code
    for prefix in ("// sequential -- ", "// unknown -- "):
        assert not [line for line in lines if line.startswith(prefix)], code


@pytest.mark.parametrize("case", list(CASES))
def test_the_c_unit_says_why_the_lifted_loop_is_parallel_and_matches_numpy(case: str):
    program, reference = CASES[case][0], CASES[case][2]
    name = f"cpf_lift_c_{case}"
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    finalize_for_target(sdfg, "cpu", validate=True)
    rendering = render(sdfg, language="c", llm_friendly=True)
    assert_class_lines(rendering.code, case)
    library = build_standalone(rendering.code, name, language="c")
    for size in (1, 777):
        host = inputs_of(case, size)
        expected = reference(host)
        call_standalone(library, rendering.sdfg, {**host, "N": size})
        for output, want in expected.items():
            assert np.array_equal(host[output], want), (case, size, output, host[output], want)


@pytest.mark.gpu
@pytest.mark.parametrize("case", list(CASES))
def test_the_cuda_units_say_why_the_lifted_loop_is_parallel_and_match_numpy(case: str):
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    program, reference = CASES[case][0], CASES[case][2]
    name = f"cpf_lift_cuda_{case}"
    rendering = render_gpu(program, name, language="cuda")
    assert_class_lines(rendering.code + rendering.device_code, case)
    with tempfile.TemporaryDirectory() as directory:
        entry = ctypes.CDLL(str(nvcc_library(rendering, pathlib.Path(directory), name)))[name]
    entry.argtypes = entry_argtypes(rendering.sdfg)
    size = 4099
    host = inputs_of(case, size)
    expected = reference(host)
    device = {key: cupy.asarray(value) for key, value in host.items()}
    entry(*[ctypes.c_void_p(device[a].data.ptr) if a in device else size for a in rendering.sdfg.arglist()])
    for output, want in expected.items():
        assert np.array_equal(cupy.asnumpy(device[output]), want), (case, output)


if __name__ == "__main__":
    for case in CASES:
        test_the_c_unit_says_why_the_lifted_loop_is_parallel_and_matches_numpy(case)
        test_the_cuda_units_say_why_the_lifted_loop_is_parallel_and_match_numpy(case)
