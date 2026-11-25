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


NAT = dace.symbol("NAT")
NH = dace.symbol("NH")
P = dace.symbol("P")


@dace.program
def triangle(lx: dace.int64[N], ly: dace.int64[N], j: dace.int64):
    # cp2k grid_integrate: pair k owns the j - k + 1 entries from n on
    n = 0
    for k in range(j + 1):
        for m in range(j - k + 1):
            lx[n + m] = k
            ly[n + m] = m
        n = n + max(0, j - k + 1)


@dace.program
def projectors(
    ityp: dace.int64[NAT], nh: dace.int64[1], sk: dace.float64[NAT, P], vkb: dace.float64[NAT * NH, P], j: dace.int64
):
    # QE vexx: atom k of type j owns nhnt rows of vkb from s on; nhnt is data, so its sign is checked at run time
    s = 0
    nhnt = nh[0]
    for k in range(NAT):
        if ityp[k] == j:
            for m in range(nhnt):
                for p in range(P):
                    vkb[s + m, p] = sk[k, p] * (m + 1)
            s = s + nhnt


NSP = dace.symbol("NSP")


@dace.program
def pairs(nh: dace.int64[NSP], tvanp: dace.int64[NSP], qgm: dace.float64[NSP * NH * NH, P]):
    # QE vexx's ijh: bumped once per pair (k, l), l >= k, of each species with tvanp
    ijh = 0
    for j in range(NSP):
        if tvanp[j] != 0:
            for k in range(nh[j]):
                for m in range(k, nh[j]):
                    ijh = ijh + 1
                    for p in range(P):
                        qgm[ijh - 1, p] = j * 100.0 + k * 10.0 + m + p * 0.5


def pairs_inputs(rng: np.random.Generator, size: int) -> Arrays:
    nsp, nh = max(size // 16, 1), 4
    return {"nh": rng.integers(0, nh + 1, nsp), "tvanp": rng.integers(0, 2, nsp), "qgm": np.zeros((nsp * nh * nh, 5))}


def pairs_in_numpy(host: Arrays) -> Arrays:
    qgm, ijh = np.zeros_like(host["qgm"]), 0
    for j, (count, flag) in enumerate(zip(host["nh"], host["tvanp"])):
        for k in range(count if flag else 0):
            for m in range(k, count):
                qgm[ijh] = j * 100.0 + k * 10.0 + m + np.arange(qgm.shape[1]) * 0.5
                ijh += 1
    return {"qgm": qgm}


def triangle_inputs(rng: np.random.Generator, size: int) -> Arrays:
    j = int((np.sqrt(8 * size + 1) - 3) // 2)  # the triangle of j + 1 rows fits in size
    return {"lx": np.full(size, -1, np.int64), "ly": np.full(size, -1, np.int64), "j": j}


def triangle_in_numpy(host: Arrays) -> Arrays:
    lx, ly = host["lx"].copy(), host["ly"].copy()
    n = 0
    for k in range(host["j"] + 1):
        lx[n : n + host["j"] - k + 1] = k
        ly[n : n + host["j"] - k + 1] = np.arange(host["j"] - k + 1)
        n += host["j"] - k + 1
    return {"lx": lx, "ly": ly}


def projectors_inputs(rng: np.random.Generator, size: int) -> Arrays:
    nat, nh = max(size // 8, 1), 3
    return {
        "ityp": rng.integers(0, 4, nat),
        "nh": np.array([nh]),
        "sk": rng.random((nat, 7)),
        "vkb": np.zeros((nat * nh, 7)),
        "j": 2,
    }


def projectors_in_numpy(host: Arrays) -> Arrays:
    vkb, s, nhnt = np.zeros_like(host["vkb"]), 0, int(host["nh"][0])
    for k in np.flatnonzero(host["ityp"] == host["j"]):
        vkb[s : s + nhnt] = host["sk"][k] * np.arange(1, nhnt + 1)[:, None]
        s += nhnt
    return {"vkb": vkb}


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


#: program, its inputs, its NumPy reference, the class lines its lifted loop must carry
CASES: dict[str, tuple[dace.program, Callable, Callable, tuple[str, ...]]] = {
    "offsets": (offsets, offsets_inputs, offsets_in_numpy, ("// parallel -- scan: ",)),
    "last_match": (last_match, match_inputs, last_match_in_numpy, ("// parallel -- reduction: max over the indices",)),
    "first_match": (
        first_match,
        match_inputs,
        first_match_in_numpy,
        ("// parallel -- reduction: min over the indices",),
    ),
    "triangle": (triangle, triangle_inputs, triangle_in_numpy, ("// parallel -- scan: ",)),
    # The sign of nhnt is data: each arm of the check says what it runs under
    "projectors": (
        projectors,
        projectors_inputs,
        projectors_in_numpy,
        ("// parallel -- scan: ", "// parallel -- guarded by if (", "// sequential -- the else arm of if ("),
    ),
    "pairs": (pairs, pairs_inputs, pairs_in_numpy, ("// parallel -- scan: ", "// sequential -- scan: ")),
}

#: The class lines of the scan a case runs inside each iteration of its lifted loop.
NESTED_SCAN = {"pairs": ("// sequential -- scan: ", "// sequential -- proven dependence on acc_")}

#: The symbols each case's entry takes besides ``N``, from its inputs.
SIZES = {
    "projectors": lambda host: {"NAT": host["ityp"].size, "NH": int(host["nh"][0]), "P": host["sk"].shape[1]},
    "pairs": lambda host: {"NSP": host["nh"].size, "NH": 4, "P": host["qgm"].shape[1]},
}


def symbols_of(case: str, host: Arrays, size: int, arguments: dict) -> dict:
    """The symbol arguments of ``case``'s entry, ``N`` the size, the rest from its inputs."""
    symbols = {"N": size, **SIZES.get(case, lambda host: {})(host)}
    return {name: value for name, value in symbols.items() if name in arguments}


def inputs_of(case: str, size: int) -> Arrays:
    program, inputs = CASES[case][:2]
    host = inputs(np.random.default_rng(size), size)
    names = set(program.to_sdfg(simplify=False).arglist())
    return {name: value for name, value in host.items() if name in names}


def assert_class_lines(code: str, case: str) -> None:
    lines = [line.strip() for line in code.splitlines()]
    for required in CASES[case][3]:
        assert required in "\n".join(lines), code
    # The in-order loops allowed: the fallback arm of a run-time check, and the scan nested in a parallel loop (one per
    # species of ``pairs``), which runs on that loop's thread; the CUDA unit spells it as a loop carrying its sum
    allowed = ("// sequential -- the else arm of if (", *NESTED_SCAN.get(case, ()))
    for prefix in ("// sequential -- ", "// unknown -- "):
        assert not [line for line in lines if line.startswith(prefix) and not line.startswith(allowed)], code


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
        call_standalone(library, rendering.sdfg, {**host, **symbols_of(case, host, size, rendering.sdfg.arglist())})
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
    symbols = symbols_of(case, host, size, rendering.sdfg.arglist())
    device = {key: cupy.asarray(value) for key, value in host.items() if isinstance(value, np.ndarray)}
    scalars = {**symbols, **{key: value for key, value in host.items() if not isinstance(value, np.ndarray)}}
    entry(*[ctypes.c_void_p(device[a].data.ptr) if a in device else scalars[a] for a in rendering.sdfg.arglist()])
    for output, want in expected.items():
        assert np.array_equal(cupy.asnumpy(device[output]), want), (case, output)


if __name__ == "__main__":
    for case in CASES:
        test_the_c_unit_says_why_the_lifted_loop_is_parallel_and_matches_numpy(case)
        test_the_cuda_units_say_why_the_lifted_loop_is_parallel_and_match_numpy(case)
