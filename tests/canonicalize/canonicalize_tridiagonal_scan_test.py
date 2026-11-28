# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Thomas tridiagonal solve: its back substitution is a reverse affine recurrence, lifted to a scan."""

from unittest import mock

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import scan
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg.state import LoopRegion
from dace.transformation.passes import loop_to_scan
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.gpu_offload_corpus_test import canon_gpu
from tests.corpus.tsvc_2_5 import tsvc_2_5
from tests.corpus.tsvc_2_5.tsvc_2_5_numpy import ref_thomas_solve

N = dace.symbol("N", dtype=dace.int64)


def scan_ops(sdfg: dace.SDFG) -> list[ScanOp]:
    return [node.op for node, parent in sdfg.all_nodes_recursive() if isinstance(node, Scan)]


def loops(sdfg: dace.SDFG) -> list[str]:
    return [r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


@dace.program
def back_substitution(c: dace.float64[N], d: dace.float64[N], x: dace.float64[N]):
    x[N - 1] = d[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]


def tridiagonal_system(n: int) -> dict[str, np.ndarray]:
    """A diagonally dominant system, so the sequential reference is well conditioned."""
    rng = np.random.default_rng(n)
    a, c = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    return {
        "a": a,
        "b": 2.5 + np.abs(a) + np.abs(c) + rng.random(n),
        "c": c,
        "d": rng.standard_normal(n),
        "x": np.zeros(n),
    }


# Reproducer: thomas_solve back substitution stayed sequential: a reverse affine recurrence (LLR thomas_solve)
@pytest.mark.parametrize("target", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [2, 3, 1000])
def test_a_reverse_affine_recurrence_lifts_to_an_affine_scan(target: str, n: int):
    sdfg = back_substitution.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target=target)
    assert scan_ops(sdfg) == [ScanOp.AFFINE]
    assert not loops(sdfg), f"the recurrence loop survived: {loops(sdfg)}"
    rng = np.random.default_rng(n)
    c, d = rng.uniform(-0.9, 0.9, n), rng.standard_normal(n)
    want = np.empty(n)
    want[n - 1] = d[n - 1]
    # The back substitution itself, sequentially: the reference
    for i in range(n - 2, -1, -1):
        want[i] = d[i] - c[i] * want[i + 1]
    x = np.zeros(n)
    sdfg(c=c, d=d, x=x, N=n)
    assert np.allclose(x, want, rtol=1e-12, atol=1e-12)


# Reproducer: thomas_solve forward sweep stayed sequential: two carriers, c linear-fractional (LLR thomas_solve)
@pytest.mark.parametrize("n", [2, 1000, 100000])
def test_the_thomas_solve_is_three_scans_and_matches_numpy(n: int):
    sdfg = tsvc_2_5.thomas_solve.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert sorted(op.value for op in scan_ops(sdfg)) == ["affine", "affine", "moebius"]
    assert not loops(sdfg), f"a sweep survived: {loops(sdfg)}"
    system = tridiagonal_system(n)
    want = {name: value.copy() for name, value in system.items()}
    ref_thomas_solve(**want)
    sdfg(**system, LEN_1D=n)
    assert np.allclose(system["x"], want["x"], rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_the_thomas_solve_runs_as_three_device_scans():
    sdfg = tsvc_2_5.thomas_solve.to_sdfg(simplify=True)
    sdfg.name = "thomas_solve_device"
    canon_gpu(sdfg)
    assert sorted(op.value for op in scan_ops(sdfg)) == ["affine", "affine", "moebius"]
    assert not loops(sdfg), f"a sweep survived: {loops(sdfg)}"
    # A short and a long system: the edge case and the numerics
    for n in (2, 100000):
        system = tridiagonal_system(n)
        want = {name: value.copy() for name, value in system.items()}
        ref_thomas_solve(**want)
        sdfg(**system, LEN_1D=n)
        assert np.allclose(system["x"], want["x"], rtol=1e-12, atol=1e-12)


@dace.program
def reads_the_other_carrier_before_its_store(
    a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]
):
    for i in range(1, N):
        d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i])
        c[i] = c[i] / (b[i] - a[i] * c[i - 1])


def test_a_carrier_reading_the_other_before_its_store_is_not_split():
    """``d`` reads ``c[i]`` before this iteration stores it; split, it would read the value the ``c`` loop stored."""
    sdfg = reads_the_other_carrier_before_its_store.to_sdfg(simplify=True)
    seen = []
    original = loop_to_scan.carrier_slices

    def record(loop: LoopRegion, owner: dace.SDFG):
        seen.append(original(loop, owner))
        return seen[-1]

    with mock.patch.object(loop_to_scan, "carrier_slices", side_effect=record):
        canonicalize(sdfg, validate=True)
    assert seen and all(result is None for result in seen), "the split was offered"
    assert loops(sdfg)
    system = tridiagonal_system(64)
    a, b, c, d = (system[k] for k in "abcd")
    want_c, want_d = c.copy(), d.copy()
    # The loop itself, sequentially: the reference
    for i in range(1, 64):
        want_d[i] = (want_d[i] - a[i] * want_d[i - 1]) / (b[i] - a[i] * want_c[i])
        want_c[i] = want_c[i] / (b[i] - a[i] * want_c[i - 1])
    sdfg(a=a, b=b, c=c, d=d, N=64)
    assert np.allclose(c, want_c) and np.allclose(d, want_d)


@dace.program
def quadratic_denominator(p: dace.float64[N], q: dace.float64[N], x: dace.float64[N]):
    for i in range(1, N):
        x[i] = p[i] / (q[i] + x[i - 1] * x[i - 1])


def test_a_denominator_quadratic_in_the_carry_is_not_a_moebius_scan():
    """``x[i] = p[i] / (q[i] + x[i-1]^2)`` is one product short of linear-fractional: it stays a loop."""
    sdfg = quadratic_denominator.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert ScanOp.MOEBIUS not in scan_ops(sdfg)
    assert loops(sdfg)
    rng = np.random.default_rng(7)
    p, q, x = rng.uniform(0.5, 1.5, 1000), rng.uniform(1.0, 2.0, 1000), np.full(1000, 0.3)
    want = x.copy()
    # The loop itself, sequentially: the reference
    for i in range(1, 1000):
        want[i] = p[i] / (q[i] + want[i - 1] * want[i - 1])
    sdfg(p=p, q=q, x=x, N=1000)
    assert np.allclose(x, want, rtol=1e-12, atol=1e-12)


def moebius_sdfg(implementation: str, seed_on_device: bool = False) -> dace.SDFG:
    on_gpu = implementation == "CUDA"
    sdfg = dace.SDFG(f"moebius_scan_{implementation}_{int(seed_on_device)}")
    storage = dace.StorageType.GPU_Global if on_gpu else dace.StorageType.Default
    # Each operand and the output: one array of N
    for name in ("p1", "p0", "q1", "q0", "out"):
        sdfg.add_array(name, [N], dace.float64, storage=storage)
    sdfg.add_array("seed", [1], dace.float64, storage=storage if seed_on_device else dace.StorageType.Default)
    state = sdfg.add_state()
    node = Scan("moebius", op=ScanOp.MOEBIUS)
    node.implementation = implementation
    if on_gpu:
        node.schedule = dace.ScheduleType.GPU_Device
    node.add_in_connector(scan.INIT_CONNECTOR_NAME)
    # Each coefficient connector: wired to its array
    for conn, name in (
        (scan.COEF_CONNECTOR_NAME, "p1"),
        (scan.INPUT_CONNECTOR_NAME, "p0"),
        (scan.COEF2_CONNECTOR_NAME, "q1"),
        (scan.DEN_CONNECTOR_NAME, "q0"),
    ):
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet(f"{name}[0:N]"))
    state.add_edge(state.add_read("seed"), None, node, scan.INIT_CONNECTOR_NAME, dace.Memlet("seed[0]"))
    state.add_edge(node, scan.OUTPUT_CONNECTOR_NAME, state.add_write("out"), None, dace.Memlet("out[0:N]"))
    return sdfg


def moebius_case(n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``x = p0[k] / (q1[k] * x + q0[k])`` over a diagonally dominant tridiagonal system, so it stays bounded."""
    system = tridiagonal_system(n)
    args = {
        "p1": np.zeros(n),
        "p0": system["c"],
        "q1": -system["a"],
        "q0": system["b"],
        "seed": np.array([0.1]),
    }
    want = np.empty(n)
    x = args["seed"][0]
    # The recurrence itself, sequentially: the reference
    for k in range(n):
        x = (args["p1"][k] * x + args["p0"][k]) / (args["q1"][k] * x + args["q0"][k])
        want[k] = x
    return args, want


@pytest.mark.parametrize("implementation", ["CPU", "pure", "sequential"])
@pytest.mark.parametrize("n", [1, 17, 100003])
def test_the_moebius_scan_matches_the_sequential_recurrence(implementation: str, n: int):
    args, want = moebius_case(n)
    out = np.zeros(n)
    moebius_sdfg(implementation)(**args, out=out, N=n)
    assert np.allclose(out, want, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("seed_on_device", [False, True])
@pytest.mark.parametrize("n", [1, 17, 100003])
def test_the_cuda_moebius_scan_matches_the_sequential_recurrence(seed_on_device: bool, n: int):
    import cupy as cp

    args, want = moebius_case(n)
    device = {name: cp.asarray(value) for name, value in args.items() if name != "seed" or seed_on_device}
    out = cp.zeros(n)
    moebius_sdfg("CUDA", seed_on_device)(**{**args, **device}, out=out, N=n)
    assert np.allclose(cp.asnumpy(out), want, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    # Every test, once per parametrized case
    for target in ("cpu", "gpu"):
        for n in (2, 3, 1000):
            test_a_reverse_affine_recurrence_lifts_to_an_affine_scan(target, n)
    for n in (2, 1000, 100000):
        test_the_thomas_solve_is_three_scans_and_matches_numpy(n)
    test_the_thomas_solve_runs_as_three_device_scans()
    test_a_carrier_reading_the_other_before_its_store_is_not_split()
    test_a_denominator_quadratic_in_the_carry_is_not_a_moebius_scan()
    for implementation in ("CPU", "pure", "sequential"):
        for n in (1, 17, 100003):
            test_the_moebius_scan_matches_the_sequential_recurrence(implementation, n)
    for seed_on_device in (False, True):
        for n in (1, 17, 100003):
            test_the_cuda_moebius_scan_matches_the_sequential_recurrence(seed_on_device, n)
