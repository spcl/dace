# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize lifts LINEAR recurrences to a parallel ``Scan`` and keeps nonlinear ones sequential."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import scan
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.corpus import corpus_suite as suite
from tests.corpus.gpu_offload_corpus_test import canon_gpu
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

N = dace.symbol("N")
M = dace.symbol("M", dtype=dace.int64)


def scan_ops(sdfg: dace.SDFG) -> list[ScanOp]:
    return [node.op for node, parent in sdfg.all_nodes_recursive() if isinstance(node, Scan)]


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def canonicalized_kernel(name: str) -> tuple[tsvc.TSVCKernel, dace.SDFG]:
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, "linear_scan", simplify=True)
    canonicalize(sdfg, validate=True)
    return kernel, sdfg


def assert_matches_reference(kernel: tsvc.TSVCKernel, sdfg: dace.SDFG) -> None:
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {name: value.copy() for name, value in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    sdfg(**arrays, **call_kwargs)
    for name, value in arrays.items():
        assert np.allclose(value, ref[name]), f"{kernel.name}: {name} diverged"


# Reproducer: s321 a[i] += a[i-1]*b[i] stayed sequential: the a[i] read was refused as a carry (LLR tsvc_2_s321)
def test_s321_lifts_to_an_affine_scan():
    kernel, sdfg = canonicalized_kernel("s321_d_single")
    assert scan_ops(sdfg) == [ScanOp.AFFINE]
    assert not loops(sdfg), f"the recurrence loop survived: {[r.label for r in loops(sdfg)]}"
    assert_matches_reference(kernel, sdfg)


def test_s256_inner_recurrence_lifts_to_an_affine_scan():
    kernel, sdfg = canonicalized_kernel("s256_d_single")
    assert scan_ops(sdfg) == [ScanOp.AFFINE]
    # Only the outer i loop is left: it re-runs the scan on a, which it carries.
    remaining = loops(sdfg)
    assert len(remaining) == 1 and any(isinstance(n, Scan) for n, p in remaining[0].all_nodes_recursive())
    assert_matches_reference(kernel, sdfg)


@dace.program
def computed_delta(out: dace.float64[N], a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        out[i + 1] = out[i] + a[i] * b[i] + np.sin(a[i])


def test_computed_delta_lifts_to_an_additive_scan():
    sdfg = computed_delta.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert scan_ops(sdfg) == [ScanOp.SUM]
    assert not loops(sdfg), f"the recurrence loop survived: {[r.label for r in loops(sdfg)]}"
    rng = np.random.default_rng(5)
    n = 257
    out, a, b = rng.random(n), rng.random(n), rng.random(n)
    want = out.copy()
    for i in range(n - 1):
        want[i + 1] = want[i] + a[i] * b[i] + np.sin(a[i])
    sdfg(out=out, a=a, b=b, N=n)
    assert np.allclose(out, want)


# Reproducer: s322 a[i] = a[i] + a[i-1]*b[i] + a[i-2]*c[i] stayed sequential: two carry offsets (LLR tsvc_2_s322)
def test_s322_lifts_to_a_second_order_scan():
    kernel, sdfg = canonicalized_kernel("s322_d_single")
    assert scan_ops(sdfg) == [ScanOp.LINEAR2]
    assert not loops(sdfg), f"the recurrence loop survived: {[r.label for r in loops(sdfg)]}"
    assert_matches_reference(kernel, sdfg)


def linear2_sdfg(implementation: str, seed_on_device: bool = False) -> dace.SDFG:
    on_gpu = implementation == "CUDA"
    sdfg = dace.SDFG(f"linear2_scan_{implementation}_{int(seed_on_device)}")
    storage = dace.StorageType.GPU_Global if on_gpu else dace.StorageType.Default
    for name in ("c1", "c2", "d", "out"):
        sdfg.add_array(name, [M], dace.float64, storage=storage)
    sdfg.add_array("seed", [2], dace.float64, storage=storage if seed_on_device else dace.StorageType.Default)
    state = sdfg.add_state()
    node = Scan("linear2", op=ScanOp.LINEAR2)
    node.implementation = implementation
    if on_gpu:
        node.schedule = dace.ScheduleType.GPU_Device
    node.add_in_connector(scan.INIT_CONNECTOR_NAME)
    for conn, name in (
        (scan.INPUT_CONNECTOR_NAME, "d"),
        (scan.COEF_CONNECTOR_NAME, "c1"),
        (scan.COEF2_CONNECTOR_NAME, "c2"),
    ):
        state.add_edge(state.add_read(name), None, node, conn, dace.Memlet(f"{name}[0:M]"))
    state.add_edge(state.add_read("seed"), None, node, scan.INIT_CONNECTOR_NAME, dace.Memlet("seed[0:2]"))
    state.add_edge(node, scan.OUTPUT_CONNECTOR_NAME, state.add_write("out"), None, dace.Memlet("out[0:M]"))
    return sdfg


def linear2_case(n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(n)
    # |c1| + |c2| < 1 keeps the recurrence contracting, so the reference itself is well conditioned
    args = {
        "c1": 0.5 * rng.random(n),
        "c2": 0.4 * rng.random(n),
        "d": rng.standard_normal(n),
        "seed": np.array([0.3, -0.7]),
    }
    want = np.empty(n)
    prev2, prev = args["seed"]
    for k in range(n):
        want[k] = args["c1"][k] * prev + args["c2"][k] * prev2 + args["d"][k]
        prev2, prev = prev, want[k]
    return args, want


@pytest.mark.parametrize("implementation", ["CPU", "pure", "sequential"])
@pytest.mark.parametrize("n", [2, 17, 40001])
def test_second_order_scan_matches_the_sequential_recurrence(implementation: str, n: int):
    args, want = linear2_case(n)
    out = np.zeros(n)
    linear2_sdfg(implementation)(**args, out=out, M=n)
    assert np.allclose(out, want, rtol=0, atol=1e-11)


@pytest.mark.gpu
@pytest.mark.parametrize("seed_on_device", [False, True])
@pytest.mark.parametrize("n", [2, 17, 100003])
def test_the_cuda_second_order_scan_matches_the_sequential_recurrence(seed_on_device: bool, n: int):
    import cupy as cp

    args, want = linear2_case(n)
    device = {name: cp.asarray(value) for name, value in args.items() if name != "seed" or seed_on_device}
    out = cp.zeros(n)
    linear2_sdfg("CUDA", seed_on_device)(**{**args, **device}, out=out, M=n)
    assert np.allclose(cp.asnumpy(out), want, rtol=0, atol=1e-11)


@pytest.mark.gpu
def test_s322_lifts_to_a_second_order_scan_on_the_gpu():
    ctx = suite.make("tsvc", "s322_d_single")
    sdfg = suite.build(ctx, canon_gpu, "linear2_gpu")
    assert scan_ops(sdfg) == [ScanOp.LINEAR2]
    assert not loops(sdfg), f"the recurrence loop survived: {[r.label for r in loops(sdfg)]}"
    assert suite.run_matches(ctx, sdfg)


def test_s222_nonlinear_recurrence_stays_sequential():
    kernel, sdfg = canonicalized_kernel("s222_d_single")
    assert not scan_ops(sdfg), "e[i] = e[i-1]**2 is not linear in the carry"
    assert loops(sdfg)
    assert_matches_reference(kernel, sdfg)


if __name__ == "__main__":
    test_s321_lifts_to_an_affine_scan()
    test_s256_inner_recurrence_lifts_to_an_affine_scan()
    test_computed_delta_lifts_to_an_additive_scan()
    test_s322_lifts_to_a_second_order_scan()
    for implementation in ("CPU", "pure", "sequential"):
        for n in (2, 17, 40001):
            test_second_order_scan_matches_the_sequential_recurrence(implementation, n)
    for seed_on_device in (False, True):
        for n in (2, 17, 100003):
            test_the_cuda_second_order_scan_matches_the_sequential_recurrence(seed_on_device, n)
    test_s322_lifts_to_a_second_order_scan_on_the_gpu()
    test_s222_nonlinear_recurrence_stays_sequential()
