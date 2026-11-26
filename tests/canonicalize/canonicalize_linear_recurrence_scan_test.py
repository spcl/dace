# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize lifts LINEAR recurrences to a parallel ``Scan`` and keeps nonlinear ones sequential."""

import numpy as np

import dace
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

N = dace.symbol("N")


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


def test_s222_nonlinear_recurrence_stays_sequential():
    kernel, sdfg = canonicalized_kernel("s222_d_single")
    assert not scan_ops(sdfg), "e[i] = e[i-1]**2 is not linear in the carry"
    assert loops(sdfg)
    assert_matches_reference(kernel, sdfg)


if __name__ == "__main__":
    test_s321_lifts_to_an_affine_scan()
    test_s256_inner_recurrence_lifts_to_an_affine_scan()
    test_computed_delta_lifts_to_an_additive_scan()
    test_s222_nonlinear_recurrence_stays_sequential()
