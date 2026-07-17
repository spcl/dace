# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regressions for the apply_assignment liveness planner (adversarial-review findings, 2026-07-17):

  * a segment whose FIRST kernel does not touch the array must still get its entry conversion when
    a LATER kernel of the segment reads live-in data (first-touch scan, not kernels[start]);
  * an untouched segment must not break the conversion chain -- the value keeps living in the last
    MATERIALIZED holder, and a later same-layout segment ALIASES onto it with no conversion;
  * a PARTIAL first write must not suppress the entry conversion (only a proven full-coverage
    write may -- ``writes_cover_array``);
  * a WCR accumulation reads-modifies its destination, so it counts as a live-in read;
  * a copy with one relaid operand becomes transposing and must be converted to a TensorTranspose
    (the shared PermuteDimensions bookkeeping), never left as a silently-transposing plain copy;
  * a NestedSDFG receiving a reassigned array through a spanning memlet is refused loudly.
"""
import numpy
import pytest

import dace
from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import (IDENTITY_LAYOUT, Layout, apply_assignment, writes_cover_array)
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N")
CM = Layout("perm10", (Permute((1, 0)), ))
ID = IDENTITY_LAYOUT


@dace.program
def gap3(A: dace.float64[N, N], X: dace.float64[N, N], C: dace.float64[N, N - 1], D: dace.float64[N, N - 1]):
    for i, j in dace.map[0:N, 0:N]:
        X[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N - 1]:
        C[i, j] = A[j, i] + A[i, j]
    for i, j in dace.map[0:N, 0:N - 1]:
        D[i, j] = X[j, i] + C[i, N - 2 - j]


@dace.program
def partial2(A: dace.float64[N, N], X: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        X[i, j] = A[i, j] * 2.0
    for j in dace.map[0:N]:
        X[0, j] = A[0, j] * 3.0


@dace.program
def wcr2(A: dace.float64[N, N], X: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        X[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N]:
        X[j, i] += A[i, j]


@dace.program
def copy3(A: dace.float64[N, N], X: dace.float64[N, N], C: dace.float64[N, N], Y: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        X[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = X[j, i] + A[i, j]
    Y[:] = X


def split(program):
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def run(sdfg, n, seed, names):
    rng = numpy.random.default_rng(seed)
    arrays = {}
    for name in names:
        shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays[name].shape)
        arrays[name] = rng.random(shape) if name == "A" else numpy.zeros(shape)
    sdfg(**arrays, N=n)
    return arrays


def kernel_data(kernel):
    return {node.data for node in kernel.state.data_nodes()}


def test_gap_segment_gets_entry_conversion(n=48):
    """X written in kernel 0, untouched in kernel 1, read transposed in kernel 2. The trajectory
    [ID, CM, CM]'s second segment STARTS at the untouched kernel; the conversion must still be
    planned (at the segment's first TOUCHING kernel) or kernel 2 reads an unpopulated clone."""
    sdfg, kernels = split(gap3)
    assert "X" not in kernel_data(kernels[1])
    applied = apply_assignment(sdfg, kernels, {"X": [ID, CM, CM]})
    assert len(applied.boundary_states) == 1
    assert applied.exit_state is None  # last write of X was under identity
    sdfg.validate()
    arrays = run(sdfg, n, seed=41, names=["A", "X", "C", "D"])
    a = arrays["A"]
    x = 2.0 * a
    c = a.T[:, :n - 1] + a[:, :n - 1]
    d = x.T[:n, :n - 1] + c[:, ::-1]
    assert numpy.allclose(arrays["X"], x)
    assert numpy.allclose(arrays["D"], d)


def test_untouched_segment_aliases_no_conversion(n=48):
    """[CM, ID, CM] with the middle kernel untouched: the identity segment stays unmaterialized and
    the third segment ALIASES onto the seg0 clone -- zero boundary conversions, one exit conversion
    restoring the original X."""
    sdfg, kernels = split(gap3)
    applied = apply_assignment(sdfg, kernels, {"X": [CM, ID, CM]})
    assert applied.segment_names["X"] == ["X__seg0_perm10"] * 3
    assert applied.boundary_states == []
    assert applied.exit_state is not None
    assert "X__seg0_perm10" in kernel_data(kernels[2]) and "X" not in kernel_data(kernels[2])
    sdfg.validate()
    arrays = run(sdfg, n, seed=42, names=["A", "X", "C", "D"])
    assert numpy.allclose(arrays["X"], 2.0 * arrays["A"])


def test_partial_write_needs_entry_conversion(n=16):
    """Kernel 1 writes only ROW 0 of X under perm10: without the coverage proof the clone would
    hold garbage everywhere else and the exit conversion would clobber X with it."""
    sdfg, kernels = split(partial2)
    assert not writes_cover_array(kernels[1].state, "X")
    assert writes_cover_array(kernels[0].state, "X")
    applied = apply_assignment(sdfg, kernels, {"X": [ID, CM]})
    assert len(applied.boundary_states) == 1  # the pre-fill of the partial writer's clone
    assert applied.exit_state is not None
    sdfg.validate()
    arrays = run(sdfg, n, seed=43, names=["A", "X"])
    expected = 2.0 * arrays["A"]
    expected[0, :] = 3.0 * arrays["A"][0, :]
    assert numpy.allclose(arrays["X"], expected)


def test_wcr_write_counts_as_read(n=32):
    """Kernel 1 only ACCUMULATES into X (a WCR write, no source access node): the segment still
    needs the live-in values, so the entry conversion must be inserted."""
    sdfg, kernels = split(wcr2)
    assert any(e.data is not None and e.data.wcr is not None for e in kernels[1].state.edges())
    applied = apply_assignment(sdfg, kernels, {"X": [ID, CM]})
    assert len(applied.boundary_states) == 1
    assert applied.exit_state is not None
    sdfg.validate()
    arrays = run(sdfg, n, seed=44, names=["A", "X"])
    assert numpy.allclose(arrays["X"], 2.0 * arrays["A"] + arrays["A"].T)


def test_relaid_copy_becomes_tensor_transpose(n=24):
    """Y[:] = X shares kernel 1's state; relaying X out by perm10 makes the plain copy transposing.
    The shared retranspose bookkeeping must convert it (Y == X, not X^T)."""
    from dace.libraries.linalg import TensorTranspose

    sdfg, kernels = split(copy3)
    applied = apply_assignment(sdfg, kernels, {"X": [ID, CM]})
    assert len(applied.boundary_states) == 1
    transposes = [node for kernel in kernels for node in kernel.state.nodes() if isinstance(node, TensorTranspose)]
    assert len(transposes) == 1
    sdfg.validate()
    arrays = run(sdfg, n, seed=45, names=["A", "X", "C", "Y"])
    assert numpy.allclose(arrays["Y"], 2.0 * arrays["A"])  # the transposing-copy bug yields X^T
    assert numpy.allclose(arrays["C"], 2.0 * arrays["A"].T + arrays["A"])


def build_nested_kernel():
    """A hand-built kernel whose map body is a NestedSDFG taking a SPANNING row of X (the
    frontend/prepare pipeline currently cannot produce this shape end to end, so the refusal is
    exercised structurally)."""
    inner = dace.SDFG("inner")
    inner.add_array("xrow", [N], dace.float64)
    inner.add_array("yrow", [N], dace.float64)
    istate = inner.add_state("body", is_start_block=True)
    tasklet = istate.add_tasklet("scale", {"a"}, {"b"}, "b = a * 2.0")
    ime, imx = istate.add_map("inner_m", {"j": "0:N"})
    istate.add_memlet_path(istate.add_access("xrow"), ime, tasklet, dst_conn="a", memlet=dace.Memlet("xrow[j]"))
    istate.add_memlet_path(tasklet, imx, istate.add_access("yrow"), src_conn="b", memlet=dace.Memlet("yrow[j]"))

    outer = dace.SDFG("outer")
    outer.add_array("X", [N, N], dace.float64)
    outer.add_array("Y", [N, N], dace.float64)
    state = outer.add_state("k", is_start_block=True)
    me, mx = state.add_map("outer_m", {"i": "0:N"})
    nsdfg = state.add_nested_sdfg(inner, {"xrow"}, {"yrow"}, symbol_mapping={"N": "N"})
    state.add_memlet_path(state.add_access("X"), me, nsdfg, dst_conn="xrow", memlet=dace.Memlet("X[i, 0:N]"))
    state.add_memlet_path(nsdfg, mx, state.add_access("Y"), src_conn="yrow", memlet=dace.Memlet("Y[i, 0:N]"))
    outer.validate()
    return outer


def test_nested_sdfg_spanning_edge_refused():
    sdfg = build_nested_kernel()
    kernels = line_graph(sdfg)
    assert len(kernels) == 1
    with pytest.raises(NotImplementedError, match="NestedSDFG"):
        apply_assignment(sdfg, kernels, {"X": [CM]})


if __name__ == "__main__":
    test_gap_segment_gets_entry_conversion()
    test_untouched_segment_aliases_no_conversion()
    test_partial_write_needs_entry_conversion()
    test_wcr_write_counts_as_read()
    test_relaid_copy_becomes_tensor_transpose()
    test_nested_sdfg_spanning_edge_refused()
    print("apply_assignment regression tests PASS")
