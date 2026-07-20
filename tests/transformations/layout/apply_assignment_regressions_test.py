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


@dace.program
def rowwise(A: dace.float64[N, N], X: dace.float64[N, N]):
    for i in dace.map[0:N]:
        X[i, 0:N] = A[i, 0:N] * 2.0


@dace.program
def readonly3(A: dace.float64[N, N], O0: dace.float64[N, N], O1: dace.float64[N, N], O2: dace.float64[N, N]):
    """A is READ-ONLY across three kernels. The reversed reads of the previous output are what keep the
    three nests from fusing into one; A itself is read straight or transposed, never written."""
    for i, j in dace.map[0:N, 0:N]:
        O0[i, j] = A[j, i]
    for i, j in dace.map[0:N, 0:N]:
        O1[i, j] = O0[i, N - 1 - j] + A[i, j]
    for i, j in dace.map[0:N, 0:N]:
        O2[i, j] = O1[i, N - 1 - j] + A[j, i]


@dace.program
def rank3_read(A: dace.float64[N, N, N], X: dace.float64[N, N, N], C: dace.float64[N, N, N],
               Y: dace.float64[N, N, N]):
    """X is RELAID and then READ by the copy `Y[:] = X` -- the 'in' side of the retranspose bookkeeping."""
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        X[i, j, k] = A[i, j, k] * 2.0
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        C[i, j, k] = X[k, i, j] + A[i, j, k]
    Y[:] = X


@dace.program
def rank3_write(A: dace.float64[N, N, N], X: dace.float64[N, N, N], C: dace.float64[N, N, N]):
    """X is RELAID and then WRITTEN by the copy `X[:] = C` -- the 'out' side, which needs the OTHER
    direction of the permutation."""
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        C[i, j, k] = A[k, i, j] * 2.0
    X[:] = C


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


def test_coverage_proof_admits_a_copy_writer_and_a_rowwise_writer():
    """The proof used to demand a MapExit producer with unit per-dimension subsets, so two very common
    full writers -- a whole-array copy (`Y[:] = X`, written by a copy library node) and a row-wise map
    writing `X[i, 0:N]` -- were unprovable. An unprovable pure OUTPUT then gets an entry conversion that
    reads the uninitialized buffer: the answer stays right, but it is a wasted full pass and it poisons
    the MALLOC_PERTURB/valgrind triage this repo leans on."""
    sdfg, kernels = split(copy3)
    assert writes_cover_array(kernels[1].state, "Y")  # produced by the copy library node
    assert writes_cover_array(kernels[1].state, "C")  # the ordinary unit-subset map writer still proves
    assert not writes_cover_array(kernels[1].state, "A")  # a pure read must never claim coverage
    row_sdfg, row_kernels = split(rowwise)
    assert writes_cover_array(row_kernels[0].state, "X")  # non-unit subset spanning a whole dimension
    assert not writes_cover_array(row_kernels[0].state, "A")
    assert row_sdfg is not None


def test_pure_output_copy_skips_the_uninitialized_entry_read(n=24):
    """The consequence of the proof above, end to end: relaying BOTH operands of `Y[:] = X` emits only
    X's entry conversion. Y is written whole, so converting its prior contents is pure waste."""
    from dace.libraries.layout import LayoutChange
    from dace.libraries.linalg import TensorTranspose

    sdfg, kernels = split(copy3)
    applied = apply_assignment(sdfg, kernels, {"X": [ID, CM], "Y": [ID, CM]})
    assert len(applied.boundary_states) == 1
    changed = [n_.name for s in applied.boundary_states for n_ in s.nodes() if isinstance(n_, LayoutChange)]
    assert len(changed) == 1 and "X" in changed[0] and "Y" not in changed[0]
    # both copy operands moved by the SAME permutation, so the copy stays elementwise -- the sibling of
    # test_relaid_copy_becomes_tensor_transpose, and the only path that merges two per-array rewrites
    assert not [n_ for k in kernels for n_ in k.state.nodes() if isinstance(n_, TensorTranspose)]
    sdfg.validate()
    arrays = run(sdfg, n, seed=61, names=["A", "X", "C", "Y"])
    assert numpy.allclose(arrays["Y"], 2.0 * arrays["A"])
    assert numpy.allclose(arrays["C"], 2.0 * arrays["A"].T + arrays["A"])


def test_readonly_array_returns_to_identity_without_touching_the_caller_buffer(n=24):
    """A read-only input keeps its original buffer valid forever, so every clone derives from the
    original and an identity segment just ALIASES it -- no restore transpose, no exit. The load-bearing
    assertion is the last one: a restore would write a transpose back into the caller's input."""
    sdfg, kernels = split(readonly3)
    assert len(kernels) == 3
    P = Layout("perm10", (Permute((1, 0)), ))
    applied = apply_assignment(sdfg, kernels, {"A": [P, ID, P]})
    assert applied.segment_names["A"] == ["A__seg0_perm10", "A", "A__seg2_perm10"]
    assert len(applied.boundary_states) == 2  # two entries; the identity segment costs nothing
    assert applied.exit_state is None  # A is never written, so there is nothing to restore
    assert "A" in {node.data for node in kernels[1].state.data_nodes()}  # reads the untouched original
    sdfg.validate()
    rng = numpy.random.default_rng(62)
    a = rng.random((n, n))
    before = a.copy()
    outs = {name: numpy.zeros((n, n)) for name in ("O0", "O1", "O2")}
    sdfg(A=a, **outs, N=n)
    assert numpy.array_equal(a, before), "the read-only input buffer was modified"
    o0 = before.T
    o1 = o0[:, ::-1] + before
    assert numpy.allclose(outs["O0"], o0)
    assert numpy.allclose(outs["O1"], o1)
    assert numpy.allclose(outs["O2"], o1[:, ::-1] + before.T)


def test_same_tag_different_ops_is_refused():
    """`segments_of` groups by TAG, and the body-uniform check compares tags too, so a tag reused for a
    different op sequence silently collapses two segments into one and the APPLIED plan stops matching
    the priced one -- with no error and a still-valid SDFG."""
    from dace.transformation.layout.apply_assignment import segments_of

    collide = [Layout("t", (Permute((1, 0)), )), Layout("t", ())]
    with pytest.raises(ValueError, match="two different op"):
        segments_of(collide)
    sdfg, kernels = split(copy3)
    with pytest.raises(ValueError, match="two different op"):
        apply_assignment(sdfg, kernels, {"X": collide})
    assert segments_of([CM, CM]) == [(0, 2, CM)]  # the honest same-tag case still groups


P201 = Layout("perm201", (Permute((2, 0, 1)), ))


def test_rank3_relaid_copy_picks_the_right_permutation_direction(n=8):
    """At rank 2 the only non-identity permutation, [1,0], is SELF-INVERSE -- so a forward/inverse mix-up
    in the retranspose axes, in `composed_permutation`, or in the clone shape is invisible to every other
    test in this directory. Rank 3 is inside the supported space (MAX_PERMUTE_NDIM == 3) and does tell the
    two directions apart, so pin BOTH sides: a relaid array being read by a copy, and one being written."""
    from dace.libraries.linalg import TensorTranspose

    read_sdfg, read_kernels = split(rank3_read)
    apply_assignment(read_sdfg, read_kernels, {"X": [ID, P201]})
    read_axes = [list(t.axes) for k in read_kernels for t in k.state.nodes() if isinstance(t, TensorTranspose)]
    read_sdfg.validate()
    arrays = run(read_sdfg, n, seed=63, names=["A", "X", "C", "Y"])
    a = arrays["A"]
    assert numpy.allclose(arrays["Y"], 2.0 * a)  # the copy must undo the relayout, not pass X^T through
    assert numpy.allclose(arrays["C"], (2.0 * a).transpose(1, 2, 0) + a)  # X[k,i,j] IS transpose(1,2,0)

    write_sdfg, write_kernels = split(rank3_write)
    apply_assignment(write_sdfg, write_kernels, {"X": [P201]})
    write_axes = [list(t.axes) for k in write_kernels for t in k.state.nodes() if isinstance(t, TensorTranspose)]
    write_sdfg.validate()
    out = run(write_sdfg, n, seed=64, names=["A", "X", "C"])
    assert numpy.allclose(out["X"], 2.0 * out["A"].transpose(1, 2, 0))

    # the load-bearing structural claim: the two sides use OPPOSITE permutations, which is precisely what
    # rank 2 cannot show
    assert read_axes and write_axes and read_axes != write_axes, (read_axes, write_axes)


if __name__ == "__main__":
    test_gap_segment_gets_entry_conversion()
    test_untouched_segment_aliases_no_conversion()
    test_partial_write_needs_entry_conversion()
    test_wcr_write_counts_as_read()
    test_relaid_copy_becomes_tensor_transpose()
    test_nested_sdfg_spanning_edge_refused()
    test_coverage_proof_admits_a_copy_writer_and_a_rowwise_writer()
    test_pure_output_copy_skips_the_uninitialized_entry_read()
    test_readonly_array_returns_to_identity_without_touching_the_caller_buffer()
    test_same_tag_different_ops_is_refused()
    test_rank3_relaid_copy_picks_the_right_permutation_direction()
    print("apply_assignment regression tests PASS")
