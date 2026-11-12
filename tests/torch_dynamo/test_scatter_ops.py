# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``slice_scatter`` / ``select_scatter`` lowerings: strided writes into a slice, with in-place elision of the copy."""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import functools  # noqa: E402
import operator  # noqa: E402

import sympy  # noqa: E402
import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402
from dace.sdfg.analysis.schedule_tree import treenodes as tn  # noqa: E402


def _check(backend, fn, inputs_per_call, expected_compiles=None):
    compiled = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        for inputs in inputs_per_call:
            torch.testing.assert_close(compiled(*inputs), fn(*inputs), equal_nan=True)
    if expected_compiles is not None:
        assert backend.compile_count == expected_compiles


def _has_full_buffer_copy(stree: tn.ScheduleTreeRoot) -> bool:
    """Whether the schedule tree contains a copy that writes a whole container (a full-buffer copy of the input)."""
    for node in stree.preorder_traversal():
        if not isinstance(node, tn.CopyNode):
            continue
        desc = stree.containers.get(node.target)
        if desc is None:
            continue
        total = functools.reduce(operator.mul, desc.shape, 1)
        written = node.memlet.other_subset.num_elements() if node.memlet.other_subset is not None else total
        if sympy.simplify(sympy.sympify(written) - sympy.sympify(total)) == 0:
            return True
    return False


# ---------------------------------------------------------------------- slice_scatter (correctness)
@pytest.mark.torch
def test_slice_scatter_constant_bounds(backend):

    def f(x, v):
        out = x * 2
        out[1:4] = v
        return out + 1

    _check(backend, f, [(torch.randn(6, 5), torch.randn(3, 5)), (torch.randn(9, 7), torch.randn(3, 7))])


@pytest.mark.torch
def test_slice_scatter_symbolic_bound(backend):

    def f(x, v):
        n = v.shape[0]  # a symbolic bound (Dynamo SymInt)
        out = x * 2
        out[:n] = v
        return out

    _check(backend, f, [(torch.randn(6, 5), torch.randn(3, 5)), (torch.randn(9, 7), torch.randn(5, 7))])


@pytest.mark.torch
def test_slice_scatter_step(backend):

    def f(x, v):
        out = x * 2
        out[0:6:2] = v
        return out

    _check(backend, f, [(torch.randn(6, 5), torch.randn(3, 5)), (torch.randn(10, 7), torch.randn(3, 7))])


@pytest.mark.torch
def test_slice_scatter_higher_dim(backend):

    def f(x, v):
        out = x * 2
        out[:, 1:3] = v
        return out

    _check(backend, f, [(torch.randn(6, 5), torch.randn(6, 2)), (torch.randn(9, 7), torch.randn(9, 2))])


@pytest.mark.torch
def test_slice_scatter_accumulate(backend):

    def f(x, v):
        out = x * 2
        out[1:4] += v
        return out

    _check(backend, f, [(torch.randn(6, 5), torch.randn(3, 5)), (torch.randn(9, 7), torch.randn(3, 7))])


# ---------------------------------------------------------------------- select_scatter (correctness)
@pytest.mark.torch
def test_select_scatter_last_dim(backend):

    def f(x, v):
        out = x * 2
        out[..., 1] = v
        return out

    _check(backend, f, [(torch.randn(6, 5), torch.randn(6)), (torch.randn(9, 7), torch.randn(9))])


@pytest.mark.torch
def test_select_scatter_negative_index(backend):

    def f(x, v):
        out = x * 2
        out[-1] = v
        return out + 1

    _check(backend, f, [(torch.randn(6, 5), torch.randn(5)), (torch.randn(9, 7), torch.randn(7))])


# ---------------------------------------------------------------------- copy path (input stays alive)
@pytest.mark.torch
def test_scatter_input_alive_uses_copy(backend):

    def f(x, v, w):
        base = x * 2
        a = torch.slice_scatter(base, v, 0, 1, 4)
        b = torch.slice_scatter(base, w, 0, 2, 5)  # `base` read again: both scatters copy it
        return a + b

    _check(
        backend,
        f,
        [
            (torch.randn(6, 5), torch.randn(3, 5), torch.randn(3, 5)),
            (torch.randn(9, 7), torch.randn(3, 7), torch.randn(3, 7)),
        ],
    )


# ---------------------------------------------------------------------- in-place elision (structural)
@pytest.mark.torch
def test_inplace_elision_emits_no_full_copy():

    def inplace(x, v):
        base = x * 2  # a dead transient after the scatter (its only user): written in place
        return torch.slice_scatter(base, v, 0, 1, 4)

    def keepalive(x, v):
        base = x * 2  # returned as well, so the scatter must copy it rather than overwrite it
        return torch.slice_scatter(base, v, 0, 1, 4), base

    be_inplace = DaceBackend()
    be_copy = DaceBackend()
    with torch.no_grad():
        torch.compile(inplace, backend=be_inplace, dynamic=True)(torch.randn(6, 8), torch.randn(3, 8))
        torch.compile(keepalive, backend=be_copy, dynamic=True)(torch.randn(6, 8), torch.randn(3, 8))

    assert not _has_full_buffer_copy(be_inplace.last_result.stree), "in-place scatter emitted a whole-buffer copy"
    assert _has_full_buffer_copy(be_copy.last_result.stree), "copy-path scatter should copy the whole buffer"


@pytest.mark.torch
def test_inplace_elision_through_functionalized_setitem():
    # `out[i:] = v` functionalizes to slice + aten.copy + slice_scatter, so `out` has more than one user, yet nothing
    # reads its data after the scatter: the write must still be elided (no whole-buffer copy). `i` is symbolic.

    def f(x, v):
        i = x.shape[0] // 2
        out = x * 2
        out[i:] = v
        return out

    be = DaceBackend()
    x, v = torch.randn(6, 8), torch.randn(3, 8)
    with torch.no_grad():
        got = torch.compile(f, backend=be, dynamic=True)(x, v)
        torch.testing.assert_close(got, f(x, v))
    assert not _has_full_buffer_copy(be.last_result.stree), "functionalized setitem should write in place"


@pytest.mark.torch
def test_scatter_input_read_after_through_view_uses_copy():
    # A view of the input is created before the scatter but read after it: the input's data is still live, so the
    # scatter must copy (not overwrite in place), and the results must stay correct.

    def f(x, v):
        out = x * 2
        s = out[0:2]  # view of `out`, created before the scatter
        out = torch.slice_scatter(out, v, 0, 2, 4)
        return out, s + 1  # `s` (hence `out`'s data) read after the scatter

    be = DaceBackend()
    x, v = torch.randn(6, 8), torch.randn(2, 8)
    with torch.no_grad():
        got = torch.compile(f, backend=be, dynamic=True)(x, v)
        ref = f(x, v)
        torch.testing.assert_close(got[0], ref[0])
        torch.testing.assert_close(got[1], ref[1])
    assert _has_full_buffer_copy(be.last_result.stree), "a view read after the scatter must force the copy path"


if __name__ == "__main__":
    for test in (
        test_slice_scatter_constant_bounds,
        test_slice_scatter_symbolic_bound,
        test_slice_scatter_step,
        test_slice_scatter_higher_dim,
        test_slice_scatter_accumulate,
        test_select_scatter_last_dim,
        test_select_scatter_negative_index,
        test_scatter_input_alive_uses_copy,
    ):
        torch._dynamo.reset()
        test(DaceBackend())
        print(test.__name__, "ok")
    torch._dynamo.reset()
    test_inplace_elision_emits_no_full_copy()
    print("test_inplace_elision_emits_no_full_copy ok")
    for test in (
        test_inplace_elision_through_functionalized_setitem,
        test_scatter_input_read_after_through_view_uses_copy,
    ):
        torch._dynamo.reset()
        test()
        print(test.__name__, "ok")
