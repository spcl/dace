"""Smoke tests for the HLFIR -> SDFG frontend.

Mirrors a narrow slice of ``tests/fortran/``'s style (inline Fortran source,
build an SDFG, validate) but exercises the new flang-based pipeline.

Per the project's E2E-numerical rule, every test that builds an SDFG also
compares its output against a non-transformed reference  --  here a numpy
equivalent, which is sufficient for these arithmetic-only kernels (no
structs, no intrinsics).  The structural assertions are kept as a
guard against silent-regressions in the SDFG shape on top of the
numerical check."""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_elementwise_loop(tmp_path):
    """One tasklet, one state  --  c(i) = a(i) + b(i)."""
    src = """
subroutine elementwise_add(a, b, c, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n), b(n)
  real(8), intent(inout) :: c(n)
  integer :: i
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine elementwise_add
"""
    b = build_sdfg(src, tmp_path, name="elementwise_add")
    sdfg = b.build()
    sdfg.validate()
    for nm in ("a", "b", "c"):
        assert nm in sdfg.arrays

    rng = np.random.default_rng(0)
    n = 16
    a = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    b_arr = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    c = np.zeros(n, dtype=np.float64)
    expected = a + b_arr
    sdfg(a=a, b=b_arr, c=c, n=n)
    np.testing.assert_allclose(c, expected, rtol=1e-12, atol=1e-12)


def test_read_after_write_shares_access_node(tmp_path):
    """Two tasklets in the same loop body: the second consumes what the first
    writes.  With the single-access-node rule, exactly one AccessNode for
    ``tmp`` must appear in the innermost state  --  anything else would mean the
    RAW dataflow edge was dropped.  Numerical check confirms the dataflow is
    actually wired (a dropped edge would leave ``out`` reading uninitialised
    transient and the result would diverge from the closed-form expectation)."""
    from dace.sdfg.state import LoopRegion
    from dace.sdfg import nodes as nd

    src = """
subroutine chained(a, out, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: out(n)
  real(8) :: tmp(n)
  integer :: i
  do i = 1, n
    tmp(i) = a(i) * 2.0d0
    out(i) = tmp(i) + 1.0d0
  end do
end subroutine chained
"""
    b = build_sdfg(src, tmp_path, name="chained")
    sdfg = b.build()
    sdfg.validate()

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif hasattr(n, "nodes"):
                yield n

    body = next(s for s in iter_states(sdfg) if any(isinstance(n, nd.Tasklet) for n in s.nodes()))
    tmp_nodes = [n for n in body.nodes() if isinstance(n, nd.AccessNode) and n.data == "tmp"]
    assert len(tmp_nodes) == 1, (f"expected a single shared access node for tmp in the body state; "
                                 f"got {len(tmp_nodes)}")

    rng = np.random.default_rng(1)
    n = 8
    a = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    out = np.zeros(n, dtype=np.float64)
    expected = a * 2.0 + 1.0
    sdfg(a=a, out=out, n=n)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
