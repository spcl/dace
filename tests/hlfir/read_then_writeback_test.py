"""Phase I — read-then-writeback to the same scalar/length-1 array
should produce two distinct access nodes in one state, never a cycle.

The velocity_tendencies pattern that surfaced this bug:

    max_vcfl_dyn = MAX(p_diag % max_vcfl_dyn, MAXVAL(vcflmax(s:e)))
    p_diag % max_vcfl_dyn = max_vcfl_dyn

After ``hlfir-lift-reduction-operands`` (committed in e3cfbcc68) emits
the lifted reduction temp, both assigns land in the same state.  The
second assign's writeback target was the first assign's RHS read; the
bridge's ``emit_scalar_assign`` was previously reusing the cached read
node for the write, producing a state where ONE access node had both
incoming and outgoing edges — invalid SDFG topology.

This test fixes the pattern at minimal scale (no MAXVAL, no struct,
no inlined callee) so a regression here surfaces independently of
Phases F / A / B / G work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_read_then_writeback_no_cycle(tmp_path: Path):
    """``out = max(out, x); ... ; out = out + 1`` — every state that
    contains both a read of ``out`` and a write to ``out`` must have
    TWO distinct access nodes for ``out`` (one input, one output)."""
    src = """
subroutine kernel(out, x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: out
  integer :: i
  do i = 1, n
    out = max(out, x(i))
  end do
end subroutine kernel
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()

    # Walk every state; for any state that touches ``out``, count the
    # number of ``out`` access nodes.  Whenever the state has BOTH an
    # in-edge AND an out-edge on data ``out``, the count must be ≥ 2.
    from dace.sdfg import nodes as nd
    bad = []
    for state in sdfg.all_states():
        out_nodes = [n for n in state.nodes() if isinstance(n, nd.AccessNode) and n.data == "out"]
        if not out_nodes:
            continue
        has_in = any(state.in_degree(n) > 0 for n in out_nodes)
        has_out = any(state.out_degree(n) > 0 for n in out_nodes)
        if has_in and has_out and len(out_nodes) < 2:
            bad.append((state.label, len(out_nodes)))
    assert not bad, f"states with read+write on a single 'out' node: {bad}"

    # End-to-end numerical correctness: kernel must compute max(initial, x[0..n-1]).
    rng = np.random.default_rng(0)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n))
    out = np.array([0.5], dtype=np.float64)
    sdfg(out=out, x=x, n=n)
    assert out[0] == max(0.5, x.max())


def test_read_then_writeback_two_assigns_same_state(tmp_path: Path):
    """The exact velocity_tendencies shape: two textual statements,
    second writes target that was the first's RHS read."""
    src = """
subroutine kernel(state, x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: state
  real(8) :: tmp
  tmp = max(state, maxval(x(1:n)))
  state = tmp
end subroutine kernel
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()

    from dace.sdfg import nodes as nd
    bad = []
    for s in sdfg.all_states():
        st_nodes = [n for n in s.nodes() if isinstance(n, nd.AccessNode) and n.data == "state"]
        if not st_nodes:
            continue
        has_in = any(s.in_degree(n) > 0 for n in st_nodes)
        has_out = any(s.out_degree(n) > 0 for n in st_nodes)
        if has_in and has_out and len(st_nodes) < 2:
            bad.append((s.label, len(st_nodes)))
    assert not bad, f"read-then-writeback on single 'state' node: {bad}"

    rng = np.random.default_rng(1)
    n = 16
    x = np.asfortranarray(rng.standard_normal(n))
    state = np.array([0.25], dtype=np.float64)
    sdfg(state=state, x=x, n=n)
    assert state[0] == max(0.25, x.max())
