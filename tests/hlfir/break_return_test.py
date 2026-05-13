"""Verify the builder-side emit handlers for BreakBlock and ReturnBlock.

Flang drops Fortran ``exit`` / ``return`` to raw ``cf.cond_br`` /
``cf.br`` before any of our bridge passes run, and ``lift-cf-to-scf``
either absorbs them into an ``scf.while`` condition or gives up  --  so
bridge-side detection is a separate workstream.  This test covers the
other side of that future pipe: given a ``SDFGBuilder`` seeded with an
AST that contains ``kind="break"`` / ``kind="return"`` nodes, the
emitted SDFG must be structurally correct, validate, compile, and run
end-to-end.

The AST is a stub class (not the nanobind-bound ASTNode) since we
can't construct those from Python  --  it just needs the fields the
emitters read.

Each emitted SDFG also gets called with concrete inputs and the
resulting array contents are checked numerically against a hand-rolled
reference.  Bridge-side ``EXIT`` is covered end-to-end through real
Fortran source by ``do_loop_exit_test.py`` (where Flang's lift-cf-to-scf
turns the EXIT into an ``scf.while`` keep-going condition); these tests
are the focused unit-test for the emit handlers themselves.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

_HLFIR_DIR = Path(__file__).resolve().parents[2] / "dace" / "frontend" / "hlfir"
if str(_HLFIR_DIR) not in sys.path:
    sys.path.insert(0, str(_HLFIR_DIR))
if str(_HLFIR_DIR / "build") not in sys.path:
    sys.path.insert(0, str(_HLFIR_DIR / "build"))


@dataclass
class _Node:
    """Minimal stand-in for the nanobind ASTNode."""
    kind: str
    target: str = ""
    expr: str = ""
    target_is_array: bool = False
    loop_iter: str = ""
    loop_lower: int = 1
    loop_bound: str = ""
    condition: str = ""
    callee: str = ""
    call_args: list = field(default_factory=list)
    reduce_src: str = ""
    reduce_wcr: str = ""
    reduce_identity: str = ""
    reduce_axes: list = field(default_factory=list)
    children: list = field(default_factory=list)
    else_children: list = field(default_factory=list)
    accesses: list = field(default_factory=list)


def test_return_block_wired_at_top_level(tmp_path):
    """An SDFG whose AST contains a top-level RETURN emits a
    ``ReturnBlock`` that codegen turns into an early ``return`` from
    the generated C++ entry point.  Calling the resulting SDFG must
    leave its inputs untouched  --  the body has no compute before the
    return."""
    import dace
    from dace import SDFG
    from hlfir_to_sdfg import SDFGBuilder

    builder = SDFGBuilder.__new__(SDFGBuilder)
    builder.variables = []
    builder.arrays = {}
    builder.symbols = {}
    builder.scalars = {}
    builder._id_counter = 0
    sdfg = SDFG("early_ret")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)

    from hlfir_to_sdfg import _Ctx
    ctx = _Ctx(sdfg, builder)

    ast = [_Node(kind="return")]
    builder._emit(ctx, ast, sdfg)
    ctx.flush(builder, sdfg)
    sdfg.validate()

    # Numerical check: the SDFG is a bare top-level RETURN  --  calling it
    # must compile, run, and leave ``a`` element-wise unchanged.
    a_init = np.array([1.5, -2.5, 3.5, 4.5], dtype=np.float64)
    a = a_init.copy()
    sdfg(a=a, n=a.size)
    np.testing.assert_array_equal(a, a_init)


def test_break_block_inside_loop_region(tmp_path):
    """A LoopRegion containing a ConditionalBlock whose true-arm is a
    BreakBlock behaves like an early-exit ``while`` in C++.  The empty
    body never writes ``a``, so a successful e2e run must return the
    array unchanged regardless of where the break fires."""
    import dace
    from dace import SDFG
    from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
    from hlfir_to_sdfg import SDFGBuilder, _Ctx

    builder = SDFGBuilder.__new__(SDFGBuilder)
    builder.variables = []
    builder.arrays = {}
    builder.symbols = {}
    builder.scalars = {}
    builder._id_counter = 0

    sdfg = SDFG("early_break")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)

    # Manually wire the shape upstream detection should produce:
    #   LoopRegion(i = 1..n) {
    #     ConditionalBlock:
    #       branch when "a[i-1] > 100": ControlFlowRegion { BreakBlock }
    #       else:                       ControlFlowRegion { /* body */ }
    #   }
    loop = LoopRegion(label="loop_0",
                      condition_expr="i < n + 1",
                      loop_var="i",
                      initialize_expr="i = 1",
                      update_expr="i = i + 1")
    sdfg.add_node(loop)

    cond_block = ConditionalBlock("if_exit")
    loop.add_node(cond_block, ensure_unique_name=True)

    break_region = ControlFlowRegion("exit_branch", sdfg=sdfg)
    cond_block.add_branch("(a[i - 1] > 100)", break_region)
    builder._emit(_Ctx(sdfg, builder), [_Node(kind="break")], break_region)

    else_region = ControlFlowRegion("body_branch", sdfg=sdfg)
    else_region.add_state("body_noop", is_start_block=True)
    cond_block.add_branch(None, else_region)

    sdfg.validate()

    # Numerical check  --  the loop body has no writes.  Whether the break
    # fires (a[3] > 100 below) or not, ``a`` must come back unchanged.
    a_init = np.array([1.0, 2.0, 3.0, 200.0, 5.0], dtype=np.float64)
    a = a_init.copy()
    sdfg(a=a, n=a.size, i=0)
    np.testing.assert_array_equal(a, a_init)

    # Also exercise the no-break path: every element below threshold so
    # the loop exits naturally on the counter.
    a_no_break = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    a_nb = a_no_break.copy()
    sdfg(a=a_nb, n=a_nb.size, i=0)
    np.testing.assert_array_equal(a_nb, a_no_break)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
