"""Array-section assignment: ``res(a:b) = <scalar>``.

Exercises the Phase-1 section-assignment path in the HLFIR bridge:
``hlfir.assign %scalar to <section-designate>`` is detected by
``asSectionDesignate`` in extract_ast.cpp and lowered into a nested
``kind="loop"`` wrapper + inner ``kind="assign"``.  No new Python
emitters are involved  --  everything reuses the existing ``emit_loop`` /
``emit_tasklet`` dispatch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
subroutine fill_range(res, a, b)
  implicit none
  integer, intent(in)    :: a, b
  integer, intent(inout) :: res(6)
  res(a:b) = 42
end subroutine
"""


def test_section_assign_ast_shape(tmp_path: Path):
    """Confirm the bridge produces a single outer loop with symbolic
    bounds + one inner assign writing the scalar constant  --  i.e. the
    shape `buildSectionScalarAssign` synthesises."""
    b = build_sdfg(_SRC, tmp_path, name="fill_range", pipeline="hlfir-propagate-shapes")
    # Only one top-level AST node.
    assert len(b.ast) == 1
    outer = b.ast[0]
    assert outer.kind == "loop"
    assert outer.loop_iter == "as_0"
    assert outer.loop_lower_expr == "a"
    assert outer.loop_bound == "b"
    # Exactly one child, an indexed scalar assign.
    children = list(outer.children)
    assert len(children) == 1
    inner = children[0]
    assert inner.kind == "assign"
    assert inner.target == "res"
    assert inner.expr == "42"
    assert inner.target_is_array is True


def test_section_assign_sdfg_structure(tmp_path: Path):
    """The built SDFG should carry exactly one LoopRegion driven by
    ``a..b`` and a single tasklet whose code writes the 42 constant."""
    from dace.sdfg.state import LoopRegion
    from dace.sdfg import nodes as nd

    sdfg = build_sdfg(_SRC, tmp_path, name="fill_range", pipeline="hlfir-propagate-shapes").build()
    loops = [n for n in sdfg.nodes() if isinstance(n, LoopRegion)]
    assert len(loops) == 1, f"expected one LoopRegion, got {len(loops)}"
    loop = loops[0]
    # Bounds appear verbatim in the Python expressions  --  symbols `a` and
    # `b` are carried through to the condition / init.
    assert "a" in loop.init_statement.as_string
    assert "b" in loop.loop_condition.as_string

    # Exactly one tasklet inside whose body writes the constant.
    tasklets = []
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, nd.Tasklet):
                tasklets.append(n)
    assert len(tasklets) == 1, f"expected one tasklet, got {len(tasklets)}"
    assert "42" in tasklets[0].code.as_string


@pytest.mark.parametrize("a,b,expected", [
    (2, 5, [0, 42, 42, 42, 42, 0]),
    (1, 6, [42, 42, 42, 42, 42, 42]),
    (3, 3, [0, 0, 42, 0, 0, 0]),
    (1, 1, [42, 0, 0, 0, 0, 0]),
])
def test_section_assign_numerical(tmp_path: Path, a, b, expected):
    """Runtime behaviour must match Fortran semantics for several
    (lower, upper) pairs including full-range, single-element, and
    asymmetric windows."""
    sdfg = build_sdfg(_SRC, tmp_path, name="fill_range", pipeline="hlfir-propagate-shapes").build()
    res = np.zeros(6, dtype=np.int32)
    sdfg(res=res, a=a, b=b)
    assert res.tolist() == expected, \
        f"res({a}:{b}) = 42 -> {res.tolist()}, expected {expected}"
