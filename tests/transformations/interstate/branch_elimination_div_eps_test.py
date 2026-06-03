# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Eps-protected division for the FP-factor branch lowering.

``EliminateBranches.make_division_tasklets_safe_for_unconditional_execution``
rewrites every ``x / y`` Python tasklet to ``x / (y + eps)`` so the
unconditional execution that the FP-factor lowering produces can no longer
trigger NaN/inf on divisor-zero paths. The pinned invariants:

1. ``eps`` is the **smallest positive normal** of the target precision --
   ``numpy.finfo(np.float64).tiny`` for fp64 and
   ``numpy.finfo(np.float32).tiny`` for fp32. An earlier ``tiny * 2``
   safety margin is no longer needed now that the literal is serialised
   with IEEE-roundtrip precision (``repr(float(x))``).
2. ``eps`` selection is driven by the ``precision`` argument so an
   fp32 input does not receive an fp64 eps (which would be smaller than
   ``FLT_MIN`` and could produce a denormal at the C++ boundary).
3. The literal stringification roundtrips through ``ast.parse`` /
   ``ast.unparse`` without precision loss -- the tasklet still divides
   by the exact ``tiny`` value, not a rounded-down neighbour.
"""
import ast

import numpy as np
import pytest

import dace
from dace.transformation.interstate.branch_elimination import BranchElimination


def _make_pass() -> BranchElimination:
    """Instantiate a bare ``BranchElimination`` for direct API testing.

    The transformation normally runs through the SDFG-matching framework
    that populates ``self.conditional`` / ``self.condition_variable``, but
    ``make_division_tasklets_safe_for_unconditional_execution`` is a pure
    state-rewrite helper that only reads its arguments. Set the matcher
    state explicitly so it can be called in isolation.
    """
    p = BranchElimination()
    # ``_processed_tasklets`` is an instance attribute the helper checks
    # before rewriting; reset to an empty set per call so each unit test
    # starts from a clean slate.
    p._processed_tasklets = set()
    p.eps_operator_type_for_log_and_div = "add"
    return p


def _make_div_sdfg(dtype: dace.dtypes.typeclass) -> dace.SDFG:
    """One-state SDFG with a single ``_out = _in1 / _in2`` Python tasklet.

    Returns ``(sdfg, state, tasklet)`` -- the state is the test target for
    :meth:`make_division_tasklets_safe_for_unconditional_execution`.
    """
    sdfg = dace.SDFG(f"div_eps_{dtype.to_string()}_test")
    sdfg.add_array("A", (4, ), dtype)
    sdfg.add_array("B", (4, ), dtype)
    sdfg.add_array("C", (4, ), dtype, transient=False)
    state = sdfg.add_state("main")
    a = state.add_read("A")
    b = state.add_read("B")
    c = state.add_write("C")
    t = state.add_tasklet(name="div",
                          inputs={"_in1", "_in2"},
                          outputs={"_out"},
                          code="_out = _in1 / _in2",
                          language=dace.dtypes.Language.Python)
    state.add_edge(a, None, t, "_in1", dace.Memlet("A[0]"))
    state.add_edge(b, None, t, "_in2", dace.Memlet("B[0]"))
    state.add_edge(t, "_out", c, None, dace.Memlet("C[0]"))
    return sdfg, state, t


def _parse_constant_in_rhs(tasklet_code: str) -> float:
    """Extract the eps constant added to the divisor of ``_out = _in1 / (_in2 + EPS)``.

    Walks the tasklet's RHS AST, finds the ``BinOp(Div, _, BinOp(Add, _in2, Constant))``,
    and returns the constant value as a Python float. Raises ``AssertionError`` if the
    expected shape isn't found.
    """
    tree = ast.parse(tasklet_code, mode="exec")
    # The tasklet code looks like ``_out = _in1 / (_in2 + EPS)``.
    assign = tree.body[0]
    assert isinstance(assign, ast.Assign), tasklet_code
    div = assign.value
    assert isinstance(div, ast.BinOp) and isinstance(div.op, ast.Div), ast.unparse(div)
    add = div.right
    assert isinstance(add, ast.BinOp) and isinstance(add.op, ast.Add), ast.unparse(add)
    const = add.right
    assert isinstance(const, ast.Constant), ast.unparse(const)
    return float(const.value)


def test_fp64_eps_is_finfo_float64_tiny():
    """For an fp64 divisor the eps equals ``numpy.finfo(np.float64).tiny``."""
    sdfg, state, tasklet = _make_div_sdfg(dace.float64)
    _make_pass().make_division_tasklets_safe_for_unconditional_execution(state, dace.float64)
    eps_in_code = _parse_constant_in_rhs(tasklet.code.as_string)
    assert eps_in_code == np.finfo(np.float64).tiny, (
        f"fp64 div eps = {eps_in_code!r}, expected {np.finfo(np.float64).tiny!r}")


def test_fp32_eps_is_finfo_float32_tiny():
    """For an fp32 divisor the eps equals ``numpy.finfo(np.float32).tiny``.

    The literal is serialised as fp64 in the Python tasklet body (Python
    has no fp32 literal); the implicit narrowing back to fp32 at the C++
    boundary preserves the exact ``FLT_MIN`` value because the textual
    representation roundtrips through both fp formats.
    """
    sdfg, state, tasklet = _make_div_sdfg(dace.float32)
    _make_pass().make_division_tasklets_safe_for_unconditional_execution(state, dace.float32)
    eps_in_code = _parse_constant_in_rhs(tasklet.code.as_string)
    fp32_tiny_as_fp64 = float(np.finfo(np.float32).tiny)
    assert eps_in_code == fp32_tiny_as_fp64, (
        f"fp32 div eps = {eps_in_code!r}, expected {fp32_tiny_as_fp64!r}")


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
def test_eps_roundtrips_through_ast_without_precision_loss(dtype):
    """Parsing the rewritten tasklet recovers the original eps value bit-for-bit.

    Regression guard for the earlier ``str(eps)`` formatting that truncated
    the literal at Python's default float precision -- the parsed-back
    value drifted from ``finfo(...).tiny`` by a few ULPs and the divisor
    occasionally produced a denormal result.
    """
    sdfg, state, tasklet = _make_div_sdfg(dtype)
    _make_pass().make_division_tasklets_safe_for_unconditional_execution(state, dtype)
    eps_in_code = _parse_constant_in_rhs(tasklet.code.as_string)
    expected_tiny = float(np.finfo(np.float64 if dtype == dace.float64 else np.float32).tiny)
    # Exact equality -- a single ULP discrepancy means the literal lost
    # precision somewhere between ``finfo().tiny`` and the AST.
    assert eps_in_code == expected_tiny, (
        f"{dtype} eps lost precision: code={eps_in_code!r}, expected={expected_tiny!r}, "
        f"ulps={abs(eps_in_code - expected_tiny) / np.spacing(expected_tiny):.1f}")


def test_eps_does_not_introduce_machine_epsilon_shift():
    """The eps is ``tiny`` (smallest positive normal), NOT ``eps`` (machine epsilon).

    A divisor of ``1.0`` produces ``1 / (1.0 + eps)`` -- if ``eps`` were
    ``machine_eps`` (~1e-16 for fp64), the quotient would shift by one ULP
    on every divide-by-one, breaking the value-equivalence guarantee the
    FP-factor unconditional execution relies on for the false-arm case
    (else-arm sets the result to ``0.0``; ``c * 1.0 + (1 - c) * 0.0`` with
    ``c == 0`` must give exactly ``0.0``). Pin the eps magnitude below
    machine eps so the quotient stays bit-equal to the un-eps'd division
    for any divisor at least 1 ULP from zero.
    """
    sdfg, state, tasklet = _make_div_sdfg(dace.float64)
    _make_pass().make_division_tasklets_safe_for_unconditional_execution(state, dace.float64)
    eps_in_code = _parse_constant_in_rhs(tasklet.code.as_string)
    assert eps_in_code < np.finfo(np.float64).eps, (
        f"eps {eps_in_code!r} >= machine eps {np.finfo(np.float64).eps!r}; "
        f"divide-by-1 would shift the quotient by a measurable amount")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
