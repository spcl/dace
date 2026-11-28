# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Precondition guards are Python tasklets ``if cond: abort()``; C++ prints ``DACE_ABORT()``, which traps in a kernel."""

import subprocess
import sys

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.codegen import cppunparse
from dace.sdfg import tasklet_utils as tutil

N = dace.symbol("N", dtype=dace.int64)


def guarded_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    guard_state = sdfg.add_state("guard", is_start_block=True)
    tutil.add_abort_guard(guard_state, "check_n", "N > 4")
    state = sdfg.add_state_after(guard_state, "body")
    state.add_mapped_tasklet("m", {"i": "0:N"}, {}, "y = 1.0", {"y": dace.Memlet("a[i]")}, external_edges=True)
    return sdfg


def test_guard_is_a_side_effecting_python_tasklet():
    sdfg = guarded_sdfg("abort_guard_shape")
    (guard,) = [n for n in sdfg.start_block.nodes() if tutil.is_abort_guard(n)]
    assert guard.language == dace.dtypes.Language.Python
    assert guard.side_effects
    assert guard.code.as_string == tutil.abort_guard_code("N > 4") == "if (N > 4):\n    abort()"


def test_cpp_spells_abort_as_the_runtime_macro():
    assert cppunparse.py2cpp("if N > 4:\n    abort()") == "if ((N > 4)) {\n    DACE_ABORT();\n}"
    code = guarded_sdfg("abort_guard_codegen").generate_code()[0].clean_code
    assert "DACE_ABORT();" in code


@pytest.mark.gpu
def test_guard_inside_a_kernel_compiles_and_passes():
    # A guard the offloading places inside a kernel: ``std::abort`` is host-only there, the macro traps instead.
    sdfg = dace.SDFG("abort_guard_kernel")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "m",
        {"i": "0:N"},
        {"x": dace.Memlet("a[i]")},
        "if x < 0:\n    abort()\ny = x + 1",
        {"y": dace.Memlet("b[i]")},
        external_edges=True,
    )
    sdfg.apply_gpu_transformations()
    a = np.arange(5, dtype=np.float64)
    b = np.zeros(5)
    sdfg(a=a, b=b, N=5)
    assert np.allclose(b, a + 1)


def test_abort_is_an_opaque_symbolic_call():
    expr = symbolic.pystr_to_symbolic("abort()")
    assert isinstance(expr, symbolic.abort)
    assert symbolic.symstr(expr) == "(abort())"
    assert "abort" in symbolic.builtin_userfunctions()


def test_guard_passes_when_the_condition_is_false():
    a = np.zeros(3)
    guarded_sdfg("abort_guard_pass")(a=a, N=3)
    assert np.all(a == 1.0)


def test_guard_aborts_when_the_condition_holds(tmp_path):
    # SIGABRT would take the test runner with it, so the violating call runs in a child.
    sdfg = guarded_sdfg("abort_guard_fire")
    path = tmp_path / "guard.sdfg"
    sdfg.save(str(path))
    script = f"import dace, numpy as np\ndace.SDFG.from_file({str(path)!r})(a=np.zeros(8), N=8)\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == -6, result.stderr[-2000:]
