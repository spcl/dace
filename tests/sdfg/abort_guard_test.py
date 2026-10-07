# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Precondition guards are Python tasklets ``if cond: abort()``; C++ prints ``std::abort()``."""

import subprocess
import sys

import numpy as np

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


def test_cpp_spells_abort_from_the_standard_library():
    assert cppunparse.py2cpp("if N > 4:\n    abort()") == "if ((N > 4)) {\n    std::abort();\n}"
    code = guarded_sdfg("abort_guard_codegen").generate_code()[0].clean_code
    assert "std::abort();" in code


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
