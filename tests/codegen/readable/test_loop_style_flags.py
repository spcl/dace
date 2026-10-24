# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the shared map-loop emitter and its ``compiler.cpu.codegen_params.loop_index_type`` knob. The emitter
lives in cpu.py, so it drives the legacy generator too -- the default must reproduce today's loop verbatim."""

import numpy
import pytest

import dace
from dace.config import set_temporary

N = dace.symbol("N")


@dace.program
def double_it(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] * 2.0


@dace.program
def strided(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N:2]:  # non-unit stride: a naive `!=` bound is stepped over (see ne test)
        B[i] = A[i] * 2.0


def generate(program, implementation="legacy", loop_index_type="auto"):
    sdfg = program.to_sdfg(simplify=True)
    with (
        set_temporary("compiler", "cpu", "implementation", value=implementation),
        set_temporary("compiler", "cpu", "codegen_params", "loop_index_type", value=loop_index_type),
    ):
        return "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")


def loop_lines(code):
    return [line.strip() for line in code.splitlines() if line.strip().startswith("for (")]


@pytest.mark.parametrize("implementation", ["legacy", "experimental_readable"])
def test_defaults_emit_the_historical_loop(implementation):
    """The default spelling must be byte-identical to the pre-knob emitter: `for (auto i = ...; i <
    end + 1; i += ...)`. This is what keeps legacy unaffected."""
    lines = loop_lines(generate(double_it, implementation))
    assert lines, "no loop emitted"
    assert any(line.startswith("for (auto ") and " < " in line for line in lines)
    assert not any("<=" in line or "!=" in line for line in lines)


@pytest.mark.parametrize(
    "loop_index_type, expected", [("auto", "for (auto "), ("int64", "for (int64_t "), ("int32", "for (int32_t ")]
)
def test_loop_index_type(loop_index_type, expected):
    lines = loop_lines(generate(double_it, loop_index_type=loop_index_type))
    assert any(line.startswith(expected) for line in lines), lines


@pytest.mark.parametrize("loop_index_type", ["auto", "int64", "int32"])
def test_every_index_type_runs_correctly(loop_index_type):
    with set_temporary("compiler", "cpu", "codegen_params", "loop_index_type", value=loop_index_type):
        A = numpy.random.default_rng(0).random(32)
        B = numpy.zeros(32)
        double_it(A=A, B=B, N=32)
        assert numpy.allclose(B, A * 2.0)


def sequential_map_sdfg(name="seq_map", nmaps=1):
    """Sequential maps (all using the parameter name `i` so sibling scoping is exercised). With
    ``simd_maps`` on, the innermost loop may be preceded by ``#pragma omp simd``; tests
    that need a genuinely non-OpenMP sequential map disable the flag explicitly."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [N], dace.float64)
    outs = ["B", "C"][:nmaps]
    for o in outs:
        sdfg.add_array(o, [N], dace.float64)
    state = sdfg.add_state()
    for k, o in enumerate(outs):
        entry, exit_node = state.add_map("m%d" % k, dict(i="0:N"), schedule=dace.dtypes.ScheduleType.Sequential)
        tasklet = state.add_tasklet("t%d" % k, {"a"}, {"b"}, "b = a * 2.0")
        state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(tasklet, exit_node, state.add_write(o), src_conn="b", memlet=dace.Memlet("%s[i]" % o))
    sdfg.validate()
    return sdfg


def generate_sdfg(sdfg, implementation="legacy"):
    with set_temporary("compiler", "cpu", "implementation", value=implementation):
        return "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")


@pytest.mark.parametrize("implementation", ["legacy", "experimental_readable"])
def test_a_sequential_map_declares_its_counter_in_the_for_init(implementation):
    lines = loop_lines(generate_sdfg(sequential_map_sdfg(), implementation))
    assert any(line.startswith("for (int i = ") for line in lines), lines
    assert not any(line.startswith("for (;") for line in lines), lines


def openmp_strided_sdfg(name="omp_strided"):
    """A CPU_Multicore map with a non-unit stride. The frontend's strided maps get a non-OMP schedule, so this
    is built directly to force the OpenMP path."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("m", dict(i="0:N:2"), schedule=dace.dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(tasklet, exit_node, state.add_write("B"), src_conn="b", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_openmp_strided_map_compiles_and_runs():
    """A non-unit stride under ``#pragma omp parallel for`` keeps the canonical ``<`` exit test."""
    sdfg = openmp_strided_sdfg()
    code = "\n".join(o.code for o in sdfg.generate_code() if o.language == "cpp")
    assert any(line.startswith("for (") and "i < " in line and "i += 2" in line for line in loop_lines(code))
    A = numpy.random.default_rng(0).random(30)
    B = numpy.zeros(30)
    sdfg(A=A, B=B, N=30)
    assert numpy.allclose(B[::2], A[::2] * 2.0)


@pytest.mark.parametrize("n", [31, 32])
def test_strided_map_covers_odd_and_even_extents(n):
    """The stride divides the range in one case and not the other; both stop at the right element."""
    A = numpy.random.default_rng(0).random(n)
    B = numpy.zeros(n)
    strided(A=A, B=B, N=n)
    assert numpy.allclose(B[::2], A[::2] * 2.0)


if __name__ == "__main__":
    test_defaults_emit_the_historical_loop("legacy")
    test_strided_map_covers_odd_and_even_extents(31)
