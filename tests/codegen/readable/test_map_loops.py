# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Map loops from the shared cpu.py emitter: the counter keeps its map parameter's type and is declared in the
for-init clause, and strided maps cover exactly their iteration space under both schedules."""

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
    for i in dace.map[0:N:2]:
        B[i] = A[i] * 2.0


def loop_lines(code):
    return [line.strip() for line in code.splitlines() if line.strip().startswith("for (")]


def generate(sdfg, implementation):
    with set_temporary("compiler", "cpu", "implementation", value=implementation):
        return "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")


def map_sdfg(name, outputs=("B",), schedule=dace.dtypes.ScheduleType.Sequential, rng="0:N"):
    """One map per output, all with the parameter ``i``, so sibling counters share a name."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [N], dace.float64)
    state = sdfg.add_state()
    for k, out in enumerate(outputs):
        sdfg.add_array(out, [N], dace.float64)
        entry, exit_node = state.add_map("m%d" % k, dict(i=rng), schedule=schedule)
        tasklet = state.add_tasklet("t%d" % k, {"a"}, {"b"}, "b = a * 2.0")
        state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
        state.add_memlet_path(tasklet, exit_node, state.add_write(out), src_conn="b", memlet=dace.Memlet(f"{out}[i]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("implementation", ["legacy", "experimental_readable"])
def test_counter_has_the_parameter_type_in_the_for_init(implementation):
    lines = loop_lines(generate(double_it.to_sdfg(simplify=True), implementation))
    assert any(line.startswith("for (int i = 0; i < ") for line in lines), lines


@pytest.mark.parametrize("implementation", ["legacy", "experimental_readable"])
def test_sibling_sequential_maps_share_a_counter_name(implementation):
    sdfg = map_sdfg(f"seq_siblings_{implementation}", outputs=("B", "C"))
    with set_temporary("compiler", "cpu", "implementation", value=implementation):
        A = numpy.random.default_rng(0).random(24)
        B, C = numpy.zeros(24), numpy.zeros(24)
        sdfg(A=A, B=B, C=C, N=24)
    assert numpy.allclose(B, A * 2.0) and numpy.allclose(C, A * 2.0)


@pytest.mark.parametrize(
    "schedule", [dace.dtypes.ScheduleType.Sequential, dace.dtypes.ScheduleType.CPU_Multicore], ids=str
)
@pytest.mark.parametrize("n", [31, 32])
def test_strided_map_covers_its_iteration_space(schedule, n):
    """The stride divides the range for one extent and not the other."""
    sdfg = map_sdfg(f"strided_{schedule.name}_{n}", schedule=schedule, rng="0:N:2")
    A = numpy.random.default_rng(0).random(n)
    B = numpy.zeros(n)
    sdfg(A=A, B=B, N=n)
    assert numpy.allclose(B[::2], A[::2] * 2.0) and not B[1::2].any()


def test_frontend_strided_map_runs():
    A = numpy.random.default_rng(0).random(31)
    B = numpy.zeros(31)
    strided(A=A, B=B, N=31)
    assert numpy.allclose(B[::2], A[::2] * 2.0)


if __name__ == "__main__":
    for impl in ("legacy", "experimental_readable"):
        test_counter_has_the_parameter_type_in_the_for_init(impl)
        test_sibling_sequential_maps_share_a_counter_name(impl)
    for sched in (dace.dtypes.ScheduleType.Sequential, dace.dtypes.ScheduleType.CPU_Multicore):
        for size in (31, 32):
            test_strided_map_covers_its_iteration_space(sched, size)
    test_frontend_strided_map_runs()
