# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient each branch of a conditional sizes and uses on its own is allocated per branch.

Fortran allocates a local array in each arm of an ``IF`` with the arm's own size (QE's ``fwfft_orbital_gamma``:
``psio(ngk(1), incr)`` in one arm, ``psio(ngk(1), brange)`` in the other). One container sized by a symbol both arms
assign has no single place to be allocated; a copy per arm has.
"""

import numpy as np
import pytest

import dace
from dace import InterstateEdge
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

N = dace.symbol("N")


def build() -> dace.SDFG:
    sdfg = dace.SDFG("sized_in_two_arms")
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_scalar("flag", dace.int32)
    sdfg.add_symbol("d0", dace.int64)
    sdfg.add_transient("buf", ["d0"], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    cond = ConditionalBlock("pick")
    sdfg.add_node(cond)
    for k, (test, size) in enumerate((("flag != 0", "N"), ("flag == 0", "N + 1"))):
        arm = ControlFlowRegion(f"arm{k}", sdfg)
        a = arm.add_state(f"a{k}", is_start_block=True)
        b = arm.add_state(f"b{k}")
        arm.add_edge(a, b, InterstateEdge(assignments={"d0": size}))
        b.add_mapped_tasklet(
            f"fill{k}", {"i": "0:d0"}, {}, f"o = i + {k}", {"o": dace.Memlet("buf[i]")}, external_edges=True
        )
        c = arm.add_state_after(b, f"c{k}")
        c.add_mapped_tasklet(
            f"cp{k}",
            {"i": "0:N"},
            {"x": dace.Memlet("buf[i]")},
            "o = x",
            {"o": dace.Memlet("out[i]")},
            external_edges=True,
        )
        cond.add_branch(CodeBlock(test), arm)
    sdfg.add_edge(init, cond, InterstateEdge())
    sdfg.add_state_after(cond, "done")
    return sdfg


# Reproducer: QE h_psi failed in codegen: "fwfft_orbital_gamma_psio is sized at 2 places: it would need a reallocation"
def test_a_transient_sized_in_each_branch_is_allocated_per_branch():
    sdfg = build()
    for flag, offset in ((1, 0.0), (0, 1.0)):
        out = np.zeros(5)
        sdfg(out=out, flag=np.int32(flag), N=5)
        assert np.allclose(out, np.arange(5) + offset), flag


def build_read_after(with_else: bool) -> dace.SDFG:
    """``if flag > 0: allocate(buf(N)) else [if flag < 0]: allocate(buf(N + 1))``, each arm filling it, then a read after the ``if``."""
    sdfg = dace.SDFG(f"sized_in_arms_read_after_{int(with_else)}")
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_scalar("flag", dace.int32)
    sdfg.add_symbol("d0", dace.int64)
    sdfg.add_transient("buf", ["d0"], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    cond = ConditionalBlock("pick")
    sdfg.add_node(cond)
    arms = (("flag > 0", "N"), (None if with_else else "flag < 0", "N + 1"))
    # Over the arms: each sizes the buffer, then fills it
    for k, (test, size) in enumerate(arms):
        arm = ControlFlowRegion(f"arm{k}", sdfg)
        a = arm.add_state(f"a{k}", is_start_block=True)
        b = arm.add_state(f"b{k}")
        arm.add_edge(a, b, InterstateEdge(assignments={"d0": size}))
        b.add_mapped_tasklet(
            f"fill{k}", {"i": "0:d0"}, {}, f"o = i + {k}", {"o": dace.Memlet("buf[i]")}, external_edges=True
        )
        cond.add_branch(None if test is None else CodeBlock(test), arm)
    # dace-fortran zeroes an allocatable's extents on entry, so the size is never an input
    sdfg.add_edge(init, cond, InterstateEdge(assignments={"d0": "0"}))
    done = sdfg.add_state_after(cond, "done")
    done.add_mapped_tasklet(
        "cp", {"i": "0:N"}, {"x": dace.Memlet("buf[i]")}, "o = x", {"o": dace.Memlet("out[i]")}, external_edges=True
    )
    return sdfg


# Reproducer: QE h_psi failed in codegen: "vloc_psi_tg_nc_tg_v is sized at 2 places: it would need a reallocation";
# each arm of an IF allocates it with its own extent, and code after the IF reads it
def test_a_transient_each_arm_sizes_and_later_code_reads_is_allocated_in_each_arm():
    sdfg = build_read_after(with_else=True)
    for flag, offset in ((1, 0.0), (0, 1.0)):
        out = np.zeros(5)
        sdfg(out=out, flag=np.int32(flag), N=5)
        assert np.allclose(out, np.arange(5) + offset), flag


def test_a_conditional_without_else_still_refuses_to_size_in_its_arms():
    """``flag == 0`` skips both arms and would read an unallocated buffer: no allocation per arm covers it."""
    with pytest.raises(NotImplementedError, match="would need a reallocation"):
        build_read_after(with_else=False).generate_code()


def allocate_fill_copy(region, label: str, condition: str, size: str, offset: int, target: str) -> None:
    """An ``IF (condition)`` after ``region``'s sink that sizes ``buf`` to ``size``, fills it with ``i + offset`` and
    copies it to ``target``."""
    sink = next(iter(region.sink_nodes()))
    cond = ConditionalBlock(f"if_{label}")
    region.add_node(cond)
    region.add_edge(sink, cond, InterstateEdge())
    arm = ControlFlowRegion(f"arm_{label}", region.sdfg)
    start = arm.add_state(f"start_{label}", is_start_block=True)
    fill = arm.add_state(f"fill_{label}")
    fill.add_mapped_tasklet(
        f"fill_{label}", {"i": "0:d0"}, {}, f"o = i + {offset}", {"o": dace.Memlet("buf[i]")}, external_edges=True
    )
    copy = arm.add_state(f"cp_{label}")
    copy.add_mapped_tasklet(
        f"cp_{label}",
        {"i": "0:N"},
        {"x": dace.Memlet("buf[i]")},
        "o = x",
        {"o": dace.Memlet(f"{target}[i]")},
        external_edges=True,
    )
    arm.add_edge(start, fill, InterstateEdge(assignments={"d0": size}))
    arm.add_edge(fill, copy, InterstateEdge())
    cond.add_branch(CodeBlock(condition), arm)
    region.add_state_after(cond, f"after_{label}")


# Reproducer: QE vexx_bp_k_gpu (USPP) aborted with "malloc_consolidate(): invalid chunk size": a buffer sized before a
# read and resized after it was allocated once at the first size and overflowed. The second fill writes every value
# the second copy reads, so the resize reallocates.
def test_a_buffer_resized_after_a_read_is_reallocated():
    sdfg = dace.SDFG("grown_after_a_read")
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_symbol("d0", dace.int64)
    sdfg.add_transient("buf", ["d0"], dace.float64)
    previous = sdfg.add_state("init", is_start_block=True)
    # Over the two sizes: fill the buffer to its size, then copy it out
    for k, size in enumerate(("N + 2", "N + 3")):
        fill = sdfg.add_state(f"fill{k}")
        sdfg.add_edge(previous, fill, InterstateEdge(assignments={"d0": size}))
        fill.add_mapped_tasklet(
            f"fill{k}", {"i": "0:d0"}, {}, f"o = i + {k}", {"o": dace.Memlet("buf[i]")}, external_edges=True
        )
        previous = sdfg.add_state_after(fill, f"cp{k}")
        previous.add_mapped_tasklet(
            f"cp{k}",
            {"i": "0:N"},
            {"x": dace.Memlet("buf[i]")},
            "o = x",
            {"o": dace.Memlet("out[i]")},
            external_edges=True,
        )
    code = sdfg.generate_code()[0].clean_code
    assert code.count("aligned_delete_array(buf") == 2, code
    out = np.zeros(5)
    sdfg(out=out, N=5)
    assert np.allclose(out, np.arange(5) + 1)


def build_resized_in_an_if() -> dace.SDFG:
    """``buf(N)`` is filled and copied to ``o0``; ``IF (flag)`` reallocates it to ``N + 3``, fills and copies it to
    ``o1``; after the ``IF`` a zero fill uses the size in force."""
    sdfg = dace.SDFG("resized_in_an_if")
    for target in ("o0", "o1"):
        sdfg.add_array(target, [N], dace.float64)
    sdfg.add_scalar("flag", dace.int32)
    sdfg.add_symbol("d0", dace.int64)
    sdfg.add_transient("buf", ["d0"], dace.float64)
    init = sdfg.add_state("init", is_start_block=True)
    fill = sdfg.add_state("fill")
    sdfg.add_edge(init, fill, InterstateEdge(assignments={"d0": "N"}))
    fill.add_mapped_tasklet("fill", {"i": "0:d0"}, {}, "o = i", {"o": dace.Memlet("buf[i]")}, external_edges=True)
    copy = sdfg.add_state_after(fill, "cp")
    copy.add_mapped_tasklet(
        "cp", {"i": "0:N"}, {"x": dace.Memlet("buf[i]")}, "o = x", {"o": dace.Memlet("o0[i]")}, external_edges=True
    )
    allocate_fill_copy(sdfg, "r", "flag != 0", "N + 3", 10, "o1")
    tail = next(iter(sdfg.sink_nodes()))
    tail.add_mapped_tasklet("zero_tail", {"i": "0:d0"}, {}, "o = 0", {"o": dace.Memlet("buf[i]")}, external_edges=True)
    return sdfg


# Reproducer: the same overflow when the resize sits in an IF without else between two uses (QE rmexx_d)
def test_a_buffer_resized_in_an_if_without_else_is_reallocated():
    sdfg = build_resized_in_an_if()
    code = sdfg.generate_code()[0].clean_code
    assert code.count("aligned_delete_array(buf") == 2, code
    for flag in (0, 1):
        o0, o1 = np.zeros(5), np.zeros(5)
        sdfg(o0=o0, o1=o1, flag=np.int32(flag), N=5)
        assert np.allclose(o0, np.arange(5)) and np.allclose(o1, (np.arange(5) + 10) * flag), flag


if __name__ == "__main__":
    test_a_transient_sized_in_each_branch_is_allocated_per_branch()
    test_a_transient_each_arm_sizes_and_later_code_reads_is_allocated_in_each_arm()
    test_a_conditional_without_else_still_refuses_to_size_in_its_arms()
    test_a_buffer_resized_after_a_read_is_reallocated()
    test_a_buffer_resized_in_an_if_without_else_is_reallocated()
