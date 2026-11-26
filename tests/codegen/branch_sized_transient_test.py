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
from dace.transformation.passes.split_branch_transients import SplitBranchTransients

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


def allocate_fill_copy(region, label: str, condition: str, size: str, offset: int, target: str, read_first: bool):
    """An ``IF (condition)`` after ``region``'s sink that sizes ``buf`` to ``size``, fills it with ``i + offset`` and
    copies it to ``target``; with ``read_first`` the copy comes first, reading the buffer as it was."""
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
    first, last = (copy, fill) if read_first else (fill, copy)
    arm.add_edge(start, first, InterstateEdge(assignments={} if read_first else {"d0": size}))
    arm.add_edge(first, last, InterstateEdge(assignments={"d0": size} if read_first else {}))
    cond.add_branch(CodeBlock(condition), arm)
    region.add_state_after(cond, f"after_{label}")


def build_in_sequence(read_first: bool) -> dace.SDFG:
    """QE p_psi's ``coeff``: ``IF (f0 > 0)`` allocates ``buf(N)``; then ``IF (f1 > 0)`` holds two ``IF``s in a row,
    each allocating it with a size of its own, filling it and copying it out before its ``deallocate``; with
    ``read_first`` the first of them copies out the values the first ``IF`` left before it allocates and fills."""
    sdfg = dace.SDFG(f"sized_in_sequence_{int(read_first)}")
    for target in ("o0", "o1", "o2"):
        sdfg.add_array(target, [N], dace.float64)
    sdfg.add_scalar("f0", dace.int32)
    sdfg.add_scalar("f1", dace.int32)
    sdfg.add_symbol("d0", dace.int64)
    sdfg.add_transient("buf", ["d0"], dace.float64)
    sdfg.add_state("init", is_start_block=True)
    allocate_fill_copy(sdfg, "a", "f0 > 0", "N", 0, "o0", False)
    outer = ConditionalBlock("if_b")
    sink = next(iter(sdfg.sink_nodes()))
    sdfg.add_node(outer)
    sdfg.add_edge(sink, outer, InterstateEdge())
    arm = ControlFlowRegion("arm_b", sdfg)
    arm.add_state("start_b", is_start_block=True)
    allocate_fill_copy(arm, "b0", "f1 > 0", "N + 2", 10, "o1", read_first)
    allocate_fill_copy(arm, "b1", "f1 > 0", "N + 1", 20, "o2", False)
    outer.add_branch(CodeBlock("f1 > 0"), arm)
    sdfg.add_state_after(outer, "done")
    return sdfg


# Reproducer: QE h_psi failed in codegen: "Array (coeff_d0, coeff_d1) is sized at 3 places: it would need a
# reallocation" -- p_psi allocates coeff in one IF, then twice in a row in the next
def test_allocations_in_sequence_take_a_container_each():
    sdfg = build_in_sequence(read_first=False)
    o0, o1, o2 = np.zeros(5), np.zeros(5), np.zeros(5)
    sdfg(o0=o0, o1=o1, o2=o2, f0=np.int32(1), f1=np.int32(1), N=5)
    assert np.allclose(o0, np.arange(5)) and np.allclose(o1, np.arange(5) + 10) and np.allclose(o2, np.arange(5) + 20)


def test_a_buffer_read_before_its_allocation_keeps_one_container():
    """The second ``IF`` reads the first one's values before it allocates the buffer again: a copy per ``IF`` would
    lose them."""
    sdfg = build_in_sequence(read_first=True)
    SplitBranchTransients().apply_pass(sdfg, {})
    assert [name for name in sdfg.arrays if name.startswith("buf")] == ["buf"]


# Reproducer: a buffer sized N + 2, read, then filled to N + 3 was allocated once at N + 2 and overflowed: the size was
# assigned at one place in the span the allocation searched, which the refusal took for a single size
def test_a_buffer_resized_after_a_read_refuses():
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
    with pytest.raises(NotImplementedError, match="would need a reallocation"):
        sdfg.generate_code()


if __name__ == "__main__":
    test_a_transient_sized_in_each_branch_is_allocated_per_branch()
    test_a_transient_each_arm_sizes_and_later_code_reads_is_allocated_in_each_arm()
    test_a_conditional_without_else_still_refuses_to_size_in_its_arms()
    test_allocations_in_sequence_take_a_container_each()
    test_a_buffer_read_before_its_allocation_keeps_one_container()
    test_a_buffer_resized_after_a_read_refuses()
