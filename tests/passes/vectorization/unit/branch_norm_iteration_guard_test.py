# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``BranchNormalization`` must REFUSE a guard on the iteration symbol.

``BranchNormalization`` is the SECOND if-conversion site (it runs right after
``SameWriteSetIfElseToITECFG``). It flattens ``if cond: arr[s] = f(...)`` into
straight-line dataflow, gating the ESCAPING write through ``arr[s] = ITE(cond, f, arr[s])``
while the READS in ``f`` become UNCONDITIONAL -- every lane evaluates them. That is sound
for a DATA condition, but UNSOUND when the guard constrains the iteration symbol itself
(``if i < N - 1: b[i] = a[i+1]``): the guard is exactly what keeps ``a[i+1]`` in range, so
making the read unconditional fabricates the out-of-range read of ``a[N]`` on lane
``i = N-1``. Such a guard needs MASKING, not if-conversion; the ``ConditionalBlock`` must
survive for the masking path. ``SameWriteSetIfElseToITECFG`` already refuses it; this pins
that ``BranchNormalization`` shares the refusal (reusing ``condition_guards_iteration_symbol``
across both its lowering call sites: ``_normalize_single_arm`` and ``_split_two_arm_disjoint``).

WHY THE CONTRACT IS STRUCTURAL (``ConditionalBlock`` survives), not a wrong OUTPUT value:
the ITE if-conversion is VALUE-PRESERVING by construction. On the excluded lane
``ITE(i < N-1, a[N], b[N-1])`` selects the ELSE operand ``b[N-1]`` (its original value), so
``b`` is computed correctly whether or not the guard is flattened -- the only defect is the
spurious read of ``a[N]``. Measured here: with ``a`` allocated one element past ``N`` (so the
read lands on a sentinel rather than faulting) the flattened kernel still returns the correct
``b``; and a genuinely one-past read of a small heap buffer does not reliably segfault either.
So no wrong output value and no reliable crash can distinguish the bug -- only the surviving
``ConditionalBlock`` can. Each test therefore asserts the refusal AND runs the refused SDFG
end-to-end (forked, bit-exact vs numpy) to prove the left-in-place conditional still compiles
and computes correctly.
"""
import os

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG

N = dace.symbol("N", nonnegative=True)


def conditional_blocks(sdfg: dace.SDFG):
    """Every :class:`ConditionalBlock` anywhere in ``sdfg`` (recurses regions / NSDFGs)."""
    return [b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)]


def add_copy_arm(sdfg: dace.SDFG, cb: ConditionalBlock, cond, read_subset: str, write_arr: str,
                 write_subset: str) -> ControlFlowRegion:
    """Attach one arm ``[if <cond>]: <write_arr>[<write_subset>] = a[<read_subset>]`` to ``cb``.

    :param sdfg: SDFG owning the arm's region.
    :param cb: conditional block the arm is added to.
    :param cond: the arm's guard source, or ``None`` for the bare ``else`` arm.
    :param read_subset: the subset ``a`` is read at inside the arm.
    :param write_arr: the array written by the arm.
    :param write_subset: the subset ``write_arr`` is written at.
    :returns: the arm's body region.
    """
    body = ControlFlowRegion(f"arm_{write_arr}", sdfg=sdfg)
    st = body.add_state("s", is_start_block=True)
    ra = st.add_access("a")
    wa = st.add_access(write_arr)
    tl = st.add_tasklet("cp", {"_a"}, {"_b"}, "_b = _a")
    st.add_edge(ra, None, tl, "_a", dace.Memlet(f"a[{read_subset}]"))
    st.add_edge(tl, "_b", wa, None, dace.Memlet(f"{write_arr}[{write_subset}]"))
    cb.add_branch(CodeBlock(cond) if cond is not None else None, body)
    return body


def build_loop_with_cb(extra_arrays=()) -> tuple:
    """``for i in range(N): <ConditionalBlock>`` scaffold with ``a`` (len ``N+1``) and ``b``.

    ``a`` is one element longer than the iteration space so the fabricated ``a[N]`` read lands
    on an allocated sentinel instead of faulting -- see the module docstring on why a crash is
    not a reliable signal.

    :param extra_arrays: extra output array names to declare with shape ``(N,)``.
    :returns: ``(sdfg, cb)``.
    """
    sdfg = dace.SDFG("branch_norm_iter_guard")
    sdfg.add_array("a", shape=(N + 1, ), dtype=dace.float64)
    sdfg.add_array("b", shape=(N, ), dtype=dace.float64)
    for name in extra_arrays:
        sdfg.add_array(name, shape=(N, ), dtype=dace.float64)
    loop = LoopRegion("loop", loop_var="i", initialize_expr="i = 0", condition_expr="i < N", update_expr="i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    head = loop.add_state("head", is_start_block=True)
    cb = ConditionalBlock("cb", sdfg=sdfg, parent=loop)
    loop.add_node(cb)
    loop.add_edge(head, cb, dace.InterstateEdge())
    return sdfg, cb


def fork_run_bit_exact(sdfg: dace.SDFG, kwargs: dict, outputs) -> None:
    """Compile + run ``sdfg`` in a forked child; assert every output is bit-exact vs its ref.

    Forked so a would-be out-of-bounds access cannot take the test process down, and because
    the child mutates the output buffers in its own (copy-on-write) address space -- the
    comparison therefore happens IN the child, which signals the verdict through its exit code.

    :param sdfg: SDFG to compile and run.
    :param kwargs: call arguments (arrays + symbols); output buffers are mutated in place.
    :param outputs: sequence of ``(buffer, reference_ndarray)`` compared bit-exact after the run.
    """
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            sdfg.compile()(**kwargs)
            code = 0 if all(np.array_equal(buf, ref) for buf, ref in outputs) else 3
        except BaseException:  # noqa: BLE001 -- child reports any failure via exit code
            code = 4
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert not os.WIFSIGNALED(status), f"kernel crashed (signal {os.WTERMSIG(status)})"
    assert os.WEXITSTATUS(status) == 0, f"kernel mismatch or error (child exit {os.WEXITSTATUS(status)})"


def test_branch_normalization_refuses_direct_iteration_guard():
    """Single-arm ``if i < N-1: b[i] = a[i+1]`` -- ``_normalize_single_arm`` must refuse it.

    Pre-fix ``BranchNormalization`` flattened it (returned ``1``, ``ConditionalBlock`` gone),
    leaving lane ``i = N-1`` reading ``a[N]``. The refused SDFG still runs bit-exact.
    """
    sdfg, cb = build_loop_with_cb()
    add_copy_arm(sdfg, cb, "i < N - 1", "i + 1", "b", "i")
    assert len(conditional_blocks(sdfg)) == 1

    assert SameWriteSetIfElseToITECFG().apply_pass(sdfg, {}) is None
    assert len(conditional_blocks(sdfg)) == 1

    rewritten = BranchNormalization().apply_pass(sdfg, {})
    assert rewritten is None, "BranchNormalization flattened the iteration guard -> a[N] read on lane i=N-1"
    assert len(conditional_blocks(sdfg)) == 1

    nval = 8
    a = np.arange(100, 100 + nval + 1, dtype=np.float64)
    b = np.full(nval, 999.0, dtype=np.float64)
    ref = b.copy()
    for i in range(nval):
        if i < nval - 1:
            ref[i] = a[i + 1]
    fork_run_bit_exact(sdfg, dict(a=a.copy(), b=b, N=nval), [(b, ref)])


def test_branch_normalization_refuses_two_arm_iteration_guard():
    """Two-arm disjoint ``if i < N-1: b[i]=a[i+1] else: c[i]=a[i]`` -- ``_split_two_arm_disjoint``
    must refuse it.

    Pre-fix ``BranchNormalization`` split the disjoint arms into two single-arm blocks and
    flattened each (``ConditionalBlock`` gone), again fabricating the ``a[N]`` read in the
    if-arm. The refused SDFG still runs bit-exact.
    """
    sdfg, cb = build_loop_with_cb(extra_arrays=("c", ))
    add_copy_arm(sdfg, cb, "i < N - 1", "i + 1", "b", "i")  # if-arm (has the OOB read)
    add_copy_arm(sdfg, cb, None, "i", "c", "i")             # bare else-arm (disjoint write)
    assert len(conditional_blocks(sdfg)) == 1

    # SameWriteSet only handles SAME-element two-arm writes; disjoint b vs c is not its case.
    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    assert len(conditional_blocks(sdfg)) == 1

    rewritten = BranchNormalization().apply_pass(sdfg, {})
    assert rewritten is None, "BranchNormalization split+flattened a two-arm iteration guard"
    assert len(conditional_blocks(sdfg)) == 1

    nval = 8
    a = np.arange(100, 100 + nval + 1, dtype=np.float64)
    b = np.full(nval, 999.0, dtype=np.float64)
    c = np.full(nval, 888.0, dtype=np.float64)
    ref_b, ref_c = b.copy(), c.copy()
    for i in range(nval):
        if i < nval - 1:
            ref_b[i] = a[i + 1]
        else:
            ref_c[i] = a[i]
    fork_run_bit_exact(sdfg, dict(a=a.copy(), b=b, c=c, N=nval), [(b, ref_b), (c, ref_c)])


def test_branch_normalization_refuses_transitive_iteration_guard():
    """Transitive guard: ``ip1 = i + 1`` on the edge into the block, guard ``ip1 < N``.

    The symbol chain resolves to the loop variable, so ``_normalize_single_arm`` must still
    refuse -- ``a[ip1]`` with ``ip1 = N`` on the last lane is the same out-of-range read.
    """
    sdfg, cb = build_loop_with_cb()
    sdfg.add_symbol("ip1", dace.int64)
    # Rebind cb's in-edge to carry ip1 = i + 1.
    loop = [b for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion)][0]
    in_edge = loop.in_edges(cb)[0]
    in_edge.data.assignments["ip1"] = "i + 1"
    add_copy_arm(sdfg, cb, "ip1 < N", "ip1", "b", "i")
    assert len(conditional_blocks(sdfg)) == 1

    assert SameWriteSetIfElseToITECFG().apply_pass(sdfg, {}) is None
    assert len(conditional_blocks(sdfg)) == 1

    rewritten = BranchNormalization().apply_pass(sdfg, {})
    assert rewritten is None, "BranchNormalization flattened a transitive iteration guard"
    assert len(conditional_blocks(sdfg)) == 1


if __name__ == "__main__":
    test_branch_normalization_refuses_direct_iteration_guard()
    test_branch_normalization_refuses_two_arm_iteration_guard()
    test_branch_normalization_refuses_transitive_iteration_guard()
