import numpy as np
import pytest

import dace
from dace.transformation.dataflow import AugAssignToWCR


def test_aug_assign_tasklet_lhs():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = a + k

    sdfg = sdfg_aug_assign_tasklet_lhs.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_brackets():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_brackets(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = a + (k + 1)

    sdfg = sdfg_aug_assign_tasklet_lhs_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = k + a

    sdfg = sdfg_aug_assign_tasklet_rhs.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs_brackets():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs_brackets(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = (k + 1) + a

    sdfg = sdfg_aug_assign_tasklet_rhs_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = a + k;
                """

    sdfg = sdfg_aug_assign_tasklet_lhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_brackets_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_brackets_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = a + (k + 1);
                """

    sdfg = sdfg_aug_assign_tasklet_lhs_brackets_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs_brackets_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs_brackets_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = (k + 1) + a;
                """

    sdfg = sdfg_aug_assign_tasklet_rhs_brackets_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_func_lhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_func_lhs_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                c << B[i]
                b >> A[i]
                """
                b = min(a, c);
                """

    sdfg = sdfg_aug_assign_tasklet_func_lhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_func_rhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_func_rhs_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                c << B[i]
                b >> A[i]
                """
                b = min(c, a);
                """

    sdfg = sdfg_aug_assign_tasklet_func_rhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_free_map():

    @dace.program
    def sdfg_aug_assign_free_map(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet(language=dace.Language.CPP):
                a << A[0]
                k << B[i]
                b >> A[0]
                """
                b = k * a;
                """

    sdfg = sdfg_aug_assign_free_map.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_state_fission_map():

    @dace.program
    def sdfg_aug_assign_state_fission(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << B[i]
                b >> A[i]
                b = a

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[0]
                b >> A[0]
                b = a * 2

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[0]
                b >> A[0]
                b = a * 2

    sdfg = sdfg_aug_assign_state_fission.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 2


def test_free_map_permissive():

    @dace.program
    def sdfg_free_map_permissive(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = k * a;
                """

    sdfg = sdfg_free_map_permissive.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=False)
    assert applied == 0

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=True)
    assert applied == 1


def test_aug_assign_same_inconns():

    @dace.program
    def sdfg_aug_assign_same_inconns(A: dace.float64[32]):
        for i in dace.map[0:31]:
            with dace.tasklet(language=dace.Language.Python):
                a << A[i]
                b << A[i + 1]
                c >> A[i]

                c = a * b

    sdfg = sdfg_aug_assign_same_inconns.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=True)
    assert applied == 1


@pytest.mark.parametrize("update,reference", [("max(a, k)", np.maximum), ("min(k, a)", np.minimum)])
def test_aug_assign_python_min_max(update, reference):
    """A Python ``a = max(a, k)`` / ``min`` update becomes a max / min WCR, and the result does not move."""
    sdfg = dace.SDFG(f"aug_assign_python_{update[:3]}")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    state = sdfg.add_state()
    _, entry, exit_node = state.add_mapped_tasklet(
        "update",
        {"i": "0:8"},
        {"a": dace.Memlet("A[0]"), "k": dace.Memlet("B[i]")},
        f"b = {update}",
        {"b": dace.Memlet("A[0]")},
        external_edges=True,
    )
    assert sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=True) == 1
    wcr = next(e.data.wcr for e in state.edges() if e.data.wcr is not None)
    assert update[:3] in wcr

    B = np.arange(8, dtype=np.float64) - 3.0
    A = np.array([0.5])
    sdfg(A=A, B=B)
    assert A[0] == reference.reduce(np.concatenate(([0.5], B)))


def test_aug_assign_python_logical_or():
    """``a = a or k`` on booleans becomes a logical-or WCR."""
    sdfg = dace.SDFG("aug_assign_python_or")
    sdfg.add_array("A", [1], dace.bool_)
    sdfg.add_array("B", [8], dace.bool_)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "update",
        {"i": "0:8"},
        {"a": dace.Memlet("A[0]"), "k": dace.Memlet("B[i]")},
        "b = a or k",
        {"b": dace.Memlet("A[0]")},
        external_edges=True,
    )
    assert sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=True) == 1
    B = np.zeros(8, dtype=np.bool_)
    B[5] = True
    A = np.array([False])
    sdfg(A=A, B=B)
    assert A[0]


def test_aug_assign_python_reads_another_element():
    """``A[0] = A[1] + k`` reads another element than it writes, so it is no update."""
    sdfg = dace.SDFG("aug_assign_python_other_element")
    sdfg.add_array("A", [2], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet("t", {"a", "k"}, {"b"}, "b = a + k")
    state.add_edge(state.add_read("A"), None, t, "a", dace.Memlet("A[1]"))
    state.add_edge(state.add_read("B"), None, t, "k", dace.Memlet("B[0]"))
    state.add_edge(t, "b", state.add_write("A"), None, dace.Memlet("A[0]"))
    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 0


def test_aug_assign_python_after_a_write_fissions_the_state():
    """The accumulator is written earlier in the same state, so the update moves to a state of its own first."""
    sdfg = dace.SDFG("aug_assign_python_isolate")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    init = state.add_tasklet("init", {}, {"o"}, "o = 1.0")
    acc = state.add_access("A")
    state.add_edge(init, "o", acc, None, dace.Memlet("A[0]"))
    update = state.add_tasklet("update", {"a", "k"}, {"b"}, "b = max(a, k)")
    state.add_edge(acc, None, update, "a", dace.Memlet("A[0]"))
    state.add_edge(state.add_read("B"), None, update, "k", dace.Memlet("B[0]"))
    state.add_edge(update, "b", state.add_write("A"), None, dace.Memlet("A[0]"))
    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 1
    A, B = np.zeros(1), np.array([3.0])
    sdfg(A=A, B=B)
    assert A[0] == 3.0


if __name__ == "__main__":
    test_aug_assign_python_after_a_write_fissions_the_state()
    test_aug_assign_tasklet_lhs()
    test_aug_assign_tasklet_lhs_brackets()
    test_aug_assign_tasklet_rhs()
    test_aug_assign_tasklet_rhs_brackets()
    test_aug_assign_tasklet_lhs_cpp()
    test_aug_assign_tasklet_lhs_brackets_cpp()
    test_aug_assign_tasklet_rhs_brackets_cpp()
    test_aug_assign_tasklet_func_lhs_cpp()
    test_aug_assign_tasklet_func_rhs_cpp()
    test_aug_assign_free_map()
    test_aug_assign_state_fission_map()
    test_free_map_permissive()
    test_aug_assign_same_inconns()
    test_aug_assign_python_min_max("max(a, k)", np.maximum)
    test_aug_assign_python_min_max("min(k, a)", np.minimum)
    test_aug_assign_python_logical_or()
    test_aug_assign_python_reads_another_element()
