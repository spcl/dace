# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy
import dace

from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.unblock_dimensions import UnblockDimensions

N = dace.symbol("N")


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])


@dace.program
def madd_blocked(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N:16, 0:N:4] @ dace.ScheduleType.Sequential:
        for ii, jj in dace.map[i:i + 16, j:j + 4] @ dace.ScheduleType.Sequential:
            C[ii, jj] = 0.5 * (A[ii, jj] + B[ii, jj])


def test_block_then_unblock_roundtrip():
    """Block an array then Unblock it with the same map: shapes and results
    return to the flat baseline (Unblock inverts Block)."""
    sdfg = madd.to_sdfg()
    split_map = {
        "A": ([True, True], [16, 4]),
        "B": ([True, True], [16, 4]),
    }
    SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    assert len(sdfg.arrays["A"].shape) == 4  # blocked: [N/16, N/4, 16, 4]

    UnblockDimensions(unblock_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    assert tuple(str(s) for s in sdfg.arrays["A"].shape) == ("N", "N")
    assert tuple(str(s) for s in sdfg.arrays["B"].shape) == ("N", "N")

    _N = 16 * 4 * 2
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C = numpy.zeros((_N, _N))
    ref = 0.5 * (A + B)
    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, ref)


def test_unblock_native_blocked_kernel():
    """Unblock a natively 5-loop blocked kernel down to flat arrays; run against a
    physically-blocked input and compare to the flat baseline."""
    baseline = madd.to_sdfg()

    sdfg = madd_blocked.to_sdfg()
    # Present A, B to the unblock pass in blocked (4D) form first.
    split_map = {
        "A": ([True, True], [16, 4]),
        "B": ([True, True], [16, 4]),
    }
    SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    UnblockDimensions(unblock_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    assert tuple(str(s) for s in sdfg.arrays["A"].shape) == ("N", "N")

    _N = 16 * 4 * 2
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))
    baseline(A=A.copy(), B=B.copy(), C=C0, N=_N)
    sdfg(A=A.copy(), B=B.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


def indexed_by(expr_str: str, name: str):
    """Every ``name[...]`` access in ``expr_str``, as a sorted list of index-string tuples."""
    expr = dace.symbolic.pystr_to_symbolic(expr_str)
    return sorted(
        tuple(str(a) for a in node.args[1:]) for node in expr.atoms(dace.symbolic.Subscript)
        if str(node.args[0]) == name)


def test_unblock_rewrites_every_access_and_keeps_the_rest_of_the_expression():
    """Interstate assignments hold whole expressions, not bare accesses.

    Accesses were located with a greedy ``A\\[(.*)\\]`` regex and folded back with a greedy
    ``re.sub`` of the same pattern, so on ``A[i, j, t] * B[i, j]`` the substitution ran from A's
    opening bracket to B's closing one and DELETED B's access outright. A rank mismatch also fell
    through unrewritten, reshaping the array while leaving its index in the blocked space."""
    unblock = UnblockDimensions(unblock_map={})
    masks, factors = [True, False], [4, 1]

    sdfg = dace.SDFG("probe")
    st0 = sdfg.add_state("s0")
    st1 = sdfg.add_state("s1")
    edge = dace.InterstateEdge(assignments={"v": "(A[i, j, t] * B[i, j])"})
    sdfg.add_edge(st0, st1, edge)
    unblock._replace_interstate_edges_recursive(sdfg, "A", masks, factors)

    out = edge.assignments["v"]
    assert indexed_by(out, "A") == [("4*i + t", "j")], out
    assert indexed_by(out, "B") == [("i", "j")], out  # the access the greedy re.sub used to delete

    # a rank that does not match the blocked rank is an error, not a silent pass-through
    edge.assignments["v"] = "A[i, j]"
    raised = False
    try:
        unblock._replace_interstate_edges_recursive(sdfg, "A", masks, factors)
    except ValueError as exc:
        raised = True
        assert "blocked rank" in str(exc)
    assert raised, "expected UnblockDimensions to reject an access that is not in the blocked index space"


if __name__ == "__main__":
    test_block_then_unblock_roundtrip()
    test_unblock_native_blocked_kernel()
    test_unblock_rewrites_every_access_and_keeps_the_rest_of_the_expression()
    print("unblock tests PASS")
