import copy
import numpy
import pytest
import dace
import dace.sdfg.construction_utils as cutil

S = dace.symbol("S")


@dace.program
def overlapping_access(A: dace.float64[2, 2, S], B: dace.float64[S]):
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def non_overlapping_access(A: dace.float64[2, 2, S], B: dace.float64[S]):
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + 1.2 * A[0, 0, i]


@dace.program
def overlapping_access_with_previous_write(A: dace.float64[2, 2, S], B: dace.float64[S]):
    A[0, 0, 0] = 3.0
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_with_intermediate_access_node(A: dace.float64[2, 2, S], B: dace.float64[2, 2, S], C: dace.float64[2,
                                                                                                                  S]):
    A[0, 0, 0] = 3.0
    for i in dace.map[0:2:1]:
        c = B[i, 0, 0]
        for j in dace.map[0:S:1]:
            C[i, j] = c * (B[1, 0, j] + B[0, 0, j]) + A[1, 0, j]


@dace.program
def jacobi2d(A: dace.float64[S, S], B: dace.float64[S, S], tsteps: dace.int64):  #, N, tsteps):
    for t in range(tsteps):
        for i, j in dace.map[0:S - 2, 0:S - 2]:
            B[i + 1, j + 1] = 0.2 * (A[i + 1, j + 1] + A[i, j + 1] + A[i + 2, j + 1] + A[i + 1, j] + A[i + 1, j + 2])

        for i, j in dace.map[0:S - 2, 0:S - 2]:
            A[i + 1, j + 1] = 0.2 * (B[i + 1, j + 1] + B[i, j + 1] + B[i + 2, j + 1] + B[i + 1, j] + B[i + 1, j + 2])


def run_comparison(sdfg1, sdfg2, arrays, params):
    # Create copies for comparison
    arrays_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arrays_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # Original SDFG
    sdfg = sdfg1
    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg = sdfg2
    copy_sdfg.name = copy_sdfg.name + "_transformed"

    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)
    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert numpy.allclose(arrays_orig[name], arrays_vec[name]), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"


@pytest.mark.parametrize("apply_only_if_subsets_not_equal", [True, False])
def test_jacobi2d(apply_only_if_subsets_not_equal: bool):
    sdfg = jacobi2d.to_sdfg()
    sdfg.name += "_apply_only_if_subsets_not_equal_" + str(apply_only_if_subsets_not_equal)
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)

    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 2
    map_entry1, state1 = map_entries.pop()
    map_entry2, state2 = map_entries.pop()

    applied1 = cutil.duplicate_memlets_sharing_single_in_connector(state1, map_entry1, apply_only_if_subsets_not_equal)
    copy_sdfg.validate()
    applied2 = cutil.duplicate_memlets_sharing_single_in_connector(state2, map_entry2, apply_only_if_subsets_not_equal)
    copy_sdfg.validate()

    assert applied1
    assert applied2

    _S = 64
    tsteps = 5
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


@pytest.mark.parametrize("apply_only_if_subsets_not_equal", [True, False])
def test_overlapping_access(apply_only_if_subsets_not_equal: bool):
    sdfg = overlapping_access.to_sdfg()
    sdfg.name += "_apply_only_if_subsets_not_equal_" + str(apply_only_if_subsets_not_equal)
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    applied1 = cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry, apply_only_if_subsets_not_equal)
    copy_sdfg.validate()

    assert applied1

    _S = 64
    tsteps = 5
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S, ))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


@pytest.mark.parametrize("apply_only_if_subsets_not_equal", [True, False])
def test_non_overlapping_access(apply_only_if_subsets_not_equal: bool):
    sdfg = non_overlapping_access.to_sdfg()
    sdfg.name += "_apply_only_if_subsets_not_equal_" + str(apply_only_if_subsets_not_equal)
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    applied1 = cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry, apply_only_if_subsets_not_equal)
    copy_sdfg.validate()

    if apply_only_if_subsets_not_equal:
        assert not applied1
    else:
        assert applied1

    _S = 64
    tsteps = 5
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S, ))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


@pytest.mark.parametrize("apply_only_if_subsets_not_equal", [True, False])
def test_overlapping_access_with_previous_write(apply_only_if_subsets_not_equal: bool):
    sdfg = overlapping_access_with_previous_write.to_sdfg()
    sdfg.name += "_apply_only_if_subsets_not_equal_" + str(apply_only_if_subsets_not_equal)
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    applied1 = cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry, apply_only_if_subsets_not_equal)
    copy_sdfg.validate()

    assert applied1

    _S = 64
    tsteps = 5
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S, ))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


@pytest.mark.parametrize("apply_only_if_subsets_not_equal", [True, False])
def test_overlapping_access_with_intermediate_access_node(apply_only_if_subsets_not_equal: bool):
    sdfg = overlapping_with_intermediate_access_node.to_sdfg()
    sdfg.name += "_apply_only_if_subsets_not_equal_" + str(apply_only_if_subsets_not_equal)
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g)
                   for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)
                   if g.scope_dict()[n] is not None}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    applied1 = cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry, apply_only_if_subsets_not_equal)
    copy_sdfg.validate()
    assert applied1

    _S = 64
    tsteps = 5
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((2, 2, _S))
    C = numpy.random.random((
        2,
        _S,
    ))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B, "C": C}, params={"tsteps": tsteps, "S": _S})


if __name__ == "__main__":
    for p in [True, False]:
        test_overlapping_access(p)
        test_non_overlapping_access(p)
        test_overlapping_access_with_previous_write(p)
        test_overlapping_access_with_intermediate_access_node(p)
        test_jacobi2d(p)
