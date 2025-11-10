import copy
import numpy
import dace
import dace.sdfg.construction_utils as cutil

S = dace.symbol("S")


@dace.program
def overlapping_access(A: dace.float64[2, 2, S], B: dace.float64[S]):
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


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
    copy_sdfg.save("x.sdfg")

    # Run both
    c_sdfg(**arrays_orig, **params)
    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert numpy.allclose(arrays_orig[name], arrays_vec[name]), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"


def test_jacobi2d():
    sdfg = jacobi2d.to_sdfg()
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)

    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 2
    map_entry1, state1 = map_entries.pop()
    map_entry2, state2 = map_entries.pop()

    cutil.duplicate_memlets_sharing_single_in_connector(state1, map_entry1)
    copy_sdfg.validate()
    cutil.duplicate_memlets_sharing_single_in_connector(state2, map_entry2)
    copy_sdfg.validate()

    _S = 64
    tsteps = 5
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


def test_overlapping_access():
    sdfg = overlapping_access.to_sdfg()
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry)
    copy_sdfg.validate()

    _S = 64
    tsteps = 5
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S, ))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


def test_overlapping_access_with_previous_write():
    sdfg = overlapping_access_with_previous_write.to_sdfg()
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g) for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry)
    copy_sdfg.validate()

    _S = 64
    tsteps = 5
    A = numpy.random.random((2, 2, _S))
    B = numpy.random.random((_S, ))
    run_comparison(sdfg, copy_sdfg, arrays={"A": A, "B": B}, params={"tsteps": tsteps, "S": _S})


def test_overlapping_access_with_intermediate_access_node():
    sdfg = overlapping_with_intermediate_access_node.to_sdfg()
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    map_entries = {(n, g)
                   for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)
                   if g.scope_dict()[n] is not None}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    sdfg.save("x.sdfg")

    cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry)
    copy_sdfg.validate()
    sdfg.save("y.sdfg")

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
    test_overlapping_access()
    test_overlapping_access_with_previous_write()
    test_overlapping_access_with_intermediate_access_node()
    test_jacobi2d()
