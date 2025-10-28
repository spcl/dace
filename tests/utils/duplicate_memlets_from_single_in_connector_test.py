import dace
import dace.sdfg.construction_utils as cutil

S = dace.symbol("S")

@dace.program
def overlapping_access(A: dace.float64[2, 2, S],
                       B: dace.float64[S]):
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_access_with_previous_write(A: dace.float64[2, 2, S],
                                           B: dace.float64[S]):
    A[0, 0, 0] = 3.0
    for i in dace.map[0:S:1]:
        B[i] = A[0, 0, i] + A[1, 0, i]


@dace.program
def overlapping_with_intermediate_access_node(A: dace.float64[2, 2, S],
                                                B: dace.float64[2, 2, S],
                                                C: dace.float64[S]):
    A[0, 0, 0] = 3.0
    for i in dace.map[0:2:1]:
        c = B[i, 0, 0]
        for j in dace.map[0:S:1]:
            C[j] = c * (B[1, 0, j] + B[0, 0, j]) + A[1, 0, j]


def test_overlapping_access():
    sdfg = overlapping_access.to_sdfg()
    sdfg.validate()

    map_entries = {(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry)
    sdfg.validate()


def test_overlapping_access_with_previous_write():
    sdfg = overlapping_access_with_previous_write.to_sdfg()
    sdfg.validate()

    map_entries = {(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry)
    sdfg.validate()




def test_overlapping_access_with_intermediate_access_node():
    sdfg = overlapping_with_intermediate_access_node.to_sdfg()
    sdfg.validate()

    map_entries = {(n, g) for n, g in sdfg.all_nodes_recursive()
                   if isinstance(n, dace.nodes.MapEntry) and g.scope_dict()[n] is not None}
    assert len(map_entries) == 1
    map_entry, state = map_entries.pop()

    cutil.duplicate_memlets_sharing_single_in_connector(state, map_entry)
    sdfg.validate()


if __name__ == "__main__":
    test_overlapping_access()
    test_overlapping_access_with_previous_write()
    test_overlapping_access_with_intermediate_access_node()
