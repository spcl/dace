# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import subsets as sbs
from dace.transformation.helpers import find_sdfg_control_flow


def test_read_write_set():
    sdfg = dace.SDFG('graph')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    sdfg.add_array('C', [10], dace.float64)
    state = sdfg.add_state('state')
    task1 = state.add_tasklet('work1', {'A'}, {'B'}, 'B = A + 1')
    task2 = state.add_tasklet('work2', {'B'},  {'C'}, 'C = B + 1')
    read_a = state.add_access('A')
    rw_b = state.add_access('B')
    write_c = state.add_access('C')
    state.add_memlet_path(read_a, task1, dst_conn='A', memlet=dace.Memlet('A[2]'))
    state.add_memlet_path(task1, rw_b, src_conn='B', memlet=dace.Memlet('B[2]'))
    state.add_memlet_path(rw_b, task2, dst_conn='B', memlet=dace.Memlet('B[2]'))
    state.add_memlet_path(task2, write_c, src_conn='C', memlet=dace.Memlet('C[2]'))

    assert 'B' not in state.read_and_write_sets()[0]


def test_read_write_set_y_formation():
    sdfg = dace.SDFG('graph')
    state = sdfg.add_state('state')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_array('C', [2], dace.float64)
    task1 = state.add_tasklet('work1', {'A'}, {'B'}, 'B = A + 1')
    task2 = state.add_tasklet('work2', {'B'},  {'C'}, 'C += B + 1')
    task3 = state.add_tasklet('work3', {'A'},  {'B'}, 'B = A + 2')
    read_a = state.add_access('A')
    rw_b = state.add_access('B')
    write_c = state.add_access('C')
    state.add_memlet_path(read_a, task1, dst_conn='A', memlet=dace.Memlet(data='A', subset='0'))
    state.add_memlet_path(read_a, task3, dst_conn='A', memlet=dace.Memlet(data='A', subset='1'))
    state.add_memlet_path(task1, rw_b, src_conn='B', memlet=dace.Memlet(data='B', subset='0'))
    state.add_memlet_path(task3, rw_b, src_conn='B', memlet=dace.Memlet(data='B', subset='0'))
    state.add_memlet_path(rw_b, task2, dst_conn='B', memlet=dace.Memlet(data='B', subset='0'))
    state.add_memlet_path(task2, write_c, src_conn='C', memlet=dace.Memlet(data='C', subset='0'))

    assert 'B' not in state.read_and_write_sets()[0]


def test_deepcopy_state():
    N = dace.symbol('N')

    @dace.program
    def double_loop(arr: dace.float32[N]):
        for i in range(N):
            arr[i] *= 2
        for i in range(N):
            arr[i] *= 2

    sdfg = double_loop.to_sdfg()
    find_sdfg_control_flow(sdfg)
    sdfg.validate()


def test_read_and_write_set_filter():
    sdfg = dace.SDFG('graph')
    state = sdfg.add_state('state')
    sdfg.add_array('A', [2, 2], dace.float64)
    sdfg.add_scalar('B', dace.float64)
    sdfg.add_array('C', [2, 2], dace.float64)
    A, B, C = (state.add_access(name) for name in ('A', 'B', 'C'))

    state.add_nedge(
            A,
            B,
            dace.Memlet("B[0] -> 0, 0"),
    )
    state.add_nedge(
            B,
            C,
            # If the Memlet would be `B[0] -> 1, 1` it would then be filtered out.
            #   This is an intentional behaviour for compatibility.
            dace.Memlet("C[1, 1] -> 0"),
    )
    state.add_nedge(
            B,
            C,
            dace.Memlet("B[0] -> 0, 0"),
    )
    sdfg.validate()

    expected_reads = {
            "A": [sbs.Range.from_string("0, 0")],
            # See comment in `state._read_and_write_sets()` why "B" is here
            #   it should actually not, but it is a bug.
            "B": [sbs.Range.from_string("0")],
    }
    expected_writes = {
            # However, this should always be here.
            "B": [sbs.Range.from_string("0")],
            "C": [sbs.Range.from_string("0, 0"), sbs.Range.from_string("1, 1")],
    }
    read_set, write_set = state._read_and_write_sets()

    for expected_sets, computed_sets in [(expected_reads, read_set), (expected_writes, write_set)]:
        assert expected_sets.keys() == computed_sets.keys(), f"Expected the set to contain '{expected_sets.keys()}' but got '{computed_sets.keys()}'."
        for access_data in expected_sets.keys():
            for exp in expected_sets[access_data]:
                found_match = False
                for res in computed_sets[access_data]:
                    if res == exp:
                        found_match = True
                        break
                assert found_match, f"Could not find the subset '{exp}' only got '{computed_sets}'"


def test_read_and_write_set_selection():
    sdfg = dace.SDFG('graph')
    state = sdfg.add_state('state')
    sdfg.add_array('A', [2, 2], dace.float64)
    sdfg.add_scalar('B', dace.float64)
    A, B = (state.add_access(name) for name in ('A', 'B'))

    state.add_nedge(
            A,
            B,
            dace.Memlet("A[0, 0]"),
    )
    sdfg.validate()

    expected_reads = {
            "A": [sbs.Range.from_string("0, 0")],
    }
    expected_writes = {
            "B": [sbs.Range.from_string("0")],
    }
    read_set, write_set = state._read_and_write_sets()

    for expected_sets, computed_sets in [(expected_reads, read_set), (expected_writes, write_set)]:
        assert expected_sets.keys() == computed_sets.keys(), f"Expected the set to contain '{expected_sets.keys()}' but got '{computed_sets.keys()}'."
        for access_data in expected_sets.keys():
            for exp in expected_sets[access_data]:
                found_match = False
                for res in computed_sets[access_data]:
                    if res == exp:
                        found_match = True
                        break
                assert found_match, f"Could not find the subset '{exp}' only got '{computed_sets}'"


if __name__ == '__main__':
    test_read_write_set()
    test_read_write_set_y_formation()
    test_deepcopy_state()
    test_read_and_write_set_selection()
    test_read_and_write_set_filter()

