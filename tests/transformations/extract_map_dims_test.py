
# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation import helpers as xfh


def test_extract_all_dims():
    """
    Test that extract_map_dims works correctly when extracting all dimensions.
    This is a regression test for an IndexError that occurred when len(dims) == len(entries).
    """

    # Create a simple SDFG with a 3D map
    sdfg = dace.SDFG('test_extract_all_dims')
    state = sdfg.add_state()

    sdfg.add_array('A', [10, 20, 30], dace.float64)
    sdfg.add_array('B', [10, 20, 30], dace.float64)

    map_entry, map_exit = state.add_map('map', dict(i='0:10', j='0:20', k='0:30'))

    a_read = state.add_read('A')
    b_write = state.add_write('B')

    tasklet = state.add_tasklet('compute', {'a_in'}, {'b_out'}, 'b_out = a_in')

    state.add_memlet_path(a_read, map_entry, tasklet, dst_conn='a_in', memlet=dace.Memlet('A[i, j, k]'))
    state.add_memlet_path(tasklet, map_exit, b_write, src_conn='b_out', memlet=dace.Memlet('B[i, j, k]'))

    # When all dimensions are extracted, there should be no remainder map
    extracted_map, remainder_map = xfh.extract_map_dims(sdfg, map_entry, [0, 1, 2])

    # Verify that both maps are returned (not None)
    assert extracted_map is not None, "Extracted map should not be None"
    assert remainder_map is None, "Remainder map should be None in this case"

    # Verify the map has 3 parameters (all extracted)
    assert len(extracted_map.map.params) == 3, f"Expected 3 parameters, got {len(extracted_map.map.params)}"


def test_extract_partial_dims():
    """
    Test that extract_map_dims works correctly when extracting only some dimensions.
    """

    # Create a simple SDFG with a 3D map
    sdfg = dace.SDFG('test_extract_partial_dims')
    state = sdfg.add_state()

    sdfg.add_array('A', [10, 20, 30], dace.float64)
    sdfg.add_array('B', [10, 20, 30], dace.float64)

    map_entry, map_exit = state.add_map('map', dict(i='0:10', j='0:20', k='0:30'))

    a_read = state.add_read('A')
    b_write = state.add_write('B')

    tasklet = state.add_tasklet('compute', {'a_in'}, {'b_out'}, 'b_out = a_in')

    state.add_memlet_path(a_read, map_entry, tasklet, dst_conn='a_in', memlet=dace.Memlet('A[i, j, k]'))
    state.add_memlet_path(tasklet, map_exit, b_write, src_conn='b_out', memlet=dace.Memlet('B[i, j, k]'))

    # Extract only first 2 dimensions
    extracted_map, remainder_map = xfh.extract_map_dims(sdfg, map_entry, [0, 1])

    assert extracted_map is not None, "Extracted map should not be None"
    assert remainder_map is not None, "Remainder map should not be None"

    # When partial dimensions are extracted, they should be different maps
    assert extracted_map != remainder_map, "Extracted and remainder maps should be different"

    # Extracted map should have 2 parameters
    assert len(extracted_map.map.params) == 2, f"Expected 2 parameters in extracted map, got {len(extracted_map.map.params)}"

    # Remainder map should have 1 parameter
    assert len(remainder_map.map.params) == 1, f"Expected 1 parameter in remainder map, got {len(remainder_map.map.params)}"


def test_extract_single_dim_from_multidim():
    """
    Test that extract_map_dims works correctly when extracting a single dimension from a multi-dimensional map.
    """

    # Create a simple SDFG with a 3D map
    sdfg = dace.SDFG('test_extract_single_dim')
    state = sdfg.add_state()

    sdfg.add_array('A', [10, 20, 30], dace.float64)
    sdfg.add_array('B', [10, 20, 30], dace.float64)

    map_entry, map_exit = state.add_map('map', dict(i='0:10', j='0:20', k='0:30'))

    a_read = state.add_read('A')
    b_write = state.add_write('B')

    tasklet = state.add_tasklet('compute', {'a_in'}, {'b_out'}, 'b_out = a_in')

    state.add_memlet_path(a_read, map_entry, tasklet, dst_conn='a_in', memlet=dace.Memlet('A[i, j, k]'))
    state.add_memlet_path(tasklet, map_exit, b_write, src_conn='b_out', memlet=dace.Memlet('B[i, j, k]'))

    # Extract only the first dimension
    extracted_map, remainder_map = xfh.extract_map_dims(sdfg, map_entry, [0])

    assert extracted_map is not None, "Extracted map should not be None"
    assert remainder_map is not None, "Remainder map should not be None"

    assert extracted_map != remainder_map, "Extracted and remainder maps should be different"

    # Extracted map should have 1 parameter
    assert len(extracted_map.map.params) == 1, f"Expected 1 parameter in extracted map, got {len(extracted_map.map.params)}"

    # Remainder map should have 2 parameters
    assert len(remainder_map.map.params) == 2, f"Expected 2 parameters in remainder map, got {len(remainder_map.map.params)}"


if __name__ == '__main__':
    test_extract_all_dims()
    test_extract_partial_dims()
    test_extract_single_dim_from_multidim()
