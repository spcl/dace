"""
Tests for data_used_by_map: given a map entry node in a state, returns a dict
mapping data container names to their storage type for all data accessed
within the map scope.
"""

import pytest
import dace
import numpy as np

from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA

N = dace.symbol('N')
M = dace.symbol('M')

def data_used_by_map(sdfg : dace.SDFG, state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> dict[str, dace.dtypes.DeviceType]:
    """
    Collects all data containers accessed within a map scope and returns
    their storage types.

    Traverses every node reachable inside the scope of `map_entry`
    (between the entry and its corresponding exit), and for each
    AccessNode found, records the container name and its storage type
    from the parent SDFG's data descriptors.

    Intended to be called on the outermost (top-level) map entry in a
    state. Nested maps are traversed implicitly since their access nodes
    live within the outer scope.

    E.g. if we have the structure:
    MapEntry (label="outer", schedule=Sequential)
      |
      |-- MapEntry (label="inner", schedule=GPU_Device)
            | -- MapEntry (label="innermost", schedule=Sequential)
            |
            |-- AccessNode (data="A")  # A is GPU_Global
            |-- AccessNode (data="B")  # B is GPU_Global


    Then calling data_used_by_map on the "outer" entry should return:
    {
        "A": dace.dtypes.DeviceType.GPU,
        "B": dace.dtypes.DeviceType.GPU
    }

    but "innermost"s return is undefined, since it is sequential it should default to 
    the default location which is CPU. Then the return would be:
    {
        "A": dace.dtypes.DeviceType.CPU,
        "B": dace.dtypes.DeviceType.CPU
    }
    """
    gpu_set, cpu_set = OtA().get_data_locations_of_map(sdfg, state, map_entry)
    d = {}
    for name in gpu_set: d[name] = dace.dtypes.DeviceType.GPU
    for name in cpu_set: d[name] = dace.dtypes.DeviceType.CPU
    print(f"gpu_set: {gpu_set}\ncpu_set: {cpu_set}\n")
    return d

def _find_map_entry(state, map_label=None, schedule=None):
    """Return the first MapEntry matching optional label/schedule filters."""
    for node in state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue
        if map_label is not None and node.map.label != map_label:
            continue
        if schedule is not None and node.map.schedule != schedule:
            continue
        return node
    raise RuntimeError(f"No MapEntry found (label={map_label}, schedule={schedule})")


def test_triple_nested_maps_gpu():
    """
    Triple-nested map: Sequential > GPU_Device > Sequential.
    Reads A (20x10) and B (10x15), accumulates into C (20x15).
    All three arrays are GPU_Global.

    data_used_by_map is called on the outermost sequential map entry.
    Since the function traverses the entire scope (including nested maps),
    it must find all three containers. The storage type is resolved from
    the SDFG's data descriptors (the global struct), not from the map
    schedule — a sequential map can still touch GPU arrays if it is
    nested inside (or is the parent of) a GPU map.
    """
    sdfg = dace.SDFG('triple_nested')
    sdfg.add_array('A', [20, 10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [10, 15], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('C', [20, 15], dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state('compute')

    # Outer: sequential over i
    outer_entry, outer_exit = state.add_map(
        'seq_outer', {'i': '0:20'}, schedule=dace.ScheduleType.Sequential)
    # Middle: GPU_Device over j
    mid_entry, mid_exit = state.add_map(
        'gpu_mid', {'j': '0:15'}, schedule=dace.ScheduleType.GPU_Device)
    # Inner: sequential over k (runs on GPU thread)
    inner_entry, inner_exit = state.add_map(
        'seq_inner', {'k': '0:10'}, schedule=dace.ScheduleType.Sequential)

    tasklet = state.add_tasklet('mac', {'a', 'b', 'cin'}, {'cout'},
                                'cout = cin + a * b')

    a_node = state.add_read('A')
    b_node = state.add_read('B')
    c_read = state.add_access('C')
    c_write = state.add_write('C')

    # A[i, k] through all three maps
    state.add_memlet_path(a_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='a', memlet=dace.Memlet('A[i, k]'))
    # B[k, j] through all three maps
    state.add_memlet_path(b_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='b', memlet=dace.Memlet('B[k, j]'))
    # C[i, j] read through all three maps
    state.add_memlet_path(c_read, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='cin', memlet=dace.Memlet('C[i, j]'))
    # C[i, j] write back out
    state.add_memlet_path(tasklet, inner_exit, mid_exit, outer_exit, c_write,
                          src_conn='cout', memlet=dace.Memlet('C[i, j]'))

    sdfg.save('triple_nested.sdfg')
    sdfg.validate()
    #sdfg.view()

    result = data_used_by_map(sdfg, state, outer_entry)

    # All three arrays must be reported
    assert set(result.keys()) == {'A', 'B', 'C'}
    # All reside on GPU
    for name in ('A', 'B', 'C'):
        assert result[name] == dace.dtypes.DeviceType.GPU, \
            f"{name} should be GPU_Global, got {result[name]}"


def test_triple_nested_maps_outer_only_array():
    """
    Triple-nested map with an extra outer-only array D.

    Outer sequential map reads D[i] and writes E[i], while inner maps
    use A, B, C as in the original test. D/E are only accessed at the
    outer level and not by inner maps.
    """
    sdfg = dace.SDFG('triple_nested_outer_only')
    sdfg.add_array('A', [20, 10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [10, 15], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('C', [20, 15], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('D', [20], dace.float64, storage=dace.StorageType.Default)
    sdfg.add_array('E', [20], dace.float64, storage=dace.StorageType.Default)

    state = sdfg.add_state('compute')

    outer_entry, outer_exit = state.add_map(
        'seq_outer', {'i': '0:20'}, schedule=dace.ScheduleType.Sequential)
    mid_entry, mid_exit = state.add_map(
        'gpu_mid', {'j': '0:15'}, schedule=dace.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map(
        'seq_inner', {'k': '0:10'}, schedule=dace.ScheduleType.Sequential)

    tasklet = state.add_tasklet('mac', {'a', 'b', 'cin'}, {'cout'},
                                'cout = cin + a * b')
    outer_tasklet = state.add_tasklet('outer_only', {'d_in'}, {'e_out'},
                                      'e_out = d_in')

    a_node = state.add_read('A')
    b_node = state.add_read('B')
    c_read = state.add_access('C')
    c_write = state.add_write('C')

    d_read = state.add_read('D')
    e_write = state.add_write('E')

    state.add_memlet_path(a_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='a', memlet=dace.Memlet('A[i, k]'))
    state.add_memlet_path(b_node, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='b', memlet=dace.Memlet('B[k, j]'))
    state.add_memlet_path(c_read, outer_entry, mid_entry, inner_entry, tasklet,
                          dst_conn='cin', memlet=dace.Memlet('C[i, j]'))
    state.add_memlet_path(tasklet, inner_exit, mid_exit, outer_exit, c_write,
                          src_conn='cout', memlet=dace.Memlet('C[i, j]'))

    state.add_memlet_path(d_read, outer_entry, outer_tasklet,
                          dst_conn='d_in', memlet=dace.Memlet('D[i]'))
    state.add_memlet_path(outer_tasklet, outer_exit, e_write,
                          src_conn='e_out', memlet=dace.Memlet('E[i]'))

    sdfg.save('triple_nested_outer_only.sdfg')
    sdfg.validate()
    sdfg.view()

    result = data_used_by_map(sdfg, state, outer_entry)

    assert set(result.keys()) == {'A', 'B', 'C', 'D', 'E'}
    for name in ('A', 'B', 'C'):
        assert result[name] == dace.dtypes.DeviceType.GPU, \
            f"{name} should be GPU, got {result[name]}"
        
    for name in ('D', 'E'):
        assert result[name] == dace.dtypes.DeviceType.CPU, \
            f"{name} should be CPU, got {result[name]}"
        


def make_row_col_add_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    sdfg = dace.SDFG('row_col_add_views')

    # --- Data descriptors ---
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('C', [N, N], dace.float64)

    # Column view of A: A[:, i] has shape [N], stride [N] in row-major
    sdfg.add_view('A_col_view', [N], dace.float64, strides=[N])
    # Row view of C: C[i, :] has shape [N], stride [1] in row-major
    sdfg.add_view('C_row_view', [N], dace.float64, strides=[1])

    state = sdfg.add_state('compute', is_start_block=True)

    # --- Access nodes (outside maps) ---
    a_read = state.add_read('A')
    c_read = state.add_read('C')
    c_write = state.add_write('C')

    # --- Outer map: for i in range(N) ---
    outer_entry, outer_exit = state.add_map('i_map', {'i': '0:N'})

    # --- Access nodes inside outer scope (for view connections) ---
    a_inside = state.add_access('A')
    c_inside_r = state.add_access('C')
    c_inside_w = state.add_access('C')

    # --- View nodes inside outer scope ---
    a_col_node = state.add_access('A_col_view')
    c_row_in = state.add_access('C_row_view')
    c_row_out = state.add_access('C_row_view')

    # --- Inner map: for j in range(N) ---
    inner_entry, inner_exit = state.add_map('j_map', {'j': '0:N'})

    # --- Tasklet ---
    tasklet = state.add_tasklet('add', {'a_val', 'c_val'}, {'c_out'},
                                'c_out = c_val + a_val')

    # =====================================================================
    #  READ path for A[:, i]
    # =====================================================================
    # A_read -> outer_entry -> A_inside  (route through outer map)
    state.add_memlet_path(a_read, outer_entry, a_inside,
                          memlet=dace.Memlet('A[0:N, i]'))
    # A_inside -> A_col_view  (views edge: column slice)
    state.add_edge(a_inside, None, a_col_node, 'views',
                   dace.Memlet('A[0:N, i]'))
    # A_col_view -> inner_entry -> tasklet
    state.add_memlet_path(a_col_node, inner_entry, tasklet,
                          dst_conn='a_val',
                          memlet=dace.Memlet('A_col_view[j]'))

    # =====================================================================
    #  READ path for C[i, :]
    # =====================================================================
    # C_read -> outer_entry -> C_inside_r  (route through outer map)
    state.add_memlet_path(c_read, outer_entry, c_inside_r,
                          memlet=dace.Memlet('C[i, 0:N]'))
    # C_inside_r -> C_row_in  (views edge: row slice)
    state.add_edge(c_inside_r, None, c_row_in, 'views',
                   dace.Memlet('C[i, 0:N]'))
    # C_row_in -> inner_entry -> tasklet
    state.add_memlet_path(c_row_in, inner_entry, tasklet,
                          dst_conn='c_val',
                          memlet=dace.Memlet('C_row_view[j]'))

    # =====================================================================
    #  WRITE path for C[i, :]
    # =====================================================================
    # tasklet -> inner_exit -> C_row_out
    state.add_memlet_path(tasklet, inner_exit, c_row_out,
                          src_conn='c_out',
                          memlet=dace.Memlet('C_row_view[j]'))
    # C_row_out -> C_inside_w  (views edge: row slice, write direction)
    state.add_edge(c_row_out, 'views', c_inside_w, None,
                   dace.Memlet('C[i, 0:N]'))
    # C_inside_w -> outer_exit -> C_write  (route through outer map)
    state.add_memlet_path(c_inside_w, outer_exit, c_write,
                          memlet=dace.Memlet('C[i, 0:N]'))

    return sdfg, state, outer_entry, inner_entry


def test_row_col_add_views_outer():
    """Validate the manually-built SDFG against numpy."""
    n = 8
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    C = rng.random((n, n))
    C_ref = C.copy()

    # Reference: for i in range(N): C[i, :] += A[:, i]
    for i in range(n):
        C_ref[i, :] += A[:, i]

    sdfg, state, outer_entry, inner_entry = make_row_col_add_sdfg()
    sdfg.validate()

    sdfg.save("row_col_add_views.sdfg")
    ##sdfg.view()

    # TODO:
    # Important! You should decide if you want to report the original array,
    # or the view, or both. I think one should report both (or either only the original)
    # as the storage of view depends on the viewed array, but the storage of both of them
    # needs to be set, therefore I would report both of them here. 
    # All three arrays must be reported

    # test with outer entry
    result = data_used_by_map(sdfg, state, outer_entry)
    assert set(result.keys()) == {'A', 'C', 'A_col_view', 'C_row_view'}
    # All reside on CPU
    for name in ('A', 'C', 'A_col_view', 'C_row_view'):
        assert result[name] == dace.dtypes.DeviceType.CPU, \
            f"{name} should be CPU, got {result[name]}"


def test_row_col_add_views_inner():
    """Validate the manually-built SDFG against numpy."""
    n = 8
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    C = rng.random((n, n))
    C_ref = C.copy()

    # Reference: for i in range(N): C[i, :] += A[:, i]
    for i in range(n):
        C_ref[i, :] += A[:, i]

    sdfg, state, outer_entry, inner_entry = make_row_col_add_sdfg()
    sdfg.validate()

    sdfg.save("row_col_add_views.sdfg")
    ##sdfg.view()
        
    # test with inner entry
    result = data_used_by_map(sdfg, state, inner_entry)
    assert set(result.keys()) == {'A', 'C', 'A_col_view', 'C_row_view'}
    # All reside on CPU
    for name in ('A', 'C', 'A_col_view', 'C_row_view'):
        assert result[name] == dace.dtypes.DeviceType.CPU, \
            f"{name} should be CPU, got {result[name]}"
        
# ---------------------------------------------------------------------------
#  Test 3 (trivial) — Single map, CPU arrays
# ---------------------------------------------------------------------------

def test_single_map_cpu():
    sdfg = dace.SDFG('single_map_cpu')
    sdfg.add_array('X', [100], dace.float32, storage=dace.StorageType.CPU_Heap)
    sdfg.add_array('Y', [100], dace.float32, storage=dace.StorageType.CPU_Heap)

    state = sdfg.add_state('s')
    entry, exit_ = state.add_map('m', {'i': '0:100'})
    tasklet = state.add_tasklet('double', {'x'}, {'y'}, 'y = 2 * x')

    x_node = state.add_read('X')
    y_node = state.add_write('Y')

    state.add_memlet_path(x_node, entry, tasklet, dst_conn='x',
                          memlet=dace.Memlet('X[i]'))
    state.add_memlet_path(tasklet, exit_, y_node, src_conn='y',
                          memlet=dace.Memlet('Y[i]'))

    result = data_used_by_map(sdfg, state, entry)

    sdfg.save("single_map_cpu.sdfg")
    #sdfg.view()

    assert set(result.keys()) == {'X', 'Y'}
    for name in ('X', 'Y'):
        assert result[name] == dace.dtypes.DeviceType.CPU, f"{name} should be CPU_Heap, got {result[name]}"


def test_empty_map():
    sdfg = dace.SDFG('empty_map')
    state = sdfg.add_state('s')
    entry, exit_ = state.add_map('m', {'i': '0:10'})
    # Connect entry directly to exit (empty body)
    state.add_edge(entry, None, exit_, None, dace.Memlet())

    sdfg.save("empty_map.sdfg")
    #sdfg.view()

    result = data_used_by_map(sdfg, state, entry)
    assert result == {}


def make_non_input_readwrite_node_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]:
    """
    C[i,j] = C[i,j] + 1
    A[i,j] = B[i,j] + C[i,j]

    A: input only
    B: input only
    C: non-transient, read-modify-write inside the map
    """
    sdfg = dace.SDFG('inplace_c')

    sdfg.add_array('A', [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [N, M], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('C', [N, M], dace.float64, storage=dace.StorageType.GPU_Global)

    state = sdfg.add_state('compute', is_start_block=True)

    # External access nodes
    b_read = state.add_read('B')
    a_write = state.add_write('A')

    # Map over i, j
    entry, exit_ = state.add_map('ij_map', {'i': '0:N', 'j': '0:M'}, schedule=dace.dtypes.ScheduleType.GPU_Device)

    # Two tasklets: first increments C, second computes A
    t_inc = state.add_tasklet('inc_c', {'b_in'}, {'c_out'}, 'c_out = b_in + 1')
    t_add = state.add_tasklet('add', {'b_in', 'c_new'}, {'a_out'}, 'a_out = b_in + c_new')


    # --- Updated C feeds into t_add ---
    c_node = state.add_access('C')   # both read and write
    
    state.add_edge(entry, "OUT_B", t_inc, 'b_in', dace.Memlet('B[i, j]'))
    state.add_edge(t_inc, "c_out", c_node, None, dace.Memlet('C[i, j]'))
    state.add_edge(c_node, None, t_add, 'c_new', dace.Memlet('C[i, j]'))


    # --- B read into t_add ---
    state.add_memlet_path(b_read, entry, t_add,
                          dst_conn='b_in',
                          memlet=dace.Memlet('B[i, j]'))

    # --- t_add writes A ---
    exit_.add_in_connector('IN_A')
    exit_.add_out_connector('OUT_A')

    state.add_edge(t_add, 'a_out', exit_, 'IN_A',
                dace.Memlet('A[i, j]'))
    state.add_edge(exit_, 'OUT_A', a_write, None,
                dace.Memlet('A[0:N, 0:M]'))

    return sdfg, state, entry

def test_map_with_non_input_readwrite_node() -> dace.SDFG:
    sdfg, state, entry = make_non_input_readwrite_node_sdfg()

    sdfg.validate()
    sdfg.save("non_input_readwrite.sdfg")
    #sdfg.view()

    result = data_used_by_map(sdfg, state, entry)
    assert set(result.keys()) == {'A', 'C', 'B'}
    # All reside on GPU
    for name in ('A', 'C', 'B'):
        assert result[name] == dace.dtypes.DeviceType.GPU, \
            f"{name} should be GPU, got {result[name]}"


if __name__ == "__main__":
    test_triple_nested_maps_gpu()
    test_triple_nested_maps_outer_only_array()
    test_row_col_add_views_outer()
    test_row_col_add_views_inner()
    test_single_map_cpu()
    test_empty_map()
    test_map_with_non_input_readwrite_node()
    print("YESS! All tests passed.")