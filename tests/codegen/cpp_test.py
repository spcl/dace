# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from functools import reduce
from operator import mul
import warnings

from dace import SDFG, Memlet, config, dtypes, symbol
from dace.codegen import codegen
from dace.codegen.targets import cpp
from dace.codegen.targets.cpu import use_aligned_operator_new
from dace.subsets import Range


def test_ndcopy_to_strided_copy_declines_broadcast_source():
    """A broadcast source (a Scalar / length-1 splatted into a width-W tile) has no
    single strided source dimension the 1D fast path can name. It must decline
    (return ``None``) so the caller falls back to the general ND-copy emitter --
    not raise ``StopIteration`` on the bare ``next`` (the adi ``stage_v_w0 ->
    v_tile_out`` scalar-broadcast miscompile)."""
    src_subset = Range([(0, 0, 1)])  # size 1 -> broadcast source
    dst_subset = Range([(0, 7, 1)])  # size 8
    assert cpp.ndcopy_to_strided_copy([8], [1], [1], [8], [1], dst_subset, src_subset, dst_subset) is None
    # Symmetric: a broadcast destination (all-ones dst shape) also declines.
    assert cpp.ndcopy_to_strided_copy([8], [8], [1], [1], [1], src_subset, dst_subset, src_subset) is None


def test_reshape_strides_multidim_array_all_dims_unit():
    r = Range([(0, 0, 1), (0, 0, 1)])

    # To smaller-sized shape
    target_dims = [1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1]
    assert strides == [1]

    # To equal-sized shape
    target_dims = [1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1, 1]
    assert strides == [1, 1]

    # To larger-sized shape
    target_dims = [1, 1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1, 1, 1]
    assert strides == [1, 1, 1]


def test_reshape_strides_multidim_array_some_dims_unit():
    r = Range([(0, 1, 1), (0, 0, 1)])

    # To smaller-sized shape
    target_dims = [2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [2, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1, 1]
    # To equal-sized shape, but units first.
    target_dims = [1, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 1]

    # To larger-sized shape.
    target_dims = [2, 1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1, 1, 1]
    # To larger-sized shape, but units first.
    target_dims = [1, 1, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 2, 1]


def test_reshape_strides_multidim_array_different_shape():
    r = Range([(0, 4, 1), (0, 5, 1)])

    # To smaller-sized shape
    target_dims = [30]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [15, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 1]

    # To larger-sized shape
    target_dims = [3, 5, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [10, 2, 1]


def test_reshape_strides_from_strided_range():
    r = Range([(0, 4, 2), (0, 6, 2)])

    # To smaller-sized shape
    target_dims = [12]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [4, 3]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [3, 1]

    # To larger-sized shape
    target_dims = [2, 3, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [6, 2, 1]


def test_reshape_strides_from_strided_and_offset_range():
    r = Range([(10, 14, 2), (10, 16, 2)])

    # To smaller-sized shape
    target_dims = [12]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [4, 3]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [3, 1]

    # To larger-sized shape
    target_dims = [2, 3, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [6, 2, 1]


def test_arrays_bigger_than_max_stack_size_get_deallocated():
    # Setup SDFG with array A that is too big to be allocated on the stack.
    sdfg = SDFG("test")
    array_a_alignment = 128
    _, a_desc = sdfg.add_array(name="A",
                               shape=(10000, ),
                               dtype=dtypes.float64,
                               storage=dtypes.StorageType.Register,
                               transient=True,
                               alignment=array_a_alignment)
    state = sdfg.add_state("state", is_start_block=True)
    read = state.add_access("A")
    tasklet = state.add_tasklet("dummy", {"a"}, {}, "a = 1")
    state.add_memlet_path(read, tasklet, dst_conn="a", memlet=Memlet("A[0]"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Generate code for the program by traversing the SDFG state by state
        program_objects = codegen.generate_code(sdfg)

        # Assert that we get the expected warning message
        assert w
        assert any("was allocated on the heap instead of" in str(warn.message) for warn in w)

        # In code, assert that we allocate _and_ deallocate on the heap
        code = program_objects[0].clean_code
        # Consult the active cpp_standard: C++ >= 17 emits the aligned
        # new/delete forms, earlier standards the plain ones.
        if use_aligned_operator_new(a_desc):
            assert f"A = new (std::align_val_t({array_a_alignment})) double" in code, "A is allocated on the heap."
            assert f"::operator delete[](A, std::align_val_t({array_a_alignment}))" in code, "A is deallocated from the heap."
        else:
            assert "A = new double" in code, "A is allocated on the heap."
            assert "delete[] A" in code, "A is deallocated from the heap."


def test_at_multiplies_the_coordinate_by_the_array_stride():
    # A strided range: the offset is coordinate * array stride, with no rational division to cancel.
    assert Range([(0, 19, 2)]).at([1], [4]) == 8


def test_pointer_argument_keeps_a_decimal_literal():
    # The dot-to-arrow rewrite for struct members must not reach a decimal literal in the index
    # expression a pointer argument carries: `&A[(0.5 * j)]` became `&A[(0->5 * j)]`.
    N = symbol('N')
    nsdfg = SDFG('inner')
    nsdfg.add_array('a', [N], dtypes.float64)
    nstate = nsdfg.add_state()
    tasklet = nstate.add_tasklet('z', {}, {'o'}, 'o = 1.0')
    nstate.add_edge(tasklet, 'o', nstate.add_write('a'), None, Memlet('a[0]'))

    sdfg = SDFG('pointer_decimal')
    sdfg.add_symbol('N', dtypes.int64)
    sdfg.add_array('A', [N], dtypes.float64)
    state = sdfg.add_state()
    entry, exit = state.add_map('m', dict(j='0:N'))
    nsdfg_node = state.add_nested_sdfg(nsdfg, {}, {'a'}, symbol_mapping=dict(N='N', j='j'))
    state.add_nedge(entry, nsdfg_node, Memlet())
    state.add_memlet_path(nsdfg_node, exit, state.add_write('A'), src_conn='a', memlet=Memlet('A[0.5*j]'))

    # Only the legacy generator keeps the nested SDFG as a call, so only there does a pointer
    # argument carry the index. experimental_readable inlines it and indexes through A_idx.
    with config.set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
        code = codegen.generate_code(sdfg)[0].clean_code
    assert '&A[(0.5 * j)]' in code
    assert '0->5' not in code

    # The rewrite must leave the literal alone on the default path too, wherever it lands.
    assert '0->5' not in codegen.generate_code(sdfg)[0].clean_code


def test_at_multiplies_the_coordinate_by_the_array_stride():
    # A strided range: the offset is coordinate * array stride, with no rational division to cancel.
    assert Range([(0, 19, 2)]).at([1], [4]) == 8


if __name__ == '__main__':
    test_reshape_strides_multidim_array_all_dims_unit()
    test_reshape_strides_multidim_array_some_dims_unit()
    test_reshape_strides_multidim_array_different_shape()
    test_reshape_strides_from_strided_range()
    test_reshape_strides_from_strided_and_offset_range()

    test_arrays_bigger_than_max_stack_size_get_deallocated()

    test_at_multiplies_the_coordinate_by_the_array_stride()
    test_pointer_argument_keeps_a_decimal_literal()
