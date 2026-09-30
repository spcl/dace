# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that validation rejects one symbol name carried at two dtypes inside a map range or memlet. """
import pytest

import dace
from dace import subsets
from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGNodeError

NARROW = dace.symbol('N', dace.int32)
WIDE = dace.symbol('N', dace.int64)
WIDE_POSITIVE = dace.symbol('N', dace.int64, positive=True)


def copy_sdfg(src_range: subsets.Range, map_range: subsets.Range) -> dace.SDFG:
    sdfg = dace.SDFG('mixed_symbol_dtypes')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [WIDE], dace.float64)
    sdfg.add_array('B', [WIDE], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': map_range})
    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
    state.add_memlet_path(state.add_read('A'),
                          entry,
                          tasklet,
                          dst_conn='a',
                          memlet=dace.Memlet(data='A', subset=src_range))
    state.add_memlet_path(tasklet, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    return sdfg


def test_one_dtype_per_name_validates():
    copy_sdfg(subsets.Range([(0, WIDE - 1, 1)]), subsets.Range([(0, WIDE - 1, 1)])).validate()


def test_default_dtype_parse_of_a_declared_symbol_validates():
    copy_sdfg(subsets.Range.from_string('0:N'), subsets.Range.from_string('0:N')).validate()


def test_map_range_mixing_two_dtypes_of_one_name_is_invalid():
    sdfg = copy_sdfg(subsets.Range([(0, WIDE - 1, 1)]), subsets.Range([(0, WIDE - NARROW, 1)]))
    with pytest.raises(InvalidSDFGNodeError, match='symbol N appears with dtypes'):
        sdfg.validate()


def test_memlet_mixing_two_dtypes_of_one_name_is_invalid():
    sdfg = copy_sdfg(subsets.Range([(NARROW - WIDE, WIDE - 1, 1)]), subsets.Range([(0, WIDE - 1, 1)]))
    with pytest.raises(InvalidSDFGEdgeError, match='symbol N appears with dtypes'):
        sdfg.validate()


def test_map_range_mixing_two_assumption_sets_of_one_name_is_invalid():
    sdfg = copy_sdfg(subsets.Range([(0, WIDE - 1, 1)]), subsets.Range([(0, WIDE - WIDE_POSITIVE, 1)]))
    with pytest.raises(InvalidSDFGNodeError, match='symbol N appears with assumptions'):
        sdfg.validate()


def test_memlet_mixing_two_assumption_sets_of_one_name_is_invalid():
    sdfg = copy_sdfg(subsets.Range([(WIDE_POSITIVE - WIDE, WIDE - 1, 1)]), subsets.Range([(0, WIDE - 1, 1)]))
    with pytest.raises(InvalidSDFGEdgeError, match='symbol N appears with assumptions'):
        sdfg.validate()


if __name__ == '__main__':
    test_one_dtype_per_name_validates()
    test_default_dtype_parse_of_a_declared_symbol_validates()
    test_map_range_mixing_two_dtypes_of_one_name_is_invalid()
    test_memlet_mixing_two_dtypes_of_one_name_is_invalid()
    test_map_range_mixing_two_assumption_sets_of_one_name_is_invalid()
    test_memlet_mixing_two_assumption_sets_of_one_name_is_invalid()
