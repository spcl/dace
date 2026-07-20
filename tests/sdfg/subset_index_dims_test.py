# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""``Range`` records which dimensions were written as an integer index.

``(begin, end, step)`` cannot express the difference between ``a[0]`` and ``a[0:1]`` -- both are
``(0, 0, 1)`` -- but numpy distinguishes them: an integer index removes its dimension from the
result rank, a slice never does, not even at extent 1. Consumers that need the rank of an access
(``x @ y`` dispatching to a matrix or a vector product, for instance) previously had to guess by
squeezing every extent-1 dimension, which is a different operation (``np.squeeze``) and is wrong
whenever a sliced dimension happens to have extent 1.
"""
import copy

import numpy as np

import dace
from dace import subsets


def test_index_reduces_rank_slice_does_not():
    indexed = subsets.Range.from_string('0:NQ, 0, 0:NP')
    sliced = subsets.Range.from_string('0:NQ, 0:1, 0:NP')

    # Identical extents, so neither size() nor equality can tell them apart ...
    assert indexed.size() == sliced.size()
    assert indexed == sliced
    # ... but the rank differs, which is what numpy cares about.
    assert indexed.rank() == 2
    assert sliced.rank() == 3
    assert indexed.rank_dims() == [0, 2]
    assert sliced.rank_dims() == [0, 1, 2]


def test_unit_extent_slice_keeps_its_rank():
    """The case squeezing cannot get right: a sliced dimension whose extent happens to be 1."""
    subset = subsets.Range.from_string('0:1, 0, 0:7')
    assert subset.size() == [1, 1, 7]
    assert subset.rank() == 2  # only the middle dimension was indexed away
    assert subset.rank_dims() == [0, 2]

    squeezed = copy.deepcopy(subset)
    squeezed.squeeze()
    assert squeezed.size() == [7]  # np.squeeze drops both, which is a different question


def test_squeeze_is_still_numpy_squeeze():
    """``squeeze`` keeps meaning ``np.squeeze``: drop every extent-1 dimension."""
    subset = subsets.Range.from_string('0:4, 0:1, 0:7')
    assert subset.rank() == 3
    subset.squeeze()
    assert subset.size() == [4, 7]


def test_from_array_preserves_rank():
    """Reading all of an array is never a rank reduction, even for its unit dimensions."""
    sdfg = dace.SDFG('from_array_rank')
    sdfg.add_array('V', [6, 1, 5], dace.float64)
    subset = subsets.Range.from_array(sdfg.arrays['V'])
    assert subset.rank() == 3
    assert not any(subset.index_dims)


def test_from_indices_is_all_indices():
    subset = subsets.Range.from_indices([2, 3])
    assert subset.rank() == 0
    assert all(subset.index_dims)


def test_string_form_is_unchanged():
    """The rendering stays exactly as it was.

    Subset strings are re-parsed as symbolic expressions in places that reject slice syntax (see
    ``dace.transformation.passes.scalar_to_symbol``), so a degenerate range must keep printing as
    ``i``. The flags therefore live in memory and in JSON, not in the string form.
    """
    assert str(subsets.Range.from_string('0:NQ, 0, 0:NP')) == '0:NQ, 0, 0:NP'
    assert str(subsets.Range.from_string('0:NQ, 0:1, 0:NP')) == '0:NQ, 0, 0:NP'
    assert str(subsets.Range.from_string('0, 0:10')) == '0, 0:10'


def test_frontend_records_mixed_subscript_ranks():
    """``A[0, 0:N]`` is rank 1 in numpy, so the frontend has to record which dimension was indexed."""
    from dace.frontend.python import memlet_parser

    subset = memlet_parser._ndslice_to_subset([0, (0, 9, 1)])
    assert subset.index_dims == [True, False]
    assert subset.rank() == 1

    both_slices = memlet_parser._ndslice_to_subset([(0, 0, 1), (0, 9, 1)])
    assert both_slices.index_dims == [False, False]
    assert both_slices.rank() == 2


def test_json_marks_only_index_dims():
    subset = subsets.Range.from_string('0:NQ, 0, 0:NP')
    entries = subset.to_json()['ranges']
    assert [entry.get('indexed', False) for entry in entries] == [False, True, False]


def test_sdfg_save_load_preserves_rank(tmp_path):
    sdfg = dace.SDFG('subset_rank_roundtrip')
    sdfg.add_array('A', [4, 1, 8], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    read, write = state.add_read('A'), state.add_write('A')
    # A[2, 0:1, 0:8]: dimension 0 is indexed away, dimension 1 is a unit slice that stays.
    state.add_nedge(read, write, dace.Memlet('A[2, 0:1, 0:8]'))
    assert state.edges()[0].data.subset.rank() == 2

    path = tmp_path / 'roundtrip.sdfg'
    sdfg.save(path)
    reloaded = dace.SDFG.from_file(path)
    subset = list(reloaded.states())[0].edges()[0].data.subset
    assert subset.index_dims == [True, False, False]
    assert subset.rank() == 2


def test_flags_survive_structural_operations():
    subset = subsets.Range.from_string('0:4, 0, 0:7, 2')
    assert subset.index_dims == [False, True, False, True]

    assert copy.deepcopy(subset).index_dims == subset.index_dims
    assert (subset + subsets.Range.from_string('0')).index_dims == [False, True, False, True, True]

    reordered = copy.deepcopy(subset)
    reordered.reorder([1, 0, 3, 2])
    assert reordered.index_dims == [True, False, True, False]

    popped = copy.deepcopy(subset)
    popped.pop([0])
    assert popped.index_dims == [True, False, True]

    unsqueezed = subsets.Range.from_string('0:10')
    unsqueezed.unsqueeze([0])
    assert unsqueezed.index_dims == [False, False]  # unsqueeze inserts slices, not indices


def test_derived_subsets_keep_their_flags():
    """Every operation that builds a new Range from an existing one must carry the flags over."""
    subset = subsets.Range.from_string('0:4, 0, 0:7')

    assert subset.offset_new(None, False).index_dims == [False, True, False]
    assert subset.offset_new([1, 0, 2], True).index_dims == [False, True, False]

    # compose: a degenerate dimension keeps its own flag, a consumed one takes the other subset's
    composed = subset.compose(subsets.Range.from_string('0:4, 0:7'))
    assert composed.index_dims == [False, True, False]
    assert composed.rank() == 2


def test_widening_a_dimension_clears_its_index_flag():
    """An index is degenerate by definition, so a widened dimension cannot stay flagged."""
    subset = subsets.Range.from_string('0:4, 0, 0:7')
    assert subset.rank() == 2

    subset[1] = (0, 3, 1)
    assert subset.rank() == 3
    assert not subset.is_index_dim(1)

    # Direct mutation of `ranges` bypasses __setitem__ entirely, so the flag is cross-checked
    # against the range rather than trusted.
    other = subsets.Range.from_string('0:4, 0, 0:7')
    other.ranges[1] = (0, 5, 1)
    assert other.rank() == 3


def test_rank_reducing_slice_still_lowers_correctly():
    """End-to-end guard: an integer index into a larger dimension still yields a 1D result."""
    R, C = (dace.symbol(s, dtype=dace.int64) for s in ('R', 'C'))

    @dace.program
    def take_row(A: dace.float64[R, C], out: dace.float64[C]):
        out[:] = A[2, :]

    rng = np.random.default_rng(0)
    A = rng.random((4, 8))
    out = np.zeros(8)
    take_row(A, out, R=4, C=8)
    assert np.allclose(out, A[2])


if __name__ == '__main__':
    test_index_reduces_rank_slice_does_not()
    test_unit_extent_slice_keeps_its_rank()
    test_squeeze_is_still_numpy_squeeze()
    test_from_array_preserves_rank()
    test_from_indices_is_all_indices()
    test_string_form_is_unchanged()
    test_frontend_records_mixed_subscript_ranks()
    test_json_marks_only_index_dims()
    test_flags_survive_structural_operations()
    test_derived_subsets_keep_their_flags()
    test_widening_a_dimension_clears_its_index_flag()
    test_rank_reducing_slice_still_lowers_correctly()
