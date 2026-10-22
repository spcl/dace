# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace import subsets, symbolic
from dace.properties import SubsetProperty


def test_an_index_string_parses_to_a_single_element_range():
    i = symbolic.pystr_to_symbolic("i")
    subset = SubsetProperty.from_string("i, 0")
    assert type(subset) is subsets.Range, type(subset)
    assert subset.ranges == [(i, i, 1), (0, 0, 1)], subset.ranges


def test_from_indices_builds_a_single_element_range_per_dimension():
    subset = subsets.Range.from_indices([3, "j"])
    j = symbolic.pystr_to_symbolic("j")
    assert type(subset) is subsets.Range, type(subset)
    assert subset.ranges == [(3, 3, 1), (j, j, 1)], subset.ranges
    assert subset.num_elements() == 1


def test_the_deprecated_indices_subset_type_is_gone():
    assert "Indices" not in vars(subsets)


if __name__ == "__main__":
    test_an_index_string_parses_to_a_single_element_range()
    test_from_indices_builds_a_single_element_range_per_dimension()
    test_the_deprecated_indices_subset_type_is_gone()
