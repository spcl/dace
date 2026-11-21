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


def test_a_single_index_with_any_step_is_one_element():
    """``b:b:p`` holds one element whatever ``p`` is: a strided slice taken per iteration of a pooling window
    (``x[..., i*p + t : ... : p]``) is one element per point, yet its size read ``ceiling(1/p)``, so the tasklet
    connector became a pointer the body then used as a value (HB conv_transpose3d_avg_pool_clamp_softmax_multiply,
    CPF hip: "invalid operands to binary expression ('float' and 'const float *')")."""
    i, p = (symbolic.pystr_to_symbolic(name) for name in ("i", "p"))
    start = symbolic.pystr_to_symbolic("p * Max(0, i - 2) + int_floor(i, p)")
    subset = subsets.Range([(i * p + 1, i * p + 1, p), (i, i, -p), (start, start, p)])
    assert subset.size() == [1, 1, 1], subset.size()
    assert subset.num_elements() == 1


if __name__ == "__main__":
    test_an_index_string_parses_to_a_single_element_range()
    test_from_indices_builds_a_single_element_range_per_dimension()
    test_the_deprecated_indices_subset_type_is_gone()
    test_a_single_index_with_any_step_is_one_element()
