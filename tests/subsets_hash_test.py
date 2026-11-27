# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A Range keeps its hash while its bounds stay the same objects, and follows a bound changed in place."""

import copy

import dace
from dace import subsets


def test_equal_ranges_hash_alike_and_an_edited_range_rehashes():
    first = subsets.Range.from_string("i:N, 0:M")
    second = copy.deepcopy(first)
    assert first == second and hash(first) == hash(second)
    before = hash(first)
    first.ranges[0] = (first.ranges[0][0] + 1, first.ranges[0][1], first.ranges[0][2])
    assert hash(first) != before and hash(first) == hash(subsets.Range.from_string("i + 1:N, 0:M"))
    # Same-named symbols with other assumptions compare equal, so they hash alike
    renamed = subsets.Range(
        [(dace.symbol("i", dace.int64, nonnegative=True), dace.symbol("N") - 1, 1), (0, "M - 1", 1)]
    )
    assert renamed == second and hash(renamed) == hash(second)


if __name__ == "__main__":
    test_equal_ranges_hash_alike_and_an_edited_range_rehashes()
