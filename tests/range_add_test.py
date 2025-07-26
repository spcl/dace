# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.subsets import Range

import pytest


@pytest.fixture
def rng1() -> Range:
    rng1 = Range([(0, 3, 1, 4)])
    assert len(rng1) == 1
    assert rng1[0][0] == 0
    assert rng1[0][1] == 3
    assert rng1[0][2] == 1
    assert len(rng1.tile_sizes) == 1
    assert rng1.tile_sizes[0] == 4
    return rng1


@pytest.fixture
def rng2() -> Range:
    rng2 = Range.from_string("0:10:2")
    assert len(rng2) == 1
    assert rng2[0][0] == 0
    assert rng2[0][1] == 9
    assert rng2[0][2] == 2
    assert len(rng2.tile_sizes) == 1
    assert rng2.tile_sizes[0] == 1
    return rng2


@pytest.fixture
def rng3() -> Range:
    rng3 = Range([(1, 4, 1, 5), (77, 88, 2)])
    assert len(rng3) == 2
    assert rng3[0][0] == 1
    assert rng3[0][1] == 4
    assert rng3[0][2] == 1
    assert rng3[1][0] == 77
    assert rng3[1][1] == 88
    assert rng3[1][2] == 2
    assert len(rng3.tile_sizes) == 2
    assert rng3.tile_sizes[0] == 5
    assert rng3.tile_sizes[1] == 1
    return rng3


def _perform_check(res, exp) -> None:
    if isinstance(exp, list):
        exp = Range(exp)
    assert len(res) == len(exp)
    assert len(res) > 0

    for i in range(len(res)):
        assert exp.ranges[i][0] == res.ranges[i][0]
        assert exp.ranges[i][1] == res.ranges[i][1]
        assert exp.ranges[i][2] == res.ranges[i][2]
        assert exp.tile_sizes[i] == exp.tile_sizes[i]


def test_range_rng1_add_rng2(rng1, rng2):
    _perform_check(rng1 + rng2, [(0, 3, 1, 4), (0, 9, 2, 1)])


def test_range_rng2_add_rng1(rng1, rng2):
    _perform_check(rng2 + rng1, [(0, 9, 2, 1), (0, 3, 1, 4)])


def test_range_rng1_add_rng3(rng1, rng3):
    _perform_check(rng1 + rng3, [(0, 3, 1, 4), (1, 4, 1, 5), (77, 88, 2)])


def test_range_rng3_add_rng1(rng1, rng3):
    _perform_check(rng3 + rng1, [(1, 4, 1, 5), (77, 88, 2), (0, 3, 1, 4)])


def test_range_rng2_add_rng2(rng2):
    _perform_check(rng2 + rng2, [(0, 9, 2, 1), (0, 9, 2, 1)])


def test_range_rng1_add_rng1(rng1):
    _perform_check(rng1 + rng1, [(0, 3, 1, 4), (0, 3, 1, 4)])
