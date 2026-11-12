# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Resident blocks: the residency arithmetic, and where the device limits come from."""

import pytest

import dace
from dace.transformation.passes.gpu_specialization import resident_blocks as rb
from dace.transformation.passes.gpu_specialization.resident_blocks import DeviceLimits, device_limits


@pytest.mark.parametrize("block, expected", [(256, 228 * 8), (128, 228 * 16), (1024, 228 * 2)])
def test_resident_blocks_divide_the_threads_of_a_multiprocessor(block: int, expected: int):
    assert DeviceLimits(228, 2048, 32).resident_blocks(block) == expected


def test_the_blocks_per_multiprocessor_limit_binds_narrow_blocks():
    assert DeviceLimits(132, 2048, 32).resident_blocks(32) == 132 * 32


def test_an_unreported_block_limit_does_not_bind():
    assert DeviceLimits(110, 2048, 0).resident_blocks(64) == 110 * 32


def test_a_block_wider_than_a_multiprocessor_holds_still_gets_one_per_multiprocessor():
    assert DeviceLimits(80, 1024, 16).resident_blocks(2048) == 80


def test_the_visible_device_wins_over_the_defaults(monkeypatch):
    monkeypatch.setattr(rb, "query_device", lambda: DeviceLimits(228, 2048, 0))
    assert device_limits() == DeviceLimits(228, 2048, 0)


def test_a_configured_device_overrides_the_query(monkeypatch):
    monkeypatch.setattr(rb, "query_device", lambda: DeviceLimits(228, 2048, 0))
    with dace.config.set_temporary("compiler", "cuda", "device_multiprocessors", value=104):
        with dace.config.set_temporary("compiler", "cuda", "device_blocks_per_multiprocessor", value=4):
            assert rb.resident_blocks(256) == 104 * 4


def test_without_a_visible_device_the_fallback_count_is_used(monkeypatch):
    monkeypatch.setattr(rb, "query_device", lambda: None)
    with pytest.warns(UserWarning, match="No GPU visible"):
        assert device_limits().multiprocessors == rb.FALLBACK_MULTIPROCESSORS


@pytest.mark.gpu
def test_the_query_reads_the_visible_device():
    found = rb.query_device()
    assert found is not None and found.multiprocessors > 0 and found.threads_per_multiprocessor >= 1024
    print("device limits", found)


if __name__ == "__main__":
    test_resident_blocks_divide_the_threads_of_a_multiprocessor(256, 228 * 8)
    test_resident_blocks_divide_the_threads_of_a_multiprocessor(128, 228 * 16)
    test_resident_blocks_divide_the_threads_of_a_multiprocessor(1024, 228 * 2)
    test_the_blocks_per_multiprocessor_limit_binds_narrow_blocks()
    test_an_unreported_block_limit_does_not_bind()
    test_a_block_wider_than_a_multiprocessor_holds_still_gets_one_per_multiprocessor()
    test_the_query_reads_the_visible_device()
