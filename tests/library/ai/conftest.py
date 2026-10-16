# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared fixtures for the AI library node tests."""

import pytest

import dace


@pytest.fixture(autouse=True)
def isolated_build_folder(request, tmp_path):
    """
    Runs offline tests with sessions and the answer cache off, in a temporary build folder root.
    Tests marked ``ai`` call a real provider and keep the defaults, so their answers are cached.
    """
    if request.node.get_closest_marker("ai") is not None:
        yield
        return

    with dace.config.set_temporary("ai", "sessions", value=False):
        with dace.config.set_temporary("ai", "cache", value=False):
            with dace.config.set_temporary("default_build_folder", value=str(tmp_path)):
                yield
