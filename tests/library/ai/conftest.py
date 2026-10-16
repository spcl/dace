# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared fixtures for the AI library node tests."""

import pytest

from tests.library.ai.ai_test_utils import offline


@pytest.fixture(autouse=True)
def isolated_build_folder(request, tmp_path):
    """
    Runs offline tests with sessions and the answer cache off, in a temporary build folder root.
    Tests marked ``ai`` call a real provider and keep the defaults, so their answers are cached.
    """
    if request.node.get_closest_marker("ai") is not None:
        yield
        return

    with offline(str(tmp_path)):
        yield
