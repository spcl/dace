# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared fixtures for the AI library node tests."""

import pytest

# The AI library needs the ``dace[agent]`` extras; without pydantic, skip this directory instead of failing collection
pytest.importorskip("pydantic", reason="the DaCe AI library needs pydantic (pip install 'dace[agent]')")

from tests.library.ai.ai_test_utils import offline  # noqa: E402


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
