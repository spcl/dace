# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Shared fixtures for the AI library node tests. """

import pytest

import dace


@pytest.fixture(autouse=True)
def isolated_from_the_users_ai_directories(tmp_path):
    """
    Keeps expansions in these tests out of the user's home directory.

    Transcripts and the answer cache are both on by default, since a real expansion is a paid call
    whose answer is worth keeping and reusing. Neither belongs in a test run: the transcripts would
    accumulate, and a cached stub answer would be served to a later test that expects its own
    provider to be called. Tests about either feature turn it back on explicitly, pointed at a
    temporary directory.
    """
    with dace.config.set_temporary('ai', 'transcripts', value=False):
        with dace.config.set_temporary('ai', 'transcript_dir', value=str(tmp_path / 'transcripts')):
            with dace.config.set_temporary('ai', 'cache', value=False):
                with dace.config.set_temporary('ai', 'cache_dir', value=str(tmp_path / 'cache')):
                    yield
