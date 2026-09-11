# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Shared fixtures for the AI library node tests. """

import pytest

import dace


@pytest.fixture(autouse=True)
def isolated_from_the_users_ai_directories(request, tmp_path):
    """
    Keeps offline tests out of the user's home directory, and leaves live tests alone.

    Sessions and the answer cache are both on by default, since a real expansion is a paid call
    whose answer is worth keeping and reusing. Neither belongs in an *offline* test run: the
    sessions would accumulate, and a cached stub answer would be served to a later test that
    expects its own provider to be called. Tests about either feature turn it back on explicitly,
    pointed at a temporary directory.

    A test marked ``ai`` calls a real provider, and for it the defaults are exactly right: the
    conversation belongs on disk where it can be read afterwards, and the cache is what makes a
    second run of an expensive test free. Isolating those would also break them outright, since
    ``history`` and ``rollback`` read the session the run just wrote.
    """
    if request.node.get_closest_marker('ai') is not None:
        yield
        return

    with dace.config.set_temporary('ai', 'sessions', value=False):
        with dace.config.set_temporary('ai', 'session_dir', value=str(tmp_path / 'sessions')):
            with dace.config.set_temporary('ai', 'cache', value=False):
                with dace.config.set_temporary('ai', 'cache_dir', value=str(tmp_path / 'cache')):
                    yield
