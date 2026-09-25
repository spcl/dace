# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A node added without explicit line info records its caller's line and file, as
``inspect.getframeinfo`` reports them, and resolves each call site only once."""
import inspect
import sys
from unittest import mock

import dace
from dace.sdfg import state as state_module


def add_tasklet_here(state: dace.SDFGState) -> tuple:
    """Adds a tasklet and returns it with the ``getframeinfo`` of the adding line."""
    here = inspect.getframeinfo(sys._getframe(0), context=0)
    tasklet = state.add_tasklet("t", {}, {"__out"}, "__out = 1")
    return tasklet, here


def test_debuginfo_is_the_callers_position():
    with dace.config.set_temporary("compiler", "lineinfo", value="inspect"):
        state = dace.SDFG("debuginfo_caller").add_state()
        tasklet, here = add_tasklet_here(state)
    assert tasklet.debuginfo.filename == here.filename
    assert tasklet.debuginfo.start_line == here.lineno + 1


def test_call_site_is_resolved_once():
    with dace.config.set_temporary("compiler", "lineinfo", value="inspect"):
        state = dace.SDFG("debuginfo_once").add_state()
        add_tasklet_here(state)
        original = inspect.getframeinfo
        with mock.patch.object(state_module.inspect, "getframeinfo", side_effect=original) as spy:
            tasklets = [add_tasklet_here(state)[0] for _ in range(5)]
    # Only the explicit call inside ``add_tasklet_here`` itself: the add_tasklet call site is cached.
    assert spy.call_count == 5
    assert len({(t.debuginfo.filename, t.debuginfo.start_line) for t in tasklets}) == 1
    # Every node owns its DebugInfo; nothing is shared through the cache.
    assert len({id(t.debuginfo) for t in tasklets}) == 5


if __name__ == "__main__":
    test_debuginfo_is_the_callers_position()
    test_call_site_is_resolved_once()
