# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ScalarWriteShadowScopes`` scans only the blocks of the SDFG it is visiting when a nested SDFG reuses a container
name."""

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis import ScalarWriteShadowScopes


def test_shadow_scopes_with_a_name_shared_by_a_nested_sdfg():
    inner = dace.SDFG("inner")
    inner.add_scalar("s", dace.int64, transient=True)
    inner.add_array("A", (4,), dace.int64)
    first = inner.add_state("first", is_start_block=True)
    task = first.add_tasklet("set", {}, {"o"}, "o = 1")
    first.add_edge(task, "o", first.add_write("s"), None, dace.Memlet("s[0]"))
    second = inner.add_state_after(first, "second", assignments={"k": "s"})
    task = second.add_tasklet("use", {}, {"o"}, "o = k")
    second.add_edge(task, "o", second.add_write("A"), None, dace.Memlet("A[0]"))

    sdfg = dace.SDFG("shadow_scopes_nested")
    sdfg.add_scalar("s", dace.int64, transient=True)
    sdfg.add_array("A", (4,), dace.int64)
    start = sdfg.add_state("start", is_start_block=True)
    task = start.add_tasklet("set", {}, {"o"}, "o = 2")
    start.add_edge(task, "o", start.add_write("s"), None, dace.Memlet("s[0]"))
    body = sdfg.add_state_after(start, "body", assignments={"m": "s"})
    nsdfg = body.add_nested_sdfg(inner, {}, {"A": None})
    body.add_edge(nsdfg, "A", body.add_write("A"), None, dace.Memlet("A[0:4]"))

    results = Pipeline([ScalarWriteShadowScopes()]).apply_pass(sdfg, {})
    assert ScalarWriteShadowScopes.__name__ in results
