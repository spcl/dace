# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The symbol-scope and allocation-scope passes must not change a single byte of generated code.

Both are pure lookup tables replacing scans, so the specification is exact equivalence. Comparing
the emitted C++ is stronger than comparing the tables: it also covers the allocation decisions
(which state declares a transient, which scope allocates it), which is what the allocation tables
feed and where an ordering mistake would show up.

The baseline is produced by monkeypatching the tables to empty, which makes ``defined_at`` fall
back to ``symbols_defined_at`` and forces the pre-pass code path.
"""

import dace
import pytest
from dace.codegen import codegen
from dace.sdfg.state import LoopRegion

N = dace.symbol('N')
M = dace.symbol('M')


def generated(sdfg: dace.SDFG) -> str:
    return '\n'.join(obj.clean_code for obj in codegen.generate_code(sdfg))


def assert_codegen_unchanged(build):
    """Generate with the passes live, then with the symbol table forced empty; compare."""
    import dace.codegen.targets.framecode as framecode

    with_passes = generated(build())

    original = framecode.DaCeCodeGenerator.determine_allocation_lifetime

    def no_symbol_table(self, top_sdfg):
        original(self, top_sdfg)
        self.symbol_scopes = {}  # force defined_at down its symbols_defined_at fallback

    framecode.DaCeCodeGenerator.determine_allocation_lifetime = no_symbol_table
    try:
        fallback = generated(build())
    finally:
        framecode.DaCeCodeGenerator.determine_allocation_lifetime = original

    assert with_passes == fallback, 'generated code differs between the pass and the fallback path'
    return with_passes


def test_nested_maps():

    def build():

        @dace.program
        def prog(A: dace.float64[N, M]):
            for i in dace.map[0:N]:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] * 2.0

        return prog.to_sdfg(simplify=False)

    assert len(assert_codegen_unchanged(build)) > 0


def test_transients_across_states():
    """Exercises the State-lifetime branch: a transient used in one state vs several."""

    def build():
        sdfg = dace.SDFG('multi')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_transient('once', [N], dace.float64, lifetime=dace.AllocationLifetime.State)
        sdfg.add_transient('twice', [N], dace.float64, lifetime=dace.AllocationLifetime.State)

        s1 = sdfg.add_state('s1', is_start_block=True)
        s1.add_nedge(s1.add_access('A'), s1.add_access('once'), dace.Memlet('A[0:N]'))
        s1.add_nedge(s1.add_access('once'), s1.add_access('twice'), dace.Memlet('once[0:N]'))

        s2 = sdfg.add_state('s2')
        sdfg.add_edge(s1, s2, dace.InterstateEdge())
        s2.add_nedge(s2.add_access('twice'), s2.add_access('B'), dace.Memlet('twice[0:N]'))
        return sdfg

    assert_codegen_unchanged(build)


def test_scope_lifetime_inside_map():
    """Exercises the Scope-lifetime branch and common_parent_scope over a cached scope dict."""

    def build():

        @dace.program
        def prog(A: dace.float64[N], B: dace.float64[N]):
            for i in dace.map[0:N]:
                tmp = A[i] * 2.0
                B[i] = tmp + 1.0

        return prog.to_sdfg(simplify=False)

    assert_codegen_unchanged(build)


def test_loop_region_states_are_reached():
    """States live inside control-flow regions, not only at the top level of the SDFG."""

    def build():
        sdfg = dace.SDFG('inloop')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_transient('acc', [1], dace.float64)
        loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state('body', is_start_block=True)
        tasklet = body.add_tasklet('w', {}, {'o'}, 'o = 1.0')
        body.add_edge(tasklet, 'o', body.add_access('acc'), None, dace.Memlet('acc[0]'))
        body.add_nedge(body.add_access('acc'), body.add_access('A'), dace.Memlet('acc[0] -> [i]'))
        return sdfg

    assert_codegen_unchanged(build)


@pytest.mark.parametrize('simplify', [False, True])
def test_nested_sdfg(simplify):

    def build():

        @dace.program
        def inner(A: dace.float64[N]):
            for i in dace.map[0:N]:
                A[i] = A[i] + 1.0

        @dace.program
        def outer(A: dace.float64[N]):
            inner(A)
            inner(A)

        return outer.to_sdfg(simplify=simplify)

    assert_codegen_unchanged(build)


if __name__ == '__main__':
    test_nested_maps()
    test_transients_across_states()
    test_scope_lifetime_inside_map()
    test_loop_region_states_are_reached()
    test_nested_sdfg(False)
    test_nested_sdfg(True)
