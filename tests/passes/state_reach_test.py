# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import dace
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation import passes


def test_loop_scope_reach():
    sdfg = SDFG('loop_scope_reach_test')
    s1 = sdfg.add_state('s1')
    s6 = sdfg.add_state('s6')
    (_, loop1, _) = sdfg.add_loop(s1, s6, 'i', 'i=0', 'i<10', 'i+1')
    loop1.label = 'loop1'
    s2 = loop1.add_state('s2')
    s5 = loop1.add_state('s5')
    (_, loop2, _) = loop1.add_loop(s2, s5, 'j', 'j=0', 'j<10', 'j+1')
    loop2.label = 'loop2'
    s3 = loop2.add_state('s3')
    s4 = loop2.add_state('s4')
    loop2.add_edge(s3, s4, dace.InterstateEdge())

    res = {}
    ppl.Pipeline([passes.analysis.LegacyStateReachability()]).apply_pass(sdfg, res)

    reach = res[passes.analysis.LegacyStateReachability.__name__][0]
    assert reach[s1] == {s2, s3, s4, s5, s6}
    assert reach[s2] == {s2, s3, s4, s5, s6}
    assert reach[s3] == {s2, s3, s4, s5, s6}
    assert reach[s4] == {s2, s3, s4, s5, s6}
    assert reach[s5] == {s2, s3, s4, s5, s6}
    assert len(reach[s6]) == 0


if __name__ == '__main__':
    test_loop_scope_reach()
