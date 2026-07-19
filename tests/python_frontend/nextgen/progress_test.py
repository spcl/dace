# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Smoke tests for pipeline progress feedback: parsing works identically with
the progress configuration enabled (bars are threshold-gated, so fast parses
show nothing either way), and the statement ticker counts lowered statements.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def test_parse_with_progress_enabled():

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0

    with dace.config.set_temporary('progress', value=True):
        tree = nextgen.parse_program(prog)
    assert not [n for n in tree.preorder_traversal() if isinstance(n, tn.PythonCallbackNode)]

    func = tree.as_sdfg().compile()
    a = np.random.rand(5)
    b = np.zeros(5)
    func(a=a, b=b, N=5)
    assert np.allclose(b, a * 2.0)


def test_statement_ticker_counts():
    from dace.cli.progress import OptionalProgressBar

    ticks = []

    class _Recorder(OptionalProgressBar):

        def __init__(self, n, title=None):
            super().__init__(n=n, progress=False)

        def next(self):
            ticks.append(1)

    @dace.program
    def prog(a: dace.float64[N]):
        a[:] = a[:] + 1.0
        a[:] = a[:] * 2.0

    # build_schedule_tree constructs its bar from the name imported into the
    # nextgen package namespace; substitute the recorder there.
    original = nextgen.OptionalProgressBar
    nextgen.OptionalProgressBar = _Recorder
    try:
        nextgen.parse_program(prog)
    finally:
        nextgen.OptionalProgressBar = original
    assert len(ticks) == 2  # One tick per lowered top-level statement


if __name__ == '__main__':
    test_parse_with_progress_enabled()
    test_statement_ticker_counts()
