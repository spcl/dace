# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests hooks that can be used to extend DaCe functionality.
"""

import dace
import numpy as np
from contextlib import contextmanager


def test_hooks():
    called_list = []

    @dace.program
    def tester():
        pass

    def before(_):
        called_list.append('before')

    def after(_):
        called_list.append('after')

    def before2(_):
        called_list.append('before2')

    def after2(_):
        called_list.append('after2')

    def before_csdfg(*args):
        called_list.append('before_csdfg')

    @contextmanager
    def ctxmgr(sdfg):
        called_list.append('ctxmgr-before-' + sdfg.name[-6:])
        yield
        called_list.append('ctxmgr-after')

    with dace.hooks.on_call(before=before, after=after):
        with dace.hooks.on_compiled_sdfg_call(before=before_csdfg):
            with dace.hooks.on_call(before=before2, after=after2):
                with dace.hooks.on_call(context_manager=ctxmgr):
                    tester()

    # Ensure hooks are called in the right order
    assert called_list == [
        'before', 'before2', 'ctxmgr-before-tester', 'before_csdfg', 'ctxmgr-after', 'after2', 'after'
    ]
    called_list.clear()

    # Ensure hooks were removed
    tester()
    assert len(called_list) == 0


def test_profile():
    @dace.program
    def test1(A: dace.float64[20]):
        return A + 1

    @dace.program
    def test2(A: dace.float64[20]):
        return A + 2

    A = np.random.rand(20)
    expected1 = A + 1
    expected2 = A + 2

    with dace.profile(repetitions=10) as prof:
        r2 = test2(A)
        r1 = test1(A)

    assert np.allclose(r1, expected1)
    assert np.allclose(r2, expected2)

    assert len(prof.times) == 2
    assert len(prof.times[0][1]) == 10
    assert len(prof.times[1][1]) == 10
    assert prof.times[0][0].name.endswith('test2')
    assert prof.times[1][0].name.endswith('test1')


if __name__ == '__main__':
    test_hooks()
    test_profile()
