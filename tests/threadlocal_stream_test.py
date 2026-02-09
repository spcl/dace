# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


def test():
    """Thread-local stream test"""

    N = dp.symbol('N')
    sdfg = SDFG('tlstream')
    sdfg.add_transient('la', [10], dp.float32)
    sdfg.add_stream('ls', dp.float32, 1, transient=True)
    sdfg.add_stream('gs', dp.float32, 1, transient=True)
    sdfg.add_array('ga', [N], dp.float32)
    state = sdfg.add_state('doit')

    localarr = state.add_access('la')
    localstream = state.add_access('ls')
    globalstream = state.add_access('gs')
    globalarr = state.add_access('ga')

    me, mx = state.add_map('par', dict(i='0:N'))
    tasklet = state.add_tasklet('arange', set(), {'a'}, 'a = i')

    state.add_nedge(me, tasklet, Memlet())
    state.add_edge(tasklet, 'a', localstream, None, Memlet.from_array(localstream.data, localstream.desc(sdfg)))
    state.add_nedge(localstream, localarr, Memlet.from_array(localarr.data, localarr.desc(sdfg)))
    state.add_nedge(localarr, mx, Memlet.from_array(globalstream.data, globalstream.desc(sdfg)))
    state.add_nedge(mx, globalstream, Memlet.from_array(globalstream.data, globalstream.desc(sdfg)))
    state.add_nedge(globalstream, globalarr, Memlet.from_array(globalarr.data, globalarr.desc(sdfg)))

    sdfg.fill_scope_connectors()

    N = 20
    output = np.ndarray([N], dtype=np.float32)
    output[:] = dp.float32(0)

    code_nonspec = sdfg.generate_code()

    assert 'Threadlocal' in code_nonspec[0].code

    func = sdfg.compile()
    func(ga=output, N=N)

    output = np.sort(output)

    diff = np.linalg.norm(output - np.arange(0, N, dtype=np.float32))
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
