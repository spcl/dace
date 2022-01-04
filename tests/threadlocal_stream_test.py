# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

N = dp.symbol('N')
sdfg = SDFG('tlstream')
state = sdfg.add_state('doit')

localarr = state.add_transient('la', [10], dp.float32)
localstream = state.add_stream('ls', dp.float32, 1, transient=True)
globalstream = state.add_stream('gs', dp.float32, 1, transient=True)
globalarr = state.add_array('ga', [N], dp.float32)

me, mx = state.add_map('par', dict(i='0:N'))
tasklet = state.add_tasklet('arange', set(), {'a'}, 'a = i')

state.add_nedge(me, tasklet, Memlet())
state.add_edge(tasklet, 'a', localstream, None, Memlet.from_array(localstream.data, localstream.desc(sdfg)))
state.add_nedge(localstream, localarr, Memlet.from_array(localarr.data, localarr.desc(sdfg)))
state.add_nedge(localarr, mx, Memlet.from_array(globalstream.data, globalstream.desc(sdfg)))
state.add_nedge(mx, globalstream, Memlet.from_array(globalstream.data, globalstream.desc(sdfg)))
state.add_nedge(globalstream, globalarr, Memlet.from_array(globalarr.data, globalarr.desc(sdfg)))

sdfg.fill_scope_connectors()


def test():
    print('Thread-local stream test')

    N.set(20)

    output = np.ndarray([N.get()], dtype=np.float32)
    output[:] = dp.float32(0)

    code_nonspec = sdfg.generate_code()

    assert 'Threadlocal' in code_nonspec[0].code

    func = sdfg.compile()
    func(ga=output, N=N)

    output = np.sort(output)

    diff = np.linalg.norm(output - np.arange(0, N.get(), dtype=np.float32))
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
