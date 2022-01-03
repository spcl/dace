# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
from dace.transformation.dataflow import MapTiling
from dace.transformation.optimizer import SDFGOptimizer
import numpy as np
from scipy import ndimage

W = dace.symbol('W')
H = dace.symbol('H')
MAXITER = dace.symbol('MAXITER')


def create_sdfg():

    sdfg = dace.SDFG('stencil_sdfg_api')
    sdfg.add_symbol('MAXITER', MAXITER.dtype)
    _, arr = sdfg.add_array('A', (H, W), dace.float32)
    _, tmparr = sdfg.add_transient('tmp', (H, W), dace.float32)

    init = sdfg.add_state('init')
    guard = sdfg.add_state('guard')
    body = sdfg.add_state('body')
    end = sdfg.add_state('end')

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': '0'}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition='i<MAXITER'))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={'i': 'i+1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i>=MAXITER'))

    init.add_mapped_tasklet('reset_tmp', {
        'y': '0:W',
        'x': '0:H'
    }, {},
                            'out = dace.float32(0)', {'out': dace.Memlet.simple('tmp', 'y, x')},
                            external_edges=True)

    inp = body.add_read('A')
    tmp = body.add_access('tmp')
    out = body.add_write('A')
    me1, mx1 = body.add_map('stencil1', {'y': '1:W-1', 'x': '1:H-1'})
    task1 = body.add_tasklet('stencil1', {'n', 's', 'w', 'e', 'c'}, {'o'},
                             'o = (n + s + w + e + c) * dace.float32(0.2)')
    body.add_nedge(inp, me1, dace.Memlet.from_array('A', arr))
    body.add_edge(me1, None, task1, 'n', dace.Memlet.simple('A', 'y-1, x'))
    body.add_edge(me1, None, task1, 's', dace.Memlet.simple('A', 'y+1, x'))
    body.add_edge(me1, None, task1, 'w', dace.Memlet.simple('A', 'y, x-1'))
    body.add_edge(me1, None, task1, 'e', dace.Memlet.simple('A', 'y, x+1'))
    body.add_edge(me1, None, task1, 'c', dace.Memlet.simple('A', 'y, x'))
    body.add_edge(task1, 'o', mx1, None, dace.Memlet.simple('tmp', 'y, x'))
    body.add_nedge(mx1, tmp, dace.Memlet.simple('tmp', '1:H-1, 1:W-1'))
    me2, mx2 = body.add_map('stencil2', {'y': '1:W-1', 'x': '1:H-1'})
    task2 = body.add_tasklet('stencil2', {'n', 's', 'w', 'e', 'c'}, {'o'},
                             'o = (n + s + w + e + c) * dace.float32(0.2)')
    body.add_nedge(tmp, me2, dace.Memlet.from_array('tmp', tmparr))
    body.add_edge(me2, None, task2, 'n', dace.Memlet.simple('tmp', 'y-1, x'))
    body.add_edge(me2, None, task2, 's', dace.Memlet.simple('tmp', 'y+1, x'))
    body.add_edge(me2, None, task2, 'w', dace.Memlet.simple('tmp', 'y, x-1'))
    body.add_edge(me2, None, task2, 'e', dace.Memlet.simple('tmp', 'y, x+1'))
    body.add_edge(me2, None, task2, 'c', dace.Memlet.simple('tmp', 'y, x'))
    body.add_edge(task2, 'o', mx2, None, dace.Memlet.simple('A', 'y, x'))
    body.add_nedge(mx2, out, dace.Memlet.simple('A', '1:H-1, 1:W-1'))

    return sdfg, body


def test():
    W.set(1024)
    H.set(1024)
    MAXITER.set(30)

    print('Jacobi 5-point Stencil %dx%d (%d steps)' % (W.get(), H.get(), MAXITER.get()))

    A = np.ndarray((H.get(), W.get()), dtype=np.float32)

    # Initialize arrays: Randomize A, zero B
    A[:] = dace.float32(0)
    A[1:H.get() - 1, 1:W.get() - 1] = np.random.rand((H.get() - 2), (W.get() - 2)).astype(dace.float32.type)
    regression = np.ndarray([H.get() - 2, W.get() - 2], dtype=np.float32)
    regression[:] = A[1:H.get() - 1, 1:W.get() - 1]

    #print(A.view(type=np.ndarray))

    #############################################
    # Run DaCe program

    sdfg, body = create_sdfg()
    sdfg.fill_scope_connectors()
    sdfg.apply_transformations(MapTiling, states=[body])
    for node in body.nodes():
        if (isinstance(node, dace.sdfg.nodes.MapEntry) and node.label[:-2] == 'stencil'):
            assert len(body.in_edges(node)) <= 1

    sdfg(A=A, H=H.get(), W=W.get(), MAXITER=MAXITER.get())

    # Regression
    kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]], dtype=np.float32)
    for i in range(2 * MAXITER.get()):
        regression = ndimage.convolve(regression, kernel, mode='constant', cval=0.0)

    residual = np.linalg.norm(A[1:H.get() - 1, 1:W.get() - 1] - regression) / (H.get() * W.get())
    print("Residual:", residual)

    assert residual <= 0.05


if __name__ == "__main__":
    test()
