import dace
from dace.data import Data
from dace.properties import make_properties
from dace.libraries.standard.nodes import CodeLibraryNode
from typing import Dict
import numpy as np
import dace.libraries.cudnn as cudnn


def test_cudnn_code():
    # Construct graph
    sdfg = dace.SDFG('custom_code')
    sdfg.add_array('A', [20, 20], dace.float32)
    sdfg.add_array('B', [20, 20], dace.float32)
    sdfg.add_array('C', [20, 20], dace.float32)
    state = sdfg.add_state()

    a = state.add_read('A')
    b = state.add_read('B')
    node = cudnn.conv2d.Conv2D('conv2d', [1,1,20,20], [1,1,20,20], 0, 0, [1,1,1,1], [0,0,0,0], 'NCHW')
    node.implementation = "cudnn"
    c = state.add_write('C')
    state.add_edge(a, None, node, 'x',
                   dace.Memlet.simple('A', '0:20, 0:20'))
    state.add_edge(b, None, node, 'f',
                   dace.Memlet.simple('B', '0:20, 0:20'))
    state.add_edge(node, 'y', c, None, dace.Memlet.simple('C', '0:20,0:20'))

    # Run graph with default node value
    A = np.random.rand(20, 20)
    B = np.random.rand(20, 20)
    C = np.random.rand(20, 20)
    sdfg(A=A, B=B, C=C)


if __name__ == '__main__':
    test_cudnn_code()
