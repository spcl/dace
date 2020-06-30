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
    sdfg.add_array('X', [10, 10], dace.float32)
    sdfg.add_array('F', [3, 3], dace.float32)
    sdfg.add_array('Y', [9, 9], dace.float32)
    state = sdfg.add_state()

    x = state.add_read('X')
    f = state.add_read('F')
    node = cudnn.conv2d.Conv2D('conv2d', [1,1,10,10], [3,3,1,1], 0, 0, [1,1,1,1], [1,1,1,1], 'NCHW')
    node.implementation = "cudnn"
    y = state.add_write('Y')
    state.add_edge(x, None, node, 'x',
                   dace.Memlet.simple('X', '0:10, 0:10'))
    state.add_edge(f, None, node, 'f',
                   dace.Memlet.simple('F', '0:3, 0:3'))
    state.add_edge(node, 'y', y, None, dace.Memlet.simple('Y', '0:9,0:9'))

    # Run graph with default node value
    X = np.random.rand(10, 10)
    F = np.random.rand(3, 3)
    Y = np.random.rand(9, 9)
    from dace.transformation.interstate.gpu_transform_sdfg import GPUTransformSDFG
    sdfg.apply_transformations(GPUTransformSDFG)
    sdfg(X=X, F=F, Y=Y)

    print(Y)


if __name__ == '__main__':
    test_cudnn_code()
