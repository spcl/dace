# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

vec3d = dace.struct('vec3d', x=dace.float32, y=dace.float32, z=dace.float32)

sdfg = dace.SDFG('sred')
sdfg.add_array('A', [1], vec3d)

state = sdfg.add_state()
t = state.add_tasklet('sredtask', {}, {'a'}, 'a = vec3d(x=float(1.0), y=float(2.0), z=float(3.0))')
a = state.add_write('A')
state.add_edge(t, 'a', a, None,
               dace.Memlet.simple('A', '0', wcr_str='lambda a, b: vec3d(x=a.x + b.x, y=a.y + b.y, z=a.z + b.z)'))


def test():
    inout = np.ndarray([1], dtype=np.dtype(vec3d.as_ctypes()))
    inout[0] = (4.0, 5.0, 6.0)

    sdfg(A=inout)

    expected = (5.0, 7.0, 9.0)
    diff = tuple(abs(x - y) for x, y in zip(inout[0], expected))

    print('Difference:', diff)
    assert all(d <= 1e-5 for d in diff)


if __name__ == "__main__":
    test()
