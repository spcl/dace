import dace
import numpy as np

sdfg = dace.SDFG('addedgepair')
state = sdfg.add_state()

# Add nodes
t = state.add_tasklet('do', {'a'}, {'b'}, 'b = 2*a')
a = state.add_array('A', [31], dace.float64)
b = state.add_array('B', [1], dace.float64)
me, mx = state.add_map('m', dict(i='0:31'))

# Add edges
state.add_edge_pair(me,
                    t,
                    a,
                    dace.Memlet.simple(a, 'i'),
                    internal_connector='a')
state.add_edge_pair(mx,
                    t,
                    b,
                    dace.Memlet.simple(b, '0', wcr_str='lambda a,b: a+b'),
                    internal_connector='b',
                    scope_connector='o')

if __name__ == '__main__':
    A = np.random.rand(31).astype(np.float64)
    B = np.array([0.], dtype=np.float64)
    sdfg(A=A, B=B)

    diff = np.linalg.norm(B[0] - np.sum(2 * A))
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)
