import pytest
import sympy as sp
import dace
from dace import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
import dace.dtypes as dtypes

from dace.sdfg.performance_evaluation.total_volume import analyze_sdfg

def make_copy_sdfg(name: str, shape, dtype) -> SDFG:
    """Minimal SDFG that copies array A into array B."""
    sdfg = SDFG(name)
    sdfg.add_array('A', shape=shape, dtype=dtype)
    sdfg.add_array('B', shape=shape, dtype=dtype)
    state = sdfg.add_state('s0')
    a = state.add_read('A')
    b = state.add_write('B')
    state.add_nedge(a, b, dace.Memlet(f'A[{",".join(f"0:{s}" for s in shape)}]'))
    return sdfg

class TestAnalyzeSdfg:

    def test_empty_sdfg(self):
        sdfg = SDFG('empty')
        sdfg.add_state('s0')
        read, write = analyze_sdfg(sdfg)
        assert sp.simplify(read) == 0
        assert sp.simplify(write) == 0

    def test_returns_sympy(self):
        sdfg = SDFG('empty')
        sdfg.add_state('s0')
        read, write = analyze_sdfg(sdfg)
        assert isinstance(read, sp.Basic)
        assert isinstance(write, sp.Basic)

    def test_copy_float64(self):
        """Copying 8 float64s: expect 64 bytes read and 64 bytes written."""
        sdfg = make_copy_sdfg('copy_f64', [8], dace.float64)
        read, write = analyze_sdfg(sdfg)
        assert sp.simplify(read - 64) == 0
        assert sp.simplify(write - 64) == 0

    def test_copy_float32_half_bytes(self):
        """float32 should produce half the volume of float64 for same shape."""
        sdfg64 = make_copy_sdfg('copy_f64', [16], dace.float64)
        sdfg32 = make_copy_sdfg('copy_f32', [16], dace.float32)
        r64, w64 = analyze_sdfg(sdfg64)
        r32, w32 = analyze_sdfg(sdfg32)
        assert sp.simplify(r64 - 2 * r32) == 0
        assert sp.simplify(w64 - 2 * w32) == 0

    def test_read_write_symmetry_on_copy(self):
        """A pure copy should read and write the same volume."""
        sdfg = make_copy_sdfg('copy_sym', [32], dace.float64)
        read, write = analyze_sdfg(sdfg)
        assert sp.simplify(read - write) == 0

    def test_two_independent_copies(self):
        """Two sequential copy states should double the volume."""
        sdfg = SDFG('two_copies')
        sdfg.add_array('A', shape=[8], dtype=dace.float64)
        sdfg.add_array('B', shape=[8], dtype=dace.float64)
        sdfg.add_array('C', shape=[8], dtype=dace.float64)

        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        sdfg.add_edge(s0, s1, dace.InterstateEdge())

        for state, src, dst in [(s0, 'A', 'B'), (s1, 'B', 'C')]:
            state.add_nedge(state.add_read(src), state.add_write(dst),
                            dace.Memlet(f'{src}[0:8]'))

        read, write = analyze_sdfg(sdfg)
        assert sp.simplify(read - 128) == 0
        assert sp.simplify(write - 128) == 0

    def test_symbolic_shape(self):
        """An SDFG with a symbolic dimension N should return a symbolic volume."""
        sdfg = SDFG('sym_shape')
        N = dace.symbol('N', dace.int32)
        sdfg.add_array('A', shape=[N], dtype=dace.float64)
        sdfg.add_array('B', shape=[N], dtype=dace.float64)
        state = sdfg.add_state('s0')
        state.add_nedge(state.add_read('A'), state.add_write('B'),
                        dace.Memlet('A[0:N]'))
        read, write = analyze_sdfg(sdfg)
        assert 8*N - read == 0 
        assert 8*N - write == 0

    def test_view_access_node_excluded(self):
        """Volumes from View arrays should not be counted."""
        sdfg = SDFG('view_test')
        sdfg.add_array('A', shape=[16], dtype=dace.float64)
        sdfg.add_view('V', shape=[8], dtype=dace.float64)
        sdfg.add_array('B', shape=[8], dtype=dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_read('A')
        v = state.add_access('V')
        b = state.add_write('B')

        # A -> V (view, should be ignored) -> B
        state.add_nedge(a, v, dace.Memlet('A[0:8]'))
        state.add_nedge(v, b, dace.Memlet('V[0:8]'))

        read, write = analyze_sdfg(sdfg)
        # V is a View so its edges must not contribute a second time —
        # the read volume should only reflect A, not A + V
        assert sp.simplify(read - 64) == 0   # 8 elements * 8 bytes from A
        assert sp.simplify(write - 64) == 0   # 8 elements * 8 bytes into B


    def test_map_doubles_volume(self):
        """A map over 2 iterations should double the access volume."""
        sdfg = SDFG('map_test')
        sdfg.add_array('A', shape=[2, 8], dtype=dace.float64)
        sdfg.add_array('B', shape=[2, 8], dtype=dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_read('A')
        b = state.add_write('B')

        me, mx = state.add_map('outer', {'i': '0:2'})
        t = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')

        state.add_memlet_path(a, me, t, memlet=dace.Memlet('A[i, 0:8]'), dst_conn='inp')
        state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[i, 0:8]'), src_conn='out')

        read, write = analyze_sdfg(sdfg)
        # 2 iterations × 8 elements × 8 bytes = 128 bytes each
        assert sp.simplify(read - 128) == 0
        assert sp.simplify(write - 128) == 0

    def test_loop_multiplies_volume(self):
        """A loop region iterating N times should scale the volume by N."""
        sdfg = SDFG('loop_test')
        N = dace.symbol('N', dace.int32)
        sdfg.add_array('A', shape=[8], dtype=dace.float64)
        sdfg.add_array('B', shape=[8], dtype=dace.float64)

        loop = LoopRegion('loop', condition_expr='i < N',
                        loop_var='i', initialize_expr='i = 0',
                        update_expr='i = i + 1', inverted=False,
                        sdfg=sdfg)
        sdfg.add_node(loop)
        sdfg.start_block = sdfg.node_id(loop) 

        body = loop.add_state('body')
        body.add_nedge(body.add_read('A'), body.add_write('B'),
                    dace.Memlet('A[0:8]'))

        read, write = analyze_sdfg(sdfg)
        # 8 elements × 8 bytes × N iterations = 64*N bytes
        expected = 64 * N
        assert sp.simplify(read - expected) == 0
        assert sp.simplify(write - expected) == 0

    def test_jacobi_1d(self):
        """Read and Write Volume of jacobi1d should be (TSTEPS-1)*16*N"""
        N = dace.symbol('N', dtype=dace.int64)
        @dace.program
        def jacobi_1d(TSTEPS: dace.int64, A: dace.float64[N], B: dace.float64[N]):

            for t in range(1, TSTEPS):
                B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
                A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
        sdfg = jacobi_1d.to_sdfg()

        read, write = analyze_sdfg(sdfg)

        expected_read = N*(dace.symbol('TSTEPS', dtype=dace.int64)-1)*16
        expected_write = (N-2)*(dace.symbol('TSTEPS', dtype=dace.int64)-1)*16
        assert read-expected_read == 0
        assert write-expected_write == 0

    def test_jacobi_1d(self):
        """Test read and write volume of jacobi 1d kernel"""
        N = dace.symbol('N', dtype=dace.int64)
        @dace.program
        def jacobi_1d(TSTEPS: dace.int64, A: dace.float64[N], B: dace.float64[N]):

            for t in range(1, TSTEPS):
                B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
                A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
        sdfg = jacobi_1d.to_sdfg()

        read, write = analyze_sdfg(sdfg)

        expected_read = N*(dace.symbol('TSTEPS', dtype=dace.int64)-1)*16
        expected_write = (N-2)*(dace.symbol('TSTEPS', dtype=dace.int64)-1)*16
        assert read-expected_read == 0
        assert write-expected_write == 0

    def test_jacobi_2d(self):
        """Test read and write volume of jacobi 2d kernel"""
        N = dace.symbol('N', dtype=dace.int64)

        @dace.program
        def jacobi_2d(TSTEPS: dace.int64, A: dace.float64[N, N], B: dace.float64[N, N]):

            for t in range(1, TSTEPS):
                B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                                    A[2:, 1:-1] + A[:-2, 1:-1])
                A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                                    B[2:, 1:-1] + B[:-2, 1:-1])
                
        sdfg = jacobi_2d.to_sdfg()

        read, write = analyze_sdfg(sdfg)

        expected_read = N**2*(dace.symbol('TSTEPS', dtype=dace.int64)-1)*16
        expected_write = (N-2)**2*(dace.symbol('TSTEPS', dtype=dace.int64)-1)*16
        assert read-expected_read == 0
        assert write-expected_write == 0

