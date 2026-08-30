# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for reshaping and reinterpretation of existing arrays. """
import dace
import numpy as np
import pytest

N = dace.symbol('N')


def test_reshape():
    """ Array->View->Tasklet """

    @dace.program
    def reshp(A: dace.float64[2, 3, 4], B: dace.float64[8, 3]):
        C = np.reshape(A, [8, 3])
        B[:] += C

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)
    expected = np.reshape(A, [8, 3]) + B

    reshp(A, B)
    assert np.allclose(expected, B)


def test_reshape_dst():
    """ Tasklet->View->Array """

    @dace.program
    def reshpdst(A: dace.float64[2, 3, 4], B: dace.float64[8, 3]):
        C = np.reshape(B, [2, 3, 4])
        C[:] = A

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)

    reshpdst(A, B)
    assert np.allclose(A, np.reshape(B, [2, 3, 4]))


def test_reshape_dst_explicit():
    """ Tasklet->View->Array """
    sdfg = dace.SDFG('reshapedst')
    sdfg.add_array('A', [2, 3, 4], dace.float64)
    sdfg.add_view('Bv', [2, 3, 4], dace.float64)
    sdfg.add_array('B', [8, 3], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map('compute', dict(i='0:2', j='0:3', k='0:4'))
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    state.add_memlet_path(state.add_read('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i,j,k]'))
    v = state.add_access('Bv')
    state.add_memlet_path(t, mx, v, src_conn='b', memlet=dace.Memlet('Bv[i,j,k]'))
    state.add_nedge(v, state.add_write('B'), dace.Memlet('B'))
    sdfg.validate()

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)
    sdfg(A=A, B=B)
    assert np.allclose(A + 1, np.reshape(B, [2, 3, 4]))


@pytest.mark.parametrize('memlet_dst', (False, True))
def test_reshape_copy(memlet_dst):
    """
    Symmetric case of Array->View->Array. Should be translated to a reference
    and a copy.
    """
    sdfg = dace.SDFG('reshpcpy')
    sdfg.add_array('A', [2, 3], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    sdfg.add_view('Av', [6], dace.float64)
    state = sdfg.add_state()
    r = state.add_read('A')
    v = state.add_access('Av')
    w = state.add_write('B')
    state.add_edge(r, None, v, 'views', dace.Memlet(data='A'))
    state.add_nedge(v, w, dace.Memlet(data='B' if memlet_dst else 'Av'))
    sdfg.validate()

    A = np.random.rand(2, 3)
    B = np.random.rand(6)
    sdfg(A=A, B=B)
    assert np.allclose(A.reshape([6]), B)


def test_reshape_copy_scoped():
    """ Array->View->Array where one array is located within a map scope. """
    sdfg = dace.SDFG('reshpcpy')
    sdfg.add_array('A', [2, 3], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    sdfg.add_view('Av', [6], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()
    r = state.add_read('A')
    me, mx = state.add_map('reverse', dict(i='0:6'))
    v = state.add_access('Av')
    t = state.add_access('tmp')
    w = state.add_write('B')
    state.add_edge_pair(me, v, r, dace.Memlet('A[0:2, 0:3]'), dace.Memlet('A[0:2, 0:3]'))
    state.add_nedge(v, t, dace.Memlet('Av[i]'))
    state.add_memlet_path(t, mx, w, memlet=dace.Memlet('B[6 - i - 1]'))
    sdfg.validate()

    A = np.random.rand(2, 3)
    B = np.random.rand(6)
    sdfg(A=A, B=B)
    assert np.allclose(A.reshape([6])[::-1], B)


def test_reshape_subset():
    """ Tests reshapes on subsets of arrays. """

    @dace.program
    def reshp(A: dace.float64[2, 3, 4], B: dace.float64[12]):
        C = np.reshape(A[1, :, :], [12])
        B[:] += C

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(12)
    expected = np.reshape(A[1, :, :], [12]) + B

    reshp(A, B)
    assert np.allclose(expected, B)


def test_reshape_subset_explicit():
    """ Tests reshapes on subsets of arrays. """
    sdfg = dace.SDFG('reshp')
    sdfg.add_array('A', [2, 3, 4], dace.float64)
    sdfg.add_array('B', [12], dace.float64)
    sdfg.add_view('Av', [12], dace.float64)
    state = sdfg.add_state()

    state.add_mapped_tasklet('compute',
                             dict(i='0:12'),
                             dict(a=dace.Memlet('Av[i]'), b=dace.Memlet('B[i]')),
                             'out = a + b',
                             dict(out=dace.Memlet('B[i]')),
                             external_edges=True)
    v = next(n for n in state.source_nodes() if n.data == 'Av')
    state.add_nedge(state.add_read('A'), v, dace.Memlet('A[1, 0:3, 0:4]'))

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(12)
    expected = np.reshape(A[1, :, :], [12]) + B

    sdfg(A=A, B=B)
    assert np.allclose(expected, B)


def test_reinterpret_smaller():

    @dace.program
    def reint(A: dace.int32[N]):
        C = A.view(dace.int16)
        C[:] += 1

    A = np.random.randint(0, 262144, size=[10], dtype=np.int32)
    expected = np.copy(A)
    B = expected.view(np.int16)
    B[:] += 1

    reint(A)
    assert np.allclose(expected, A)


def test_reinterpret_larger():

    @dace.program
    def reint(A: dace.int16[N]):
        C = A.view(dace.int32)
        C[:] += 1

    A = np.random.randint(0, 32767, size=[10], dtype=np.int16)
    expected = np.copy(A)
    B = expected.view(np.int32)
    B[:] += 1

    reint(A)
    assert np.allclose(expected, A)


def test_reinterpret_invalid():

    @dace.program
    def reint_invalid(A: dace.float32[5]):
        C = A.view(dace.float64)
        C[:] += 1

    A = np.random.rand(5).astype(np.float32)
    with pytest.raises(ValueError,
                       match="When changing to a larger dtype, its size must be a divisor of the total size "
                       "in bytes of the last axis of the array."):
        reint_invalid(A)


def test_reinterpret_symbolic_stride_uses_int_floor():
    """View descriptors must divide with int_floor, never `//`.

    `//` on a symbolic stride builds sympy `floor(expr / d)`, and sym2cpp prints that argument
    WITHOUT the floor, leaving each term of the sum to truncate on its own. A non-contiguous
    symbolic stride is the case where the division does not fold away.
    """

    @dace.program
    def reint(A: dace.data.Array(dace.int16, [N, 4], strides=[2 * N + 1, 1], total_size=N * (2 * N + 1))):
        C = A.view(dace.int32)
        C[:] += 1

    sdfg = reint.to_sdfg(simplify=False)
    exprs = [
        str(e) for d in sdfg.arrays.values() if isinstance(d, dace.data.View)
        for e in (*d.shape, *d.strides, d.total_size)
    ]
    assert any('int_floor' in e for e in exprs), exprs
    assert not any('floor' in e.replace('int_floor', '') for e in exprs), exprs


B = dace.symbol('B')
K = dace.symbol('K')
M = dace.symbol('M')
OH = dace.symbol('OH')
OW = dace.symbol('OW')
C = dace.symbol('C')


def views_of(sdfg: dace.SDFG) -> list:
    return [(name, desc) for name, desc in sdfg.arrays.items() if isinstance(desc, dace.data.View)]


def test_reshape_of_a_strided_slice_reads_the_slice():
    """numpy returns a VIEW only when the new shape can be walked with the SOURCE's own strides; a
    non-contiguous source is copied. Reinterpreting one as packed reads the elements the slice skips
    -- densenet's im2col reshapes a strided ``nhwc[:, ky:ky+OH, kx:kx+OW, :]`` window, so every tap
    came out of the zero padding and the network was 1e140 off with nothing to catch it."""

    @dace.program
    def one_block(nhwc: dace.float64[B, OH + 2, OW + 2, C], col: dace.float64[B * OH * OW, C]):
        patch = nhwc[:, 1:1 + OH, 2:2 + OW, :]
        col[:] = np.reshape(patch, (B * OH * OW, C))

    b, oh, ow, c = 2, 3, 4, 5
    sdfg = one_block.to_sdfg(simplify=False)
    # Structural: the reshaped view must sit on a PACKED container. A view whose strides are the
    # padded parent's is the miscompile, and it is invisible in the shapes alone.
    reshaped = [d for name, d in views_of(sdfg) if list(d.shape) == [B * OH * OW, C]]
    assert reshaped, [str(d.shape) for _, d in views_of(sdfg)]
    for desc in reshaped:
        assert list(desc.strides) == [C, 1], f'reshaped view has strides {desc.strides}'

    nhwc = np.random.default_rng(20260830).random((b, oh + 2, ow + 2, c))
    col = np.zeros((b * oh * ow, c))
    sdfg(nhwc=nhwc.copy(), col=col, B=b, OH=oh, OW=ow, C=c)
    assert np.allclose(col, np.reshape(nhwc[:, 1:1 + oh, 2:2 + ow, :], (b * oh * ow, c))), col


def test_reshape_of_a_contiguous_source_adds_no_copy():
    """The materialization is for non-contiguous sources ONLY. A whole-array reshape is a view, and
    turning it into a copy would cost every reshape in the corpus a buffer."""

    @dace.program
    def whole(a: dace.float64[B, OH, OW, C], col: dace.float64[B * OH * OW, C]):
        col[:] = np.reshape(a, (B * OH * OW, C))

    sdfg = whole.to_sdfg(simplify=False)
    transients = [name for name, desc in sdfg.arrays.items() if desc.transient and not isinstance(desc, dace.data.View)]
    assert not transients, f'a contiguous reshape materialized {transients}'


def test_im2col_gathers_every_tap_from_its_own_window():
    """The shape densenet actually runs: one buffer filled block by block, each block a reshape of a
    different strided window. Every block reads a window the previous one did not."""

    @dace.program
    def im2col(nhwc: dace.float64[B, OH + 2, OW + 2, C], col: dace.float64[B * OH * OW, 9 * C]):
        for ky in range(3):
            for kx in range(3):
                patch = nhwc[:, ky:ky + OH, kx:kx + OW, :]
                col[:, (ky * 3 + kx) * C:(ky * 3 + kx) * C + C] = np.reshape(patch, (B * OH * OW, C))

    b, oh, ow, c = 2, 3, 4, 5
    nhwc = np.random.default_rng(20260831).random((b, oh + 2, ow + 2, c))
    col = np.full((b * oh * ow, 9 * c), np.nan)
    im2col(nhwc=nhwc.copy(), col=col, B=b, OH=oh, OW=ow, C=c)

    assert not np.isnan(col).any(), 'the buffer was left partly unwritten'
    expected = np.empty((b * oh * ow, 9 * c))
    for ky in range(3):
        for kx in range(3):
            window = nhwc[:, ky:ky + oh, kx:kx + ow, :]
            expected[:, (ky * 3 + kx) * c:(ky * 3 + kx) * c + c] = np.reshape(window, (b * oh * ow, c))
    assert np.allclose(col, expected), np.abs(col - expected).max()


def test_a_split_of_a_strided_slice_stays_a_view():
    """The other side of the rule, and the one that miscompiles QUIETLY. A non-contiguous source is
    still viewable when the new shape only splits an axis that is contiguous in it, and numpy aliases
    there -- so a write through the result must reach the source. Copying whenever the source is not
    packed drops that write with nothing to see."""

    @dace.program
    def split_axis(a: dace.float64[4, 6]):
        v = a[:, 0:4]
        c = np.reshape(v, (4, 2, 2))
        c[:] += 1.0

    a = np.zeros((4, 6))
    split_axis(a=a)
    expected = np.zeros((4, 6))
    expected[:, 0:4] += 1.0
    assert np.allclose(a, expected), a


def test_writing_through_an_undecided_reshape_is_refused():
    """Symbolic extents leave cases numpy never meets, and there the two answers are different
    PROGRAMS: a view propagates the write to the source, a copy keeps it local. Merging the rows of
    ``a[:, 0:M]`` out of an ``(N, K)`` array is exactly that -- contiguous, hence a view, if ``K`` is
    ``M``, and a copy otherwise -- so it is reported rather than guessed."""

    @dace.program
    def undecided_write(a: dace.float64[N, K], out: dace.float64[N * M]):
        c = np.reshape(a[:, 0:M], (N * M, ))
        c[:] = 1.0
        out[:] = c

    with pytest.raises(Exception, match='not provably expressible as a view'):
        undecided_write.to_sdfg(simplify=False)


def test_a_contiguous_source_reshapes_to_any_shape_as_a_view():
    """A contiguous source reshapes to ANY shape of the same size, whatever the extents are -- numpy
    takes that path before it tries to factor anything. ``(N, M) -> (M, N)`` is the case that exposes
    a missing fast path: it is transpose-SHAPED, the factoring cannot order ``N`` against ``M``, and
    calling it undecidable would refuse a write numpy propagates."""
    n, m = 3, 4
    strides, decided = dace.data.core.nocopy_reshape_strides([N, M], [M, 1], [M, N])
    assert decided and strides == [N, 1], (strides, decided)

    @dace.program
    def swap_shape(a: dace.float64[N, M]):
        c = np.reshape(a, (M, N))
        c[:] += 1.0

    a = np.zeros((n, m))
    swap_shape(a=a, N=n, M=m)
    assert np.allclose(a, 1.0), a


def test_a_provable_copy_takes_the_write_like_numpy():
    """A reshape numpy itself would copy is NOT refused: numpy drops the propagation there too, and
    the write is perfectly meaningful against the copy. Merging rows across a padded stride is that
    case -- ``M + 2`` is provably not ``M``, so no guess is involved."""

    @dace.program
    def padded_merge(a: dace.float64[N, M + 2], out: dace.float64[N * M]):
        c = np.reshape(a[:, 0:M], (N * M, ))
        c[:] = 1.0
        out[:] = c

    n, m = 3, 4
    a = np.zeros((n, m + 2))
    out = np.zeros(n * m)
    padded_merge(a=a, out=out, N=n, M=m)
    assert np.allclose(out, 1.0), out
    assert np.allclose(a, 0.0), 'the write reached the source, so the reshape aliased when it copied'


@pytest.mark.parametrize('source', ['contiguous', 'row slice', 'stride 2', 'last axis slice', 'transposed'])
def test_the_view_rule_agrees_with_numpy(source):
    """numpy IS the specification here, so it is the oracle: for each source layout, the decision and
    the strides must match what ``np.reshape`` actually does, checked with ``shares_memory``."""
    base = np.arange(2 * 6 * 4 * 3, dtype=np.float64).reshape(2, 6, 4, 3)
    src = {
        'contiguous': base,
        'row slice': base[:, 1:5, :, :],
        'stride 2': base[:, ::2, :, :],
        'last axis slice': base[:, :, :, 0:2],
        'transposed': base.transpose(0, 2, 1, 3),
    }[source]
    strides = [st // src.itemsize for st in src.strides]

    checked = 0
    for shape in ((src.size, ), (2, -1), tuple(src.shape), (src.shape[0], -1), (src.size // 3, 3), (2, 2, -1)):
        want = tuple(np.reshape(src, shape).shape)
        out = np.reshape(src, want)
        got, _ = dace.data.core.nocopy_reshape_strides(list(src.shape), strides, list(want))
        assert (got is not None) == np.shares_memory(out, src), f'{src.shape}->{want}: numpy view'
        if got is not None:
            # numpy leaves an arbitrary stride on a length-1 axis, so only the axes that walk compare.
            walked = [(g, st // src.itemsize) for g, st, e in zip(got, out.strides, want) if e != 1]
            assert [g for g, _ in walked] == [e for _, e in walked], f'{src.shape}->{want}: {got}'
        checked += 1
    assert checked, 'no shape was exercised'


if __name__ == "__main__":
    test_reshape()
    test_reshape_dst()
    test_reshape_dst_explicit()
    test_reshape_copy(False)
    test_reshape_copy(True)
    test_reshape_copy_scoped()
    test_reshape_subset()
    test_reshape_subset_explicit()
    test_reinterpret_smaller()
    test_reinterpret_larger()
    test_reinterpret_invalid()
    test_reinterpret_symbolic_stride_uses_int_floor()
    test_reshape_of_a_strided_slice_reads_the_slice()
    test_reshape_of_a_contiguous_source_adds_no_copy()
    test_im2col_gathers_every_tap_from_its_own_window()
    test_a_split_of_a_strided_slice_stays_a_view()
    test_writing_through_an_undecided_reshape_is_refused()
    test_a_contiguous_source_reshapes_to_any_shape_as_a_view()
    test_a_provable_copy_takes_the_write_like_numpy()
    test_the_view_rule_agrees_with_numpy('row slice')
