# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Proof-of-concept / feasibility test for a data-dependent-size result inside
a single compiled SDFG: ``B = A[mask]`` for a boolean array ``mask``, where
the result's element count is not known until the mask has actually been
read at runtime.

This does NOT exercise a Python frontend (classic or nextgen) -- neither
currently lowers boolean-mask reads to real dataflow; both categorize them
as an unsupported ``data-dependent-subscript`` and fall back to a callback.
It hand-builds the SDFG directly to answer a prior question: does DaCe's
existing machinery (WCR reductions, interstate-edge symbol assignment,
symbol-dependent transient/view shapes) support this at all, and if so,
what does the generated code look like? See the design discussion this
test resulted from for the full writeup; the short version:

- An SDFG can compute a runtime count into a scalar, assign it to a symbol
  via a plain ``InterstateEdge`` (``M = __nnz[0]``, the same primitive
  ``ScalarToSymbolPromotion`` produces post-hoc, done here directly at
  construction time instead), and use that symbol in a LATER state's data
  descriptor shapes. ``tests/symbol_dependent_transients_test.py`` already
  proves codegen defers allocation correctly for loop-bounded symbols; this
  file extends that to a genuinely data-dependent (not loop-bounded) count.
- Two configurable compaction strategies, both valid, verified against
  NumPy across normal and boundary (empty/full mask) cases:

  * ``'view'``  -- ONE pass over the source: a stream push into an
    upper-bound-sized (``N``) backing buffer, fused with a WCR sum computing
    the count; the result is exposed as a VIEW over the buffer's first
    ``M`` elements. Cheaper in compute, wastes ``N - M`` elements of
    backing memory.
  * ``'exact'`` -- TWO passes over ``mask``: a pure count-only reduction
    first (resolves ``M``, no per-element writes at all), then a second
    compaction pass whose stream capacity is exactly ``M`` (the just-learned
    array shape), so nothing is ever over-allocated. Costs a second read of
    ``mask``.

- A load-bearing pitfall found while building this, worth documenting: the
  WCR accumulator transient (``__nnz``) MUST be explicitly zeroed in an
  ``init`` state before the reduction runs. Its allocated memory is not
  zero-initialized by default (unlike ``samples/explicit/filter.py``, whose
  equivalent accumulator is a CALLER-supplied argument that the sample's
  ``__main__`` explicitly zeroes before the call) -- omitting the zero
  state produces a build that compiles cleanly and often *appears* to
  work, then segfaults non-deterministically (an uninitialized start value
  occasionally landing on zeroed heap memory by chance). Caught by running
  the exact same construction fresh across several sizes/fractions, not by
  a single successful run.
- A genuine ABI boundary, NOT solved here: a compiled SDFG's ctypes call
  convention requires every argument's backing memory to exist before the
  call, so a result whose size is discovered mid-execution cannot be handed
  back from ONE opaque external call (confirmed via ``__return`` shape
  resolution in ``dace/codegen/compiled_sdfg.py::_initialize_return_values``,
  which evaluates symbolic shapes from caller-supplied ``kwargs`` BEFORE
  invoking the compiled function body). Both strategies below therefore
  still require ``M`` and ``B`` to be supplied by the caller at this SDFG's
  external boundary -- exactly as ``filter.py`` already does for its
  upper-bound output. What this test demonstrates is that the deferred
  symbol resolves correctly and is usable for INTERNAL/intermediate
  computation on the exact- or view-sized result within a larger program
  (e.g. a subsequent map over ``B[M]`` inside the same SDFG) without the
  caller needing to know ``M`` in advance for THAT part. A genuinely opaque
  single-call ``B = A[mask]`` at a program's external boundary needs a
  two-kernel (count, then fill) dispatch pattern, which is future work.

To inspect the generated C++ directly: run this file's ``__main__`` block
and look under ``.dacecache/bool_index_view/src/cpu/`` and
``.dacecache/bool_index_exact/src/cpu/``.
"""
import numpy as np
import pytest

import dace

N = dace.symbol('N')
M = dace.symbol('M')


def _add_wcr_zero_init(sdfg: dace.SDFG, next_state: dace.SDFGState) -> dace.SDFGState:
    """
    A state that zeroes the ``__nnz`` WCR accumulator, wired before
    ``next_state``. See the module docstring's "load-bearing pitfall" note:
    this is not optional.
    """
    init = sdfg.add_state('init')
    tasklet = init.add_tasklet('zero_nnz', {}, {'z'}, 'z = 0')
    init.add_memlet_path(tasklet, init.add_write('__nnz'), src_conn='z', memlet=dace.Memlet('__nnz[0]'))
    sdfg.add_edge(init, next_state, dace.InterstateEdge())
    return init


def _build_view(dtype=dace.float64) -> dace.SDFG:
    """Single-pass strategy: over-allocated backing buffer + a view over its first M elements."""
    sdfg = dace.SDFG('bool_index_view')
    sdfg.add_array('A', [N], dtype)
    sdfg.add_array('mask', [N], dace.bool_)
    sdfg.add_array('__buf', [N], dtype, transient=True)
    sdfg.add_array('__nnz', [1], dace.uint32, transient=True)
    sdfg.add_stream('__s', dtype, transient=True)

    fill = sdfg.add_state('fill')
    _add_wcr_zero_init(sdfg, fill)

    me, mx = fill.add_map('filter_map', dict(i='0:N'))
    tasklet = fill.add_tasklet('predicate', {'a', 'm'}, {'b', 'osz'}, 'if m:\n    b = a\nosz = 1 if m else 0')
    fill.add_memlet_path(fill.add_read('A'), me, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    fill.add_memlet_path(fill.add_read('mask'), me, tasklet, dst_conn='m', memlet=dace.Memlet('mask[i]'))
    s_write = fill.add_write('__s')
    fill.add_memlet_path(tasklet,
                         mx,
                         s_write,
                         src_conn='b',
                         memlet=dace.Memlet(data='__s', subset='0', dynamic=True, volume=0))
    fill.add_memlet_path(tasklet,
                         mx,
                         fill.add_write('__nnz'),
                         src_conn='osz',
                         memlet=dace.Memlet(data='__nnz', subset='0', dynamic=True, volume=0, wcr='lambda x, y: x + y'))
    # DaCe auto-detects an adjacent stream->array pair and emits a direct push.
    fill.add_nedge(s_write, fill.add_write('__buf'), dace.Memlet(data='__buf', subset='0:N', dynamic=False, volume=1))

    expose = sdfg.add_state('expose')
    sdfg.add_edge(fill, expose, dace.InterstateEdge(assignments={'M': '__nnz[0]'}))

    sdfg.add_view('result_view', [M], dtype)
    buf_read = expose.add_read('__buf')
    view_node = expose.add_access('result_view')
    expose.add_edge(buf_read, None, view_node, 'views', dace.Memlet('__buf[0:M]'))

    sdfg.add_array('B', [M], dtype)
    expose.add_nedge(view_node, expose.add_write('B'), dace.Memlet('B[0:M]'))

    return sdfg


def _build_exact(dtype=dace.float64) -> dace.SDFG:
    """Two-pass strategy: a pure count pass, then a compaction sized exactly to the count."""
    sdfg = dace.SDFG('bool_index_exact')
    sdfg.add_array('A', [N], dtype)
    sdfg.add_array('mask', [N], dace.bool_)
    sdfg.add_array('__nnz', [1], dace.uint32, transient=True)

    count = sdfg.add_state('count')
    _add_wcr_zero_init(sdfg, count)

    me, mx = count.add_map('count_map', dict(i='0:N'))
    tasklet = count.add_tasklet('predicate', {'m'}, {'osz'}, 'osz = 1 if m else 0')
    count.add_memlet_path(count.add_read('mask'), me, tasklet, dst_conn='m', memlet=dace.Memlet('mask[i]'))
    count.add_memlet_path(tasklet,
                          mx,
                          count.add_write('__nnz'),
                          src_conn='osz',
                          memlet=dace.Memlet(data='__nnz', subset='0', dynamic=True, volume=0,
                                             wcr='lambda x, y: x + y'))

    fill = sdfg.add_state('fill')
    sdfg.add_edge(count, fill, dace.InterstateEdge(assignments={'M': '__nnz[0]'}))

    sdfg.add_array('B', [M], dtype)
    sdfg.add_stream('__s', dtype, transient=True)
    me2, mx2 = fill.add_map('fill_map', dict(i='0:N'))
    tasklet2 = fill.add_tasklet('select', {'a', 'm'}, {'b'}, 'if m:\n    b = a')
    fill.add_memlet_path(fill.add_read('A'), me2, tasklet2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    fill.add_memlet_path(fill.add_read('mask'), me2, tasklet2, dst_conn='m', memlet=dace.Memlet('mask[i]'))
    s_write = fill.add_write('__s')
    fill.add_memlet_path(tasklet2,
                         mx2,
                         s_write,
                         src_conn='b',
                         memlet=dace.Memlet(data='__s', subset='0', dynamic=True, volume=0))
    fill.add_nedge(s_write, fill.add_write('B'), dace.Memlet(data='B', subset='0:M', dynamic=False, volume=1))

    return sdfg


_BUILDERS = {'view': _build_view, 'exact': _build_exact}

#: (seed, N, fraction of True in mask); fraction outside (0, 1) means a
#: constant all-False/all-True mask rather than a random one, exercising
#: the M=0 and M=N boundaries where the uninitialized-accumulator bug above
#: was actually found (a "normal" mid-range case alone did not reproduce it
#: reliably).
_CASES = [
    (1, 200, 0.4),
    (2, 50, 1.0),
    (3, 50, 0.0),
    (4, 200, 0.425),
    (5, 1, 1.0),
    (6, 1, 0.0),
    (7, 500, 0.9),
]


@pytest.mark.parametrize('strategy', ['view', 'exact'])
def test_boolean_filter_matches_numpy(strategy):
    compiled = _BUILDERS[strategy]().compile()
    for seed, n, frac in _CASES:
        rng = np.random.default_rng(seed)
        A = rng.random(n)
        mask = rng.random(n) < frac if 0 < frac < 1 else np.full(n, frac >= 1.0, dtype=bool)
        m = int(mask.sum())
        B = np.zeros(m, dtype=np.float64)
        compiled(A=A, mask=mask, B=B, N=n, M=m)
        expected = A[mask]
        if m:
            assert np.allclose(np.sort(B), np.sort(expected)), f'{strategy} N={n} M={m} seed={seed}'
        else:
            assert B.shape == (0, )


if __name__ == '__main__':
    test_boolean_filter_matches_numpy('view')
    test_boolean_filter_matches_numpy('exact')
