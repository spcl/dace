# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MaterializeLoopExitSymbols`.

A loop-defined symbol (``k = k + step`` on a body interstate edge) whose
final value is read after the loop blocks ``LoopToMap``. The pass materialises
the closed-form exit value under a fresh unique name and rewrites every
post-loop reader to use it, so the original symbol is no longer "used after the
loop" and the body can parallelise.
"""
import contextlib
import os

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import (MaterializeLoopExitSymbols,
                                                                                   _POST_PREFIX)

N = dace.symbol('N')
step = dace.symbol('step')


def _n_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _has_loop_exit_sym(sdfg, base_name):
    return any(s.startswith(f"{_POST_PREFIX}{base_name}_") for s in sdfg.symbols)


def _l2m(sdfg):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg.apply_transformations_repeated(LoopToMap)


def test_post_loop_iv_symbol_materialised_with_unique_name():
    """Build an SDFG by hand: pre-loop ``k=0`` + loop body ``k = k + step`` +
    post-loop tasklet reading ``k``. The pass should add a ``_loop_exit_k_<N>``
    symbol whose post-loop assignment is the closed form, and rewrite the
    post-loop tasklet to read it."""
    sdfg = dace.SDFG('mat_iv_post')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('step', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('out', [1], dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'k': '0'}))
    body = loop.add_state('body', is_start_block=True)
    body2 = loop.add_state('body2')
    loop.add_edge(body, body2, dace.InterstateEdge(assignments={'k': 'k + step'}))

    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())
    out_w = post.add_write('out')
    t = post.add_tasklet('write_k', {}, {'__o'}, '__o = k', language=dace.dtypes.Language.Python)
    post.add_edge(t, '__o', out_w, None, dace.Memlet(data='out', subset='0'))

    res = MaterializeLoopExitSymbols().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _has_loop_exit_sym(sdfg, 'k')
    assert 'k' in t.code.as_string or '_loop_exit_k' in t.code.as_string
    assert '_loop_exit_k' in t.code.as_string, (
        f"post-loop tasklet should read the materialised symbol; got code={t.code.as_string!r}")


def test_no_post_loop_use_is_noop():
    """If the loop-defined symbol is never read after the loop, the pass refuses."""
    sdfg = dace.SDFG('mat_iv_unused')
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('step', dace.int64)
    sdfg.add_symbol('N', dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'k': '0'}))
    body = loop.add_state('body', is_start_block=True)
    body2 = loop.add_state('body2')
    loop.add_edge(body, body2, dace.InterstateEdge(assignments={'k': 'k + step'}))
    post = sdfg.add_state('post')
    sdfg.add_edge(loop, post, dace.InterstateEdge())  # post does not read k

    res = MaterializeLoopExitSymbols().apply_pass(sdfg, {})
    assert res is None
    assert not _has_loop_exit_sym(sdfg, 'k')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
