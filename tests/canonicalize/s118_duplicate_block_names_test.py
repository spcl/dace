# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression: ``TrivialLoopElimination`` must not reparent a loop body under a name the
destination region already uses.

TSVC ``s118`` (``for i in range(1, N): for j in range(0, i): a[i] += bb[j, i] * a[i-j-1]``)
canonicalized to an INVALID SDFG -- ``Found multiple blocks with the same name in for_259_par``.

The chain: loop peeling clones the inner ``j`` loop into siblings ``for_259_p0``/``p1``/``p2``
inside ``for_259_par``. Each clone is a deepcopy, so all three carry a body block named
``slice_a_260``. That is legal -- a LoopRegion is its own name scope, and block labels only
have to be unique WITHIN a region. The SDFG is valid at that point.

``TrivialLoopElimination`` then splices each single-iteration peel's body straight into
``for_259_par``, moving those blocks into a DIFFERENT name scope. It did so with a bare
``graph.add_node(self.loop.start_block)`` (plus ``add_edge``'s implicit auto-add for the rest),
neither of which renames. Eliminating the first peel was fine; the second landed a second
``slice_a_260`` next to the first. Nothing downstream repairs block labels --
``UniqueLoopIterators`` renames loop ITERATORS (symbols), not blocks -- so the invalid graph
survived to the validate.

The fix names the block uniquely AT the reparenting point (``ensure_unique_name=True``, as every
other reparenting site in the codebase already does), so the region is never left holding two
blocks under one name.

``test_trivial_loop_elimination_reparents_clone_bodies_uniquely`` is the load-bearing one and the
strict gate on the fix: it builds the shape directly (a loop and its deepcopy as siblings) out of
loops that run EXACTLY once, and fails deterministically without it.

The two ``s118`` tests pin the reported kernel end to end, but they do NOT gate this fix on their
own: ``can_be_applied``'s zero-trip guard independently removes s118's trigger by refusing to
splice a never-executed peel, so s118 canonicalizes cleanly with either change alone. The
underlying reparenting bug is unaffected by that guard -- a merely single-iteration loop is
trivially eliminable and still collides -- which is what the unit test above pins.
"""
import copy
import os

import numpy as np
import pytest

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.trivial_loop_elimination import TrivialLoopElimination
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from tests.corpus.measure_parallelization import cpu_params
from tests.corpus.tsvc import tsvc as TS
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

_N = 8
_TRIALS = 10


def _duplicate_block_names(sdfg: dace.SDFG):
    """``[(region_label, [duplicated_block_label, ...]), ...]`` -- empty iff every region's
    block labels are unique, which is what ``SDFG.validate`` requires."""
    found = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        seen = set()
        dups = sorted({b.label for b in region.nodes() if b.label in seen or seen.add(b.label)})
        if dups:
            found.append((region.label, dups))
    return found


def _single_iteration_loop(sdfg: dace.SDFG, label: str, body_label: str) -> LoopRegion:
    """A ``for i in range(1)`` loop whose body block is named ``body_label`` and writes ``a[0]``."""
    loop = LoopRegion(label, condition_expr='i < 1', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    body = loop.add_state(body_label, is_start_block=True)
    tasklet = body.add_tasklet(f'{label}_t', {}, {'out'}, 'out = 1.0')
    body.add_edge(tasklet, 'out', body.add_access('a'), None, dace.Memlet('a[0]'))
    return loop


def test_trivial_loop_elimination_reparents_clone_bodies_uniquely():
    """Two sibling trivial loops cloned from one original share a body-block name. Splicing both
    into the shared parent must rename on arrival rather than duplicate the label.

    Pre-fix this leaves two ``dup_body`` blocks in the SDFG root and ``validate`` raises
    ``Found multiple blocks with the same name``.
    """
    sdfg = dace.SDFG('tle_clone_body_names')
    sdfg.add_array('a', [_N], dace.float64)
    sdfg.add_symbol('i', dace.int64)

    original = _single_iteration_loop(sdfg, 'loop_p0', 'dup_body')
    clone = copy.deepcopy(original)  # a deepcopy carries the body label over verbatim -- the real shape
    clone.label = 'loop_p1'
    sdfg.add_node(original, is_start_block=True)
    sdfg.add_node(clone)
    sdfg.add_edge(original, clone, InterstateEdge())

    # Sibling loops each owning a `dup_body` is legal: a LoopRegion is its own name scope.
    assert _duplicate_block_names(sdfg) == [], 'precondition: the input SDFG is valid'
    sdfg.validate()

    applied = PatternMatchAndApplyRepeated([TrivialLoopElimination()]).apply_pass(sdfg, {})
    assert applied, 'both single-iteration loops should have been eliminated'
    assert not [cfr for cfr in sdfg.all_control_flow_regions() if isinstance(cfr, LoopRegion)]

    assert _duplicate_block_names(sdfg) == [], 'TrivialLoopElimination reparented a body under a taken name'
    sdfg.validate()


def test_s118_canonicalizes_to_a_valid_sdfg():
    """The reported failure: ``s118`` canonicalization raised ``Found multiple blocks with the same
    name in for_259_par``. Repeated in-process because the producing order is not deterministic."""
    kernel = TS.collect(name='s118_d_single')[0]
    base = TS.to_sdfg(kernel, tag='s118_dupnames', simplify=True)
    for trial in range(_TRIALS):
        sdfg = copy.deepcopy(base)
        canonicalize(sdfg, validate=True, validate_all=False, **cpu_params(4))
        assert _duplicate_block_names(sdfg) == [], f'trial {trial}: duplicate block names survived'


def test_s118_is_bit_exact_after_canonicalization():
    """Canonicalization is value-preserving: ``s118`` must match its numpy oracle exactly.

    Run in a forked child so a miscompiled kernel segfaulting cannot take the session down.
    """
    kernel = TS.collect(name='s118_d_single')[0]
    arrays, call_kwargs = TS.make_inputs(kernel, seed=1234)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES['s118_d_single'](**ref, **call_kwargs)

    sdfg = TS.to_sdfg(kernel, tag='s118_dupnames_exact', simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params(4))
    fin = finalize_for_target(copy.deepcopy(sdfg), 'cpu')
    fin.name = f'{fin.name}_s118_dupnames_exact'
    csdfg = fin.compile()

    got = {n: a.copy() for n, a in arrays.items()}
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: never return into pytest, and never raise past os._exit
        code = 1
        try:
            csdfg(**got, **call_kwargs)
            # A pure reassociation-free recurrence: equality is exact, no tolerance.
            code = 0 if all(np.array_equal(got[n], ref[n]) for n in arrays) else 2
        except BaseException:  # noqa: BLE001 -- report as an exit code, never as an exception
            code = 3
        finally:
            os.write(write_fd, bytes([code]))
            os._exit(code)

    os.close(write_fd)
    _, status = os.waitpid(pid, 0)
    os.close(read_fd)

    assert not os.WIFSIGNALED(status), f's118 killed by signal {os.WTERMSIG(status)}'
    assert os.WEXITSTATUS(status) == 0, f's118 not bit-exact against the numpy oracle (code {os.WEXITSTATUS(status)})'


if __name__ == '__main__':
    pytest.main([__file__])
