# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for skipping resets of containers that nothing wrote since they were last reset."""
import copy
from typing import Dict

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import reuse_transients, skip_redundant_resets
from dace.sdfg.state import LoopRegion


def _loop(var: str, start: int, end: int, body: list) -> tn.ForScope:
    loop = LoopRegion(f'loop_{var}', f'{var} < {end}', var, f'{var} = {start}', f'{var} = {var} + 1')
    return tn.ForScope(loop=loop, children=body)


def _tasklet(code: str, inputs: Dict[str, str], outputs: Dict[str, str]) -> tn.TaskletNode:
    tasklet = dace.nodes.Tasklet('compute', set(inputs), set(outputs), code)
    return tn.TaskletNode(node=tasklet,
                          in_memlets={
                              c: dace.Memlet(m)
                              for c, m in inputs.items()
                          },
                          out_memlets={
                              c: dace.Memlet(m)
                              for c, m in outputs.items()
                          })


def _tree(reset_code: str = 'z = 0', guarded: bool = True, outer: bool = True) -> tn.ScheduleTreeRoot:
    """``for t: L = 0; for k, i: if A[t, k, i] < 0: L[k, i] = A[t, k, i]; out[t] = L`` (``L`` a persistent transient)."""
    sdfg = dace.SDFG('resets')
    sdfg.add_array('A', [4, 8, 16], dace.float64)
    sdfg.add_array('out', [4, 8, 16], dace.float64)
    sdfg.add_array('L', [8, 16], dace.float64, transient=True, lifetime=dace.AllocationLifetime.Persistent)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []
    reset = _loop('k', 0, 8, [_loop('i', 0, 16, [_tasklet(reset_code, {}, {'z': 'L[k, i]'})])])
    write = _tasklet('l = a', {'a': 'A[t, k, i]'}, {'l': 'L[k, i]'})
    fix = _loop('k', 0, 8, [
        _loop('i', 0, 16, [tn.IfScope(condition=CodeBlock('A[t, k, i] < 0'), children=[write])] if guarded else [write])
    ])
    copy_out = _loop('k', 0, 8, [_loop('i', 0, 16, [_tasklet('o = l', {'l': 'L[k, i]'}, {'o': 'out[t, k, i]'})])])
    body = [reset, fix, copy_out]
    stree.add_children([_loop('t', 0, 4, body)] if outer else body[:1] + [_loop('t', 0, 4, body[1:])])
    return stree


def _run(stree: tn.ScheduleTreeRoot, a: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 8, 16))
    sdfg = stree.as_sdfg(simplify=dace.config.Config.get_bool('optimizer', 'automatic_simplification'))
    sdfg(A=a.copy(), out=out)
    return out


def test_reset_skipped_when_clean():
    rng = np.random.default_rng(0)
    clean = rng.random((4, 8, 16))
    dirty = clean.copy()
    dirty[1, 2, 3] = -1.0  # Written in t = 1: the reset must run again in t = 2
    stree = _tree()
    assert skip_redundant_resets(stree) == 1
    tn.validate_children_and_parents_align(stree, root=True)
    for a in (clean, dirty):
        assert np.array_equal(_run(copy.deepcopy(stree), a), _run(_tree(), a))
    assert _run(copy.deepcopy(stree), dirty)[2, 2, 3] == 0.0


def test_reset_kept_if_written_unconditionally():
    assert skip_redundant_resets(_tree(guarded=False)) == 0


def test_reset_kept_if_value_reads_a_name():
    assert skip_redundant_resets(_tree(reset_code='z = t')) == 0


def test_reset_kept_outside_loops():
    assert skip_redundant_resets(_tree(outer=False)) == 0


def test_reset_kept_if_written_by_other_nodes():
    stree = _tree()
    sdfg_copy = tn.CopyNode(target='L', memlet=dace.Memlet('A[0, 0:8, 0:16] -> [0:8, 0:16]'))
    stree.children[0].add_child(sdfg_copy)
    assert skip_redundant_resets(stree) == 0


def test_reuse_leaves_skipped_reset_alone():
    """After the pass, the container may keep values between iterations: no other transient may share it."""

    def with_other():
        stree = _tree()
        stree.containers['M'] = dace.data.Array(dace.float64, [8, 16],
                                                transient=True,
                                                lifetime=dace.AllocationLifetime.Persistent)
        write = _tasklet('m = a', {'a': 'A[0, k, i]'}, {'m': 'M[k, i]'})
        read = _tasklet('o = m', {'m': 'M[k, i]'}, {'o': 'out[0, k, i]'})
        stree.add_children(
            [_loop('k', 0, 8, [_loop('i', 0, 16, [write])]),
             _loop('k', 0, 8, [_loop('i', 0, 16, [read])])])
        return stree

    assert reuse_transients(with_other()) == 2  # L and M share memory without the pass
    stree = with_other()
    assert skip_redundant_resets(stree) == 1
    assert reuse_transients(stree) == 0


if __name__ == '__main__':
    test_reset_skipped_when_clean()
    test_reset_kept_if_written_unconditionally()
    test_reset_kept_if_value_reads_a_name()
    test_reset_kept_outside_loops()
    test_reset_kept_if_written_by_other_nodes()
    test_reuse_leaves_skipped_reset_alone()
