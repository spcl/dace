# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Round-tripping an SDFG that carries a transformation history.

The history stores each transformation by CLASS NAME only, and the reader resolves it against the
already-imported ``PatternTransformation`` subclasses. A process that just did ``import dace`` has
none of the transformation packages loaded, so the lookup used to raise a bare ``StopIteration`` that
``dace.serialize.from_json`` turned into a warning plus an opaque placeholder -- the SDFG loaded, and
silently lost the element.
"""
import gzip
import os
import subprocess
import sys

import pytest

import dace
from dace.transformation.interstate.state_fusion import StateFusion
from dace.transformation.transformation import PatternTransformation, SubgraphTransformation, transformation_by_name


def fused_sdfg() -> dace.SDFG:
    """Two states writing one element each, fused -- ``apply_to`` records the match in the history."""
    sdfg = dace.SDFG('hist_roundtrip')
    sdfg.add_array('A', [2], dace.float64)
    first = sdfg.add_state('s0')
    second = sdfg.add_state_after(first, 's1')
    for i, state in enumerate((first, second)):
        tasklet = state.add_tasklet(f't{i}', {}, {'out'}, 'out = 1.0')
        state.add_edge(tasklet, 'out', state.add_write('A'), None, dace.Memlet(f'A[{i}]'))
    StateFusion.apply_to(sdfg, first_state=first, second_state=second)
    return sdfg


def load_in_fresh_process(path: str, expected_hist: int) -> None:
    """Load ``path`` in a subprocess that only imports ``dace``, with every deserialization warning
    promoted to an error. That is the state the original bug needed to show up."""
    script = ('import dace\n'
              f'sdfg = dace.SDFG.from_file({path!r})\n'
              f'assert len(sdfg.transformation_hist) == {expected_hist}, len(sdfg.transformation_hist)\n'
              'assert all(type(x).__name__ == "StateFusion" for x in sdfg.transformation_hist), '
              '[type(x).__name__ for x in sdfg.transformation_hist]\n')
    env = dict(os.environ, DACE_testing_deserialize_exception='1')
    proc = subprocess.run([sys.executable, '-W', 'error::UserWarning', '-c', script],
                          capture_output=True,
                          text=True,
                          env=env)
    assert proc.returncode == 0, f'fresh-process load failed:\n{proc.stdout}\n{proc.stderr}'


def test_transformation_history_roundtrip(tmp_path):
    sdfg = fused_sdfg()
    assert len(sdfg.transformation_hist) == 1

    path = tmp_path / 'hist.sdfgz'
    sdfg.save(str(path), compress=True, include_transformation_history=True)
    load_in_fresh_process(str(path), expected_hist=1)

    # save -> load -> save is a fixed point: nothing is lost on the way through JSON.
    reloaded = dace.SDFG.from_file(str(path))
    again = tmp_path / 'hist2.sdfgz'
    reloaded.save(str(again), compress=True, include_transformation_history=True)
    assert dace.SDFG.from_file(str(again)).hash_sdfg() == reloaded.hash_sdfg() == sdfg.hash_sdfg()
    assert gzip.open(again, 'rt').read() == gzip.open(path, 'rt').read()


def test_unknown_transformation_name_raises():
    """A name that really does not exist must fail loudly, not resolve to nothing."""
    with pytest.raises(TypeError, match='NoSuchTransformation'):
        transformation_by_name('NoSuchTransformation', PatternTransformation)
    with pytest.raises(TypeError, match='NoSuchTransformation'):
        transformation_by_name('NoSuchTransformation', SubgraphTransformation)


def test_builtin_transformation_resolves_without_prior_import():
    """The resolver imports the built-in packages itself, so a name is found regardless of what the
    calling process happened to import."""
    script = ('from dace.transformation.transformation import PatternTransformation, transformation_by_name\n'
              'assert transformation_by_name("LoopUnroll", PatternTransformation).__name__ == "LoopUnroll"\n')
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
