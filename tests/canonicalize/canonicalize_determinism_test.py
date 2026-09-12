# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize must be a FUNCTION OF ITS INPUT: same SDFG in, same SDFG out.

Iteration over a ``set`` is the standard way this breaks. Sets of strings are ordered by
``PYTHONHASHSEED``; sets of graph NODE objects are ordered by ``id()`` and so vary run to run even
at a fixed seed. When such an order feeds a first-match-wins decision it changes the RESULT (which
loop becomes a Map); when it feeds an allocation it changes the NAMES (``find_new_name`` suffixes).
Both make builds irreproducible.

``PYTHONHASHSEED`` can only be set before the interpreter starts, so each run is a subprocess.
"""
import json
import os
import subprocess
import sys

import pytest

# Kernels that previously diverged. The dominant cause was a set of ``SDFGState`` objects in
# ScalarWriteShadowScopes: those hash by id(), so the order tracked allocation history rather than
# the seed -- which is why several of these are stable ALONE and only diverge when canonicalized
# alongside another kernel in the same process. Hence they are deliberately run together here.
#   s352/s4115 -- privatized accumulator came out as ``_priv_dot_1`` vs ``_priv_dot_0``.
#   s118       -- ScalarFission allocated the peeled loop copies' transients in a permuted order.
#   s471       -- final map count differed (2 vs 1) in a full-corpus run.
KERNELS = ['s352_d_single', 's4115_d_single', 's118_d_single', 's471_d_single']
SEEDS = ['1', '3']

_PROBE = r'''
import hashlib, json, os, sys
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.tsvc import tsvc

out = {}
for name in sys.argv[1].split(','):
    sd = tsvc.to_sdfg(tsvc.collect(name=name)[0], tag='det', simplify=True)
    canonicalize(sd, validate=True, peel_limit=4, break_anti_dependence=True)
    maps = sorted(f'{len(n.map.params)}:{n.map.range}' for n, _ in sd.all_nodes_recursive()
                  if isinstance(n, nd.MapEntry))
    libs = sorted(type(n).__name__ for n, _ in sd.all_nodes_recursive() if isinstance(n, nd.LibraryNode))
    loops = sorted(c.loop_condition.as_string if c.loop_condition else '?'
                   for c in sd.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion))
    arrays = sorted(f'{k}:{v.shape}:{v.dtype}' for k, v in sd.arrays.items())
    out[name] = {'maps': maps, 'loops': loops, 'libs': libs, 'arrays': arrays}
print('__RESULT__' + json.dumps(out))
'''


def _canonicalize_under(seed: str, tmp_path) -> dict:
    """Canonicalize KERNELS in a subprocess pinned to ``seed``; return the structural signature."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {
        **os.environ,
        'PYTHONHASHSEED': seed,
        'PYTHONPATH': root,
        # Per-seed build folder: a shared cache would hide a naming difference.
        'DACE_default_build_folder': str(tmp_path / f'build_{seed}'),
        'MPI4PY_RC_INITIALIZE': '0',
        'OMP_NUM_THREADS': '1',
    }
    proc = subprocess.run([sys.executable, '-c', _PROBE, ','.join(KERNELS)],
                          cwd=root,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=1800)
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith('__RESULT__')]
    assert marker, f'probe failed under PYTHONHASHSEED={seed}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}'
    return json.loads(marker[-1][len('__RESULT__'):])


@pytest.mark.integration
def test_canonicalize_is_deterministic_across_hash_seeds(tmp_path):
    """Same input, different PYTHONHASHSEED -> byte-identical structure AND names."""
    results = {seed: _canonicalize_under(seed, tmp_path) for seed in SEEDS}
    first, *rest = SEEDS

    for kernel in KERNELS:
        base = results[first][kernel]
        for seed in rest:
            other = results[seed][kernel]
            # Structure first: a map/loop-count difference means a transformation fired under one
            # seed and not the other -- a far worse bug than a naming difference.
            assert other['maps'] == base['maps'], (f'{kernel}: map structure differs between '
                                                   f'PYTHONHASHSEED={first} and {seed}')
            assert other['loops'] == base['loops'], f'{kernel}: loop structure differs'
            assert other['libs'] == base['libs'], f'{kernel}: library nodes differ'
            assert other['arrays'] == base['arrays'], (f'{kernel}: array names differ (nondeterministic '
                                                       f'find_new_name allocation order)')


if __name__ == '__main__':
    import pathlib
    test_canonicalize_is_deterministic_across_hash_seeds(pathlib.Path('/tmp/canon_det'))
    print('OK')
