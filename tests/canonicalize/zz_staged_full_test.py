# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scratch: full staged walk from scratch, so the cleaned pipeline is verified phase by phase.

The cached snapshots were built before ``PruneUnreferencedTransients`` joined ``StructuralCleanup``
and before the terminal symbol trio started running twice, so resuming from them would verify a
graph the current recipe does not produce. Fresh cache directory, ``resume=False``.
"""
import os

from tests.canonicalize.cloudsc_canonicalize_staged_test import SPECIES_CONSTANTS, run_staged

CACHE = os.path.expanduser('~/.cache/cloudsc_staged_clean')


def test_staged_from_scratch():
    records = run_staged(CACHE, verify_numerics=True, resume=False, specialize_constants=SPECIES_CONSTANTS)
    bad = [r for r in records if not r.get('ok', True)]
    print(f'[STAGED] {len(records)} phases, {len(bad)} bad', flush=True)
    for r in bad:
        print(f"[BAD] {r.get('label')}: {r.get('detail')}", flush=True)
    assert not bad, f'{len(bad)} phases diverged'
