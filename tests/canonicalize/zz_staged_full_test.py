# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scratch: full staged walk from scratch, so the cleaned pipeline is verified phase by phase.

The cached snapshots were built before ``PruneUnreferencedTransients`` joined ``StructuralCleanup``
and before the terminal symbol trio started running twice, so resuming from them would verify a
graph the current recipe does not produce. Fresh cache directory, ``resume=False``.
"""
import os

import pytest

from tests.canonicalize.cloudsc_canonicalize_staged_test import SPECIES_CONSTANTS, run_staged

CACHE = os.path.expanduser('~/.cache/cloudsc_staged_clean')


# ``integration``, like the staged walk it calls: this rebuilds and verifies CloudSC once per
# phase from an empty cache, which is far past the 600s per-test cap on the canonicalize CI leg.
# Unmarked it was collected there and timed out, which kills the whole session, not just this test.
@pytest.mark.integration
def test_staged_from_scratch():
    records = run_staged(CACHE, verify_numerics=True, resume=False, specialize_constants=SPECIES_CONSTANTS)
    bad = [r for r in records if not r.get('ok', True)]
    print(f'[STAGED] {len(records)} phases, {len(bad)} bad', flush=True)
    for r in bad:
        print(f"[BAD] {r.get('label')}: {r.get('detail')}", flush=True)
    assert not bad, f'{len(bad)} phases diverged'
