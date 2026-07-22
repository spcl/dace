# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``run_pipeline`` phase checkpointing.

Three things have to hold before resuming past a checkpoint is safe:

* a checkpoint round-trips FAITHFULLY -- same edges (ordering edges included: they carry the WAR
  anti-dependences, and losing one silently reorders the emitted code) and the same generated C;
* resuming lands on the graph the from-scratch run would have produced;
* a checkpoint that does not deserialize cleanly is REFUSED, so a lossy file falls back to recompute
  instead of being resumed from.

Driven on a tiny SDFG rather than CloudSC so all of that runs in seconds; the machinery under test is
the one the CloudSC runs use.
"""
import copy
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.codegen.codegen import generate_code
from dace.sdfg.state import SDFGState
from dace.serialize import SerializableObject
from tests.corpus.cloudsc.pipelines import load_checkpoint, run_pipeline, uniquely_named

PHASES = ('start', 'prep', 'loop_to_x', 'parallelize', 'coalesce')


@dace.program
def rowsum_and_scale(A: dace.float64[24, 16], B: dace.float64[24], C: dace.float64[24, 16]):
    for i in range(24):
        acc = 0.0
        for j in range(16):
            C[i, j] = A[i, j] * 2.0 + 1.0
            acc += A[i, j]
        B[i] = acc


@pytest.fixture(scope='module')
def base_sdfg():
    return uniquely_named(rowsum_and_scale.to_sdfg(simplify=False), 'resume_probe')


@pytest.fixture(scope='module')
def inputs():
    a = np.random.default_rng(0).random((24, 16))
    return a, a.sum(axis=1), a * 2.0 + 1.0


def edge_fingerprint(sdfg: dace.SDFG) -> Counter:
    """Every edge in the hierarchy, keyed by endpoints AND payload. A zero-volume happens-before edge
    carries an empty memlet, so it is only distinguishable if the payload is part of the key -- a
    count-only comparison would miss a dropped ordering edge."""
    counts = Counter()
    for edge, parent in sdfg.all_edges_recursive():
        if isinstance(parent, SDFGState):
            empty = edge.data is None or edge.data.is_empty()
            key = ('dataflow', str(edge.src), str(edge.dst), str(edge.data), empty)
        else:
            key = ('interstate', str(edge.src), str(edge.dst), edge.data.condition.as_string,
                   str(sorted(edge.data.assignments.items())))
        counts[key] += 1
    return counts


def emitted_code(sdfg: dace.SDFG):
    # generate_code raises control flow and infers types in place -- keep the caller's graph intact.
    # Sorted by target name: the code is what must match, not the order the backends ran in.
    return sorted((obj.name, obj.clean_code) for obj in generate_code(copy.deepcopy(sdfg)))


def drive(base_sdfg, inputs, dump: Path, resume: bool, capture=None):
    """Run the ``parallelize`` pipeline on a fresh copy. Returns the final SDFG and the phases that
    actually executed (a resume skips the rest). ``capture`` collects the live per-phase graph."""
    a, ref_b, ref_c = inputs
    executed = []

    def numeric_check(sdfg, phase_name):
        executed.append(phase_name)
        if capture is not None:
            capture[phase_name] = copy.deepcopy(sdfg)
        b, c = np.zeros(24), np.zeros((24, 16))
        sdfg(A=a.copy(), B=b, C=c)
        assert np.allclose(b, ref_b), phase_name
        assert np.allclose(c, ref_c), phase_name

    sdfg = run_pipeline(copy.deepcopy(base_sdfg),
                        'parallelize',
                        dump,
                        tag='resume_probe',
                        numeric_check=numeric_check,
                        resume=resume)
    return sdfg, executed


def checkpoints(dump: Path):
    return sorted(dump.glob('*.sdfgz'))


def test_checkpoint_roundtrip_is_faithful(base_sdfg, inputs, tmp_path):
    """The graph read back from a checkpoint is the graph that was written: same edges, same
    ordering edges, same generated C."""
    dump = tmp_path / 'dump'
    live = {}
    drive(base_sdfg, inputs, dump, resume=False, capture=live)
    assert sorted(live) == sorted(PHASES)

    for path in checkpoints(dump):
        phase = path.stem.split('__')[-1]
        before, after = live[phase], load_checkpoint(path)
        assert edge_fingerprint(after) == edge_fingerprint(before), phase
        assert emitted_code(after) == emitted_code(before), phase
        assert after.hash_sdfg() == before.hash_sdfg(), phase


def test_resume_reproduces_from_scratch(base_sdfg, inputs, tmp_path):
    dump = tmp_path / 'dump'
    scratch, executed = drive(base_sdfg, inputs, dump, resume=False)
    assert executed == list(PHASES)
    assert [p.stem.split('__')[-1] for p in checkpoints(dump)] == list(PHASES)

    # Everything is on disk, so no phase re-runs: the pipeline hands back the final graph as saved,
    # after one numeric check of the graph it resumed from.
    resumed, executed = drive(base_sdfg, inputs, dump, resume=True)
    assert executed == [PHASES[-1]]
    assert resumed.hash_sdfg() == scratch.hash_sdfg()
    assert emitted_code(resumed) == emitted_code(scratch)

    # Partial resume: drop the tail one checkpoint at a time and check that exactly the missing
    # phases re-run. ``drive``'s numeric_check runs the transformed SDFG against the reference on
    # every one of them, so a re-derived phase that computes something else fails here.
    #
    # Not compared against the from-scratch graph: re-DERIVING a phase from a reloaded checkpoint is
    # not bit-identical to deriving it in-process, because reading a checkpoint back rebuilds each
    # symbol with the dtype the SDFG declares for it while the in-process graph may still carry the
    # frontend default (int vs int64_t on a loop index). Re-USING a checkpoint is exact -- that is
    # the full-resume case asserted above, and the one the CloudSC runs rely on.
    for drop in range(1, len(PHASES) + 1):
        for path in checkpoints(dump)[-drop:]:
            path.unlink()
        _, executed = drive(base_sdfg, inputs, dump, resume=True)
        # the phase resumed from is re-checked, then the dropped ones re-run
        rechecked = PHASES[-drop - 1:-drop] if drop < len(PHASES) else ()
        assert executed == list(rechecked) + list(PHASES[-drop:]), drop


def corrupt(path: Path):
    """Give the checkpoint a history entry naming a class that does not exist. History entries
    resolve by class name, so this is the shape of every lossy checkpoint seen in the wild."""
    obj = json.loads(gzip.open(path, 'rt').read())
    obj['attributes']['transformation_hist'] = [{
        'type': 'PatternTransformation',
        'transformation': 'TransformationFromTheFuture',
        'cfg_id': 0,
        'state_id': -1,
        '_subgraph': {},
    }]
    with gzip.open(path, 'wt') as fp:
        json.dump(obj, fp)


def test_lossy_checkpoint_is_refused(base_sdfg, inputs, tmp_path):
    """The default reader warns and hands back an SDFG that quietly lost what it could not rebuild.
    Resume must reject that graph and fall back to an earlier checkpoint."""
    dump = tmp_path / 'dump'
    drive(base_sdfg, inputs, dump, resume=False)
    last = checkpoints(dump)[-1]
    corrupt(last)

    # The default reader swaps the element it could not rebuild for an opaque placeholder and hands
    # back an SDFG -- that is the silent corruption. The strict load raises on the same file.
    lossy = dace.SDFG.from_file(str(last))
    assert [type(x) for x in lossy.transformation_hist] == [SerializableObject]
    with pytest.raises(TypeError, match='TransformationFromTheFuture'):
        load_checkpoint(last)

    _, executed = drive(base_sdfg, inputs, dump, resume=True)
    assert executed == list(PHASES[-2:]), 'the corrupt tail checkpoint should have been skipped'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
