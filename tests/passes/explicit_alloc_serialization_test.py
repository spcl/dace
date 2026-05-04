# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Round-trip tests for the Explicit-allocation extensions.

Asserts that the new InterstateEdge properties (alloc / free / reuse) and the
AllocationLifetime.Explicit enum value survive an SDFG -> JSON -> SDFG cycle.
This is the serialisation contract the design relies on: passes that mutate
allocation state (make_explicit, buffer_reuse_same_pass, buffer_reuse_same_pass_ua) must
be observable through .sdfg files saved to disk and reloaded by code-gen.
"""
import json

import dace
from dace import dtypes
from dace.sdfg import SDFG, InterstateEdge


def _explicit_array(sdfg: SDFG, name: str, shape=(4,)):
    sdfg.add_array(name, shape, dace.float64, transient=True,
                   lifetime=dtypes.AllocationLifetime.Explicit)


def _three_state_with_annotations() -> SDFG:
    """SDFG carrying every annotation the new properties expose:
    alloc, free, and a reuse pair."""
    sdfg = SDFG('rtrip')
    _explicit_array(sdfg, 'tmp1')
    _explicit_array(sdfg, 'tmp2')
    s0 = sdfg.add_state('init')
    s1 = sdfg.add_state('use')
    s2 = sdfg.add_state('done')
    e0 = sdfg.add_edge(s0, s1, InterstateEdge())
    e1 = sdfg.add_edge(s1, s2, InterstateEdge())
    s1.add_access('tmp1')
    s1.add_access('tmp2')
    e0.data.alloc = ['tmp1']
    e0.data.reuse = [['tmp2', 'tmp1']]   # tmp2 rebinds onto tmp1
    e1.data.free  = ['tmp1']
    return sdfg


def _only_edge(sdfg: SDFG, src_label: str, dst_label: str) -> InterstateEdge:
    src = next(s for s in sdfg.nodes() if s.label == src_label)
    dst = next(s for s in sdfg.nodes() if s.label == dst_label)
    return sdfg.edges_between(src, dst)[0].data


def test_alloc_free_reuse_round_trip():
    sdfg = _three_state_with_annotations()
    blob = sdfg.to_json()
    # ensure we are exercising the JSON path (not in-memory aliasing)
    blob = json.loads(json.dumps(blob))
    rt = SDFG.from_json(blob)

    e0 = _only_edge(rt, 'init', 'use')
    e1 = _only_edge(rt, 'use', 'done')

    assert e0.alloc == ['tmp1']
    assert e0.free == []
    assert e0.reuse == [['tmp2', 'tmp1']]
    assert e1.alloc == []
    assert e1.free == ['tmp1']
    assert e1.reuse == []


def test_explicit_lifetime_round_trip():
    sdfg = _three_state_with_annotations()
    blob = json.loads(json.dumps(sdfg.to_json()))
    rt = SDFG.from_json(blob)

    for name in ('tmp1', 'tmp2'):
        assert rt.arrays[name].lifetime == dtypes.AllocationLifetime.Explicit, (
            f'{name} lost its Explicit lifetime through the JSON round-trip'
        )


def test_save_load_round_trip(tmp_path):
    """Same contract via SDFG.save / SDFG.from_file — the path code-gen uses."""
    sdfg = _three_state_with_annotations()
    fname = tmp_path / 'rtrip.sdfg'
    sdfg.save(str(fname))
    rt = SDFG.from_file(str(fname))

    e0 = _only_edge(rt, 'init', 'use')
    e1 = _only_edge(rt, 'use', 'done')

    assert e0.alloc == ['tmp1']
    assert e0.reuse == [['tmp2', 'tmp1']]
    assert e1.free == ['tmp1']
    assert rt.arrays['tmp1'].lifetime == dtypes.AllocationLifetime.Explicit
    assert rt.arrays['tmp2'].lifetime == dtypes.AllocationLifetime.Explicit


def test_empty_lists_default_round_trip():
    """Edges without annotations must come back with empty alloc/free/reuse —
    not None, not missing — so downstream code can iterate unconditionally."""
    sdfg = SDFG('empty_edges')
    s0 = sdfg.add_state('a'); s1 = sdfg.add_state('b')
    sdfg.add_edge(s0, s1, InterstateEdge())

    rt = SDFG.from_json(json.loads(json.dumps(sdfg.to_json())))
    e = _only_edge(rt, 'a', 'b')
    assert e.alloc == []
    assert e.free == []
    assert e.reuse == []
