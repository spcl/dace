# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Round-trip tests for JSON (de)serialization of schedule trees.

Covers every concrete node type (through the ``coverage_test`` factories) and a set of small end-to-end programs (from
``formal/scripts/corpus_programs.py``), checking that

* ``to_json`` is idempotent across a ``json.dumps``/``json.loads``/``from_json`` round trip, and
* ``as_string`` of the original and reloaded trees match,

and that an SDFG rebuilt from a reloaded tree computes the same results as the original for a couple of programs.
"""

import json
import os
import sys

import numpy as np
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import serialize as ser
from dace.sdfg.analysis.schedule_tree import treenodes as tn

# ``coverage_test`` lives next to this file; ``corpus_programs`` lives under ``formal/scripts``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "formal", "scripts"))

from corpus_programs import PROGRAMS  # noqa: E402
from coverage_test import FACTORIES  # noqa: E402

_CORPUS = {name: (prog, factory) for name, prog, factory in PROGRAMS}


def _reload(tree: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
    """Serialize, round-trip through a JSON string, and deserialize a schedule tree."""
    return ser.from_json(json.loads(json.dumps(ser.to_json(tree), sort_keys=True, indent=2)))


def _assert_roundtrip(tree: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
    reloaded = _reload(tree)
    assert ser.to_json(reloaded) == ser.to_json(tree)
    assert reloaded.as_string() == tree.as_string()
    # Parent pointers are restored, not serialized.
    tn.validate_children_and_parents_align(reloaded, root=True)
    return reloaded


@pytest.mark.parametrize("node_type", list(FACTORIES), ids=lambda t: t.__name__)
def test_roundtrip_covers_every_node_type(node_type):
    """Each concrete node type round-trips with identical JSON and ``as_string``."""
    tree = FACTORIES[node_type]()
    assert any(type(node) is node_type for node in tree.preorder_traversal())
    _assert_roundtrip(tree)


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_roundtrip_corpus_programs(name):
    """Each corpus program's schedule tree round-trips.

    The SDFG is first canonicalized through a native JSON round trip (as a loaded ``.sdfg`` would be), so that
    deserialization-only normalizations such as the parenthesization of library-node lambda properties do not make the
    original and reloaded ``as_string`` differ.
    """
    prog, _ = _CORPUS[name]
    sdfg = dace.SDFG.from_json(prog.to_sdfg(simplify=True).to_json())
    _assert_roundtrip(sdfg.as_schedule_tree())


@pytest.mark.parametrize("name", ["axpy", "matmul"])
def test_reloaded_tree_compiles_and_matches(name):
    """An SDFG rebuilt from a reloaded tree computes the same results as the original program's SDFG."""
    prog, factory = _CORPUS[name]
    sdfg = prog.to_sdfg(simplify=True)
    reloaded = _reload(sdfg.as_schedule_tree())
    new_sdfg = reloaded.as_sdfg()
    new_sdfg.name = f"{sdfg.name}_serialize_roundtrip"

    args = factory()
    expected_args = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    actual_args = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    expected = sdfg.compile()(**expected_args)
    actual = new_sdfg.compile()(**actual_args)

    for key, value in expected_args.items():
        if isinstance(value, np.ndarray):
            assert np.allclose(actual_args[key], value), f'Argument "{key}" differs'
    if expected is None:
        assert actual is None
    else:
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        assert len(actual) == len(expected)
        for e, a in zip(expected, actual):
            assert np.allclose(a, e), "Return value differs"


if __name__ == "__main__":
    for node_type in FACTORIES:
        test_roundtrip_covers_every_node_type(node_type)
    for program in sorted(_CORPUS):
        test_roundtrip_corpus_programs(program)
