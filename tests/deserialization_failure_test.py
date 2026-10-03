# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A failed deserialization must be reported where it happened, not smuggled onward as a placeholder.

``SerializableObject`` exists so an SDFG carrying a type THIS build never registered still round-trips
(the ``testing.deserialize_exception`` config text names the case: "a missing library node"). It used
to be handed out for registered types too, whenever their parser raised -- so an unparsable shape
became a placeholder, the ``Subset`` holding it became one, then the ``Memlet``, then the block, and
the load finally died on ``Expected ControlFlowBlock, got SerializableObject``: an error that names
neither the real failure nor where it happened.
"""
import pytest

import dace
from dace import serialize
from dace.config import set_temporary
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def context():
    return {'version': dace.__version__}


def test_an_unregistered_type_still_becomes_a_placeholder(monkeypatch):
    """The documented case, unchanged: this build cannot know the type, so keep the raw JSON.

    ``testing.deserialize_exception`` is the knob that turns this placeholder into a raise, and CI
    exports it, so the ambient value would otherwise decide which of the two behaviours below runs.
    Both are pinned and asserted here. The environment override wins over ``Config.set``, hence the
    ``delenv``."""
    unknown = {'type': 'TransformationFromTheFuture', 'attributes': {}}
    monkeypatch.delenv('DACE_testing_deserialize_exception', raising=False)

    with set_temporary('testing', 'deserialize_exception', value=False):
        with pytest.warns(UserWarning, match='Failed to deserialize element'):
            got = serialize.from_json(unknown, context())
        assert isinstance(got, serialize.SerializableObject)
        assert got.typename == 'TransformationFromTheFuture'

    with set_temporary('testing', 'deserialize_exception', value=True):
        with pytest.raises(KeyError, match='TransformationFromTheFuture'):
            serialize.from_json(unknown, context())


def test_a_registered_type_that_fails_to_parse_raises_where_it_failed():
    """``Range`` IS registered, so a parse failure is a genuine error about data we understand.
    The original exception must survive -- callers catch these by type -- carrying a note that says
    which type was being read."""
    broken = {'type': 'Range', 'ranges': [{'start': '0', 'end': '@@bad@@', 'step': '1', 'tile': '1'}]}

    with pytest.raises(Exception) as excinfo:
        serialize.from_json(broken, context())

    assert not isinstance(excinfo.value, TypeError) or 'ControlFlowBlock' not in str(excinfo.value), \
        'the failure must not resurface later as a placeholder type mismatch'
    assert 'while deserializing a Range' in (excinfo.value.__notes__ or []), \
        f'no note naming the failing type: {excinfo.value.__notes__!r}'


def test_a_registered_failure_does_not_become_a_placeholder():
    """The load must not hand back a lossy object it can neither validate nor round-trip.

    Written as try/fail rather than ``pytest.raises``: the natural spelling puts the ``assert`` inside
    the ``with`` block, where its own ``AssertionError`` satisfies the expected exception and the test
    passes no matter what came back.
    """
    broken = {'type': 'Range', 'ranges': [{'start': '0', 'end': '@@bad@@', 'step': '1', 'tile': '1'}]}
    try:
        got = serialize.from_json(broken, context())
    except Exception:
        return
    pytest.fail(f'returned {type(got).__name__} instead of raising')


def test_add_branch_refuses_a_non_region():
    """``ConditionalBlock.from_json`` builds branches directly and nothing downstream type-checks the
    region, so a wrong object here would sit in the branch list and fail in some later pass instead.
    ``add_node`` already checks its argument; this is the same check for the same reason."""
    block = ConditionalBlock('cond')

    with pytest.raises(TypeError, match='Expected ControlFlowRegion'):
        block.add_branch(None, serialize.SerializableObject({}, 'NotARegion'))
    assert block.branches == [], 'a rejected branch must not have been recorded'

    block.add_branch(None, ControlFlowRegion('body'))
    assert len(block.branches) == 1


if __name__ == '__main__':
    pytest.main([__file__])
