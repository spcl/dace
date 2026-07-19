# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the in-path schedule-tree callback discrepancy check: the parallel
lowering check compares the classic frontend's callbacks (ground truth for
intent, recorded in ``sdfg.callback_mapping``) against the nextgen tree's
``PythonCallbackNode`` count, and fails the parse when nextgen falls back
more than classic does.
"""
import warnings

import pytest

import dace
from dace.frontend.python.nextgen.common import UnsupportedFeatureError


@pytest.fixture(autouse=True)
def _isolate_stree_config(monkeypatch):
    """Environment variables override ``set_temporary`` in dace's config
    (e.g. a triage run exporting DACE_frontend_stree_callback_check=warn);
    strip them so these tests control the configuration."""
    monkeypatch.delenv('DACE_frontend_stree_callback_check', raising=False)
    monkeypatch.delenv('DACE_frontend_stree_report', raising=False)


def dace_inhibitor(f):
    return f


@dace_inhibitor
def intended_callback(x):
    return x * 3


def _force_fallback(monkeypatch):
    """Monkeypatch the elementwise mechanism so every computation falls back
    to a Python callback, on programs the classic frontend lowers cleanly."""
    from dace.frontend.python.nextgen.lowering.mechanisms import elementwise

    def raising(*args, **kwargs):
        raise UnsupportedFeatureError('forced fallback', category='test-category')

    monkeypatch.setattr(elementwise, 'emit_computation', raising)


def test_clean_program_passes():

    @dace.program
    def clean(A: dace.float64[10]):
        A[0] = A[1] + 1.0

    clean.to_sdfg()


def test_intended_callback_passes():
    """A callback mirrored in classic's callback_mapping is not a discrepancy."""

    @dace.program
    def with_callback(A: dace.float64[10]):
        y = intended_callback(5)
        A[0] = y

    with_callback.to_sdfg()


def test_discrepancy_raises(monkeypatch):
    _force_fallback(monkeypatch)

    @dace.program
    def clean(A: dace.float64[10]):
        A[0] = A[1] + 1.0

    with pytest.raises(RuntimeError, match='discrepancy') as excinfo:
        clean.to_sdfg()
    # The error lists the categorized reasons of the excess callbacks
    assert 'test-category' in str(excinfo.value)


def test_discrepancy_warn_mode(monkeypatch):
    _force_fallback(monkeypatch)

    @dace.program
    def clean(A: dace.float64[10]):
        A[0] = A[1] + 1.0

    with dace.config.set_temporary('frontend', 'stree_callback_check', value='warn'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clean.to_sdfg()
    assert any('discrepancy' in str(w.message) for w in caught)


def test_discrepancy_off_mode(monkeypatch):
    _force_fallback(monkeypatch)

    @dace.program
    def clean(A: dace.float64[10]):
        A[0] = A[1] + 1.0

    with dace.config.set_temporary('frontend', 'stree_callback_check', value='off'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clean.to_sdfg()
    assert not any('discrepancy' in str(w.message) for w in caught)


if __name__ == '__main__':
    import os
    os.environ.pop('DACE_frontend_stree_callback_check', None)
    os.environ.pop('DACE_frontend_stree_report', None)
    test_clean_program_passes()
    test_intended_callback_passes()
    test_discrepancy_raises(pytest.MonkeyPatch())
    test_discrepancy_warn_mode(pytest.MonkeyPatch())
    test_discrepancy_off_mode(pytest.MonkeyPatch())
