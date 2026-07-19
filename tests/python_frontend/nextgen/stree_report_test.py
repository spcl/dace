# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the schedule-tree gap report: the parallel lowering check appends
one JSON line per parsed program when ``frontend.stree_report`` is set, and
the ``python -m dace.frontend.python.nextgen.coverage`` CLI aggregates the
collected lines into a prioritized gap worklist.
"""
import contextlib
import io
import json

import pytest

import dace
from dace.frontend.python.nextgen import coverage
from dace.frontend.python.nextgen.common import UnsupportedFeatureError


def dace_inhibitor(f):
    return f


@dace_inhibitor
def intended_callback(x):
    return x * 3


def _force_fallback(monkeypatch):
    from dace.frontend.python.nextgen.lowering.mechanisms import elementwise

    def raising(*args, **kwargs):
        raise UnsupportedFeatureError('forced fallback', category='test-category')

    monkeypatch.setattr(elementwise, 'emit_computation', raising)


def _parse_report(path):
    with open(path, 'r') as report:
        return [json.loads(line) for line in report if line.strip()]


def test_report_lines_written(tmp_path):
    report_path = str(tmp_path / 'stree.jsonl')

    @dace.program
    def clean(A: dace.float64[10]):
        A[0] = A[1] + 1.0

    @dace.program
    def with_callback(A: dace.float64[10]):
        y = intended_callback(5)
        A[0] = y

    with dace.config.set_temporary('frontend', 'stree_report', value=report_path):
        clean.to_sdfg()
        with_callback.to_sdfg()

    # Program names are module-prefixed (e.g. 'stree_report_test_clean')
    records = {record['program'].rsplit('_test_', 1)[-1]: record for record in _parse_report(report_path)}
    assert set(records) == {'clean', 'with_callback'}

    assert records['clean']['nextgen_nodes'] == 0
    assert records['clean']['discrepancy'] is False

    callback_record = records['with_callback']
    assert callback_record['classic_callbacks'] == 1
    assert callback_record['nextgen_nodes'] == 1
    assert callback_record['discrepancy'] is False
    assert 'detected-callback' in callback_record['category_counts']
    assert any('detected-callback' in reason for reason in callback_record['reasons'])
    for key in ('test', 'statement_nodes', 'refset_warnings', 'pythonclass'):
        assert key in callback_record


def test_report_records_discrepancy(tmp_path, monkeypatch):
    """Discrepant programs are recorded even in warn mode (full-picture triage)."""
    report_path = str(tmp_path / 'stree.jsonl')
    _force_fallback(monkeypatch)

    @dace.program
    def clean(A: dace.float64[10]):
        A[0] = A[1] + 1.0

    with dace.config.set_temporary('frontend', 'stree_report', value=report_path):
        with dace.config.set_temporary('frontend', 'stree_callback_check', value='warn'):
            with pytest.warns(UserWarning, match='discrepancy'):
                clean.to_sdfg()

    (record, ) = _parse_report(report_path)
    assert record['discrepancy'] is True
    assert record['category_counts'] == {'test-category': 1}


def test_report_cli_aggregation(tmp_path):
    report_path = str(tmp_path / 'stree.jsonl')
    records = [
        # Two specializations of the same discrepant program: max wins
        {
            'program': 'prog_a',
            'test': 'tests/a_test.py::test_a',
            'classic_callbacks': 0,
            'nextgen_nodes': 1,
            'discrepancy': True,
            'reasons': ['[unknown-call:numpy.linalg.svd] no lowering for call "numpy.linalg.svd"'],
            'category_counts': {
                'unknown-call:numpy.linalg.svd': 1
            },
            'statement_nodes': 0,
            'refset_warnings': 0,
            'pythonclass': [],
        },
        {
            'program':
            'prog_a',
            'test':
            'tests/a_test.py::test_a',
            'classic_callbacks':
            0,
            'nextgen_nodes':
            2,
            'discrepancy':
            True,
            'reasons': [
                '[unknown-call:numpy.linalg.svd] no lowering for call "numpy.linalg.svd"',
                '[memlet-parse] cannot parse',
            ],
            'category_counts': {
                'unknown-call:numpy.linalg.svd': 1,
                'memlet-parse': 1
            },
            'statement_nodes':
            0,
            'refset_warnings':
            0,
            'pythonclass': [],
        },
        # A clean program and an intended-callback program: not discrepant
        {
            'program': 'prog_b',
            'test': None,
            'classic_callbacks': 1,
            'nextgen_nodes': 1,
            'discrepancy': False,
            'reasons': ['[detected-callback] call to Python callback "cb"'],
            'category_counts': {
                'detected-callback': 1
            },
            'statement_nodes': 0,
            'refset_warnings': 0,
            'pythonclass': [],
        },
    ]
    with open(report_path, 'w') as report:
        for record in records:
            report.write(json.dumps(record) + '\n')

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert coverage.main(['report', report_path]) == 0
    output = buffer.getvalue()

    assert '3 record(s), 2 program(s), 1 discrepant' in output
    # The discrepant program prints first, aggregated to its worst specialization
    assert 'prog_a: nextgen=2 classic=0' in output
    assert 'tests/a_test.py::test_a' in output
    # unknown-call qualnames aggregate into their own worklist
    assert 'numpy.linalg.svd' in output
    assert 'unknown-call' in output
    assert 'memlet-parse' in output


def test_reason_categories():
    reason = ('[opaque-syntax:Assign] non-canonical; '
              '[pyobject-propagation] operates on an opaque Python object')
    assert coverage.reason_categories(reason) == ['opaque-syntax:Assign', 'pyobject-propagation']
    assert coverage.reason_categories('[unknown-call:a.b] x [not-a-prefix] y') == ['unknown-call:a.b']


if __name__ == '__main__':
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_report_lines_written(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_report_records_discrepancy(Path(tmp), pytest.MonkeyPatch())
    with tempfile.TemporaryDirectory() as tmp:
        test_report_cli_aggregation(Path(tmp))
    test_reason_categories()
