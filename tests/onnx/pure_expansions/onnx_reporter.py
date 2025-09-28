"""
Simple plugin to record the results of the ONNX op cases.

It's very similar to the built-in one, but with an extra check for whether the
schemas are defined before trying to add them.

Can be used by begin enabled as a pytest plugin, for example by passing -p
'tests.pure_expansions.onnx_reporter'

It writes a report to both the pytest terminal output, and the file
'daceml/onnx_coverage.txt'
"""

import os

import pytest
import onnx.backend.test.report
import dace

COVERAGE = onnx.backend.test.report.Coverage()
MARKS = {}


def _add_mark(mark, bucket) -> None:
    proto = mark.args[0]
    if isinstance(proto, list):
        assert len(proto) == 1
        proto = proto[0]
    if proto is not None:
        for node in proto.graph.node:
            if not onnx.defs.has(node.op_type):
                return
        COVERAGE.add_proto(proto, bucket, mark.args[1] == 'RealModel')


def pytest_runtest_call(item):
    mark = item.get_closest_marker('onnx_coverage')
    if mark:
        assert item.nodeid not in MARKS
        MARKS[item.nodeid] = mark


def pytest_runtest_logreport(report):
    if (report.when == 'call' and report.outcome == 'passed'
            and report.nodeid in MARKS):
        mark = MARKS[report.nodeid]
        _add_mark(mark, 'passed')


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus):
    for mark in MARKS.values():
        _add_mark(mark, 'loaded')
    COVERAGE.report_text(terminalreporter)
    terminalreporter.write("\n")

    # write out to a file
    with open(
            os.path.join(os.path.dirname(dace.__file__),
                         "onnx_coverage.txt"), "w") as report_f:
        COVERAGE.report_text(report_f)
        report_f.write("\n")
