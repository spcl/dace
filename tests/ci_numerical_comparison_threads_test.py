# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A numerical comparison is graded on the build that ships, and the build that ships is threaded.

Pinning ``OMP_NUM_THREADS=1`` makes a comparison bit-exact, which is why it keeps getting added:
a reduction stops reassociating and two generators agree to the last bit. What it also does is
grade a configuration nobody runs, so a generator that only diverges once its maps are split
across threads passes. That happened here -- ``test_readable_smoke`` compared with
``np.array_equal`` and was green for exactly as long as the step ran on one thread.

The answer is a tolerance sized for fp64 reassociation, not a thread count of one. This pins the
rule so the next person to meet a last-ulp failure reaches for the tolerance rather than the pin.
"""
import pathlib
import re

import pytest

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / '.github' / 'workflows'

#: ``OMP_NUM_THREADS`` written as a shell export or as a YAML ``env:`` entry, capturing the value.
SETTING = re.compile(r'OMP_NUM_THREADS\s*[:=]\s*[\'"]?(\d+)[\'"]?')


def _workflow_files():
    return sorted(WORKFLOWS.glob('*.yml')) + sorted(WORKFLOWS.glob('*.yaml'))


def test_no_workflow_pins_openmp_to_a_single_thread():
    """No CI step may set ``OMP_NUM_THREADS=1``.

    A step that needs a serial reference for one specific reason should say so at the call site --
    a single test forcing its own environment -- rather than serialising a whole phase, which
    silently downgrades every numerical comparison the phase happens to contain.
    """
    assert _workflow_files(), f'no workflow files under {WORKFLOWS}; this test would assert nothing'
    pinned = []
    for path in _workflow_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith('#'):
                continue
            match = SETTING.search(line)
            if match and int(match.group(1)) == 1:
                pinned.append(f'{path.name}:{lineno}: {line.strip()}')
    assert not pinned, ('a numerical comparison graded on one thread is graded on a build nobody '
                        'ships; raise the tolerance instead of the pin:\n  ' + '\n  '.join(pinned))


@pytest.mark.parametrize('value', ['1', '4'])
def test_the_scan_recognises_both_spellings(value):
    """The regex must catch the shell export and the YAML mapping, or the guard is decorative."""
    assert SETTING.search(f'        export OMP_NUM_THREADS={value}').group(1) == value
    assert SETTING.search(f"        OMP_NUM_THREADS: '{value}'").group(1) == value


#: The CloudSC numeric harness. A module importing one of these is comparing a TRANSFORMED graph
#: against a reference, which is the comparison the pin invalidates. Membership is derived from the
#: import rather than a path list so a new harness is covered the day it is written.
HARNESS_IMPORTS = ('compare_outputs', 'make_sequential', 'build_reference_outputs', 'numeric_check_from')

#: ``OMP_NUM_THREADS`` pinned to 1 in Python source: an ``os.environ`` write in either spelling
#: (``setdefault(name, '1')`` / ``environ[name] = '1'``), or a shell line inside a docstring telling
#: the next reader to run it that way. ``(?!\d)`` keeps ``=10`` out; the separator class is what
#: keeps the ``!= '1'`` ASSERTION that enforces this same rule out.
SOURCE_PIN = re.compile(r'OMP_NUM_THREADS[\'"]?\]?\s*[,:=]\s*[\'"]?1[\'"]?(?!\d)')

TESTS = pathlib.Path(__file__).resolve().parent


def _numeric_harness_modules():
    """Modules that compare a transformed CloudSC graph against a reference.

    ``*.py``, not ``*_test.py``: the pin that started this scan was in ``pipelines.py``, the shared
    harness the test modules import, which a ``_test`` suffix filter walks straight past.
    """
    this_file = pathlib.Path(__file__).resolve()
    found = []
    for path in sorted(TESTS.rglob('*.py')):
        # This module names every harness symbol it scans for, so it matches itself.
        if path.resolve() == this_file:
            continue
        text = path.read_text()
        if any(name in text for name in HARNESS_IMPORTS):
            found.append((path, text))
    return found


def test_no_cloudsc_numeric_harness_pins_openmp_to_a_single_thread():
    """The rule the workflow scan enforces for CI, enforced for the harnesses themselves.

    The rule is against ONE thread, not against a fixed count: pinning 4 keeps a run reproducible and
    still exercises the OpenMP path. Canonicalization's job is to EXPOSE parallelism, so the mistake
    it can make is a Map over a loop that carries a dependence -- bit-exact on one thread, wrong on
    many. A harness pinned to one thread -- in its own environment, or in a docstring telling the
    reader to -- passes precisely that defect through, invisibly to the call site asserting numbers.
    """
    modules = _numeric_harness_modules()
    assert modules, 'no CloudSC numeric harness found; this test would assert nothing'
    pinned = []
    for path, text in modules:
        for lineno, line in enumerate(text.splitlines(), start=1):
            if SOURCE_PIN.search(line):
                pinned.append(f'{path.relative_to(TESTS)}:{lineno}: {line.strip()}')
    assert not pinned, ('a CloudSC numeric comparison pinned to one thread grades a build nobody '
                        'ships and cannot see a wrongly parallelized loop:\n  ' + '\n  '.join(pinned))


def test_the_source_scan_recognises_the_shapes_it_must_catch():
    """The environment write, the docstring command line, and the negative -- or the guard is decorative."""
    assert SOURCE_PIN.search("os.environ.setdefault('OMP_NUM_THREADS', '1')")
    assert SOURCE_PIN.search('    OMP_NUM_THREADS=1 pytest tests/corpus/cloudsc -m integration')
    assert SOURCE_PIN.search('os.environ["OMP_NUM_THREADS"] = "1"')
    assert SOURCE_PIN.search("os.environ['OMP_NUM_THREADS'] = '1'")
    assert not SOURCE_PIN.search("os.environ.setdefault('OMP_NUM_THREADS', '4')")
    assert not SOURCE_PIN.search("os.environ.setdefault('OMP_NUM_THREADS', '16')")
    assert not SOURCE_PIN.search('    OMP_NUM_THREADS=4 pytest tests/corpus/cloudsc -m integration')
    assert not SOURCE_PIN.search("assert os.environ.get('OMP_NUM_THREADS') != '1'")
