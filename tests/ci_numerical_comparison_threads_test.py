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


#: A numeric assertion: the file compares VALUES, so what thread count produced them is part of what
#: it grades. A timing or structural harness carries none of these and may pin whatever it likes.
NUMERIC_ASSERTION = re.compile(r'\ballclose\b|\barray_equal\b|\bassert_allclose\b|\bassert_array_\w+\b')

#: ``OMP_NUM_THREADS`` pinned to 1 in Python source: an ``os.environ`` write in either spelling
#: (``setdefault(name, '1')`` / ``environ[name] = '1'``), or a shell line inside a docstring telling
#: the next reader to run it that way. ``(?!\d)`` keeps ``=10`` out; the separator class is what
#: keeps the ``!= '1'`` ASSERTION that enforces this same rule out.
SOURCE_PIN = re.compile(r'OMP_NUM_THREADS[\'"]?\]?\s*[,:=]\s*[\'"]?1[\'"]?(?!\d)')

#: An ``rst`` literal -- ``OMP_NUM_THREADS=1`` written in prose. Every occurrence in the tree is a
#: comment ABOUT the pin (this file's own rationale, the fork-hang measurement in
#: ``helpers/isolation.py``), never one in force, so the spans come out before the scan.
RST_LITERAL = re.compile(r'``.*?``')

#: Waiver for the rare line where one thread is the thing under test rather than a shortcut around
#: reassociation -- a subprocess asserting that a guard ABORTS, say. Must carry a reason.
WAIVER = re.compile(r'#\s*openmp-pin-ok:\s*\S')

TESTS = pathlib.Path(__file__).resolve().parent


def value_asserting_modules():
    """Modules under ``tests/`` that assert numeric values.

    ``*.py``, not ``*_test.py``: the pin that prompted this scan was in ``pipelines.py``, the shared
    CloudSC harness the test modules import, which a ``_test`` suffix filter walks straight past.
    """
    this_file = pathlib.Path(__file__).resolve()
    found = []
    for path in sorted(TESTS.rglob('*.py')):
        # This module names every pattern it scans for, so it matches itself.
        if path.resolve() == this_file:
            continue
        text = path.read_text()
        if NUMERIC_ASSERTION.search(text):
            found.append((path, text))
    return found


def test_no_value_asserting_module_pins_openmp_to_a_single_thread():
    """The rule the workflow scan enforces for CI, enforced for the test sources themselves.

    The rule is against ONE thread, not against a fixed count: pinning 4 keeps a run reproducible and
    still exercises the OpenMP path. What one thread hides is the mistake a parallelizing transform
    can actually commit -- a Map over a loop that carries a dependence is bit-exact on one thread and
    wrong on many. A module pinned to one thread, in its own environment or in a docstring telling
    the reader to, passes precisely that defect through, invisibly to the call site asserting the
    numbers.
    """
    modules = value_asserting_modules()
    assert modules, 'no value-asserting module found; this test would assert nothing'
    pinned = []
    for path, text in modules:
        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            # The waiver sits on the pinned line or in the comment block directly above it, which is
            # where a multi-line reason has to go.
            if any(WAIVER.search(above) for above in lines[max(0, lineno - 4):lineno]):
                continue
            if SOURCE_PIN.search(RST_LITERAL.sub('', line)):
                pinned.append(f'{path.relative_to(TESTS)}:{lineno}: {line.strip()}')
    assert not pinned, ('a numeric comparison pinned to one thread grades a build nobody ships and '
                        'cannot see a wrongly parallelized loop; raise the tolerance, not the pin:\n  ' +
                        '\n  '.join(pinned))


def test_the_source_scan_recognises_the_shapes_it_must_catch():
    """Both environment spellings, the docstring command line, and the negatives -- or it is decorative."""
    assert SOURCE_PIN.search("os.environ.setdefault('OMP_NUM_THREADS', '1')")
    assert SOURCE_PIN.search('    OMP_NUM_THREADS=1 pytest tests/corpus/cloudsc -m integration')
    assert SOURCE_PIN.search("os.environ['OMP_NUM_THREADS'] = '1'")
    assert not SOURCE_PIN.search("os.environ.setdefault('OMP_NUM_THREADS', '4')")
    assert not SOURCE_PIN.search("os.environ.setdefault('OMP_NUM_THREADS', '16')")
    assert not SOURCE_PIN.search('    OMP_NUM_THREADS=4 pytest tests/corpus/cloudsc -m integration')
    assert not SOURCE_PIN.search("assert os.environ.get('OMP_NUM_THREADS') != '1'")
    assert not SOURCE_PIN.search(RST_LITERAL.sub('', 'the same parent under ``OMP_NUM_THREADS=1`` builds no team'))
    assert SOURCE_PIN.search(RST_LITERAL.sub('', "os.environ['OMP_NUM_THREADS'] = '1'  # ``see above``"))
    assert WAIVER.search("OMP_NUM_THREADS='1',  # openmp-pin-ok: the abort under test is per-thread")
    assert WAIVER.search('        # openmp-pin-ok: the child asserts a signal, not values')
    assert not WAIVER.search("OMP_NUM_THREADS='1',  # openmp-pin-ok:")


def test_the_numeric_assertion_scan_separates_values_from_timings():
    """A value comparison is in scope; a timing or structural harness is not."""
    assert NUMERIC_ASSERTION.search('assert np.allclose(out, want, rtol=1e-12)')
    assert NUMERIC_ASSERTION.search('np.testing.assert_array_almost_equal(a, b)')
    assert not NUMERIC_ASSERTION.search('assert elapsed < budget, f"took {elapsed}s"')
    assert not NUMERIC_ASSERTION.search('assert len(maps) == 314')
