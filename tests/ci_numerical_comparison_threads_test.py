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
