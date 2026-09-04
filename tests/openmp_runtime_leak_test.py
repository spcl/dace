# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collecting the suite must not pull an OpenMP runtime into the interpreter.

Six modules used to ``ctypes.CDLL("libgomp.so.1", RTLD_GLOBAL)`` at module scope and a seventh
compiled and ran an SDFG there, so merely COLLECTING them -- not running them -- left libgomp
mapped for every test that came after. On the heterogeneous runner that made ScaLAPACK's ``pdgemm``
return a different wrong product on every call at two ranks: correct operands, correct descriptors,
no error, just wrong numbers. Collecting the three directories those modules live in reproduced it;
collecting ten others did not, and ``LD_PRELOAD``-ing libgomp alone reproduced it from a
single-file collection.

Collection-time imports are shared by the whole session, so this is a property of the suite rather
than of any one test, and it is asserted the only way it can be: by collecting in a child and
asking that child what it mapped.
"""
import os
import subprocess
import sys

import pytest

#: Where the leaks were. Collecting these is enough; the rest of the suite adds only time.
SUSPECT_DIRS = ('tests/library', 'tests/passes', 'tests/sdfg')

#: Every OpenMP runtime soname, matched against the child's own memory map.
OMP_RUNTIME_MARKERS = ('libgomp', 'libomp', 'libiomp', 'libnvomp')

#: A plugin for the child: report what is mapped once collection has imported everything.
REPORTER = '''
def pytest_collection_finish(session):
    try:
        with open("/proc/self/maps") as maps:
            mapped = maps.read()
    except OSError:
        mapped = ""
    hits = sorted({{line.rsplit(" ", 1)[-1] for line in mapped.splitlines()
                    if any(marker in line for marker in {markers!r})}})
    print("OMPRUNTIME " + ("|".join(hits) if hits else "(none)"))
'''


@pytest.mark.skipif(not os.path.exists('/proc/self/maps'), reason='needs a Linux memory map')
def test_collecting_the_suite_maps_no_openmp_runtime(tmp_path):
    """No collected module may load an OpenMP runtime as an import side effect."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plugin = tmp_path / 'ompreport.py'
    plugin.write_text(REPORTER.format(markers=OMP_RUNTIME_MARKERS))

    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '-p', 'ompreport', '--collect-only', '-q', '-s',
            '--continue-on-collection-errors', '-p', 'no:cacheprovider', *SUSPECT_DIRS
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={
            **os.environ, 'PYTHONPATH': f'{tmp_path}{os.pathsep}' + os.environ.get('PYTHONPATH', ''),
            'CUDA_VISIBLE_DEVICES': ''
        },
    )

    reported = [line for line in result.stdout.splitlines() if line.startswith('OMPRUNTIME ')]
    assert reported, f'the child never reported:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}'
    assert reported[-1] == 'OMPRUNTIME (none)', (
        f'collecting {" ".join(SUSPECT_DIRS)} loaded an OpenMP runtime: {reported[-1]}. A module '
        f'is dlopening one, or compiling and running an SDFG, at import rather than inside a test.')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
