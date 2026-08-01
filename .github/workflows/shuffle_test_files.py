# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Prints every test file under ``tests/``, shuffled with the seed given as the only argument.

Process-global caches (SymPy's ``@cacheit``, ``functools`` LRUs) survive across tests, so a test can pass or fail
depending on what ran before it. A fixed collection order hides that; feeding pytest a seeded permutation of the
files exposes it while still reproducing exactly from the seed alone.
"""
import pathlib
import random
import sys

# The patterns pytest.ini declares under ``python_files``.
PATTERNS = ('test_*.py', '*_test.py', '*_cudatest.py')


def main() -> int:
    seed = int(sys.argv[1])
    root = pathlib.Path(__file__).resolve().parents[2] / 'tests'
    files = sorted({str(path.relative_to(root.parent)) for pattern in PATTERNS for path in root.rglob(pattern)})
    random.Random(seed).shuffle(files)
    print('\n'.join(files))
    return 0


if __name__ == '__main__':
    sys.exit(main())
