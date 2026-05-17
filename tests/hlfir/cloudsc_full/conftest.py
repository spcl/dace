"""Shared fixtures for the full-CLOUDSC test family.

Every ``test_cloudsc_*`` module needs the same strict-FP DaCe C++
compiler override so the SDFG side matches the gfortran reference's
IEEE arithmetic; it lives here once instead of being copy-pasted per
module.
"""

import dace
import pytest


@pytest.fixture
def _strict_fp_cpu_args():
    """Match the gfortran reference's FP semantics on the SDFG side.

    DaCe's default ``-O3 -march=native -ffast-math`` lets gcc fuse and
    reassociate floating-point ops, diverging from the gfortran-built
    f2py reference (strict IEEE, no ``-ffast-math``).  Drop ``-ffast-
    math``, ``-O3``->``-O0``, disable FMA contraction; restore the
    previous flags afterwards.  The flag set is the LLVM-flang-portable
    core (see :data:`cloudsc_full._registries.CLOUDSC_F90FLAGS`).
    """
    prev = dace.Config.get('compiler', 'cpu', 'args')
    dace.Config.set(
        'compiler',
        'cpu',
        'args',
        value='-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off '
        '-Wno-unused-parameter -Wno-unused-label',
    )
    try:
        yield
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=prev)
