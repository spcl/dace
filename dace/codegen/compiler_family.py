# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-compiler-family defaults for the ``compiler.*.args`` flag strings.

The shipped defaults are written for GCC and Clang. nvc++ rejects four of those switches outright
(``-fno-math-errno``, ``-fno-trapping-math``, ``-freciprocal-math``, ``-Wno-unused-label``), so
without a family-specific default an NVHPC user cannot compile anything until they rewrite the
config by hand.

This is resolved here rather than in :mod:`dace.config` because a compiler family is not knowable
when the defaults are built. ``default_<platform>`` works at import time (``platform.system()`` is
always available); ``compiler.cpu.executable`` defaults to empty, meaning "whatever CMake picks", so
the family is only settled where the flags are assembled.

Family names match CMake's ``CMAKE_<LANG>_COMPILER_ID`` lowercased, so the CMake path and the
precompiled header -- which is built outside CMake -- name the same thing.
"""
import functools
import os
import subprocess
from typing import Tuple

from dace.config import Config

#: Predefined macro -> family, MOST SPECIFIC FIRST. Clang and nvc++ both define ``__GNUC__`` (they
#: advertise GCC compatibility), so a ``__GNUC__`` test placed first would claim all three.
FAMILY_MACROS: Tuple[Tuple[str, str], ...] = (
    ('__NVCOMPILER', 'nvhpc'),
    ('__INTEL_LLVM_COMPILER', 'intelllvm'),
    ('__clang__', 'clang'),
    ('_MSC_VER', 'msvc'),
    ('__GNUC__', 'gnu'),
)

#: Returned when the compiler cannot be probed -- not on PATH, not accepting ``-dM``, or too slow.
#: The shipped defaults are the GNU ones, so this keeps the previous behaviour rather than failing a
#: build over a detection that is only ever an optimization.
FALLBACK_FAMILY: str = 'gnu'


@functools.lru_cache(maxsize=None, typed=True)
def detect(executable: str) -> str:
    """Compiler family of ``executable``, from the macros it predefines.

    Asks the compiler instead of reading its filename: a distribution wrapper, a ``ccache`` shim or a
    site module can name nvc++ anything, and the macros are what the code is actually compiled with.
    Cached because the answer cannot change within a process for a given executable.
    """
    try:
        probe = subprocess.run([executable, '-dM', '-E', '-x', 'c++', '-'],
                               input='',
                               capture_output=True,
                               text=True,
                               timeout=60)
    except (OSError, subprocess.SubprocessError):
        return FALLBACK_FAMILY
    if probe.returncode != 0:
        return FALLBACK_FAMILY
    defined = {
        parts[1]
        for parts in (line.split() for line in probe.stdout.splitlines()) if len(parts) > 1 and parts[0] == '#define'
    }
    for macro, family in FAMILY_MACROS:
        if macro in defined:
            return family
    return FALLBACK_FAMILY


def host_compiler() -> str:
    """The C++ compiler this build will actually use.

    ``compiler.cpu.executable`` is DaCe's own knob and wins. ``$CXX`` is consulted next because CMake
    honours it when DaCe passes no ``-DCMAKE_CXX_COMPILER``: without this, ``CXX=nvc++`` gets a build
    where CMake picks nvc++ while the flags were chosen for GCC, and the configure dies on
    ``nvc++-Error-Unknown switch: -fno-math-errno``. Detection has to look where CMake looks.
    """
    return Config.get('compiler', 'cpu', 'executable') or os.environ.get('CXX') or 'c++'


def cpu_args() -> str:
    """``compiler.cpu.args``, specialized to the host compiler this build will use.

    The shipped default is the GCC/Clang spelling; a family needing different switches has its
    ``default_<family>`` sibling substituted for it.

    Substitution replaces the shipped default as a PREFIX rather than swapping the whole string,
    because appending is how DaCe itself adds flags: LIKWID appends ``-DLIKWID_PERFMON -fopenmp`` to
    activate its headers, PAPI a vectorization-report flag, and ``Config.append`` is literally
    ``current += value``. Treating any difference from the default as "the user chose this" would let
    those appends drag the GCC flags back in -- on NVHPC that turns a working build into
    ``nvc++-Error-Unknown switch: -fno-math-errno`` the moment instrumentation is switched on.

    A value that does NOT start with the shipped default was written by hand and is returned
    untouched: a family default is a better starting point, never a reason to overrule someone who
    stated what they wanted.
    """
    configured = Config.get('compiler', 'cpu', 'args')
    shipped = Config.get_default('compiler', 'cpu', 'args')
    family = Config.get_metadata('compiler', 'cpu', 'args').get('default_' + detect(host_compiler()))
    if family is None or not configured.startswith(shipped):
        return configured
    return family + configured[len(shipped):]
