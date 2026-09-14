# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-compiler-family defaults for ``compiler.*.args``.

The shipped flags are GCC/Clang spellings, several of which nvc++ rejects outright. Family names
match CMake's ``CMAKE_<LANG>_COMPILER_ID``, lowercased.
"""
import functools
import os
import subprocess
from typing import Tuple

from dace.config import Config

#: Predefined macro -> family, most specific first: clang and nvc++ also define ``__GNUC__``.
FAMILY_MACROS: Tuple[Tuple[str, str], ...] = (
    ('__NVCOMPILER', 'nvhpc'),
    ('__INTEL_LLVM_COMPILER', 'intelllvm'),
    ('__clang__', 'clang'),
    ('_MSC_VER', 'msvc'),
    ('__GNUC__', 'gnu'),
)

#: Used when the compiler cannot be probed; the shipped defaults are the GNU ones.
FALLBACK_FAMILY: str = 'gnu'


def host_compiler() -> str:
    """The C++ compiler DaCe pins CMake to."""
    return Config.get('compiler', 'cpu', 'executable') or os.environ.get('CXX') or 'c++'


@functools.lru_cache(maxsize=None, typed=True)
def detect(executable: str) -> str:
    """Compiler family of ``executable``, from the macros it predefines.

    Asks the compiler rather than reading its filename, which a wrapper or ccache shim can change.
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


def cpu_args() -> str:
    """``compiler.cpu.args`` with the shipped default swapped for the host family's default.

    Substitutes the default as a prefix rather than the whole string: DaCe appends to these args
    itself, so an appended flag must not drag the GCC defaults back in. A value that does not start
    with the shipped default was hand-written and is returned untouched.
    """
    configured = Config.get('compiler', 'cpu', 'args')
    shipped = Config.get_default('compiler', 'cpu', 'args')
    family = Config.get_metadata('compiler', 'cpu', 'args').get('default_' + detect(host_compiler()))
    if family is None or not configured.startswith(shipped):
        return configured
    return family + configured[len(shipped):]
