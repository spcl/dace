# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-compiler-family defaults for ``compiler.*.args``.

The shipped flags are GCC/Clang spellings, several of which nvc++ rejects outright. Family names
match CMake's ``CMAKE_<LANG>_COMPILER_ID``, lowercased.
"""
import functools
import os
import subprocess
from typing import Dict, Optional, Tuple

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
def _predefined_macros(executable: str) -> Optional[Dict[str, str]]:
    """``{macro: value}`` for everything ``executable`` predefines, or None if it cannot be probed."""
    try:
        probe = subprocess.run([executable, '-dM', '-E', '-x', 'c++', '-'],
                               input='',
                               capture_output=True,
                               text=True,
                               timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    macros: Dict[str, str] = {}
    for line in probe.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) > 1 and parts[0] == '#define':
            macros[parts[1]] = parts[2] if len(parts) > 2 else ''
    return macros


def detect(executable: str) -> str:
    """Compiler family of ``executable``, from the macros it predefines.

    Asks the compiler rather than reading its filename, which a wrapper or ccache shim can change.
    """
    defined = _predefined_macros(executable)
    if defined is None:
        return FALLBACK_FAMILY
    for macro, family in FAMILY_MACROS:
        if macro in defined:
            return family
    return FALLBACK_FAMILY


def detect_version(executable: str) -> Optional[Tuple[int, ...]]:
    """``(major, minor, patch)`` of ``executable``, or None if it cannot be determined.

    From the same predefined macros as :func:`detect`, not from the ``--version`` banner: invoked as
    ``c++`` a GCC banner never names its vendor, and distributions rewrite the rest of the line.
    """
    defined = _predefined_macros(executable)
    if defined is None:
        return None
    # clang defines __GNUC__ too, so its own macros are checked first (as in FAMILY_MACROS).
    triples = {
        'clang': ('__clang_major__', '__clang_minor__', '__clang_patchlevel__'),
        'gnu': ('__GNUC__', '__GNUC_MINOR__', '__GNUC_PATCHLEVEL__')
    }
    names = triples.get(detect(executable))
    if names is None or names[0] not in defined:
        return None
    try:
        return tuple(int(defined.get(name, '0')) for name in names)
    except ValueError:
        return None


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
