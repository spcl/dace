# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Machine-global build caches shared by both build backends.

Two things are expensive to produce and identical across the SDFGs built on one machine:

* the precompiled ``<dace/dace.h>``. The runtime umbrella header dominates the compile time of a
  small kernel (~1s of parsing and template instantiation), and every generated translation unit
  includes it;
* CMake's configure results -- the ``project()`` compiler/ABI detection plus every
  ``find_package`` -- which a fresh build folder otherwise repeats for every single SDFG.

Both are cached outside the build folder, keyed on everything that can change them, so the Nth
SDFG pays neither cost. Both caches are advisory: a stale or mismatched PCH is ignored by the
compiler, and a stale configure seed is simply re-derived by CMake, so a cache miss or a bad
entry can only cost speed, never correctness.

The cache lives in RAM (``/dev/shm``) by default. On HPC nodes the user cache dir is usually
NFS-backed ``$HOME``, and re-reading a ~100 MB PCH over NFS for every translation unit costs more
than the PCH saves. Override the root with ``DACE_BUILD_CACHE_DIR``.
"""
import getpass
import hashlib
import os
import re
import shutil
from typing import Callable, List, Optional, Sequence

#: Name of the generated one-line header that pulls in the DaCe runtime umbrella.
PREWARM_HEADER = 'dace_prewarm.h'


def cache_root(kind: str) -> str:
    """Root directory for one cache ``kind``, RAM-backed when possible.

    Per-user, because ``/dev/shm`` is shared by everyone on the node and the entries are written
    with the creating user's ownership.
    """
    root = os.environ.get('DACE_BUILD_CACHE_DIR')
    if not root:
        if os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK):
            root = os.path.join('/dev/shm', f'dace_build_cache_{getpass.getuser()}')
        else:
            root = os.path.expanduser('~/.cache/dace/build_cache')
    return os.path.join(root, kind)


def signature(*parts: object) -> str:
    """Short stable digest of everything an entry depends on."""
    return hashlib.sha256('\0'.join(str(p) for p in parts).encode()).hexdigest()[:16]


def newest_mtime(directory: str) -> float:
    """Modification time of the most recently touched file under ``directory``."""
    newest = 0.0
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass
    return newest


def ensure_dace_pch(cxx: str, pch_flags: Sequence[str], runtime_inc: str, runtime_mtime: float,
                    run: Callable[[List[str]], None]) -> Optional[List[str]]:
    """Precompile ``<dace/dace.h>`` once per (compiler, flags) and cache it.

    Returns the extra ``-I``/``-include`` flags that make g++/clang++ pick up the cached PCH, or
    ``None`` when one could not be produced (the caller then compiles normally -- only speed is
    affected). ``pch_flags`` must be the flags the translation units are actually compiled with,
    minus per-program ``-D``/``-I``, which the compiler tolerates as extras on the compile line.

    An invalid or flag-mismatched PCH is silently ignored by the compiler, so this can never change
    the produced object; the only failure mode is the one-off PCH build itself, which is swallowed.
    """
    try:
        pch_dir = os.path.join(cache_root('pch'), signature(cxx, runtime_inc, *pch_flags))
        header = os.path.join(pch_dir, PREWARM_HEADER)
        gch = header + '.gch'
        # Strictly newer: a .gch sharing the newest runtime header's mtime counts as stale, since the
        # compiler would otherwise keep silently using a PCH built from the pre-edit headers.
        if not (os.path.isfile(gch) and os.path.getmtime(gch) > runtime_mtime):
            os.makedirs(pch_dir, exist_ok=True)
            if not os.path.isfile(header):
                with open(header, 'w') as f:
                    f.write('#include <dace/dace.h>\n')
            # Compile to a per-process temp then rename into place, so a concurrent build (pytest -n4
            # shares this global cache) can never observe a half-written .gch.
            tmp_gch = f'{gch}.tmp.{os.getpid()}'
            run([cxx] + list(pch_flags) + ['-I', runtime_inc, '-x', 'c++-header', header, '-o', tmp_gch])
            os.replace(tmp_gch, gch)
        return ['-I', pch_dir, '-include', PREWARM_HEADER]
    except Exception:
        return None  # any trouble -> compile without the PCH


# ---------------------------------------------------------------------------
# CMake configure cache
# ---------------------------------------------------------------------------

#: The configure state worth transplanting into a fresh build folder. ``CMakeCache.txt`` holds the
#: ``find_package`` results; ``CMakeFiles/<cmake-version>/`` holds the compiler identification and
#: ABI detection that ``project()`` performs. Seeding only one of the two is close to worthless --
#: each still forces the other half of the work -- while both together turn a ~0.95s fresh configure
#: into a ~0.25s reconfigure. Deliberately NOT the whole ``CMakeFiles/``: under Ninja that directory
#: also holds the build's object files, which must never be transplanted between programs.
_CACHE_FILE = 'CMakeCache.txt'
_CMAKEFILES = 'CMakeFiles'


def _version_dir(cmakefiles: str) -> Optional[str]:
    """The ``CMakeFiles/<cmake-version>/`` subdirectory holding the compiler detection results."""
    try:
        entries = [e for e in os.listdir(cmakefiles) if e[:1].isdigit() and os.path.isdir(os.path.join(cmakefiles, e))]
    except OSError:
        return None
    return max(entries) if entries else None


def seed_configure_cache(build_folder: str, key: str) -> bool:
    """Transplant a previously published configure result into a fresh ``build_folder``.

    Does nothing (returning ``False``) when the folder is already configured or no entry exists for
    ``key``, so the caller simply runs the normal -- then merely slower -- configure.
    """
    if os.path.isfile(os.path.join(build_folder, _CACHE_FILE)):
        return False  # already configured; its own cache is newer than anything we could seed
    entry = os.path.join(cache_root('configure'), key)
    cached_cache, cached_files = os.path.join(entry, _CACHE_FILE), os.path.join(entry, _CMAKEFILES)
    if not (os.path.isfile(cached_cache) and os.path.isdir(cached_files)):
        return False
    try:
        # CMake records the directory a cache was created in and REFUSES a cache it finds anywhere
        # else ("The current CMakeCache.txt directory ... is different than the directory ... where
        # CMakeCache.txt was created"), which aborts the configure outright. Retarget that one entry
        # so the transplanted cache belongs to this build folder.
        with open(cached_cache) as f:
            cache_text = f.read()
        cache_text = re.sub(r'^CMAKE_CACHEFILE_DIR:INTERNAL=.*$',
                            'CMAKE_CACHEFILE_DIR:INTERNAL=' + build_folder.replace('\\', '/'),
                            cache_text,
                            count=1,
                            flags=re.MULTILINE)
        with open(os.path.join(build_folder, _CACHE_FILE), 'w') as f:
            f.write(cache_text)
        shutil.copytree(cached_files, os.path.join(build_folder, _CMAKEFILES), dirs_exist_ok=True)
        return True
    except OSError:
        # A partially copied seed would make the configure slower, not wrong, but leave nothing
        # behind: CMake reads whatever is there, and a truncated cache is worse than no cache.
        drop_configure_cache(key)
        shutil.rmtree(os.path.join(build_folder, _CMAKEFILES), ignore_errors=True)
        return False


def publish_configure_cache(build_folder: str, key: str) -> None:
    """Publish a freshly configured ``build_folder`` as the entry for ``key``.

    Never overwrites an existing entry: a concurrent build that got there first published the same
    thing, and replacing a directory another process may be reading from is not worth the race.
    """
    entry = os.path.join(cache_root('configure'), key)
    if os.path.isdir(entry):
        return
    cache_file = os.path.join(build_folder, _CACHE_FILE)
    cmakefiles = os.path.join(build_folder, _CMAKEFILES)
    version = _version_dir(cmakefiles)
    if not (os.path.isfile(cache_file) and version):
        return
    staging = f'{entry}.tmp.{os.getpid()}'
    try:
        os.makedirs(os.path.join(staging, _CMAKEFILES), exist_ok=True)
        shutil.copy2(cache_file, os.path.join(staging, _CACHE_FILE))
        shutil.copytree(os.path.join(cmakefiles, version), os.path.join(staging, _CMAKEFILES, version))
        os.rename(staging, entry)  # atomic; fails harmlessly if another build won the race
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)


def drop_configure_cache(key: str) -> None:
    """Remove the entry for ``key``, so a configure failure cannot poison every later build."""
    shutil.rmtree(os.path.join(cache_root('configure'), key), ignore_errors=True)
