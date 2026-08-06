# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import ctypes.util
import functools
import glob
import os
import shutil
import subprocess
import warnings

import dace.library

# Environment variables that may point at an OpenBLAS install. ``OPENBLAS_DIR`` is the
# name our own job scripts export (alongside ``LD_LIBRARY_PATH``); the others are the
# de-facto CMake/spack conventions. Each may name either an install prefix (with
# ``lib``/``lib64`` + ``include`` under it) or a lib dir directly.
OPENBLAS_ENV_VARS = ('OPENBLAS_DIR', 'OPENBLAS_ROOT', 'OPENBLAS_HOME', 'OpenBLAS_HOME')

# ``openblas_get_parallel()`` return codes, and the spack variant each corresponds to.
OPENBLAS_PARALLEL_NAMES = {
    0: 'sequential (spack threads=none)',
    1: 'pthreads (spack threads=pthreads)',
    2: 'OpenMP (spack threads=openmp)'
}


def _include_dir_for(libdir):
    """Sibling ``include`` dir of a lib dir, if it actually holds ``cblas.h`` -- spack/conda
    keep the OpenBLAS headers off the default compiler search path, so the build needs it."""
    inc = os.path.join(os.path.dirname(os.path.realpath(libdir)), 'include')
    return inc if os.path.isfile(os.path.join(inc, 'cblas.h')) else None


def _libopenblas_in(libdir):
    """First ``libopenblas.so*`` in ``libdir`` -- the exact ``.so`` symlink if present, else the
    highest-sorting versioned soname (``libopenblas.so.0``). ``None`` if the dir has none.

    A full path to a versioned soname links fine, so a spack/module install that ships only
    ``libopenblas.so.0`` (no unversioned dev symlink) is still usable."""
    if not libdir or not os.path.isdir(libdir):
        return None
    exact = os.path.join(libdir, 'libopenblas.so')
    if os.path.exists(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(libdir, 'libopenblas.so*')))
    return hits[-1] if hits else None


def _scan_ld_library_path():
    """Walk ``LD_LIBRARY_PATH`` directly for ``libopenblas.so*``. This is the crux of the
    spack/module case: ``ctypes.util.find_library`` consults only the ``ldconfig`` cache and
    the default loader dirs, so it returns ``None`` for an OpenBLAS that lives *solely* on
    ``LD_LIBRARY_PATH`` -- and the detection then wrongly reports "not installed" and MatMul
    falls back to a naive pure-Python-style loop. Globbing the path ourselves fixes that."""
    for d in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        lib = _libopenblas_in(d)
        if lib:
            return lib
    return None


def _resolve_lib_dir(soname_or_path):
    """Directory that actually contains a library, given whatever
    ``ctypes.util.find_library`` returned -- which on Linux is often a bare soname
    (``libopenblas.so.0``) rather than an absolute path. Resolved against
    ``LD_LIBRARY_PATH`` in that case (e.g. a ``spack load``/``module load`` install)."""
    if not soname_or_path:
        return None
    if os.path.isabs(soname_or_path) and os.path.exists(soname_or_path):
        return os.path.dirname(os.path.realpath(soname_or_path))
    for d in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        cand = os.path.join(d, soname_or_path) if d else ''
        if cand and os.path.exists(cand):
            return os.path.dirname(os.path.realpath(cand))
    return None


def _standalone_libopenblas():
    """``(lib_path, include_dir)`` for a single-library OpenBLAS install, or ``(None, None)``.

    spack and conda ship OpenBLAS as one ``libopenblas`` that provides BLAS, CBLAS, LAPACK
    and LAPACKE together -- with no reference-BLAS symlinks (``libblas``/``libcblas``/
    ``liblapacke``) and with the headers (``cblas.h``/``lapacke.h``) under the install's own
    ``include`` dir, off the default compiler search path. This locates that library and its
    header dir so both can be passed to the build directly.

    Search order: explicit env vars (:data:`OPENBLAS_ENV_VARS`), then the ``ldconfig`` cache
    (``find_library``), then a direct ``LD_LIBRARY_PATH`` scan for the off-cache spack/module
    case."""
    for var in OPENBLAS_ENV_VARS:
        root = os.environ.get(var)
        if not root or not os.path.isdir(root):
            continue
        # ``root`` may be an install prefix (lib/lib64 under it) or a lib dir itself.
        for libdir in (os.path.join(root, 'lib'), os.path.join(root, 'lib64'), root):
            lib = _libopenblas_in(libdir)
            if lib:
                return lib, _include_dir_for(libdir)
    found = ctypes.util.find_library('openblas')
    libdir = _resolve_lib_dir(found)
    if libdir:
        lib = _libopenblas_in(libdir) or found  # else let the linker resolve the soname
        return lib, _include_dir_for(libdir)
    lib = _scan_ld_library_path()
    if lib:
        return lib, _include_dir_for(os.path.dirname(lib))
    root = _spack_openblas_prefix()
    if root:
        for libdir in (os.path.join(root, 'lib'), os.path.join(root, 'lib64')):
            lib = _libopenblas_in(libdir)
            if lib:
                return lib, _include_dir_for(libdir)
    return None, None


@functools.lru_cache(maxsize=1, typed=True)
def _spack_openblas_prefix() -> str | None:
    """Install prefix of a spack-managed OpenBLAS, or ``None``.

    Last resort, after the env vars and the loader paths: an OpenBLAS that is merely *installed*
    under spack, never ``spack load``ed, is on no search path at all and every earlier probe misses
    it. Asking spack costs a subprocess, so it runs once per process and only once nothing cheaper
    found a library. A spack that is absent, slow, or reports several matching installs leaves the
    detection exactly where it was.
    """
    exe = shutil.which('spack')
    if not exe:
        return None
    try:
        out = subprocess.run([exe, 'location', '-i', 'openblas'],
                             capture_output=True,
                             text=True,
                             timeout=30,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    prefix = out.stdout.strip()
    return prefix if out.returncode == 0 and os.path.isdir(prefix) else None


def _openblas_present() -> bool:
    """Whether a ``libopenblas`` exists at all. The ldconfig cache counts: ``BLA_VENDOR=OpenBLAS``
    makes FindBLAS search the standard link paths, so it needs no resolvable directory."""
    return bool(ctypes.util.find_library('openblas')) or bool(_standalone_libopenblas()[0])


def _system_blas_libs():
    """Generic distro BLAS shared libs (``liblapacke``/``libcblas``/``libblas``) on the loader
    path, or ``[]``. These are update-alternatives symlinks and may point at ANY implementation."""
    return [p for p in (ctypes.util.find_library(l) for l in ('lapacke', 'cblas', 'blas')) if p]


def _mapped_openblas_path() -> str | None:
    """Absolute path of the ``libopenblas`` this process actually mapped, or ``None``. Needed
    because the loader is given a bare soname, and Debian and spack both keep several flavors
    side by side -- only the concrete path tells the user which one they got."""
    try:
        with open('/proc/self/maps') as f:  # Linux-only; absent elsewhere, hence the guard
            for line in f:
                if 'libopenblas' in line:
                    return line.rstrip().rsplit(' ', 1)[-1]
    except OSError:
        pass
    return None


@functools.lru_cache(maxsize=None, typed=True)
def _openblas_threading_flavor() -> tuple[int | None, str | None, str | None]:
    """``(parallel_code, config, path)`` of the OpenBLAS that would be loaded, ``(None, None, None)``
    if none can be probed.

    ``threads=openmp``, ``threads=pthreads`` and ``threads=none`` all install a file named
    ``libopenblas.so``, so presence checks and filenames cannot tell them apart -- only calling
    ``openblas_get_parallel()`` can. Loaded ``RTLD_LOCAL`` (ctypes' default) so it cannot interpose
    on symbols numpy's own BLAS already provides, and cached so this costs one ``dlopen`` per
    process."""
    lib_path = _standalone_libopenblas()[0] or ctypes.util.find_library('openblas')
    if not lib_path:
        return None, None, None
    try:
        lib = ctypes.CDLL(lib_path)
        lib.openblas_get_parallel.restype = ctypes.c_int
        code = lib.openblas_get_parallel()
    except (OSError, AttributeError):
        # Unloadable here, or too old to export the symbol. Diagnostics only -- never fatal.
        return None, None, None
    try:
        lib.openblas_get_config.restype = ctypes.c_char_p
        raw = lib.openblas_get_config()
        config = raw.decode() if raw else None
    except AttributeError:
        config = None
    return code, config, (lib_path if os.path.isabs(lib_path) else _mapped_openblas_path() or lib_path)


@functools.lru_cache(maxsize=None, typed=True)
def _warn_unless_openmp_threaded() -> None:
    """Name the resolved threading flavor when it is not the OpenMP one. Zero args, so the
    ``lru_cache`` makes this warn at most once per process.

    Deliberately a warning and not an error: a pthreads or sequential OpenBLAS computes the
    right answer, it only times badly. A pthreads build spawns a pool of its own rather than
    sharing libgomp's, so called from inside DaCe's OpenMP maps the two pools multiply and
    oversubscribe the machine; BLAS-heavy kernels (gemm, syrk, symm, cholesky, ...) then get slow
    and non-deterministic with nothing in a results table revealing why. Note the pthreads build
    still *seeds* its pool from ``OMP_NUM_THREADS`` -- it is the separate pool that hurts, so
    matching the two thread counts does not make the arms comparable."""
    code, config, path = _openblas_threading_flavor()
    if code is None or code == 2:
        return
    flavor = OPENBLAS_PARALLEL_NAMES.get(code, f'unknown ({code})')
    effect = ('spawns a thread pool of its own instead of sharing libgomp\'s, so inside DaCe OpenMP '
              'maps the two pools multiply and oversubscribe the machine'
              if code == 1 else 'leaves every BLAS call single-threaded')
    warnings.warn(
        f'OpenBLAS at {path} is the {flavor} build: it {effect}. Results stay correct, '
        'timings do not -- load an OpenBLAS built with threads=openmp before measuring. '
        f'openblas_get_config()={config}',
        stacklevel=2)


@dace.library.environment
class OpenBLAS:

    # Two install shapes: distro alternatives (liblapacke/libcblas/libblas pointing at OpenBLAS,
    # headers on the default path) and a lone libopenblas from spack/conda (headers under its
    # own include dir). The mode names say who locates the library, which is what the build needs.

    cmake_minimum_version = "3.6"
    cmake_compile_flags = []
    cmake_files = []

    headers = ["cblas.h", "lapacke.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def _mode():
        """Who locates the library: ``'find_package'`` (CMake's FindBLAS does), ``'direct_link'``
        (we pass the full path), or ``None`` (no OpenBLAS).

        ``find_package`` pins ``BLA_VENDOR=OpenBLAS``, so FindBLAS searches for ``libopenblas``
        specifically. The distro alternatives resolving is NOT sufficient evidence -- they point
        at any implementation -- so an OpenBLAS must exist too, or configure fails with
        "Could NOT find BLAS". Existence is the ldconfig-aware check, not the path-resolving one:
        FindBLAS searches the standard link paths, while a full path is needed only to link one
        directly.
        """
        if _system_blas_libs() and _openblas_present():
            return 'find_package'
        return 'direct_link' if _standalone_libopenblas()[0] else None

    @staticmethod
    def cmake_packages():
        # direct_link passes a full library path below, and find_package(BLAS REQUIRED) cannot
        # be satisfied for an off-default-path spack/conda install -- asking would only fail.
        return ["LAPACK", "BLAS"] if OpenBLAS._mode() == 'find_package' else []

    @staticmethod
    def cmake_variables():
        return {"BLA_VENDOR": "OpenBLAS"} if OpenBLAS._mode() == 'find_package' else {}

    @staticmethod
    def cmake_link_flags():
        # These vars only exist once find_package(LAPACK/BLAS) has run.
        return ["${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"] if OpenBLAS._mode() == 'find_package' else []

    @staticmethod
    def cmake_includes():
        _, inc = _standalone_libopenblas()
        return [inc] if inc else []

    @staticmethod
    def cmake_libraries():
        # Driven by _mode() like the other four, so is_installed() cannot report an OpenBLAS
        # the mode has already rejected.
        mode = OpenBLAS._mode()
        if mode is None:
            return []
        _warn_unless_openmp_threaded()  # only once an OpenBLAS is actually going to be used
        if mode == 'find_package':
            return _system_blas_libs()
        lib, _ = _standalone_libopenblas()
        return [lib]

    @staticmethod
    def is_installed():
        return len(OpenBLAS.cmake_libraries()) > 0
