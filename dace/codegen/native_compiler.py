# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""No-CMake native build backend for DaCe.

Emits the ``g++`` / ``nvcc`` / ``ar`` / link commands directly and runs them via subprocess,
skipping the CMake configure overhead that dominates wall-clock for small kernels. Enabled with
``compiler.build_mode = native``. Linux only.

The module is self-contained: the only integration points elsewhere are the ``compiler.build_mode``
config key and a single dispatch branch in :func:`dace.codegen.compiler.configure_and_compile`. It
reads all other settings from :class:`dace.config.Config` and resolves libraries from the
environments the SDFG already collected -- there is no separate list of "supported" libraries.

Library resolution mirrors what CMake's ``find_package`` would do, but only for what a program
actually uses: the CUDA toolkit (for ``-L``/``-l``/rpath of ``cudart`` and bare names like
``cublas``) and MPI (via the wrapper compiler). Everything an environment can already express as a
concrete ``-I`` / ``-l`` / absolute path (cuBLAS, MKL, OpenBLAS, ...) is passed through. Anything
that would need real CMake -- an unknown ``find_package``, an environment shipping ``.cmake`` files,
or an unexpanded ``${...}`` -- raises a clear error telling the user to switch to ``build_mode=cmake``.
"""
import functools
import os
import re
import shlex
import shutil
import subprocess
import warnings
from typing import Dict, List, Optional

from dace.config import Config
from dace.dtypes import deduplicate
from dace.codegen import common
from dace.codegen import exceptions as cgx
from dace.codegen.compiler import _get_or_eval
from dace.codegen.target import make_absolute

#: ``compiler.build_type`` -> optimization/debug flags, appended after ``compiler.cpu.args`` so the
#: last ``-O`` wins exactly as CMake's ``CMAKE_CXX_FLAGS_<CONFIG>`` does.
_BUILD_TYPE_FLAGS = {
    'Debug': ['-O0', '-g'],
    'Release': ['-O3', '-DNDEBUG'],
    'RelWithDebInfo': ['-O2', '-g', '-DNDEBUG'],
    'MinSizeRel': ['-Os', '-DNDEBUG'],
}

#: The same, for nvcc (``CMAKE_CUDA_FLAGS_<CONFIG>``). Deliberately a separate table rather than
#: reusing the host one: nvcc rejects ``-Os`` outright (``nvcc fatal : 's': expected a number``), so
#: MinSizeRel must map to ``-O1``, and CMake's CUDA defaults differ from its host defaults for
#: exactly this reason.
_CUDA_BUILD_TYPE_FLAGS = {
    'Debug': ['-g'],
    'Release': ['-O3', '-DNDEBUG'],
    'RelWithDebInfo': ['-O2', '-g', '-DNDEBUG'],
    'MinSizeRel': ['-O1', '-DNDEBUG'],
}

#: Target names whose sources are GPU device code (compiled by nvcc, not g++). Native mode is
#: CUDA-only; a HIP/ROCm backend is rejected early with a clear error (use compiler.build_mode=cmake).
_GPU_TARGETS = ('cuda', 'experimental_cuda')


class _LinkSpec:
    """Include/library/flag fragments accumulated while resolving toolkits and environments."""

    def __init__(self):
        self.includes: List[str] = []  # -I values (no prefix)
        self.compile_flags: List[str] = []  # raw compile tokens
        self.libdirs: List[str] = []  # -L values (no prefix)
        self.libs: List[str] = []  # bare names for -l
        self.link_flags: List[str] = []  # raw link tokens (abs paths, -l.., -Wl,..)


def _dace_root() -> str:
    """``<repo>/dace`` -- the parent of this ``codegen`` package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cxx() -> str:
    exe = Config.get('compiler', 'cpu', 'executable')
    return make_absolute(exe) if exe else 'c++'


def _is_deferred(token: str) -> bool:
    """A ``${CMAKE_VAR}`` fragment that only CMake could expand."""
    return '${' in token


# ---------------------------------------------------------------------------
# Toolkit / library resolution
# ---------------------------------------------------------------------------

#: CUDA library / header files to locate. Runtime is mandatory; the math libraries live in a
#: separate ``math_libs`` subtree on the HPC SDK and are added when present (cuBLAS/cuSOLVER/... use
#: bare ``-l`` names that need those ``-L`` dirs).
_CUDA_LIB_FILES = ('libcudart.so', 'libcublas.so', 'libcusolver.so', 'libcufft.so', 'libcurand.so', 'libcusparse.so')
_CUDA_HEADER_FILES = ('cuda_runtime.h', 'cublas_v2.h', 'cusolverDn.h', 'cufft.h')


def _search_for(fname: str, roots: List[str]) -> Optional[str]:
    """Directory under any ``root`` containing ``fname``, handling both the standard CUDA layout
    (``<root>/lib64`` | ``<root>/include``) and the NVIDIA HPC-SDK layout
    (``<root>/{cuda,math_libs}/<ver>/targets/<arch>/{lib,include}``, nvcc under ``compilers/bin``)."""
    import glob
    kind = 'lib' if fname.endswith('.so') else 'include'
    subdirs = ['', 'lib64', 'lib', 'include', os.path.join('targets', 'x86_64-linux', kind)]
    for root in roots:
        if not root:
            continue
        for sub in subdirs:
            d = os.path.join(root, sub)
            if os.path.isfile(os.path.join(d, fname)):
                return d
        for base in (root, os.path.dirname(root.rstrip(os.sep))):
            for subtree in ('cuda', 'math_libs'):
                for d in sorted(glob.glob(os.path.join(base, subtree, '*', 'targets', '*', kind)), reverse=True):
                    if os.path.isfile(os.path.join(d, fname)):
                        return d
    return None


@functools.lru_cache(maxsize=None)
def _cuda_paths_cached(root: Optional[str], path_env: str) -> tuple:
    """Filesystem search for the CUDA toolkit. Cached because the toolkit does not move within a run;
    the inputs that determine the result (the config/env root hint and ``$PATH`` used by ``which``)
    are the cache key, so a later config/PATH change still re-resolves.
    """
    nvcc = None
    if root:
        cand = os.path.join(root, 'bin', 'nvcc')
        nvcc = cand if os.path.isfile(cand) else shutil.which('nvcc')
    else:
        nvcc = shutil.which('nvcc')
    if not nvcc:
        raise cgx.CompilerConfigurationError(
            'Native build cannot locate nvcc. Set compiler.cuda.path or the CUDA_HOME environment '
            'variable, add nvcc to PATH, or use compiler.build_mode=cmake.')

    nvcc_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    roots = [root, nvcc_root, os.path.dirname(nvcc_root), '/usr/local/cuda']

    lib_dirs = deduplicate([d for d in (_search_for(f, roots) for f in _CUDA_LIB_FILES) if d])
    header_roots = roots + [os.path.dirname(d) for d in lib_dirs]
    inc_dirs = deduplicate([d for d in (_search_for(h, header_roots) for h in _CUDA_HEADER_FILES) if d])
    if not lib_dirs or not inc_dirs:
        raise cgx.CompilerConfigurationError(
            f'Native build found nvcc at {nvcc} but could not locate libcudart.so / cuda_runtime.h. '
            f'Set compiler.cuda.path to the toolkit root, or use compiler.build_mode=cmake.')
    return nvcc, inc_dirs, lib_dirs


def _cuda_paths() -> tuple:
    """Return ``(nvcc, include_dirs, lib_dirs)`` for the CUDA toolkit, or raise a clear error.

    Locates nvcc (``compiler.cuda.path``/``$CUDA_HOME``/``$CUDA_PATH``/``which``), then finds the
    directories actually containing the CUDA runtime + math libraries and their headers by search --
    this is what ``find_package(CUDAToolkit)`` does, and unlike a fixed ``<root>/lib64`` it works for
    the HPC SDK (where the math libraries live under a separate ``math_libs`` tree). The search is
    cached (see :func:`_cuda_paths_cached`).
    """
    root = (Config.get('compiler', 'cuda', 'path') or os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'))
    return _cuda_paths_cached(root, os.environ.get('PATH', ''))


@functools.lru_cache(maxsize=None)
def _mpi_flags(mpicxx: str, path_env: str) -> tuple:
    """Query the MPI wrapper compiler for its include/lib flags, as
    ``(includes, libdirs, libs, compile_flags, link_flags)`` tuples (or raise).

    OpenMPI answers ``--showme:{incdirs,libdirs,libs}``; MPICH answers ``-compile_info`` /
    ``-link_info`` with full flag strings. Cached per wrapper (keyed also on ``$PATH``): the answer is
    fixed for a given toolchain and the subprocess launches are not free when many SDFGs are built.
    """

    def run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.run([mpicxx] + args, capture_output=True, text=True)
        except OSError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    incdirs, libdirs, libs = run(['--showme:incdirs']), run(['--showme:libdirs']), run(['--showme:libs'])
    if incdirs is not None and libdirs is not None and libs is not None:
        return tuple(incdirs.split()), tuple(libdirs.split()), tuple(libs.split()), (), ()

    compile_info, link_info = run(['-compile_info']), run(['-link_info'])
    if compile_info is not None and link_info is not None:
        cflags = tuple(t for t in shlex.split(compile_info) if t.startswith('-I'))
        lflags = tuple(t for t in shlex.split(link_info) if t.startswith(('-l', '-L', '-Wl', '/')))
        return (), (), (), cflags, lflags

    raise cgx.CompilerConfigurationError(
        f"Native build cannot query the MPI compiler wrapper '{mpicxx}'. Set compiler.mpi.executable "
        f"or use compiler.build_mode=cmake.")


def _resolve_mpi(spec: _LinkSpec) -> None:
    """Fill ``spec`` with MPI include/lib flags from the (cached) MPI wrapper query."""
    mpicxx = Config.get('compiler', 'mpi', 'executable') or 'mpicxx'
    includes, libdirs, libs, compile_flags, link_flags = _mpi_flags(mpicxx, os.environ.get('PATH', ''))
    spec.includes += includes
    spec.libdirs += libdirs
    spec.libs += libs
    spec.compile_flags += compile_flags
    spec.link_flags += link_flags


#: A bare library FILENAME, which several environments legitimately put in ``cmake_libraries`` --
#: ``ctypes.util.find_library()`` returns e.g. ``libblas.so.3`` (OpenBLAS), and the reference
#: ScaLAPACK environments hardcode ``libscalapack-mpich.so``. CMake normalizes such an item to
#: ``-l<stem>``; ``-llibblas.so.3`` is not resolvable by ld, so native must strip it the same way.
_LIB_FILENAME = re.compile(r'lib(.+?)\.(?:so|a)(?:\.\d+)*')


def _classify_library(spec: _LinkSpec, lib: str) -> None:
    """Route one ``cmake_libraries`` entry onto the link line."""
    lib = lib.strip()
    if not lib:
        return
    if lib.startswith(('-l', '-L', '-Wl')):
        spec.link_flags.append(lib)
    elif os.path.isabs(lib):  # absolute path to a .so/.a (MKL, OpenBLAS, reference libs)
        spec.link_flags.append(lib)
    else:  # bare name ("cublas") or a library filename ("libblas.so.3"), which becomes its stem
        filename = _LIB_FILENAME.fullmatch(lib)
        spec.libs.append(filename.group(1) if filename else lib)


def _resolve_environment(env, spec: _LinkSpec) -> None:
    """Resolve one collected environment into ``spec``, or raise if it needs real CMake."""
    name = env.__name__

    if _get_or_eval(env.cmake_files):
        files = ', '.join(_get_or_eval(env.cmake_files))
        raise cgx.CompilerConfigurationError(
            f"Native build cannot resolve environment '{name}': it ships CMake files ({files}). "
            f"Use compiler.build_mode=cmake for this program.")

    # find_package packages: MPI via the wrapper; the toolkit/threading ones are handled globally;
    # BLAS/LAPACK vendor selection only picks concrete libs that arrive via cmake_libraries().
    for pkg in _get_or_eval(env.cmake_packages):
        if pkg == 'MPI':
            _resolve_mpi(spec)
        elif pkg in ('CUDA', 'CUDAToolkit', 'Threads', 'OpenMP', 'BLAS', 'LAPACK'):
            continue
        else:
            raise cgx.CompilerConfigurationError(
                f"Native build cannot resolve environment '{name}': it needs find_package({pkg}), which "
                f"native mode does not implement. Use compiler.build_mode=cmake for this program.")

    for inc in _get_or_eval(env.cmake_includes):
        if not _is_deferred(inc):
            spec.includes.append(inc)

    # Headers stored alongside the environment imply its directory is an include path.
    env_dir = os.path.dirname(env._dace_file_path)
    headers = _get_or_eval(env.headers)
    header_groups = headers.values() if isinstance(headers, dict) else [headers]
    for group in header_groups:
        for header in group:
            if os.path.isabs(header):
                spec.includes.append(os.path.dirname(header))
            elif os.path.isfile(os.path.join(env_dir, header)):
                spec.includes.append(env_dir)
                break

    # An environment may return SEVERAL flags in one string (e.g. IntelMKLScaLAPACKMPICH returns its
    # whole "-L <dir> -lmkl_scalapack_lp64 ... -ldl" link line as a single entry). CMake pastes that
    # string into CMAKE_*_FLAGS, where the shell tokenizes it; native builds an argv list, so it must
    # tokenize here -- otherwise the whole string arrives as one quoted, unusable argument.
    # Deferred ${...} fragments are dropped per token: a known package (MPI) already supplied the real
    # flags, and an unknown one would have raised on cmake_packages above.
    def tokenize(flags):
        try:
            return [tok for flag in flags for tok in shlex.split(flag) if not _is_deferred(tok)]
        except ValueError as ex:  # unbalanced quote etc. -- keep the module's clear-error contract
            raise cgx.CompilerConfigurationError(f"Native build cannot parse a flag from environment '{name}': {ex}. "
                                                 f"Use compiler.build_mode=cmake for this program.")

    spec.compile_flags += tokenize(_get_or_eval(env.cmake_compile_flags))
    spec.link_flags += tokenize(_get_or_eval(env.cmake_link_flags))
    for lib in _get_or_eval(env.cmake_libraries):
        if not _is_deferred(lib):
            _classify_library(spec, lib)


# ---------------------------------------------------------------------------
# GPU architecture flags
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _nvcc_supported_arches(nvcc: str) -> Optional[frozenset]:
    """The set of ``sm_XX`` numbers ``nvcc`` can target, from ``--list-gpu-arch``; ``None`` if the
    probe fails (then no filtering is applied). Lets us drop archs a newer toolkit dropped -- e.g.
    CUDA 13 no longer builds ``sm_60`` -- so a stale ``compiler.cuda.cuda_arch`` cannot fail the build.

    Cached per nvcc binary: the toolkit's arch list is fixed, and the probe is a subprocess launch."""
    try:
        out = subprocess.run([nvcc, '--list-gpu-arch'], capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return frozenset(int(m) for m in re.findall(r'compute_(\d+)', out.stdout))


@functools.lru_cache(maxsize=None)
def _can_use_arch_native(nvcc: str) -> bool:
    """Whether ``nvcc -arch=native`` can resolve a local GPU. Native detection queries the driver, so
    it fails on a host without a visible CUDA device (e.g. a GPU-less build node); there we must fall
    back to the explicitly configured architectures instead of emitting an unbuildable command.

    Cached per nvcc binary: the local GPU visibility is fixed for a run, and the probe compiles."""
    try:
        out = subprocess.run([nvcc, '-arch=native', '--dryrun', '-x', 'cu', '-c', os.devnull, '-o', os.devnull],
                             capture_output=True,
                             text=True)
    except OSError:
        return False
    return out.returncode == 0


#: One ``compiler.cuda.cuda_arch`` entry: a number with an optional ``sm_``/``compute_`` prefix and an
#: optional feature suffix -- ``90``, ``sm_90`` and ``90a`` all name architecture 90.
_ARCH_TOKEN = re.compile(r'(?:sm_|compute_)?(\d+)([a-z]?)')


def _cuda_arch_flags(supported: Optional[set], allow_native: bool = True) -> List[str]:
    """CUDA ``-arch`` / ``-gencode`` flags, targeting what the CMake build targets.

    CMake compiles for the locally detected GPU and consults ``compiler.cuda.cuda_arch`` *only* when
    that detection fails -- ``DACE_CUDA_ARCHITECTURES_DEFAULT`` is read in the else-branch alone (see
    ``CMakeLists.txt``, "Default CUDA architectures in case native not found"). Native mirrors that:
    ``-arch=native`` lets nvcc detect the local GPU, and the configured architectures serve as the
    fallback for a host without one. Compiling them *in addition* would emit architectures the cmake
    build never produces -- a fatter binary and a slower compile -- so despite the config describing
    cuda_arch as "additional" (wording cmake has never honored), parity with the cmake artifact wins.

    Each fallback entry is normalized: an ``sm_``/``compute_`` prefix is stripped, an ``auto``/
    ``native`` entry is skipped, and a feature suffix (e.g. ``90a``) is preserved in the emitted flag
    while only the number is matched against the toolkit's supported set -- an architecture the
    toolkit dropped is skipped with a warning rather than failing the compile. Anything unparseable
    is warned about and skipped instead of being interpolated into an unbuildable
    ``arch=compute_sm_90``. An empty result raises rather than emitting no target at all.
    """
    if allow_native:
        return ['-arch=native']

    flags: List[str] = []
    for arch in Config.get('compiler', 'cuda', 'cuda_arch').strip().lower().split(','):
        arch = arch.strip()
        if not arch or arch in ('auto', 'native'):
            continue
        token = _ARCH_TOKEN.fullmatch(arch)
        if not token:
            warnings.warn(f'Native build: ignoring unparseable compiler.cuda.cuda_arch entry {arch!r}.')
            continue
        number, suffix = int(token.group(1)), token.group(2)
        if supported is not None and number not in supported:
            warnings.warn(f'Native build: the CUDA toolkit does not support architecture sm_{arch}; skipping it.')
            continue
        target = f'{number}{suffix}'
        flags += ['-gencode', f'arch=compute_{target},code=sm_{target}']
    if not flags:
        raise cgx.CompilerConfigurationError(
            'Native build: nvcc -arch=native found no local GPU and compiler.cuda.cuda_arch names no '
            'architecture this toolkit supports. Set compiler.cuda.cuda_arch to a supported target, or '
            'use compiler.build_mode=cmake.')
    return flags


def _newest_mtime(directory: str) -> float:
    newest = 0.0
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return newest


def _depfile_headers(depfile: str) -> List[str]:
    """The sources and headers a ``-MMD -MF`` compile recorded for one object; empty if unreadable.

    Walking fixed directories can only ever see the DaCe runtime and generated includes, so a header
    an environment contributes through its own ``-I`` directory would never invalidate the object.
    The compiler already knows exactly which files it opened, so ask it -- this is what
    CMake/make/ninja do. The file lists ``<obj>: <src> <header> ...``, continuing lines with ``\\``
    and escaping spaces in paths as ``\\ ``; ``-MMD`` omits system headers, which never change here.
    """
    try:
        with open(depfile) as f:
            text = f.read()
    except OSError:
        return []
    # Split off the ``<obj>:`` target on the first colon that ends the target -- i.e. one followed by
    # whitespace. A bare ``split(':')`` would break on a colon inside the object path (the compiler
    # escapes spaces but not colons), so match the separator, not any colon.
    parts = re.split(r':\s', text, maxsplit=1)
    if len(parts) < 2:
        return []
    # Protect escaped spaces before splitting, or a path containing one would be torn into pieces
    # that never exist on disk -- which would silently rebuild everything on every build.
    text = parts[1].replace('\\\n', ' ').replace('\\ ', '\0')
    return [entry.replace('\0', ' ') for entry in text.split()]


def _ensure_dace_pch(cxx: str, pch_flags: List[str], runtime_inc: str, runtime_mtime: float,
                     run) -> Optional[List[str]]:
    """Precompile ``<dace/dace.h>`` once per (compiler, flags) and cache it in the user cache dir.

    The DaCe runtime umbrella header dominates the compile time of a small kernel (~1s of parsing +
    template instantiation); precompiling it cuts a host translation unit by ~3x. Returns the extra
    ``-I``/``-include`` flags that make g++/clang++ use the cached PCH, or ``None`` when a PCH could
    not be produced (the caller then compiles normally -- correctness is unaffected, only speed).

    An invalid or flag-mismatched PCH is silently ignored by the compiler, so this can never change
    the produced object; the only failure mode is the one-off PCH build itself, which is swallowed.
    """
    try:
        import getpass
        import hashlib
        key = hashlib.md5(('\0'.join([cxx, runtime_inc] + pch_flags)).encode()).hexdigest()[:16]
        # PCH root: keep the ~120 MB .gch in RAM (/dev/shm) when available -- on HPC login/compute
        # nodes the default user cache dir is NFS-backed $HOME, and re-reading the PCH over NFS on
        # every translation unit costs more than the PCH saves, silently eroding the native
        # backend's compile-time advantage. Override with DACE_NATIVE_PCH_DIR; fall back to the
        # user cache dir where /dev/shm is absent (macOS, containers without shm).
        pch_root = os.environ.get('DACE_NATIVE_PCH_DIR')
        if not pch_root:
            if os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK):
                pch_root = os.path.join('/dev/shm', f'dace_native_pch_{getpass.getuser()}')
            else:
                pch_root = os.path.expanduser('~/.cache/dace/native_pch')
        pch_dir = os.path.join(pch_root, key)
        header = os.path.join(pch_dir, 'dace_prewarm.h')
        gch = header + '.gch'
        # Strictly newer, matching the coarse-mtime convention used for objects/libraries below: a
        # .gch sharing the newest runtime header's mtime counts as stale, since g++ would otherwise
        # keep silently using a PCH built from the pre-edit headers.
        if not (os.path.isfile(gch) and os.path.getmtime(gch) > runtime_mtime):
            os.makedirs(pch_dir, exist_ok=True)
            if not os.path.isfile(header):
                with open(header, 'w') as f:
                    f.write('#include <dace/dace.h>\n')
            # Compile to a per-process temp then atomically rename into place, so a concurrent build
            # (pytest -n4 shares this global cache) can never observe a half-written .gch.
            tmp_gch = f'{gch}.tmp.{os.getpid()}'
            run([cxx] + pch_flags + ['-I', runtime_inc, '-x', 'c++-header', header, '-o', tmp_gch])
            os.replace(tmp_gch, gch)
        return ['-I', pch_dir, '-include', 'dace_prewarm.h']
    except Exception:
        return None  # any trouble -> compile without the PCH


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_native(program_folder: str,
                 program_name: str,
                 files: List[str],
                 targets: Dict,
                 environments,
                 build_folder: str,
                 build_env: dict,
                 output_stream=None) -> None:
    """Compile and link a prepared program folder into ``build/lib<name>.<ext>`` (+ loader stub).

    Writes the same artifacts, in the same place, as the CMake path -- so the shared tail of
    :func:`dace.codegen.compiler.configure_and_compile` (``get_binary_name`` lookup, production
    relocation) works unchanged.

    :param files: source paths relative to ``<program_folder>/src`` (from ``dace_files.csv``).
    :param targets: ``{target_name: TargetCodeGenerator}`` for the linkable sources.
    :param environments: resolved environment classes the SDFG uses.
    :param build_env: subprocess environment (MPI-rank identity stripped by the caller).
    """
    # Reuse the caller's subprocess runner (keeps the SIGCHLD/MPI-env safeguards). Lazy import keeps
    # this module free of an import cycle with compiler.py.
    from dace.codegen.compiler import _run_liveoutput, identical_file_exists

    if os.name != 'posix':
        raise cgx.CompilerConfigurationError('Native build mode is Linux-only; use compiler.build_mode=cmake.')

    dace_root = _dace_root()
    runtime_inc = os.path.join(dace_root, 'runtime', 'include')
    generated_inc = os.path.join(program_folder, 'include')
    src_folder = os.path.join(program_folder, 'src')
    lib_ext = Config.get('compiler', 'library_extension')
    # Walked once per build (not cached across builds): the runtime headers are stable within a build,
    # but a developer editing them between builds in one long-lived process must still invalidate
    # objects/PCH -- a process-lifetime cache here would wrongly reuse stale objects.
    runtime_mtime = _newest_mtime(runtime_inc)

    def run(cmd: List[str], stream=output_stream) -> None:
        line = ' '.join(shlex.quote(c) for c in cmd)
        if Config.get_bool('debugprint'):
            print(f'Native build: {line}')
        try:
            _run_liveoutput(line, shell=True, cwd=build_folder, output_stream=stream, env=build_env)
        except subprocess.CalledProcessError as ex:
            raise cgx.CompilationError('Compiler failure:\n' + ex.output)

    def up_to_date(product: str, *sources: str) -> bool:
        if not os.path.isfile(product):
            return False
        ptime = os.path.getmtime(product)
        # Strict ``<``: an input sharing the product's mtime (possible on coarse-granularity
        # filesystems where a same-second edit lands on the same tick) counts as stale, so the
        # product is rebuilt rather than silently reused.
        return all(os.path.isfile(s) and os.path.getmtime(s) < ptime for s in sources)

    # --- classify sources ---------------------------------------------------
    host_objs: List[str] = []  # (.cpp -> .o)
    cuda_objs: List[str] = []  # (.cu  -> .o via nvcc)
    compile_jobs: List[tuple] = []  # (kind, src, obj)

    def add_source(abspath: str, tag: str) -> None:
        obj = os.path.join(build_folder, tag + '.o')
        if abspath.endswith('.cu'):
            cuda_objs.append(obj)
            compile_jobs.append(('cuda', abspath, obj))
        else:
            host_objs.append(obj)
            compile_jobs.append(('host', abspath, obj))

    for rel in files:
        add_source(os.path.join(src_folder, rel), rel.replace(os.sep, '__'))

    has_gpu = bool(cuda_objs) or any(t in _GPU_TARGETS for t in targets)
    # Native mode is CUDA-only. A HIP/ROCm backend is rejected with a clear, actionable error
    # instead of attempting an unvalidated build that would fail confusingly downstream.
    if has_gpu and common.get_gpu_backend() == 'hip':
        raise cgx.CompilerConfigurationError(
            'Native build mode supports CUDA GPUs only, not HIP/ROCm. Use compiler.build_mode=cmake '
            'for this program.')

    # --- resolve toolkits + environment libraries ---------------------------
    spec = _LinkSpec()
    nvcc = None
    if has_gpu:
        nvcc, cuda_incdirs, cuda_libdirs = _cuda_paths()
        spec.includes += cuda_incdirs
        spec.libdirs += cuda_libdirs

    # Sorted by name: ``environments`` is a topological sort of a SET of class objects, whose
    # iteration order is id-based and therefore differs between processes. That order reaches the
    # flag lists and thus the recorded command strings, so leaving it unsorted would make the .cmd
    # sidecars mismatch on every re-run and defeat the incremental fast path. CMake sorts its
    # environment flags for the same reason; only independent shared libraries are involved here, so
    # discarding the topological order is safe.
    for env in sorted(environments, key=lambda e: e.__name__):
        _resolve_environment(env, spec)

    # Per-target extra libraries from config (compiler.<target>.libs).
    target_libs: List[str] = []
    for tname in targets:
        try:
            raw = Config.get('compiler', tname, 'libs')
        except (KeyError, TypeError):
            continue
        if raw:
            target_libs += shlex.split(raw)

    # --- shared flag context ------------------------------------------------
    std = Config.get('compiler', 'cpp_standard')
    cpu_args = shlex.split(Config.get('compiler', 'cpu', 'args'))
    build_type_flags = _BUILD_TYPE_FLAGS.get(Config.get('compiler', 'build_type'), [])
    cuda_build_type_flags = _CUDA_BUILD_TYPE_FLAGS.get(Config.get('compiler', 'build_type'), [])
    # Base flags shared by the host compile and the PCH. ``-fPIC`` is appended LAST so it always wins
    # over any -fno-pic/-fPIE a user put in cpu.args or a build_type flag (the last -f code-model
    # option wins) -- the artifact is always a shared library, and the distro toolchain default may be
    # -fPIE. ``-pthread`` matches CMake's Threads::Threads on the compile line, not just the link line.
    host_base_flags = [f'-std=c++{std}', '-fopenmp', '-pthread'] + cpu_args + build_type_flags + ['-fPIC']
    defines = [f'-DDACE_BINARY_DIR="{build_folder}"']
    if has_gpu:
        defines.append('-DWITH_CUDA')
    includes = ['-I' + runtime_inc, '-I' + generated_inc] + ['-I' + d for d in deduplicate(spec.includes)]

    # --- compile ------------------------------------------------------------
    cuda_arch_flags: List[str] = []
    if has_gpu:
        # The supported-arch list is only consulted to filter the fallback entries, so probing for it
        # when the local GPU is detected would spawn an nvcc it never reads.
        allow_native = _can_use_arch_native(nvcc)
        supported = None if allow_native else _nvcc_supported_arches(nvcc)
        cuda_arch_flags = _cuda_arch_flags(supported, allow_native=allow_native)
    ccbin = (['-ccbin', _cxx()] if Config.get('compiler', 'cpu', 'executable') else [])

    # Precompile the DaCe runtime header once (per compiler+flags) to speed up host translation
    # units. WITH_CUDA changes what dace.h pulls in, so it is part of the PCH's flags; the per-program
    # -DDACE_BINARY_DIR and -I dirs are tolerated as extras on the compile line. Only the generated
    # framecode gets the forced ``-include``; an environment's auxiliary .cpp is left alone, since
    # force-including <dace/dace.h> into a TU that does not expect it can break it.
    generated_prefix = src_folder + os.sep
    host_pch: List[str] = []
    if any(kind == 'host' and src.startswith(generated_prefix) for kind, src, _ in compile_jobs):
        pch_flags = list(host_base_flags)
        if has_gpu:
            pch_flags += ['-DWITH_CUDA']
        host_pch = _ensure_dace_pch(_cxx(), pch_flags, runtime_inc, runtime_mtime, run) or []

    def obj_current(obj: str, cmd: List[str]) -> bool:
        """An object is current if every file it was built from is older than it and the command that
        built it is unchanged (a changed flag/define/build_type must rebuild, as CMake reconfigures).

        The depfile the previous compile emitted names the source and every header the compiler
        actually opened, which a fixed directory walk cannot do -- an environment's own bundled
        header only shows up here. No depfile means no proof of currency, so rebuild.
        """
        if not os.path.isfile(obj):
            return False
        otime = os.path.getmtime(obj)
        # The PCH already satisfies the generated framecode's <dace/dace.h>, so the compiler never
        # parses the runtime headers behind it and -MMD cannot list them -- the depfile records
        # dace.h alone. The runtime tree therefore needs its own comparison, or editing a header it
        # includes would leave this object stale. ``runtime_mtime`` is already computed for the PCH.
        if runtime_mtime >= otime:
            return False
        dependencies = _depfile_headers(obj + '.d')
        if not dependencies:
            return False
        # ``>=``: an input sharing the object's mtime counts as newer (coarse-mtime safety). A
        # dependency that vanished also forces a rebuild.
        for dependency in dependencies:
            if not os.path.isfile(dependency) or os.path.getmtime(dependency) >= otime:
                return False
        return identical_file_exists(obj + '.cmd', ' '.join(cmd))

    def compile_one(obj: str, cmd: List[str], stream=output_stream) -> None:
        run(cmd, stream)
        with open(obj + '.cmd', 'w') as f:  # record the exact command for the staleness check above
            f.write(' '.join(cmd))

    # Assemble one command per translation unit; they are independent (each writes its own object),
    # so the host .cpp and device .cu compiles run concurrently.
    compile_units: List[tuple] = []  # (obj, cmd) for the out-of-date units only
    for kind, src, obj in compile_jobs:
        # ``-MMD -MF`` makes the compiler record the headers it opened, so obj_current can track
        # every dependency instead of only the directories native happens to know about.
        depfile = ['-MMD', '-MF', obj + '.d']
        if kind == 'host':
            pch = host_pch if src.startswith(generated_prefix) else []
            cmd = ([_cxx()] + host_base_flags + defines + pch + includes + spec.compile_flags + depfile +
                   ['-c', src, '-o', obj])
        else:  # cuda
            # The CUDA build-type flags go straight on the nvcc line, as CMake applies
            # CMAKE_CUDA_FLAGS_<CONFIG>: without them the device TU builds unoptimized and, worse,
            # without -DNDEBUG, leaving asserts live that the cmake build compiles out.
            # Host-side flags are forwarded via ``-Xcompiler``: ``-fPIC`` (after cuda.args so it wins
            # over any conflicting host code-model flag) and ``-fopenmp`` so host-side OpenMP in the
            # generated .cu links against libgomp like the .cpp TUs do.
            cmd = ([nvcc, '-std=c++17'] + ccbin + cuda_arch_flags +
                   shlex.split(Config.get('compiler', 'cuda', 'args')) + cuda_build_type_flags +
                   ['-Xcompiler', '-fPIC,-fopenmp'] + defines + includes + depfile + ['-dc', src, '-o', obj])
        if not obj_current(obj, cmd):
            compile_units.append((obj, cmd))

    if len(compile_units) <= 1:
        for obj, cmd in compile_units:
            compile_one(obj, cmd)
    else:
        import concurrent.futures
        # Bounded so a program with many translation units cannot fan out into an OOM of heavy -O3
        # compiles on a shared machine; the common case (one host .cpp + one device .cu) is 2. Each
        # concurrent compile suppresses live streaming (stream=None) so the worker threads never write
        # the shared output_stream at once; a failure still carries its output via the raised exception.
        workers = min(len(compile_units), 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(compile_one, obj, cmd, None) for obj, cmd in compile_units]
            errors = [f.exception() for f in concurrent.futures.as_completed(futures)]
        errors = [e for e in errors if e is not None]
        if errors:
            raise errors[0]

    # --- CUDA device link + archive ----------------------------------------
    cuda_archive = None
    if cuda_objs:
        cuda_archive = os.path.join(build_folder, f'lib{program_name}_cuda.a')
        if not up_to_date(cuda_archive, *cuda_objs):
            dlink_obj = os.path.join(build_folder, 'cuda_dlink.o')
            run([nvcc] + ccbin + cuda_arch_flags + ['--compiler-options', '-fPIC', '-dlink'] + cuda_objs +
                ['-o', dlink_obj])
            if os.path.exists(cuda_archive):
                os.remove(cuda_archive)
            run(['ar', 'rcs', cuda_archive] + cuda_objs + [dlink_obj])

    # --- final shared library ----------------------------------------------
    lib_path = os.path.join(build_folder, f'lib{program_name}.{lib_ext}')
    link_cmd = [_cxx(), '-shared', '-fopenmp', '-o', lib_path]
    link_cmd += host_objs
    if cuda_archive:
        link_cmd += [cuda_archive]
    link_cmd += ['-pthread']
    libdirs = deduplicate(spec.libdirs)
    link_cmd += ['-L' + d for d in libdirs]
    # NOT deduplicated: link flags are positional. An environment may legitimately repeat a token --
    # MKL's ScaLAPACK line wraps each archive in its own -Wl,--whole-archive/-Wl,--no-whole-archive
    # pair -- and dropping the second pair would silently stop whole-archiving that library.
    link_cmd += spec.link_flags
    link_cmd += ['-l' + lib for lib in deduplicate(spec.libs)]
    # RPATH every directory we link out of, so the loader can find those libraries without
    # LD_LIBRARY_PATH: -L alone only satisfies the linker, and the stub's dlopen would fail at run
    # time for anything outside ldconfig (MKL via MKLROOT, a module-installed MPI). A link flag may
    # be an absolute library file (rpath its directory) or an absolute directory carried as the
    # value of a separate '-L' token (rpath it as-is).
    rpath_dirs = list(libdirs)
    for flag in spec.link_flags:
        if not os.path.isabs(flag):
            continue
        if os.path.isfile(flag):
            rpath_dirs.append(os.path.dirname(flag))
        elif os.path.isdir(flag):
            rpath_dirs.append(flag)
    link_cmd += [f'-Wl,-rpath,{d}' for d in deduplicate(rpath_dirs) if d]
    # The CUDA runtime is placed last so libraries that depend on it (e.g. cublas) precede it.
    # ``cudadevrt`` resolves the device-link registration symbols from separable compilation.
    if has_gpu:
        link_cmd += ['-lcudadevrt', '-lcudart']
    link_cmd += target_libs
    link_cmd += shlex.split(Config.get('compiler', 'linker', 'args') or '')
    # Relink only when an input object/archive is newer than the library or the link command itself
    # changed (a lib/flag edit that leaves the objects untouched). The ``.cmd`` sidecar records the
    # exact command, mirroring the per-object staleness check.
    link_inputs = host_objs + ([cuda_archive] if cuda_archive else [])
    if not (up_to_date(lib_path, *link_inputs) and identical_file_exists(lib_path + '.cmd', ' '.join(link_cmd))):
        run(link_cmd)
        with open(lib_path + '.cmd', 'w') as f:
            f.write(' '.join(link_cmd))

    # --- loader stub (rebuilt only when missing; dacestub.cpp never changes) -
    stub_path = os.path.join(build_folder, f'libdacestub_{program_name}.{lib_ext}')
    stub_src = os.path.join(dace_root, 'codegen', 'tools', 'dacestub.cpp')
    if not up_to_date(stub_path, stub_src):
        run([_cxx(), '-shared', '-fopenmp', '-fPIC', '-o', stub_path, stub_src, '-pthread', '-ldl'])
