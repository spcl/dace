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
import os
import shlex
import shutil
import subprocess
from typing import Dict, List, Optional

from dace.config import Config
from dace.codegen import common
from dace.codegen import exceptions as cgx
from dace.codegen.target import make_absolute

#: ``compiler.build_type`` -> optimization/debug flags, appended after ``compiler.cpu.args`` so the
#: last ``-O`` wins exactly as CMake's ``CMAKE_CXX_FLAGS_<CONFIG>`` does.
_BUILD_TYPE_FLAGS = {
    'Debug': ['-O0', '-g'],
    'Release': ['-O3', '-DNDEBUG'],
    'RelWithDebInfo': ['-O2', '-g', '-DNDEBUG'],
    'MinSizeRel': ['-Os', '-DNDEBUG'],
}

#: Target names whose sources are GPU device code (compiled by nvcc / amdclang++, not g++).
_GPU_TARGETS = ('cuda', 'experimental_cuda')


class _LinkSpec:
    """Include/library/flag fragments accumulated while resolving toolkits and environments."""

    def __init__(self):
        self.includes: List[str] = []  # -I values (no prefix)
        self.compile_flags: List[str] = []  # raw compile tokens
        self.libdirs: List[str] = []  # -L values (no prefix)
        self.libs: List[str] = []  # bare names for -l
        self.link_flags: List[str] = []  # raw link tokens (abs paths, -l.., -Wl,..)


def _dedup(seq: List[str]) -> List[str]:
    """Order-preserving de-duplication (linker order matters)."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_or_eval(value_or_function):
    """Environment attributes may be plain values or zero-arg callables (runtime probes)."""
    return value_or_function() if callable(value_or_function) else value_or_function


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


def _cuda_paths() -> tuple:
    """Return ``(nvcc, include_dirs, lib_dirs)`` for the CUDA toolkit, or raise a clear error.

    Locates nvcc (``compiler.cuda.path``/``$CUDA_HOME``/``$CUDA_PATH``/``which``), then finds the
    directories actually containing the CUDA runtime + math libraries and their headers by search --
    this is what ``find_package(CUDAToolkit)`` does, and unlike a fixed ``<root>/lib64`` it works for
    the HPC SDK (where the math libraries live under a separate ``math_libs`` tree). ``libcudart.so``
    / ``cuda_runtime.h`` are mandatory; the rest are added only when found.
    """
    root = (Config.get('compiler', 'cuda', 'path') or os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'))
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

    lib_dirs = _dedup([d for d in (_search_for(f, roots) for f in _CUDA_LIB_FILES) if d])
    header_roots = roots + [os.path.dirname(d) for d in lib_dirs]
    inc_dirs = _dedup([d for d in (_search_for(h, header_roots) for h in _CUDA_HEADER_FILES) if d])
    if not lib_dirs or not inc_dirs:
        raise cgx.CompilerConfigurationError(
            f'Native build found nvcc at {nvcc} but could not locate libcudart.so / cuda_runtime.h. '
            f'Set compiler.cuda.path to the toolkit root, or use compiler.build_mode=cmake.')
    return nvcc, inc_dirs, lib_dirs


def _resolve_rocm_root() -> str:
    cand = os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if not cand:
        hipcc = shutil.which('hipcc') or shutil.which('amdclang++')
        if hipcc:
            cand = os.path.dirname(os.path.dirname(hipcc))
    if not cand and os.path.isdir('/opt/rocm'):
        cand = '/opt/rocm'
    if not cand or not os.path.isdir(cand):
        raise cgx.CompilerConfigurationError(
            'Native build cannot locate ROCm. Set ROCM_PATH, add amdclang++ to PATH, or use '
            'compiler.build_mode=cmake.')
    return cand


def _resolve_mpi(spec: _LinkSpec, build_env: dict) -> None:
    """Fill ``spec`` with MPI include/lib flags by querying the MPI wrapper compiler.

    OpenMPI answers ``--showme:{incdirs,libdirs,libs}``; MPICH answers ``-compile_info`` /
    ``-link_info`` with full flag strings.
    """
    mpicxx = Config.get('compiler', 'mpi', 'executable') or 'mpicxx'

    def run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.run([mpicxx] + args, capture_output=True, text=True, env=build_env)
        except OSError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    incdirs, libdirs, libs = run(['--showme:incdirs']), run(['--showme:libdirs']), run(['--showme:libs'])
    if incdirs is not None and libdirs is not None and libs is not None:
        spec.includes += incdirs.split()
        spec.libdirs += libdirs.split()
        spec.libs += libs.split()
        return

    compile_info, link_info = run(['-compile_info']), run(['-link_info'])
    if compile_info is not None and link_info is not None:
        spec.compile_flags += [t for t in shlex.split(compile_info) if t.startswith('-I')]
        spec.link_flags += [t for t in shlex.split(link_info) if t.startswith(('-l', '-L', '-Wl', '/'))]
        return

    raise cgx.CompilerConfigurationError(
        f"Native build cannot query the MPI compiler wrapper '{mpicxx}'. Set compiler.mpi.executable "
        f"or use compiler.build_mode=cmake.")


def _classify_library(spec: _LinkSpec, lib: str) -> None:
    """Route one ``cmake_libraries`` entry onto the link line."""
    lib = lib.strip()
    if not lib:
        return
    if lib.startswith(('-l', '-L', '-Wl')):
        spec.link_flags.append(lib)
    elif os.path.isabs(lib):  # absolute path to a .so/.a (MKL, OpenBLAS, reference libs)
        spec.link_flags.append(lib)
    else:  # bare soname/name -- e.g. "cublas", "mkl_rt"
        spec.libs.append(lib)


def _resolve_environment(env, spec: _LinkSpec, build_env: dict) -> None:
    """Resolve one collected environment into ``spec``, or raise if it needs real CMake."""
    name = getattr(env, '__name__', str(env))

    if _get_or_eval(env.cmake_files):
        files = ', '.join(_get_or_eval(env.cmake_files))
        raise cgx.CompilerConfigurationError(
            f"Native build cannot resolve environment '{name}': it ships CMake files ({files}). "
            f"Use compiler.build_mode=cmake for this program.")

    # find_package packages: MPI via the wrapper; the toolkit/threading ones are handled globally;
    # BLAS/LAPACK vendor selection only picks concrete libs that arrive via cmake_libraries().
    for pkg in _get_or_eval(env.cmake_packages):
        if pkg == 'MPI':
            _resolve_mpi(spec, build_env)
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

    # Deferred ${...} fragments are dropped: a known package (MPI) already supplied the real flags,
    # and an unknown one would have raised on cmake_packages above.
    for flag in _get_or_eval(env.cmake_compile_flags):
        if not _is_deferred(flag):
            spec.compile_flags.append(flag)
    for flag in _get_or_eval(env.cmake_link_flags):
        if not _is_deferred(flag):
            spec.link_flags.append(flag)
    for lib in _get_or_eval(env.cmake_libraries):
        if not _is_deferred(lib):
            _classify_library(spec, lib)


def _collect_auxiliary_sources(environments) -> List[str]:
    """Absolute paths of ``.cu`` translation units contributed by environments (e.g. libnode wrappers)."""
    out: List[str] = []
    for env in environments:
        env_dir = os.path.dirname(env._dace_file_path)
        for src in _get_or_eval(getattr(env, 'auxiliary_sources', [])):
            out.append(src if os.path.isabs(src) else os.path.join(env_dir, src))
    return out


# ---------------------------------------------------------------------------
# GPU architecture flags
# ---------------------------------------------------------------------------


def _cuda_arch_flags() -> List[str]:
    """``-arch=native`` for auto-detection, or explicit ``-gencode`` per configured architecture.

    ``compiler.cuda.cuda_arch == 'auto'`` (or empty) uses nvcc's built-in local-GPU detection (CUDA
    11.5+), which replaces CMake's separate get_cuda_arch.cpp compile+run. An explicit
    comma-separated list emits one ``-gencode`` each.
    """
    cfg = Config.get('compiler', 'cuda', 'cuda_arch').strip()
    if cfg in ('', 'auto', 'native'):
        return ['-arch=native']
    flags: List[str] = []
    for arch in cfg.split(','):
        arch = arch.strip()
        if arch:
            flags += ['-gencode', f'arch=compute_{arch},code=sm_{arch}']
    return flags or ['-arch=native']


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
    from dace.codegen.compiler import _run_liveoutput

    if os.name != 'posix':
        raise cgx.CompilerConfigurationError('Native build mode is Linux-only; use compiler.build_mode=cmake.')

    dace_root = _dace_root()
    runtime_inc = os.path.join(dace_root, 'runtime', 'include')
    generated_inc = os.path.join(program_folder, 'include')
    src_folder = os.path.join(program_folder, 'src')
    lib_ext = Config.get('compiler', 'library_extension')

    def run(cmd: List[str]) -> None:
        line = ' '.join(shlex.quote(c) for c in cmd)
        if Config.get('debugprint') == 'verbose':
            print(f'Native build: {line}')
        try:
            _run_liveoutput(line, shell=True, cwd=build_folder, output_stream=output_stream, env=build_env)
        except subprocess.CalledProcessError as ex:
            raise cgx.CompilationError('Compiler failure:\n' + ex.output)

    def up_to_date(product: str, *sources: str) -> bool:
        if not os.path.isfile(product):
            return False
        ptime = os.path.getmtime(product)
        return all(os.path.isfile(s) and os.path.getmtime(s) <= ptime for s in sources)

    # --- classify sources ---------------------------------------------------
    host_objs: List[str] = []  # (.cpp -> .o)
    cuda_objs: List[str] = []  # (.cu  -> .o via nvcc)
    hip_objs: List[str] = []  # (.cpp -> .o via amdclang++, target_type 'hip')
    compile_jobs: List[tuple] = []  # (kind, src, obj)

    def add_source(abspath: str, tag: str) -> None:
        obj = os.path.join(build_folder, tag + '.o')
        if abspath.endswith('.cu'):
            cuda_objs.append(obj)
            compile_jobs.append(('cuda', abspath, obj))
        elif os.sep + 'hip' + os.sep in abspath:
            hip_objs.append(obj)
            compile_jobs.append(('hip', abspath, obj))
        else:
            host_objs.append(obj)
            compile_jobs.append(('host', abspath, obj))

    for rel in files:
        add_source(os.path.join(src_folder, rel), rel.replace(os.sep, '__'))
    for aux in _collect_auxiliary_sources(environments):
        add_source(aux, 'aux__' + os.path.basename(aux))

    has_gpu = bool(cuda_objs or hip_objs) or any(t in _GPU_TARGETS for t in targets)
    backend = common.get_gpu_backend() if has_gpu else None
    is_hip = backend == 'hip' or bool(hip_objs)

    # --- resolve toolkits + environment libraries ---------------------------
    spec = _LinkSpec()
    nvcc = None
    if has_gpu and not is_hip:
        nvcc, cuda_incdirs, cuda_libdirs = _cuda_paths()
        spec.includes += cuda_incdirs
        for d in cuda_libdirs:
            spec.libdirs.append(d)
            spec.link_flags.append(f'-Wl,-rpath,{d}')
    rocm_root = None
    if is_hip:
        rocm_root = _resolve_rocm_root()
        rocm_libdir = _lib_subdir(rocm_root)
        spec.includes.append(os.path.join(rocm_root, 'include'))
        spec.libdirs.append(rocm_libdir)
        spec.link_flags.append(f'-Wl,-rpath,{rocm_libdir}')
        spec.libs.append('amdhip64')

    for env in environments:
        _resolve_environment(env, spec, build_env)

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
    defines = [f'-DDACE_BINARY_DIR="{build_folder}"']
    if has_gpu:
        defines.append('-DWITH_CUDA')
        if is_hip:
            defines.append('-DWITH_HIP')
    includes = ['-I' + runtime_inc, '-I' + generated_inc] + ['-I' + d for d in _dedup(spec.includes)]

    # --- compile ------------------------------------------------------------
    cuda_arch_flags = _cuda_arch_flags() if (has_gpu and not is_hip) else []
    ccbin = (['-ccbin', _cxx()] if Config.get('compiler', 'cpu', 'executable') else [])

    for kind, src, obj in compile_jobs:
        if up_to_date(obj, src):
            continue
        if kind == 'host':
            run([_cxx(), f'-std=c++{std}', '-fopenmp'] + cpu_args + build_type_flags + defines + includes +
                spec.compile_flags + ['-c', src, '-o', obj])
        elif kind == 'cuda':
            run([nvcc, '-std=c++17'] + ccbin + ['--compiler-options', '-fPIC'] + cuda_arch_flags +
                shlex.split(Config.get('compiler', 'cuda', 'args')) + defines + includes + ['-dc', src, '-o', obj])
        else:  # hip  (AMD path -- structurally validated; unvalidated on this NVIDIA host)
            amdclang = shutil.which('amdclang++') or os.path.join(rocm_root, 'llvm', 'bin', 'amdclang++')
            run([amdclang, '-x', 'hip', f'-std=c++{std}', '--offload-arch=native', '-fPIC', '-D__HIP_PLATFORM_AMD__'] +
                shlex.split(Config.get('compiler', 'cuda', 'hip_args')) + defines + includes + ['-c', src, '-o', obj])

    # --- CUDA device link + archive ----------------------------------------
    cuda_archive = None
    if cuda_objs:
        dlink_obj = os.path.join(build_folder, 'cuda_dlink.o')
        run([nvcc] + ccbin + cuda_arch_flags + ['--compiler-options', '-fPIC', '-dlink'] + cuda_objs +
            ['-o', dlink_obj])
        cuda_archive = os.path.join(build_folder, f'lib{program_name}_cuda.a')
        if os.path.exists(cuda_archive):
            os.remove(cuda_archive)
        run(['ar', 'rcs', cuda_archive] + cuda_objs + [dlink_obj])

    # --- final shared library ----------------------------------------------
    lib_path = os.path.join(build_folder, f'lib{program_name}.{lib_ext}')
    link_cmd = [_cxx(), '-shared', '-fopenmp', '-o', lib_path]
    link_cmd += host_objs + hip_objs
    if cuda_archive:
        link_cmd += [cuda_archive]
    link_cmd += ['-pthread']
    link_cmd += ['-L' + d for d in _dedup(spec.libdirs)]
    link_cmd += _dedup(spec.link_flags)
    link_cmd += ['-l' + lib for lib in _dedup(spec.libs)]
    # The CUDA runtime is placed last so libraries that depend on it (e.g. cublas) precede it.
    # ``cudadevrt`` resolves the device-link registration symbols from separable compilation.
    if has_gpu and not is_hip:
        link_cmd += ['-lcudadevrt', '-lcudart']
    link_cmd += target_libs
    link_cmd += shlex.split(Config.get('compiler', 'linker', 'args') or '')
    run(link_cmd)

    # --- loader stub (rebuilt only when missing; dacestub.cpp never changes) -
    stub_path = os.path.join(build_folder, f'libdacestub_{program_name}.{lib_ext}')
    stub_src = os.path.join(dace_root, 'codegen', 'tools', 'dacestub.cpp')
    if not up_to_date(stub_path, stub_src):
        run([_cxx(), '-shared', '-fopenmp', '-fPIC', '-o', stub_path, stub_src, '-pthread', '-ldl'])
