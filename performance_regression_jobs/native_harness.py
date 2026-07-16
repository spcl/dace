"""Native C++ reference harness: compile tsvc{2,2_5}_core.cpp and call each
kernel's ``<name>_run_timed`` function via ctypes -- it times itself
(std::chrono) and writes the elapsed nanoseconds to its trailing ``time_ns``
output pointer.

Two native baselines the DaCe pipelines are compared against:
  * SINGLE-CORE: ``native-clang`` -- plain serial clang.
  * MULTI-CORE AUTO-PAR (compiler auto-parallelization): ``native-clang-polly-
    autopar`` (clang + Polly ``-polly-parallel``) and ``native-gcc-autopar``
    (gcc ``-ftree-parallelize-loops=<n> -floop-parallelize-all -fopenmp``).
    Both are OpenMP/GOMP-threaded; either one supplies the multi-core baseline
    (whichever this machine's toolchain can actually build).

A lane whose compiler isn't on PATH (or whose auto-par flag isn't supported) is
skipped for that lane -- compile_lane returns an error and the sweep moves on --
never falling back to a different vendor.
"""
import ctypes
import os
import re
import shutil
import subprocess

#: Serial (single-core) and two multi-core auto-parallelizing forms, plus the
#: two experiment-facing lanes the unified run_perf.py sweeps:
#:   compiler-seq      single-core -O3 -march=native -ffast-math (autovectorized,
#:                     single thread) -- the sequential C++ baseline
#:   compiler-autopar  multi-core auto-parallel (gcc -ftree-parallelize-loops=N
#:                     -floop-parallelize-all -fopenmp); at -O3 this also autovecs
#: A lane is skipped entirely if its compiler isn't on PATH.
LANES = ('native-clang', 'native-clang-polly-autopar', 'native-gcc-autopar', 'compiler-seq', 'compiler-autopar')

#: Roles used by the perf scripts / boxplot: the single-core native baseline and
#: the multi-core auto-par native baselines (first one with data is preferred).
SINGLE_CORE_LANE = 'native-clang'
MULTICORE_LANES = ('native-clang-polly-autopar', 'native-gcc-autopar')


def _autopar_threads():
    """Thread count baked into gcc's ``-ftree-parallelize-loops=<n>`` -- take
    OMP_NUM_THREADS (the same knob the runtime honors), default 4."""
    try:
        return max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))
    except ValueError:
        return 4

#: Optimization flags shared with DaCe's own compiler.cpu.args (see
#: engine.configure_dace_process, which ensures these are present there too)
#: so a native lane and a DaCe lane are compiled at the same optimization
#: level -- otherwise a "canon is faster than native" or vice versa result
#: could just be reflecting a flags mismatch, not a real difference.
#: `-fopenmp` is always on so every lane honors OpenMP pragmas (DaCe's parallel
#: maps need it; the serial native cores have no `#pragma omp`, so it only links
#: the runtime there and stays single-threaded) and links against the same
#: OpenMP runtime across lanes -- see openmp_rpath_flags for making that runtime
#: loadable at ctypes time.
OPT_FLAGS = ('-O3', '-march=native', '-ffast-math', '-fopenmp')


def openmp_rpath_flags(cc):
    """Linker ``-rpath`` entries so a compiled ``.so`` can find the OpenMP runtime it
    links (``libomp`` for clang, ``libgomp`` for gcc) when it is loaded via ctypes.

    ``spack load`` does not put these lib dirs on ``LD_LIBRARY_PATH`` and the compilers
    bake no ``RUNPATH`` of their own, so a ``-fopenmp`` library otherwise fails to load
    with ``libomp.so: cannot open shared object file``. The dirs are asked of the compiler
    itself (``-print-file-name``), so this follows whatever toolchain is on PATH and adds
    an rpath only for a runtime that actually resolves to a real file."""
    dirs = []
    for lib in ('libomp.so', 'libgomp.so'):
        try:
            out = subprocess.run([cc, f'-print-file-name={lib}'], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
        except Exception:
            continue
        if out and out != lib and os.path.isfile(out):
            d = os.path.dirname(os.path.realpath(out))
            if d not in dirs:
                dirs.append(d)
    return [f'-Wl,-rpath,{d}' for d in dirs]


def library_discovery_flags():
    """``-isystem`` / ``-L`` / ``-rpath`` flags so a kernel that expands a DaCe library node
    (BLAS/LAPACK via ``cblas.h`` / ``lapacke.h``, MKL, ...) finds its headers and libraries in
    the common install layouts the bare compiler does not search on its own.

    The compiler already honors ``CPATH`` / ``C_INCLUDE_PATH`` / ``CPLUS_INCLUDE_PATH`` and
    ``LIBRARY_PATH`` from the inherited environment, so this ADDS only the prefix layouts those
    miss: the ``include`` / ``lib`` / ``lib64`` siblings of every ``PATH`` entry (the standard
    ``bin/ include/ lib/`` prefix used by conda / spack / venv), the common prefix variables,
    every ``CMAKE_PREFIX_PATH`` entry, and the Debian multiarch cblas/openblas header subdirs.
    Purely additive: a nonexistent dir is dropped and re-adding a default dir is a no-op, so a
    build that already resolved is unchanged."""
    inc, lib = [], []

    def add_prefix(pfx):
        inc.append(os.path.join(pfx, 'include'))
        lib.extend((os.path.join(pfx, 'lib'), os.path.join(pfx, 'lib64')))

    for entry in os.environ.get('PATH', '').split(os.pathsep):
        if entry:
            add_prefix(os.path.dirname(entry.rstrip(os.sep)))
    for var in ('CONDA_PREFIX', 'VIRTUAL_ENV', 'OPENBLAS_ROOT', 'BLAS_ROOT', 'LAPACK_ROOT', 'MKLROOT', 'CUDA_HOME',
                'CUDA_PATH'):
        if os.environ.get(var):
            add_prefix(os.environ[var])
    for pfx in os.environ.get('CMAKE_PREFIX_PATH', '').split(os.pathsep):
        if pfx:
            add_prefix(pfx)
    for base in ('/usr/include', '/usr/include/x86_64-linux-gnu'):
        for sub in ('openblas', 'openblas-pthread', 'openblas-openmp', 'openblas-serial', 'cblas', 'lapacke', 'mkl'):
            inc.append(os.path.join(base, sub))

    flags, seen = [], set()
    for d in inc:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            flags.extend(('-isystem', d))
    seen = set()
    for d in lib:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            flags.extend(('-L', d, f'-Wl,-rpath,{d}'))
    return flags


_CTYPE = {'double': ctypes.c_double, 'float': ctypes.c_float, 'int': ctypes.c_int, 'int64': ctypes.c_int64}


def find_compiler(name):
    """Plain PATH lookup for `name` (e.g. 'g++', 'clang++'). Trusts whatever
    environment/module setup (spack load, an HPC module, a venv) already put
    the intended version on PATH under its bare name -- no guessing across
    versioned suffixes or vendor install directories."""
    return shutil.which(name)


def find_best_cpp_compiler():
    """The compiler used for DaCe's own C++ codegen (--cxx / DACE_PERF_CXX):
    it needs no *specific* vendor (unlike every native lane below, which each
    test one vendor's compiler/auto-parallelizer by construction and always
    use that vendor), so this picks clang++ if it's on PATH, else g++."""
    return find_compiler('clang++') or find_compiler('g++')


def find_gcc_install_dir():
    """Clang needs an explicit --gcc-install-dir to find libstdc++ headers.

    Uses find_compiler('g++') -- the C++ compiler, not 'gcc' the C compiler --
    since the two can be different versions with only one of them having a
    matching libstdc++-dev headers package installed (observed concretely:
    a gcc present as a C-only compiler with no matching libstdc++-dev, while
    a different g++ on the same PATH was the actual complete C++ toolchain)."""
    gxx = find_compiler('g++')
    if not gxx:
        return None
    try:
        out = subprocess.run([gxx, '-print-search-dirs'], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith('install:'):
            path = line.split(':', 1)[1].strip().rstrip('/')
            if os.path.isdir(path):
                return path
    return None


def needs_gcc_install_dir(cc):
    """clang++/icpx (both LLVM-based) need an explicit --gcc-install-dir on a
    machine with several GCC versions, to find a *matching* libstdc++
    (find_gcc_install_dir) -- plain g++/nvc++ never need this. Public: also
    used by engine.configure_dace_process() for DaCe's codegen compiler."""
    base = os.path.basename(cc)
    return 'clang' in base or 'icpx' in base or 'icpc' in base


def _gcc_install_dir_flag(cc):
    gcc_dir = find_gcc_install_dir() if needs_gcc_install_dir(cc) else None
    return [f'--gcc-install-dir={gcc_dir}'] if gcc_dir else []


def _perf_phase_cxx():
    """The compiler for THIS phase -- DACE_PERF_CXX, set by run_perf from ``--cxx``. Both the
    DaCe codegen AND the native experiment lanes (compiler-seq, compiler-autopar) use it, so a
    phase is FULLY-LLVM or FULLY-GCC, never mixed. Falls back to clang++ else g++."""
    cxx = os.environ.get('DACE_PERF_CXX')
    if cxx and shutil.which(cxx):
        return shutil.which(cxx)
    return find_compiler('clang++') or find_compiler('g++')


def _is_clang(cc):
    return cc is not None and 'clang' in os.path.basename(cc).lower()


def _autopar_flags(cc):
    """Auto-parallelization flags matching the compiler family (user: cxx=clang -> clang+Polly,
    cxx=gcc -> gcc+Graphite). clang uses Polly (``-mllvm -polly -polly-parallel``); gcc uses the
    tree parallelizer + Graphite ``-floop-parallelize-all`` (needs a gcc built with isl)."""
    if _is_clang(cc):
        return _gcc_install_dir_flag(cc) + [
            '-mllvm', '-polly', '-mllvm', '-polly-parallel', '-mllvm', '-polly-parallel-force', '-mllvm',
            '-polly-process-unprofitable', '-lgomp'
        ]
    return [f'-ftree-parallelize-loops={_autopar_threads()}', '-floop-parallelize-all', '-fopenmp']


#: lane -> (finder() -> compiler path or None, cc -> extra flags beyond
#: '-O3 ... -shared -fPIC <src> -o <so>').
_LANE_SPEC = {
    'native-clang': (lambda: find_compiler('clang++'), lambda cc: _gcc_install_dir_flag(cc)),
    'native-clang-polly-autopar':
    (lambda: find_compiler('clang++'), lambda cc: _gcc_install_dir_flag(cc) + [
        '-mllvm', '-polly', '-mllvm', '-polly-parallel', '-mllvm', '-polly-parallel-force', '-mllvm',
        '-polly-process-unprofitable', '-lgomp'
    ]),
    'native-gcc-autopar':
    (lambda: find_compiler('g++'), lambda cc: [
        f'-ftree-parallelize-loops={_autopar_threads()}', '-floop-parallelize-all', '-fopenmp'
    ]),
    # -- experiment-facing lanes (run_perf.py). Both follow the PHASE compiler
    #    (_perf_phase_cxx == DACE_PERF_CXX == run_perf --cxx) so a phase is fully
    #    LLVM or fully GCC, never mixed: 'seq' is a single-core autovectorized build
    #    (OPT_FLAGS: -O3 -march=native -ffast-math) and 'autopar' adds the matching
    #    auto-parallelizer (clang -> Polly, gcc -> Graphite; see _autopar_flags).
    'compiler-seq': (_perf_phase_cxx, _gcc_install_dir_flag),
    'compiler-autopar': (_perf_phase_cxx, _autopar_flags),
}


#: A compiler that doesn't recognize a flag often warns and exits 0 rather
#: than erroring -- e.g. a newer icpx silently dropping '-parallel' would
#: otherwise report a serial binary as an auto-parallelizer measurement.
#: Scanning stderr for this applies to every lane/vendor uniformly.
_IGNORED_FLAG_RE = re.compile(
    r'unknown argument|argument unused during compilation|unrecognized command[- ]line option|'
    r'unrecognized option|ignoring unknown option|unsupported option', re.IGNORECASE)


def compile_lane(cpp_path, so_path, lane, timeout=1200):
    """Compile one lane's shared library. Returns (ok, error_message).
    Every lane finds its own vendor's compiler (see _LANE_SPEC) -- no
    cross-lane override, so a lane always measures its named vendor."""
    if lane not in _LANE_SPEC:
        raise ValueError(lane)
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    find_cc, extra_flags = _LANE_SPEC[lane]
    cc = find_cc()
    if not cc:
        return False, f'{lane}: compiler not found'

    cmd = [cc, *OPT_FLAGS] + extra_flags(cc) + openmp_rpath_flags(cc) + library_discovery_flags() + [
        '-shared', '-fPIC', cpp_path, '-o', so_path
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'compile timeout'
    if proc.returncode != 0:
        return False, proc.stderr[-400:]
    if _IGNORED_FLAG_RE.search(proc.stderr):
        return False, f'{lane}: compiler ignored a requested flag: {proc.stderr[-400:]}'
    return True, ''


# --------------------------------------------------------------------------
# Signature parsing: one .cpp compiles ALL of a corpus's kernels into a single
# translation unit, so this runs once per corpus, not per kernel.
# --------------------------------------------------------------------------
_SIG_RE = re.compile(r'(\w+)_run_timed\s*\((.*?)\)\s*\{', re.DOTALL)


def _parse_param(part):
    part = part.strip()
    is_pointer = '*' in part
    name = part.replace('__restrict__', ' ').replace('*', ' ').split()[-1]
    if 'int64' in part:
        ctype = 'int64'
    elif 'double' in part:
        ctype = 'double'
    elif 'float' in part:
        ctype = 'float'
    elif 'int' in part:
        ctype = 'int'
    else:
        raise ValueError(f'unrecognized C type in parameter: {part!r}')
    return dict(name=name, ctype=ctype, is_pointer=is_pointer)


def parse_signatures(cpp_path):
    """kernel_name -> [{'name', 'ctype', 'is_pointer'}, ...] in declaration order."""
    text = open(cpp_path).read()
    out = {}
    for m in _SIG_RE.finditer(text):
        params_str = m.group(2).strip()
        out[m.group(1)] = [_parse_param(p) for p in params_str.split(',')] if params_str else []
    return out


def call_kernel(lib, c_name, sig, *, arrays, len_1d, len_2d, scalar_params, symbols, vlen=8, iterations=1):
    """One ctypes call to <c_name>; returns the elapsed nanoseconds the C function measured.

    Every non-pointer/non-time_ns parameter is resolved BY NAME (never position)
    against a pool of known sizes + the kernel's own scalar/symbol values.
    """
    pool = {'len_1d': len_1d, 'len_2d': len_2d, 'vlen': vlen, 'iterations': iterations}
    pool.update({k.lower(): v for k, v in scalar_params.items()})
    pool.update({k.lower(): v for k, v in symbols.items()})

    time_ns = ctypes.c_int64(0)
    argtypes, call_args = [], []
    for p in sig:
        base = _CTYPE[p['ctype']]
        if p['is_pointer']:
            if p['name'] == 'time_ns':
                argtypes.append(ctypes.POINTER(ctypes.c_int64))
                call_args.append(ctypes.byref(time_ns))
            elif p['name'] in arrays:
                argtypes.append(ctypes.POINTER(base))
                call_args.append(arrays[p['name']].ctypes.data_as(ctypes.POINTER(base)))
            else:
                raise KeyError(f'unresolved pointer parameter {p["name"]!r} for {c_name}')
        else:
            if p['name'] not in pool:
                raise KeyError(f'unresolved scalar parameter {p["name"]!r} for {c_name}')
            argtypes.append(base)
            call_args.append(pool[p['name']])

    fn = getattr(lib, c_name)
    fn.argtypes = argtypes
    fn.restype = None
    fn(*call_args)
    return time_ns.value


def load_library(so_path):
    return ctypes.CDLL(so_path)
