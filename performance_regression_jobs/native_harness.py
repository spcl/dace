"""Native C++ reference harness: compile the per-kernel .cpp microkernels from the
sibling VectraArtifacts repo (see corpus_sources) into one .so per lane, and call
each kernel's function (named after its file stem) via ctypes -- it times itself
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
#:   compiler-seq      single-core -O3 -march=native + the fp guarantee flags (autovectorized,
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
OPT_FLAGS = ('-O3', '-march=native', '-fno-math-errno', '-fno-trapping-math', '-fno-signed-zeros', '-freciprocal-math',
             '-fopenmp')


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


# --------------------------------------------------------------------------
# Source discovery: the TSVC native baselines compile one .cpp per kernel
# subfolder, out of the sibling VectraArtifacts repo (its microkernels
# supersede the old tsvc2_core.cpp / tsvc_2_5_core.cpp monoliths).
# --------------------------------------------------------------------------
def _dace_repo_root() -> str:
    """dace repo root -- two levels up from this file (performance_regression_jobs/native_harness.py)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vectra_artifacts_root() -> str:
    """VectraArtifacts checkout holding the split TSVC microkernel sources: $VECTRA_ARTIFACTS_ROOT
    if set, else a sibling of the dace repo root (both repos checked out side by side)."""
    env = os.environ.get('VECTRA_ARTIFACTS_ROOT')
    if env:
        return env
    sibling = os.path.join(os.path.dirname(_dace_repo_root()), 'VectraArtifacts')
    if os.path.isdir(sibling):
        return sibling
    raise RuntimeError(f'VectraArtifacts checkout not found: set VECTRA_ARTIFACTS_ROOT or '
                        f'place it at {sibling!r} (sibling of the dace repo)')


#: corpus (matching CORPUS in tsvc2_perf.py / tsvc2_5_perf.py) -> (microkernel root relative
#: to VectraArtifacts, the file-stem suffix picked per kernel subfolder -- the ONE variant
#: matching this corpus's existing DaCe kernel-name convention: tsvc2 kernel names already
#: end '_d_single'; tsvc2_5 kernel names are bare, matching the plain double '_d' variant).
_CORPUS_MICROKERNELS = {
    'tsvc2': (os.path.join('tsvc_2', 'tsvc_cpp_microkernels'), '_d_single'),
    'tsvc2_5': (os.path.join('tsvc_2_5', 'tsvc_2_5_cpp_microkernels'), '_d'),
}


def corpus_sources(corpus: str) -> list[str]:
    """One .cpp per kernel subfolder for `corpus` ('tsvc2' | 'tsvc2_5'):
    root/<kernel>/<kernel><suffix>.cpp. Sorted so link order (and therefore the
    built .so) is deterministic."""
    subdir, suffix = _CORPUS_MICROKERNELS[corpus]
    root = os.path.join(vectra_artifacts_root(), subdir)
    sources = []
    for kernel_dir in sorted(os.listdir(root)):
        kdir = os.path.join(root, kernel_dir)
        if not os.path.isdir(kdir):
            continue
        src = os.path.join(kdir, kernel_dir + suffix + '.cpp')
        if not os.path.isfile(src):
            raise FileNotFoundError(f'{corpus}: expected {src!r}')
        sources.append(src)
    return sorted(sources)


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


def _gxx_version(gxx):
    """(major, minor) of a g++ binary, or (0, 0) if it can't be queried."""
    try:
        dump = subprocess.run([gxx, '-dumpversion'], capture_output=True, text=True, timeout=10).stdout.strip()
        parts = dump.split('.')
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:
        return (0, 0)


def newest_gxx():
    """Path to the HIGHEST-version g++ on PATH, across the bare ``g++`` (a spack-loaded gcc
    puts its own here) and versioned ``g++-NN`` names. Distros ship a stale ``/usr/bin/g++``
    (e.g. gcc 7) alongside newer ``g++-14``/``g++-13``; the newest is the one that both
    compiles DaCe's C++23 (clang's libstdc++ source, see find_gcc_install_dir) AND supports
    the Graphite loop optimizer (``-floop-parallelize-all``, needs a gcc built with isl --
    the ancient default g++ predates it). Returns None if no g++ is found at all."""
    best, best_ver, seen = None, (-1, -1), set()
    for name in ['g++'] + [f'g++-{m}' for m in range(30, 6, -1)]:
        gxx = find_compiler(name)
        if not gxx:
            continue
        real = os.path.realpath(gxx)
        if real in seen:
            continue
        seen.add(real)
        ver = _gxx_version(gxx)
        if ver > best_ver:
            best, best_ver = gxx, ver
    return best


def _gcc_install_dir_of(gxx):
    """The libstdc++ 'install:' dir clang's --gcc-install-dir wants, for a given g++."""
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


def find_gcc_install_dir():
    """Clang needs an explicit --gcc-install-dir to find libstdc++ headers.

    Uses g++ -- the C++ compiler, not 'gcc' the C compiler -- since the two can be
    different versions with only one having matching libstdc++-dev headers (observed:
    a C-only gcc with no libstdc++-dev, while a different g++ on PATH was the complete
    toolchain).

    Resolved from :func:`newest_gxx` (the highest-version g++ on PATH): DaCe codegen emits
    C++23, and clang resolves its libstdc++ from THIS directory -- a stale one (a distro's
    ``/usr/bin/g++`` fixed at gcc 7, whose libstdc++ lacks ``std::ranges::fold_left`` and
    other C++23 features) makes clang fail to compile the generated code even though clang
    itself is modern. A spack-loaded modern gcc wins; failing that a system ``g++-14`` beats
    the ancient default ``g++``."""
    gxx = newest_gxx()
    return _gcc_install_dir_of(gxx) if gxx else None


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
    'native-clang-polly-autopar': (lambda: find_compiler('clang++'), lambda cc: _gcc_install_dir_flag(cc) + [
        '-mllvm', '-polly', '-mllvm', '-polly-parallel', '-mllvm', '-polly-parallel-force', '-mllvm',
        '-polly-process-unprofitable', '-lgomp'
    ]),
    # newest_gxx (not bare 'g++'): the Graphite loop optimizer '-floop-parallelize-all' needs a
    # gcc built with isl, which the ancient distro-default '/usr/bin/g++' (gcc 7) lacks -- a
    # newer system g++-14/13/12 has it. Picking the newest g++ makes this lane a real Graphite
    # auto-parallelizer instead of silently failing to compile.
    'native-gcc-autopar':
    (newest_gxx, lambda cc: [f'-ftree-parallelize-loops={_autopar_threads()}', '-floop-parallelize-all', '-fopenmp']),
    # -- experiment-facing lanes (run_perf.py). Both follow the PHASE compiler
    #    (_perf_phase_cxx == DACE_PERF_CXX == run_perf --cxx) so a phase is fully
    #    LLVM or fully GCC, never mixed: 'seq' is a single-core autovectorized build
    #    (OPT_FLAGS: -O3 -march=native + the fp guarantee flags) and 'autopar' adds the matching
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


def compile_lane(cpp_paths, so_path, lane, timeout=1200):
    """Compile one lane's shared library from one source or many (e.g. every kernel's
    per-kernel .cpp, see corpus_sources) -- always ONE .so per lane, exactly as the old
    single-TU monolith produced. Returns (ok, error_message). Every lane finds its own
    vendor's compiler (see _LANE_SPEC) -- no cross-lane override, so a lane always
    measures its named vendor."""
    if lane not in _LANE_SPEC:
        raise ValueError(lane)
    sources = [cpp_paths] if isinstance(cpp_paths, str) else list(cpp_paths)
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    find_cc, extra_flags = _LANE_SPEC[lane]
    cc = find_cc()
    if not cc:
        return False, f'{lane}: compiler not found'

    cmd = [cc, *OPT_FLAGS] + extra_flags(cc) + openmp_rpath_flags(cc) + library_discovery_flags() + [
        '-shared', '-fPIC', *sources, '-o', so_path
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
# Signature parsing: one .cpp per kernel (see corpus_sources), each defining
# exactly one function whose name is the file stem -- so each file contributes
# exactly one entry, keyed by that stem.
# --------------------------------------------------------------------------
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


def parse_signatures(cpp_paths):
    """kernel_name -> [{'name', 'ctype', 'is_pointer'}, ...] in declaration order.

    Accepts one file or many (see corpus_sources); each file's single function is
    keyed by its own file stem -- the C symbol name -- and only ITS parameter
    list is parsed (unlike the old monolith, no other kernel's signature is in
    the same file to worry about)."""
    paths = [cpp_paths] if isinstance(cpp_paths, str) else list(cpp_paths)
    out = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        text = open(path).read()
        m = re.search(re.escape(name) + r'\s*\((.*?)\)\s*\{', text, re.DOTALL)
        if not m:
            raise ValueError(f'{name!r} signature not found in {path!r}')
        params_str = m.group(1).strip()
        out[name] = [_parse_param(p) for p in params_str.split(',')] if params_str else []
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
