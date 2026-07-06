"""Native C++ reference harness: compile tsvc{2,2_5}_core.cpp with each of
4 vendor toolchains (GCC, Clang/LLVM, Intel oneAPI, NVIDIA HPC SDK), each in
both a plain-serial and that vendor's own auto-parallelized form, and call
each kernel's ``<name>_run_timed`` function via ctypes. Each such function
times itself (std::chrono) and writes the elapsed nanoseconds to its trailing
``time_ns`` output pointer -- built for exactly this kind of external driving.

Every lane finds its own compiler independently (no cross-lane override): a
lane like 'native-gcc-autopar' exists specifically to measure THAT vendor's
auto-parallelizer, so a lane silently building with a different vendor's
compiler would invalidate what it claims to measure. A vendor whose compiler
isn't installed is simply skipped (compile_lane returns an error for that
one lane; see callers' "compiler not found" handling) -- not every machine
has all 4 toolchains.
"""
import ctypes
import os
import re
import shutil
import subprocess

#: One (serial, autopar) pair per vendor. LLVM/clang first -- same
#: LLVM-first preference as find_best_cpp_compiler() below, which resolves
#: DaCe's own codegen compiler -- so a time-limited sweep collects the LLVM
#: seq + autopar lanes (and the LLVM-compiled DaCe baseline/canon/fast-canon
#: lanes, all under one shared --cxx choice) before the other vendors. A
#: vendor with no compiler installed is skipped lane-by-lane (see
#: compile_lane), not corpus-wide.
LANES = ('native-clang', 'native-clang-polly-autopar', 'native-gcc', 'native-gcc-autopar', 'native-icpx',
         'native-icpx-autopar', 'native-nvc', 'native-nvc-autopar')

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
    """clang++ and Intel's LLVM-based icpx both need an explicit
    --gcc-install-dir on a machine with several GCC versions installed side
    by side, to find a *matching* libstdc++ headers package (see
    find_gcc_install_dir) -- plain g++/nvc++ never need this. Public: also
    used by engine.configure_dace_process() for DaCe's own codegen compiler,
    not just the native lanes below."""
    base = os.path.basename(cc)
    return 'clang' in base or 'icpx' in base or 'icpc' in base


def _gcc_install_dir_flag(cc):
    gcc_dir = find_gcc_install_dir() if needs_gcc_install_dir(cc) else None
    return [f'--gcc-install-dir={gcc_dir}'] if gcc_dir else []


#: lane -> (finder() -> compiler path or None, (cc, nthreads) -> extra flags
#: beyond '-O3 ... -shared -fPIC <src> -o <so>'). Each finder is independent
#: -- no lane ever takes an override -- so a lane always measures its own
#: named vendor's compiler, never a substitute.
_LANE_SPEC = {
    'native-gcc': (lambda: find_compiler('g++'), lambda cc, nthreads: []),
    'native-gcc-autopar':
    (lambda: find_compiler('g++'), lambda cc, nthreads: [f'-ftree-parallelize-loops={nthreads}', '-lgomp']),
    'native-clang': (lambda: find_compiler('clang++'), lambda cc, nthreads: _gcc_install_dir_flag(cc)),
    'native-clang-polly-autopar':
    (lambda: find_compiler('clang++'), lambda cc, nthreads: _gcc_install_dir_flag(cc) + [
        '-mllvm', '-polly', '-mllvm', '-polly-parallel', '-mllvm', '-polly-parallel-force', '-mllvm',
        '-polly-process-unprofitable', '-lgomp'
    ]),
    # oneAPI icpx only -- no fallback to the older classic icpc: silently
    # substituting a different Intel compiler product under the same
    # "native-icpx" label would mislabel whichever one actually built it.
    'native-icpx': (lambda: find_compiler('icpx'), lambda cc, nthreads: _gcc_install_dir_flag(cc)),
    'native-icpx-autopar': (lambda: find_compiler('icpx'),
                            lambda cc, nthreads: _gcc_install_dir_flag(cc) + ['-parallel', '-qopenmp']),
    'native-nvc': (lambda: find_compiler('nvc++'), lambda cc, nthreads: []),
    'native-nvc-autopar': (lambda: find_compiler('nvc++'), lambda cc, nthreads: ['-Mconcur']),
}


#: A compiler that doesn't recognize a flag we asked for often warns and
#: keeps going (exit 0) rather than erroring -- e.g. a newer icpx silently
#: dropping the classic-icc '-parallel' autopar flag would otherwise produce
#: a "successfully compiled" serial binary that native-icpx-autopar reports
#: as an auto-parallelizer measurement. Catching this by scanning stderr
#: applies uniformly to every lane/vendor, not just the one flag combination
#: we've been able to test locally.
_IGNORED_FLAG_RE = re.compile(
    r'unknown argument|argument unused during compilation|unrecognized command[- ]line option|'
    r'unrecognized option|ignoring unknown option|unsupported option', re.IGNORECASE)


def compile_lane(cpp_path, so_path, lane, nthreads=4, timeout=180):
    """Compile one lane's shared library. Returns (ok, error_message).

    Every lane finds its own vendor's compiler (see _LANE_SPEC) -- there is
    no cross-lane override, since e.g. 'native-clang-polly-autopar' exists
    specifically to measure clang+Polly's auto-parallelizer and silently
    building it with a different compiler would invalidate the measurement."""
    if lane not in _LANE_SPEC:
        raise ValueError(lane)
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    find_cc, extra_flags = _LANE_SPEC[lane]
    cc = find_cc()
    if not cc:
        return False, f'{lane}: compiler not found'

    cmd = [cc, '-O3'] + extra_flags(cc, nthreads) + ['-shared', '-fPIC', cpp_path, '-o', so_path]
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
