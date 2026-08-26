# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Acceptance harness for MPR (maximal parallel rendering) output.

MPR's contract is one sentence: the emitted C++ builds with a bare host compiler -- no DaCe
include directory, no ``libdace``, no BLAS -- and reproduces the SDFG's numbers. Everything in
this directory is checked against that sentence, so the harness has to be able to state it
without the feature existing yet:

* :func:`assert_standalone` is a pure string check for the tokens MPR must never emit.
* :func:`build_standalone` compiles a translation unit in an EMPTY directory with NO ``-I`` flag
  at all. A leaked ``#include <dace/dace.h>`` therefore fails to compile rather than silently
  picking the header up off an inherited include path -- the failure mode a ``-I``-carrying build
  would hide.
* :func:`call_standalone` invokes the emitted entry point through ctypes, so the numeric compare
  needs no generated driver ``main`` and no file round-trip.

There is no ``mpr_available()`` skip gate: a skip that can fire on a healthy box reports green
while verifying nothing. Tests arrive as the phases land, and each one asserts.
"""
import ctypes
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

import dace
from dace import data as dt

#: C++ standard MPR output is emitted against. DaCe is >= C++20 everywhere.
CXX_STANDARD = 'c++20'
#: C standard MPR's C output is emitted against.
C_STANDARD = 'c23'

#: ``language`` -> ``(standard, source suffix, environment variable naming the compiler, fallback
#: compiler names)``. The two legs are built by DIFFERENT drivers on purpose: ``g++`` would accept
#: much of the C output as C++ and hide exactly the constructs the C dialect exists to avoid.
TOOLCHAINS = {
    'c++': (CXX_STANDARD, '.cpp', 'CXX', ('g++', 'c++')),
    'c': (C_STANDARD, '.c', 'CC', ('gcc', 'cc')),
}

#: Flags every MPR translation unit is built with, before the language standard. Deliberately NO
#: ``-I``: see the module docstring.
BASE_FLAGS = ('-O2', '-fopenmp', '-fPIC', '-shared')
#: Warning flags, kept apart from :data:`BASE_FLAGS` so the numeric gate and the zero-warning gate
#: fail independently -- a warning must not be reported as "MPR produced wrong numbers".
#: The conversion flags are in here rather than opt-in per test because an IMPLICIT conversion is
#: what a self-contained render gets wrong silently: an extent reaching a ``size_t``, an int32
#: symbol taken by a nested body, a double stored into an ``int64_t``. ``-Wall -Wextra`` diagnoses
#: none of the three.
WARNING_FLAGS = ('-Wall', '-Wextra', '-Wconversion', '-Wsign-conversion')

#: Tokens MPR output must not contain, and what each one means when it appears. Checked as plain
#: substrings/regexes on the emitted text: the compile in an empty directory catches a leaked
#: header, but a leaked ``dace::`` symbol from a header that happens to be self-contained, or a
#: state-struct dereference, would only surface as a link error much later.
#: Order matters: the first match wins, so a SPECIFIC construct is listed ahead of the general
#: pattern that would also cover it (``dace::CopyND`` before ``dace::``, ``__dace_init_cuda``
#: before ``__state``), and the failure message names the narrower cause.
BANNED_PATTERNS = (
    (re.compile(r'#\s*include\s*[<"][^>"]*dace/'), 'DaCe runtime header include'),
    (re.compile(r'#\s*include\s*"'), 'quoted (relative) include -- MPR may only use system headers'),
    (re.compile(r'CopyND'), 'dace::CopyND copy fallback'),
    (re.compile(r'__dace_(init|exit)\w*'), 'DaCe init/exit entry point'),
    (re.compile(r'\bdace\s*::'), 'DaCe runtime namespace reference'),
    (re.compile(r'\bDACE_[A-Z]'), 'DaCe preprocessor macro'),
    (re.compile(r'__state\b'), 'DaCe state-struct dereference'),
)

#: DaCe runtime functions the code generators emit UNQUALIFIED, so no ``dace::`` appears and the
#: namespace pattern above cannot see them. They are declared at global scope by ``math.h``,
#: ``pyinterop.h`` and ``ITE.h`` -- which MPR does not include, so each one is a link failure
#: waiting to happen. Only names with NO ``std`` counterpart are listed: a bare ``max``/``abs``/
#: ``round``/``conj`` is ambiguous (``std::max`` is spelled the same after a ``using``), and
#: flagging those would reject correct output. Every name here needs an MPR mapping -- a ``std::``
#: equivalent, a rewritten expression, or an emitted inline definition.
UNQUALIFIED_RUNTIME_FUNCTIONS = frozenset({
    'Abs', 'Max', 'Min', 'ITE', 'ROUND', 'iround', 'ceiling', 'int_ceil', 'int_floor', 'int_floor_ni', 'reciprocal',
    'sign', 'sgn', 'sign_numpy_2', 'heaviside', 'mod', 'Mod', 'Mod_float', 'Modulo', 'Modulo_float', 'py_mod',
    'py_floor', 'py_divmod', 'cpp_mod', 'cpp_divmod', 'floor_mod', 'deg2rad', 'rad2deg', 'np_float_pow', 'np_frexp',
    'np_modf', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_invert', 'left_shift', 'right_shift',
    'logical_left_shift', 'logical_right_shift'
})

#: A call to one of the above: the name at a word boundary, not already namespace-qualified, and
#: not a declaration of the same name (MPR is allowed to EMIT an inline ``reciprocal`` of its own,
#: which is exactly the fix -- so a preceding ``inline``/type keyword is not a violation).
_UNQUALIFIED_CALL = re.compile(r'(?<![\w:.])(' + '|'.join(sorted(UNQUALIFIED_RUNTIME_FUNCTIONS)) + r')\s*\(')

#: What counts as MPR DEFINING one of those names rather than calling it: a C++ function definition
#: (``static constexpr inline int64_t int_ceil(...)``) or a C function-like macro
#: (``#define int_ceil(a, b) _Generic(...)``), which is the C dialect's form of the same fix.
_DEFINITION_OF = re.compile(r'(?:\b(?:inline|constexpr|static)\b[^;{()\n]*?|#\s*define\s+)'
                            r'(?<![\w:.])(\w+)\s*\(')


def assert_no_unqualified_runtime_calls(code: str, label: str = 'mpr') -> None:
    """Assert ``code`` calls no unqualified DaCe runtime function.

    Split out from :func:`assert_standalone` because these names carry no ``dace::`` marker: a leak
    surfaces only as an "undeclared identifier" from the compiler, at which point nothing points at
    the printer that emitted it. Definitions MPR emits itself are excluded -- a line that declares
    the name is the fix, not the defect.
    """
    defined = {match.group(1) for match in _DEFINITION_OF.finditer(code)}
    for match in _UNQUALIFIED_CALL.finditer(code):
        name = match.group(1)
        if name in defined:
            continue
        raise AssertionError(f'{label}: MPR output calls the unqualified DaCe runtime function {name!r} at offset '
                             f'{match.start()}; it is declared by the DaCe headers MPR does not include\n'
                             f'{_context(code, match.start())}')


#: Tokens the C output must not contain, on top of :data:`BANNED_PATTERNS`. Stated here as well as
#: in ``dace.codegen.mpr.BANNED_C`` on purpose: this file is the acceptance spec, written from
#: outside, and a table that forgot an entry cannot fool both.
BANNED_PATTERNS_C = BANNED_PATTERNS + (
    (re.compile(r'\bstd\s*::'), 'C++ standard-library symbol'),
    (re.compile(r'\btemplate\s*<'), 'C++ template'),
    (re.compile(r'extern\s*"C"'), 'C++ language linkage specifier'),
    (re.compile(r'\bstatic_cast\s*<'), 'C++ static_cast'),
    (re.compile(r'\bnew\s'), 'C++ new-expression'),
    (re.compile(r'\bdelete\b'), 'C++ delete-expression'),
)


def host_compiler(language: str = 'c++') -> str:
    """The compiler MPR output for ``language`` is built with.

    Taken from ``CXX``/``CC`` when set, else the host compiler DaCe itself configures (C++ only),
    else the toolchain default. Asserts rather than skips: a supported box has both compilers, and
    a missing one is a broken environment that must fail loudly.
    """
    _, _, variable, fallbacks = TOOLCHAINS[language]
    candidate = os.environ.get(variable)
    if not candidate and language == 'c++':
        from dace.config import Config
        candidate = Config.get('compiler', 'cpu', 'executable')
    resolved = shutil.which(candidate) if candidate else None
    for fallback in fallbacks:
        if resolved is not None:
            break
        resolved = shutil.which(fallback)
    assert resolved is not None, (f'no {language} compiler found (tried {candidate!r}, {fallbacks}); MPR output is '
                                  'defined by what a bare host compiler accepts, so this box cannot test it')
    return resolved


def assert_standalone(code: str, label: str = 'mpr', language: str = 'c++') -> None:
    """Assert ``code`` carries none of the banned tokens for ``language``.

    :param code: the emitted translation unit.
    :param label: prefix for the failure message (usually the kernel name).
    :param language: ``'c++'`` or ``'c'``; the C table bans the C++ constructs as well.
    """
    for pattern, meaning in (BANNED_PATTERNS_C if language == 'c' else BANNED_PATTERNS):
        match = pattern.search(code)
        assert match is None, (f'{label}: MPR output contains {meaning} -- {match.group(0)!r} at offset '
                               f'{match.start()}\n{_context(code, match.start())}')
    assert_no_unqualified_runtime_calls(code, label)


def _context(code: str, offset: int, radius: int = 160) -> str:
    """The source line around ``offset``, for a failure message."""
    start = code.rfind('\n', 0, max(0, offset - radius)) + 1
    end = code.find('\n', offset + radius)
    return code[start:end if end != -1 else len(code)]


def compile_standalone(code: str, name: str = 'mpr_kernel', extra_flags: Any = (), language: str = 'c++') -> str:
    """Build ``code`` into a shared object and return its path.

    The translation unit is written into a FRESH temporary directory and compiled from there with
    no include path, so any header it names must be a system header. The directory is left in
    place for the caller's process lifetime -- ctypes needs the ``.so`` to stay on disk.

    :param code: the emitted translation unit.
    :param name: basename for the source/``.so`` pair.
    :param extra_flags: additional compiler flags (e.g. :data:`WARNING_FLAGS`).
    :param language: ``'c++'`` or ``'c'``, choosing the driver and the ``-std=`` flag.
    :returns: absolute path of the built shared object.
    :raises AssertionError: if the compile fails; the message carries the compiler's own diagnostics.
    """
    standard, suffix, _, _ = TOOLCHAINS[language]
    workdir = tempfile.mkdtemp(prefix=f'mpr_{name}_')
    source = os.path.join(workdir, name + suffix)
    library = os.path.join(workdir, f'lib{name}.so')
    with open(source, 'w') as handle:
        handle.write(code)
    command = [host_compiler(language), '-std=' + standard, *BASE_FLAGS, *extra_flags, source, '-o', library]
    proc = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
    assert proc.returncode == 0, (f'{name}: MPR output does not build with a bare host compiler\n'
                                  f'command: {" ".join(command)}\n{proc.stderr}')
    return library


def compile_diagnostics(code: str, name: str = 'mpr_kernel', language: str = 'c++') -> str:
    """Compiler stderr for ``code`` built with :data:`WARNING_FLAGS`.

    Separate from :func:`compile_standalone` so a zero-warning assertion reads as one, instead of
    riding on the numeric gate. Returns the raw stderr; empty means clean.
    """
    standard, suffix, _, _ = TOOLCHAINS[language]
    workdir = tempfile.mkdtemp(prefix=f'mpr_warn_{name}_')
    source = os.path.join(workdir, name + suffix)
    with open(source, 'w') as handle:
        handle.write(code)
    command = [
        host_compiler(language), '-std=' + standard, *BASE_FLAGS, *WARNING_FLAGS, source, '-o',
        os.path.join(workdir, f'lib{name}.so')
    ]
    proc = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
    assert proc.returncode == 0, f'{name}: MPR output does not build\ncommand: {" ".join(command)}\n{proc.stderr}'
    shutil.rmtree(workdir, ignore_errors=True)
    return proc.stderr


def build_standalone(code: str, name: str = 'mpr_kernel', language: str = 'c++') -> ctypes.CDLL:
    """Compile ``code`` and load the result. See :func:`compile_standalone` for the build rules."""
    return ctypes.CDLL(compile_standalone(code, name, language=language))


def entry_argtypes(sdfg: dace.SDFG) -> List[Any]:
    """ctypes argument types for ``sdfg``'s MPR entry point.

    MPR emits ``void <sdfg.name>(<arglist>)`` with the SAME argument order DaCe's own
    ``__program_<name>`` uses -- :meth:`dace.SDFG.arglist`, arrays first then scalars, each group
    sorted. Arrays are plain pointers; scalars and free symbols are passed by value.
    """
    argtypes: List[Any] = []
    for name, desc in sdfg.arglist().items():
        if isinstance(desc, dt.Scalar):
            argtypes.append(desc.dtype.as_ctypes())
        else:
            argtypes.append(ctypes.c_void_p)
    return argtypes


def call_standalone(library: ctypes.CDLL, sdfg: dace.SDFG, arguments: Dict[str, Any]) -> None:
    """Invoke ``sdfg``'s MPR entry point in ``library`` with ``arguments``.

    Array arguments are numpy arrays, passed by data pointer (so the kernel writes in place);
    scalar and symbol arguments are python numbers. Every entry of the SDFG's arglist must be
    supplied -- a missing one is a test bug, not a tolerated default.

    :param library: the loaded shared object from :func:`build_standalone`.
    :param sdfg: the SDFG whose arglist defines the signature.
    :param arguments: name -> value for every arglist entry.
    """
    arglist = sdfg.arglist()
    missing = sorted(set(arglist) - set(arguments))
    assert not missing, f'{sdfg.name}: MPR call is missing arguments {missing}'
    # An EXTRA name is the dangerous direction: a symbol the SDFG never used is absent from the
    # arglist, so it would be silently dropped and the kernel would run on an uninitialized extent.
    extra = sorted(set(arguments) - set(arglist))
    assert not extra, (f'{sdfg.name}: {extra} are not in the SDFG arglist {list(arglist)} and would be dropped; '
                       'a symbol the SDFG does not use never reaches the entry point')
    function = getattr(library, sdfg.name)
    function.argtypes = entry_argtypes(sdfg)
    function.restype = None
    values: List[Any] = []
    for name, desc in arglist.items():
        value = arguments[name]
        if isinstance(desc, dt.Scalar):
            values.append(desc.dtype.as_ctypes()(value))
        else:
            array = np.ascontiguousarray(value)
            assert array is value or array.base is value or np.shares_memory(array, value), (
                f'{sdfg.name}/{name}: argument was copied to make it contiguous, so writes would be lost; '
                'pass a C-contiguous array')
            values.append(ctypes.c_void_p(array.ctypes.data))
    function(*values)


def tolerance_for(dtype) -> Any:
    """``(rtol, atol)`` matched to precision: fp64 tight, fp32 relaxed, ints exact."""
    dt_ = np.dtype(dtype)
    if dt_.kind in 'iub':
        return 0.0, 0.0
    single = (dt_.kind == 'f' and dt_.itemsize <= 4) or (dt_.kind == 'c' and dt_.itemsize <= 8)
    return (1e-5, 1e-6) if single else (1e-9, 1e-11)


def assert_matches(reference: Dict[str, np.ndarray], mpr: Dict[str, np.ndarray], label: str = 'mpr') -> None:
    """Assert the MPR run reproduced ``reference`` (dtype-aware tolerance; exact for integers)."""
    assert set(reference) == set(mpr), f'{label}: output-key mismatch {sorted(reference)} vs {sorted(mpr)}'
    for name, expected in reference.items():
        got = mpr[name]
        assert expected.shape == got.shape, f'{label}/{name}: shape {expected.shape} vs {got.shape}'
        rtol, atol = tolerance_for(expected.dtype)
        assert np.allclose(expected, got, rtol=rtol, atol=atol, equal_nan=True), (
            f'{label}/{name}: MPR output diverges from the SDFG, '
            f'max|diff|={float(np.nanmax(np.abs(expected.astype(np.float64) - got.astype(np.float64)))):.3e}')


def wcr_sdfg(name: str, resolution: str, length: int = 32) -> dace.SDFG:
    """A map whose every iteration resolves into its OWN element through ``resolution``.

    Distinct targets, so the write does not conflict and the WCR reaches the NON-atomic lowering --
    the only conflict resolution MPR admits. Built directly rather than through ``@dace.program``
    because the Python frontend has no syntax that yields a non-conflicting custom resolution: an
    augmented assignment to distinct elements is a plain write, and a shared accumulator conflicts.

    :param name: the SDFG (and entry point) name.
    :param resolution: the WCR, as a lambda source string.
    :param length: the map extent and the array size.
    :returns: the built SDFG, ready to render.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [length], dace.float64)
    sdfg.add_array('out', [length], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': f'0:{length}'})
    tasklet = state.add_tasklet('t', {'x': None}, {'y': None}, 'y = x')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='x', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(tasklet,
                          exit_node,
                          state.add_write('out'),
                          src_conn='y',
                          memlet=dace.Memlet(data='out', subset='i', wcr=resolution))
    return sdfg


@pytest.fixture(scope='session')
def cxx() -> str:
    """The resolved host C++ compiler MPR output is built with."""
    return host_compiler('c++')
