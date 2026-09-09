# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How CPF spells the functions the DaCe runtime headers normally provide.

CPF (canonical parallel form) emits C++ -- or C23, which is the same semantics in a language
with no templates and no overloading -- that builds against a bare host compiler: no
``-I dace/runtime/include``, no ``libdace``. Every function the ordinary code generators reach for
therefore has to be re-expressed, and there are four ways to do it:

``STD_RENAMES``
    The C++ standard library has the same function under a different name. A pure rename.

``REWRITES``
    No standard function exists, but the operation is a short expression over its arguments. The
    call site is replaced by that expression, so nothing has to be declared.

``INLINE_DEFINITIONS``
    Neither of the above: the operation needs a real function (it is recursive, generic over
    signedness, or simply too long to inline at every use). CPF emits the definition once, at the
    top of the translation unit, and only for the helpers that translation unit actually calls.

``C_REWRITTEN_IN_NATIVE_CODE``
    C only, and the rare one: the construct cannot be a callable in C at all, so the CALL SITE is
    rewritten. The scan identities need the element type, which only the call site spells; the
    find-first takes a predicate, which is a C++ lambda and in C has to be pasted into the search
    as a macro argument. Every entry here is a C++ helper whose C answer is a rewrite rather than a
    definition, which is what lets the anti-rot tests still demand an answer for each one.

This module is deliberately a LEAF: it imports nothing from ``dace``. ``dace.symbolic`` (which
imports no code generator) and ``dace.codegen.cppunparse`` both consume it, and a shared table is
what keeps the two printers from drifting -- the same expression reaching C++ through a memlet
subset and through a tasklet body must produce the same text.

Names absent from all three tables are NOT silently passed through: see :func:`lowering_for`.
"""
import contextlib
import enum
import re
from typing import Dict, FrozenSet, NamedTuple, Optional, Set, Tuple


class Dialect(enum.Enum):
    """Which C++ vocabulary a printer may emit.

    An enum rather than a boolean, for two reasons that both cost correctness otherwise.

    The printers below are memoized (``symstr`` and ``_sym2cpp`` are both ``lru_cache``d), so the
    dialect MUST reach the cache key: a mode read from configuration inside a printer would let a
    ``RUNTIME`` call warm the cache and a later ``STANDALONE`` call read that entry back, emitting
    ``dace::math::pow`` into a translation unit that never includes the header. Passing the dialect
    as an argument is what puts it in the key.

    And a boolean fourth argument sits next to ``cpp_mode``, which is also a boolean: transposing
    the two at one call site is a silent behaviour change that no type checker would see. A
    ``Dialect`` cannot be mistaken for ``cpp_mode``.
    """
    #: The DaCe runtime headers are available; emit ``dace::`` names as usual.
    RUNTIME = 'runtime'
    #: No DaCe headers; emit only the C++ standard library and CPF's own inline definitions.
    STANDALONE = 'standalone'
    #: No DaCe headers and no C++ either: one C23 translation unit. The C standard library is not
    #: type-generic and has no templates, so every helper is a ``_Generic`` dispatch macro over a
    #: closed set of typed ``static inline`` functions (see :data:`C_INLINE_DEFINITIONS`).
    STANDALONE_C = 'standalone_c'
    #: No DaCe headers, one HIP translation unit holding both the host code and the kernels.
    #: C++ like :attr:`STANDALONE`, so every host-side helper and lowering it has applies here
    #: too; what it adds is the device vocabulary.
    #:
    #: The unit may include the ROCm toolkit's own headers -- ``hip_runtime.h`` and hipCUB --
    #: on the same footing as ``<math.h>`` and OpenMP on the host: they ship WITH the compiler
    #: that builds the unit, so requiring them adds no dependency a caller did not already have
    #: by choosing to compile for a GPU. What stays banned is ``dace/``, which does not.
    STANDALONE_HIP = 'standalone_hip'


#: The dialect a printer uses when its caller names none. Set only through :func:`dialect_scope`.
#:
#: An AMBIENT default exists because the dialect has to reach hundreds of ``sym2cpp`` call sites --
#: every memlet subset, every allocation extent, every loop bound -- spread over the code
#: generators. Threading a parameter through all of them would be a large diff whose every line is
#: identical, and one missed site emits a ``dace::`` name into a standalone unit.
#:
#: This does NOT reintroduce the memoization hazard the :class:`Dialect` docstring describes. The
#: ambient value is resolved by the OUTERMOST wrapper (``sym2cpp``, ``symstr``) and passed to the
#: memoized inner function as an ordinary argument, so it is still part of the cache key. Nothing
#: inside a cached function reads it.
_active_dialect: Dialect = Dialect.RUNTIME


def active_dialect() -> Dialect:
    """The dialect currently in force (see :func:`dialect_scope`)."""
    return _active_dialect


@contextlib.contextmanager
def dialect_scope(dialect: Dialect):
    """Make ``dialect`` the ambient default for the duration of the block.

    Restores the previous value on the way out, exception or not -- a code generation that raises
    must not leave every later ``sym2cpp`` in the process emitting standalone text.
    """
    global _active_dialect
    previous = _active_dialect
    _active_dialect = dialect
    try:
        yield dialect
    finally:
        _active_dialect = previous


#: The dialects that emit a self-contained translation unit. Everything CPF refuses -- device
#: code, a state struct, an external buffer handshake -- it refuses for both of them, so the many
#: call sites that ask "is this an CPF rendering" ask through :func:`standalone`.
STANDALONE_DIALECTS = frozenset({Dialect.STANDALONE, Dialect.STANDALONE_C, Dialect.STANDALONE_HIP})

#: The standalone dialects that render DEVICE code. A rendering under one of these admits GPU
#: storages and schedules, and emits both the host code and the kernels into the one unit.
DEVICE_DIALECTS = frozenset({Dialect.STANDALONE_HIP})


def standalone() -> bool:
    """Whether the ambient dialect renders a self-contained unit, C++ or C."""
    return _active_dialect in STANDALONE_DIALECTS


def device() -> bool:
    """Whether the ambient dialect renders device code into the standalone unit."""
    return _active_dialect in DEVICE_DIALECTS


def standalone_c() -> bool:
    """Whether the ambient dialect is :attr:`Dialect.STANDALONE_C`.

    Separate from :func:`standalone` because the two answer different questions: "may I name a
    DaCe runtime symbol" (no, for both) versus "may I write C++" (no, only for this one).
    """
    return _active_dialect is Dialect.STANDALONE_C


#: Node GUID -> ``(origin GUID, description)`` of the library node it came from, for the rendering
#: in progress. Set only through :func:`provenance_scope`.
#:
#: A pure expansion replaces ``Gemm`` with a loop nest, and the loop nest does not say it was a
#: matrix product. CPF restores that: the description is recorded when the node is expanded and
#: written as a comment where the expansion's code is emitted. Keyed by GUID rather than by node
#: object because the code generator runs its own lowering (inlining, copy lifting) between the
#: expansion and the emission.
#:
#: Ambient for the same reason the dialect is: the emission points are inside the code generators,
#: several call layers below anything that knows an expansion happened. Nothing memoized reads it.
_provenance: Dict[str, Tuple[str, str]] = {}


def describe(guid: str) -> Optional[Tuple[str, str]]:
    """``(origin, description)`` for the library node that produced ``guid``, if CPF recorded one.

    ``origin`` is the GUID of the library node itself. The emitter dedupes on it rather than on the
    description text: two separate ``Gemm`` nodes in one program are two things worth commenting,
    while the forty tasklets of one expansion are one.
    """
    return _provenance.get(guid)


def hint_comment(hint: Optional[str], indent: str = '') -> str:
    """Render a ``specialization_hint`` as a comment block, or ``''`` when there is none.

    A no-op outside a standalone rendering: the hint is a note to whoever reads the maximally
    parallel form -- a specializing pass, or a person -- and the ordinary build has a target
    already, so emitting it there would be noise in code nobody reads.

    Multi-line hints stay multi-line. The alternatives a hint describes are per-device and do not
    read as one sentence: ``CPU: ... / GPU: ...`` is the shape, one line each.
    """
    if not hint or not standalone():
        return ''
    return ''.join(f'{indent}// {line}\n' for line in str(hint).splitlines() if line.strip())


@contextlib.contextmanager
def provenance_scope(provenance: Dict[str, Tuple[str, str]]):
    """Make ``provenance`` the ambient GUID -> description map for the duration of the block."""
    global _provenance
    previous = _provenance
    _provenance = provenance
    try:
        yield provenance
    finally:
        _provenance = previous


#: Runtime function -> the ``std`` function with identical semantics and arity.
#:
#: ``ROUND`` is here rather than in :data:`REWRITES` because the runtime's ``ROUND`` is literally
#: ``return round(value);`` -- both round half away from zero, so the rename is exact.
STD_RENAMES: Dict[str, str] = {
    'Abs': 'std::abs',
    'abs': 'std::abs',
    'ceiling': 'std::ceil',
    'ceil': 'std::ceil',
    'floor': 'std::floor',
    'ROUND': 'std::round',
    'round': 'std::round',
    'conj': 'std::conj',
    'exp2': 'std::exp2',
    'expm1': 'std::expm1',
    'log1p': 'std::log1p',
    'log2': 'std::log2',
    'frexp': 'std::frexp',
    'ldexp': 'std::ldexp',
    'ilogb': 'std::ilogb',
    'isfinite': 'std::isfinite',
    'isinf': 'std::isinf',
    'isnan': 'std::isnan',
    'signbit': 'std::signbit',
    'gcd': 'std::gcd',
    'lcm': 'std::lcm',
    'sin': 'std::sin',
    'cos': 'std::cos',
    'tan': 'std::tan',
    'asin': 'std::asin',
    'acos': 'std::acos',
    'atan': 'std::atan',
    'atan2': 'std::atan2',
    'sinh': 'std::sinh',
    'cosh': 'std::cosh',
    'tanh': 'std::tanh',
    'exp': 'std::exp',
    'fabs': 'std::fabs',
    'log': 'std::log',
    'log10': 'std::log10',
    'sqrt': 'std::sqrt',
    'cbrt': 'std::cbrt',
    'pow': 'std::pow',
    'fma': 'std::fma',
    'erf': 'std::erf',
    'erfc': 'std::erfc',
    'tgamma': 'std::tgamma',
    'lgamma': 'std::lgamma',
    'trunc': 'std::trunc',
    'hypot': 'std::hypot',
}

#: Runtime function -> ``(arity, format string over the printed arguments)``.
#:
#: Every replacement parenthesizes each argument and the whole result: these strings are spliced
#: into a larger expression, and an unparenthesized ``a - b`` would rebind against a neighbouring
#: operator. Each argument appears EXACTLY ONCE -- a template repeating ``{0}`` would duplicate
#: whatever expression the caller printed, and anything needing its argument twice, or needing to
#: name its argument's type, is an inline definition instead (see :data:`INLINE_DEFINITIONS`).
#:
#: ``reciprocal`` keeps the runtime's integer-division behaviour for an integer argument: the
#: runtime is ``T(1) / a``, which for ``T = int`` truncates exactly as ``1 / (a)`` does here.
#: ``Mod`` is Fortran ``MOD`` on integers, which is plain C++ ``%``; the floating Fortran ``MOD``
#: is ``Mod_float`` and needs its argument twice, so it is a definition.
REWRITES: Dict[str, Tuple[int, str]] = {
    'reciprocal': (1, '(1 / ({0}))'),
    # A complex's components. The runtime helpers forward to ``.real()`` / ``.imag()`` on the
    # underlying ``std::complex``, which standalone output can call directly.
    're': (1, '(({0}).real())'),
    'im': (1, '(({0}).imag())'),
    'iround': (1, '(static_cast<int>(std::round({0})))'),
    'ITE': (3, '(({0}) ? ({1}) : ({2}))'),
    'IfExpr': (3, '(({0}) ? ({1}) : ({2}))'),
    'deg2rad': (1, '(({0}) * 0.017453292519943295)'),
    'rad2deg': (1, '(({0}) * 57.29577951308232)'),
    'left_shift': (2, '(({0}) << ({1}))'),
    'right_shift': (2, '(({0}) >> ({1}))'),
    'bitwise_and': (2, '(({0}) & ({1}))'),
    'bitwise_or': (2, '(({0}) | ({1}))'),
    'bitwise_xor': (2, '(({0}) ^ ({1}))'),
    'bitwise_invert': (1, '(~({0}))'),
    'int_floor': (2, '(({0}) / ({1}))'),
    'Mod': (2, '(({0}) % ({1}))'),
    'np_float_pow': (2, '(std::pow(static_cast<double>({0}), static_cast<double>({1})))'),
}

#: Runtime function -> the C++ definition CPF emits for it.
#:
#: Emitted only when the translation unit calls the helper (see :func:`definitions_for`), so a
#: kernel that never rounds up never carries an ``int_ceil``. Each is ``static`` so several CPF
#: translation units can be compiled together, and templated so it works for whatever width the
#: index arithmetic settled on. A helper needs a definition rather than a rewrite when it names its
#: argument's type, uses an argument more than once, dispatches on integral-vs-floating, or writes
#: through out-parameters.
#:
#: The modulo family is three DIFFERENT operations and the names do not say which is which:
#: ``mod``/``py_mod``/``floor_mod``/``Modulo`` are FLOORED (result takes the sign of the divisor,
#: ``mod(-1, 5) == 4``), while ``cpp_mod``/``Mod``/``Mod_float`` TRUNCATE toward zero
#: (``cpp_mod(-1, 5) == -1``). Collapsing them onto one spelling would be a silent wrong answer for
#: half of them.
#:
#: ``sign`` and ``heaviside`` are here rather than in :data:`REWRITES` for a dtype reason. The
#: runtime's ``sign`` is ``T((T(0) < x) - (x < T(0)))``: the comparisons yield ``bool``, their
#: difference is ``int``, and the cast back to ``T`` is what keeps ``sign(2.5)`` a ``double``.
#: Written as a textual rewrite the cast has no ``T`` to name, so the result would decay to ``int``
#: and a following ``sign(x) / 2`` would silently become integer division.
#:
#: ``cpp_divmod``, ``py_divmod``, ``np_modf`` and ``np_frexp`` return ``void`` and write through
#: reference out-parameters -- they are statements, not expressions, so they only ever needed a
#: definition.
#:
#: Every definition that CAN be evaluated at compile time is ``constexpr`` (never ``consteval``,
#: which would forbid the runtime calls that are the normal case). ``Modulo``, ``Modulo_float``,
#: ``np_modf`` and ``np_frexp`` are not: each reaches a standard function that is not ``constexpr``
#: before C++23 for every instantiation. Marking those ``constexpr`` anyway is ill-formed with no
#: diagnostic required, and it LOOKS fine -- GCC folds ``std::floor`` as a builtin and accepts it,
#: while clang rejects the same code. Measured, not assumed; see the constexpr probes in
#: ``tests/codegen/cpf/test_lowering_table.py``.
INLINE_DEFINITIONS: Dict[str, str] = {
    'cpf_max':
    'template <typename T>\n'
    'static constexpr inline T cpf_max(const T& value) {\n'
    '    return value;\n'
    '}\n'
    'template <typename T, typename... Ts>\n'
    'static constexpr inline typename std::common_type<T, Ts...>::type cpf_max(const T& a, const Ts&... rest) {\n'
    '    return (a < cpf_max(rest...)) ? cpf_max(rest...) : a;\n'
    '}',
    'cpf_min':
    'template <typename T>\n'
    'static constexpr inline T cpf_min(const T& value) {\n'
    '    return value;\n'
    '}\n'
    'template <typename T, typename... Ts>\n'
    'static constexpr inline typename std::common_type<T, Ts...>::type cpf_min(const T& a, const Ts&... rest) {\n'
    '    return (cpf_min(rest...) < a) ? cpf_min(rest...) : a;\n'
    '}',
    'sign':
    'template <typename T>\n'
    'static constexpr inline T sign(const T& value) {\n'
    '    return T((T(0) < value) - (value < T(0)));\n'
    '}',
    'sgn':
    'template <typename T>\n'
    'static constexpr inline T sgn(const T& value) {\n'
    '    return T((T(0) < value) - (value < T(0)));\n'
    '}',
    'sign_numpy_2':
    'template <typename T>\n'
    'static constexpr inline T sign_numpy_2(const T& value) {\n'
    '    return T((T(0) < value) - (value < T(0)));\n'
    '}\n'
    'template <typename T>\n'
    'static inline std::complex<T> sign_numpy_2(const std::complex<T>& value) {\n'
    '    return (value.real() != 0 && value.imag() != 0) ? value / std::abs(value) : std::complex<T>(0, 0);\n'
    '}',
    'heaviside':
    'template <typename T>\n'
    'static constexpr inline T heaviside(const T& value, const T& at_zero) {\n'
    '    return (value < T(0)) ? T(0) : ((value > T(0)) ? T(1) : at_zero);\n'
    '}\n'
    'template <typename T>\n'
    'static constexpr inline T heaviside(const T& value) {\n'
    '    return (value > T(0)) ? T(1) : T(0);\n'
    '}',
    # ``ifloor`` is the UNARY floor-to-integer the tasklet printer emits for a floor division
    # (``dace::math::ifloor(a / b)``), not the binary ``int_floor``. Integral input is already
    # floored, so the runtime returns it unchanged -- and that identity case is why this is a
    # definition and not a rewrite: ``(int)std::floor(x)`` on an int64 would truncate it to 32 bits.
    'ifloor':
    'template <typename T>\n'
    'static constexpr inline auto ifloor(const T& value) {\n'
    '    if constexpr (std::is_integral_v<T>) {\n'
    '        return value;\n'
    '    } else {\n'
    '        return static_cast<int>(std::floor(value));\n'
    '    }\n'
    '}',
    # --- prefix scans -------------------------------------------------------------------------
    # The DaCe runtime provides these in ``dace/scan.hpp``, one function per (op, inclusive) pair
    # because an OpenMP reduction identifier cannot be a template parameter -- the operator has to
    # be spelled into the clause. CPF reproduces them rather than rewriting a scan into a
    # sequential loop: the ``inscan`` form IS the parallel one, and a rendering that quietly
    # serialized every prefix sum would not be a canonical parallel form.
    'min_identity':
    'template <typename T>\n'
    'static inline T min_identity() {\n'
    '    return std::numeric_limits<T>::has_infinity ? T(std::numeric_limits<T>::infinity())\n'
    '                                                : std::numeric_limits<T>::max();\n'
    '}',
    'max_identity':
    'template <typename T>\n'
    'static inline T max_identity() {\n'
    '    return std::numeric_limits<T>::has_infinity ? T(-std::numeric_limits<T>::infinity())\n'
    '                                                : std::numeric_limits<T>::lowest();\n'
    '}',
    'scan_incl_sum':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_incl_sum(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, +:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = acc + f[i];\n'
    '        #pragma omp scan inclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_incl_product':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_incl_product(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, *:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = acc * f[i];\n'
    '        #pragma omp scan inclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_incl_min':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_incl_min(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, min:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = cpf_min(acc, static_cast<T>(f[i]));\n'
    '        #pragma omp scan inclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_incl_max':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_incl_max(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, max:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = cpf_max(acc, static_cast<T>(f[i]));\n'
    '        #pragma omp scan inclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    # The exclusive form runs its two phases the other way round: the ``scan`` directive splits the
    # body into an input phase and a scan phase, and for ``exclusive`` the SCAN phase is the one
    # before the directive. Written the inclusive way round it still compiles and stores the seed
    # into every element, so these mirror ``dace/scan.hpp`` statement for statement.
    'scan_excl_sum':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_sum(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, +:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        o[i] = acc;\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        acc = acc + f[i];\n'
    '    }\n'
    '}',
    'scan_excl_product':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_product(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, *:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        o[i] = acc;\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        acc = acc * f[i];\n'
    '    }\n'
    '}',
    'scan_excl_min':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_min(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, min:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        o[i] = acc;\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        acc = cpf_min(acc, static_cast<T>(f[i]));\n'
    '    }\n'
    '}',
    'scan_excl_max':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_max(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, max:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        o[i] = acc;\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        acc = cpf_max(acc, static_cast<T>(f[i]));\n'
    '    }\n'
    '}',
    # --- find-first ---------------------------------------------------------------------------
    # An early-exit loop lifts to a ``FindFirst`` library node whose expansion calls the runtime's
    # short-circuiting parallel search. CPF emits that search rather than unrolling it back into a
    # sequential scan, for the same reason it emits the inscan form of a prefix sum: the cancelling
    # parallel shape IS the rendering, and serializing it would answer a different question.
    'find_first_chunk':
    '#ifdef _OPENMP\n'
    '#include <omp.h>\n'
    '#endif\n'
    'static inline long long find_first_chunk(long long span, bool parallel) {\n'
    '    // Grows as sqrt(span): too big a chunk scans past the answer on one thread, too small a\n'
    '    // one pays dispatch on chunks the answer makes dead. The floor binds below ~64k elements.\n'
    '    constexpr double chunk_scale = 8.0;\n'
    '    constexpr long long chunks_per_thread = 4;\n'
    '    long long chunk = (long long)(chunk_scale * std::sqrt((double)span));\n'
    '    long long threads = 1;\n'
    '#ifdef _OPENMP\n'
    '    if (parallel) threads = (long long)omp_get_max_threads();\n'
    '#endif\n'
    '    long long ceiling = span / (chunks_per_thread * threads);\n'
    '    if (ceiling < 1) ceiling = 1;\n'
    '    if (chunk > ceiling) chunk = ceiling;\n'
    '    if (chunk < 1) chunk = 1;\n'
    '    return chunk;\n'
    '}',
    'find_first_index':
    'template <typename Pred>\n'
    'static inline long long find_first_index(long long begin, long long end, Pred pred, bool parallel) {\n'
    '    // The answer is a min-reduction and is exact; the hint is shared and races by design --\n'
    '    // every value it takes is a real firing index, so a lost update costs pruning, never\n'
    '    // correctness. Folding the two into one word is exactly that lost-update bug.\n'
    '    constexpr long long simd_block = 64;\n'
    '    if (begin >= end) return end;\n'
    '    const long long span = end - begin;\n'
    '    const long long chunk = find_first_chunk(span, parallel);\n'
    '    const long long nchunks = (span + chunk - 1) / chunk;\n'
    '    long long best = end;\n'
    '    long long hint = end;\n'
    '    #pragma omp parallel for schedule(dynamic, 1) if (parallel : parallel) reduction(min : best)\n'
    '    for (long long c = 0; c < nchunks; ++c) {\n'
    '        long long seen;\n'
    '        #pragma omp atomic read\n'
    '        seen = hint;\n'
    '        const long long lo = begin + c * chunk;\n'
    '        if (lo >= seen) continue;\n'
    '        long long hi = lo + chunk;\n'
    '        if (hi > end) hi = end;\n'
    '        if (hi > seen) hi = seen;\n'
    '        long long found = end;\n'
    '        for (long long b = lo; b < hi; b += simd_block) {\n'
    '            long long be = b + simd_block;\n'
    '            if (be > hi) be = hi;\n'
    '            long long block = end;\n'
    '            // A vectorized loop cannot break, so the block is the early-exit granularity.\n'
    '            #pragma omp simd reduction(min : block)\n'
    '            for (long long i = b; i < be; ++i) {\n'
    '                const long long v = pred(i) ? i : end;\n'
    '                block = v < block ? v : block;\n'
    '            }\n'
    '            if (block < end) { found = block; break; }\n'
    '        }\n'
    '        if (found < end) {\n'
    '            if (found < best) best = found;\n'
    '            long long cur;\n'
    '            #pragma omp atomic read\n'
    '            cur = hint;\n'
    '            if (found < cur) {\n'
    '                #pragma omp atomic write\n'
    '                hint = found;\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '    return best;\n'
    '}',
    'int_ceil':
    'template <typename T, typename U>\n'
    'static constexpr inline auto int_ceil(const T& numerator, const U& denominator) {\n'
    '    return (numerator + denominator - 1) / denominator;\n'
    '}',
    'int_floor_ni':
    'template <typename T, typename U>\n'
    'static constexpr inline auto int_floor_ni(const T& numerator, const U& denominator) {\n'
    '    auto quotient = numerator / denominator;\n'
    '    auto remainder = numerator % denominator;\n'
    '    return quotient - ((remainder != 0) && ((remainder < 0) != (denominator < 0)));\n'
    '}',
    'py_floor':
    'template <typename T, typename U>\n'
    'static constexpr inline auto py_floor(const T& numerator, const U& denominator) {\n'
    '    if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {\n'
    '        return int_floor_ni(numerator, denominator);\n'
    '    } else {\n'
    '        return std::floor(numerator / denominator);\n'
    '    }\n'
    '}',
    'py_mod':
    'template <typename T, typename U>\n'
    'static constexpr inline auto py_mod(const T& numerator, const U& denominator) {\n'
    '    return numerator - py_floor(numerator, denominator) * denominator;\n'
    '}',
    'floor_mod':
    'template <typename T, typename U>\n'
    'static constexpr inline auto floor_mod(const T& numerator, const U& denominator) {\n'
    '    return py_mod(numerator, denominator);\n'
    '}',
    'mod':
    'template <typename T, typename U>\n'
    'static constexpr inline auto mod(const T& value, const U& modulus) {\n'
    '    return ((value % modulus) + modulus) % modulus;\n'
    '}',
    'cpp_mod':
    'template <typename T, typename U>\n'
    'static constexpr inline auto cpp_mod(const T& numerator, const U& denominator) {\n'
    '    if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {\n'
    '        return numerator % denominator;\n'
    '    } else {\n'
    '        return std::fmod(numerator, denominator);\n'
    '    }\n'
    '}',
    'Mod_float':
    'template <typename T>\n'
    'static constexpr inline T Mod_float(const T& value, const T& modulus) {\n'
    '    return value - static_cast<int>(value / modulus) * modulus;\n'
    '}',
    'Modulo':
    'template <typename T>\n'
    'static inline T Modulo(const T& value, const T& modulus) {\n'
    '    return value - static_cast<T>(std::floor(static_cast<double>(value) / modulus)) * modulus;\n'
    '}',
    'Modulo_float':
    'template <typename T>\n'
    'static inline T Modulo_float(const T& value, const T& modulus) {\n'
    '    return value - static_cast<T>(std::floor(value / modulus)) * modulus;\n'
    '}',
    'cpp_divmod':
    'template <typename T>\n'
    'static constexpr inline void cpp_divmod(const T& numerator, const T& denominator, T& quotient,\n'
    '                                        T& remainder) {\n'
    '    quotient = static_cast<T>(numerator / denominator);\n'
    '    remainder = static_cast<T>(numerator % denominator);\n'
    '}',
    'py_divmod':
    'template <typename T>\n'
    'static constexpr inline void py_divmod(const T& numerator, const T& denominator, T& quotient,\n'
    '                                       T& remainder) {\n'
    '    cpp_divmod(numerator, denominator, quotient, remainder);\n'
    '    T correction = (remainder != 0 && ((remainder < 0) != (denominator < 0)));\n'
    '    quotient -= correction;\n'
    '    remainder += correction * denominator;\n'
    '}',
    'np_modf':
    'template <typename T>\n'
    'static inline void np_modf(const T& value, T& integral, T& fractional) {\n'
    '    if constexpr (std::is_integral_v<T>) {\n'
    '        integral = value;\n'
    '        fractional = T(0);\n'
    '    } else {\n'
    '        fractional = std::modf(value, &integral);\n'
    '    }\n'
    '}',
    'np_frexp':
    'template <typename T>\n'
    'static inline void np_frexp(const T& value, T& mantissa, int& exponent) {\n'
    '    mantissa = std::frexp(value, &exponent);\n'
    '}',
    'ipow':
    'template <typename T, typename U>\n'
    'static constexpr inline T ipow(T base, U exponent) {\n'
    '    T result = 1;\n'
    '    while (exponent > 0) {\n'
    '        if (exponent & 1) { result *= base; }\n'
    '        base *= base;\n'
    '        exponent >>= 1;\n'
    '    }\n'
    '    return result;\n'
    '}',
    'logical_left_shift':
    'template <typename T, typename U>\n'
    'static constexpr inline T logical_left_shift(const T& value, const U& amount) {\n'
    '    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) << amount);\n'
    '}',
    'logical_right_shift':
    'template <typename T, typename U>\n'
    'static constexpr inline T logical_right_shift(const T& value, const U& amount) {\n'
    '    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) >> amount);\n'
    '}',
}

#: Definitions each definition calls. Emission is dependency-first (see :func:`definitions_for`).
DEFINITION_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    'scan_incl_min': ('min_identity', 'cpf_min'),
    'scan_incl_max': ('max_identity', 'cpf_max'),
    'scan_excl_min': ('min_identity', 'cpf_min'),
    'scan_excl_max': ('max_identity', 'cpf_max'),
    'find_first_index': ('find_first_chunk', ),
    'py_floor': ('int_floor_ni', ),
    'py_mod': ('py_floor', ),
    'floor_mod': ('py_mod', ),
    'py_divmod': ('cpp_divmod', ),
}

#: System headers each inline definition needs, beyond :data:`BASE_HEADERS`.
DEFINITION_HEADERS: Dict[str, Tuple[str, ...]] = {
    'ifloor': ('<type_traits>', ),
    'min_identity': ('<limits>', ),
    'max_identity': ('<limits>', ),
    'logical_left_shift': ('<type_traits>', ),
    'logical_right_shift': ('<type_traits>', ),
    'py_floor': ('<type_traits>', ),
    'cpp_mod': ('<type_traits>', ),
    'np_modf': ('<type_traits>', ),
}

#: ``Max``/``Min`` are variadic in the runtime, which ``std::max``/``std::min`` are not: those are
#: binary or take an ``initializer_list``. The ORDER now matches (a later argument wins only by
#: comparing strictly better), so the difference is arity and mixed-type promotion, but CPF still
#: emits the runtime's own definition (see :data:`INLINE_DEFINITIONS`) so the two cannot drift.
VARIADIC_MINMAX: Dict[str, str] = {'Max': 'cpf_max', 'Min': 'cpf_min', 'max': 'cpf_max', 'min': 'cpf_min'}

#: Headers CPF always includes: the exact-width integer types and the maths every kernel may reach,
#: plus the two the readable generator's own allocations need -- ``<new>`` for the aligned
#: ``operator new[](std::align_val_t)`` it allocates heap transients with, and ``<type_traits>``
#: for the ``std::is_trivially_destructible`` static assertion it pairs with the matching delete.
#: ``<cstdlib>`` is for ``std::abort``, which canonicalization writes into the assumption-guard
#: tasklet (``if ((N < 0)) { std::abort(); }``) -- a body no printer sees, so nothing else would
#: pull the declaration in. ``<cassert>`` is the same case one level down: code generation guards
#: every map with a non-unit step by ``assert((step) > 0 && "...")``, which it writes directly into
#: the stream rather than through a printer, so no call-site table can discover it.
BASE_HEADERS: Tuple[str, ...] = ('<cstdint>', '<cmath>', '<cstring>', '<cstdlib>', '<algorithm>', '<cassert>',
                                 '<complex>', '<numeric>', '<new>', '<type_traits>')

#: What a HIP unit adds to :data:`BASE_HEADERS`. Both ship with the ROCm toolkit, so a unit that
#: includes them needs nothing a caller compiling for an AMD GPU does not already have -- the same
#: standing OpenMP has on the host side, and the reason they are not what CPF exists to avoid.
#: hipCUB supplies the device scan, reduce and arg-reduce the library nodes expand into.
HIP_BASE_HEADERS: Tuple[str, ...] = ('<hip/hip_runtime.h>', '<hipcub/hipcub.hpp>')

#: What ``dace/dace.h`` supplies that a DEVICE unit still needs, written out inline.
#:
#: These are not lowerings -- the generated device code is already correct C++ -- they are the
#: handful of spellings the header defines and the unit therefore has to define for itself: the
#: annotation macros, the backend-neutral ``gpu*`` aliases the code generator emits so one text
#: serves CUDA and HIP, and an error check. Everything else the header would have brought (the
#: reduction functors, the copy templates, the runtime context) is lowered or replaced.
#:
#: ``cpf_gpu_context`` is the replacement for ``dace::cuda::Context``: the generated frame reaches
#: it as ``__state->gpu_context->streams``, so keeping that SHAPE is what lets the device code
#: stand unaltered. One stream, because canon offloads onto the default stream
#: (``max_concurrent_streams = -1``); the array is what the emitted indexing expects.
#:
#: This is the part EVERY device unit needs. What only some need is in
#: :data:`HIP_DEVICE_BLOCKS`, selected the same way the C helpers are -- from the finished text.
HIP_DEVICE_CORE: str = """\
#define DACE_EXPORTED
#define DACE_HDFI __host__ __device__ __forceinline__
#define DACE_HFI __host__ __forceinline__
#define DACE_DFI __device__ __forceinline__
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;
static constexpr gpuError_t gpuSuccess = hipSuccess;
static constexpr gpuError_t gpuErrorMemoryAllocation = hipErrorOutOfMemory;
#define DACE_GPU_CHECK(expr) do {                                                             \\
        gpuError_t __cpf_status = (expr);                                                     \\
        if (__cpf_status != gpuSuccess) {                                                     \\
            fprintf(stderr, "%s:%d: GPU error %d (%s) in %s\\n", __FILE__, __LINE__,           \\
                    (int)__cpf_status, hipGetErrorString(__cpf_status), #expr);               \\
            abort();                                                                          \\
        }                                                                                     \\
    } while (0)

//: The one stream canon offloads onto, in the shape the generated frame indexes.
struct cpf_gpu_context {
    gpuStream_t streams[1];
    gpuEvent_t events[1];
    gpuError_t lasterror;
};
"""

#: Device preamble blocks only SOME units need, keyed by the identifier that pulls each one in.
#:
#: Same principle as :func:`helpers_used`, and for the same reason: what a unit needs is a property
#: of the finished text, not of the SDFG. Emitting the lot unconditionally cost every GPU form the
#: 30-line atomic template and the cub aliases whether or not it reduced anything -- measured over
#: the 40-kernel corpus, the atomic block was dead in 33 of 40 and the cub aliases in 32, about 40
#: lines of a 200-line form. The form is read by an agent under a token budget, so text it has no
#: use for is not free.
HIP_DEVICE_BLOCKS: Dict[str, str] = {
    'DACE_KERNEL_LAUNCH_CHECK':
    """\
#define DACE_KERNEL_LAUNCH_CHECK(err, name, gx, gy, gz, bx, by, bz)                            \\
    do {                                                                                      \\
        if ((err) != gpuSuccess) {                                                            \\
            fprintf(stderr, "%s launch failed (grid %d,%d,%d block %d,%d,%d): %s\\n", (name),  \\
                    (int)(gx), (int)(gy), (int)(gz), (int)(bx), (int)(by), (int)(bz),         \\
                    hipGetErrorString(err));                                                  \\
            abort();                                                                          \\
        }                                                                                     \\
    } while (0)
""",
    'gpucub':
    """\
//: The backend cub, under the name the code generator writes -- what gpucub.cuh aliases. The
//: DACE_CUB_*_OP macros below are the binary-operator functors a DeviceScan / DeviceReduce
//: expansion passes by macro, in the spelling cub_compat.cuh selects for HIP. Only that arm
//: applies: hipCUB keeps the functor structs CCCL 3 dropped, and a unit built by hipcc is never
//: built against CCCL.
namespace gpucub = hipcub;
""",
    'DACE_CUB_SUM_OP':
    '#define DACE_CUB_SUM_OP ::gpucub::Sum()\n',
    'DACE_CUB_MIN_OP':
    '#define DACE_CUB_MIN_OP ::gpucub::Min()\n',
    'DACE_CUB_MAX_OP':
    '#define DACE_CUB_MAX_OP ::gpucub::Max()\n',
    'DACE_CUB_MUL_OP':
    '#define DACE_CUB_MUL_OP [] __device__(auto __cpf_a, auto __cpf_b) { return __cpf_a * __cpf_b; }\n',
    'cpf_gpu_atomic':
    """\
//: A conflicting accumulation, applied atomically under any binary operator.
//:
//: HIP spells only a few operator/type pairs as an intrinsic (atomicAdd on the arithmetic types,
//: atomicMin/atomicMax on the integers), and an SDFG's write-conflict resolution is any of nine
//: reductions over any element type, so one CAS loop covers the set where a table of intrinsics
//: would leave holes. On a tree-reduced accumulator this runs ONCE PER BLOCK, after
//: gpucub::BlockReduce has folded the block's partials, so the loop is off the per-element path.
template <typename T, typename V, typename Op>
__device__ inline void cpf_gpu_atomic(T *address, V value, Op op) {
    using cpf_atomic_word =
        typename std::conditional<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;
    static_assert(sizeof(T) == sizeof(cpf_atomic_word), "no atomic word as wide as the accumulator");
    cpf_atomic_word *word = reinterpret_cast<cpf_atomic_word *>(address);
    cpf_atomic_word old = *word;
    cpf_atomic_word assumed;
    do {
        assumed = old;
        T updated = op(__builtin_bit_cast(T, assumed), static_cast<T>(value));
        old = atomicCAS(word, assumed, __builtin_bit_cast(cpf_atomic_word, updated));
    } while (assumed != old);
}
""",
}

#: Emission order for :data:`HIP_DEVICE_BLOCKS`. A dict preserves insertion order, but the blocks
#: are selected into a set, so the order a unit gets them in has to be stated rather than inherited
#: from however the set happened to iterate -- two runs of the same SDFG must render byte-identical.
HIP_DEVICE_BLOCK_ORDER: Tuple[str, ...] = tuple(HIP_DEVICE_BLOCKS)

#: What each block needs in turn. The cub operator macros expand to ``::gpucub::`` names, so a unit
#: that mentions one needs the namespace alias even when it never writes ``gpucub`` itself.
HIP_DEVICE_BLOCK_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    'DACE_CUB_SUM_OP': ('gpucub', ),
    'DACE_CUB_MIN_OP': ('gpucub', ),
    'DACE_CUB_MAX_OP': ('gpucub', ),
    'DACE_CUB_MUL_OP': ('gpucub', ),
    'cpf_gpu_atomic': ('gpucub', ),
}


def hip_device_preamble(code: str) -> str:
    """:data:`HIP_DEVICE_CORE` plus the blocks ``code`` actually reaches for.

    :param code: the emitted translation unit, WITHOUT its preamble -- so a block's own definition
                 never counts as a use of it.
    :returns: the device preamble, core first and blocks in :data:`HIP_DEVICE_BLOCK_ORDER`.
    """
    needed: Set[str] = set()
    pending = [name for name in HIP_DEVICE_BLOCKS if re.search(rf'\b{re.escape(name)}\b', code)]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(dep for dep in HIP_DEVICE_BLOCK_DEPENDENCIES.get(name, ()) if dep not in needed)
    parts = [HIP_DEVICE_CORE]
    parts += [HIP_DEVICE_BLOCKS[name] for name in HIP_DEVICE_BLOCK_ORDER if name in needed]
    return '\n'.join(parts)


def device_entry_prologue(state_struct: str) -> str:
    """The device setup CPF's single entry function opens with.

    An ordinary build does this in ``__dace_init_<name>`` and undoes it in ``__dace_exit_<name>``,
    both taking the state pointer the caller kept between invocations. CPF has one entry point and
    no handshake, so the context is a LOCAL: created here, destroyed by the scope guard on the way
    out, whichever way the function leaves.

    The stream is the default one, which is what canon offloads onto
    (``compiler.cuda.max_concurrent_streams = -1``), so there is nothing to create and nothing to
    destroy -- only the device to select and the work to drain.

    :param state_struct: the mangled state struct name the generated body dereferences.
    :returns: the prologue lines, indented one level.
    """
    return f"""\
    // The device handshake, which a single entry point has nowhere else to put.
    cpf_gpu_context __cpf_context{{}};
    __cpf_context.streams[0] = nullptr;
    {state_struct} __cpf_state{{&__cpf_context}};
    {state_struct} *__state = &__cpf_state;
    struct __cpf_drain {{
        ~__cpf_drain() {{ DACE_GPU_CHECK(hipDeviceSynchronize()); }}
    }} __cpf_drain_guard;
"""


#: Runtime functions CPF deliberately does NOT lower, and why. Reaching one is a refusal, not a
#: pass-through: the name is declared by a DaCe header CPF does not include, so passing it through
#: would produce a translation unit that does not build.
UNSUPPORTED: Dict[str, str] = {}

#: Math functions a RUNTIME-dialect printer must qualify as ``dace::math::``, rather than leave
#: bare for unqualified lookup to resolve.
#:
#: A bare ``sqrt(x)`` binds to ``std::sqrt``, whose ``float`` / ``double`` / ``long double``
#: overloads are all equally good for a 16-bit float: ``dace::float16`` IS CUDA's ``half``, and its
#: many non-explicit conversion operators each reach a different overload through a different
#: conversion, so nvcc calls the call ambiguous and rejects the translation unit. Each name here
#: carries a non-template ``dace::float16`` / ``dace::bfloat16`` overload in ``dace/math.h``, an
#: exact type match that wins outright -- qualifying the call is what reaches it. A math name with
#: no such overload (``sin``, ``cos``, ...) is deliberately absent: qualifying it would move the
#: same ambiguity one frame down, into ``dace::math``'s own template body.
#:
#: Shared because the SAME expression reaches C++ through two printers -- a tasklet body through
#: ``cppunparse``, a memlet subset or interstate assignment through ``dace.symbolic`` -- and a name
#: qualified by only one of them builds in one place and is ambiguous in the other. This table has
#: nothing to say about the standalone dialects, which resolve these names through
#: :data:`STD_RENAMES` before any of this applies.
RUNTIME_QUALIFIED_MATH: Dict[str, str] = {
    'fma': 'dace::math::fma',
    'sqrt': 'dace::math::sqrt',
    'exp': 'dace::math::exp',
    'log': 'dace::math::log',
}

#: Every runtime function this module knows about, in any lane.
KNOWN: Set[str] = (set(STD_RENAMES) | set(REWRITES) | set(INLINE_DEFINITIONS) | set(VARIADIC_MINMAX) | set(UNSUPPORTED))

#: ``dace::``-namespaced C++ type -> the standalone spelling. Most DaCe ctypes are already plain
#: (``float64`` is ``double``, ``int32`` is ``int32_t``), so only these few ever leak.
CTYPE_RENAMES: Dict[str, str] = {
    'dace::bool_': 'bool',
    'dace::uint': 'uint32_t',
    'dace::uint8': 'uint8_t',
    'dace::uint16': 'uint16_t',
    'dace::uint32': 'uint32_t',
    'dace::uint64': 'uint64_t',
    'dace::int8': 'int8_t',
    'dace::int16': 'int16_t',
    'dace::int32': 'int32_t',
    'dace::int64': 'int64_t',
    'dace::float32': 'float',
    'dace::float64': 'double',
    'dace::complex64': 'std::complex<float>',
    'dace::complex128': 'std::complex<double>',
}

#: Types with no portable standalone spelling. ``float16``/``bfloat16``/fp8 exist in DaCe only as
#: the CUDA and ROCm vendor types, or as an emulation the runtime headers carry -- neither is
#: reachable from a translation unit that includes nothing but the standard library.
UNSUPPORTED_CTYPES: Dict[str, str] = {
    'dace::float16': 'no portable C++ half type; use float32 or keep the DaCe runtime',
    'dace::bfloat16': 'no portable C++ bfloat16 type; use float32 or keep the DaCe runtime',
    'dace::float8_e4m3fn': 'no portable C++ fp8 type',
    'dace::float8_e5m2': 'no portable C++ fp8 type',
}


def ctype_for(ctype: str, dialect: Optional[Dialect] = None) -> str:
    """The standalone spelling of a type name.

    :param ctype: the type as the ordinary generators spell it.
    :param dialect: which standalone dialect to spell it for; ambient when omitted.
    :returns: the standalone spelling, or ``ctype`` unchanged when it is already plain.
    :raises NotImplementedError: if the type has no standalone spelling at all.
    """
    if ctype in UNSUPPORTED_CTYPES:
        raise NotImplementedError(f'CPF cannot emit the type {ctype!r}: {UNSUPPORTED_CTYPES[ctype]}.')
    return tables_for(dialect).ctype_renames.get(ctype, ctype)


def variadic_minmax(name: str, arguments: Tuple[str, ...], dialect: Optional[Dialect] = None) -> Optional[str]:
    """Spell a variadic ``Max``/``Min`` for ``dialect``.

    The runtime's ``Max`` takes any number of arguments, and so does the C++ dialect's own
    ``cpf_max`` template, so that one is called with the arguments as they stand. C has no
    variadic macro to fold over, so the C dialect NESTS the binary macro instead -- left to right,
    which is the association the recursive template has too.

    :param name: the runtime function name.
    :param arguments: already-printed argument expressions.
    :param dialect: which standalone dialect to spell it for; ambient when omitted.
    :returns: the expression, or ``None`` if ``name`` is not a min/max.
    """
    resolved = dialect if dialect is not None else _active_dialect
    target = tables_for(resolved).variadic_minmax.get(name)
    if target is None:
        return None
    if len(arguments) == 1:
        return '(%s)' % arguments[0]
    if len(arguments) == 2:
        return '%s(%s, %s)' % (target, arguments[0], arguments[1])
    if resolved is Dialect.STANDALONE_C:
        nested = arguments[0]
        for argument in arguments[1:]:
            nested = '%s(%s, %s)' % (target, nested, argument)
        return nested
    return '%s(%s)' % (target, ', '.join(arguments))


def needs_definition(name: str, dialect: Optional[Dialect] = None) -> bool:
    """Whether CPF calls ``name`` unchanged and emits a definition for it."""
    return name in tables_for(dialect).inline_definitions


def lowering_for(name: str, arguments: Tuple[str, ...], dialect: Optional[Dialect] = None) -> Optional[str]:
    """The CPF spelling of a call to ``name`` with ``arguments`` already printed.

    :param name: the runtime function name as the ordinary generators would emit it.
    :param arguments: already-printed argument expressions.
    :returns: the C++ expression, or ``None`` if ``name`` needs no rewriting -- either it is not a
              runtime function at all, or it is one CPF emits a definition for and calls unchanged
              (:func:`needs_definition` separates those two).
    :raises ValueError: if ``name`` is a known rewrite but the argument count does not match, which
                        means the caller and this table disagree about the function's shape.
    :raises NotImplementedError: if ``name`` is a runtime function CPF cannot express (see
                                 :data:`UNSUPPORTED`).
    """
    tables = tables_for(dialect)
    if name in tables.unsupported:
        raise NotImplementedError(f'CPF cannot lower {name!r}: {tables.unsupported[name]}.')
    variadic = variadic_minmax(name, arguments, dialect)
    if variadic is not None:
        return variadic
    if name in tables.rewrites:
        arity, template = tables.rewrites[name]
        if len(arguments) != arity:
            raise ValueError(f'CPF lowering of {name!r} expects {arity} arguments, got {len(arguments)}')
        return template.format(*arguments)
    if name in tables.std_renames:
        return '%s(%s)' % (tables.std_renames[name], ', '.join(arguments))
    return None


#: A ``dace::``-qualified name in hand-written C++ (a native tasklet body, a library expansion's
#: code), with the trailing identifier captured. Matches a type and a function alike -- the
#: distinction is made by which table the identifier is found in.
_QUALIFIED_NAME = re.compile(r'(?:::)?\bdace::(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)\b')


def rewrite_ctypes(code: str, dialect: Optional[Dialect] = None) -> str:
    """Spell every DaCe ctype in ``code`` the standalone way.

    Type names reach the emitted text from places no expression printer sees -- the entry
    signature, transient declarations, casts -- so this runs over the finished unit as well as over
    hand-written tasklet bodies. Purely a rename, from the same table
    (:data:`CTYPE_RENAMES` / :data:`C_CTYPE_RENAMES`) both callers share, so a container's type and
    a tasklet's cast cannot be spelled differently.

    :param code: emitted text, of any size.
    :param dialect: which standalone dialect emitted it; ambient when omitted.
    :returns: the text with DaCe type names replaced.
    :raises NotImplementedError: if the text names a type with no standalone spelling.
    """
    for qualified, reason in UNSUPPORTED_CTYPES.items():
        if re.search(r'(?:::)?\b%s\b' % re.escape(qualified), code):
            raise NotImplementedError(f'CPF cannot emit the type {qualified!r}: {reason}')
    for qualified, plain in tables_for(dialect).ctype_renames.items():
        code = re.sub(r'(?:::)?\b%s\b' % re.escape(qualified), plain, code)
    return code


def rewrite_native_code(code: str, dialect: Optional[Dialect] = None) -> str:
    """Rewrite the ``dace::`` names in a hand-written C++ body to their standalone spellings.

    Native tasklet bodies never reach the expression printers -- they are emitted verbatim -- so
    this is the only point at which a library expansion's own C++ can be re-spelled. Which is
    needed for the real cases: the ``Scan`` expansion calls ``::dace::scan::detail::scan_incl_sum``
    and the ``FindFirst`` expansion calls ``dace::find_first_index``, and CPF emits both functions
    itself rather than serializing a prefix sum or a cancelling search into a sequential loop.

    In C the same pass also rewrites the two call shapes C cannot express as a call at all -- the
    scan identities and the find-first over a lambda predicate (:func:`c_scan_identities`,
    :func:`c_find_first`) -- and re-spells the ``std::`` names a pass wrote directly
    (:func:`c_native_renames`), before the name table is consulted.

    Textual by necessity, and deliberately conservative: only the qualified name is rewritten, only
    when the identifier is one CPF knows, and never with knowledge of the arguments. A
    ``dace::`` name with no standalone spelling is LEFT ALONE, so it reaches
    ``dace.codegen.cpf.verify`` and is reported against the construct that emitted it -- a silent
    partial rewrite would be worse than none.

    :param code: the C++ body as the expansion wrote it.
    :param dialect: which standalone dialect to rewrite for; ambient when omitted.
    :returns: the body with the names CPF can spell rewritten.
    :raises NotImplementedError: if the body names a type or function this dialect cannot express.
    """
    tables = tables_for(dialect)
    code = rewrite_ctypes(code, dialect)

    if (dialect if dialect is not None else _active_dialect) is Dialect.STANDALONE_C:
        code = c_native_renames(c_find_first(c_scan_identities(code)))

    def replace(match: 're.Match') -> str:
        name = match.group(1)
        if name in tables.unsupported:
            raise NotImplementedError(f'CPF cannot lower {name!r}: {tables.unsupported[name]}.')
        if name in tables.inline_definitions:
            return name  # CPF emits this one's definition at the top of the unit
        if name in tables.std_renames:
            return tables.std_renames[name]
        return match.group(0)  # unknown: left for verify() to report

    code = _QUALIFIED_NAME.sub(replace, code)
    if (dialect if dialect is not None else _active_dialect) is Dialect.STANDALONE_C:
        code = c_cast_native_code(code)
    return code


def helpers_used(code: str, dialect: Optional[Dialect] = None) -> Set[str]:
    """Which inline-definition helpers ``code`` calls.

    Recovered from the finished text rather than accumulated while printing, for two reasons. The
    symbolic printer is reached through memoized entry points, so per-printer state does not
    survive a cache hit and would under-report. And a translation unit is written by several
    emitters -- memlet subsets, tasklet bodies, loop bounds -- so scanning the result is the only
    place that sees all of them at once.

    :param code: the emitted translation unit, or any fragment of it.
    :param dialect: which standalone dialect emitted it; ambient when omitted.
    :returns: the helper names called, for :func:`definitions_for`.
    """
    return {match.group(1) for match in tables_for(dialect).helper_call.finditer(code)}


def required_definitions(names: Set[str], dialect: Optional[Dialect] = None) -> Set[str]:
    """Close ``names`` over :data:`DEFINITION_DEPENDENCIES`.

    ``py_mod`` calls ``py_floor``, which calls ``int_floor_ni``: a unit that mentions only the
    first still has to carry all three.

    :param names: every function name the emitted code calls.
    :param dialect: which standalone dialect emitted them; ambient when omitted.
    :returns: every definition the unit needs, callers and callees alike.
    """
    tables = tables_for(dialect)
    needed: Set[str] = set()
    pending = [name for name in names if name in tables.inline_definitions]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(dependency for dependency in tables.definition_dependencies.get(name, ())
                       if dependency not in needed)
    return needed


def definitions_for(names: Set[str], dialect: Optional[Dialect] = None) -> Tuple[str, ...]:
    """The inline definitions a translation unit calling ``names`` has to carry, callees first.

    C++ needs a function declared before it is called, so the order is a topological one over
    :data:`DEFINITION_DEPENDENCIES` rather than alphabetical -- emitting ``py_mod`` before
    ``py_floor`` would not compile. Ties break on the name, so the same input always produces
    byte-identical output.

    :param names: every function name the emitted code calls.
    :param dialect: which standalone dialect emitted them; ambient when omitted.
    :returns: the definitions to place at the top of the translation unit.
    :raises ValueError: if the dependencies contain a cycle, which no valid ordering satisfies.
    """
    tables = tables_for(dialect)
    needed = required_definitions(names, dialect)
    emitted: list = []
    placed: Set[str] = set()
    while len(placed) < len(needed):
        ready = sorted(name for name in needed - placed
                       if all(dependency in placed for dependency in tables.definition_dependencies.get(name, ())
                              if dependency in needed))
        if not ready:
            raise ValueError(f'CPF inline definitions have a dependency cycle among {sorted(needed - placed)}')
        for name in ready:
            emitted.append(tables.inline_definitions[name])
            placed.add(name)
    return tuple(emitted)


def headers_for(names: Set[str], dialect: Optional[Dialect] = None) -> Tuple[str, ...]:
    """The system headers a translation unit calling ``names`` has to include, in a stable order.

    Closed over the dependencies too: a unit calling only ``py_mod`` still ends up with
    ``py_floor``'s body, and that is what needs ``<type_traits>``.

    :param names: every function name the emitted code calls.
    :param dialect: which standalone dialect emitted them; ambient when omitted.
    :returns: the include list, base headers first.
    """
    tables = tables_for(dialect)
    extra: Set[str] = set()
    for name in required_definitions(names, dialect) | set(names):
        extra.update(tables.definition_headers.get(name, ()))
    return tables.base_headers + tuple(sorted(extra - set(tables.base_headers)))


# ======================================================================================
# The C dialect
# ======================================================================================
#
# C23 has no templates, no function overloading and no ``constexpr`` on functions, and its maths
# library is not type-generic: ``sqrt(x)`` on a ``float`` promotes to ``double`` and rounds twice,
# which ``std::sqrt(float)`` does not. ``<tgmath.h>`` would fix the second problem and create a
# worse one -- its macros are named ``exp``, ``pow``, ``log``, ``round``, which is exactly the set
# of names a scientific SDFG gives its containers.
#
# So every generic operation becomes a ``_Generic`` dispatch macro over a closed set of typed
# ``static inline`` functions. The controlling expression of ``_Generic`` is UNEVALUATED, so each
# argument is still evaluated exactly once, in the selected call.

#: Arithmetic types a ``_Generic`` dispatch enumerates, paired with the suffix its typed helper is
#: named after. SIGNED integers only: an unsigned instantiation of a sign-sensitive body ("comparison
#: of unsigned expression < 0 is always false") warns under ``-Wextra``, and CPF output must build
#: warning-free. A helper reached with an unsigned value fails to select, which is the loud direction.
#:
#: The types are the FUNDAMENTAL spellings, not the ``<stdint.h>`` typedefs: ``int32_t`` IS ``int``
#: on every platform DaCe targets, so listing both would give one ``_Generic`` two associations for
#: the same type, which does not compile.
C_SIGNED_INTS: Tuple[Tuple[str, str], ...] = (('int', 'i'), ('long', 'l'), ('long long', 'll'))
C_FLOATS: Tuple[Tuple[str, str], ...] = (('float', 'f'), ('double', 'd'), ('long double', 'ld'))
C_COMPLEX: Tuple[Tuple[str, str],
                 ...] = (('float _Complex', 'fc'), ('double _Complex', 'dc'), ('long double _Complex', 'ldc'))
C_ARITHMETIC: Tuple[Tuple[str, str], ...] = C_SIGNED_INTS + C_FLOATS

#: Every type surviving the usual arithmetic conversions of ``(a) + (b)``, which is what
#: ``cpf_max`` / ``cpf_min`` dispatch on. Unsigned types belong HERE (the bodies compare two values
#: of one type and cannot warn), and the list is closed on purpose: no ``default:`` association, so
#: a type outside it is a compile error rather than a silent widening through ``double`` -- which
#: is how an int64 argument would lose its low bits.
C_MINMAX_TYPES: Tuple[Tuple[str, str],
                      ...] = (('int', 'i'), ('unsigned int', 'u'), ('long', 'l'), ('unsigned long', 'ul'),
                              ('long long', 'll'), ('unsigned long long',
                                                    'ull'), ('float', 'f'), ('double', 'd'), ('long double', 'ld'))


def c_generic_macro(name: str,
                    parameters: Tuple[str, ...],
                    control: str,
                    dispatch: Tuple[Tuple[str, str], ...],
                    call: Optional[Tuple[str, ...]] = None) -> str:
    """One ``_Generic`` dispatch macro.

    :param name: the macro's name -- the same name the printers already emit, so no call site moves.
    :param parameters: the macro parameters, which are also the typed functions' parameter names.
    :param control: the controlling expression, over ``parameters``. Never evaluated.
    :param dispatch: ``(type, target function)`` associations, in emission order. A ``'default'``
                     type is written as the ``default:`` association.
    :param call: what to pass to the selected function, defaulting to ``parameters`` unchanged. An
                 out-parameter is passed as ``&(name)``, which is why the caller may override it.
    :returns: the ``#define`` line.
    """
    associations = ', '.join('%s: %s' % (ctype, target) for ctype, target in dispatch)
    return '#define %s(%s) _Generic(%s, %s)(%s)' % (name, ', '.join(parameters), control, associations,
                                                    ', '.join(call if call is not None else parameters))


def c_typed_family(name: str,
                   parameters: Tuple[Tuple[str, str], ...],
                   groups: Tuple[Tuple[Tuple[Tuple[str, str], ...], str, str], ...],
                   control: str,
                   call: Optional[Tuple[str, ...]] = None) -> str:
    """A helper as C: one ``static inline`` per type, plus the ``_Generic`` macro that selects it.

    An unused ``static inline`` warns under neither ``-Wall`` nor ``-Wextra``, so the whole typed
    set is emitted whenever the helper is used at all -- which is what lets one macro serve every
    width the index arithmetic or the element type settled on.

    :param name: the helper's name, as the printers emit it. Becomes the macro's name.
    :param parameters: ``(type template, name)`` per parameter. ``{T}`` is the group's type.
    :param groups: ``(types, return type template, body)``. Several groups exist where C++ used
                   ``if constexpr`` to branch on integral-vs-floating: the branch becomes two
                   groups, and ``_Generic`` picks between them.
    :param control: the ``_Generic`` controlling expression, over the parameter names.
    :param call: what to pass to the selected function (see :func:`c_generic_macro`).
    :returns: the definitions and the macro, as one block.
    """
    blocks = []
    dispatch = []
    # A family whose own name already carries the prefix (``cpf_max``) must not get it twice.
    stem = name[4:] if name.startswith('cpf_') else name
    for types, returns, body in groups:
        for ctype, suffix in types:
            target = 'cpf_%s_%s' % (stem, suffix)
            declared = ', '.join(ptype.replace('{T}', ctype) + ' ' + pname for ptype, pname in parameters)
            statements = '\n'.join('    ' + line if line.strip() else line
                                   for line in body.replace('{T}', ctype).split('\n'))
            blocks.append('static inline %s %s(%s) {\n%s\n}' %
                          (returns.replace('{T}', ctype), target, declared, statements))
            dispatch.append((ctype, target))
    blocks.append(c_generic_macro(name, tuple(pname for _, pname in parameters), control, tuple(dispatch), call))
    return '\n'.join(blocks)


#: ``(runtime name, C base name, family, arity)`` for every :data:`STD_RENAMES` entry that has a C
#: counterpart. The family names the suffix set ``_Generic`` picks between:
#:
#: ``real``
#:     ``<base>f`` for ``float``, ``<base>l`` for ``long double``, ``<base>`` otherwise -- which is
#:     also what an integer argument gets, matching ``std::sqrt(int) -> double``.
#: ``abs``
#:     the integer, floating and complex absolute values, which C spells with five different names.
#: ``complex``
#:     ``conjf`` / ``conj`` / ``conjl``.
#:
#: Arity 1 dispatches on ``+(a0)``; the unary plus applies the integer promotions, so a ``short`` or
#: an ``int8_t`` selects the ``int`` association instead of failing to select. Arity 2 and 3
#: dispatch on the SUM of the arguments, which is the type the call would convert them to anyway --
#: except ``frexp`` and ``ldexp``, whose second argument is an ``int`` exponent and would drag the
#: dispatch to the wrong type, so they dispatch on the first argument alone (``first`` arity 2).
C_MATH_SPEC: Tuple[Tuple[str, str, str, object], ...] = (
    ('Abs', 'abs', 'abs', 1),
    ('abs', 'abs', 'abs', 1),
    ('ceiling', 'ceil', 'real', 1),
    ('ceil', 'ceil', 'real', 1),
    ('floor', 'floor', 'real', 1),
    ('ROUND', 'round', 'real', 1),
    ('round', 'round', 'real', 1),
    ('conj', 'conj', 'complex', 1),
    ('exp2', 'exp2', 'real', 1),
    ('expm1', 'expm1', 'real', 1),
    ('log1p', 'log1p', 'real', 1),
    ('log2', 'log2', 'real', 1),
    ('frexp', 'frexp', 'real', 'first2'),
    ('ldexp', 'ldexp', 'real', 'first2'),
    ('ilogb', 'ilogb', 'real', 1),
    ('sin', 'sin', 'real', 1),
    ('cos', 'cos', 'real', 1),
    ('tan', 'tan', 'real', 1),
    ('asin', 'asin', 'real', 1),
    ('acos', 'acos', 'real', 1),
    ('atan', 'atan', 'real', 1),
    ('atan2', 'atan2', 'real', 2),
    ('sinh', 'sinh', 'real', 1),
    ('cosh', 'cosh', 'real', 1),
    ('tanh', 'tanh', 'real', 1),
    ('exp', 'exp', 'real', 1),
    ('fabs', 'fabs', 'real', 1),
    ('log', 'log', 'real', 1),
    ('log10', 'log10', 'real', 1),
    ('sqrt', 'sqrt', 'real', 1),
    ('cbrt', 'cbrt', 'real', 1),
    ('pow', 'pow', 'real', 2),
    ('fma', 'fma', 'real', 3),
    ('erf', 'erf', 'real', 1),
    ('erfc', 'erfc', 'real', 1),
    ('tgamma', 'tgamma', 'real', 1),
    ('lgamma', 'lgamma', 'real', 1),
    ('trunc', 'trunc', 'real', 1),
    ('hypot', 'hypot', 'real', 2),
)

#: Maths CPF emits for its OWN definitions rather than for a runtime rename: ``cpp_mod`` needs
#: ``fmod``, ``np_modf`` needs ``modf``, and the complex ``sign_numpy_2`` needs the component
#: accessors that :data:`REWRITES` spells ``.real()`` / ``.imag()`` in C++.
C_INTERNAL_MATH_SPEC: Tuple[Tuple[str, str, str, object], ...] = (
    ('fmod', 'fmod', 'real', 2),
    ('modf', 'modf', 'real', 'first2'),
    ('creal', 'creal', 'complex_component', 1),
    ('cimag', 'cimag', 'complex_component', 1),
)

#: Runtime maths C already spells type-generically, as a ``<math.h>`` MACRO. Wrapping these in an
#: ``cpf_`` dispatch would be wrong as well as pointless: there is no ``isnanf`` to dispatch TO.
C_TYPE_GENERIC_MATH: Dict[str, str] = {
    'isfinite': 'isfinite',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'signbit': 'signbit',
}

_C_FAMILY_DISPATCH: Dict[str, Tuple[Tuple[str, str], ...]] = {
    'real': (('float', '{base}f'), ('long double', '{base}l'), ('default', '{base}')),
    'abs':
    (('int', 'abs'), ('long', 'labs'), ('long long', 'llabs'), ('float', 'fabsf'), ('long double', 'fabsl'),
     ('float _Complex', 'cabsf'), ('double _Complex', 'cabs'), ('long double _Complex', 'cabsl'), ('default', 'fabs')),
    'complex': (('float _Complex', '{base}f'), ('long double _Complex', '{base}l'), ('default', '{base}')),
    # A real argument has no imaginary part to read, and C's ``creal``/``cimag`` accept one, so the
    # default association keeps working for a complex-valued expression that folded to a real type.
    'complex_component': (('float _Complex', '{base}f'), ('long double _Complex', '{base}l'), ('default', '{base}')),
}


def c_math_macro(base: str, family: str, arity) -> Tuple[str, str]:
    """``(macro name, #define line)`` for one C maths dispatch.

    :param base: the C function's base name (``sqrt``, ``pow``).
    :param family: which suffix set to dispatch over -- see :data:`C_MATH_SPEC`.
    :param arity: the argument count, or ``'first2'`` for a two-argument call whose dispatch is
                  decided by the first argument alone.
    :returns: the macro's name and its definition.
    """
    count = 2 if arity == 'first2' else arity
    parameters = tuple('a%d' % index for index in range(count))
    control = '+(a0)' if (count == 1 or arity == 'first2') else ' + '.join('(%s)' % p for p in parameters)
    dispatch = tuple((ctype, target.replace('{base}', base)) for ctype, target in _C_FAMILY_DISPATCH[family])
    name = 'cpf_' + base
    return name, c_generic_macro(name, parameters, control, dispatch)


#: Runtime function -> its C spelling, for the names C has under a different name.
C_STD_RENAMES: Dict[str, str] = dict(C_TYPE_GENERIC_MATH)

#: C macro name -> its ``#define``. Merged into :data:`C_INLINE_DEFINITIONS` below, so the same
#: use-scan and dependency ordering that places an inline definition places a macro.
_C_MATH_MACROS: Dict[str, str] = {}
for _runtime_name, _base, _family, _arity in C_MATH_SPEC:
    _macro, _definition = c_math_macro(_base, _family, _arity)
    C_STD_RENAMES[_runtime_name] = _macro
    _C_MATH_MACROS[_macro] = _definition
for _runtime_name, _base, _family, _arity in C_INTERNAL_MATH_SPEC:
    _macro, _definition = c_math_macro(_base, _family, _arity)
    _C_MATH_MACROS[_macro] = _definition

#: ``Max``/``Min`` in C. Not the ``<stdlib.h>`` integer ``max``, which does not exist: CPF emits its
#: own typed pair (see :data:`C_MINMAX_TYPES`).
C_VARIADIC_MINMAX: Dict[str, str] = {'Max': 'cpf_max', 'Min': 'cpf_min', 'max': 'cpf_max', 'min': 'cpf_min'}

#: Rewrites that differ from :data:`REWRITES` because their C++ form names a C++ construct: a
#: member call on ``std::complex``, or a ``static_cast``.
C_REWRITES: Dict[str, Tuple[int, str]] = dict(REWRITES)
C_REWRITES.update({
    're': (1, '(cpf_creal({0}))'),
    'im': (1, '(cpf_cimag({0}))'),
    'iround': (1, '((int)cpf_round({0}))'),
    'np_float_pow': (2, '(cpf_pow((double)({0}), (double)({1})))'),
})

#: ``dace::``-namespaced C++ type -> its C spelling. Only the two complex types differ from the C++
#: dialect's table: C spells a complex ``float _Complex`` rather than ``std::complex<float>``.
C_CTYPE_RENAMES: Dict[str, str] = dict(CTYPE_RENAMES)
C_CTYPE_RENAMES.update({
    'dace::complex64': 'float _Complex',
    'dace::complex128': 'double _Complex',
    'std::complex<float>': 'float _Complex',
    'std::complex<double>': 'double _Complex',
})

#: ``b`` wins only by comparing STRICTLY better, so a tie -- and a comparison false because an
#: operand is NaN -- keeps ``a``. Same rule as the runtime's ``max``/``min``, which is what these
#: stand in for.
_C_MINMAX_DEFINITIONS: Dict[str, str] = {
    name:
    c_typed_family(name, (('{T}', 'a'), ('{T}', 'b')), ((C_MINMAX_TYPES, '{T}', 'return (%s) ? b : a;' % condition), ),
                   '(a) + (b)')
    for name, condition in (('cpf_max', 'a < b'), ('cpf_min', 'b < a'))
}

_C_SIGN_BODY = 'return ({T})((({T})0 < value) - (value < ({T})0));'

#: The C form of every :data:`INLINE_DEFINITIONS` entry, plus the two ``<numeric>`` functions C has
#: no counterpart for at all (``gcd`` / ``lcm``, which are a rename in C++ and a definition here).
#:
#: The eight prefix scans keep their ``#pragma omp simd reduction(inscan, ...)`` bodies verbatim.
#: That form IS the parallel scan; a rendering that quietly serialized every prefix sum would not be
#: a canonical parallel form.
#:
#: The four out-parameter helpers took C++ references. Their C macros take the same LVALUES the
#: printers already pass and apply ``&`` themselves, so no call site changes shape.
C_INLINE_DEFINITIONS: Dict[str, str] = dict(_C_MATH_MACROS)
C_INLINE_DEFINITIONS.update(_C_MINMAX_DEFINITIONS)
C_INLINE_DEFINITIONS.update({
    'sign':
    c_typed_family('sign', (('{T}', 'value'), ), ((C_ARITHMETIC, '{T}', _C_SIGN_BODY), ), '+(value)'),
    'sgn':
    c_typed_family('sgn', (('{T}', 'value'), ), ((C_ARITHMETIC, '{T}', _C_SIGN_BODY), ), '+(value)'),
    'sign_numpy_2':
    c_typed_family(
        'sign_numpy_2', (('{T}', 'value'), ),
        ((C_ARITHMETIC, '{T}', _C_SIGN_BODY),
         (C_COMPLEX, '{T}', 'return (cpf_creal(value) != 0 && cpf_cimag(value) != 0) ? value / cpf_abs(value) : 0;')),
        '+(value)'),
    # Two arities, which no single C macro can have. The three-argument pick chooses between the
    # unary and binary dispatch macros by counting what the caller wrote.
    'heaviside':
    '\n'.join((
        '#define cpf_pick3(a0, a1, a2, ...) a2',
        c_typed_family('cpf_heaviside_1', (('{T}', 'value'), ),
                       ((C_ARITHMETIC, '{T}', 'return (value > ({T})0) ? ({T})1 : ({T})0;'), ), '+(value)'),
        c_typed_family(
            'cpf_heaviside_2', (('{T}', 'value'), ('{T}', 'at_zero')),
            ((C_ARITHMETIC, '{T}', 'return (value < ({T})0) ? ({T})0 : ((value > ({T})0) ? ({T})1 : at_zero);'), ),
            '(value) + (at_zero)'),
        '#define heaviside(...) cpf_pick3(__VA_ARGS__, cpf_heaviside_2, cpf_heaviside_1)(__VA_ARGS__)',
    )),
    # Integral input is already floored, so it comes back unchanged -- narrowing an int64 through
    # ``(int)floor(...)`` would truncate it to 32 bits.
    'ifloor':
    c_typed_family('ifloor', (('{T}', 'value'), ),
                   ((C_SIGNED_INTS, '{T}', 'return value;'), (C_FLOATS, 'int', 'return (int)cpf_floor(value);')),
                   '+(value)'),
    'int_ceil':
    c_typed_family('int_ceil', (('{T}', 'numerator'), ('{T}', 'denominator')),
                   ((C_SIGNED_INTS, '{T}', 'return (numerator + denominator - 1) / denominator;'), ),
                   '(numerator) + (denominator)'),
    'int_floor_ni':
    c_typed_family('int_floor_ni', (('{T}', 'numerator'), ('{T}', 'denominator')),
                   ((C_SIGNED_INTS, '{T}', '{T} quotient = numerator / denominator;\n'
                     '{T} remainder = numerator % denominator;\n'
                     'return quotient - ((remainder != 0) && ((remainder < 0) != (denominator < 0)));'), ),
                   '(numerator) + (denominator)'),
    'py_floor':
    c_typed_family('py_floor', (('{T}', 'numerator'), ('{T}', 'denominator')),
                   ((C_SIGNED_INTS, '{T}', 'return int_floor_ni(numerator, denominator);'),
                    (C_FLOATS, '{T}', 'return cpf_floor(numerator / denominator);')), '(numerator) + (denominator)'),
    'py_mod':
    c_typed_family('py_mod', (('{T}', 'numerator'), ('{T}', 'denominator')),
                   ((C_ARITHMETIC, '{T}', 'return numerator - py_floor(numerator, denominator) * denominator;'), ),
                   '(numerator) + (denominator)'),
    'floor_mod':
    c_typed_family('floor_mod', (('{T}', 'numerator'), ('{T}', 'denominator')),
                   ((C_ARITHMETIC, '{T}', 'return py_mod(numerator, denominator);'), ), '(numerator) + (denominator)'),
    'mod':
    c_typed_family('mod', (('{T}', 'value'), ('{T}', 'modulus')),
                   ((C_SIGNED_INTS, '{T}', 'return ((value % modulus) + modulus) % modulus;'), ),
                   '(value) + (modulus)'),
    'cpp_mod':
    c_typed_family('cpp_mod', (('{T}', 'numerator'), ('{T}', 'denominator')),
                   ((C_SIGNED_INTS, '{T}', 'return numerator % denominator;'),
                    (C_FLOATS, '{T}', 'return cpf_fmod(numerator, denominator);')), '(numerator) + (denominator)'),
    'Mod_float':
    c_typed_family('Mod_float', (('{T}', 'value'), ('{T}', 'modulus')),
                   ((C_FLOATS, '{T}', 'return value - (int)(value / modulus) * modulus;'), ), '(value) + (modulus)'),
    'Modulo':
    c_typed_family(
        'Modulo', (('{T}', 'value'), ('{T}', 'modulus')),
        ((C_ARITHMETIC, '{T}', 'return value - ({T})cpf_floor((double)(value) / (double)(modulus)) * modulus;'), ),
        '(value) + (modulus)'),
    'Modulo_float':
    c_typed_family('Modulo_float', (('{T}', 'value'), ('{T}', 'modulus')),
                   ((C_FLOATS, '{T}', 'return value - ({T})cpf_floor(value / modulus) * modulus;'), ),
                   '(value) + (modulus)'),
    'cpp_divmod':
    c_typed_family('cpp_divmod',
                   (('{T}', 'numerator'), ('{T}', 'denominator'), ('{T} *', 'quotient'), ('{T} *', 'remainder')),
                   ((C_SIGNED_INTS, 'void', '*quotient = ({T})(numerator / denominator);\n'
                     '*remainder = ({T})(numerator % denominator);'), ), '(numerator) + (denominator)',
                   ('numerator', 'denominator', '&(quotient)', '&(remainder)')),
    'py_divmod':
    c_typed_family('py_divmod',
                   (('{T}', 'numerator'), ('{T}', 'denominator'), ('{T} *', 'quotient'), ('{T} *', 'remainder')),
                   ((C_SIGNED_INTS, 'void', '{T} correction;\n'
                     'cpp_divmod(numerator, denominator, *quotient, *remainder);\n'
                     'correction = (*remainder != 0 && ((*remainder < 0) != (denominator < 0)));\n'
                     '*quotient -= correction;\n'
                     '*remainder += correction * denominator;'), ), '(numerator) + (denominator)',
                   ('numerator', 'denominator', '&(quotient)', '&(remainder)')),
    'np_modf':
    c_typed_family('np_modf', (('{T}', 'value'), ('{T} *', 'integral'), ('{T} *', 'fractional')),
                   ((C_SIGNED_INTS, 'void', '*integral = value;\n*fractional = 0;'),
                    (C_FLOATS, 'void', '*fractional = cpf_modf(value, integral);')), '+(value)',
                   ('value', '&(integral)', '&(fractional)')),
    'np_frexp':
    c_typed_family('np_frexp', (('{T}', 'value'), ('{T} *', 'mantissa'), ('int *', 'exponent')),
                   ((C_FLOATS, 'void', '*mantissa = cpf_frexp(value, exponent);'), ), '+(value)',
                   ('value', '&(mantissa)', '&(exponent)')),
    'ipow':
    c_typed_family('ipow', (('{T}', 'base'), ('long long', 'exponent')), ((C_ARITHMETIC, '{T}', '{T} result = 1;\n'
                                                                           'while (exponent > 0) {\n'
                                                                           '    if (exponent & 1) { result *= base; }\n'
                                                                           '    base *= base;\n'
                                                                           '    exponent >>= 1;\n'
                                                                           '}\n'
                                                                           'return result;'), ), '+(base)'),
    'logical_left_shift':
    c_typed_family(
        'logical_left_shift', (('{T}', 'value'), ('int', 'amount')),
        tuple((((ctype, suffix), ), '{T}', 'return ({T})((%s)(value) << amount);' % unsigned)
              for ctype, suffix, unsigned in (('int', 'i', 'unsigned int'), ('long', 'l', 'unsigned long'),
                                              ('long long', 'll', 'unsigned long long'))), '+(value)'),
    'logical_right_shift':
    c_typed_family(
        'logical_right_shift', (('{T}', 'value'), ('int', 'amount')),
        tuple((((ctype, suffix), ), '{T}', 'return ({T})((%s)(value) >> amount);' % unsigned)
              for ctype, suffix, unsigned in (('int', 'i', 'unsigned int'), ('long', 'l', 'unsigned long'),
                                              ('long long', 'll', 'unsigned long long'))), '+(value)'),
    'gcd':
    c_typed_family('gcd', (('{T}', 'a'), ('{T}', 'b')), ((C_SIGNED_INTS, '{T}', '{T} x = a < 0 ? -a : a;\n'
                                                          '{T} y = b < 0 ? -b : b;\n'
                                                          'while (y != 0) {\n'
                                                          '    {T} t = x % y;\n'
                                                          '    x = y;\n'
                                                          '    y = t;\n'
                                                          '}\n'
                                                          'return x;'), ), '(a) + (b)'),
    'lcm':
    c_typed_family('lcm', (('{T}', 'a'), ('{T}', 'b')), ((C_SIGNED_INTS, '{T}', '{T} divisor = gcd(a, b);\n'
                                                          '{T} product;\n'
                                                          'if (divisor == 0) { return 0; }\n'
                                                          'product = (a / divisor) * b;\n'
                                                          'return product < 0 ? -product : product;'), ), '(a) + (b)'),
})


def _c_scan_family(kind: str, operation: str, clause: str, step: str) -> str:
    """One prefix-scan helper as C: a statement macro over ``typeof``, not a typed function set.

    A scan touches THREE independent types -- input element, output element, accumulator -- which
    is why the C++ form is a template over ``<It, OutIt, T>``. A ``_Generic`` family cannot say
    that: it dispatches on one operand and then declares the other two at whatever type it picked,
    so the compaction prefix sums exist for -- an ``int8_t`` 0/1 mask scanned into ``int64_t``
    ranks -- fails to select. Saying it with a cross product costs 36 functions per family, and
    CPF output is meant to be read. ``typeof`` gives the same three degrees of freedom, at
    the price of being a statement rather than a call -- the trade ``cpf_find_first`` already makes,
    for the same reason.

    The accumulator is ``typeof_unqual(seed)`` and never the input's: a 0/1 mask folded at ``int8_t``
    wraps at 128, and the seed is the one argument that names the type the caller wants the fold
    carried out in. Every argument is bound to a local before the loop, so each is evaluated
    exactly once even though the loop names it on every iteration.

    :param kind: ``'inclusive'`` or ``'exclusive'``, as the OpenMP ``scan`` clause spells it.
    :param operation: the fold's name, which the helper is named after.
    :param clause: the OpenMP reduction identifier for ``operation``.
    :param step: the accumulator's update, written against the macro's own locals.
    :returns: the ``#define``.
    """
    update = '            cpf_scan_acc = %s;' % step
    store = '            cpf_scan_out[cpf_scan_i] = cpf_scan_acc;'
    # Input phase, directive, scan phase -- and ``exclusive`` names them in the other order.
    phases = (update, store) if kind == 'inclusive' else (store, update)
    return '\\\n'.join((
        '#define scan_%s_%s(f, o, lo, hi, seed) ' % ('incl' if kind == 'inclusive' else 'excl', operation),
        '    do {',
        '        typeof(*(f)) * cpf_scan_in = (f);',
        '        typeof(*(o)) * cpf_scan_out = (o);',
        '        const long cpf_scan_lo = (lo);',
        '        const long cpf_scan_hi = (hi);',
        # ``typeof_unqual``, not ``typeof``: the seed is normally a read-only scalar the backend
        # already emitted as ``const double _scan_seed_b = ...``, and ``typeof`` keeps that
        # qualifier, so the accumulator comes out const -- the fold cannot assign it and OpenMP
        # refuses it outright ("may appear only in shared or firstprivate clauses"). The INPUT
        # binding deliberately keeps its qualifiers; only the accumulator is written.
        '        typeof_unqual(seed) cpf_scan_acc = (seed);',
        '        _Pragma("omp simd reduction(inscan, %s:cpf_scan_acc)")' % clause,
        '        for (long cpf_scan_i = cpf_scan_lo; cpf_scan_i < cpf_scan_hi; ++cpf_scan_i) {',
        phases[0],
        '            _Pragma("omp scan %s(cpf_scan_acc)")' % kind,
        phases[1],
        '        }',
        '    } while (0)',
    ))


#: ``(operation, OpenMP reduction identifier, accumulator update)``. The update is written against
#: the macro's own locals, and ``min`` / ``max`` cast the input to the accumulator's type first so
#: the comparison happens where the fold does -- the ``static_cast<T>`` the C++ templates write.
_C_SCAN_STEPS: Tuple[Tuple[str, str, str], ...] = (
    ('sum', '+', 'cpf_scan_acc + cpf_scan_in[cpf_scan_i]'),
    ('product', '*', 'cpf_scan_acc * cpf_scan_in[cpf_scan_i]'),
    ('min', 'min', 'cpf_min(cpf_scan_acc, (typeof(cpf_scan_acc))cpf_scan_in[cpf_scan_i])'),
    ('max', 'max', 'cpf_max(cpf_scan_acc, (typeof(cpf_scan_acc))cpf_scan_in[cpf_scan_i])'),
)

for _kind in ('inclusive', 'exclusive'):
    for _operation, _clause, _step in _C_SCAN_STEPS:
        C_INLINE_DEFINITIONS['scan_%s_%s' % ('incl' if _kind == 'inclusive' else 'excl', _operation)] = _c_scan_family(
            _kind, _operation, _clause, _step)

#: The chunk sizer, identical to the C++ one: it is already a single concrete type, so it needs no
#: ``_Generic`` dispatch and is a plain function in both dialects.
C_INLINE_DEFINITIONS['find_first_chunk'] = (
    '#ifdef _OPENMP\n'
    '#include <omp.h>\n'
    '#endif\n'
    'static inline long long find_first_chunk(long long span, bool parallel) {\n'
    '    const double chunk_scale = 8.0;\n'
    '    const long long chunks_per_thread = 4;\n'
    '    long long chunk = (long long)(chunk_scale * sqrt((double)span));\n'
    '    long long threads = 1;\n'
    '    long long ceiling;\n'
    '#ifdef _OPENMP\n'
    '    if (parallel) threads = (long long)omp_get_max_threads();\n'
    '#endif\n'
    '    ceiling = span / (chunks_per_thread * threads);\n'
    '    if (ceiling < 1) ceiling = 1;\n'
    '    if (chunk > ceiling) chunk = ceiling;\n'
    '    if (chunk < 1) chunk = 1;\n'
    '    return chunk;\n'
    '}')

#: The search itself, which is where the two dialects genuinely part. The C++ form takes the
#: predicate as a lambda; C has none, so the predicate arrives as a macro ARGUMENT and is pasted
#: into the innermost loop, with the search's own index bound to the name the expansion wrote its
#: subscripts against (:func:`c_find_first` supplies both). That makes it a statement macro rather
#: than an expression: the assignment target is the first argument, because a C expression cannot
#: contain the loop this needs. ``_Pragma`` rather than ``#pragma`` for the same reason -- a
#: directive cannot be produced by a macro expansion.
C_INLINE_DEFINITIONS['cpf_find_first'] = '\\\n'.join((
    '#define cpf_find_first(out, ff_begin, ff_end, ff_index, ff_parallel, ff_pred) ',
    '    do {',
    '        const long long cpf_ff_lo = (ff_begin);',
    '        const long long cpf_ff_end = (ff_end);',
    '        const bool cpf_ff_par = (ff_parallel);',
    # The block is the early-exit granularity: a vectorized loop cannot break.
    '        const long long cpf_ff_simd = 64;',
    '        long long cpf_ff_best = cpf_ff_end;',
    '        if (cpf_ff_lo < cpf_ff_end) {',
    '            const long long cpf_ff_span = cpf_ff_end - cpf_ff_lo;',
    '            const long long cpf_ff_chunk = find_first_chunk(cpf_ff_span, cpf_ff_par);',
    '            const long long cpf_ff_chunks = (cpf_ff_span + cpf_ff_chunk - 1) / cpf_ff_chunk;',
    '            long long cpf_ff_hint = cpf_ff_end;',
    '            _Pragma("omp parallel for schedule(dynamic, 1) if (parallel : cpf_ff_par) '
    'reduction(min : cpf_ff_best)")',
    '            for (long long cpf_ff_c = 0; cpf_ff_c < cpf_ff_chunks; ++cpf_ff_c) {',
    '                long long cpf_ff_seen, cpf_ff_hi, cpf_ff_found, cpf_ff_b;',
    '                const long long cpf_ff_from = cpf_ff_lo + cpf_ff_c * cpf_ff_chunk;',
    '                _Pragma("omp atomic read")',
    '                cpf_ff_seen = cpf_ff_hint;',
    '                if (cpf_ff_from >= cpf_ff_seen) continue;',
    '                cpf_ff_hi = cpf_ff_from + cpf_ff_chunk;',
    '                if (cpf_ff_hi > cpf_ff_end) cpf_ff_hi = cpf_ff_end;',
    '                if (cpf_ff_hi > cpf_ff_seen) cpf_ff_hi = cpf_ff_seen;',
    '                cpf_ff_found = cpf_ff_end;',
    '                for (cpf_ff_b = cpf_ff_from; cpf_ff_b < cpf_ff_hi; cpf_ff_b += cpf_ff_simd) {',
    '                    long long cpf_ff_block = cpf_ff_end;',
    '                    long long cpf_ff_to = cpf_ff_b + cpf_ff_simd;',
    '                    if (cpf_ff_to > cpf_ff_hi) cpf_ff_to = cpf_ff_hi;',
    '                    _Pragma("omp simd reduction(min : cpf_ff_block)")',
    '                    for (long long ff_index = cpf_ff_b; ff_index < cpf_ff_to; ++ff_index) {',
    '                        const long long cpf_ff_v = (ff_pred) ? ff_index : cpf_ff_end;',
    '                        cpf_ff_block = cpf_ff_v < cpf_ff_block ? cpf_ff_v : cpf_ff_block;',
    '                    }',
    '                    if (cpf_ff_block < cpf_ff_end) { cpf_ff_found = cpf_ff_block; break; }',
    '                }',
    '                if (cpf_ff_found < cpf_ff_end) {',
    '                    long long cpf_ff_cur;',
    '                    if (cpf_ff_found < cpf_ff_best) cpf_ff_best = cpf_ff_found;',
    '                    _Pragma("omp atomic read")',
    '                    cpf_ff_cur = cpf_ff_hint;',
    '                    if (cpf_ff_found < cpf_ff_cur) {',
    '                        _Pragma("omp atomic write")',
    '                        cpf_ff_hint = cpf_ff_found;',
    '                    }',
    '                }',
    '            }',
    '        }',
    '        (out) = cpf_ff_best;',
    '    } while (0)',
))

#: Definitions each C definition calls -- macros included, since a macro must be ``#define``d before
#: the function body that expands it is compiled.
C_DEFINITION_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    'sign_numpy_2': ('cpf_creal', 'cpf_cimag', 'cpf_abs'),
    'ifloor': ('cpf_floor', ),
    'py_floor': ('int_floor_ni', 'cpf_floor'),
    'py_mod': ('py_floor', ),
    'floor_mod': ('py_mod', ),
    'cpp_mod': ('cpf_fmod', ),
    'Modulo': ('cpf_floor', ),
    'Modulo_float': ('cpf_floor', ),
    'py_divmod': ('cpp_divmod', ),
    'np_modf': ('cpf_modf', ),
    'np_frexp': ('cpf_frexp', ),
    'lcm': ('gcd', ),
    'scan_incl_min': ('cpf_min', ),
    'scan_incl_max': ('cpf_max', ),
    'scan_excl_min': ('cpf_min', ),
    'scan_excl_max': ('cpf_max', ),
    'cpf_find_first': ('find_first_chunk', ),
}

#: What the C dialect refuses, and why. Empty: every construct CPF reaches has a C spelling.
C_UNSUPPORTED: Dict[str, str] = {}

#: Helpers C answers with a REWRITE of the CALL SITE rather than a definition or a refusal -- a
#: third lane, and the only one, so the anti-rot tests can still insist every C++ helper is
#: accounted for. Each is a shape C cannot spell as a callable at all: the scan's neutral elements
#: need the element type, which only the call site names (:func:`c_scan_identities`), and the
#: find-first takes a predicate, which in C++ is a lambda and in C has to be pasted into the search
#: as a macro argument (:func:`c_find_first`).
C_REWRITTEN_IN_NATIVE_CODE: FrozenSet[str] = frozenset({'min_identity', 'max_identity', 'find_first_index'})

#: Headers CPF's C output always includes. ``<stdbool.h>`` is deliberately absent: ``bool`` /
#: ``true`` / ``false`` are C23 keywords. ``<tgmath.h>`` is deliberately absent too -- see the
#: section header above.
C_BASE_HEADERS: Tuple[str, ...] = ('<stdint.h>', '<math.h>', '<limits.h>', '<stdlib.h>', '<string.h>', '<assert.h>',
                                   '<complex.h>')

#: ``<complex.h>`` defines ``I``, and ``I`` is a plausible loop-index name in scientific code. The
#: macro is removed immediately after the include; complex literals are built with ``CMPLX``.
C_UNDEF_LINE: str = '#undef I  // <complex.h> defines I, which an SDFG may use as a container name'


class Tables(NamedTuple):
    """One dialect's complete lowering vocabulary.

    Bundled rather than looked up table by table so a new dialect cannot be half-added: every
    consumer takes the bundle, so a missing member is a construction error here instead of a
    ``dace::`` name reaching the output through the one table nobody remapped.
    """
    #: Runtime function -> the standard-library function with identical semantics.
    std_renames: Dict[str, str]
    #: Runtime function -> ``(arity, format string over the printed arguments)``.
    rewrites: Dict[str, Tuple[int, str]]
    #: Function name -> the definition CPF emits for it.
    inline_definitions: Dict[str, str]
    #: ``Max``/``Min`` -> the binary function they nest into.
    variadic_minmax: Dict[str, str]
    #: Function name -> why this dialect cannot express it.
    unsupported: Dict[str, str]
    #: ``dace::``-namespaced type -> its plain spelling.
    ctype_renames: Dict[str, str]
    #: Headers every unit includes.
    base_headers: Tuple[str, ...]
    #: Definition -> the definitions it calls.
    definition_dependencies: Dict[str, Tuple[str, ...]]
    #: Definition -> the headers its body needs, beyond ``base_headers``.
    definition_headers: Dict[str, Tuple[str, ...]]
    #: A call to one of ``inline_definitions`` in already-emitted code.
    helper_call: 're.Pattern'
    #: Every name this dialect knows, in any lane.
    known: Set[str]


#: An OPTIONAL explicit template-argument list between a helper's name and its call parentheses
#: (``get_scratch<ScanTag>(...)``). Deliberately narrow -- no parentheses, no nesting, one line --
#: so a comparison chain (``a < b && c > (d)``) cannot read as one.
_EXPLICIT_TEMPLATE_ARGUMENTS = r'(?:<[^<>();{}\n]*>\s*)?'


def _tables(std_renames, rewrites, inline_definitions, minmax, unsupported, ctype_renames, base_headers, dependencies,
            definition_headers) -> Tables:
    return Tables(std_renames=std_renames,
                  rewrites=rewrites,
                  inline_definitions=inline_definitions,
                  variadic_minmax=minmax,
                  unsupported=unsupported,
                  ctype_renames=ctype_renames,
                  base_headers=base_headers,
                  definition_dependencies=dependencies,
                  definition_headers=definition_headers,
                  helper_call=re.compile(r'(?<![\w:.])(' + '|'.join(sorted(inline_definitions, key=len, reverse=True)) +
                                         r')\s*' + _EXPLICIT_TEMPLATE_ARGUMENTS + r'\('),
                  known=(set(std_renames) | set(rewrites) | set(inline_definitions) | set(minmax) | set(unsupported)))


#: What a DEVICE unit adds to :data:`INLINE_DEFINITIONS`: the runtime functions a device library
#: expansion calls, written out the same way the host ones are. Device-only, so they are NOT in the
#: shared table -- every body here holds a kernel launch or ``__global__``, which a host compiler
#: cannot parse.
#:
#: They are reached the same way too. A ``FindFirst`` or ``Scan`` on a device graph expands to a
#: HOST tasklet plus a wrapper in the device unit, and the wrapper calls one of these; CPF's own
#: definition stands in for the runtime header the wrapper would otherwise include, exactly as
#: ``find_first_index`` and ``scan_incl_sum`` stand in for ``dace/scan.hpp`` on the host.
HIP_DEVICE_INLINE_DEFINITIONS: Dict[str, str] = {
    'get_scratch':
    '//: The CUB scratch pool, as much of it as one entry point can hold: one buffer per tag,\n'
    '//: allocated on first use and grown, never shrunk, so a repeated call pays no allocation.\n'
    '//: The runtime pre-allocates and releases it from the entry points a repeated invocation\n'
    '//: shares; one self-contained call has neither, so the buffer owns itself and frees at\n'
    '//: static destruction. It is also not keyed by stream, as the runtime pool is: a CPF unit\n'
    '//: issues every launch on the one stream (see cpf_gpu_context), so its calls are ordered.\n'
    'struct ScanTag {};\n'
    'struct DetectFlagTag {};\n'
    'struct cpf_scratch_block {\n'
    '    void *ptr = nullptr;\n'
    '    size_t capacity = 0;\n'
    '    ~cpf_scratch_block() { if (ptr != nullptr) { (void)hipFree(ptr); } }\n'
    '};\n'
    'template <typename Tag>\n'
    'static inline void *get_scratch(size_t bytes, gpuStream_t stream, gpuError_t *status) {\n'
    '    static cpf_scratch_block block;\n'
    '    *status = gpuSuccess;\n'
    '    if (bytes <= block.capacity) return block.ptr;\n'
    '    if (block.ptr != nullptr) {\n'
    '        // In flight work may still be reading the old buffer.\n'
    '        *status = hipStreamSynchronize(stream);\n'
    '        if (*status != gpuSuccess) return nullptr;\n'
    '        *status = hipFree(block.ptr);\n'
    '        block.ptr = nullptr;\n'
    '        block.capacity = 0;\n'
    '        if (*status != gpuSuccess) return nullptr;\n'
    '    }\n'
    '    *status = hipMalloc(&block.ptr, bytes);\n'
    '    if (*status != gpuSuccess) { block.ptr = nullptr; return nullptr; }\n'
    '    block.capacity = bytes;\n'
    '    return block.ptr;\n'
    '}',
    'find_first_index_device':
    '//: The device find-first: the smallest index in [begin, end) at which ``pred`` fires, or\n'
    '//: ``end``. The answer is an atomic min over the firing indices and is exact; the read of it\n'
    '//: at the top of each step races by design, and every value it can take is a real firing\n'
    '//: index, so a stale one costs pruning and never correctness.\n'
    '//:\n'
    '//: ``pred`` is a FUNCTOR and not a lambda so no caller needs --extended-lambda: the\n'
    '//: expansion appends the struct to the device code beside the wrapper that instantiates it.\n'
    'template <typename Pred>\n'
    '__global__ void find_first_kernel(long long begin, long long end, Pred pred,\n'
    '                                  unsigned long long *result) {\n'
    '    const long long stride = (long long)gridDim.x * (long long)blockDim.x;\n'
    '    long long i = begin + (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n'
    '    for (; i < end; i += stride) {\n'
    '        // Every index below this thread\'s next one is another thread\'s, so the first hit it\n'
    '        // finds walking upwards is the smallest it can contribute: it is done either way.\n'
    '        if (i >= (long long)*(volatile unsigned long long *)result) return;\n'
    '        if (pred(i)) { atomicMin(result, (unsigned long long)i); return; }\n'
    '    }\n'
    '}\n'
    'template <typename Pred>\n'
    'static inline gpuError_t find_first_index_device(long long begin, long long end, Pred pred, long long *out,\n'
    '                                                gpuStream_t stream) {\n'
    '    constexpr int block_threads = 256;\n'
    '    constexpr long long max_blocks = 1024;\n'
    '    *out = end;\n'
    '    if (begin >= end) return gpuSuccess;\n'
    '    gpuError_t status = gpuSuccess;\n'
    '    unsigned long long *result = (unsigned long long *)get_scratch<DetectFlagTag>(\n'
    '        sizeof(unsigned long long), stream, &status);\n'
    '    if (result == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n'
    '    const unsigned long long sentinel = (unsigned long long)end;\n'
    '    status = hipMemcpyAsync(result, &sentinel, sizeof(sentinel), hipMemcpyHostToDevice, stream);\n'
    '    if (status != gpuSuccess) return status;\n'
    '    // ``sentinel`` is a local, so the copy has to be done before this frame goes away.\n'
    '    status = hipStreamSynchronize(stream);\n'
    '    if (status != gpuSuccess) return status;\n'
    '    long long blocks = (end - begin + block_threads - 1) / block_threads;\n'
    '    if (blocks > max_blocks) blocks = max_blocks;\n'
    '    find_first_kernel<<<(unsigned)blocks, block_threads, 0, stream>>>(begin, end, pred, result);\n'
    '    status = hipGetLastError();\n'
    '    if (status != gpuSuccess) return status;\n'
    '    unsigned long long found = sentinel;\n'
    '    status = hipMemcpyAsync(&found, result, sizeof(found), hipMemcpyDeviceToHost, stream);\n'
    '    if (status != gpuSuccess) return status;\n'
    '    status = hipStreamSynchronize(stream);\n'
    '    *out = (long long)found;\n'
    '    return status;\n'
    '}',
    'inclusive_affine':
    '//: The first-order linear recurrence out[k] = c[k]*out[k-1] + d[k], entered at out[-1] = seed.\n'
    '//: The carry is the affine MAP x -> a*x + b rather than a value, and map composition is\n'
    '//: associative, so a plain prefix scan over the maps computes the recurrence.\n'
    '//:\n'
    '//: The seed is folded into element 0 rather than handed to cub as an init value, and that is\n'
    '//: numerically load-bearing: element 0 comes out as the CONSTANT map {0, c[0]*seed + d[0]},\n'
    '//: so every prefix including it carries a == 0 and the coefficient product never spans more\n'
    '//: than one composed segment -- which is what keeps it off the overflow the closed form hits.\n'
    'template <typename E>\n'
    'struct cpf_affine_map { E a; E b; };\n'
    'template <typename E>\n'
    'struct cpf_affine_compose {\n'
    '    __device__ __forceinline__ cpf_affine_map<E> operator()(const cpf_affine_map<E> &x,\n'
    '                                                            const cpf_affine_map<E> &y) const {\n'
    '        return cpf_affine_map<E>{y.a * x.a, y.a * x.b + y.b};\n'
    '    }\n'
    '};\n'
    'template <typename E, typename C, typename D, typename S>\n'
    '__global__ void cpf_affine_pack_kernel(const C *__restrict__ c, const D *__restrict__ d,\n'
    '                                       cpf_affine_map<E> *__restrict__ m, const S *__restrict__ seed_ptr,\n'
    '                                       E seed_val, long long n) {\n'
    '    const long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n'
    '    if (k >= n) return;\n'
    '    const E ck = (E)c[k];\n'
    '    const E dk = (E)d[k];\n'
    '    // A device-resident seed arrives as a pointer; a host-readable one by value.\n'
    '    if (k == 0) {\n'
    '        const E s = (seed_ptr != nullptr) ? (E)(*seed_ptr) : seed_val;\n'
    '        m[0] = cpf_affine_map<E>{(E)0, ck * s + dk};\n'
    '    } else {\n'
    '        m[k] = cpf_affine_map<E>{ck, dk};\n'
    '    }\n'
    '}\n'
    'template <typename E>\n'
    '__global__ void cpf_affine_unpack_kernel(const cpf_affine_map<E> *__restrict__ m, E *__restrict__ out,\n'
    '                                         long long n) {\n'
    '    const long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n'
    '    if (k < n) out[k] = m[k].b;  // every composed prefix is constant, so b IS the value\n'
    '}\n'
    'template <typename E, typename C, typename D, typename S>\n'
    'static inline gpuError_t inclusive_affine(const C *coef, const D *delta, const S *seed_ptr, E seed_val,\n'
    '                                          E *out, long long n, gpuStream_t stream) {\n'
    '    using M = cpf_affine_map<E>;\n'
    '    constexpr int block_threads = 256;\n'
    '    if (n <= 0) return gpuSuccess;\n'
    '    cpf_affine_compose<E> op;\n'
    '    size_t cub_bytes = 0;\n'
    '    gpuError_t status = gpucub::DeviceScan::InclusiveScan(nullptr, cub_bytes, (M *)nullptr, (M *)nullptr,\n'
    '                                                         op, n, stream);\n'
    '    if (status != gpuSuccess) return status;\n'
    '    // 256-byte alignment for the workspace that follows: cub assumes an allocation at least as\n'
    '    // aligned as hipMalloc gives, and the maps sit in front of it in the one block.\n'
    '    const size_t map_bytes = (((size_t)n * sizeof(M)) + 255u) & ~(size_t)255u;\n'
    '    void *scratch = get_scratch<ScanTag>(map_bytes + cub_bytes, stream, &status);\n'
    '    if (scratch == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n'
    '    M *maps = (M *)scratch;\n'
    '    void *workspace = (char *)scratch + map_bytes;\n'
    '    const unsigned blocks = (unsigned)((n + block_threads - 1) / block_threads);\n'
    '    cpf_affine_pack_kernel<E, C, D, S><<<blocks, block_threads, 0, stream>>>(coef, delta, maps, seed_ptr,\n'
    '                                                                            seed_val, n);\n'
    '    status = hipGetLastError();\n'
    '    if (status != gpuSuccess) return status;\n'
    '    status = gpucub::DeviceScan::InclusiveScan(workspace, cub_bytes, maps, maps, op, n, stream);\n'
    '    if (status != gpuSuccess) return status;\n'
    '    cpf_affine_unpack_kernel<E><<<blocks, block_threads, 0, stream>>>(maps, out, n);\n'
    '    return hipGetLastError();\n'
    '}',
}

HIP_INLINE_DEFINITIONS: Dict[str, str] = {**INLINE_DEFINITIONS, **HIP_DEVICE_INLINE_DEFINITIONS}

HIP_DEFINITION_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    **DEFINITION_DEPENDENCIES,
    'find_first_index_device': ('get_scratch', ),
    'inclusive_affine': ('get_scratch', ),
}

#: The CUB scratch pool's tag types, which reach the text as template ARGUMENTS rather than as
#: calls. Renamed here, with the type table, because that is what runs over the whole unit -- a
#: name table entry would need a call to fire on.
HIP_CTYPE_RENAMES: Dict[str, str] = {
    **CTYPE_RENAMES,
    'dace::cub::ScanTag': 'ScanTag',
    'dace::cub::DetectFlagTag': 'DetectFlagTag',
}

#: Environments a DEVICE rendering supplies for itself, so that
#: ``framecode.generate_standalone_footer`` does not refuse them. Everything else with code to run
#: or a library to link is still refused: a single entry point has nowhere to run an initializer.
#:
#: ``CUDA`` declares the toolkit as a CMake package. A device dialect is BUILT by that toolkit's
#: compiler, so the package is there by construction, and there is nothing to find and nothing to
#: link. ``ScanScratch`` and ``DetectScratch`` pre-allocate and release the CUB scratch pool;
#: :data:`HIP_DEVICE_INLINE_DEFINITIONS`'s ``get_scratch`` allocates on first use and frees at
#: static destruction, so both halves of that handshake are inside the unit.
DEVICE_PROVIDED_ENVIRONMENTS: FrozenSet[str] = frozenset({'CUDA', 'ScanScratch', 'DetectScratch'})

#: Dialect -> its vocabulary. ``RUNTIME`` has none: a runtime rendering emits ``dace::`` names and
#: never consults these tables at all, so asking for its bundle is a bug worth a ``KeyError``.
TABLES: Dict[Dialect, Tables] = {
    Dialect.STANDALONE:
    _tables(STD_RENAMES, REWRITES, INLINE_DEFINITIONS, VARIADIC_MINMAX, UNSUPPORTED, CTYPE_RENAMES, BASE_HEADERS,
            DEFINITION_DEPENDENCIES, DEFINITION_HEADERS),
    Dialect.STANDALONE_C:
    _tables(C_STD_RENAMES, C_REWRITES, C_INLINE_DEFINITIONS, C_VARIADIC_MINMAX, C_UNSUPPORTED, C_CTYPE_RENAMES,
            C_BASE_HEADERS, C_DEFINITION_DEPENDENCIES, {}),
    # The HIP unit is C++, so it takes the C++ vocabulary and adds to it: the ROCm toolkit's own
    # headers, which ship with the compiler that builds the unit, and the device counterparts of
    # the runtime functions a DEVICE library expansion calls (:data:`HIP_INLINE_DEFINITIONS`).
    Dialect.STANDALONE_HIP:
    _tables(STD_RENAMES, REWRITES, HIP_INLINE_DEFINITIONS, VARIADIC_MINMAX, UNSUPPORTED, HIP_CTYPE_RENAMES,
            BASE_HEADERS + HIP_BASE_HEADERS, HIP_DEFINITION_DEPENDENCIES, DEFINITION_HEADERS),
}

#: Every runtime function the C dialect knows about, in any lane.
C_KNOWN: Set[str] = TABLES[Dialect.STANDALONE_C].known


def tables_for(dialect: Optional[Dialect] = None) -> Tables:
    """The lowering vocabulary of ``dialect``, or of the ambient dialect when none is given.

    :raises ValueError: for :attr:`Dialect.RUNTIME`, which has no vocabulary at all -- a runtime
                        rendering emits ``dace::`` names and never consults these tables, so
                        reaching here under it means the caller forgot a :func:`dialect_scope`.
    """
    resolved = dialect if dialect is not None else _active_dialect
    if resolved not in TABLES:
        raise ValueError(f'{resolved} has no CPF lowering tables; the standalone dialects are '
                         f'{sorted(d.value for d in TABLES)}. Name one, or run inside a dialect_scope.')
    return TABLES[resolved]


#: Plain scalar type spellings a library expansion may write a FUNCTIONAL cast with
#: (``double(0)``, the ``Scan`` expansion's seed). C has no functional cast, so these become
#: ordinary cast expressions. Anchored on a closed list of type names rather than on "identifier
#: followed by ``(``", which would rewrite every call in the body.
C_CAST_TYPES: Tuple[str,
                    ...] = ('long double', 'unsigned long long', 'unsigned long', 'unsigned int', 'unsigned char',
                            'long long', 'double', 'float', 'bool', 'char', 'short', 'int', 'long', 'int8_t', 'int16_t',
                            'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'size_t')

_C_STATIC_CAST = re.compile(r'\bstatic_cast\s*<\s*([^<>;{}]+?)\s*>\s*\(')
_C_FUNCTIONAL_CAST = re.compile(r'(?<![\w:.])(' + '|'.join(C_CAST_TYPES) + r')\s*\(')

#: The scan's min / max neutral element per element type, as a C constant expression. Mirrors
#: ``dace::scan::min_identity`` / ``max_identity``: infinity where the type has one, otherwise the
#: extreme value. The C++ version is one function template; C needs the type spelled out, and the
#: call site already spells it (``min_identity<double>()``), so a constant is enough and no
#: definition has to be emitted.
C_SCAN_IDENTITIES: Dict[str, Tuple[str, str]] = {
    'int': ('INT_MAX', 'INT_MIN'),
    'long': ('LONG_MAX', 'LONG_MIN'),
    'long long': ('LLONG_MAX', 'LLONG_MIN'),
    'float': ('INFINITY', '-INFINITY'),
    'double': ('INFINITY', '-INFINITY'),
    'long double': ('INFINITY', '-INFINITY'),
}

#: ``min_identity<double>()`` as a hand-written expansion writes it, qualified or not.
_C_IDENTITY_CALL = re.compile(r'(?:::)?(?:\w+::)*\b(min|max)_identity\s*<\s*([A-Za-z_][A-Za-z_ ]*?)\s*>\s*\(\s*\)')


def c_scan_identities(code: str) -> str:
    """Replace every ``min_identity<T>()`` / ``max_identity<T>()`` with its C constant.

    Must run BEFORE the qualified-name rewrite: the name carries an explicit template argument, so
    it is not an ordinary call the name table could map.

    :param code: the C++ body, with its ctypes already renamed.
    :returns: the body with the identities spelled as C.
    :raises NotImplementedError: on an element type with no ordered extreme (complex).
    """

    def replace(match: 're.Match') -> str:
        kind, ctype = match.group(1), match.group(2)
        identities = C_SCAN_IDENTITIES.get(ctype)
        if identities is None:
            raise NotImplementedError(f'CPF cannot spell the scan {kind} identity for {ctype!r}: it has no ordered '
                                      'extreme value, so only sum and product scans are defined for it.')
        return identities[0] if kind == 'min' else identities[1]

    return _C_IDENTITY_CALL.sub(replace, code)


#: ``target = dace::find_first_index((begin), (end), [&](long long __i) -> bool { return (pred); },
#: parallel);`` -- the one statement ``ExpandFindFirstPure`` and ``ExpandFindFirstOpenMP`` write.
#: Anchored on the whole statement, target included, because the C replacement is a statement macro
#: and needs somewhere to put the result. The bounds are captured as ONE group and spliced through
#: unread: the expansion parenthesizes each of them, so they arrive as two macro arguments however
#: many commas the extents contain. The predicate is parenthesized by the same expansion, which is
#: what keeps a comma inside it (``cpf_max(a, b) > 0``) from splitting the macro argument.
_C_FIND_FIRST_CALL = re.compile(
    r'([^;{}\n]+?)\s*=\s*(?:::)?(?:[A-Za-z_]\w*::)*find_first_index\s*\(\s*'
    r'(.+?),\s*\[&\]\s*\(\s*long long\s+([A-Za-z_]\w*)\s*\)\s*->\s*bool\s*\{\s*return\s+(.+?)\s*;\s*\}'
    r'\s*,\s*([A-Za-z_]\w*)\s*\)\s*;', re.S)


def c_find_first(code: str) -> str:
    """Rewrite a ``find_first_index`` call over a C++ lambda into the C statement macro.

    C has no lambda and no way to hand a capturing predicate to a function, so the predicate cannot
    stay an argument to anything callable -- it has to be pasted into the search's innermost loop,
    which makes the search a macro. This is the only construct CPF answers by rewriting a call site
    rather than by naming a helper, so it is deliberately narrow: it matches the exact statement the
    two CPU expansions write, and anything else is left alone for ``dace.codegen.cpf.verify`` to
    report as an unlowered ``dace::`` name rather than half-rewritten into something that builds.

    Must run BEFORE the qualified-name rewrite, which would otherwise leave the C++ call shape in
    place with only its namespace stripped.

    :param code: the C++ body, with its ctypes already renamed.
    :returns: the body with the search spelled as C.
    """
    return _C_FIND_FIRST_CALL.sub(
        lambda match: 'cpf_find_first(%s, %s, %s, %s, %s);' %
        (match.group(1).strip(), match.group(2).strip(), match.group(3), match.group(5), match.group(4).strip()), code)


#: A ``std::`` name DaCe's own passes write STRAIGHT into a tasklet body, and the C spelling of it.
#: These never reach an expression printer and are not ``dace::`` names either, so neither lowering
#: lane sees them -- and in C a ``std::`` name is not a name at all.
#:
#: One entry, and it is the assumption guard: canonicalization's last pass traps a violated symbol
#: assumption with ``if ((N < 0)) { std::abort(); }``, and that pass DEDUPS its own guards by
#: searching tasklet bodies for the literal ``std::abort``, so the spelling is fixed at the source
#: and cannot be changed there without breaking the dedup. C declares ``abort`` in ``<stdlib.h>``,
#: which :data:`C_BASE_HEADERS` already includes.
C_NATIVE_RENAMES: Dict[str, str] = {'std::abort': 'abort'}


def c_native_renames(code: str) -> str:
    """Spell the ``std::`` names a hand-written body carries the C way.

    :param code: the body as the pass wrote it.
    :returns: the body with each :data:`C_NATIVE_RENAMES` name replaced.
    """
    for qualified, plain in C_NATIVE_RENAMES.items():
        code = re.sub(r'(?:::)?\b%s\b' % re.escape(qualified), plain, code)
    return code


def c_cast_native_code(code: str) -> str:
    """Rewrite the C++ casts in a hand-written body to C casts.

    Native tasklet bodies are emitted verbatim, so this is the only point at which a library
    expansion's own C++ can be re-spelled. The two forms that actually appear are the ``Scan``
    expansion's ``static_cast<long>(n)`` length and its ``double(0)`` seed.

    :param code: the body, after :func:`rewrite_native_code` has re-spelled its ``dace::`` names.
    :returns: the body with both cast forms written as ``(type)(...)``.
    """
    code = _C_STATIC_CAST.sub(lambda match: '(%s)(' % match.group(1), code)
    return _C_FUNCTIONAL_CAST.sub(lambda match: '(%s)(' % match.group(1), code)
