# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How MPR spells the functions the DaCe runtime headers normally provide.

MPR (maximal parallel rendering) emits C++ -- or C23, which is the same semantics in a language
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
    signedness, or simply too long to inline at every use). MPR emits the definition once, at the
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
    #: No DaCe headers; emit only the C++ standard library and MPR's own inline definitions.
    STANDALONE = 'standalone'
    #: No DaCe headers and no C++ either: one C23 translation unit. The C standard library is not
    #: type-generic and has no templates, so every helper is a ``_Generic`` dispatch macro over a
    #: closed set of typed ``static inline`` functions (see :data:`C_INLINE_DEFINITIONS`).
    STANDALONE_C = 'standalone_c'


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


#: The dialects that emit a self-contained translation unit. Everything MPR refuses -- device
#: code, a state struct, an external buffer handshake -- it refuses for both of them, so the many
#: call sites that ask "is this an MPR rendering" ask through :func:`standalone`.
STANDALONE_DIALECTS = frozenset({Dialect.STANDALONE, Dialect.STANDALONE_C})


def standalone() -> bool:
    """Whether the ambient dialect renders a self-contained unit, C++ or C."""
    return _active_dialect in STANDALONE_DIALECTS


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
#: matrix product. MPR restores that: the description is recorded when the node is expanded and
#: written as a comment where the expansion's code is emitted. Keyed by GUID rather than by node
#: object because the code generator runs its own lowering (inlining, copy lifting) between the
#: expansion and the emission.
#:
#: Ambient for the same reason the dialect is: the emission points are inside the code generators,
#: several call layers below anything that knows an expansion happened. Nothing memoized reads it.
_provenance: Dict[str, Tuple[str, str]] = {}


def describe(guid: str) -> Optional[Tuple[str, str]]:
    """``(origin, description)`` for the library node that produced ``guid``, if MPR recorded one.

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

#: Runtime function -> the C++ definition MPR emits for it.
#:
#: Emitted only when the translation unit calls the helper (see :func:`definitions_for`), so a
#: kernel that never rounds up never carries an ``int_ceil``. Each is ``static`` so several MPR
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
#: ``tests/codegen/mpr/test_lowering_table.py``.
INLINE_DEFINITIONS: Dict[str, str] = {
    'mpr_max':
    'template <typename T>\n'
    'static constexpr inline T mpr_max(const T& value) {\n'
    '    return value;\n'
    '}\n'
    'template <typename T, typename... Ts>\n'
    'static constexpr inline typename std::common_type<T, Ts...>::type mpr_max(const T& a, const Ts&... rest) {\n'
    '    return (a < mpr_max(rest...)) ? mpr_max(rest...) : a;\n'
    '}',
    'mpr_min':
    'template <typename T>\n'
    'static constexpr inline T mpr_min(const T& value) {\n'
    '    return value;\n'
    '}\n'
    'template <typename T, typename... Ts>\n'
    'static constexpr inline typename std::common_type<T, Ts...>::type mpr_min(const T& a, const Ts&... rest) {\n'
    '    return (mpr_min(rest...) < a) ? mpr_min(rest...) : a;\n'
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
    # be spelled into the clause. MPR reproduces them rather than rewriting a scan into a
    # sequential loop: the ``inscan`` form IS the parallel one, and a rendering that quietly
    # serialized every prefix sum would not be a maximal parallel rendering.
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
    '        acc = mpr_min(acc, static_cast<T>(f[i]));\n'
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
    '        acc = mpr_max(acc, static_cast<T>(f[i]));\n'
    '        #pragma omp scan inclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_excl_sum':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_sum(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, +:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = acc + f[i];\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_excl_product':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_product(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, *:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = acc * f[i];\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_excl_min':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_min(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, min:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = mpr_min(acc, static_cast<T>(f[i]));\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    'scan_excl_max':
    'template <typename It, typename OutIt, typename T>\n'
    'static inline void scan_excl_max(It f, OutIt o, long lo, long hi, T seed) {\n'
    '    T acc = seed;\n'
    '    #pragma omp simd reduction(inscan, max:acc)\n'
    '    for (long i = lo; i < hi; ++i) {\n'
    '        acc = mpr_max(acc, static_cast<T>(f[i]));\n'
    '        #pragma omp scan exclusive(acc)\n'
    '        o[i] = acc;\n'
    '    }\n'
    '}',
    # --- find-first ---------------------------------------------------------------------------
    # An early-exit loop lifts to a ``FindFirst`` library node whose expansion calls the runtime's
    # short-circuiting parallel search. MPR emits that search rather than unrolling it back into a
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
    'scan_incl_min': ('min_identity', 'mpr_min'),
    'scan_incl_max': ('max_identity', 'mpr_max'),
    'scan_excl_min': ('min_identity', 'mpr_min'),
    'scan_excl_max': ('max_identity', 'mpr_max'),
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
#: comparing strictly better), so the difference is arity and mixed-type promotion, but MPR still
#: emits the runtime's own definition (see :data:`INLINE_DEFINITIONS`) so the two cannot drift.
VARIADIC_MINMAX: Dict[str, str] = {'Max': 'mpr_max', 'Min': 'mpr_min', 'max': 'mpr_max', 'min': 'mpr_min'}

#: Headers MPR always includes: the exact-width integer types and the maths every kernel may reach,
#: plus the two the readable generator's own allocations need -- ``<new>`` for the aligned
#: ``operator new[](std::align_val_t)`` it allocates heap transients with, and ``<type_traits>``
#: for the ``std::is_trivially_destructible`` static assertion it pairs with the matching delete.
#: ``<cstdlib>`` is for ``std::abort``, which canonicalization writes into the assumption-guard
#: tasklet (``if ((N < 0)) { std::abort(); }``) -- a body no printer sees, so nothing else would
#: pull the declaration in.
BASE_HEADERS: Tuple[str, ...] = ('<cstdint>', '<cmath>', '<cstring>', '<cstdlib>', '<algorithm>', '<complex>',
                                 '<numeric>', '<new>', '<type_traits>')

#: Runtime functions MPR deliberately does NOT lower, and why. Reaching one is a refusal, not a
#: pass-through: the name is declared by a DaCe header MPR does not include, so passing it through
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
        raise NotImplementedError(f'MPR cannot emit the type {ctype!r}: {UNSUPPORTED_CTYPES[ctype]}.')
    return tables_for(dialect).ctype_renames.get(ctype, ctype)


def variadic_minmax(name: str, arguments: Tuple[str, ...], dialect: Optional[Dialect] = None) -> Optional[str]:
    """Spell a variadic ``Max``/``Min`` for ``dialect``.

    The runtime's ``Max`` takes any number of arguments, and so does the C++ dialect's own
    ``mpr_max`` template, so that one is called with the arguments as they stand. C has no
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
    """Whether MPR calls ``name`` unchanged and emits a definition for it."""
    return name in tables_for(dialect).inline_definitions


def lowering_for(name: str, arguments: Tuple[str, ...], dialect: Optional[Dialect] = None) -> Optional[str]:
    """The MPR spelling of a call to ``name`` with ``arguments`` already printed.

    :param name: the runtime function name as the ordinary generators would emit it.
    :param arguments: already-printed argument expressions.
    :returns: the C++ expression, or ``None`` if ``name`` needs no rewriting -- either it is not a
              runtime function at all, or it is one MPR emits a definition for and calls unchanged
              (:func:`needs_definition` separates those two).
    :raises ValueError: if ``name`` is a known rewrite but the argument count does not match, which
                        means the caller and this table disagree about the function's shape.
    :raises NotImplementedError: if ``name`` is a runtime function MPR cannot express (see
                                 :data:`UNSUPPORTED`).
    """
    tables = tables_for(dialect)
    if name in tables.unsupported:
        raise NotImplementedError(f'MPR cannot lower {name!r}: {tables.unsupported[name]}.')
    variadic = variadic_minmax(name, arguments, dialect)
    if variadic is not None:
        return variadic
    if name in tables.rewrites:
        arity, template = tables.rewrites[name]
        if len(arguments) != arity:
            raise ValueError(f'MPR lowering of {name!r} expects {arity} arguments, got {len(arguments)}')
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
            raise NotImplementedError(f'MPR cannot emit the type {qualified!r}: {reason}')
    for qualified, plain in tables_for(dialect).ctype_renames.items():
        code = re.sub(r'(?:::)?\b%s\b' % re.escape(qualified), plain, code)
    return code


def rewrite_native_code(code: str, dialect: Optional[Dialect] = None) -> str:
    """Rewrite the ``dace::`` names in a hand-written C++ body to their standalone spellings.

    Native tasklet bodies never reach the expression printers -- they are emitted verbatim -- so
    this is the only point at which a library expansion's own C++ can be re-spelled. Which is
    needed for the real cases: the ``Scan`` expansion calls ``::dace::scan::detail::scan_incl_sum``
    and the ``FindFirst`` expansion calls ``dace::find_first_index``, and MPR emits both functions
    itself rather than serializing a prefix sum or a cancelling search into a sequential loop.

    In C the same pass also rewrites the two call shapes C cannot express as a call at all -- the
    scan identities and the find-first over a lambda predicate (:func:`c_scan_identities`,
    :func:`c_find_first`) -- and re-spells the ``std::`` names a pass wrote directly
    (:func:`c_native_renames`), before the name table is consulted.

    Textual by necessity, and deliberately conservative: only the qualified name is rewritten, only
    when the identifier is one MPR knows, and never with knowledge of the arguments. A
    ``dace::`` name with no standalone spelling is LEFT ALONE, so it reaches
    ``dace.codegen.mpr.verify`` and is reported against the construct that emitted it -- a silent
    partial rewrite would be worse than none.

    :param code: the C++ body as the expansion wrote it.
    :param dialect: which standalone dialect to rewrite for; ambient when omitted.
    :returns: the body with the names MPR can spell rewritten.
    :raises NotImplementedError: if the body names a type or function this dialect cannot express.
    """
    tables = tables_for(dialect)
    code = rewrite_ctypes(code, dialect)

    if (dialect if dialect is not None else _active_dialect) is Dialect.STANDALONE_C:
        code = c_native_renames(c_find_first(c_scan_identities(code)))

    def replace(match: 're.Match') -> str:
        name = match.group(1)
        if name in tables.unsupported:
            raise NotImplementedError(f'MPR cannot lower {name!r}: {tables.unsupported[name]}.')
        if name in tables.inline_definitions:
            return name  # MPR emits this one's definition at the top of the unit
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
            raise ValueError(f'MPR inline definitions have a dependency cycle among {sorted(needed - placed)}')
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
#: of unsigned expression < 0 is always false") warns under ``-Wextra``, and MPR output must build
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
#: ``mpr_max`` / ``mpr_min`` dispatch on. Unsigned types belong HERE (the bodies compare two values
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
    # A family whose own name already carries the prefix (``mpr_max``) must not get it twice.
    stem = name[4:] if name.startswith('mpr_') else name
    for types, returns, body in groups:
        for ctype, suffix in types:
            target = 'mpr_%s_%s' % (stem, suffix)
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

#: Maths MPR emits for its OWN definitions rather than for a runtime rename: ``cpp_mod`` needs
#: ``fmod``, ``np_modf`` needs ``modf``, and the complex ``sign_numpy_2`` needs the component
#: accessors that :data:`REWRITES` spells ``.real()`` / ``.imag()`` in C++.
C_INTERNAL_MATH_SPEC: Tuple[Tuple[str, str, str, object], ...] = (
    ('fmod', 'fmod', 'real', 2),
    ('modf', 'modf', 'real', 'first2'),
    ('creal', 'creal', 'complex_component', 1),
    ('cimag', 'cimag', 'complex_component', 1),
)

#: Runtime maths C already spells type-generically, as a ``<math.h>`` MACRO. Wrapping these in an
#: ``mpr_`` dispatch would be wrong as well as pointless: there is no ``isnanf`` to dispatch TO.
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
    name = 'mpr_' + base
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

#: ``Max``/``Min`` in C. Not the ``<stdlib.h>`` integer ``max``, which does not exist: MPR emits its
#: own typed pair (see :data:`C_MINMAX_TYPES`).
C_VARIADIC_MINMAX: Dict[str, str] = {'Max': 'mpr_max', 'Min': 'mpr_min', 'max': 'mpr_max', 'min': 'mpr_min'}

#: Rewrites that differ from :data:`REWRITES` because their C++ form names a C++ construct: a
#: member call on ``std::complex``, or a ``static_cast``.
C_REWRITES: Dict[str, Tuple[int, str]] = dict(REWRITES)
C_REWRITES.update({
    're': (1, '(mpr_creal({0}))'),
    'im': (1, '(mpr_cimag({0}))'),
    'iround': (1, '((int)mpr_round({0}))'),
    'np_float_pow': (2, '(mpr_pow((double)({0}), (double)({1})))'),
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
    for name, condition in (('mpr_max', 'a < b'), ('mpr_min', 'b < a'))
}

_C_SIGN_BODY = 'return ({T})((({T})0 < value) - (value < ({T})0));'

#: The C form of every :data:`INLINE_DEFINITIONS` entry, plus the two ``<numeric>`` functions C has
#: no counterpart for at all (``gcd`` / ``lcm``, which are a rename in C++ and a definition here).
#:
#: The eight prefix scans keep their ``#pragma omp simd reduction(inscan, ...)`` bodies verbatim.
#: That form IS the parallel scan; a rendering that quietly serialized every prefix sum would not be
#: a maximal parallel rendering.
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
         (C_COMPLEX, '{T}', 'return (mpr_creal(value) != 0 && mpr_cimag(value) != 0) ? value / mpr_abs(value) : 0;')),
        '+(value)'),
    # Two arities, which no single C macro can have. The three-argument pick chooses between the
    # unary and binary dispatch macros by counting what the caller wrote.
    'heaviside':
    '\n'.join((
        '#define mpr_pick3(a0, a1, a2, ...) a2',
        c_typed_family('mpr_heaviside_1', (('{T}', 'value'), ),
                       ((C_ARITHMETIC, '{T}', 'return (value > ({T})0) ? ({T})1 : ({T})0;'), ), '+(value)'),
        c_typed_family(
            'mpr_heaviside_2', (('{T}', 'value'), ('{T}', 'at_zero')),
            ((C_ARITHMETIC, '{T}', 'return (value < ({T})0) ? ({T})0 : ((value > ({T})0) ? ({T})1 : at_zero);'), ),
            '(value) + (at_zero)'),
        '#define heaviside(...) mpr_pick3(__VA_ARGS__, mpr_heaviside_2, mpr_heaviside_1)(__VA_ARGS__)',
    )),
    # Integral input is already floored, so it comes back unchanged -- narrowing an int64 through
    # ``(int)floor(...)`` would truncate it to 32 bits.
    'ifloor':
    c_typed_family('ifloor', (('{T}', 'value'), ),
                   ((C_SIGNED_INTS, '{T}', 'return value;'), (C_FLOATS, 'int', 'return (int)mpr_floor(value);')),
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
                    (C_FLOATS, '{T}', 'return mpr_floor(numerator / denominator);')), '(numerator) + (denominator)'),
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
                    (C_FLOATS, '{T}', 'return mpr_fmod(numerator, denominator);')), '(numerator) + (denominator)'),
    'Mod_float':
    c_typed_family('Mod_float', (('{T}', 'value'), ('{T}', 'modulus')),
                   ((C_FLOATS, '{T}', 'return value - (int)(value / modulus) * modulus;'), ), '(value) + (modulus)'),
    'Modulo':
    c_typed_family(
        'Modulo', (('{T}', 'value'), ('{T}', 'modulus')),
        ((C_ARITHMETIC, '{T}', 'return value - ({T})mpr_floor((double)(value) / (double)(modulus)) * modulus;'), ),
        '(value) + (modulus)'),
    'Modulo_float':
    c_typed_family('Modulo_float', (('{T}', 'value'), ('{T}', 'modulus')),
                   ((C_FLOATS, '{T}', 'return value - ({T})mpr_floor(value / modulus) * modulus;'), ),
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
                    (C_FLOATS, 'void', '*fractional = mpr_modf(value, integral);')), '+(value)',
                   ('value', '&(integral)', '&(fractional)')),
    'np_frexp':
    c_typed_family('np_frexp', (('{T}', 'value'), ('{T} *', 'mantissa'), ('int *', 'exponent')),
                   ((C_FLOATS, 'void', '*mantissa = mpr_frexp(value, exponent);'), ), '+(value)',
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
    """One prefix-scan helper as C: one typed function per element type, plus its dispatch macro."""
    body = ('{T} acc = seed;\n'
            '#pragma omp simd reduction(inscan, %s:acc)\n'
            'for (long i = lo; i < hi; ++i) {\n'
            '    acc = %s;\n'
            '    #pragma omp scan %s(acc)\n'
            '    o[i] = acc;\n'
            '}') % (clause, step, kind)
    return c_typed_family('scan_%s_%s' % ('incl' if kind == 'inclusive' else 'excl', operation),
                          (('const {T} *', 'f'), ('{T} *', 'o'), ('long', 'lo'), ('long', 'hi'), ('{T}', 'seed')),
                          ((C_ARITHMETIC, 'void', body), ), '+(seed)')


for _kind in ('inclusive', 'exclusive'):
    for _operation, _clause, _step in (('sum', '+', 'acc + f[i]'), ('product', '*', 'acc * f[i]'),
                                       ('min', 'min', 'mpr_min(acc, ({T})f[i])'), ('max', 'max',
                                                                                   'mpr_max(acc, ({T})f[i])')):
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
C_INLINE_DEFINITIONS['mpr_find_first'] = '\\\n'.join((
    '#define mpr_find_first(out, ff_begin, ff_end, ff_index, ff_parallel, ff_pred) ',
    '    do {',
    '        const long long mpr_ff_lo = (ff_begin);',
    '        const long long mpr_ff_end = (ff_end);',
    '        const bool mpr_ff_par = (ff_parallel);',
    # The block is the early-exit granularity: a vectorized loop cannot break.
    '        const long long mpr_ff_simd = 64;',
    '        long long mpr_ff_best = mpr_ff_end;',
    '        if (mpr_ff_lo < mpr_ff_end) {',
    '            const long long mpr_ff_span = mpr_ff_end - mpr_ff_lo;',
    '            const long long mpr_ff_chunk = find_first_chunk(mpr_ff_span, mpr_ff_par);',
    '            const long long mpr_ff_chunks = (mpr_ff_span + mpr_ff_chunk - 1) / mpr_ff_chunk;',
    '            long long mpr_ff_hint = mpr_ff_end;',
    '            _Pragma("omp parallel for schedule(dynamic, 1) if (parallel : mpr_ff_par) '
    'reduction(min : mpr_ff_best)")',
    '            for (long long mpr_ff_c = 0; mpr_ff_c < mpr_ff_chunks; ++mpr_ff_c) {',
    '                long long mpr_ff_seen, mpr_ff_hi, mpr_ff_found, mpr_ff_b;',
    '                const long long mpr_ff_from = mpr_ff_lo + mpr_ff_c * mpr_ff_chunk;',
    '                _Pragma("omp atomic read")',
    '                mpr_ff_seen = mpr_ff_hint;',
    '                if (mpr_ff_from >= mpr_ff_seen) continue;',
    '                mpr_ff_hi = mpr_ff_from + mpr_ff_chunk;',
    '                if (mpr_ff_hi > mpr_ff_end) mpr_ff_hi = mpr_ff_end;',
    '                if (mpr_ff_hi > mpr_ff_seen) mpr_ff_hi = mpr_ff_seen;',
    '                mpr_ff_found = mpr_ff_end;',
    '                for (mpr_ff_b = mpr_ff_from; mpr_ff_b < mpr_ff_hi; mpr_ff_b += mpr_ff_simd) {',
    '                    long long mpr_ff_block = mpr_ff_end;',
    '                    long long mpr_ff_to = mpr_ff_b + mpr_ff_simd;',
    '                    if (mpr_ff_to > mpr_ff_hi) mpr_ff_to = mpr_ff_hi;',
    '                    _Pragma("omp simd reduction(min : mpr_ff_block)")',
    '                    for (long long ff_index = mpr_ff_b; ff_index < mpr_ff_to; ++ff_index) {',
    '                        const long long mpr_ff_v = (ff_pred) ? ff_index : mpr_ff_end;',
    '                        mpr_ff_block = mpr_ff_v < mpr_ff_block ? mpr_ff_v : mpr_ff_block;',
    '                    }',
    '                    if (mpr_ff_block < mpr_ff_end) { mpr_ff_found = mpr_ff_block; break; }',
    '                }',
    '                if (mpr_ff_found < mpr_ff_end) {',
    '                    long long mpr_ff_cur;',
    '                    if (mpr_ff_found < mpr_ff_best) mpr_ff_best = mpr_ff_found;',
    '                    _Pragma("omp atomic read")',
    '                    mpr_ff_cur = mpr_ff_hint;',
    '                    if (mpr_ff_found < mpr_ff_cur) {',
    '                        _Pragma("omp atomic write")',
    '                        mpr_ff_hint = mpr_ff_found;',
    '                    }',
    '                }',
    '            }',
    '        }',
    '        (out) = mpr_ff_best;',
    '    } while (0)',
))

#: Definitions each C definition calls -- macros included, since a macro must be ``#define``d before
#: the function body that expands it is compiled.
C_DEFINITION_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    'sign_numpy_2': ('mpr_creal', 'mpr_cimag', 'mpr_abs'),
    'ifloor': ('mpr_floor', ),
    'py_floor': ('int_floor_ni', 'mpr_floor'),
    'py_mod': ('py_floor', ),
    'floor_mod': ('py_mod', ),
    'cpp_mod': ('mpr_fmod', ),
    'Modulo': ('mpr_floor', ),
    'Modulo_float': ('mpr_floor', ),
    'py_divmod': ('cpp_divmod', ),
    'np_modf': ('mpr_modf', ),
    'np_frexp': ('mpr_frexp', ),
    'lcm': ('gcd', ),
    'scan_incl_min': ('mpr_min', ),
    'scan_incl_max': ('mpr_max', ),
    'scan_excl_min': ('mpr_min', ),
    'scan_excl_max': ('mpr_max', ),
    'mpr_find_first': ('find_first_chunk', ),
}

#: What the C dialect refuses, and why. Empty: every construct MPR reaches has a C spelling.
C_UNSUPPORTED: Dict[str, str] = {}

#: Helpers C answers with a REWRITE of the CALL SITE rather than a definition or a refusal -- a
#: third lane, and the only one, so the anti-rot tests can still insist every C++ helper is
#: accounted for. Each is a shape C cannot spell as a callable at all: the scan's neutral elements
#: need the element type, which only the call site names (:func:`c_scan_identities`), and the
#: find-first takes a predicate, which in C++ is a lambda and in C has to be pasted into the search
#: as a macro argument (:func:`c_find_first`).
C_REWRITTEN_IN_NATIVE_CODE: FrozenSet[str] = frozenset({'min_identity', 'max_identity', 'find_first_index'})

#: Headers MPR's C output always includes. ``<stdbool.h>`` is deliberately absent: ``bool`` /
#: ``true`` / ``false`` are C23 keywords. ``<tgmath.h>`` is deliberately absent too -- see the
#: section header above.
C_BASE_HEADERS: Tuple[str, ...] = ('<stdint.h>', '<math.h>', '<limits.h>', '<stdlib.h>', '<string.h>', '<complex.h>')

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
    #: Function name -> the definition MPR emits for it.
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
                                         r')\s*\('),
                  known=(set(std_renames) | set(rewrites) | set(inline_definitions) | set(minmax) | set(unsupported)))


#: Dialect -> its vocabulary. ``RUNTIME`` has none: a runtime rendering emits ``dace::`` names and
#: never consults these tables at all, so asking for its bundle is a bug worth a ``KeyError``.
TABLES: Dict[Dialect, Tables] = {
    Dialect.STANDALONE:
    _tables(STD_RENAMES, REWRITES, INLINE_DEFINITIONS, VARIADIC_MINMAX, UNSUPPORTED, CTYPE_RENAMES, BASE_HEADERS,
            DEFINITION_DEPENDENCIES, DEFINITION_HEADERS),
    Dialect.STANDALONE_C:
    _tables(C_STD_RENAMES, C_REWRITES, C_INLINE_DEFINITIONS, C_VARIADIC_MINMAX, C_UNSUPPORTED, C_CTYPE_RENAMES,
            C_BASE_HEADERS, C_DEFINITION_DEPENDENCIES, {}),
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
        raise ValueError(f'{resolved} has no MPR lowering tables; the standalone dialects are '
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
            raise NotImplementedError(f'MPR cannot spell the scan {kind} identity for {ctype!r}: it has no ordered '
                                      'extreme value, so only sum and product scans are defined for it.')
        return identities[0] if kind == 'min' else identities[1]

    return _C_IDENTITY_CALL.sub(replace, code)


#: ``target = dace::find_first_index((begin), (end), [&](long long __i) -> bool { return (pred); },
#: parallel);`` -- the one statement ``ExpandFindFirstPure`` and ``ExpandFindFirstOpenMP`` write.
#: Anchored on the whole statement, target included, because the C replacement is a statement macro
#: and needs somewhere to put the result. The bounds are captured as ONE group and spliced through
#: unread: the expansion parenthesizes each of them, so they arrive as two macro arguments however
#: many commas the extents contain. The predicate is parenthesized by the same expansion, which is
#: what keeps a comma inside it (``mpr_max(a, b) > 0``) from splitting the macro argument.
_C_FIND_FIRST_CALL = re.compile(
    r'([^;{}\n]+?)\s*=\s*(?:::)?(?:[A-Za-z_]\w*::)*find_first_index\s*\(\s*'
    r'(.+?),\s*\[&\]\s*\(\s*long long\s+([A-Za-z_]\w*)\s*\)\s*->\s*bool\s*\{\s*return\s+(.+?)\s*;\s*\}'
    r'\s*,\s*([A-Za-z_]\w*)\s*\)\s*;', re.S)


def c_find_first(code: str) -> str:
    """Rewrite a ``find_first_index`` call over a C++ lambda into the C statement macro.

    C has no lambda and no way to hand a capturing predicate to a function, so the predicate cannot
    stay an argument to anything callable -- it has to be pasted into the search's innermost loop,
    which makes the search a macro. This is the only construct MPR answers by rewriting a call site
    rather than by naming a helper, so it is deliberately narrow: it matches the exact statement the
    two CPU expansions write, and anything else is left alone for ``dace.codegen.mpr.verify`` to
    report as an unlowered ``dace::`` name rather than half-rewritten into something that builds.

    Must run BEFORE the qualified-name rewrite, which would otherwise leave the C++ call shape in
    place with only its namespace stripped.

    :param code: the C++ body, with its ctypes already renamed.
    :returns: the body with the search spelled as C.
    """
    return _C_FIND_FIRST_CALL.sub(
        lambda match: 'mpr_find_first(%s, %s, %s, %s, %s);' %
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
