# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Stockham autosort Fast Fourier Transform (FFT) as a native SDFG.

The transform of a length-``N`` axis is factored into radices ``R_0 * R_1 * ... = N``. Stage ``s`` is a single map
over the batch dimensions (all other axes), ``b`` in ``[0, L)`` and ``k`` in ``[0, M)``, where ``L`` is the product of
the radices of the previous stages and ``M = N / (L * R_s)``. Each map iteration reads ``R_s`` elements at
``(b * R_s + a) * M + k``, multiplies them by the twiddle factors ``exp(+-2 pi i a b / (L * R_s))``, computes a
radix-``R_s`` DFT in registers, and writes the results to ``(c * L + b) * M + k`` in another buffer. All stages of all
transformed axes are chained through two ping-pong buffers in one state, so the expansion can be inlined. The
twiddle factors of each stage are in a ``[R - 1, L]`` table, which a map in the same state computes in every call, or
a tasklet in the first call of the program for a persistent table in host memory.

On the CPU, a tasklet computes several consecutive iterations of a stage in a SIMD loop (with interleaved complex AVX
code for complex128 if the compiler targets AVX and FMA). All stages of an axis whose transforms fit into
``LOCAL_BYTES`` are one map over the other dimensions instead, in which each iteration transforms a row (or a block of
columns) in buffers on the stack (see ``add_local_axis_kernel``), so that the array is read and written once rather
than once per stage. On the GPU, the stage maps compute one butterfly per thread and fuse with elementwise maps.

Prime factors without a generated butterfly (see ``PRIME_RADICES``) use general-radix stages, in which each map
iteration computes one output ``j = c * L + b`` as the sum over ``a`` of the inputs times ``exp(+-2 pi i a j / (L *
R_s))``, read from a table of ``L * R_s`` roots of unity. Such a stage costs ``O(N * R_s)`` operations, so any fixed
length can be transformed, but a length with a large prime factor is transformed in up to ``O(N^2)`` operations.

The radices and all stage sizes are fixed at expansion time, so the lengths of the transformed axes must be known
integers. Batch dimensions may be symbolic.

The stages can also be generated for a transform that a single GPU thread block computes in shared memory (see
``block_local_fft_sdfg``). This is the GPU expansion of transforms that fit, and the ``BlockLocalFFTFusion``
transformation uses it to fuse an FFT, a computation in the frequency domain, and an inverse FFT into one kernel.
"""

import copy
import dataclasses
import itertools
import math

from dace import SDFG, Memlet, SDFGState, data, dtypes, nodes, properties, symbolic
from dace.sdfg import propagation

#: Radices that butterflies are generated for. Powers of two are split recursively inside the butterfly and odd
#: primes use a direct DFT. Other prime factors use general-radix stages.
POWER_OF_TWO_RADICES = (16, 8, 4, 2)
PRIME_RADICES = (3, 5, 7, 11, 13)

#: Arrays with fewer elements than this are transformed by sequential maps on the CPU, where the work of a stage is
#: too small to amortize starting a parallel region.
SEQUENTIAL_SIZE_THRESHOLD = 2048

#: Numbers of consecutive iterations that a CPU tasklet may compute in a SIMD loop, in order of preference.
VECTOR_LANES = (8, 4, 2)

#: Storage types of host memory, in which persistent twiddle factor tables are computed once.
HOST_STORAGES = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)

#: Stack memory of the two buffers in which a CPU map iteration may transform an axis (see ``local_lanes``).
LOCAL_BYTES = 128 * 1024

#: Thread block size of the GPU kernels (64-256 threads perform the same on an RTX 3070).
GPU_BLOCK_SIZE = [128, 1, 1]

#: Maximum number of threads of a thread block that computes a transform in shared memory. Larger transforms loop.
BLOCK_LOCAL_MAX_THREADS = 256

#: Shared memory that a block-local transform may use for its two buffers: the static shared memory limit of CUDA.
BLOCK_LOCAL_SHARED_BYTES = 48 * 1024


#: Unique identifiers of the flags of twiddle factor tables that are computed once.
_twiddle_flag_ids = itertools.count()


def factorize(n: int) -> list[int] | None:
    """
    Splits a transform length into the radices of the Stockham stages.

    :param n: The transform length.
    :return: The list of radices whose product is ``n`` (empty for ``n == 1``), or None if ``n`` is not positive.
             Prime factors with no generated butterfly come first, where their stages have the smallest tables,
             followed by the other odd primes and by powers of two in ascending order, so that the last stage of a
             power-of-two length has the largest radix (see ``local_padding``).
    """
    if n < 1:
        return None
    radices = []
    for radix in POWER_OF_TWO_RADICES + PRIME_RADICES:
        while n % radix == 0:
            radices.append(radix)
            n //= radix
    general = []
    prime = PRIME_RADICES[-1] + 2
    while prime * prime <= n:
        while n % prime == 0:
            general.append(prime)
            n //= prime
        prime += 2
    if n > 1:
        general.append(n)
    odd = [radix for radix in radices if radix in PRIME_RADICES]
    powers = sorted(radix for radix in radices if radix in POWER_OF_TWO_RADICES)
    return general + odd + powers


def is_general_radix(radix: int) -> bool:
    """
    Returns True if a stage of the given radix is a general-radix stage, which has no generated butterfly.
    """
    return radix > 1 and radix not in POWER_OF_TWO_RADICES + PRIME_RADICES


def fixed_length(length: symbolic.SymbolicType, constants: dict) -> int | None:
    """
    Returns the integer value of an axis length, or None if it depends on symbols that are not constants.
    """
    if symbolic.issymbolic(length, constants):
        return None
    return int(symbolic.evaluate(length, constants))


def plan_stages(
    shape: tuple[symbolic.SymbolicType, ...], axes: list[int], constants: dict
) -> list[tuple[int, int, int, int]]:
    """
    Returns the stages that transform the given axes of an array, in order.

    :param shape: The shape of the array.
    :param axes: The (normalized) axes to transform. Their lengths must be fixed (see ``can_expand``).
    :param constants: Constants of the SDFG, used to evaluate symbolic axis lengths.
    :return: A list of ``(axis, radix, L, M)`` tuples. A transform of length-one axes only has a single radix-1 stage.
    """
    stages = []
    for axis in axes:
        length = fixed_length(shape[axis], constants)
        span = 1
        for radix in factorize(length):
            stages.append((axis, radix, span, length // (span * radix)))
            span *= radix
    if not stages:
        stages.append((axes[0], 1, 1, 1))
    return stages


def can_expand(shape: tuple[symbolic.SymbolicType, ...], axes: list[int], constants: dict) -> bool:
    """
    Checks whether the Stockham expansion can transform the given axes of an array.

    :param shape: The shape of the array.
    :param axes: The (normalized) axes to transform.
    :param constants: Constants of the SDFG, used to evaluate symbolic axis lengths.
    :return: True if every transformed axis has a fixed, positive length.
    """
    for axis in axes:
        length = fixed_length(shape[axis], constants)
        if length is None or factorize(length) is None:
            return False
    return True


class _ButterflyEmitter:
    """
    Generates straight-line C++ code for small DFTs on values held in pairs of real variables.
    """

    def __init__(self, real_ctype: str):
        self.real_ctype = real_ctype
        self.lines: list[str] = []
        self.counter = 0

    def literal(self, value: float) -> str:
        text = repr(float(value))
        return f"{text}f" if self.real_ctype == "float" else text

    def define(self, expr: str) -> str:
        name = f"__v{self.counter}"
        self.counter += 1
        self.lines.append(f"const {self.real_ctype} {name} = {expr};")
        return name

    def add(self, x: tuple[str, str], y: tuple[str, str]) -> tuple[str, str]:
        return self.define(f"{x[0]} + {y[0]}"), self.define(f"{x[1]} + {y[1]}")

    def sub(self, x: tuple[str, str], y: tuple[str, str]) -> tuple[str, str]:
        return self.define(f"{x[0]} - {y[0]}"), self.define(f"{x[1]} - {y[1]}")

    def rotate(self, x: tuple[str, str], numerator: int, denominator: int, sign: int) -> tuple[str, str]:
        """
        Multiplies a value by the constant ``exp(sign * 2 pi i numerator / denominator)``.
        """
        numerator %= denominator
        if numerator == 0:
            return x
        if (4 * numerator) % denominator == 0:
            quarter = (4 * numerator // denominator * sign) % 4
            if quarter == 1:  # Multiplication by i
                return self.define(f"-{x[1]}"), x[0]
            if quarter == 2:
                return self.define(f"-{x[0]}"), self.define(f"-{x[1]}")
            return x[1], self.define(f"-{x[0]}")  # Multiplication by -i
        angle = sign * 2 * math.pi * numerator / denominator
        cos, sin = self.literal(math.cos(angle)), self.literal(math.sin(angle))
        return (self.define(f"{x[0]} * {cos} - {x[1]} * {sin}"), self.define(f"{x[0]} * {sin} + {x[1]} * {cos}"))

    def dft(self, values: list[tuple[str, str]], sign: int) -> list[tuple[str, str]]:
        """
        Emits a DFT of the given values and returns the output values in natural order.
        """
        n = len(values)
        if n == 1:
            return list(values)
        if n % 2 == 0:  # Radix-2 decimation in time
            even = self.dft(values[0::2], sign)
            odd = self.dft(values[1::2], sign)
            result = [None] * n
            for k in range(n // 2):
                rotated = self.rotate(odd[k], k, n, sign)
                result[k] = self.add(even[k], rotated)
                result[k + n // 2] = self.sub(even[k], rotated)
            return result
        return self._odd_dft(values, sign)

    def _odd_dft(self, values: list[tuple[str, str]], sign: int) -> list[tuple[str, str]]:
        # Pairs x[a] with x[n - a]: W^{ca} x[a] + W^{-ca} x[n-a] = cos(t) (x[a] + x[n-a]) + i sign sin(t) (x[a] - x[n-a])
        n = len(values)
        half = (n - 1) // 2
        sums = [self.add(values[a], values[n - a]) for a in range(1, half + 1)]
        diffs = [self.sub(values[a], values[n - a]) for a in range(1, half + 1)]
        result = [
            (
                self.define(" + ".join([values[0][0]] + [s[0] for s in sums])),
                self.define(" + ".join([values[0][1]] + [s[1] for s in sums])),
            )
        ]
        for c in range(1, n):
            real_terms, imag_terms = [values[0][0]], [values[0][1]]
            for a in range(1, half + 1):
                cos = self.literal(math.cos(2 * math.pi * c * a / n))
                sin = sign * math.sin(2 * math.pi * c * a / n)
                real_terms += [f"{cos} * {sums[a - 1][0]}", f"{self.literal(-sin)} * {diffs[a - 1][1]}"]
                imag_terms += [f"{cos} * {sums[a - 1][1]}", f"{self.literal(sin)} * {diffs[a - 1][0]}"]
            result.append((self.define(" + ".join(real_terms)), self.define(" + ".join(imag_terms))))
        return result


@dataclasses.dataclass(frozen=True)
class Vector:
    """
    Consecutive iterations of a stage that one tasklet computes in a SIMD loop, through pointers to the elements of
    the first iteration.
    """

    #: The number of iterations.
    length: int
    #: C++ expression of the distance between the inputs of consecutive iterations, in elements of the input array.
    input_stride: str
    #: C++ expression of the distance between the outputs of consecutive iterations.
    output_stride: str
    #: C++ expression of the distance between the twiddle factors of consecutive iterations, or None if they share
    #: their twiddle factors.
    twiddle_stride: str | None


class _AVXButterflyEmitter(_ButterflyEmitter):
    """
    Generates straight-line C++ code for small DFTs on ``__m256d`` values, each holding two interleaved complex
    numbers (of two consecutive iterations), with the helpers of ``AVX_HELPERS``.
    """

    def __init__(self):
        super().__init__("double")

    def define(self, expr: str) -> str:
        name = f"__v{self.counter}"
        self.counter += 1
        self.lines.append(f"const __m256d {name} = {expr};")
        return name

    def add(self, x: str, y: str) -> str:
        return self.define(f"_mm256_add_pd({x}, {y})")

    def sub(self, x: str, y: str) -> str:
        return self.define(f"_mm256_sub_pd({x}, {y})")

    def rotate(self, x: str, numerator: int, denominator: int, sign: int) -> str:
        numerator %= denominator
        if numerator == 0:
            return x
        if (4 * numerator) % denominator == 0:
            quarter = (4 * numerator // denominator * sign) % 4
            if quarter == 1:
                return self.define(f"__stockham_mul_i({x})")
            if quarter == 2:
                return self.define(f"__stockham_neg({x})")
            return self.define(f"__stockham_mul_minus_i({x})")
        angle = sign * 2 * math.pi * numerator / denominator
        return self.define(f"__stockham_rot({x}, {math.cos(angle)!r}, {math.sin(angle)!r})")

    def _odd_dft(self, values: list[str], sign: int) -> list[str]:
        # The same pairing as in the scalar code: cos(t) (x[a] + x[n-a]) + sign sin(t) i (x[a] - x[n-a])
        n = len(values)
        half = (n - 1) // 2
        sums = [self.add(values[a], values[n - a]) for a in range(1, half + 1)]
        rotated = [self.define(f"__stockham_mul_i({self.sub(values[a], values[n - a])})") for a in range(1, half + 1)]
        total = values[0]
        for term in sums:
            total = f"_mm256_add_pd({total}, {term})"
        result = [self.define(total)]
        for c in range(1, n):
            acc = values[0]
            for a in range(1, half + 1):
                cos = math.cos(2 * math.pi * c * a / n)
                sin = sign * math.sin(2 * math.pi * c * a / n)
                acc = f"_mm256_fmadd_pd(_mm256_set1_pd({cos!r}), {sums[a - 1]}, {acc})"
                acc = f"_mm256_fmadd_pd(_mm256_set1_pd({sin!r}), {rotated[a - 1]}, {acc})"
            result.append(self.define(acc))
        return result


#: Helpers of the AVX code of complex128 stages (see ``_AVXButterflyEmitter``), declared once per file.
AVX_HELPERS = """
#if defined(__AVX__) && defined(__FMA__)
#include <immintrin.h>
#ifndef DACE_STOCKHAM_AVX_HELPERS
#define DACE_STOCKHAM_AVX_HELPERS
inline __m256d __stockham_swap(__m256d x) { return _mm256_permute_pd(x, 5); }
inline __m256d __stockham_neg(__m256d x) { return _mm256_xor_pd(x, _mm256_set1_pd(-0.0)); }
inline __m256d __stockham_mul_i(__m256d x) { return _mm256_addsub_pd(_mm256_setzero_pd(), __stockham_swap(x)); }
inline __m256d __stockham_mul_minus_i(__m256d x) {
    return _mm256_xor_pd(__stockham_swap(x), _mm256_set_pd(-0.0, 0.0, -0.0, 0.0));
}
// Multiplication by the constant c + i s
inline __m256d __stockham_rot(__m256d x, double c, double s) {
    return _mm256_fmaddsub_pd(x, _mm256_set1_pd(c), _mm256_mul_pd(__stockham_swap(x), _mm256_set1_pd(s)));
}
// Complex multiplication of each of the two numbers
inline __m256d __stockham_mul(__m256d x, __m256d w) {
    return _mm256_fmaddsub_pd(x, _mm256_movedup_pd(w), _mm256_mul_pd(__stockham_swap(x), _mm256_permute_pd(w, 15)));
}
inline __m256d __stockham_load(const void *p) { return _mm256_loadu_pd(static_cast<const double *>(p)); }
inline __m256d __stockham_load2(const void *lo, const void *hi) {
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(static_cast<const double *>(lo))),
                                _mm_loadu_pd(static_cast<const double *>(hi)), 1);
}
inline __m256d __stockham_broadcast(const void *p) {
    return _mm256_broadcast_pd(static_cast<const __m128d *>(p));
}
inline void __stockham_store(void *p, __m256d x) { _mm256_storeu_pd(static_cast<double *>(p), x); }
inline void __stockham_store2(void *lo, void *hi, __m256d x) {
    _mm_storeu_pd(static_cast<double *>(lo), _mm256_castpd256_pd128(x));
    _mm_storeu_pd(static_cast<double *>(hi), _mm256_extractf128_pd(x, 1));
}
#endif
#endif
"""


def _avx_stage_code(radix: int, sign: int, twiddle: bool, factor: str | None, vector: Vector) -> str:
    """
    Generates the AVX loop of a complex128 stage that computes ``vector.length`` (even) iterations, two at a time.
    """

    def pointer(name: str, stride: str, lane: str) -> str:
        return f"{name} + ({lane}) * ({stride})"

    def load(name: str, stride: str) -> str:
        if stride == "1":
            return f"__stockham_load({name} + __lane)"
        return f"__stockham_load2({pointer(name, stride, '__lane')}, {pointer(name, stride, '__lane + 1')})"

    emitter = _AVXButterflyEmitter()
    values = []
    for a in range(radix):
        value = emitter.define(load(f"_x{a}", vector.input_stride))
        if twiddle and a > 0:
            if vector.twiddle_stride is None:
                w = f"__stockham_broadcast(&_w{a})"
            else:
                w = load(f"_w{a}", vector.twiddle_stride)
            value = emitter.define(f"__stockham_mul({value}, {w})")
        values.append(value)
    for c, y in enumerate(emitter.dft(values, sign)):
        if factor is not None:
            y = emitter.define(f"_mm256_mul_pd({y}, _mm256_set1_pd({factor}))")
        if vector.output_stride == "1":
            emitter.lines.append(f"__stockham_store(_y{c} + __lane, {y});")
        else:
            lo = pointer(f"_y{c}", vector.output_stride, "__lane")
            hi = pointer(f"_y{c}", vector.output_stride, "__lane + 1")
            emitter.lines.append(f"__stockham_store2({lo}, {hi}, {y});")
    body = "\n".join(f"    {line}" for line in emitter.lines)
    return f"for (int __lane = 0; __lane < {vector.length}; __lane += 2) {{\n{body}\n}}"


def stage_code(
    radix: int,
    sign: int,
    twiddle: bool,
    input_is_complex: bool,
    out_dtype: dtypes.typeclass,
    factor: str | None,
    vector: Vector | None = None,
) -> str:
    """
    Generates the C++ tasklet code of one Stockham stage.

    :param radix: The radix of the stage.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param twiddle: Whether inputs ``_x<a>`` (``a > 0``) are multiplied by twiddle factors read from ``_w<a>``.
    :param input_is_complex: Whether the inputs ``_x<a>`` are complex (real inputs only reach the first stage).
    :param out_dtype: The complex type of the outputs ``_y<c>``.
    :param factor: C++ expression of the factor to multiply outputs by, or None for no multiplication.
    :param vector: If given, the connectors are pointers and the tasklet computes ``vector.length`` iterations. Where
                   ``uses_avx``, the code has an AVX variant that is used if the compiler targets AVX and FMA.
    :return: The tasklet code.
    """

    def access(name: str, stride: str | None) -> str:
        if vector is None or stride is None:
            return name
        return f"{name}[__lane * ({stride})]"

    real_ctype = "float" if out_dtype == dtypes.complex64 else "double"
    emitter = _ButterflyEmitter(real_ctype)
    values = []
    for a in range(radix):
        x = access(f"_x{a}", vector and vector.input_stride)
        if input_is_complex:
            value = (emitter.define(f"{x}.real()"), emitter.define(f"{x}.imag()"))
        else:
            value = (emitter.define(x), emitter.literal(0.0))
        if twiddle and a > 0:
            w = access(f"_w{a}", vector and vector.twiddle_stride)
            cos, sin = emitter.define(f"{w}.real()"), emitter.define(f"{w}.imag()")
            value = (
                emitter.define(f"{value[0]} * {cos} - {value[1]} * {sin}"),
                emitter.define(f"{value[0]} * {sin} + {value[1]} * {cos}"),
            )
        values.append(value)

    outputs = emitter.dft(values, sign)
    for c, (real, imag) in enumerate(outputs):
        if factor is not None:
            real, imag = f"{real} * {factor}", f"{imag} * {factor}"
        y = access(f"_y{c}", vector and vector.output_stride)
        emitter.lines.append(f"{y} = {out_dtype.ctype}({real}, {imag});")
    if vector is None:
        return "\n".join(emitter.lines)
    body = "\n".join(f"    {line}" for line in emitter.lines)
    code = f"#pragma omp simd\nfor (int __lane = 0; __lane < {vector.length}; ++__lane) {{\n{body}\n}}"
    if not uses_avx(input_is_complex, out_dtype, vector):
        return code
    avx = _avx_stage_code(radix, sign, twiddle, factor, vector)
    return f"#if defined(__AVX__) && defined(__FMA__)\n{avx}\n#else\n{code}\n#endif"


def uses_avx(input_is_complex: bool, out_dtype: dtypes.typeclass, vector: Vector | None) -> bool:
    """
    Returns True if the code of a stage has an AVX variant, which needs ``AVX_HELPERS`` in the global code of the
    tasklet: for complex128 inputs and outputs, and an even number of iterations.
    """
    return vector is not None and input_is_complex and out_dtype == dtypes.complex128 and vector.length % 2 == 0


def twiddle_code(radix: int, span: int, sign: int, out_dtype: dtypes.typeclass) -> str:
    """
    Generates the C++ tasklet code that computes the twiddle factors ``w^a = exp(sign * 2 pi i a __b / (L * R))`` of a
    stage into ``_w<a>`` for ``a`` in ``[1, R)``. Only ``w`` is evaluated with trigonometric functions; its powers are
    computed by repeated multiplication. All computations are in double precision for all output types.
    """
    step = repr(sign * 2 * math.pi / (span * radix))
    lines = [
        f"const double __angle = double(__b) * {step};",
        "const double __c1 = cos(__angle), __s1 = sin(__angle);",
        "double __c = __c1, __s = __s1;",
        f"_w1 = {out_dtype.ctype}(__c, __s);",
    ]
    for a in range(2, radix):
        lines += [
            "{ const double __cn = __c * __c1 - __s * __s1; __s = __c * __s1 + __s * __c1; __c = __cn; }",
            f"_w{a} = {out_dtype.ctype}(__c, __s);",
        ]
    return "\n".join(lines)


def twiddle_table_code(radix: int, span: int, sign: int, dtype: dtypes.typeclass) -> str:
    """
    Generates C++ code that computes a whole table of a stage into ``_w``: the twiddle factors of a stage (see
    ``twiddle_code``) as a ``[R - 1, L]`` table, or the roots of unity of a general-radix stage as a ``[L * R]`` table.
    """
    step = repr(sign * 2 * math.pi / (span * radix))
    if is_general_radix(radix):
        return "\n".join(
            [
                f"for (long long __t = 0; __t < {span * radix}; ++__t) {{",
                f"    const double __angle = double(__t) * {step};",
                f"    _w[__t] = {dtype.ctype}(cos(__angle), sin(__angle));",
                "}",
            ]
        )
    return "\n".join(
        [
            f"for (long long __b = 0; __b < {span}; ++__b) {{",
            f"    const double __angle = double(__b) * {step};",
            "    const double __c1 = cos(__angle), __s1 = sin(__angle);",
            "    double __c = __c1, __s = __s1;",
            f"    _w[__b] = {dtype.ctype}(__c, __s);",
            f"    for (int __a = 2; __a < {radix}; ++__a) {{",
            "        const double __cn = __c * __c1 - __s * __s1;",
            "        __s = __c * __s1 + __s * __c1;",
            "        __c = __cn;",
            f"        _w[(__a - 1) * {span} + __b] = {dtype.ctype}(__c, __s);",
            "    }",
            "}",
        ]
    )


def general_stage_code(
    radix: int,
    span: int,
    output: str,
    input_stride: str,
    input_is_complex: bool,
    out_dtype: dtypes.typeclass,
    factor: str | None,
) -> str:
    """
    Generates the C++ tasklet code that computes one output ``_y`` of a general-radix stage: the sum over ``a`` of the
    inputs ``_x[a * input_stride]`` times the roots ``_w[(a * j) mod (L * R)]`` of a table of ``exp(sign * 2 pi i t /
    (L * R))``, which includes the twiddle factors.

    :param radix: The radix ``R`` of the stage.
    :param span: The product ``L`` of the radices of the previous stages.
    :param output: C++ expression of the output index ``j = c * L + b`` in ``[0, L * R)``.
    :param input_stride: C++ expression of the distance between consecutive inputs, in elements of the input array.
    :param input_is_complex: Whether the inputs are complex (real inputs only reach the first stage).
    :param out_dtype: The complex type of the output.
    :param factor: C++ expression of the factor to multiply the output by, or None for no multiplication.
    :return: The tasklet code.
    """
    real_ctype = "float" if out_dtype == dtypes.complex64 else "double"
    size = span * radix
    if input_is_complex:
        accumulate = [
            "__re += __x.real() * __w.real() - __x.imag() * __w.imag();",
            "__im += __x.real() * __w.imag() + __x.imag() * __w.real();",
        ]
    else:
        accumulate = ["__re += __x * __w.real();", "__im += __x * __w.imag();"]
    scale = "" if factor is None else f" * {factor}"
    lines = [
        f"const long long __j = {output};",
        f"{real_ctype} __re = 0, __im = 0;",
        "long long __m = 0;",
        f"for (int __a = 0; __a < {radix}; ++__a) {{",
        f"    const auto __x = _x[__a * ({input_stride})];",
        "    const auto __w = _w[__m];",
        *(f"    {line}" for line in accumulate),
        "    __m += __j;",
        f"    if (__m >= {size}) __m -= {size};",
        "}",
        f"_y = {out_dtype.ctype}(__re{scale}, __im{scale});",
    ]
    return "\n".join(lines)


def factor_code(factor: symbolic.SymbolicType, dtype: dtypes.typeclass) -> str | None:
    """
    Returns the C++ expression of an output factor in the real type of a complex type, or None if the factor is one.
    """
    if factor == 1:
        return None
    real_ctype = "float" if dtype == dtypes.complex64 else "double"
    return f"{real_ctype}({symbolic.symstr(factor, cpp_mode=True)})"


def add_twiddle_tables(
    sdfg: SDFG,
    state: SDFGState,
    stages: list[tuple[int, int, int, int]],
    sign: int,
    dtype: dtypes.typeclass,
    storage: dtypes.StorageType,
    lifetime: dtypes.AllocationLifetime,
    schedule: dtypes.ScheduleType,
    prefix: str = "__stockham_tw",
) -> dict[tuple[int, int], nodes.AccessNode]:
    """
    Adds a ``[R - 1, L]`` twiddle factor table for every distinct ``(L, R)`` of the stages that need twiddle factors,
    and a ``[L * R]`` table of roots of unity for every distinct ``(L, R)`` of the general-radix stages (see
    ``general_stage_code``), each computed by a map in the given state. Persistent tables in host memory are instead
    computed once, by a tasklet that only computes them in the first call of the program.

    :param sdfg: The SDFG to add the tables to.
    :param state: The state to compute the tables in.
    :param stages: The stages, as returned by ``plan_stages``.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param dtype: The complex type of the tables.
    :param storage: The storage of the tables.
    :param lifetime: The allocation lifetime of the tables.
    :param schedule: The schedule of the maps that compute the tables.
    :param prefix: Prefix of the table names.
    :return: A dictionary from ``(L, R)`` to the access node that the table is written to.
    """
    tables = {}
    for _, radix, span, _ in stages:
        general = is_general_radix(radix)
        if (span == 1 and not general) or (span, radix) in tables:
            continue
        table, _ = sdfg.add_transient(
            f"{prefix}_{span}_{radix}",
            (span * radix,) if general else (radix - 1, span),
            dtype,
            storage,
            lifetime=lifetime,
            find_new_name=True,
        )
        table_node = state.add_access(table)
        if lifetime == dtypes.AllocationLifetime.Persistent and storage in HOST_STORAGES:
            flag = f"__stockham_twiddles_ready_{next(_twiddle_flag_ids)}"
            tasklet = state.add_tasklet(
                f"stockham_twiddles_{span}_{radix}",
                {},
                {"_w"},
                f"if (!__state->{flag}) {{\n{twiddle_table_code(radix, span, sign, dtype)}\n__state->{flag} = true;\n}}",
                language=dtypes.Language.CPP,
                state_fields=[f"bool {flag};"],
                code_init=f"__state->{flag} = false;",
            )
            memlet = Memlet.from_array(table, sdfg.arrays[table])
            memlet.dynamic = True
            state.add_edge(tasklet, "_w", table_node, None, memlet)
            tables[span, radix] = table_node
            continue
        if general:
            step = repr(sign * 2 * math.pi / (span * radix))
            ranges = {"__t": f"0:{span * radix}"}
            code = f"const double __angle = double(__t) * {step};\n_w = {dtype.ctype}(cos(__angle), sin(__angle));"
            outputs = {"_w": Memlet(f"{table}[__t]")}
        else:
            ranges = {"__b": f"0:{span}"}
            code = twiddle_code(radix, span, sign, dtype)
            outputs = {f"_w{a}": Memlet(f"{table}[{a - 1}, __b]") for a in range(1, radix)}
        _, table_entry, _ = state.add_mapped_tasklet(
            f"stockham_twiddles_{span}_{radix}",
            ranges,
            {},
            code,
            outputs,
            language=dtypes.Language.CPP,
            external_edges=True,
            output_nodes={table: table_node},
            schedule=schedule,
        )
        if storage == dtypes.StorageType.GPU_Global:
            table_entry.map.gpu_block_size = GPU_BLOCK_SIZE
        tables[span, radix] = table_node
    return tables


def stockham_sdfg(
    indesc: data.Array,
    outdesc: data.Array,
    axes: list[int],
    inverse: bool,
    factor: symbolic.SymbolicType,
    constants: dict,
    schedule: dtypes.ScheduleType = dtypes.ScheduleType.Default,
    persistent: bool = False,
    name: str = "stockham_fft",
) -> SDFG:
    """
    Creates an SDFG that computes the (inverse) FFT of ``_inp`` over the given axes into ``_out``.

    On the CPU, all stages of an axis whose transforms fit into ``LOCAL_BYTES`` (see ``local_lanes``) are computed by
    one map over the other dimensions, in which each iteration transforms one row (or several adjacent columns) in
    buffers on the stack. Each other stage is one map over the whole array.

    :param indesc: Descriptor of the input array (real or complex).
    :param outdesc: Descriptor of the complex output array, with the same shape as the input.
    :param axes: The (normalized) axes to transform. Their lengths must be fixed (see ``can_expand``).
    :param inverse: Whether to compute the inverse transform.
    :param factor: Factor to multiply the outputs by (e.g., for normalization).
    :param constants: Constants of the parent SDFG, used to evaluate symbolic axis lengths.
    :param schedule: Schedule of the library node. A GPU device schedule places the buffers in GPU global memory,
                     since they would otherwise default to shared memory.
    :param persistent: Whether the buffers and twiddle tables may be allocated once per program rather than per call.
                       Only valid if the transform is not inside a map, where each iteration needs its own buffers.
    :param name: Name of the generated SDFG.
    :return: A single-state SDFG.
    """
    if not can_expand(indesc.shape, axes, constants):
        raise ValueError(f"Stockham FFT requires fixed axis lengths, got shape {indesc.shape} and axes {axes}")
    if outdesc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f"Stockham FFT output must be complex, got {outdesc.dtype}")
    if len(indesc.shape) != len(outdesc.shape):
        raise ValueError("Stockham FFT input and output must have the same number of dimensions")

    sdfg = SDFG(name)
    shape = tuple(outdesc.shape)
    dtype = outdesc.dtype
    sdfg.add_array("_inp", indesc.shape, indesc.dtype, indesc.storage, strides=indesc.strides)
    sdfg.add_array("_out", outdesc.shape, outdesc.dtype, outdesc.storage, strides=outdesc.strides)
    state = sdfg.add_state("stockham")

    stages = plan_stages(shape, axes, constants)
    gpu = schedule == dtypes.ScheduleType.GPU_Device

    storage = dtypes.StorageType.Default
    if gpu:
        storage = dtypes.StorageType.GPU_Global
    elif persistent:
        storage = dtypes.StorageType.CPU_Heap

    # Small fixed-size transforms on the CPU run sequentially
    map_schedule = dtypes.ScheduleType.Default
    total_size = fixed_length(math.prod(shape), constants)
    if not gpu and total_size is not None and total_size < SEQUENTIAL_SIZE_THRESHOLD:
        map_schedule = dtypes.ScheduleType.Sequential

    # The steps of the transform: single stages, or on the CPU all stages of an axis computed locally
    steps = []
    for stage in stages:
        lanes = None if gpu else local_lanes(shape, [stage[0]], dtype, constants)
        if lanes is not None and steps and steps[-1][0] == "local" and steps[-1][1][0][0] == stage[0]:
            steps[-1][1].append(stage)
        elif lanes is not None:
            steps.append(("local", [stage], lanes))
        else:
            steps.append(("global", [stage], None))

    # Allocate the buffers once if their size is fixed, as per-call allocations of large buffers dominate the runtime
    table_lifetime = dtypes.AllocationLifetime.Persistent if persistent else dtypes.AllocationLifetime.Scope
    lifetime = dtypes.AllocationLifetime.Scope
    if persistent and not any(symbolic.issymbolic(size, constants) for size in shape):
        lifetime = dtypes.AllocationLifetime.Persistent
    buffers = []
    for i in range(min(len(steps) - 1, 2)):
        buffer_name, _ = sdfg.add_transient(f"__stockham_buf{i}", shape, dtype, storage, lifetime=lifetime)
        buffers.append(buffer_name)

    sign = 1 if inverse else -1
    factor_cpp = factor_code(factor, dtype)
    tables = add_twiddle_tables(sdfg, state, stages, sign, dtype, storage, table_lifetime, map_schedule)
    src_node = state.add_read("_inp")
    stage_index = 0
    for s, (kind, step_stages, lanes) in enumerate(steps):
        last = s == len(steps) - 1
        dst_node = state.add_write("_out" if last else buffers[s % 2])
        if kind == "local":
            add_local_axis_kernel(
                state,
                f"stockham_axis_{s}",
                step_stages,
                sign,
                src_node,
                dst_node,
                tables,
                factor_cpp if last else None,
                lanes,
                map_schedule,
                constants,
            )
        else:
            label = f"stockham_stage_{stage_index}"
            extents = list(shape)
            entry = add_stage_map(
                state,
                label,
                step_stages[0],
                sign,
                src_node,
                dst_node,
                extents,
                {},
                tables,
                factor_cpp if last else None,
                map_schedule,
                vectorize=not gpu,
                linearize=gpu,
            )
            if gpu:
                entry.map.gpu_block_size = GPU_BLOCK_SIZE
        stage_index += len(step_stages)
        src_node = dst_node
    return sdfg


def add_stage_map(
    state: SDFGState,
    label: str,
    stage: tuple[int, int, int, int],
    sign: int,
    src_node: nodes.AccessNode,
    dst_node: nodes.AccessNode,
    extents: list[symbolic.SymbolicType],
    bases: dict[str, list[str | None]],
    tables: dict[tuple[int, int], nodes.AccessNode | str],
    factor: str | None,
    schedule: dtypes.ScheduleType,
    vectorize: bool = False,
    linearize: bool = False,
    padding: dict[str, int] | None = None,
) -> nodes.MapEntry:
    """
    Adds a map that computes one stage from ``src_node`` to ``dst_node``. Each iteration computes one butterfly (or
    one output of a general-radix stage), or with ``vectorize``, several consecutive ones in a SIMD loop.

    Map dimensions, outermost to innermost: the dimensions before the axis, b in [0, L) (or the output j in [0, L * R)
    of a general-radix stage), k in [0, M), and the dimensions after the axis (contiguous in row-major arrays, and the
    GPU thread x dimension).

    :param state: The state to add the map to.
    :param label: The label of the map.
    :param stage: The ``(axis, radix, L, M)`` of the stage, as returned by ``plan_stages``.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param src_node: The access node to read from.
    :param dst_node: The access node to write to.
    :param extents: For each dimension other than the axis, the number of indices that the map iterates over (one for
                    no iteration).
    :param bases: For arrays that are not indexed from zero, the index of each dimension other than the axis at which
                  the iterations start (or None for zero).
    :param tables: The twiddle factor tables (see ``add_twiddle_tables``), as access nodes or names, by ``(L, R)``.
    :param factor: C++ expression of a factor to multiply the outputs by, or None.
    :param schedule: The schedule of the map.
    :param vectorize: Whether a tasklet computes several consecutive iterations of the innermost dimension.
    :param linearize: Whether to iterate over b and k with one index (for GPU kernels).
    :param padding: For arrays whose axis is split into chunks of padded rows (see ``block_local_sdfg``), the number
                    of elements of a chunk.
    :return: The entry node of the map.
    """
    padding = padding or {}
    sdfg = state.sdfg
    axis, radix, span, stride = stage
    ndim = len(extents)
    src, dst = src_node.data, dst_node.data
    general = is_general_radix(radix)
    rows = span * radix if general else span

    index, map_ranges = {}, {}
    for dim in range(axis):
        if extents[dim] != 1:
            map_ranges[f"__i{dim}"] = f"0:{symbolic.symstr(extents[dim])}"
            index[dim] = f"__i{dim}"
    if linearize and rows > 1 and stride > 1:
        # One linear thread index for (b, k): either range alone may be too short to fill a thread block, and a long
        # one could exceed the grid limits of the y and z dimensions
        map_ranges["__t"] = f"0:{rows * stride}"
        row, k = f"int_floor(__t, {stride})", f"(__t % {stride})"
    else:
        if rows > 1:
            map_ranges["__b"] = f"0:{rows}"
        if stride > 1:
            map_ranges["__k"] = f"0:{stride}"
        row = "__b" if rows > 1 else "0"
        k = "__k" if stride > 1 else "0"
    for dim in range(axis + 1, ndim):
        if extents[dim] != 1:
            map_ranges[f"__i{dim}"] = f"0:{symbolic.symstr(extents[dim])}"
            index[dim] = f"__i{dim}"
    if not map_ranges:
        map_ranges["__k"] = "0:1"

    # The innermost dimension, if the tasklet computes several of its iterations
    vector_dim, lanes = None, 1
    if vectorize and not general:
        after = [dim for dim in range(axis + 1, ndim) if extents[dim] != 1]
        if after:
            candidates = [(after[-1], extents[after[-1]])]
        else:
            candidates = [("k", stride), ("b", rows)]
        for dim, extent in candidates:
            lanes = next((n for n in VECTOR_LANES if _divides(n, extent)), 1)
            if lanes > 1:
                vector_dim = dim
                param = {"b": "__b", "k": "__k"}.get(dim, f"__i{dim}")
                begin, end = map_ranges[param].split(":")
                map_ranges[param] = f"{begin}:{end}:{lanes}"
                break

    def element(dataname: str, axis_index: str, axis_step: int = 0) -> Memlet:
        """
        The element at ``axis_index`` along the axis, or the elements of the iterations of a tasklet that computes
        several, which are ``axis_step`` apart along the axis if it vectorizes over b or k.
        """
        subset = []
        chunk = padding.get(dataname)
        for dim in range(ndim):
            if dim == axis and chunk is not None:
                row, column = f"int_floor({axis_index}, {chunk})", f"(({axis_index}) % {chunk})"
                if axis_step and axis_step % chunk == 0:
                    row_step = axis_step // chunk
                    row = f"{row}:{row} + {(lanes - 1) * row_step + 1}:{row_step}"
                elif axis_step:
                    column = f"{column}:{column} + {(lanes - 1) * axis_step + 1}:{axis_step}"
                subset += [row, column]
                continue
            if dim == axis:
                if axis_step:
                    axis_index = f"{axis_index}:{axis_index} + {(lanes - 1) * axis_step + 1}:{axis_step}"
                subset.append(axis_index)
                continue
            base = bases.get(dataname, [None] * ndim)[dim]
            parts = [part for part in (base, index.get(dim)) if part is not None]
            expr = " + ".join(parts) if parts else "0"
            subset.append(f"{expr}:{expr} + {lanes}" if dim == vector_dim else expr)
        return Memlet(f"{dataname}[{', '.join(subset)}]")

    def table_name(key: tuple[int, int]) -> str:
        table = tables[key]
        return table.data if isinstance(table, nodes.AccessNode) else table

    input_is_complex = sdfg.arrays[src].dtype in (dtypes.complex64, dtypes.complex128)
    out_dtype = sdfg.arrays[dst].dtype
    if general:
        table = table_name((span, radix))
        first = f"({row} % {span}) * {radix * stride} + {k}"
        inputs = {
            "_x": element(src, f"{first}:{first} + {(radix - 1) * stride + 1}:{stride}"),
            "_w": Memlet.from_array(table, sdfg.arrays[table]),
        }
        outputs = {"_y": element(dst, f"{row} * {stride} + {k}")}
        input_stride = _cpp(stride * sdfg.arrays[src].strides[axis])
        code = general_stage_code(radix, span, _cpp(row), input_stride, input_is_complex, out_dtype, factor)
    else:
        b = row
        in_step = {"b": radix * stride, "k": 1}.get(vector_dim, 0)
        out_step = {"b": stride, "k": 1}.get(vector_dim, 0)
        inputs = {f"_x{a}": element(src, f"({b} * {radix} + {a}) * {stride} + {k}", in_step) for a in range(radix)}
        outputs = {f"_y{c}": element(dst, f"({c} * {span} + {b}) * {stride} + {k}", out_step) for c in range(radix)}
        if span > 1:
            table = table_name((span, radix))
            twiddle_index = f"{b}:{b} + {lanes}" if vector_dim == "b" else b
            inputs.update({f"_w{a}": Memlet(f"{table}[{a - 1}, {twiddle_index}]") for a in range(1, radix)})
        vector = None
        if vector_dim is not None:

            def distance(dataname: str, step: int) -> symbolic.SymbolicType:
                """
                The distance in memory between the elements of consecutive iterations of the tasklet.
                """
                strides = sdfg.arrays[dataname].strides
                chunk = padding.get(dataname)
                if vector_dim not in ("b", "k"):
                    return strides[vector_dim + 1 if chunk is not None else vector_dim]
                if chunk is None:
                    return step * strides[axis]
                if step % chunk == 0:
                    return step // chunk * strides[axis]
                return step * strides[axis + 1]

            in_stride, out_stride = distance(src, in_step), distance(dst, out_step)
            vector = Vector(lanes, _cpp(in_stride), _cpp(out_stride), "1" if vector_dim == "b" else None)
        code = stage_code(radix, sign, span > 1, input_is_complex, out_dtype, factor, vector)

    input_nodes = {src: src_node}
    for memlet in inputs.values():
        if memlet.data not in input_nodes:
            table = tables[next(key for key in tables if table_name(key) == memlet.data)]
            input_nodes[memlet.data] = table if isinstance(table, nodes.AccessNode) else state.add_read(table)
    tasklet, map_entry, _ = state.add_mapped_tasklet(
        label,
        map_ranges,
        inputs,
        code,
        outputs,
        language=dtypes.Language.CPP,
        external_edges=True,
        input_nodes=input_nodes,
        output_nodes={dst: dst_node},
        schedule=schedule,
    )
    if not general and uses_avx(input_is_complex, out_dtype, vector):
        tasklet.code_global = properties.CodeBlock(AVX_HELPERS, dtypes.Language.CPP)
    # Parallelize over all dimensions: L is small in the first stages and M is small in the last ones
    map_entry.map.collapse = len(map_ranges)
    return map_entry


def _divides(lanes: int, extent: symbolic.SymbolicType) -> bool:
    """
    Returns True if an extent is a known multiple of a number of lanes, and at least that number.
    """
    if symbolic.issymbolic(extent):
        return False
    return int(extent) >= lanes and int(extent) % lanes == 0


def local_lanes(
    shape: tuple[symbolic.SymbolicType, ...], axes: list[int], dtype: dtypes.typeclass, constants: dict
) -> int | None:
    """
    Returns how many transforms of the given axes one map iteration computes in stack buffers on the CPU (see
    ``add_local_axis_kernel``), or None if they do not fit into ``LOCAL_BYTES``. If the innermost dimension is
    transformed, an iteration computes one transform. Otherwise, it computes adjacent ones along the innermost
    dimension, so that its accesses to the array are contiguous, and that dimension must be a fixed multiple of their
    number.

    :param shape: The shape of the array.
    :param axes: The (normalized) axes.
    :param dtype: The complex type of the buffers.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :return: The number of transforms of an iteration, or None.
    """
    lengths = [fixed_length(shape[axis], constants) for axis in axes]
    if any(length is None for length in lengths) or math.prod(lengths) < 2:
        return None
    size = math.prod(lengths)
    innermost = len(shape) - 1
    if innermost in axes:
        return 1 if 2 * size * dtype.bytes <= LOCAL_BYTES else None
    extent = fixed_length(shape[innermost], constants)
    for lanes in VECTOR_LANES[:1]:
        if extent is not None and _divides(lanes, extent) and 2 * size * lanes * dtype.bytes <= LOCAL_BYTES:
            return lanes
    return None


def _cpp(expr: str | symbolic.SymbolicType) -> str:
    """
    Returns the C++ code of an index expression (e.g., ``int_floor(__t, 4)``).
    """
    if isinstance(expr, str):
        expr = symbolic.pystr_to_symbolic(expr)
    return symbolic.symstr(expr, cpp_mode=True)


def delinearize(index: str, extents: list[int]) -> list[str]:
    """
    Returns expressions of the row-major multi-dimensional index that a linear index corresponds to.

    :param index: The linear index.
    :param extents: The extents of the dimensions, outermost first.
    :return: One expression per dimension.
    """
    result = []
    inner = math.prod(extents)
    for i, extent in enumerate(extents):
        inner //= extent
        expr = index if inner == 1 else f"int_floor({index}, {inner})"
        result.append(expr if i == 0 else f"({expr} % {extent})")
    return result


def block_local_shape(
    shape: tuple[symbolic.SymbolicType, ...], axes: list[int], constants: dict, lanes: int = 1
) -> list[int]:
    """
    Returns the shape of one transform of an array: the lengths of the transformed axes, and one for the other
    dimensions, except for the innermost one with ``lanes`` adjacent transforms.
    """
    result = [fixed_length(shape[dim], constants) if dim in axes else 1 for dim in range(len(shape))]
    if lanes > 1:
        result[-1] = lanes
    return result


def fits_block_local(
    shape: tuple[symbolic.SymbolicType, ...], axes: list[int], dtype: dtypes.typeclass, constants: dict
) -> bool:
    """
    Checks whether one GPU thread block can compute a transform of the given axes of an array in shared memory, i.e.,
    whether the axis lengths are fixed and two buffers of one transform fit into ``BLOCK_LOCAL_SHARED_BYTES``.

    :param shape: The shape of the array.
    :param axes: The (normalized) axes to transform.
    :param dtype: The complex type of the transform.
    :param constants: Constants of the SDFG, used to evaluate symbolic axis lengths.
    """
    if not can_expand(shape, axes, constants):
        return False
    return 2 * math.prod(block_local_shape(shape, axes, constants)) * dtype.bytes <= BLOCK_LOCAL_SHARED_BYTES


def block_local_threads(stages: list[tuple[int, int, int, int]], shape: list[int]) -> int:
    """
    Returns the number of threads of a thread block that computes the given stages of one transform: one thread per
    butterfly (or output of a general-radix stage) of the largest stage, rounded up to full warps and capped at
    ``BLOCK_LOCAL_MAX_THREADS``.

    :param stages: The stages, as returned by ``plan_stages``.
    :param shape: The shape of one transform (one for dimensions that are not transformed).
    """
    elements = math.prod(shape)
    work = max(elements if is_general_radix(radix) else elements // radix for _, radix, _, _ in stages)
    return min(BLOCK_LOCAL_MAX_THREADS, 32 * math.ceil(work / 32))


def block_local_loop(
    state: SDFGState, label: str, work: int, threads: int
) -> tuple[nodes.MapEntry, nodes.MapExit, nodes.MapEntry, nodes.MapExit]:
    """
    Adds a ``GPU_ThreadBlock`` map of ``threads`` threads and a sequential map in it that strides the threads over
    ``work`` linear indices ``__u``. Every thread-block map of a block-local transform spans all of its threads, so
    the synchronization between states is outside of any per-thread condition.

    :return: The entry and exit nodes of the thread-block map, followed by those of the sequential map.
    """
    tb_entry, tb_exit = state.add_map(
        f"{label}_threads", {"__tid": f"0:{threads}"}, dtypes.ScheduleType.GPU_ThreadBlock
    )
    seq_entry, seq_exit = state.add_map(
        f"{label}_loop", {"__u": f"__tid:{work}:{threads}"}, dtypes.ScheduleType.Sequential
    )
    return tb_entry, tb_exit, seq_entry, seq_exit


def add_block_local_stage(
    state: SDFGState,
    stage: tuple[int, int, int, int],
    sign: int,
    src: str,
    dst: str,
    shape: list[int],
    batch_index: dict[str, list[str | None]],
    tables: dict[tuple[int, int], str],
    threads: int,
    factor: str | None = None,
) -> None:
    """
    Adds one stage of a transform that one GPU thread block computes. The threads of the block stride over the
    butterflies of the stage, so the state must be inside a kernel and the stages must be in consecutive states, which
    synchronizes the threads of the block between them.

    :param state: The (empty) state to add the stage to.
    :param stage: The ``(axis, radix, L, M)`` of the stage, as returned by ``plan_stages``.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param src: The array to read from (e.g., a global input or a shared memory buffer).
    :param dst: The array to write to.
    :param shape: The shape of one transform (one for dimensions that are not transformed).
    :param batch_index: For ``src`` and ``dst``, the index of each dimension that is not transformed (None for the
                        transformed dimensions), i.e., which transform of the array the block computes.
    :param tables: Names of the twiddle factor tables by ``(L, R)`` (see ``add_twiddle_tables``).
    :param threads: The number of threads of the block.
    :param factor: C++ expression of a factor to multiply the outputs by, or None.
    """
    sdfg = state.sdfg
    axis, radix, span, stride = stage
    general = is_general_radix(radix)

    # Linear iteration space, outermost first: transformed dimensions before the axis, b (or the output j of a
    # general-radix stage), k, and the transformed dimensions after the axis, so that consecutive threads access
    # consecutive elements where possible
    loop = [(dim, shape[dim]) for dim in range(axis) if shape[dim] > 1]
    loop += [("row", span * radix if general else span), ("k", stride)]
    loop += [(dim, shape[dim]) for dim in range(axis + 1, len(shape)) if shape[dim] > 1]
    loop = [(name, extent) for name, extent in loop if extent > 1]
    work = math.prod(extent for _, extent in loop)
    indices = dict(zip((name for name, _ in loop), delinearize("__u", [extent for _, extent in loop])))
    row, k = indices.get("row", "0"), indices.get("k", "0")

    def element(dataname: str, axis_index: str) -> Memlet:
        subset = []
        for dim in range(len(shape)):
            if dim == axis:
                subset.append(axis_index)
            elif batch_index[dataname][dim] is not None:
                subset.append(batch_index[dataname][dim])
            else:
                subset.append(indices.get(dim, "0"))
        return Memlet(f"{dataname}[{', '.join(subset)}]")

    input_is_complex = sdfg.arrays[src].dtype in (dtypes.complex64, dtypes.complex128)
    out_dtype = sdfg.arrays[dst].dtype
    if general:
        table = tables[span, radix]
        first = f"({row} % {span}) * {radix * stride} + {k}"
        inputs = {
            src: {"_x": element(src, f"{first}:{first} + {(radix - 1) * stride + 1}:{stride}")},
            table: {"_w": Memlet.from_array(table, sdfg.arrays[table])},
        }
        outputs = {"_y": element(dst, f"{row} * {stride} + {k}")}
        input_stride = _cpp(stride * sdfg.arrays[src].strides[axis])
        code = general_stage_code(radix, span, _cpp(row), input_stride, input_is_complex, out_dtype, factor)
    else:
        b = row
        inputs = {src: {f"_x{a}": element(src, f"({b} * {radix} + {a}) * {stride} + {k}") for a in range(radix)}}
        if span > 1:
            table = tables[span, radix]
            inputs[table] = {f"_w{a}": Memlet(f"{table}[{a - 1}, {b}]") for a in range(1, radix)}
        outputs = {f"_y{c}": element(dst, f"({c} * {span} + {b}) * {stride} + {k}") for c in range(radix)}
        code = stage_code(radix, sign, span > 1, input_is_complex, out_dtype, factor)

    tasklet = state.add_tasklet(
        f"stockham_butterfly_{radix}",
        dict.fromkeys(conn for memlets in inputs.values() for conn in memlets),
        dict.fromkeys(outputs),
        code,
        language=dtypes.Language.CPP,
    )
    tb_entry, tb_exit, seq_entry, seq_exit = block_local_loop(state, f"stockham_block_{state.label}", work, threads)
    for dataname, memlets in inputs.items():
        access = state.add_read(dataname)
        state.add_edge(access, None, tb_entry, f"IN_{dataname}", Memlet.from_array(dataname, sdfg.arrays[dataname]))
        state.add_edge(
            tb_entry, f"OUT_{dataname}", seq_entry, f"IN_{dataname}", Memlet.from_array(dataname, sdfg.arrays[dataname])
        )
        tb_entry.add_in_connector(f"IN_{dataname}")
        tb_entry.add_out_connector(f"OUT_{dataname}")
        seq_entry.add_in_connector(f"IN_{dataname}")
        seq_entry.add_out_connector(f"OUT_{dataname}")
        for conn, memlet in memlets.items():
            state.add_edge(seq_entry, f"OUT_{dataname}", tasklet, conn, memlet)
    access = state.add_write(dst)
    for conn, memlet in outputs.items():
        state.add_edge(tasklet, conn, seq_exit, f"IN_{dst}", memlet)
    state.add_edge(seq_exit, f"OUT_{dst}", tb_exit, f"IN_{dst}", Memlet.from_array(dst, sdfg.arrays[dst]))
    state.add_edge(tb_exit, f"OUT_{dst}", access, None, Memlet.from_array(dst, sdfg.arrays[dst]))
    seq_exit.add_in_connector(f"IN_{dst}")
    seq_exit.add_out_connector(f"OUT_{dst}")
    tb_exit.add_in_connector(f"IN_{dst}")
    tb_exit.add_out_connector(f"OUT_{dst}")


def block_local_sdfg(
    name: str,
    arrays: dict[str, data.Data],
    shape: tuple[symbolic.SymbolicType, ...],
    axes: list[int],
    dtype: dtypes.typeclass,
    constants: dict,
    num_buffers: int = 2,
    storage: dtypes.StorageType = dtypes.StorageType.GPU_Shared,
    lanes: int = 1,
    padding: int = 0,
) -> tuple[SDFG, dict[int, str], dict[str, list[str | None]], list[str]]:
    """
    Creates the SDFG that each thread block of a block-local kernel runs (see ``add_block_local_kernel``), without
    states. It accesses the given arrays of the parent, which the kernel passes whole, has one symbol per dimension
    that is not transformed (the transform of the block), and has shared memory buffers for one transform.

    :param name: The name of the SDFG.
    :param arrays: The arrays of the parent SDFG that the SDFG accesses, by name.
    :param shape: The shape of the transformed arrays.
    :param axes: The (normalized) transformed axes. Their lengths must be fixed.
    :param dtype: The complex type of the buffers.
    :param constants: Constants of the parent SDFG, used to evaluate symbolic axis lengths.
    :param num_buffers: The number of shared memory buffers (at most two are needed).
    :param storage: The storage of the buffers.
    :param lanes: The number of adjacent transforms along the innermost dimension (not transformed) that a block
                  computes. Its symbol is the first of them.
    :param padding: If nonzero, the axis of the buffers (of a single transformed axis) is split into rows of this
                    many elements, which are padded by one element, so that accesses with power-of-two strides do
                    not map to the same cache sets (see ``local_padding``).
    :return: The SDFG, the symbol of each dimension that is not transformed, the batch index of each array (see
             ``add_block_local_stage``), and the names of the buffers.
    """
    block = SDFG(name)
    ndim = len(shape)
    batch_symbols = {dim: f"__fft_batch{dim}" for dim in range(ndim) if dim not in axes}
    for symbol in batch_symbols.values():
        block.add_symbol(symbol, dtypes.int64)
    for array_name, desc in sorted(arrays.items()):
        desc = copy.deepcopy(desc)
        desc.transient = False
        desc.lifetime = dtypes.AllocationLifetime.Scope
        block.add_datadesc(array_name, desc)
    block_shape = block_local_shape(shape, axes, constants, lanes)
    buffer_shape, strides, total_size = block_shape, None, None
    if padding:
        (axis,) = axes
        inner = math.prod(block_shape[axis + 1 :])
        rows = block_shape[axis] // padding
        buffer_shape = block_shape[:axis] + [rows, padding] + block_shape[axis + 1 :]
        total_size = rows * (padding * inner + 1)
        after = [math.prod(block_shape[dim + 1 :]) for dim in range(axis + 1, ndim)]
        strides = [total_size] * axis + [padding * inner + 1, inner] + after
    buffers = [
        block.add_transient(
            f"__fft_block_buf{i}",
            buffer_shape,
            dtype,
            storage,
            strides=strides,
            total_size=total_size,
            find_new_name=True,
        )[0]
        for i in range(num_buffers)
    ]
    global_index = [batch_symbols.get(dim) for dim in range(ndim)]
    shared_index = ["0" if dim in batch_symbols and block_shape[dim] == 1 else None for dim in range(ndim)]
    batch_index = dict.fromkeys(arrays, global_index)
    batch_index.update(dict.fromkeys(buffers, shared_index))
    return block, batch_symbols, batch_index, buffers


def add_block_local_steps(
    block: SDFG,
    steps: list[tuple[tuple[int, int, int, int], int, dict[tuple[int, int], str], str | None] | None],
    src: str,
    dst: str,
    buffers: list[str],
    shape: list[int],
    batch_index: dict[str, list[str | None]],
    threads: int | None,
    padding: int = 0,
) -> list[tuple[SDFGState, str, str]]:
    """
    Adds one state per step to the SDFG of a thread block (see ``block_local_sdfg``), in order. The first step reads
    ``src``, the last one writes ``dst``, and the steps in between alternate between the buffers. On the CPU, the
    stages are sequential maps that compute several butterflies per iteration (see ``add_stage_map``).

    :param block: The SDFG of a thread block, without states.
    :param steps: The steps, each a stage as ``(stage, sign, tables, factor)`` (see ``add_block_local_stage``), or
                  None for a state that the caller fills (e.g., with a computation from one buffer to the other).
    :param src: The array that the first step reads.
    :param dst: The array that the last step writes.
    :param buffers: The shared memory buffers.
    :param shape: The shape of one transform (one for dimensions that are not transformed).
    :param batch_index: The batch index of each array (see ``add_block_local_stage``).
    :param threads: The number of threads of the block, or None for a CPU map iteration.
    :param padding: The number of elements of a padded row of the buffers (see ``block_local_sdfg``), or 0.
    :return: For each step, its state and the arrays it reads and writes.
    """
    locations = [src] + [buffers[i % 2] for i in range(len(steps) - 1)] + [dst]
    result = []
    prev_state = None
    for i, step in enumerate(steps):
        if prev_state is None:
            state = block.add_state(f"step_{i}", is_start_block=True)
        else:
            state = block.add_state_after(prev_state, f"step_{i}")
        prev_state = state
        if step is not None and threads is None:
            stage, sign, tables, factor = step
            add_stage_map(
                state,
                f"{block.name}_stage_{i}",
                stage,
                sign,
                state.add_read(locations[i]),
                state.add_write(locations[i + 1]),
                shape,
                batch_index,
                tables,
                factor,
                dtypes.ScheduleType.Sequential,
                vectorize=True,
                padding=dict.fromkeys(buffers, padding) if padding else None,
            )
        elif step is not None:
            stage, sign, tables, factor = step
            add_block_local_stage(
                state, stage, sign, locations[i], locations[i + 1], shape, batch_index, tables, threads, factor
            )
        result.append((state, locations[i], locations[i + 1]))
    return result


def add_block_local_kernel(
    state: SDFGState,
    block: SDFG,
    inputs: dict[str, nodes.AccessNode],
    output: nodes.AccessNode,
    shape: tuple[symbolic.SymbolicType, ...],
    batch_symbols: dict[int, str],
    schedule: dtypes.ScheduleType = dtypes.ScheduleType.GPU_Device,
    lanes: int = 1,
    constants: dict | None = None,
) -> tuple[nodes.MapEntry, nodes.NestedSDFG]:
    """
    Adds a GPU kernel that runs the SDFG of a thread block (see ``block_local_sdfg``) with one thread block per
    transform, i.e., a ``GPU_Device`` map over the dimensions that are not transformed. With another schedule, it
    adds a map with that schedule, in which each iteration runs the SDFG, or a sequential one for a single iteration.

    :param state: The state to add the kernel to.
    :param block: The SDFG of a thread block, with all of its states.
    :param inputs: The access nodes of the arrays that the block reads, by name.
    :param output: The access node of the array that the block writes.
    :param shape: The shape of the transformed arrays.
    :param batch_symbols: The symbol of each dimension that is not transformed.
    :param schedule: The schedule of the map.
    :param lanes: The number of adjacent transforms along the innermost dimension that an iteration computes.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :return: The entry node of the kernel map and the nested SDFG node.
    """
    sdfg = state.sdfg
    if schedule != dtypes.ScheduleType.GPU_Device:
        # A single iteration does not need a parallel region
        if fixed_length(math.prod(shape[dim] for dim in batch_symbols) / lanes, constants or {}) == 1:
            schedule = dtypes.ScheduleType.Sequential
    ranges = {symbol: f"0:{symbolic.symstr(shape[dim])}" for dim, symbol in sorted(batch_symbols.items())}
    if lanes > 1:
        innermost = batch_symbols[len(shape) - 1]
        ranges[innermost] += f":{lanes}"
    kernel_entry, kernel_exit = state.add_map(block.name, ranges or {"__fft_block": "0:1"}, schedule)
    if schedule != dtypes.ScheduleType.GPU_Device:
        kernel_entry.map.collapse = len(ranges) or 1
    for symbol in sorted(block.free_symbols - set(batch_symbols.values()) - set(block.symbols)):
        block.add_symbol(symbol, sdfg.symbols.get(symbol, dtypes.int64))
    nsdfg = state.add_nested_sdfg(
        block,
        dict.fromkeys(sorted(inputs)),
        dict.fromkeys([output.data]),
        {symbol: symbol for symbol in block.symbols},
        name=block.name,
    )
    for name in sorted(inputs):
        state.add_memlet_path(
            inputs[name], kernel_entry, nsdfg, dst_conn=name, memlet=Memlet.from_array(name, sdfg.arrays[name])
        )
    state.add_memlet_path(
        nsdfg,
        kernel_exit,
        output,
        src_conn=output.data,
        memlet=Memlet.from_array(output.data, sdfg.arrays[output.data]),
    )
    propagation.propagate_memlets_sdfg(block)
    propagation.propagate_memlets_state(sdfg, state)
    return kernel_entry, nsdfg


def local_padding(stages: list[tuple[int, int, int, int]]) -> int:
    """
    Returns the number of elements after which the stack buffers of a transform of one axis on the CPU (see
    ``add_local_axis_kernel``) are padded by one element, or 0 if they are not padded. The stages access the buffers
    with power-of-two strides that would otherwise map to the same cache sets. The rows must keep the iterations of
    each SIMD tasklet together: those of k (at most ``VECTOR_LANES[0]`` consecutive elements) within a row, and those
    of b in the last stage (``R`` elements apart) at the same position of rows.

    :param stages: All stages of the axis, in order.
    :return: The number of elements of a padded row, or 0.
    """
    if any(is_general_radix(radix) for _, radix, _, _ in stages):
        return 0
    length = math.prod(radix for _, radix, _, _ in stages)
    for chunk in (16, 8):
        if chunk >= VECTOR_LANES[0] and length % chunk == 0 and stages[-1][1] % chunk == 0:
            return chunk
    return 0


def add_local_axis_kernel(
    state: SDFGState,
    label: str,
    stages: list[tuple[int, int, int, int]],
    sign: int,
    src_node: nodes.AccessNode,
    dst_node: nodes.AccessNode,
    tables: dict[tuple[int, int], nodes.AccessNode],
    factor: str | None,
    lanes: int,
    schedule: dtypes.ScheduleType,
    constants: dict,
) -> nodes.MapEntry:
    """
    Adds a CPU map over the dimensions other than an axis, in which each iteration computes all stages of the axis
    for ``lanes`` adjacent transforms (see ``local_lanes``) in buffers on the stack, so that the array is read and
    written once rather than once per stage. Each iteration runs a nested SDFG with one state per stage, in which the
    stages are sequential maps (see ``add_stage_map``).

    :param state: The state to add the map to.
    :param label: The label of the map and of the nested SDFG.
    :param stages: All stages of the axis, in order.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param src_node: The access node to read from.
    :param dst_node: The access node to write to.
    :param tables: The twiddle factor tables (see ``add_twiddle_tables``).
    :param factor: C++ expression of a factor to multiply the outputs of the last stage by, or None.
    :param lanes: The number of adjacent transforms of an iteration.
    :param schedule: The schedule of the map.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :return: The entry node of the map.
    """
    sdfg = state.sdfg
    src, dst = src_node.data, dst_node.data
    shape = tuple(sdfg.arrays[dst].shape)
    dtype = sdfg.arrays[dst].dtype
    axis = stages[0][0]
    keys = {(span, radix) for _, radix, span, _ in stages}
    stage_tables = {key: node for key, node in tables.items() if key in keys}
    table_names = {key: node.data for key, node in stage_tables.items()}
    arrays = {name: sdfg.arrays[name] for name in (src, dst, *table_names.values())}
    padding = local_padding(stages)
    block, batch_symbols, batch_index, buffers = block_local_sdfg(
        label,
        arrays,
        shape,
        [axis],
        dtype,
        constants,
        min(len(stages) - 1, 2),
        dtypes.StorageType.Register(force=True),
        lanes,
        padding,
    )
    block_shape = block_local_shape(shape, [axis], constants, lanes)
    steps = [(stage, sign, table_names, factor if i == len(stages) - 1 else None) for i, stage in enumerate(stages)]
    add_block_local_steps(block, steps, src, dst, buffers, block_shape, batch_index, None, padding)
    inputs = {src: src_node, **{node.data: node for node in stage_tables.values()}}
    entry, _ = add_block_local_kernel(state, block, inputs, dst_node, shape, batch_symbols, schedule, lanes, constants)
    return entry


def block_local_fft_sdfg(
    indesc: data.Array,
    outdesc: data.Array,
    axes: list[int],
    inverse: bool,
    factor: symbolic.SymbolicType,
    constants: dict,
    name: str = "stockham_fft_block_local",
) -> SDFG:
    """
    Creates an SDFG that computes the (inverse) FFT of ``_inp`` over the given axes into ``_out``, both in GPU global
    memory, as one kernel in which each thread block computes one transform in shared memory. The twiddle factor
    tables are allocated once and computed per call by small kernels before it, so the SDFG must not be inside a map.

    :param indesc: Descriptor of the input array (real or complex).
    :param outdesc: Descriptor of the complex output array, with the same shape as the input.
    :param axes: The (normalized) axes to transform. A transform must fit into shared memory (see
                 ``fits_block_local``).
    :param inverse: Whether to compute the inverse transform.
    :param factor: Factor to multiply the outputs by (e.g., for normalization).
    :param constants: Constants of the parent SDFG, used to evaluate symbolic axis lengths.
    :param name: Name of the generated SDFG.
    :return: A single-state SDFG with the table maps and the kernel.
    """
    shape = tuple(outdesc.shape)
    if not fits_block_local(shape, axes, outdesc.dtype, constants):
        raise ValueError(f"A transform of axes {axes} of shape {shape} does not fit into the shared memory of a block")
    if outdesc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f"Stockham FFT output must be complex, got {outdesc.dtype}")
    if len(indesc.shape) != len(outdesc.shape):
        raise ValueError("Stockham FFT input and output must have the same number of dimensions")

    sdfg = SDFG(name)
    sdfg.add_array("_inp", indesc.shape, indesc.dtype, indesc.storage, strides=indesc.strides)
    sdfg.add_array("_out", outdesc.shape, outdesc.dtype, outdesc.storage, strides=outdesc.strides)
    state = sdfg.add_state("stockham")

    dtype = outdesc.dtype
    sign = 1 if inverse else -1
    block_shape = block_local_shape(shape, axes, constants)
    stages = plan_stages(block_shape, axes, {})
    threads = block_local_threads(stages, block_shape)
    table_nodes = add_twiddle_tables(
        sdfg,
        state,
        stages,
        sign,
        dtype,
        dtypes.StorageType.GPU_Global,
        dtypes.AllocationLifetime.Persistent,
        dtypes.ScheduleType.GPU_Device,
    )
    tables = {key: node.data for key, node in table_nodes.items()}
    arrays = {array: sdfg.arrays[array] for array in ("_inp", "_out", *tables.values())}
    block, batch_symbols, batch_index, buffers = block_local_sdfg(
        f"{name}_block", arrays, shape, axes, dtype, constants, min(len(stages) - 1, 2)
    )
    factor_cpp = factor_code(factor, dtype)
    steps = [(stage, sign, tables, factor_cpp if s == len(stages) - 1 else None) for s, stage in enumerate(stages)]
    add_block_local_steps(block, steps, "_inp", "_out", buffers, block_shape, batch_index, threads)

    inputs = {"_inp": state.add_read("_inp"), **{node.data: node for node in table_nodes.values()}}
    add_block_local_kernel(state, block, inputs, state.add_write("_out"), shape, batch_symbols)
    return sdfg
