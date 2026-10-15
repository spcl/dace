# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Stockham autosort Fast Fourier Transform (FFT) as a native SDFG.

The transform of a length-``N`` axis is factored into radices ``R_0 * R_1 * ... = N``. Stage ``s`` is a single map
over the batch dimensions (all other axes), ``b`` in ``[0, L)`` and ``k`` in ``[0, M)``, where ``L`` is the product of
the radices of the previous stages and ``M = N / (L * R_s)``. Each map iteration reads ``R_s`` elements at
``(b * R_s + a) * M + k``, multiplies them by the twiddle factors ``exp(+-2 pi i a b / (L * R_s))``, computes a
radix-``R_s`` DFT in registers, and writes the results to ``(c * L + b) * M + k`` in another buffer. All stages of all
transformed axes are chained through two ping-pong buffers in one state, so the expansion can be inlined and fused
with its surroundings. The twiddle factors of each stage are computed once per call into a small ``[L, R - 1]``
table by a map in the same state.

The radices and all stage sizes are fixed at expansion time, so the lengths of the transformed axes must be known
integers. Batch dimensions may be symbolic.
"""

import math

from dace import SDFG, Memlet, data, dtypes, symbolic

#: Radices that butterflies are generated for. Powers of two are split recursively inside the butterfly and odd
#: primes use a direct DFT, so other prime factors would need a different algorithm.
POWER_OF_TWO_RADICES = (16, 8, 4, 2)
PRIME_RADICES = (3, 5, 7, 11, 13)

#: Arrays with fewer elements than this are transformed by sequential maps on the CPU, where the work of a stage is
#: too small to amortize starting a parallel region.
SEQUENTIAL_SIZE_THRESHOLD = 2048

#: Thread block size of the GPU kernels (64-256 threads perform the same on an RTX 3070).
GPU_BLOCK_SIZE = [128, 1, 1]


def factorize(n: int) -> list[int] | None:
    """
    Splits a transform length into the radices of the Stockham stages.

    :param n: The transform length.
    :return: The list of radices whose product is ``n`` (empty for ``n == 1``), or None if ``n`` has a prime factor
             with no butterfly.
    """
    if n < 1:
        return None
    radices = []
    for radix in POWER_OF_TWO_RADICES + PRIME_RADICES:
        while n % radix == 0:
            radices.append(radix)
            n //= radix
    return radices if n == 1 else None


def fixed_length(length: symbolic.SymbolicType, constants: dict) -> int | None:
    """
    Returns the integer value of an axis length, or None if it depends on symbols that are not constants.
    """
    if symbolic.issymbolic(length, constants):
        return None
    return int(symbolic.evaluate(length, constants))


def can_expand(shape: tuple[symbolic.SymbolicType, ...], axes: list[int], constants: dict) -> bool:
    """
    Checks whether the Stockham expansion can transform the given axes of an array.

    :param shape: The shape of the array.
    :param axes: The (normalized) axes to transform.
    :param constants: Constants of the SDFG, used to evaluate symbolic axis lengths.
    :return: True if every transformed axis has a fixed length with supported prime factors.
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


def stage_code(
    radix: int,
    sign: int,
    twiddle: bool,
    input_is_complex: bool,
    out_dtype: dtypes.typeclass,
    factor: str | None,
) -> str:
    """
    Generates the C++ tasklet code of one Stockham stage.

    :param radix: The radix of the stage.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param twiddle: Whether inputs ``_x<a>`` (``a > 0``) are multiplied by twiddle factors read from ``_w<a>``.
    :param input_is_complex: Whether the inputs ``_x<a>`` are complex (real inputs only reach the first stage).
    :param out_dtype: The complex type of the outputs ``_y<c>``.
    :param factor: C++ expression of the factor to multiply outputs by, or None for no multiplication.
    :return: The tasklet code.
    """
    real_ctype = "float" if out_dtype == dtypes.complex64 else "double"
    emitter = _ButterflyEmitter(real_ctype)
    values = []
    for a in range(radix):
        if input_is_complex:
            value = (emitter.define(f"_x{a}.real()"), emitter.define(f"_x{a}.imag()"))
        else:
            value = (emitter.define(f"_x{a}"), emitter.literal(0.0))
        if twiddle and a > 0:
            cos, sin = emitter.define(f"_w{a}.real()"), emitter.define(f"_w{a}.imag()")
            value = (
                emitter.define(f"{value[0]} * {cos} - {value[1]} * {sin}"),
                emitter.define(f"{value[0]} * {sin} + {value[1]} * {cos}"),
            )
        values.append(value)

    outputs = emitter.dft(values, sign)
    for c, (real, imag) in enumerate(outputs):
        if factor is not None:
            real, imag = f"{real} * {factor}", f"{imag} * {factor}"
        emitter.lines.append(f"_y{c} = {out_dtype.ctype}({real}, {imag});")
    return "\n".join(emitter.lines)


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
    :return: A single-state SDFG with one map per stage.
    """
    if not can_expand(indesc.shape, axes, constants):
        raise ValueError(
            f"Stockham FFT requires fixed axis lengths with prime factors in {(2,) + PRIME_RADICES}, "
            f"got shape {indesc.shape} and axes {axes}"
        )
    if outdesc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise TypeError(f"Stockham FFT output must be complex, got {outdesc.dtype}")
    if len(indesc.shape) != len(outdesc.shape):
        raise ValueError("Stockham FFT input and output must have the same number of dimensions")

    sdfg = SDFG(name)
    shape = tuple(outdesc.shape)
    sdfg.add_array("_inp", indesc.shape, indesc.dtype, indesc.storage, strides=indesc.strides)
    sdfg.add_array("_out", outdesc.shape, outdesc.dtype, outdesc.storage, strides=outdesc.strides)
    state = sdfg.add_state("stockham")

    # Stages: (axis, radix, L, M)
    stages = []
    for axis in axes:
        length = fixed_length(shape[axis], constants)
        span = 1
        for radix in factorize(length):
            stages.append((axis, radix, span, length // (span * radix)))
            span *= radix
    if not stages:  # Every transformed axis has length one
        stages.append((axes[0], 1, 1, 1))

    storage = dtypes.StorageType.Default
    if schedule == dtypes.ScheduleType.GPU_Device:
        storage = dtypes.StorageType.GPU_Global
    elif persistent:
        storage = dtypes.StorageType.CPU_Heap

    # Allocate the buffers once if their size is fixed, as per-call allocations of large buffers dominate the runtime
    table_lifetime = dtypes.AllocationLifetime.Persistent if persistent else dtypes.AllocationLifetime.Scope
    lifetime = dtypes.AllocationLifetime.Scope
    if persistent and not any(symbolic.issymbolic(size, constants) for size in shape):
        lifetime = dtypes.AllocationLifetime.Persistent

    buffers = []
    for i in range(min(len(stages) - 1, 2)):
        buffer_name, _ = sdfg.add_transient(f"__stockham_buf{i}", shape, outdesc.dtype, storage, lifetime=lifetime)
        buffers.append(buffer_name)

    factor_cpp = None
    if factor != 1:
        real_ctype = "float" if outdesc.dtype == dtypes.complex64 else "double"
        factor_cpp = f"{real_ctype}({symbolic.symstr(factor, cpp_mode=True)})"

    # Small fixed-size transforms on the CPU run sequentially
    map_schedule = dtypes.ScheduleType.Default
    total_size = fixed_length(math.prod(shape), constants)
    if (
        schedule in (dtypes.ScheduleType.Default, dtypes.ScheduleType.CPU_Multicore)
        and total_size is not None
        and total_size < SEQUENTIAL_SIZE_THRESHOLD
    ):
        map_schedule = dtypes.ScheduleType.Sequential

    sign = 1 if inverse else -1
    src_node = state.add_read("_inp")

    # Twiddle factor tables [L, R - 1], one per distinct (L, R), computed by a map before the first stage using them
    twiddle_nodes = {}
    for _, radix, span, _ in stages:
        if span == 1 or (span, radix) in twiddle_nodes:
            continue
        table, _ = sdfg.add_transient(
            f"__stockham_tw_{span}_{radix}",
            (span, radix - 1),
            outdesc.dtype,
            storage,
            lifetime=table_lifetime,
        )
        table_node = state.add_access(table)
        _, table_entry, _ = state.add_mapped_tasklet(
            f"stockham_twiddles_{span}_{radix}",
            {"__b": f"0:{span}"},
            {},
            twiddle_code(radix, span, sign, outdesc.dtype),
            {f"_w{a}": Memlet(f"{table}[__b, {a - 1}]") for a in range(1, radix)},
            language=dtypes.Language.CPP,
            external_edges=True,
            output_nodes={table: table_node},
            schedule=map_schedule,
        )
        if storage == dtypes.StorageType.GPU_Global:
            table_entry.map.gpu_block_size = GPU_BLOCK_SIZE
        twiddle_nodes[span, radix] = table_node
    for s, (axis, radix, span, stride) in enumerate(stages):
        last = s == len(stages) - 1
        dst = "_out" if last else buffers[s % 2]
        dst_node = state.add_write(dst)

        # Map dimensions, outermost to innermost: batch dimensions before the axis, b in [0, L), k in [0, M), and
        # batch dimensions after the axis (contiguous in row-major arrays, and the GPU thread x dimension)
        index = [f"__i{dim}" for dim in range(len(shape))]
        map_ranges = {f"__i{dim}": f"0:{symbolic.symstr(shape[dim])}" for dim in range(axis)}
        if storage == dtypes.StorageType.GPU_Global and span > 1 and stride > 1:
            # One linear thread index for (b, k): either range alone may be too short to fill a thread block, and
            # a long one could exceed the grid limits of the y and z dimensions
            map_ranges["__t"] = f"0:{span * stride}"
            b, k = f"int_floor(__t, {stride})", f"(__t % {stride})"
        else:
            if span > 1:
                map_ranges["__b"] = f"0:{span}"
            if stride > 1:
                map_ranges["__k"] = f"0:{stride}"
            b = "__b" if span > 1 else "0"
            k = "__k" if stride > 1 else "0"
        map_ranges.update({f"__i{dim}": f"0:{symbolic.symstr(shape[dim])}" for dim in range(axis + 1, len(shape))})
        if not map_ranges:
            map_ranges["__k"] = "0:1"

        def element(dataname: str, axis_index: str) -> Memlet:
            subset = ", ".join(axis_index if dim == axis else index[dim] for dim in range(len(shape)))
            return Memlet(f"{dataname}[{subset}]")

        src = src_node.data
        inputs = {f"_x{a}": element(src, f"({b} * {radix} + {a}) * {stride} + {k}") for a in range(radix)}
        outputs = {f"_y{c}": element(dst, f"({c} * {span} + {b}) * {stride} + {k}") for c in range(radix)}
        input_nodes = {src: src_node}
        if span > 1:
            table_node = twiddle_nodes[span, radix]
            inputs.update({f"_w{a}": Memlet(f"{table_node.data}[{b}, {a - 1}]") for a in range(1, radix)})
            input_nodes[table_node.data] = table_node
        code = stage_code(
            radix,
            sign,
            span > 1,
            input_is_complex=sdfg.arrays[src].dtype in (dtypes.complex64, dtypes.complex128),
            out_dtype=outdesc.dtype,
            factor=factor_cpp if last else None,
        )
        _, map_entry, _ = state.add_mapped_tasklet(
            f"stockham_stage_{s}",
            map_ranges,
            inputs,
            code,
            outputs,
            language=dtypes.Language.CPP,
            external_edges=True,
            input_nodes=input_nodes,
            output_nodes={dst: dst_node},
            schedule=map_schedule,
        )
        # Parallelize over all dimensions: L is small in the first stages and M is small in the last ones
        map_entry.map.collapse = len(map_ranges)
        if storage == dtypes.StorageType.GPU_Global:
            map_entry.map.gpu_block_size = GPU_BLOCK_SIZE
        src_node = dst_node

    return sdfg
