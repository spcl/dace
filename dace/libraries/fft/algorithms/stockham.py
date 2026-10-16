# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Stockham autosort Fast Fourier Transform (FFT) as a native SDFG.

The transform of a length-``N`` axis is factored into radices ``R_0 * R_1 * ... = N``. Stage ``s`` is a single map
over the batch dimensions (all other axes), ``b`` in ``[0, L)`` and ``k`` in ``[0, M)``, where ``L`` is the product of
the radices of the previous stages and ``M = N / (L * R_s)``. Each map iteration reads ``R_s`` elements at
``(b * R_s + a) * M + k``, multiplies them by the twiddle factors ``exp(+-2 pi i a b / (L * R_s))``, computes a
radix-``R_s`` DFT in registers, and writes the results to ``(c * L + b) * M + k`` in another buffer. All stages of all
transformed axes are chained through two ping-pong buffers in one state, so the expansion can be inlined. The
twiddle factors of each stage are in a ``[R - 1, L]`` table, which a cache in the program state computes when the
program first requests it and shares between all transforms that need it (see ``dace_stockham.h`` and ``add_table``).

Each butterfly is a Python tasklet with explicit real arithmetic, and the maps are laid out so that their innermost
dimension has affine accesses, which compilers vectorize. On the CPU, all stages of an axis whose transforms fit into
``LOCAL_BYTES`` are one map over the other dimensions, in which each iteration transforms a row (or a block of
columns) in buffers on the stack (see ``add_local_axis_kernel``), so that the array is read and written once rather
than once per stage. Longer axes are split into groups of stages that fit (see ``group_lengths``), each of which is
one such map. On the GPU, the stage maps compute one butterfly per thread and fuse with elementwise maps.

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
import math

from dace import SDFG, Memlet, SDFGState, data, dtypes, nodes, symbolic
from dace.codegen import common
from dace.libraries.fft.environments import stockham as stockham_environment
from dace.sdfg import propagation

#: Radices that butterflies are generated for. Powers of two are split recursively inside the butterfly and odd
#: primes use a direct DFT. Other prime factors use general-radix stages.
POWER_OF_TWO_RADICES = (16, 8, 4, 2)
PRIME_RADICES = (3, 5, 7, 11, 13)

#: Arrays with fewer elements than this are transformed by sequential maps on the CPU, where the work of a stage is
#: too small to amortize starting a parallel region.
SEQUENTIAL_SIZE_THRESHOLD = 2048

#: Number of adjacent transforms that a CPU map iteration of a local axis kernel computes if the innermost dimension
#: is not transformed (see ``local_lanes``).
LOCAL_LANES = 8

#: Storage types of host memory.
HOST_STORAGES = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)

#: Stack memory of the two buffers in which a CPU map iteration may transform an axis (see ``local_lanes``).
LOCAL_BYTES = 128 * 1024

#: Number of adjacent transforms that a CPU map iteration of a group of stages computes (see ``axis_groups``).
GROUP_LANES = 8

#: Thread block size of the GPU kernels (64-256 threads perform the same on an RTX 3070).
GPU_BLOCK_SIZE = [128, 1, 1]

#: Maximum number of threads of a thread block that computes a transform in shared memory. Larger transforms loop.
BLOCK_LOCAL_MAX_THREADS = 256


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
    Generates straight-line Python tasklet code for small DFTs on values held in pairs of real variables.
    """

    def __init__(self, real_ctype: str):
        self.real_ctype = real_ctype
        self.lines: list[str] = []
        self.counter = 0

    def literal(self, value: float) -> str:
        text = repr(float(value))
        return f"dace.float32({text})" if self.real_ctype == "float" else text

    def define(self, expr: str) -> str:
        name = f"__v{self.counter}"
        self.counter += 1
        self.lines.append(f"{name} = {expr}")
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
    span: int,
    b: str,
    sign: int,
    input_is_complex: bool,
    out_dtype: dtypes.typeclass,
    factor: symbolic.SymbolicType | None,
    input_parts: bool = False,
    output_parts: bool = False,
) -> str:
    """
    Generates the Python tasklet code of one butterfly of a Stockham stage, which reads inputs ``_x<a>`` and writes
    outputs ``_y<c>``, or their real and imaginary parts (``_x<a>_re`` and ``_x<a>_im``) where the complex numbers
    are stored as pairs of real numbers (see ``add_real_axis_kernel``).

    :param radix: The radix ``R`` of the stage.
    :param span: The product ``L`` of the radices of the previous stages. If it is more than one, the inputs ``_x<a>``
                 (``a > 0``) are multiplied by twiddle factors read from the stage's table ``_tw`` (see
                 ``StockhamTwiddles::Stage`` in ``dace_stockham.h``).
    :param b: Python expression of the index of the butterfly in ``[0, L)``.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param input_is_complex: Whether the inputs are complex (real inputs only reach the first stage).
    :param out_dtype: The complex type of the outputs.
    :param factor: The factor to multiply outputs by, or None for no multiplication.
    :param input_parts: Whether the inputs are connected as their real and imaginary parts.
    :param output_parts: Whether the outputs are connected as their real and imaginary parts.
    :return: The tasklet code.
    """
    emitter = _ButterflyEmitter("float" if out_dtype == dtypes.complex64 else "double")
    values = []
    for a in range(radix):
        if input_parts:
            value = (f"_x{a}_re", f"_x{a}_im")
        elif input_is_complex:
            value = (emitter.define(f"_x{a}.real"), emitter.define(f"_x{a}.imag"))
        else:
            value = (emitter.define(f"_x{a}"), emitter.literal(0.0))
        if span > 1 and a > 0:
            w = emitter.define(f"_tw[{(a - 1) * span} + {b}]")
            cos, sin = emitter.define(f"{w}.real"), emitter.define(f"{w}.imag")
            value = (
                emitter.define(f"{value[0]} * {cos} - {value[1]} * {sin}"),
                emitter.define(f"{value[0]} * {sin} + {value[1]} * {cos}"),
            )
        values.append(value)

    complex_type = f"dace.{out_dtype.to_string()}"
    scale = "" if factor is None else f" * {factor_code(factor, out_dtype)}"
    for c, (real, imag) in enumerate(emitter.dft(values, sign)):
        real, imag = f"{real}{scale}", f"{imag}{scale}"
        if output_parts:
            emitter.lines += [f"_y{c}_re = {real}", f"_y{c}_im = {imag}"]
        else:
            emitter.lines.append(f"_y{c} = {complex_type}({real}, {imag})")
    return "\n".join(emitter.lines)


def general_stage_code(
    radix: int,
    span: int,
    output: str,
    input_is_complex: bool,
    out_dtype: dtypes.typeclass,
    factor: symbolic.SymbolicType | None,
) -> str:
    """
    Generates the Python tasklet code that computes one output ``_y`` of a general-radix stage: the sum over ``a`` of
    the inputs ``_x[a]`` times the roots ``_w[(a * j) mod (L * R)]`` of a table of ``exp(sign * 2 pi i t / (L * R))``,
    which includes the twiddle factors.

    :param radix: The radix ``R`` of the stage.
    :param span: The product ``L`` of the radices of the previous stages.
    :param output: Python expression of the output index ``j = c * L + b`` in ``[0, L * R)``.
    :param input_is_complex: Whether the inputs are complex (real inputs only reach the first stage).
    :param out_dtype: The complex type of the output.
    :param factor: The factor to multiply the output by, or None for no multiplication.
    :return: The tasklet code.
    """
    real_type = "dace.float32" if out_dtype == dtypes.complex64 else "dace.float64"
    size = span * radix
    if input_is_complex:
        accumulate = [
            "__re += _x[__a].real * __w.real - _x[__a].imag * __w.imag",
            "__im += _x[__a].real * __w.imag + _x[__a].imag * __w.real",
        ]
    else:
        accumulate = ["__re += _x[__a] * __w.real", "__im += _x[__a] * __w.imag"]
    scale = "" if factor is None else f" * {factor_code(factor, out_dtype)}"
    lines = [
        f"__j = {output}",
        f"__re = {real_type}(0)",
        f"__im = {real_type}(0)",
        "__m = 0",
        f"for __a in range({radix}):",
        "    __w = _w[__m]",
        *(f"    {line}" for line in accumulate),
        "    __m += __j",
        f"    if __m >= {size}:",
        f"        __m -= {size}",
        f"_y = dace.{out_dtype.to_string()}(__re{scale}, __im{scale})",
    ]
    return "\n".join(lines)


def complex_type(dtype: dtypes.typeclass) -> dtypes.typeclass:
    """
    Returns the complex type of a real or complex type.
    """
    return {dtypes.float32: dtypes.complex64, dtypes.float64: dtypes.complex128}.get(dtype, dtype)


def factor_code(factor: symbolic.SymbolicType, dtype: dtypes.typeclass) -> str:
    """
    Returns the Python tasklet expression of an output factor in the real type of a complex type.
    """
    real_type = "dace.float32" if dtype == dtypes.complex64 else "dace.float64"
    return f"{real_type}({symbolic.symstr(factor)})"


def add_table(state: SDFGState, name: str, dtype: dtypes.typeclass, call: str) -> nodes.AccessNode:
    """
    Adds a pointer to a twiddle factor table of the program's cache (see ``dace_stockham.h``), which a tasklet gets
    from the cache, computing the table in the first call of the program.

    :param state: The state to add the pointer and the tasklet to.
    :param name: The name of the pointer.
    :param dtype: The complex type of the table.
    :param call: The call of the ``dace::fft::StockhamTwiddles`` method that returns the table.
    :return: The access node of the pointer.
    """
    sdfg = state.sdfg
    name, _ = sdfg.add_scalar(
        name, dtypes.pointer(dtype), dtypes.StorageType.Register, transient=True, find_new_name=True
    )
    tasklet = state.add_tasklet(
        f"get{name}",
        {},
        {"_table": dtypes.pointer(dtype)},
        f"_table = __state->stockham_twiddles.{call};",
        language=dtypes.Language.CPP,
    )
    tasklet.environments = {stockham_environment.Stockham.full_class_path()}
    node = state.add_access(name)
    state.add_edge(tasklet, "_table", node, None, Memlet(f"{name}[0]"))
    return node


def add_twiddle_tables(
    state: SDFGState,
    stages: list[tuple[int, int, int, int]],
    sign: int,
    dtype: dtypes.typeclass,
    device: bool,
    prefix: str = "__stockham_tw",
) -> dict[tuple[int, int], nodes.AccessNode]:
    """
    Adds a pointer to the ``[R - 1, L]`` twiddle factor table of every distinct ``(L, R)`` of the stages that need
    twiddle factors, and to the ``L * R`` roots of unity of every distinct ``(L, R)`` of the general-radix stages (see
    ``general_stage_code``), from the program's cache of tables (see ``add_table``).

    :param state: The state to add the pointers to.
    :param stages: The stages, as returned by ``plan_stages``.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param dtype: The complex type of the tables.
    :param device: Whether the tables are in GPU memory.
    :param prefix: Prefix of the pointer names.
    :return: A dictionary from ``(L, R)`` to the access node of the pointer.
    """
    tables = {}
    device_arg = "true" if device else "false"
    for _, radix, span, _ in stages:
        general = is_general_radix(radix)
        if (span == 1 and not general) or (span, radix) in tables:
            continue
        if general:
            call = f"Roots<{dtype.ctype}>({span * radix}, {sign}, {device_arg})"
        else:
            call = f"Stage<{dtype.ctype}>({radix}, {span}, {sign}, {device_arg})"
        tables[span, radix] = add_table(state, f"{prefix}_{span}_{radix}", dtype, call)
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
    buffers on the stack. A longer axis is split into groups of stages that fit, each computed by one such map (see
    ``add_axis_group``). Each other stage is one map over the whole array.

    :param indesc: Descriptor of the input array (real or complex).
    :param outdesc: Descriptor of the complex output array, with the same shape as the input.
    :param axes: The (normalized) axes to transform. Their lengths must be fixed (see ``can_expand``).
    :param inverse: Whether to compute the inverse transform.
    :param factor: Factor to multiply the outputs by (e.g., for normalization).
    :param constants: Constants of the parent SDFG, used to evaluate symbolic axis lengths.
    :param schedule: Schedule of the library node. A GPU device schedule places the buffers in GPU global memory,
                     since they would otherwise default to shared memory.
    :param persistent: Whether the buffers may be allocated once per program rather than per call.
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
    sdfg.add_array("_inp", indesc.shape, indesc.dtype, indesc.storage, strides=indesc.strides)
    sdfg.add_array("_out", outdesc.shape, outdesc.dtype, outdesc.storage, strides=outdesc.strides)
    state = sdfg.add_state("stockham")
    add_stockham(
        state, state.add_read("_inp"), state.add_write("_out"), axes, inverse, factor, constants, schedule, persistent
    )
    return sdfg


def add_stockham(
    state: SDFGState,
    src_node: nodes.AccessNode,
    dst_node: nodes.AccessNode,
    axes: list[int],
    inverse: bool,
    factor: symbolic.SymbolicType,
    constants: dict,
    schedule: dtypes.ScheduleType = dtypes.ScheduleType.Default,
    persistent: bool = False,
    prefix: str = "stockham",
) -> None:
    """
    Adds the (inverse) FFT of ``src_node`` over the given axes into ``dst_node`` to a state (see ``stockham_sdfg``).

    :param state: The state to add the transform to.
    :param src_node: The access node of the input array (real or complex).
    :param dst_node: The access node of the complex output array, with the same shape as the input.
    :param axes: The (normalized) axes to transform. Their lengths must be fixed (see ``can_expand``).
    :param inverse: Whether to compute the inverse transform.
    :param factor: Factor to multiply the outputs by (e.g., for normalization).
    :param constants: Constants of the SDFG, used to evaluate symbolic axis lengths.
    :param schedule: Schedule of the transform (see ``stockham_sdfg``).
    :param persistent: Whether the buffers may be allocated once per program (see ``stockham_sdfg``).
    :param prefix: Prefix of the labels of the maps and kernels, which must be unique in the SDFG.
    """
    sdfg = state.sdfg
    outdesc = sdfg.arrays[dst_node.data]
    shape = tuple(outdesc.shape)
    dtype = outdesc.dtype
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

    # The steps of the transform: single stages, or on the CPU all stages of an axis (or of a group of its stages)
    # computed locally
    steps, table_stages = [], []
    for axis in dict.fromkeys(stage[0] for stage in stages):
        axis_stages = [stage for stage in stages if stage[0] == axis]
        lanes = None if gpu else local_lanes(shape, [axis], dtype, constants)
        groups = None if gpu or lanes is not None else axis_groups(shape, axis, dtype, constants)
        if lanes is not None:
            steps.append(("local", axis_stages, lanes))
        elif groups is not None:
            steps += [("group", (axis, group), None) for group in groups]
            table_stages += [stage for _, length, _ in groups for stage in plan_stages((length,), [0], {})]
            continue
        else:
            steps += [("global", [stage], None) for stage in axis_stages]
        table_stages += axis_stages

    # Allocate the buffers once if their size is fixed, as per-call allocations of large buffers dominate the runtime
    lifetime = dtypes.AllocationLifetime.Scope
    if persistent and not any(symbolic.issymbolic(size, constants) for size in shape):
        lifetime = dtypes.AllocationLifetime.Persistent
    buffers = []
    for i in range(min(len(steps) - 1, 2)):
        buffer_name, _ = sdfg.add_transient(
            f"__stockham_buf{i}", shape, dtype, storage, lifetime=lifetime, find_new_name=True
        )
        buffers.append(buffer_name)

    sign = 1 if inverse else -1
    out_factor = None if factor == 1 else factor
    tables = add_twiddle_tables(state, table_stages, sign, dtype, gpu)
    group_tables = {}
    stage_index = 0
    final_node = dst_node
    for s, (kind, step_stages, lanes) in enumerate(steps):
        last = s == len(steps) - 1
        dst_node = final_node if last else state.add_access(buffers[s % 2])
        if kind == "group":
            add_axis_group(
                state,
                f"{prefix}_group_{s}",
                *step_stages,
                sign,
                src_node,
                dst_node,
                tables,
                group_tables,
                out_factor if last else None,
                map_schedule,
                constants,
            )
            src_node = dst_node
            continue
        if kind == "local":
            add_local_axis_kernel(
                state,
                f"{prefix}_axis_{s}",
                step_stages,
                sign,
                src_node,
                dst_node,
                tables,
                out_factor if last else None,
                lanes,
                map_schedule,
                constants,
            )
        else:
            label = f"{prefix}_stage_{stage_index}"
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
                out_factor if last else None,
                map_schedule,
                linearize=gpu,
            )
            if gpu:
                entry.map.gpu_block_size = GPU_BLOCK_SIZE
        stage_index += len(step_stages)
        src_node = dst_node


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
    factor: symbolic.SymbolicType | None,
    schedule: dtypes.ScheduleType,
    linearize: bool = False,
    padding: dict[str, int] | None = None,
    pairs: set[str] | None = None,
) -> nodes.MapEntry:
    """
    Adds a map that computes one stage from ``src_node`` to ``dst_node``. Each iteration computes one butterfly (or
    one output of a general-radix stage) in a Python tasklet.

    Map dimensions, outermost to innermost: the dimensions before the axis, b in [0, L) (or the output j in [0, L * R)
    of a general-radix stage), k in [0, M), and the dimensions after the axis (contiguous in row-major arrays, and the
    GPU thread x dimension). The accesses are affine in the innermost dimension, so that compilers can vectorize it.

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
    :param factor: A factor to multiply the outputs by, or None.
    :param schedule: The schedule of the map.
    :param linearize: Whether to iterate over b and k with one index (for GPU kernels).
    :param padding: For arrays whose axis is split into chunks of padded rows (see ``block_local_sdfg``), the number
                    of elements of a chunk.
    :param pairs: Real arrays that store the complex elements of the stage as pairs of consecutive real numbers along
                  the axis (see ``add_real_axis_kernel``).
    :return: The entry node of the map.
    """
    padding = padding or {}
    pairs = pairs or set()
    sdfg = state.sdfg
    axis, radix, span, stride = stage
    ndim = len(extents)
    src, dst = src_node.data, dst_node.data
    general = is_general_radix(radix)
    rows = span * radix if general else span

    # If k spans several padded rows of the buffers, the map iterates over the rows and over the elements of a row in
    # separate dimensions, so that the accesses are affine in the innermost one
    chunks = set(padding.values())
    chunk = None
    if not general and len(chunks) == 1:
        chunk = next(iter(chunks))
        if stride <= chunk or stride % chunk != 0:
            chunk = None

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
        if chunk is not None:
            map_ranges["__kr"] = f"0:{stride // chunk}"
            map_ranges["__kc"] = f"0:{chunk}"
            k = f"(__kr * {chunk} + __kc)"
        elif stride > 1:
            map_ranges["__k"] = f"0:{stride}"
            k = "__k"
        else:
            k = "0"
        row = "__b" if rows > 1 else "0"
    for dim in range(axis + 1, ndim):
        if extents[dim] != 1:
            map_ranges[f"__i{dim}"] = f"0:{symbolic.symstr(extents[dim])}"
            index[dim] = f"__i{dim}"
    if not map_ranges:
        map_ranges["__k"] = "0:1"

    def padded(chunk: int, major: str, minor: str | int, scale: int) -> list[str] | None:
        """
        The row and the column in rows of ``chunk`` elements of the index ``(major * scale + minor) * M + k``, as
        expressions that are affine in the indices of the map, or None if they are not.
        """
        if stride % chunk == 0:
            if "__kr" in map_ranges:
                return [f"(({major}) * {scale} + {minor}) * {stride // chunk} + __kr", "__kc"]
            return [f"(({major}) * {scale} + {minor}) * {stride // chunk}", k]
        if stride == 1 and isinstance(minor, int) and scale % chunk == 0:
            return [f"({major}) * {scale // chunk} + {minor // chunk}", str(minor % chunk)]
        return None

    def element(
        dataname: str, axis_index: str, terms: tuple[str, str | int, int] | None = None, part: int | None = None
    ) -> Memlet:
        """
        The element at ``axis_index`` along the axis. ``terms`` are the ``(major, minor, scale)`` of ``axis_index =
        (major * scale + minor) * M + k`` (see ``padded``), if known. In an array of pairs, ``part`` selects the real
        or the imaginary part.
        """
        subset = []
        chunk = padding.get(dataname)
        for dim in range(ndim):
            if dim == axis and part is not None:
                subset.append(f"2 * ({axis_index}) + {part}")
                continue
            if dim == axis and chunk is not None:
                exact = padded(chunk, *terms) if terms is not None else None
                subset += exact or [f"int_floor({axis_index}, {chunk})", f"(({axis_index}) % {chunk})"]
                continue
            if dim == axis:
                subset.append(axis_index)
                continue
            base = bases.get(dataname, [None] * ndim)[dim]
            parts = [part for part in (base, index.get(dim)) if part is not None]
            subset.append(" + ".join(parts) if parts else "0")
        return Memlet(f"{dataname}[{', '.join(subset)}]")

    def table_name(key: tuple[int, int]) -> str:
        table = tables[key]
        return table.data if isinstance(table, nodes.AccessNode) else table

    input_is_complex = sdfg.arrays[src].dtype in (dtypes.complex64, dtypes.complex128)
    out_dtype = sdfg.arrays[dst].dtype
    if general and (src in pairs or dst in pairs):
        raise NotImplementedError("General-radix stages of arrays of pairs are not supported")
    if general:
        table = table_name((span, radix))
        first = f"({row} % {span}) * {radix * stride} + {k}"
        inputs = {
            "_x": element(src, f"{first}:{first} + {(radix - 1) * stride + 1}:{stride}"),
            "_w": Memlet(f"{table}[0]"),
        }
        outputs = {"_y": element(dst, f"{row} * {stride} + {k}")}
        code = general_stage_code(radix, span, row, input_is_complex, out_dtype, factor)
    else:

        def elements(conn: str, dataname: str, axis_index: str, terms: tuple) -> dict[str, Memlet]:
            if dataname in pairs:
                return {f"{conn}_{name}": element(dataname, axis_index, part=p) for p, name in enumerate(("re", "im"))}
            return {conn: element(dataname, axis_index, terms)}

        b = row
        inputs, outputs = {}, {}
        for a in range(radix):
            inputs.update(elements(f"_x{a}", src, f"({b} * {radix} + {a}) * {stride} + {k}", (b, a, radix)))
        for c in range(radix):
            outputs.update(elements(f"_y{c}", dst, f"({c} * {span} + {b}) * {stride} + {k}", (str(c), b, span)))
        if span > 1:
            inputs["_tw"] = Memlet(f"{table_name((span, radix))}[0]")
        input_is_complex = input_is_complex or src in pairs
        code = stage_code(
            radix, span, b, sign, input_is_complex, complex_type(out_dtype), factor, src in pairs, dst in pairs
        )

    input_nodes = {src: src_node}
    for memlet in inputs.values():
        if memlet.data not in input_nodes:
            table = tables[next(key for key in tables if table_name(key) == memlet.data)]
            input_nodes[memlet.data] = table if isinstance(table, nodes.AccessNode) else state.add_read(table)
    _, map_entry, _ = state.add_mapped_tasklet(
        label,
        map_ranges,
        inputs,
        code,
        outputs,
        external_edges=True,
        input_nodes=input_nodes,
        output_nodes={dst: dst_node},
        schedule=schedule,
    )
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
    shape: tuple[symbolic.SymbolicType, ...],
    axes: list[int],
    dtype: dtypes.typeclass,
    constants: dict,
    lanes: int = LOCAL_LANES,
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
    :param lanes: The number of adjacent transforms of an iteration if the innermost dimension is not transformed.
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
    if extent is not None and _divides(lanes, extent) and 2 * size * lanes * dtype.bytes <= LOCAL_BYTES:
        return lanes
    return None


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
    whether the axis lengths are fixed and two buffers of one transform fit into the shared memory of a thread block
    (see ``compiler.cuda.max_shared_memory``). Code generation moves buffers that exceed the static limit to dynamic
    shared memory.

    :param shape: The shape of the array.
    :param axes: The (normalized) axes to transform.
    :param dtype: The complex type of the transform.
    :param constants: Constants of the SDFG, used to evaluate symbolic axis lengths.
    """
    if not can_expand(shape, axes, constants):
        return False
    return 2 * math.prod(block_local_shape(shape, axes, constants)) * dtype.bytes <= common.gpu_max_shared_memory()


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
    factor: symbolic.SymbolicType | None = None,
    pairs: set[str] | None = None,
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
    :param factor: A factor to multiply the outputs by, or None.
    :param pairs: Real arrays that store the complex elements as pairs of real numbers (see ``add_stage_map``).
    """
    sdfg = state.sdfg
    pairs = pairs or set()
    axis, radix, span, stride = stage
    general = is_general_radix(radix)
    if general and (src in pairs or dst in pairs):
        raise NotImplementedError("General-radix stages of arrays of pairs are not supported")

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

    def element(dataname: str, axis_index: str, part: int | None = None) -> Memlet:
        subset = []
        for dim in range(len(shape)):
            if dim == axis:
                subset.append(axis_index if part is None else f"2 * ({axis_index}) + {part}")
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
            table: {"_w": Memlet(f"{table}[0]")},
        }
        outputs = {"_y": element(dst, f"{row} * {stride} + {k}")}
        code = general_stage_code(radix, span, row, input_is_complex, out_dtype, factor)
    else:

        def elements(conn: str, dataname: str, axis_index: str) -> dict[str, Memlet]:
            if dataname in pairs:
                return {f"{conn}_{name}": element(dataname, axis_index, p) for p, name in enumerate(("re", "im"))}
            return {conn: element(dataname, axis_index)}

        b = row
        inputs = {src: {}}
        for a in range(radix):
            inputs[src].update(elements(f"_x{a}", src, f"({b} * {radix} + {a}) * {stride} + {k}"))
        if span > 1:
            table = tables[span, radix]
            inputs[table] = {"_tw": Memlet(f"{table}[0]")}
        outputs = {}
        for c in range(radix):
            outputs.update(elements(f"_y{c}", dst, f"({c} * {span} + {b}) * {stride} + {k}"))
        input_is_complex = input_is_complex or src in pairs
        code = stage_code(
            radix, span, b, sign, input_is_complex, complex_type(out_dtype), factor, src in pairs, dst in pairs
        )

    add_block_local_tasklet(state, f"stockham_butterfly_{radix}", work, threads, inputs, dst, outputs, code)


def add_block_local_tasklet(
    state: SDFGState,
    label: str,
    work: int,
    threads: int,
    inputs: dict[str, dict[str, Memlet]],
    dst: str,
    outputs: dict[str, Memlet],
    code: str,
) -> None:
    """
    Adds a Python tasklet that the threads of a GPU thread block compute for ``work`` linear indices ``__u`` (see
    ``block_local_loop``).

    :param state: The (empty) state to add the tasklet to.
    :param label: The label of the tasklet.
    :param work: The number of indices.
    :param threads: The number of threads of the block.
    :param inputs: The memlets of the input connectors, by array.
    :param dst: The array that the tasklet writes.
    :param outputs: The memlets of the output connectors (all of ``dst``).
    :param code: The code of the tasklet.
    """
    sdfg = state.sdfg
    tasklet = state.add_tasklet(
        label, dict.fromkeys(conn for memlets in inputs.values() for conn in memlets), dict.fromkeys(outputs), code
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
    rows: bool = False,
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
    :param rows: Whether the SDFG accesses the arrays through views of the elements of its transforms (see
                 ``add_row_views``), so that the indices of its accesses do not depend on the transform.
    :return: The SDFG, the symbol of each dimension that is not transformed, the batch index of each array (see
             ``add_block_local_stage``), and the names of the buffers.
    """
    block = SDFG(name)
    ndim = len(shape)
    batch_symbols = {dim: f"__fft_batch{dim}" for dim in range(ndim) if dim not in axes}
    for symbol in batch_symbols.values():
        block.add_symbol(symbol, dtypes.int64)
    block_shape = block_local_shape(shape, axes, constants, lanes)
    row_arrays = set()
    for array_name, desc in sorted(arrays.items()):
        desc = desc.as_array() if isinstance(desc, data.View) else copy.deepcopy(desc)
        desc.transient = False
        desc.lifetime = dtypes.AllocationLifetime.Scope
        if rows and isinstance(desc, data.Array) and len(desc.shape) == ndim:
            row_arrays.add(array_name)
        block.add_datadesc(array_name, desc)
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
    row_index = ["0" if dim in batch_symbols else None for dim in range(ndim)]
    batch_index = dict.fromkeys(arrays, global_index)
    batch_index.update(dict.fromkeys(row_arrays, row_index))
    batch_index.update(dict.fromkeys(buffers, shared_index))
    return block, batch_symbols, batch_index, buffers


def add_row_views(block: SDFG, names: list[str], batch_symbols: dict[int, str], block_shape: list[int]) -> None:
    """
    Makes the states of the SDFG of a CPU map iteration (see ``block_local_sdfg`` with ``rows``) access arrays through
    views of the elements of its transforms. The code then computes their address once, rather than in every access,
    which lets compilers vectorize the accesses.

    :param block: The SDFG, with all of its states.
    :param names: The arrays to access through views.
    :param batch_symbols: The symbol of each dimension that is not transformed.
    :param block_shape: The shape of the transforms of an iteration. The transformed dimensions of a view have the
                        extents of its array (e.g., an array of real pairs, see ``add_real_axis_kernel``).
    """
    for name in names:
        desc = block.arrays[name]
        shape = [block_shape[dim] if dim in batch_symbols else desc.shape[dim] for dim in range(len(block_shape))]
        view, _ = block.add_view(f"{name}_row", shape, desc.dtype, strides=desc.strides, find_new_name=True)
        subset = []
        for dim, extent in enumerate(shape):
            first = batch_symbols.get(dim, "0")
            subset.append(f"{first}:{first} + {extent}")
        row = Memlet(f"{name}[{', '.join(subset)}]")
        for state in block.states():
            for node in [n for n in state.data_nodes() if n.data == name]:
                for edge in state.all_edges(node):
                    for path_edge in state.memlet_tree(edge):
                        if path_edge.data.data == name:
                            path_edge.data.data = view
                node.data = view
                array = state.add_access(name)
                if state.out_degree(node) > 0:
                    state.add_edge(array, None, node, "views", copy.deepcopy(row))
                else:
                    state.add_edge(node, "views", array, None, copy.deepcopy(row))


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
    pairs: set[str] | None = None,
) -> list[tuple[SDFGState, str, str]]:
    """
    Adds one state per step to the SDFG of a thread block (see ``block_local_sdfg``), in order. The first step reads
    ``src``, the last one writes ``dst``, and the steps in between alternate between the buffers. On the CPU, the
    stages are sequential maps (see ``add_stage_map``).

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
    :param pairs: Real arrays that store the complex elements as pairs of real numbers (see ``add_stage_map``).
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
                padding=dict.fromkeys(buffers, padding) if padding else None,
                pairs=pairs,
            )
        elif step is not None:
            stage, sign, tables, factor = step
            add_block_local_stage(
                state, stage, sign, locations[i], locations[i + 1], shape, batch_index, tables, threads, factor, pairs
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
    with power-of-two strides that would otherwise map to the same cache sets. The radix of the last stage must be a
    multiple of the length of a row, so that its accesses are affine in b (see ``add_stage_map``).

    :param stages: All stages of the axis, in order.
    :return: The number of elements of a padded row, or 0.
    """
    if any(is_general_radix(radix) for _, radix, _, _ in stages):
        return 0
    length = math.prod(radix for _, radix, _, _ in stages)
    for chunk in (16, 8):
        if length % chunk == 0 and stages[-1][1] % chunk == 0:
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
    factor: symbolic.SymbolicType | None,
    lanes: int,
    schedule: dtypes.ScheduleType,
    constants: dict,
    twiddle: "GroupTwiddle | None" = None,
) -> nodes.MapEntry:
    """
    Adds a CPU map over the dimensions other than an axis, in which each iteration computes all stages of the axis
    for ``lanes`` adjacent transforms (see ``local_lanes``) in buffers on the stack, so that the array is read and
    written once rather than once per stage. Each iteration runs a nested SDFG with one state per stage, in which the
    stages are sequential maps (see ``add_stage_map``), preceded by a state that multiplies the inputs by the twiddle
    factors of a group of stages, if given.

    :param state: The state to add the map to.
    :param label: The label of the map and of the nested SDFG.
    :param stages: All stages of the axis, in order.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param src_node: The access node to read from.
    :param dst_node: The access node to write to.
    :param tables: The twiddle factor tables (see ``add_twiddle_tables``).
    :param factor: A factor to multiply the outputs of the last stage by, or None.
    :param lanes: The number of adjacent transforms of an iteration.
    :param schedule: The schedule of the map.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :param twiddle: The twiddle factors of a group of stages (see ``add_axis_group``), or None.
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
    inputs = {src: src_node, **{node.data: node for node in stage_tables.values()}}
    if twiddle is not None:
        inputs.update({node.data: node for node in (twiddle.low, twiddle.high, twiddle.lane) if node is not None})
    arrays = {name: sdfg.arrays[name] for name in (dst, *inputs)}
    steps = [(stage, sign, table_names, factor if i == len(stages) - 1 else None) for i, stage in enumerate(stages)]
    if twiddle is not None:
        steps.insert(0, None)
    padding = local_padding(stages)
    block, batch_symbols, batch_index, buffers = block_local_sdfg(
        label,
        arrays,
        shape,
        [axis],
        dtype,
        constants,
        min(len(steps) - 1, 2),
        dtypes.StorageType.Register(force=True),
        lanes,
        padding,
        rows=True,
    )
    block_shape = block_local_shape(shape, [axis], constants, lanes)
    block_states = add_block_local_steps(block, steps, src, dst, buffers, block_shape, batch_index, None, padding)
    if twiddle is not None:
        twiddle_state, step_src, step_dst = block_states[0]
        _add_group_twiddle_step(
            twiddle_state, step_src, step_dst, block_shape, axis, batch_index, batch_symbols, padding, twiddle
        )
    add_row_views(block, [src, dst], batch_symbols, block_shape)
    entry, _ = add_block_local_kernel(state, block, inputs, dst_node, shape, batch_symbols, schedule, lanes, constants)
    return entry


def _prime_factors(n: int) -> list[int]:
    """
    Returns the prime factors of a positive integer, with multiplicity, in ascending order.
    """
    factors, p = [], 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    return factors + ([n] if n > 1 else [])


def group_lengths(length: int, max_length: int) -> list[int] | None:
    """
    Splits the length of an axis into the fewest (at least two) factors of at most ``max_length``, as equal as
    possible. Each factor is the length of a group of stages that one map computes locally (see ``add_axis_group``).

    :param length: The length of the axis.
    :param max_length: The largest length of a group.
    :return: The lengths of the groups, in order, or None if a prime factor is larger than ``max_length``.
    """
    primes = sorted(_prime_factors(length), reverse=True)
    if len(primes) < 2 or primes[0] > max_length:
        return None
    for count in range(2, len(primes) + 1):
        # Assign each prime, largest first, to the group with the smallest product
        groups = [1] * count
        for prime in primes:
            smallest = min(range(count), key=lambda g: groups[g])
            groups[smallest] *= prime
        if max(groups) <= max_length:
            return sorted(groups, reverse=True)
    return None


def _group_view(
    shape: tuple[symbolic.SymbolicType, ...], axis: int, group: tuple[int, int, int]
) -> list[tuple[int | str, symbolic.SymbolicType]]:
    """
    Returns the dimensions of the view of an array in which a group of stages ``(L, P, M)`` transforms one dimension:
    the dimensions of the array, with the axis replaced by ``b`` in ``[0, L)``, ``a`` in ``[0, P)``, and ``k`` in
    ``[0, M)``, of which those of length one are omitted. If ``a`` would be innermost, ``b`` is moved after it, so
    that adjacent iterations of ``b`` write adjacent outputs.

    :return: For each dimension of the view, the dimension of the array (or ``"b"``, ``"a"``, ``"k"``) and its extent.
    """
    span, length, stride = group
    dims = [(dim, shape[dim]) for dim in range(axis)] + [("b", span), ("a", length), ("k", stride)]
    dims += [(dim, shape[dim]) for dim in range(axis + 1, len(shape))]
    dims = [(dim, extent) for dim, extent in dims if dim not in ("b", "k") or extent != 1]
    if dims[-1][0] == "a" and span > 1:
        dims.remove(("b", span))
        dims.append(("b", span))
    return dims


def _group_strides(
    desc: data.Array,
    axis: int,
    dims: list[tuple[int | str, symbolic.SymbolicType]],
    group: tuple[int, int, int],
    output: bool,
) -> list[symbolic.SymbolicType]:
    """
    Returns the strides of the view of an array (see ``_group_view``): a group reads its input at ``(b * P + a) * M +
    k`` and writes its output at ``(a * L + b) * M + k`` along the axis.
    """
    span, length, stride = group
    unit = desc.strides[axis]
    if output:
        strides = {"b": stride * unit, "a": span * stride * unit, "k": unit}
    else:
        strides = {"b": length * stride * unit, "a": stride * unit, "k": unit}
    return [strides[dim] if isinstance(dim, str) else desc.strides[dim] for dim, _ in dims]


def axis_groups(
    shape: tuple[symbolic.SymbolicType, ...], axis: int, dtype: dtypes.typeclass, constants: dict
) -> list[tuple[int, int, int]] | None:
    """
    Returns the groups of stages that transform an axis on the CPU if its transforms do not fit into ``LOCAL_BYTES``
    whole (see ``local_lanes``), or None if one map per stage transforms it. Each group is ``(L, P, M)``: one map
    computes a length-``P`` transform of span ``L`` and stride ``M`` (see ``add_axis_group``) per iteration, for
    adjacent transforms along the innermost dimension of its view.

    :param shape: The shape of the array.
    :param axis: The axis.
    :param dtype: The complex type of the buffers.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :return: The groups, in order, or None.
    """
    length = fixed_length(shape[axis], constants)
    if length is None:
        return None
    lengths = group_lengths(length, LOCAL_BYTES // (2 * GROUP_LANES * dtype.bytes))
    if lengths is None:
        return None
    groups, span = [], 1
    for group_length in lengths:
        groups.append((span, group_length, length // (span * group_length)))
        span *= group_length
    for group in groups:
        dims = _group_view(shape, axis, group)
        view_axis = next(i for i, (dim, _) in enumerate(dims) if dim == "a")
        if local_lanes(tuple(extent for _, extent in dims), [view_axis], dtype, constants, GROUP_LANES) is None:
            return None
    return groups


@dataclasses.dataclass
class GroupTwiddle:
    """
    The twiddle factors ``exp(sign 2 pi i t / Q)`` with ``t = a * b mod Q`` that a group of stages multiplies its
    inputs by (see ``add_axis_group``), as the product of ``low[t % S]`` and ``high[t / S]``. If the adjacent
    transforms of an iteration are along ``b``, those of ``b + l`` are additionally multiplied by ``lane[a, l]``,
    the twiddle factor of ``t = a * l``.
    """

    #: The dimension of ``b`` in the view of the group.
    dim: int
    #: ``Q``, the span of the group times its length.
    modulus: int
    #: ``S``, the length of the ``low`` table.
    split: int
    low: nodes.AccessNode
    high: nodes.AccessNode
    lane: nodes.AccessNode | None = None


def add_group_twiddle_tables(
    state: SDFGState,
    modulus: int,
    sign: int,
    dtype: dtypes.typeclass,
    lane_shape: tuple[int, int] | None = None,
) -> tuple[nodes.AccessNode, nodes.AccessNode, int, nodes.AccessNode | None]:
    """
    Adds pointers to the tables of the twiddle factors of a group of stages (see ``GroupTwiddle``), of which ``low``
    and ``high`` have about ``sqrt(Q)`` entries each, from the program's cache of tables (see ``add_table``).

    :param lane_shape: The length of the group and the number of adjacent transforms along ``b`` of an iteration,
                       if it needs a ``lane`` table.
    :return: The access nodes of the ``low`` and ``high`` pointers, the length of the ``low`` table, and the access
             node of the ``lane`` pointer or None.
    """
    split = math.isqrt(modulus - 1) + 1
    ctype = dtype.ctype
    prefix = f"__stockham_gtw_{modulus}"
    low = add_table(state, f"{prefix}_low", dtype, f"GroupLow<{ctype}>({modulus}, {split}, {sign})")
    high = add_table(state, f"{prefix}_high", dtype, f"GroupHigh<{ctype}>({modulus}, {split}, {sign})")
    lane = None
    if lane_shape is not None:
        call = f"GroupLane<{ctype}>({modulus}, {lane_shape[0]}, {lane_shape[1]}, {sign})"
        lane = add_table(state, f"{prefix}_lane", dtype, call)
    return low, high, split, lane


def add_axis_group(
    state: SDFGState,
    label: str,
    axis: int,
    group: tuple[int, int, int],
    sign: int,
    src_node: nodes.AccessNode,
    dst_node: nodes.AccessNode,
    tables: dict[tuple[int, int], nodes.AccessNode],
    group_tables: dict[tuple[int, int], tuple[nodes.AccessNode, nodes.AccessNode, int]],
    factor: symbolic.SymbolicType | None,
    schedule: dtypes.ScheduleType,
    constants: dict,
) -> nodes.MapEntry:
    """
    Adds a CPU map that computes a group ``(L, P, M)`` of consecutive stages of an axis (see ``axis_groups``). The
    stages of span ``L`` to ``L * P`` are equivalent to one radix-``P`` stage: for each ``b`` in ``[0, L)`` and ``k``
    in ``[0, M)``, the group multiplies the inputs at ``(b * P + a) * M + k`` by ``exp(sign 2 pi i a b / (L * P))``,
    computes their length-``P`` DFT, and writes output ``c`` to ``(c * L + b) * M + k``. The map is a local axis
    kernel (see ``add_local_axis_kernel``) on views of the arrays in which the DFT is along one dimension.

    :param state: The state to add the map to.
    :param label: The label of the map and of its views.
    :param axis: The axis.
    :param group: The ``(L, P, M)`` of the group.
    :param sign: -1 for the forward transform, 1 for the inverse transform.
    :param src_node: The access node to read from.
    :param dst_node: The access node to write to.
    :param tables: The twiddle factor tables of the stages of the groups (see ``add_twiddle_tables``).
    :param group_tables: The twiddle factor tables of groups by ``(Q, sign, lane shape)`` (see
                         ``add_group_twiddle_tables``), to which missing tables are added.
    :param factor: A factor to multiply the outputs by, or None.
    :param schedule: The schedule of the map.
    :param constants: Constants of the SDFG, used to evaluate symbolic lengths.
    :return: The entry node of the map.
    """
    sdfg = state.sdfg
    span, length, _ = group
    src, dst = src_node.data, dst_node.data
    dtype = sdfg.arrays[dst].dtype
    dims = _group_view(sdfg.arrays[dst].shape, axis, group)
    view_shape = [extent for _, extent in dims]
    view_axis = next(i for i, (dim, _) in enumerate(dims) if dim == "a")

    # Views of the input and the output in which the group transforms one dimension
    views = []
    for node, output in ((src_node, False), (dst_node, True)):
        desc = sdfg.arrays[node.data]
        view, _ = sdfg.add_view(
            f"{label}_{'out' if output else 'in'}",
            view_shape,
            desc.dtype,
            desc.storage,
            strides=_group_strides(desc, axis, dims, group, output),
            total_size=desc.total_size,
            find_new_name=True,
        )
        views.append(state.add_access(view))
        memlet = Memlet.from_array(node.data, desc)
        if output:
            state.add_edge(views[-1], None, node, None, memlet)
        else:
            state.add_edge(node, None, views[-1], None, memlet)

    stages = plan_stages(tuple(view_shape), [view_axis], {})
    lanes = local_lanes(tuple(view_shape), [view_axis], dtype, constants, GROUP_LANES)
    twiddle = None
    if span > 1:
        b_dim = next(i for i, (dim, _) in enumerate(dims) if dim == "b")
        lane_shape = (length, lanes) if b_dim == len(dims) - 1 and lanes > 1 else None
        key = (span * length, sign, lane_shape)
        if key not in group_tables:
            group_tables[key] = add_group_twiddle_tables(state, span * length, sign, dtype, lane_shape)
        low, high, split, lane = group_tables[key]
        twiddle = GroupTwiddle(b_dim, span * length, split, low, high, lane)
    return add_local_axis_kernel(
        state, label, stages, sign, views[0], views[1], tables, factor, lanes, schedule, constants, twiddle
    )


def _complex_product(left: str, right: str, dtype: dtypes.typeclass) -> str:
    """
    Returns a Python tasklet expression of the product of two complex values with explicit arithmetic, which
    vectorizes (unlike ``std::complex`` multiplication without fast-math options).
    """
    return (
        f"dace.{dtype.to_string()}({left}.real * {right}.real - {left}.imag * {right}.imag, "
        f"{left}.real * {right}.imag + {left}.imag * {right}.real)"
    )


def _add_group_twiddle_step(
    state: SDFGState,
    src: str,
    dst: str,
    shape: list[int],
    axis: int,
    batch_index: dict[str, list[str | None]],
    batch_symbols: dict[int, str],
    padding: int,
    twiddle: GroupTwiddle,
) -> None:
    """
    Adds maps to a state of a local axis kernel (see ``add_local_axis_kernel``) that multiply the elements of ``src``
    by the twiddle factors of a group of stages and write them to buffer ``dst``. The first map computes the twiddle
    factors of the first transform of the iteration from the tables, and the second one multiplies all transforms.

    :param state: The (empty) state to add the maps to.
    :param src: The array to read from.
    :param dst: The buffer to write to.
    :param shape: The shape of the transforms of an iteration.
    :param axis: The transformed dimension.
    :param batch_index: The batch index of each array (see ``add_block_local_stage``).
    :param batch_symbols: The symbol of each dimension that is not transformed.
    :param padding: The number of elements of a padded row of the buffers (see ``block_local_sdfg``), or 0.
    :param twiddle: The twiddle factors.
    """
    sdfg = state.sdfg
    dtype = sdfg.arrays[dst].dtype
    length = shape[axis]
    lane_dim = next((dim for dim, extent in enumerate(shape) if extent > 1 and dim != axis), None)
    lanes = 1 if lane_dim is None else shape[lane_dim]
    tables = {"_low": twiddle.low, "_high": twiddle.high, "_lane": twiddle.lane}
    tables = {conn: node.data for conn, node in tables.items() if node is not None}

    row, _ = sdfg.add_transient(
        "__fft_group_row", [length], dtype, dtypes.StorageType.Register(force=True), find_new_name=True
    )
    split = twiddle.split
    code = "\n".join(
        [
            f"__t = __a * {batch_symbols[twiddle.dim]} % {twiddle.modulus}",
            f"__low = _low[__t % {split}]",
            f"__high = _high[__t // {split}]",
            f"_w = {_complex_product('__low', '__high', dtype)}",
        ]
    )
    row_node = state.add_access(row)
    state.add_mapped_tasklet(
        "stockham_group_row",
        {"__a": f"0:{length}"},
        {conn: Memlet(f"{tables[conn]}[0]") for conn in ("_low", "_high")},
        code,
        {"_w": Memlet(f"{row}[__a]")},
        external_edges=True,
        output_nodes={row: row_node},
        schedule=dtypes.ScheduleType.Sequential,
    )

    def element(name: str) -> Memlet:
        subset = []
        for dim in range(len(shape)):
            base = batch_index[name][dim]
            if dim == axis and padding and name == dst:
                subset += [f"int_floor(__a, {padding})", f"__a % {padding}"]
            elif dim == axis:
                subset.append("__a")
            elif dim == lane_dim:
                subset.append("__l" if base is None else f"{base} + __l")
            else:
                subset.append("0" if base is None else base)
        return Memlet(f"{name}[{', '.join(subset)}]")

    inputs = {"_x": element(src), "_w": Memlet(f"{row}[__a]")}
    factor = "_w"
    lines = []
    if twiddle.lane is not None:
        inputs["_lane"] = Memlet(f"{tables['_lane']}[0]")
        lines.append(f"__lane = _lane[__a * {lanes} + __l]")
        lines.append(f"__w = {_complex_product('_w', '__lane', dtype)}")
        factor = "__w"
    lines.append(f"_y = {_complex_product('_x', factor, dtype)}")
    state.add_mapped_tasklet(
        "stockham_group_twiddle",
        {"__a": f"0:{length}", "__l": f"0:{lanes}"},
        inputs,
        "\n".join(lines),
        {"_y": element(dst)},
        external_edges=True,
        input_nodes={row: row_node},
        schedule=dtypes.ScheduleType.Sequential,
    )


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
    table_nodes = add_twiddle_tables(state, stages, sign, dtype, True)
    tables = {key: node.data for key, node in table_nodes.items()}
    arrays = {array: sdfg.arrays[array] for array in ("_inp", "_out", *tables.values())}
    block, batch_symbols, batch_index, buffers = block_local_sdfg(
        f"{name}_block", arrays, shape, axes, dtype, constants, min(len(stages) - 1, 2)
    )
    out_factor = None if factor == 1 else factor
    steps = [(stage, sign, tables, out_factor if s == len(stages) - 1 else None) for s, stage in enumerate(stages)]
    add_block_local_steps(block, steps, "_inp", "_out", buffers, block_shape, batch_index, threads)

    inputs = {"_inp": state.add_read("_inp"), **{node.data: node for node in table_nodes.values()}}
    add_block_local_kernel(state, block, inputs, state.add_write("_out"), shape, batch_symbols)
    return sdfg
