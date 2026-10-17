# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Implements Forward and Inverse Fast Fourier Transform (FFT) library nodes
"""

import functools
import itertools
import operator
import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple

from dace import SDFG, Memlet, SDFGState, cpf_lowering, data, dtypes, library, nodes, properties, symbolic
from dace import transformation as xf
from dace.frontend.python.replacements.utils import normalize_axes
from dace.libraries.blas.environments.intel_mkl import IntelMKL
from dace.libraries.fft import environments as env
from dace.libraries.fft.gpu_dialect import CUFFT, HIPFFT, GpuFftDialect
from dace.sdfg import utils as sdutil


def normalize_fft_axes(ndim: int, axes: Sequence[int] | None) -> list[int]:
    """The axes an FFT over ``axes`` transforms, in order, as indices in ``0..ndim-1``; ``None`` means every axis."""
    result = list(range(ndim)) if axes is None else normalize_axes([int(axis) for axis in axes], ndim)
    if any(not 0 <= axis < ndim for axis in result):
        raise ValueError(f"FFT axes {axes} out of range for rank-{ndim} input")
    return result


@properties.make_properties
class FourierTransform(nodes.LibraryNode):
    """``_out`` is the discrete Fourier transform of ``_inp`` over ``axes``, scaled by ``factor``.

    With ``axes is None`` (default) every axis is transformed (``np.fft.fftn``). With ``axes`` set the listed axes
    are transformed in order and the others are batch dimensions, matching ``np.fft.fftn(x, axes=...)`` and
    ``np.fft.fft(x, axis=k)`` (``axes=[k]``). With ``half_spectrum`` the complex side holds only the non-negative
    frequencies of the last transformed axis (``np.fft.rfftn`` / ``np.fft.irfftn``): :class:`FFT` reads a real
    input and :class:`IFFT` writes a real output, whose extents are the transform lengths.
    """

    factor = properties.SymbolicProperty(
        category="Semantics", desc="Coefficient to multiply outputs. Used for normalization", default=1.0
    )
    axes = properties.ListProperty(
        element_type=int,
        allow_none=True,
        default=None,
        category="Semantics",
        desc="Axes transformed in order (0..rank-1); unlisted axes are batch dimensions. ``None`` means every axis.",
    )
    half_spectrum = properties.Property(
        dtype=bool,
        default=False,
        category="Semantics",
        desc="The complex side keeps only the non-negative frequencies of the last transformed axis (rfft/irfft).",
    )

    def __init__(self, name, *args, schedule=None, axes=None, half_spectrum=False, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={"_inp"}, outputs={"_out"}, **kwargs)
        self.axes = axes
        self.half_spectrum = half_spectrum


@library.node
class FFT(FourierTransform):
    """Forward FFT. See :class:`FourierTransform`."""

    implementations = {}
    default_implementation = "pure"


@library.node
class IFFT(FourierTransform):
    """Inverse FFT, unnormalized unless ``factor`` says otherwise. See :class:`FourierTransform`."""

    implementations = {}
    default_implementation = "pure"


##################################################################################################
# Native SDFG expansions
##################################################################################################


def pure_sdfg(node: FourierTransform, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
    from dace.libraries.fft.algorithms import dft  # Lazy import functions

    indesc, outdesc = input_and_output(parent_state, parent_sdfg, node)
    return dft.dft_nd_sdfg(indesc, outdesc, node.factor, isinstance(node, IFFT), node.axes, node.half_spectrum)


@library.register_expansion(FFT, "pure")
class DFTExpansion(xf.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return pure_sdfg(node, parent_state, parent_sdfg)


@library.register_expansion(IFFT, "pure")
class IDFTExpansion(xf.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return pure_sdfg(node, parent_state, parent_sdfg)


##################################################################################################
# Vendor expansions: FFTW3 / MKL on the CPU, cuFFT / hipFFT on the GPU
##################################################################################################

#: The complex types every vendor transform computes in.
COMPLEX_TYPES = (dtypes.complex64, dtypes.complex128)

#: The most axes one cuFFT / hipFFT plan transforms.
MAX_GPU_FFT_RANK = 3


class Transform(NamedTuple):
    """One vendor call: ``out`` is the transform of ``src`` over ``axes``."""

    src: data.Data
    out: data.Data
    axes: list[int]
    inverse: bool
    half_spectrum: bool

    @property
    def real_input(self) -> bool:
        return self.half_spectrum and not self.inverse

    @property
    def real_output(self) -> bool:
        return self.half_spectrum and self.inverse

    @property
    def lengths(self) -> list[symbolic.SymbolicType]:
        """The transform lengths: the extents of the transformed axes on the real (or the only) side."""
        signal = self.out if self.real_output else self.src
        return [signal.shape[a] for a in self.axes]

    @property
    def double(self) -> bool:
        return (self.src if self.real_output else self.out).dtype == dtypes.complex128


#: Builds the vendor call for a :class:`Transform`, or returns ``None`` when the library cannot address its layout.
VendorCall = Callable[[Transform], nodes.Tasklet | None]


def vendor_fft_sdfg(
    node: FourierTransform, parent_state: SDFGState, parent_sdfg: SDFG, call: VendorCall, backend: str
) -> SDFG:
    """Wrap one vendor FFT call in the numpy semantics the library node carries.

    The vendor transforms are unnormalized and read the complex type, or the real one for a half-spectrum forward
    transform. An input of another type, or one the vendor cannot address, is staged into a copy of the right type
    and layout first; a half-spectrum inverse is always staged, because the vendor overwrites its input. The
    ``factor`` scales that staging copy when there is one (the transform is linear), and the output otherwise. A
    layout the vendor cannot address even staged falls back to the ``pure`` expansion, with a warning naming the
    backend.
    """
    from dace.libraries.fft.algorithms.dft import complex_of, floating_text, real_of  # avoid loop

    inverse = isinstance(node, IFFT)
    indesc, outdesc = input_and_output(parent_state, parent_sdfg, node)
    axes = normalize_fft_axes(len(indesc.shape), node.axes)
    transform = Transform(indesc, outdesc, axes, inverse, node.half_spectrum)
    complex_type = complex_of(outdesc.dtype)
    read_type = real_of(complex_type) if transform.real_input else complex_type
    # The staged copy takes the output's layout where the shapes agree: the relayout a source the vendor cannot
    # plan against needs (a batch-innermost view into a Fortran-order result). One copy beats the separable DFT.
    staged = data.Array(
        read_type,
        indesc.shape,
        storage=outdesc.storage,
        strides=outdesc.strides if indesc.shape == outdesc.shape else None,
    )
    tasklet = None
    stage = indesc.dtype != read_type or transform.real_output
    if complex_type in COMPLEX_TYPES:
        if not stage:
            tasklet = call(transform)
            stage = tasklet is None
        if stage:
            tasklet = call(transform._replace(src=staged))
    if tasklet is None:
        warnings.warn(
            f"{backend} cannot transform axes {axes} of {indesc.dtype}{list(indesc.shape)} "
            f"(strides {list(indesc.strides)}); falling back to the pure expansion",
            stacklevel=2,
        )
        return pure_sdfg(node, parent_state, parent_sdfg)

    sdfg = SDFG(f"{node.label}_{backend}")
    sdfg.add_array(
        "_inp", indesc.shape, indesc.dtype, storage=indesc.storage, strides=indesc.strides, offset=indesc.offset
    )
    sdfg.add_array(
        "_out", outdesc.shape, outdesc.dtype, storage=outdesc.storage, strides=outdesc.strides, offset=outdesc.offset
    )
    # Explicit on the GPU: the maps sit at host level next to a host-side library call.
    schedule = (
        dtypes.ScheduleType.GPU_Device
        if outdesc.storage == dtypes.StorageType.GPU_Global
        else dtypes.ScheduleType.Default
    )
    # Divide in floating point: ``1/N`` over an integer extent symbol is C integer division.
    scale = (
        ""
        if str(node.factor) == "1"
        else f" * dace.{'float64' if complex_type == dtypes.complex128 else 'float32'}({floating_text(node.factor)})"
    )

    from dace.libraries.fft.algorithms.dft import index_ranges  # avoid loop

    def elementwise(state: SDFGState, label: str, src: str, dst: str, shape, code: str) -> None:
        ranges = index_ranges(shape, "__i")
        subset = ", ".join(ranges)
        state.add_mapped_tasklet(
            label,
            ranges,
            {"i": Memlet(data=src, subset=subset)},
            code,
            {"o": Memlet(data=dst, subset=subset)},
            schedule=schedule,
            external_edges=True,
        )

    source = "_inp"
    state = sdfg.add_state("fft", is_start_block=True)
    if stage:
        sdfg.add_transient("__staged", staged.shape, staged.dtype, storage=staged.storage, strides=staged.strides)
        # The cast names the type: ``decltype(o)`` would be a reference to the target element.
        elementwise(state, "stage", "_inp", "__staged", indesc.shape, f"o = dace.{read_type.to_string()}(i){scale}")
        source = "__staged"
        state = sdfg.add_state_after(state, "fft")
    state.add_edge(state.add_read(source), None, tasklet, "__in", Memlet.from_array(source, sdfg.arrays[source]))
    state.add_edge(tasklet, "__out", state.add_write("_out"), None, Memlet.from_array("_out", sdfg.arrays["_out"]))
    if scale and not stage:
        elementwise(
            sdfg.add_state_after(state, "normalize"), "normalize", "_out", "_out", outdesc.shape, f"o = i{scale}"
        )
    return sdfg


def dense_block_order(
    descs: Sequence[data.Data], axes: Sequence[int]
) -> tuple[list[int], list[symbolic.SymbolicType]] | None:
    """An order of ``axes``, slowest first, in which every descriptor lays them out as ONE dense block.

    Returned with each descriptor's element stride (its fastest axis's stride): axis ``order[k]``
    steps by that stride times the extents of every later axis. The order is shared, since a plan's
    dimension list maps the same logical axis in the input and the output. Compared through
    :func:`~dace.symbolic.relax_ipow`, since canonicalization respells a packed stride ``N**2`` as ``ipow(N, 2)``.
    """
    for order in itertools.permutations(axes):
        steps = []
        for desc in descs:
            step = span = desc.strides[order[-1]]
            for axis in reversed(order):
                if symbolic.equal(symbolic.relax_ipow(desc.strides[axis]), symbolic.relax_ipow(span)) is not True:
                    break
                span = span * desc.shape[axis]
            else:
                steps.append(step)
                continue
            break
        if len(steps) == len(descs):
            return list(order), steps
    return None


def fftw3_call(transform: Transform) -> nodes.Tasklet | None:
    """One ``fftw_plan_guru64_dft`` (``_r2c``, ``_c2r``) over the transformed axes, every other axis a ``howmany``
    batch dimension, planned once and kept in the library's plan cache.

    Both dimension lists step by the descriptors' own strides, so any rank, any axis set and any strided view is
    one plan, and one buffer on both connectors is FFTW's in-place transform. A plan is keyed by the transform
    kind, the extents and strides, whether the transform is in place, the planner's thread count and the arrays'
    alignment, the conditions under which FFTW's new-array execute may reuse a plan.
    """
    from dace.codegen.targets import cpp  # avoid import loop
    from dace.libraries.fft.environments import fftw3

    src, out, axes = transform.src, transform.out, transform.axes
    if len(set(axes)) != len(axes):
        return None
    prefix = "fftw" if transform.double else "fftwf"
    real_t = "double" if transform.double else "float"
    complex_t = f"{prefix}_complex"
    in_t = real_t if transform.real_input else complex_t
    out_t = real_t if transform.real_output else complex_t
    kind = "dft_r2c" if transform.real_input else "dft_c2r" if transform.real_output else "dft"
    batch = [d for d in range(len(src.shape)) if d not in axes]

    def iodim(
        extent: symbolic.SymbolicType, istride: symbolic.SymbolicType, ostride: symbolic.SymbolicType
    ) -> list[str]:
        return [cpp.sym2cpp(extent), cpp.sym2cpp(istride), cpp.sym2cpp(ostride)]

    dims = [iodim(n, src.strides[d], out.strides[d]) for n, d in zip(transform.lengths, axes)]
    # MKL's FFTW3 interface plans at most one batch dimension and returns no plan otherwise, so a batch that is one
    # dense block goes as one dimension, and any other batch plans its innermost dimension and loops over the rest,
    # one plan executed per slice.
    block = dense_block_order((src, out), batch) if len(batch) > 1 else None
    looped: list[int] = []
    if block is not None:
        howmany = [iodim(functools.reduce(operator.mul, (src.shape[d] for d in batch), 1), *block[1])]
    else:
        looped, planned = batch[:-1], batch[-1:]
        howmany = [iodim(src.shape[d], src.strides[d], out.strides[d]) for d in planned]

    def iodims(name: str, entries: list[list[str]]) -> str:
        values = ", ".join(f"{{{', '.join(entry)}}}" for entry in entries)
        return f"{prefix}_iodim64 {name}[{len(entries)}] = {{{values}}};" if entries else ""

    loop_open = "".join(f"for (long long __b{d} = 0; __b{d} < {cpp.sym2cpp(src.shape[d])}; ++__b{d}) " for d in looped)
    in_offset = "".join(f" + __b{d} * ({cpp.sym2cpp(src.strides[d])})" for d in looped)
    out_offset = "".join(f" + __b{d} * ({cpp.sym2cpp(out.strides[d])})" for d in looped)
    # Slices of the planned arrays have any alignment; a whole array keeps the planned one, part of the key.
    flags = "FFTW_ESTIMATE | FFTW_UNALIGNED" if looped else "FFTW_ESTIMATE"
    alignment = (
        "0, 0"
        if looped
        else "(long long)(reinterpret_cast<uintptr_t>(__in) % 64), (long long)(reinterpret_cast<uintptr_t>(__out) % 64)"
    )
    sign = f"{'FFTW_BACKWARD' if transform.inverse else 'FFTW_FORWARD'}, " if kind == "dft" else ""
    threaded = fftw3.threaded()
    # A threaded FFTW plans for the team the program runs (pocketfft's ``workers``); inside a parallel region, one.
    team = (
        f"static const int __threads_ready = {prefix}_init_threads(); (void)__threads_ready;\n"
        f"{prefix}_plan_with_nthreads(__threads);"
        if threaded
        else ""
    )
    threads = "omp_in_parallel() ? 1 : omp_get_max_threads()" if threaded else "1"
    # The cache is shared by every transform of this precision, so the key names the transform kind too.
    kind_key = (
        f"{['dft', 'dft_r2c', 'dft_c2r'].index(kind)}, {int(transform.inverse)}, {int(bool(looped))}, {len(dims)}"
    )
    key = ", ".join([kind_key] + [value for entry in dims + howmany for value in entry])
    plan = (
        f"{prefix}_plan_guru64_{kind}({len(dims)}, __dims, {len(howmany)}, {'__howmany' if howmany else 'NULL'}, "
        f"({in_t} *)__in, ({out_t} *)__out, {sign}{flags})"
    )
    execute = f"{loop_open}{prefix}_execute_{kind}(__plan, ({in_t} *)__in{in_offset}, ({out_t} *)__out{out_offset});"
    if cpf_lowering.standalone_c():
        # C has no templated plan cache: plan, run and destroy per call, the planner serialized.
        code = f"""
    {iodims("__dims", dims)}
    {iodims("__howmany", howmany)}
    {prefix}_plan __plan;
    #pragma omp critical(dace_fftw_planner)
    __plan = {plan};
    if (!__plan)
        throw std::runtime_error("FFTW returned no plan for this transform layout");
    {execute}
    #pragma omp critical(dace_fftw_planner)
    {prefix}_destroy_plan(__plan);
    """
        return nodes.Tasklet(
            f"fftw3_{'i' if transform.inverse else ''}fft", {"__in"}, {"__out"}, code, language=dtypes.Language.CPP
        )
    code = f"""
    {iodims("__dims", dims)}
    {iodims("__howmany", howmany)}
    const int __threads = {threads};
    {prefix}_plan __plan = dace::fft::Plans<{prefix}_plan>({prefix}_destroy_plan).Get(
        {{{key}, (long long)((const void *)__in == (const void *)__out), (long long)__threads, {alignment}}}, [&] {{
            {team}
            {prefix}_plan __made = {plan};
            if (!__made)
                throw std::runtime_error("FFTW returned no plan for this transform layout");
            return __made;
        }});
    {execute}
    """
    return nodes.Tasklet(
        f"fftw3_{'i' if transform.inverse else ''}fft", {"__in"}, {"__out"}, code, language=dtypes.Language.CPP
    )


class GpuFftLayout(NamedTuple):
    """One ``MakePlanMany`` plan: ``lengths`` in plan order, each side's embed (its own extents in that order),
    element stride and batch distance, and the batch count."""

    lengths: list[symbolic.SymbolicType]
    in_embed: list[symbolic.SymbolicType]
    in_stride: symbolic.SymbolicType
    in_distance: symbolic.SymbolicType
    out_embed: list[symbolic.SymbolicType]
    out_stride: symbolic.SymbolicType
    out_distance: symbolic.SymbolicType
    batch: symbolic.SymbolicType


def gpu_fft_layout(transform: Transform) -> GpuFftLayout | None:
    """``transform`` as ONE ``MakePlanMany`` plan, else ``None``.

    A plan reads element ``(p, q, r)`` of batch ``b`` at ``stride * (p*n1*n2 + q*n2 + r) + dist * b``, separately
    for input and output with each side's embed. So each side's transformed axes must be one dense block under a
    single element stride, and its other axes one dense block under a single batch distance, in an axis order both
    sides share. That covers the C-order trailing and leading runs, a Fortran-order batch (the C-order batch of its
    reversed axes), and a batch-innermost view of an ``order='F'`` reshape. numpy's result does not depend on the
    axis order, so the order is free to choose -- except for a half-spectrum transform, whose halved axis must stay
    the last.
    """
    src, out, axes = transform.src, transform.out, transform.axes
    if len(set(axes)) != len(axes) or len(axes) > MAX_GPU_FFT_RANK:
        return None
    if any(d.storage != dtypes.StorageType.GPU_Global for d in (src, out)):
        return None
    block = dense_block_order((src, out), axes)
    if block is None or (transform.half_spectrum and block[0][-1] != axes[-1]):
        return None
    order, (istride, ostride) = block
    lengths = dict(zip(axes, transform.lengths))
    rest = [a for a in range(len(src.shape)) if a not in axes]
    in_size = functools.reduce(operator.mul, (src.shape[a] for a in order), 1)
    out_size = functools.reduce(operator.mul, (out.shape[a] for a in order), 1)
    if not rest:
        idist, odist, batch = istride * in_size, ostride * out_size, 1
    else:
        batched = dense_block_order((src, out), rest)
        if batched is None:
            return None
        batch_order, (idist, odist) = batched
        batch = functools.reduce(operator.mul, (src.shape[a] for a in batch_order), 1)
    return GpuFftLayout(
        [lengths[a] for a in order],
        [src.shape[a] for a in order],
        istride,
        idist,
        [out.shape[a] for a in order],
        ostride,
        odist,
        batch,
    )


def gpu_fft_call(dialect: GpuFftDialect) -> VendorCall:
    """The cuFFT / hipFFT call in ``dialect``'s spelling: a ``MakePlanMany64`` plan from the library's plan cache,
    run on the SDFG's stream, so a transform inside a loop plans once. One buffer on both connectors is the vendor's
    in-place transform.
    """

    def call(transform: Transform) -> nodes.Tasklet | None:
        from dace.codegen.targets import cpp  # avoid import loop

        layout = gpu_fft_layout(transform)
        if layout is None:
            return None
        rank = len(layout.lengths)
        key = [
            cpp.sym2cpp(e)
            for e in (
                *layout.lengths,
                *layout.in_embed,
                layout.in_stride,
                layout.in_distance,
                *layout.out_embed,
                layout.out_stride,
                layout.out_distance,
                layout.batch,
            )
        ]
        lengths, in_embed, out_embed = "__key", f"__key + {rank}", f"__key + {2 * rank + 2}"
        in_layout = f"{in_embed}, __key[{2 * rank}], __key[{2 * rank + 1}]"
        out_layout = f"{out_embed}, __key[{3 * rank + 2}], __key[{3 * rank + 3}]"
        batch = f"__key[{3 * rank + 4}]"
        letter = "Z" if transform.double else "C"
        real_letter = "D" if transform.double else "R"
        kind = (
            f"{real_letter}2{letter}"
            if transform.real_input
            else f"{letter}2{real_letter}"
            if transform.real_output
            else f"{letter}2{letter}"
        )
        direction = dialect.inverse if transform.inverse else f"{dialect.enum}FORWARD"
        # The cache is shared by every transform, so the key ends with the transform type.
        key += [str(int(transform.real_input) + 2 * int(transform.real_output)), str(int(transform.double))]
        check = (
            f"void (*__check)(const char *, {dialect.api}Result) = [](const char *what, {dialect.api}Result result) {{ "
            f'if (result != {dialect.enum}SUCCESS) throw std::runtime_error(std::string("{dialect.name} ") + '
            f'what + " failed with status " + std::to_string((int)result)); }};'
        )
        code = f"""
        {check}
        long long __key[{len(key)}] = {{{", ".join(key)}}};
        // An empty transform is numpy's empty result: nothing to plan, and the vendor planner raises
        // SIGFPE on a zero extent or batch.
        bool __empty = {batch} == 0;
        for (int __k = 0; __k < {rank}; ++__k)
            __empty = __empty || __key[__k] == 0;
        if (!__empty) {{
        {dialect.api}Handle __plan = dace::fft::Plans<{dialect.api}Handle>([]({dialect.api}Handle plan) {{
            {dialect.api}Destroy(plan);
        }}).Get({{std::begin(__key), std::end(__key)}}, [&] {{
            {dialect.api}Handle __made;
            __check("{dialect.api}Create", {dialect.api}Create(&__made));
            size_t __work_size = 0;
            __check("{dialect.api}MakePlanMany64", {dialect.api}MakePlanMany64(__made, {rank}, {lengths},
                {in_layout}, {out_layout}, {dialect.enum}{kind}, {batch}, &__work_size));
            return __made;
        }});
        __check("{dialect.api}SetStream", {dialect.api}SetStream(__plan, __dace_current_stream));
        __check("{dialect.api}XtExec", {dialect.api}XtExec(__plan, (void *)__in, (void *)__out, {direction}));
        }}
        """
        return nodes.Tasklet(
            f"{dialect.api}_{'i' if transform.inverse else ''}fft",
            {"__in"},
            {"__out"},
            code,
            language=dtypes.Language.CPP,
        )

    return call


class ExpandFFTW3(xf.ExpandTransformation):
    """CPU FFTW3 backend: any rank, axes, strides, real or complex input, any ``norm``."""

    environments = [env.FFTW3, env.FFTPlans]

    @staticmethod
    def expansion(node: FourierTransform, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return vendor_fft_sdfg(node, parent_state, parent_sdfg, fftw3_call, "FFTW3")


@library.register_expansion(FFT, "FFTW3")
class FFTW3FFTExpansion(ExpandFFTW3):
    pass


@library.register_expansion(IFFT, "FFTW3")
class FFTW3IFFTExpansion(ExpandFFTW3):
    pass


class ExpandGPUFFT(xf.ExpandTransformation):
    """The one GPU FFT expansion; a backend subclass names its environment and :class:`GpuFftDialect`."""

    environments = []
    dialect: GpuFftDialect

    @classmethod
    def expansion(cls, node: FourierTransform, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return vendor_fft_sdfg(node, parent_state, parent_sdfg, gpu_fft_call(cls.dialect), cls.dialect.name)


@library.register_expansion(FFT, "cuFFT")
class cuFFTFFTExpansion(ExpandGPUFFT):
    environments = [env.cuFFT, env.FFTPlans]
    dialect = CUFFT


@library.register_expansion(IFFT, "cuFFT")
class cuFFTIFFTExpansion(ExpandGPUFFT):
    environments = [env.cuFFT, env.FFTPlans]
    dialect = CUFFT


@library.register_expansion(FFT, "hipFFT")
class hipFFTFFTExpansion(ExpandGPUFFT):
    environments = [env.hipFFT, env.FFTPlans]
    dialect = HIPFFT


@library.register_expansion(IFFT, "hipFFT")
class hipFFTIFFTExpansion(ExpandGPUFFT):
    environments = [env.hipFFT, env.FFTPlans]
    dialect = HIPFFT


class ExpandMKL(ExpandFFTW3):
    """MKL backend: MKL exposes the FFTW3 C ABI, so the code is the FFTW3 expansion's, linked against MKL."""

    environments = [IntelMKL, env.FFTPlans]


@library.register_expansion(FFT, "MKL")
class MKLFFTExpansion(ExpandMKL):
    pass


@library.register_expansion(IFFT, "MKL")
class MKLIFFTExpansion(ExpandMKL):
    pass


##################################################################################################
# Helper functions
##################################################################################################


def input_and_output(state: SDFGState, sdfg: SDFG, node: nodes.LibraryNode) -> tuple[data.Data, data.Data]:
    return sdutil.in_desc_with_name(node, state, sdfg, "_inp"), sdutil.out_desc_with_name(node, state, sdfg, "_out")
