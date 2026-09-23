# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Implements Forward and Inverse Fast Fourier Transform (FFT) library nodes
"""

import functools
import itertools
import operator
import warnings
from typing import Any
from collections.abc import Callable, Sequence

from dace import data, dtypes, Memlet, SDFG, SDFGState, symbolic, library, nodes, properties
from dace import transformation as xf
from dace.libraries.fft import environments as env
from dace.libraries.fft.gpu_dialect import CUFFT, HIPFFT, GpuFftDialect


def normalize_fft_axes(ndim: int, axes: Sequence[int] | None) -> list[int]:
    """The axes an FFT over ``axes`` transforms, in order, as indices in ``0..ndim-1``; ``None`` means every axis."""
    if axes is None:
        return list(range(ndim))
    result = []
    for axis in axes:
        index = int(axis) + ndim if int(axis) < 0 else int(axis)
        if not 0 <= index < ndim:
            raise ValueError(f'FFT axis {axis} out of range for rank-{ndim} input')
        result.append(index)
    return result


# Define the library nodes
@library.node
class FFT(nodes.LibraryNode):
    """Forward FFT.

    With ``axes is None`` (default) the lib node transforms every axis of the input (``np.fft.fftn``).
    With ``axes`` set it transforms the listed axes in order and treats the others as batch
    dimensions, matching ``np.fft.fftn(x, axes=...)`` and ``np.fft.fft(x, axis=k)`` (``axes=[k]``).
    """
    implementations = {}
    default_implementation = 'pure'

    factor = properties.SymbolicProperty(desc='Coefficient to multiply outputs. Used for normalization', default=1.0)
    axes = properties.ListProperty(element_type=int,
                                   allow_none=True,
                                   default=None,
                                   desc='Axes transformed in order (0..rank-1); unlisted axes are batch dimensions. '
                                   '``None`` means every axis.')

    def __init__(self, name, *args, schedule=None, axes=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={'_inp'}, outputs={'_out'}, **kwargs)
        self.axes = axes


@library.node
class IFFT(nodes.LibraryNode):
    """Inverse FFT.  See :class:`FFT` for ``axes`` semantics."""

    implementations = {}
    default_implementation = 'pure'

    factor = properties.SymbolicProperty(desc='Coefficient to multiply outputs. Used for normalization', default=1.0)
    axes = properties.ListProperty(element_type=int,
                                   allow_none=True,
                                   default=None,
                                   desc='Axes transformed in order (0..rank-1); unlisted axes are batch dimensions. '
                                   '``None`` means every axis.')

    def __init__(self, name, *args, schedule=None, axes=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={'_inp'}, outputs={'_out'}, **kwargs)
        self.axes = axes


##################################################################################################
# Native SDFG expansions
##################################################################################################


@library.register_expansion(FFT, 'pure')
class DFTExpansion(xf.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries.fft.algorithms import dft  # Lazy import functions
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if len(indesc.shape) > 1 or node.axes is not None:
            return dft.dft_nd_sdfg(indesc, outdesc, factor=node.factor, inverse=False, axes=node.axes)

        return dft.dft_explicit.to_sdfg(indesc, outdesc, N=indesc.shape[0], factor=dft.floating_factor(node.factor))


@library.register_expansion(IFFT, 'pure')
class IDFTExpansion(xf.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries.fft.algorithms import dft  # Lazy import functions
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if len(indesc.shape) > 1 or node.axes is not None:
            return dft.dft_nd_sdfg(indesc, outdesc, factor=node.factor, inverse=True, axes=node.axes)

        return dft.idft_explicit.to_sdfg(indesc, outdesc, N=indesc.shape[0], factor=dft.floating_factor(node.factor))


##################################################################################################
# Vendor expansions: FFTW3 / MKL on the CPU, cuFFT / hipFFT on the GPU
##################################################################################################

#: The operand types every vendor transform takes: complex-to-complex only.
COMPLEX_TYPES = (dtypes.complex64, dtypes.complex128)

#: The most axes one cuFFT / hipFFT plan transforms.
MAX_GPU_FFT_RANK = 3

#: Suffix of each GPU plan's state fields, unique across every program in the process.
PLAN_IDS = itertools.count()

#: Builds the vendor call for ``(complex input descriptor, output descriptor, transformed axes,
#: is_inverse)``, or returns ``None`` when the library cannot address that layout.
VendorCall = Callable[[data.Data, data.Data, list[int], bool], nodes.Tasklet | None]


def vendor_fft_sdfg(node: nodes.LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, call: VendorCall,
                    backend: str) -> SDFG:
    """Wrap one vendor FFT call in the numpy semantics the library node carries.

    The vendor transforms are complex-to-complex and unnormalised, so the nested SDFG casts a real
    input into a complex buffer first and scales the output by ``node.factor`` afterwards. A layout
    the vendor cannot address (a repeated axis, or a GPU transform that is not one batched block)
    falls back to the separable ``pure`` expansion, with a warning naming the backend.
    """
    from dace.libraries.fft.algorithms.dft import FloatingPrinter  # avoid import loop
    is_inverse = isinstance(node, IFFT)
    inp, out = _get_input_and_output(parent_state, node)
    indesc, outdesc = parent_sdfg.arrays[inp], parent_sdfg.arrays[out]
    transformed = normalize_fft_axes(len(indesc.shape), node.axes)
    # A staged copy of the input in the output's dtype AND layout: the cast a real input needs, and the
    # relayout a source the vendor cannot plan against the output needs (cegterg's ifftn reads a
    # batch-innermost view into a Fortran-order result). One device copy beats the separable DFT.
    staged = data.Array(outdesc.dtype, outdesc.shape, storage=outdesc.storage, strides=outdesc.strides)
    cast = indesc.dtype != outdesc.dtype
    tasklet = None
    if outdesc.dtype in COMPLEX_TYPES:
        tasklet = None if cast else call(indesc, outdesc, transformed, is_inverse)
        if tasklet is None:
            tasklet = call(staged, outdesc, transformed, is_inverse)
            cast = tasklet is not None
    if tasklet is None:
        warnings.warn(
            f'{backend} cannot transform axes {transformed} of {outdesc.dtype}{list(indesc.shape)} '
            f'(strides {list(indesc.strides)}); falling back to the pure expansion',
            stacklevel=2)
        pure = IDFTExpansion if is_inverse else DFTExpansion
        return pure.expansion(node, parent_state, parent_sdfg)

    sdfg = SDFG(f'{node.label}_{backend}')
    sdfg.add_array('_inp',
                   indesc.shape,
                   indesc.dtype,
                   storage=indesc.storage,
                   strides=indesc.strides,
                   offset=indesc.offset)
    sdfg.add_array('_out',
                   outdesc.shape,
                   outdesc.dtype,
                   storage=outdesc.storage,
                   strides=outdesc.strides,
                   offset=outdesc.offset)
    # Explicit on the GPU: the maps sit at host level next to a host-side library call.
    schedule = (dtypes.ScheduleType.GPU_Device
                if outdesc.storage == dtypes.StorageType.GPU_Global else dtypes.ScheduleType.Default)
    ranges = {f'__i{d}': f'0:{symbolic.symstr(extent)}' for d, extent in enumerate(outdesc.shape)}
    subset = ', '.join(ranges)

    source = '_inp'
    state = sdfg.add_state('fft', is_start_block=True)
    if cast:
        sdfg.add_transient('__cinp', outdesc.shape, outdesc.dtype, storage=outdesc.storage, strides=outdesc.strides)
        # The cast names the type: ``decltype(o)`` would be a reference to the target element.
        state.add_mapped_tasklet('cast_in',
                                 ranges, {'i': Memlet(data='_inp', subset=subset)},
                                 f'o = dace.{outdesc.dtype.to_string()}(i)',
                                 {'o': Memlet(data='__cinp', subset=subset)},
                                 schedule=schedule,
                                 external_edges=True)
        source = '__cinp'
        state = sdfg.add_state_after(state, 'fft')
    state.add_edge(state.add_read(source), None, tasklet, '__in', Memlet.from_array(source, sdfg.arrays[source]))
    state.add_edge(tasklet, '__out', state.add_write('_out'), None, Memlet.from_array('_out', sdfg.arrays['_out']))

    if str(node.factor) != '1':
        # Divide in floating point: ``1/N`` over an integer extent symbol is C integer division.
        factor = FloatingPrinter().doprint(symbolic.pystr_to_symbolic(node.factor))
        real = 'float32' if outdesc.dtype == dtypes.complex64 else 'float64'
        state = sdfg.add_state_after(state, 'normalize')
        state.add_mapped_tasklet('normalize',
                                 ranges, {'i': Memlet(data='_out', subset=subset)},
                                 f'o = i * dace.{real}({factor})', {'o': Memlet(data='_out', subset=subset)},
                                 schedule=schedule,
                                 external_edges=True)
    return sdfg


def fftw3_call(src: data.Data, out: data.Data, transformed: list[int], is_inverse: bool) -> nodes.Tasklet | None:
    """One ``fftw_plan_guru64_dft`` over the transformed axes, every other axis a ``howmany`` batch dimension.

    Both dimension lists step by the descriptors' own strides, so any rank, any axis set and any
    strided view is one plan, and one buffer on both connectors is FFTW's in-place transform. FFTW's planner is
    not thread-safe, so plan creation and destruction are serialised; execution is not.
    """
    from dace.codegen.targets import cpp  # avoid import loop
    if len(set(transformed)) != len(transformed):
        return None
    prefix, complex_t = ('fftw_', 'fftw_complex') if out.dtype == dtypes.complex128 else ('fftwf_', 'fftwf_complex')
    batch = [d for d in range(len(src.shape)) if d not in transformed]

    def iodims(dims: list[int]) -> str:
        return ', '.join(
            f'{{{cpp.sym2cpp(src.shape[d])}, {cpp.sym2cpp(src.strides[d])}, {cpp.sym2cpp(out.strides[d])}}}'
            for d in dims)

    howmany = f'{prefix}iodim64 __howmany[{len(batch)}] = {{{iodims(batch)}}};' if batch else ''
    direction = 'FFTW_BACKWARD' if is_inverse else 'FFTW_FORWARD'
    code = f"""
    {prefix}iodim64 __dims[{len(transformed)}] = {{{iodims(transformed)}}};
    {howmany}
    {prefix}plan __plan;
    #pragma omp critical(dace_fftw_planner)
    __plan = {prefix}plan_guru64_dft({len(transformed)}, __dims, {len(batch)}, {'__howmany' if batch else 'NULL'},
                                    ({complex_t} *)__in, ({complex_t} *)__out, {direction}, FFTW_ESTIMATE);
    {prefix}execute(__plan);
    #pragma omp critical(dace_fftw_planner)
    {prefix}destroy_plan(__plan);
    """
    return nodes.Tasklet(f'fftw3_{"i" if is_inverse else ""}fft', {'__in'}, {'__out'},
                         code,
                         language=dtypes.Language.CPP)


def dense_block_order(descs: Sequence[data.Data], axes: Sequence[int]) -> tuple[list[int], list[Any]] | None:
    """An order of ``axes``, slowest first, in which every descriptor lays them out as ONE dense block.

    Returned with each descriptor's element stride (its fastest axis's stride): axis ``order[k]``
    steps by that stride times the extents of every later axis. The order is shared, since a plan's
    dimension list maps the same logical axis in the input and the output.
    """
    for order in itertools.permutations(axes):
        steps = []
        for desc in descs:
            step = span = desc.strides[order[-1]]
            for axis in reversed(order):
                if symbolic.equal(desc.strides[axis], span) is not True:
                    break
                span = span * desc.shape[axis]
            else:
                steps.append(step)
                continue
            break
        if len(steps) == len(descs):
            return list(order), steps
    return None


def gpu_fft_layout(src: data.Data, out: data.Data,
                   transformed: list[int]) -> tuple[list, Any, Any, Any, Any, Any] | None:
    """``(extents, istride, idist, ostride, odist, batch)`` of ``transformed`` as ONE ``MakePlanMany`` plan, else ``None``.

    With the embeds equal to the extents, a plan reads element ``(p, q, r)`` of batch ``b`` at
    ``stride * (p*n1*n2 + q*n2 + r) + dist * b``, separately for input and output. So each side's
    transformed axes must be one dense block under a single element stride, and its other axes one
    dense block under a single batch distance, in an axis order both sides share. That covers the
    C-order trailing and leading runs, a Fortran-order batch (the C-order batch of its reversed
    axes), and cegterg's batch-innermost view of an ``order='F'`` reshape. numpy's result does not
    depend on the axis order, so the order is free to choose.
    """
    if len(set(transformed)) != len(transformed) or len(transformed) > MAX_GPU_FFT_RANK:
        return None
    if any(d.storage != dtypes.StorageType.GPU_Global for d in (src, out)):
        return None
    block = dense_block_order((src, out), transformed)
    if block is None:
        return None
    order, (istride, ostride) = block
    extents = [src.shape[a] for a in order]
    rest = [a for a in range(len(src.shape)) if a not in transformed]
    if not rest:
        size = functools.reduce(operator.mul, extents, 1)
        return extents, istride, istride * size, ostride, ostride * size, 1
    batched = dense_block_order((src, out), rest)
    if batched is None:
        return None
    batch_order, (idist, odist) = batched
    return extents, istride, idist, ostride, odist, functools.reduce(operator.mul, (src.shape[a] for a in batch_order),
                                                                     1)


def gpu_fft_call(dialect: GpuFftDialect) -> VendorCall:
    """The cuFFT / hipFFT call in ``dialect``'s spelling: a cached ``MakePlanMany64`` plan run on the SDFG's stream.

    The plan lives in the SDFG state and is rebuilt only when its extents change between calls, so a
    transform inside a loop plans once. One buffer on both connectors is the vendor's in-place transform.
    """

    def call(src: data.Data, out: data.Data, transformed: list[int], is_inverse: bool) -> nodes.Tasklet | None:
        from dace.codegen.targets import cpp  # avoid import loop
        layout = gpu_fft_layout(src, out, transformed)
        if layout is None:
            return None
        extents, istride, idist, ostride, odist, batch = layout
        plan = f'{dialect.api}_plan_{next(PLAN_IDS)}'
        key = [cpp.sym2cpp(e) for e in (*extents, istride, idist, ostride, odist, batch)]
        rank = len(extents)
        in_layout = f'__key[{rank}], __key[{rank + 1}]'
        out_layout = f'__key[{rank + 2}], __key[{rank + 3}]'
        kind = f'{dialect.enum}{"Z2Z" if out.dtype == dtypes.complex128 else "C2C"}'
        direction = dialect.inverse if is_inverse else f'{dialect.enum}FORWARD'
        check = (f'auto __check = [](const char *what, {dialect.api}Result result) {{ '
                 f'if (result != {dialect.enum}SUCCESS) throw std::runtime_error(std::string("{dialect.name} ") + '
                 f'what + " failed with status " + std::to_string((int)result)); }};')
        code = f"""
        {check}
        const long long __key[{len(key)}] = {{{', '.join(key)}}};
        bool __same = __state->{plan}_made;
        for (int __k = 0; __same && __k < {len(key)}; ++__k)
            __same = __state->{plan}_key[__k] == __key[__k];
        if (!__same) {{
            if (__state->{plan}_made)
                __check("{dialect.api}Destroy", {dialect.api}Destroy(__state->{plan}));
            __check("{dialect.api}Create", {dialect.api}Create(&__state->{plan}));
            long long __n[{rank}] = {{{', '.join(key[:rank])}}};
            size_t __work_size = 0;
            // The embeds are the extents themselves: a NULL embed makes the vendor ignore stride and distance.
            __check("{dialect.api}MakePlanMany64", {dialect.api}MakePlanMany64(__state->{plan}, {rank}, __n,
                __n, {in_layout}, __n, {out_layout}, {kind}, __key[{rank + 4}], &__work_size));
            for (int __k = 0; __k < {len(key)}; ++__k)
                __state->{plan}_key[__k] = __key[__k];
            __state->{plan}_made = true;
        }}
        __check("{dialect.api}SetStream", {dialect.api}SetStream(__state->{plan}, __dace_current_stream));
        __check("{dialect.api}XtExec", {dialect.api}XtExec(__state->{plan}, (void *)__in, (void *)__out, {direction}));
        """
        return nodes.Tasklet(
            f'{dialect.api}_{"i" if is_inverse else ""}fft', {'__in'}, {'__out'},
            code,
            language=dtypes.Language.CPP,
            state_fields=[f'{dialect.api}Handle {plan};', f'long long {plan}_key[{len(key)}];', f'bool {plan}_made;'],
            code_init=f'__state->{plan}_made = false;',
            code_exit=f'if (__state->{plan}_made) {dialect.api}Destroy(__state->{plan});')

    return call


@library.register_expansion(FFT, 'FFTW3')
class FFTW3FFTExpansion(xf.ExpandTransformation):
    """CPU FFTW3 backend for :class:`FFT`: any rank, axes, strides, real or complex input, any ``norm``."""

    environments = [env.FFTW3]

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return vendor_fft_sdfg(node, parent_state, parent_sdfg, fftw3_call, 'FFTW3')


@library.register_expansion(IFFT, 'FFTW3')
class FFTW3IFFTExpansion(xf.ExpandTransformation):
    """CPU FFTW3 backend for :class:`IFFT`. Same coverage as :class:`FFTW3FFTExpansion`."""

    environments = [env.FFTW3]

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return vendor_fft_sdfg(node, parent_state, parent_sdfg, fftw3_call, 'FFTW3')


class ExpandGPUFFT(xf.ExpandTransformation):
    """The one GPU FFT expansion; a backend subclass names its environment and :class:`GpuFftDialect`."""

    environments = []
    dialect: GpuFftDialect

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return vendor_fft_sdfg(node, parent_state, parent_sdfg, gpu_fft_call(cls.dialect), cls.dialect.name)


@library.register_expansion(FFT, 'cuFFT')
class cuFFTFFTExpansion(ExpandGPUFFT):
    environments = [env.cuFFT]
    dialect = CUFFT


@library.register_expansion(IFFT, 'cuFFT')
class cuFFTIFFTExpansion(ExpandGPUFFT):
    environments = [env.cuFFT]
    dialect = CUFFT


@library.register_expansion(FFT, 'hipFFT')
class hipFFTFFTExpansion(ExpandGPUFFT):
    environments = [env.hipFFT]
    dialect = HIPFFT


@library.register_expansion(IFFT, 'hipFFT')
class hipFFTIFFTExpansion(ExpandGPUFFT):
    environments = [env.hipFFT]
    dialect = HIPFFT


# MKL backend (uses FFTW-compat layer of MKL via the same FFTW3 C ABI)


@library.register_expansion(FFT, 'MKL')
class MKLFFTExpansion(xf.ExpandTransformation):
    """MKL backend: routes through MKL's FFTW3 compatibility layer.

    MKL exposes the FFTW3 C ABI when ``libmkl_*`` is linked instead of
    ``libfftw3``, so the emitted code is identical to :class:`FFTW3FFTExpansion`.
    Selecting this implementation simply pulls in the MKL environment / link
    flags from :mod:`dace.libraries.blas.environments.intel_mkl`.
    """

    from dace.libraries.blas import environments as _blas_envs
    environments = [_blas_envs.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return FFTW3FFTExpansion.expansion(*args, **kwargs)


@library.register_expansion(IFFT, 'MKL')
class MKLIFFTExpansion(xf.ExpandTransformation):
    """MKL backend for :class:`IFFT` (routes through FFTW3-compat ABI)."""

    from dace.libraries.blas import environments as _blas_envs
    environments = [_blas_envs.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return FFTW3IFFTExpansion.expansion(*args, **kwargs)


##################################################################################################
# Helper functions
##################################################################################################


def _get_input_and_output(state: SDFGState, node: nodes.LibraryNode):
    """
    Helper function that returns the input and output arrays of the library node
    """
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn)
    out_edge = next(e for e in state.out_edges(node) if e.src_conn)
    return in_edge.data.data, out_edge.data.data
