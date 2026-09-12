# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Implements Forward and Inverse Fast Fourier Transform (FFT) library nodes
"""

from typing import List, Optional, Sequence

from dace import data, dtypes, SDFG, SDFGState, symbolic, library, nodes, properties
from dace import transformation as xf
from dace.libraries.fft import environments as env


def normalize_fft_axes(ndim: int, axes: Optional[Sequence[int]]) -> List[int]:
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

        return dft.dft_explicit.to_sdfg(indesc, outdesc, N=indesc.shape[0], factor=node.factor)


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

        return dft.idft_explicit.to_sdfg(indesc, outdesc, N=indesc.shape[0], factor=node.factor)


##################################################################################################
# cuFFT expansions
##################################################################################################


@library.register_expansion(FFT, 'cuFFT')
class cuFFTFFTExpansion(xf.ExpandTransformation):
    environments = [env.cuFFT]
    plan_uid = 0

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if str(node.factor) != '1':
            raise NotImplementedError('Multiplicative post-FFT factors are not yet implemented')
        return _generate_cufft_code(indesc, outdesc, parent_sdfg, False, node.axes)


@library.register_expansion(IFFT, 'cuFFT')
class cuFFTIFFTExpansion(xf.ExpandTransformation):
    environments = [env.cuFFT]
    plan_uid = 0

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if str(node.factor) != '1':
            raise NotImplementedError('Multiplicative post-FFT factors are not yet implemented')
        return _generate_cufft_code(indesc, outdesc, parent_sdfg, True, node.axes)


def _generate_cufft_code(indesc: data.Data,
                         outdesc: data.Data,
                         sdfg: SDFG,
                         is_inverse: bool,
                         axes: Optional[List[int]] = None):
    from dace.codegen.targets import cpp  # Avoid import loops
    if axes is not None and len(axes) != 1:
        raise NotImplementedError(f'cuFFT expansion transforms every axis or a single one (got axes={axes})')
    axis = None if axes is None else axes[0]
    if len(indesc.shape) not in (1, 2, 3):
        raise ValueError('cuFFT only supports 1/2/3-dimensional FFT')
    if indesc.storage != dtypes.StorageType.GPU_Global:
        raise ValueError('cuFFT implementation requires input array to be on GPU')
    if outdesc.storage != dtypes.StorageType.GPU_Global:
        raise ValueError('cuFFT implementation requires output array to be on GPU')

    cufft_type = _types_to_cufft(indesc.dtype, outdesc.dtype)
    init_code = ''
    exit_code = ''
    callsite_code = ''

    # Make a unique name for this plan
    if not is_inverse:
        plan_name = f'fwdplan{cuFFTFFTExpansion.plan_uid}'
        cuFFTFFTExpansion.plan_uid += 1
        direction = 'CUFFT_FORWARD'
        tasklet_prefix = ''
    else:
        plan_name = f'invplan{cuFFTIFFTExpansion.plan_uid}'
        cuFFTIFFTExpansion.plan_uid += 1
        direction = 'CUFFT_INVERSE'
        tasklet_prefix = 'i'

    fields = [
        f'cufftHandle {plan_name};',
    ]
    plan_name = f'__state->{plan_name}'

    init_code += f'''
    cufftCreate(&{plan_name});
    '''
    exit_code += f'''
    cufftDestroy({plan_name});
    '''

    # Axis-aware lowering: ``cufftMakePlanMany`` (which fits both the
    # full-N-D case via ``rank>=2`` and the batched-1-D case via
    # ``rank=1, howmany=...``).  For the simple N-D case we keep the old
    # ``cufftMakePlan{N}d`` since its ABI is leaner.
    if axis is None:
        cdims = ', '.join([cpp.sym2cpp(s) for s in indesc.shape])
        # ``cufftMakePlan1d`` is the only variant that takes a ``batch`` argument;
        # the 2-D / 3-D entry points do not.  Passing batch=1 to the higher-rank
        # plans raised a "too many arguments" build error.
        batch_arg = ", /*batch=*/1" if len(indesc.shape) == 1 else ""
        make_plan = f'''
        {{
            size_t __work_size = 0;
            cufftMakePlan{len(indesc.shape)}d({plan_name}, {cdims}, {cufft_type}{batch_arg}, &__work_size);
        }}
        '''
    else:
        ndim = len(indesc.shape)
        axis_norm = int(axis) if axis >= 0 else ndim + int(axis)
        if axis_norm not in (0, ndim - 1):
            raise NotImplementedError(f"cuFFT axis-aware expansion only handles axis=0 or axis=ndim-1 "
                                      f"(got axis={axis} on shape {indesc.shape}); intermediate axes need "
                                      f"``cufftXtMakePlanMany`` with explicit per-dim strides.")
        n_sym = indesc.shape[axis_norm]
        other_dims = [d for i, d in enumerate(indesc.shape) if i != axis_norm]
        howmany_sym = 1
        for d in other_dims:
            howmany_sym = howmany_sym * d
        if axis_norm == ndim - 1:
            stride_sym, dist_sym = 1, n_sym
        else:
            stride_sym, dist_sym = howmany_sym, 1
        make_plan = f'''
        {{
            size_t __work_size = 0;
            int __n_arr[1] = {{ (int){cpp.sym2cpp(n_sym)} }};
            int __stride = (int){cpp.sym2cpp(stride_sym)};
            int __dist = (int){cpp.sym2cpp(dist_sym)};
            int __howmany = (int){cpp.sym2cpp(howmany_sym)};
            cufftMakePlanMany({plan_name}, /*rank=*/1, __n_arr,
                              /*inembed=*/NULL, __stride, __dist,
                              /*onembed=*/NULL, __stride, __dist,
                              {cufft_type}, __howmany, &__work_size);
        }}
        '''

    # Make plan in init if not symbolic or not data-dependent, otherwise make at callsite.
    symbols_that_change = set(s for ise in sdfg.edges() for s in ise.data.assignments.keys())
    symbols_that_change &= set(map(str, sdfg.symbols.keys()))

    def _fsyms(x):
        if symbolic.issymbolic(x):
            return set(map(str, x.free_symbols))
        return set()

    if symbols_that_change and any(_fsyms(s) & symbols_that_change for s in indesc.shape):
        callsite_code += make_plan
    else:
        init_code += make_plan

    # Execute plan
    callsite_code += f'''
    cufftSetStream({plan_name}, __dace_current_stream);
    cufftXtExec({plan_name}, _inp, _out, {direction});
    '''

    return nodes.Tasklet(f'cufft_{tasklet_prefix}fft', {'_inp'}, {'_out'},
                         callsite_code,
                         language=dtypes.Language.CPP,
                         state_fields=fields,
                         code_init=init_code,
                         code_exit=exit_code)


##################################################################################################
# FFTW3 expansions
##################################################################################################


@library.register_expansion(FFT, 'FFTW3')
class FFTW3FFTExpansion(xf.ExpandTransformation):
    """CPU FFTW3 backend for :class:`FFT` over complex64 / complex128, for any ``node.axes`` without repeats."""

    environments = [env.FFTW3]

    @staticmethod
    def expansion(node: 'FFT', parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if str(node.factor) != '1':
            raise NotImplementedError('Multiplicative post-FFT factors are not yet implemented')
        return _generate_fftw3_code(indesc, outdesc, is_inverse=False, axes=node.axes)


@library.register_expansion(IFFT, 'FFTW3')
class FFTW3IFFTExpansion(xf.ExpandTransformation):
    """CPU FFTW3 backend for :class:`IFFT`. Same shape/dtype/axes constraints as :class:`FFTW3FFTExpansion`."""

    environments = [env.FFTW3]

    @staticmethod
    def expansion(node: 'IFFT', parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if str(node.factor) != '1':
            raise NotImplementedError('Multiplicative post-FFT factors are not yet implemented')
        return _generate_fftw3_code(indesc, outdesc, is_inverse=True, axes=node.axes)


def _generate_fftw3_code(indesc: data.Data,
                         outdesc: data.Data,
                         is_inverse: bool,
                         axes: Optional[List[int]] = None) -> nodes.Tasklet:
    """Emit a self-contained ``fftw_plan_*`` -> ``execute`` -> ``destroy_plan`` tasklet.

    A full transform of a rank 1-3 array drives ``fftw_plan_dft_{rank}d`` and a single leading or trailing axis
    drives ``fftw_plan_many_dft``. Any other ``axes`` drives ``fftw_plan_guru_dft``: the transformed axes are the
    plan dimensions, every other axis is a ``howmany`` batch dimension, and both step by the descriptors' strides.
    """
    from dace.codegen.targets import cpp  # avoid import loop

    if indesc.dtype not in (dtypes.complex64, dtypes.complex128):
        raise ValueError(f'FFTW3 expansion requires complex inputs (got {indesc.dtype})')
    if outdesc.dtype != indesc.dtype:
        raise ValueError('FFTW3 expansion requires matching input/output dtypes')

    if indesc.dtype == dtypes.complex128:
        prefix, complex_t = 'fftw_', 'fftw_complex'
    else:
        prefix, complex_t = 'fftwf_', 'fftwf_complex'
    direction = 'FFTW_BACKWARD' if is_inverse else 'FFTW_FORWARD'
    ndim = len(indesc.shape)

    if axes is None and ndim in (1, 2, 3):
        cdims = ', '.join(cpp.sym2cpp(s) for s in indesc.shape)
        plan = f'{prefix}plan_dft_{ndim}d({cdims}, ({complex_t}*)_inp, ({complex_t}*)_out, {direction}, FFTW_ESTIMATE)'
        code = f"""
        {{
            {prefix}plan __plan = {plan};
            {prefix}execute(__plan);
            {prefix}destroy_plan(__plan);
        }}
        """
    elif axes is not None and len(axes) == 1 and axes[0] in (0, ndim - 1):
        axis = axes[0]
        n_sym = indesc.shape[axis]
        # Consecutive transforms along the last axis are contiguous; along axis 0 they interleave with stride howmany.
        howmany_sym = 1
        for d in (d for i, d in enumerate(indesc.shape) if i != axis):
            howmany_sym = howmany_sym * d
        if axis == ndim - 1:
            istride, idist = 1, n_sym
        else:
            istride, idist = howmany_sym, 1
        code = f"""
        {{
            int __n = {cpp.sym2cpp(n_sym)};
            {prefix}plan __plan = {prefix}plan_many_dft(
                /*rank=*/1, &__n, /*howmany=*/{cpp.sym2cpp(howmany_sym)},
                ({complex_t}*)_inp, /*inembed=*/NULL,
                /*istride=*/{cpp.sym2cpp(istride)}, /*idist=*/{cpp.sym2cpp(idist)},
                ({complex_t}*)_out, /*onembed=*/NULL,
                /*ostride=*/{cpp.sym2cpp(istride)}, /*odist=*/{cpp.sym2cpp(idist)},
                {direction}, FFTW_ESTIMATE);
            {prefix}execute(__plan);
            {prefix}destroy_plan(__plan);
        }}
        """
    else:
        transformed = list(range(ndim)) if axes is None else list(axes)
        if not transformed or len(set(transformed)) != len(transformed):
            raise NotImplementedError(f'FFTW3 expansion transforms each of one or more axes once (got axes={axes})')
        batch = [d for d in range(ndim) if d not in transformed]

        def iodims(dims: List[int]) -> str:
            return ', '.join(f'{{(int)({cpp.sym2cpp(indesc.shape[d])}), (int)({cpp.sym2cpp(indesc.strides[d])}), '
                             f'(int)({cpp.sym2cpp(outdesc.strides[d])})}}' for d in dims)

        batch_decl = f'{prefix}iodim __howmany[{len(batch)}] = {{{iodims(batch)}}};' if batch else ''
        code = f"""
        {{
            {prefix}iodim __dims[{len(transformed)}] = {{{iodims(transformed)}}};
            {batch_decl}
            {prefix}plan __plan = {prefix}plan_guru_dft({len(transformed)}, __dims,
                {len(batch)}, {'__howmany' if batch else 'NULL'},
                ({complex_t}*)_inp, ({complex_t}*)_out, {direction}, FFTW_ESTIMATE);
            {prefix}execute(__plan);
            {prefix}destroy_plan(__plan);
        }}
        """

    name = f'fftw3_{"i" if is_inverse else ""}fft'
    return nodes.Tasklet(name, {'_inp'}, {'_out'}, code, language=dtypes.Language.CPP)


##################################################################################################
# MKL backend (uses FFTW-compat layer of MKL via the same FFTW3 C ABI)
##################################################################################################


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


def _types_to_cufft(indtype: dtypes.typeclass, outdtype: dtypes.typeclass):
    typedict = {
        dtypes.float32: 'R',
        dtypes.float64: 'D',
        dtypes.complex64: 'C',
        dtypes.complex128: 'Z',
    }
    return f'CUFFT_{typedict[indtype]}2{typedict[outdtype]}'
