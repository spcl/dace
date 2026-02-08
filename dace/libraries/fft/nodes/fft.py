# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Implements Forward and Inverse Fast Fourier Transform (FFT) library nodes
"""
import warnings

from dace import data, dtypes, SDFG, SDFGState, symbolic, library, nodes, properties
from dace import transformation as xf
from dace.libraries.fft import environments as env


# Define the library nodes
@library.node
class FFT(nodes.LibraryNode):
    implementations = {}
    default_implementation = 'pure'

    factor = properties.SymbolicProperty(desc='Coefficient to multiply outputs. Used for normalization', default=1.0)

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={'_inp'}, outputs={'_out'}, **kwargs)


@library.node
class IFFT(nodes.LibraryNode):
    implementations = {}
    default_implementation = 'pure'

    factor = properties.SymbolicProperty(desc='Coefficient to multiply outputs. Used for normalization', default=1.0)

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={'_inp'}, outputs={'_out'}, **kwargs)


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
        if len(indesc.shape) != 1:
            raise NotImplementedError('Native SDFG expansion for FFT does not yet support N-dimensional inputs')

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
        if len(indesc.shape) != 1:
            raise NotImplementedError('Native SDFG expansion for IFFT does not yet support N-dimensional inputs')

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
        return _generate_cufft_code(indesc, outdesc, parent_sdfg, False)


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
        return _generate_cufft_code(indesc, outdesc, parent_sdfg, True)


def _generate_cufft_code(indesc: data.Data, outdesc: data.Data, sdfg: SDFG, is_inverse: bool):
    from dace.codegen.targets import cpp  # Avoid import loops
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

    cdims = ', '.join([cpp.sym2cpp(s) for s in indesc.shape])
    make_plan = f'''
    {{
        size_t __work_size = 0;
        cufftMakePlan{len(indesc.shape)}d({plan_name}, {cdims}, {cufft_type}, /*batch=*/1, &__work_size);
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
