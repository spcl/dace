# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Implements Forward and Inverse Fast Fourier Transform (FFT) library nodes
"""

from typing import Any, NamedTuple

from dace import SDFG, SDFGState, data, dtypes, library, nodes, properties, symbolic
from dace import transformation as xf
from dace.libraries.fft import environments as env
from dace.utils import prod


@properties.make_properties
class FourierTransform(nodes.LibraryNode):
    """A transform of ``_inp`` into ``_out`` along every axis, or along ``axis`` only, batched over the others."""

    factor = properties.SymbolicProperty(
        category="Semantics", desc="Coefficient to multiply outputs. Used for normalization", default=1.0
    )
    axis = properties.Property(
        dtype=int, allow_none=True, default=None, desc="The one axis to transform along; None transforms along all."
    )

    def __init__(self, name, *args, schedule=None, axis=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={"_inp"}, outputs={"_out"}, **kwargs)
        self.axis = axis


# Define the library nodes
@library.node
class FFT(FourierTransform):
    implementations = {}
    default_implementation = "pure"


@library.node
class IFFT(FourierTransform):
    implementations = {}
    default_implementation = "pure"


##################################################################################################
# Native SDFG expansions
##################################################################################################


def _expand_dft(node: FourierTransform, parent_state: SDFGState, parent_sdfg: SDFG, inverse: bool) -> SDFG:
    from dace.libraries.fft.algorithms import dft  # Lazy import functions

    input, output = _get_input_and_output(parent_state, node)
    indesc = parent_sdfg.arrays[input]
    outdesc = parent_sdfg.arrays[output]
    if len(indesc.shape) > 1 or node.axis is not None:
        return dft.dft_nd_sdfg(indesc, outdesc, factor=node.factor, inverse=inverse, axis=node.axis)
    program = dft.idft_explicit if inverse else dft.dft_explicit
    return program.to_sdfg(indesc, outdesc, N=indesc.shape[0], factor=node.factor)


@library.register_expansion(FFT, "pure")
class DFTExpansion(xf.ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_dft(node, parent_state, parent_sdfg, inverse=False)


@library.register_expansion(IFFT, "pure")
class IDFTExpansion(xf.ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_dft(node, parent_state, parent_sdfg, inverse=True)


##################################################################################################
# Vendor expansions: cuFFT and hipFFT (one API, two spellings) and FFTW3
##################################################################################################


class BatchedTransform(NamedTuple):
    """A vendor plan's view of a transform: ``count`` transforms of ``length`` elements, each ``stride`` apart, the
    first elements of consecutive transforms ``distance`` apart."""

    length: Any
    count: Any
    stride: Any
    distance: Any


def batched_transform(desc: data.Data, axis: int) -> BatchedTransform | None:
    """The transform along ``axis`` of a C-ordered ``desc``, or ``None`` for an axis other than the first and the
    last, which a vendor plan does not express as one batch."""
    rank = len(desc.shape)
    length, count = desc.shape[axis], prod(s for d, s in enumerate(desc.shape) if d != axis)
    if axis == rank - 1:
        return BatchedTransform(length, count, 1, length)
    if axis == 0:
        return BatchedTransform(length, count, count, 1)
    return None


def vendor_supported(indesc: data.Data, outdesc: data.Data, node: FourierTransform, scales: bool) -> bool:
    """Whether a vendor plan can compute ``node``: rank 1 to 3, C-ordered arrays, an axis it can batch, and a factor
    of 1 unless the expansion ``scales`` its output. Otherwise the pure expansion computes it."""
    rank = len(indesc.shape)
    c_order = [prod(indesc.shape[d + 1 :]) for d in range(rank)]
    return (
        rank in (1, 2, 3)
        and list(indesc.strides) == c_order
        and list(outdesc.strides) == c_order
        and (scales or str(node.factor) == "1")
        and (node.axis is None or batched_transform(indesc, node.axis) is not None)
    )


class GpuFftApi(NamedTuple):
    """Spelling of a GPU FFT API: the function prefix, the constant prefix and the inverse direction."""

    name: str
    prefix: str
    constant: str
    inverse: str


CUFFT = GpuFftApi("cuFFT", "cufft", "CUFFT", "CUFFT_INVERSE")
HIPFFT = GpuFftApi("hipFFT", "hipfft", "HIPFFT", "HIPFFT_BACKWARD")


def _expand_gpu_fft(node, parent_state: SDFGState, parent_sdfg: SDFG, is_inverse: bool, api: GpuFftApi):
    input, output = _get_input_and_output(parent_state, node)
    indesc = parent_sdfg.arrays[input]
    outdesc = parent_sdfg.arrays[output]
    if not vendor_supported(indesc, outdesc, node, scales=False):
        return _expand_dft(node, parent_state, parent_sdfg, is_inverse)
    return _generate_gpu_fft_code(indesc, outdesc, parent_sdfg, is_inverse, node.axis, api)


@library.register_expansion(FFT, "cuFFT")
class cuFFTFFTExpansion(xf.ExpandTransformation):
    environments = [env.cuFFT]
    plan_uid = 0

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_gpu_fft(node, parent_state, parent_sdfg, False, CUFFT)


@library.register_expansion(IFFT, "cuFFT")
class cuFFTIFFTExpansion(xf.ExpandTransformation):
    environments = [env.cuFFT]
    plan_uid = 0

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_gpu_fft(node, parent_state, parent_sdfg, True, CUFFT)


@library.register_expansion(FFT, "hipFFT")
class hipFFTFFTExpansion(xf.ExpandTransformation):
    environments = [env.hipFFT]

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_gpu_fft(node, parent_state, parent_sdfg, False, HIPFFT)


@library.register_expansion(IFFT, "hipFFT")
class hipFFTIFFTExpansion(xf.ExpandTransformation):
    environments = [env.hipFFT]

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_gpu_fft(node, parent_state, parent_sdfg, True, HIPFFT)


def _generate_gpu_fft_code(
    indesc: data.Data, outdesc: data.Data, sdfg: SDFG, is_inverse: bool, axis: int | None, api: GpuFftApi
):
    from dace.codegen.targets import cpp  # Avoid import loops

    p = api.prefix
    if indesc.storage != dtypes.StorageType.GPU_Global:
        raise ValueError(f"{api.name} implementation requires input array to be on GPU")
    if outdesc.storage != dtypes.StorageType.GPU_Global:
        raise ValueError(f"{api.name} implementation requires output array to be on GPU")

    fft_type = _types_to_gpu_fft(indesc.dtype, outdesc.dtype, api)
    init_code = ""
    exit_code = ""
    callsite_code = ""

    # Make a unique name for this plan
    if not is_inverse:
        plan_name = f"fwdplan{cuFFTFFTExpansion.plan_uid}"
        cuFFTFFTExpansion.plan_uid += 1
        direction = f"{api.constant}_FORWARD"
        tasklet_prefix = ""
    else:
        plan_name = f"invplan{cuFFTIFFTExpansion.plan_uid}"
        cuFFTIFFTExpansion.plan_uid += 1
        direction = api.inverse
        tasklet_prefix = "i"

    fields = [
        f"{p}Handle {plan_name};",
    ]
    plan_name = f"__state->{plan_name}"

    init_code += f"""
    {p}Create(&{plan_name});
    """
    exit_code += f"""
    {p}Destroy({plan_name});
    """

    if axis is None:
        cdims = ", ".join([cpp.sym2cpp(s) for s in indesc.shape])
        # Of the MakePlan{N}d functions only the 1-D one takes a batch count
        batch = ", /*batch=*/1" if len(indesc.shape) == 1 else ""
        make_plan = f"""
        {{
            size_t __work_size = 0;
            {p}MakePlan{len(indesc.shape)}d({plan_name}, {cdims}, {fft_type}{batch}, &__work_size);
        }}
        """
    else:
        batch = batched_transform(indesc, axis)
        assert batch is not None  # vendor_supported checked it
        length, count, stride, distance = (cpp.sym2cpp(v) for v in batch)
        make_plan = f"""
        {{
            size_t __work_size = 0;
            int __length = {length};
            {p}MakePlanMany({plan_name}, /*rank=*/1, &__length, /*inembed=*/&__length, {stride}, {distance},
                            /*onembed=*/&__length, {stride}, {distance}, {fft_type}, {count}, &__work_size);
        }}
        """

    # Make plan in init if not symbolic or not data-dependent, otherwise make at callsite.
    symbols_that_change = {s for ise in sdfg.edges() for s in ise.data.assignments.keys()}
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
    callsite_code += f"""
    {p}SetStream({plan_name}, __dace_current_stream);
    {p}XtExec({plan_name}, _inp, _out, {direction});
    """

    return nodes.Tasklet(
        f"{p}_{tasklet_prefix}fft",
        {"_inp"},
        {"_out"},
        callsite_code,
        language=dtypes.Language.CPP,
        state_fields=fields,
        code_init=init_code,
        code_exit=exit_code,
    )


def _expand_fftw3(node, parent_state: SDFGState, parent_sdfg: SDFG, is_inverse: bool):
    from dace.codegen.targets import cpp  # Avoid import loops

    input, output = _get_input_and_output(parent_state, node)
    indesc = parent_sdfg.arrays[input]
    outdesc = parent_sdfg.arrays[output]
    if (
        not vendor_supported(indesc, outdesc, node, scales=True)
        or indesc.dtype not in (dtypes.complex64, dtypes.complex128)
        or outdesc.dtype != indesc.dtype
    ):
        return _expand_dft(node, parent_state, parent_sdfg, is_inverse)

    prefix, complex_t = ("fftw_", "fftw_complex") if indesc.dtype == dtypes.complex128 else ("fftwf_", "fftwf_complex")
    direction = "FFTW_BACKWARD" if is_inverse else "FFTW_FORWARD"
    if node.axis is None:
        cdims = ", ".join(cpp.sym2cpp(s) for s in indesc.shape)
        plan = f"{prefix}plan_dft_{len(indesc.shape)}d({cdims}, ({complex_t} *)_inp, ({complex_t} *)_out, {direction}, "
        plan += "FFTW_ESTIMATE)"
    else:
        batch = batched_transform(indesc, node.axis)
        assert batch is not None  # vendor_supported checked it
        length, count, stride, distance = (cpp.sym2cpp(v) for v in batch)
        plan = (
            f"{prefix}plan_many_dft(1, &__length, {count}, ({complex_t} *)_inp, NULL, {stride}, {distance}, "
            f"({complex_t} *)_out, NULL, {stride}, {distance}, {direction}, FFTW_ESTIMATE)"
        )
    code = f"""
    int __length = {cpp.sym2cpp(indesc.shape[node.axis or 0])};
    {prefix}plan __plan = {plan};
    {prefix}execute(__plan);
    {prefix}destroy_plan(__plan);
    """
    if str(node.factor) != "1":
        code += f"""
    for (long long __i = 0; __i < {cpp.sym2cpp(prod(outdesc.shape))}; ++__i) _out[__i] *= {cpp.sym2cpp(node.factor)};
    """
    return nodes.Tasklet(
        f"fftw3_{'i' if is_inverse else ''}fft", {"_inp"}, {"_out"}, code, language=dtypes.Language.CPP
    )


@library.register_expansion(FFT, "FFTW3")
class FFTW3FFTExpansion(xf.ExpandTransformation):
    environments = [env.FFTW3]

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_fftw3(node, parent_state, parent_sdfg, is_inverse=False)


@library.register_expansion(IFFT, "FFTW3")
class FFTW3IFFTExpansion(xf.ExpandTransformation):
    environments = [env.FFTW3]

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _expand_fftw3(node, parent_state, parent_sdfg, is_inverse=True)


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


def _types_to_gpu_fft(indtype: dtypes.typeclass, outdtype: dtypes.typeclass, api: GpuFftApi):
    typedict = {
        dtypes.float32: "R",
        dtypes.float64: "D",
        dtypes.complex64: "C",
        dtypes.complex128: "Z",
    }
    return f"{api.constant}_{typedict[indtype]}2{typedict[outdtype]}"
