# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Implements Forward and Inverse Fast Fourier Transform (FFT) library nodes
"""

from dace import SDFG, SDFGState, data, dtypes, library, nodes, properties, symbolic
from dace import transformation as xf
from dace.libraries.fft import environments as env
from dace.libraries.fft.algorithms import stockham, stockham_real


# Define the library nodes
@library.node
class FFT(nodes.LibraryNode):
    implementations = {}
    default_implementation = "pure"

    factor = properties.SymbolicProperty(
        category="Semantics", desc="Coefficient to multiply outputs. Used for normalization", default=1.0
    )
    axes = properties.ListProperty(
        element_type=int,
        allow_none=True,
        default=None,
        category="Semantics",
        desc="Axes to transform over. If None, all axes are transformed",
    )

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={"_inp"}, outputs={"_out"}, **kwargs)


@library.node
class IFFT(nodes.LibraryNode):
    implementations = {}
    default_implementation = "pure"

    factor = properties.SymbolicProperty(
        category="Semantics", desc="Coefficient to multiply outputs. Used for normalization", default=1.0
    )
    axes = properties.ListProperty(
        element_type=int,
        allow_none=True,
        default=None,
        category="Semantics",
        desc="Axes to transform over. If None, all axes are transformed",
    )

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={"_inp"}, outputs={"_out"}, **kwargs)


@library.node
class RFFT(nodes.LibraryNode):
    """
    Real-to-complex FFT with the half spectrum of ``numpy.fft.rfftn``: the last of the transformed axes is the real
    one, of length ``N`` in the real input and ``N // 2 + 1`` in the complex output.
    """

    implementations = {}
    default_implementation = "pure"

    factor = properties.SymbolicProperty(
        category="Semantics", desc="Coefficient to multiply outputs. Used for normalization", default=1.0
    )
    axes = properties.ListProperty(
        element_type=int,
        allow_none=True,
        default=None,
        category="Semantics",
        desc="Axes to transform over, the real one last. If None, all axes are transformed",
    )

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={"_inp"}, outputs={"_out"}, **kwargs)


@library.node
class IRFFT(nodes.LibraryNode):
    """
    Complex-to-real inverse FFT of a half spectrum, like ``numpy.fft.irfftn``: the last of the transformed axes is the
    real one, of length ``N // 2 + 1`` in the complex input and ``N`` in the real output.
    """

    implementations = {}
    default_implementation = "pure"

    factor = properties.SymbolicProperty(
        category="Semantics", desc="Coefficient to multiply outputs. Used for normalization", default=1.0
    )
    axes = properties.ListProperty(
        element_type=int,
        allow_none=True,
        default=None,
        category="Semantics",
        desc="Axes to transform over, the real one last. If None, all axes are transformed",
    )

    def __init__(self, name, *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, inputs={"_inp"}, outputs={"_out"}, **kwargs)


##################################################################################################
# Native SDFG expansions
##################################################################################################


@library.register_expansion(RFFT, "pure")
class RFFTExpansion(xf.ExpandTransformation):
    """
    Native SDFG expansion: a Stockham FFT of half the length of the real axis (see ``stockham_real``). Requires fixed
    transformed axis lengths.
    """

    environments = []

    @staticmethod
    def expansion(node: RFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _real_expansion(node, parent_state, parent_sdfg, False)


@library.register_expansion(IRFFT, "pure")
class IRFFTExpansion(xf.ExpandTransformation):
    """
    Native SDFG expansion: a Stockham inverse FFT of half the length of the real axis (see ``stockham_real``).
    Requires fixed transformed axis lengths.
    """

    environments = []

    @staticmethod
    def expansion(node: IRFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _real_expansion(node, parent_state, parent_sdfg, True)


def _real_expansion(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, is_inverse: bool) -> SDFG:
    indesc, outdesc = _connected_descriptors(state, sdfg, node)
    axes = real_transformed_axes(node, len(indesc.shape))
    persistent = not _in_map_scope(state, node)
    return stockham_real.real_stockham_sdfg(
        indesc, outdesc, axes, is_inverse, node.factor, sdfg.constants, node.schedule, persistent
    )


@library.register_expansion(FFT, "pure")
class DFTExpansion(xf.ExpandTransformation):
    """
    Native SDFG expansion: a Stockham FFT if the transformed axis lengths are fixed, otherwise a direct DFT
    (one-dimensional inputs only).
    """

    environments = []

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _native_expansion(node, parent_state, parent_sdfg, False)


@library.register_expansion(IFFT, "pure")
class IDFTExpansion(xf.ExpandTransformation):
    """
    Native SDFG expansion: a Stockham FFT if the transformed axis lengths are fixed, otherwise a direct DFT
    (one-dimensional inputs only).
    """

    environments = []

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _native_expansion(node, parent_state, parent_sdfg, True)


@library.register_expansion(FFT, "Stockham")
class StockhamFFTExpansion(xf.ExpandTransformation):
    """
    Stockham autosort FFT with one map per stage, or one kernel with a thread block per transform on a GPU (see
    ``stockham.block_local_fft_sdfg``). Requires fixed transformed axis lengths.
    """

    environments = []

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _stockham_expansion(node, parent_state, parent_sdfg, False)


@library.register_expansion(IFFT, "Stockham")
class StockhamIFFTExpansion(xf.ExpandTransformation):
    """
    Stockham autosort inverse FFT with one map per stage, or one kernel with a thread block per transform on a GPU
    (see ``stockham.block_local_fft_sdfg``). Requires fixed transformed axis lengths.
    """

    environments = []

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        return _stockham_expansion(node, parent_state, parent_sdfg, True)


def _stockham_expansion(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, is_inverse: bool) -> SDFG:
    indesc, outdesc = _connected_descriptors(state, sdfg, node)
    axes = transformed_axes(node, len(indesc.shape))
    persistent = not _in_map_scope(state, node)
    if persistent and _block_local_applies(node, indesc, outdesc, axes, sdfg.constants):
        return stockham.block_local_fft_sdfg(indesc, outdesc, axes, is_inverse, node.factor, sdfg.constants)
    return stockham.stockham_sdfg(
        indesc, outdesc, axes, is_inverse, node.factor, sdfg.constants, node.schedule, persistent
    )


def _block_local_applies(
    node: nodes.LibraryNode, indesc: data.Array, outdesc: data.Array, axes: list[int], constants: dict
) -> bool:
    """
    Returns True if a GPU (I)FFT node is computed by one thread block per transform: if a transform fits into shared
    memory and is contiguous in global memory (it includes the dimension of unit stride), so that the threads of a
    block access consecutive elements.
    """
    if node.schedule != dtypes.ScheduleType.GPU_Device:
        return False
    if any(desc.storage != dtypes.StorageType.GPU_Global for desc in (indesc, outdesc)):
        return False
    innermost = len(outdesc.shape) - 1
    if innermost not in axes or any(desc.strides[innermost] != 1 for desc in (indesc, outdesc)):
        return False
    return outdesc.dtype in (dtypes.complex64, dtypes.complex128) and stockham.fits_block_local(
        outdesc.shape, axes, outdesc.dtype, constants
    )


def _native_expansion(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, is_inverse: bool) -> SDFG:
    from dace.libraries.fft.algorithms import dft  # Lazy import functions

    indesc, outdesc = _connected_descriptors(state, sdfg, node)
    if stockham.can_expand(outdesc.shape, transformed_axes(node, len(outdesc.shape)), sdfg.constants):
        return _stockham_expansion(node, state, sdfg, is_inverse)
    if tuple(indesc.shape) != tuple(outdesc.shape):
        raise NotImplementedError("Zero-padded FFTs require fixed axis lengths")
    if len(indesc.shape) != 1:
        raise NotImplementedError("Native SDFG expansion of N-dimensional (I)FFT requires fixed axis lengths")
    program = dft.idft_explicit if is_inverse else dft.dft_explicit
    return program.to_sdfg(indesc, outdesc, N=indesc.shape[0], factor=node.factor)


##################################################################################################
# cuFFT expansions
##################################################################################################


@library.register_expansion(FFT, "cuFFT")
class cuFFTFFTExpansion(xf.ExpandTransformation):
    environments = [env.cuFFT]
    plan_uid = 0

    @staticmethod
    def expansion(node: FFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if str(node.factor) != "1":
            raise NotImplementedError("Multiplicative post-FFT factors are not yet implemented")
        return _generate_cufft_code(node, indesc, outdesc, parent_sdfg, False)


@library.register_expansion(IFFT, "cuFFT")
class cuFFTIFFTExpansion(xf.ExpandTransformation):
    environments = [env.cuFFT]
    plan_uid = 0

    @staticmethod
    def expansion(node: IFFT, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        input, output = _get_input_and_output(parent_state, node)
        indesc = parent_sdfg.arrays[input]
        outdesc = parent_sdfg.arrays[output]
        if str(node.factor) != "1":
            raise NotImplementedError("Multiplicative post-FFT factors are not yet implemented")
        return _generate_cufft_code(node, indesc, outdesc, parent_sdfg, True)


def _generate_cufft_code(node: nodes.LibraryNode, indesc: data.Data, outdesc: data.Data, sdfg: SDFG, is_inverse: bool):
    from dace.codegen.targets import cpp  # Avoid import loops

    if len(indesc.shape) not in (1, 2, 3):
        raise ValueError("cuFFT only supports 1/2/3-dimensional FFT")
    if len(transformed_axes(node, len(indesc.shape))) != len(indesc.shape):
        raise NotImplementedError("cuFFT expansion only supports transforming all axes")
    if indesc.storage != dtypes.StorageType.GPU_Global:
        raise ValueError("cuFFT implementation requires input array to be on GPU")
    if tuple(indesc.shape) != tuple(outdesc.shape):
        raise NotImplementedError("cuFFT expansion does not support zero-padded inputs")
    if outdesc.storage != dtypes.StorageType.GPU_Global:
        raise ValueError("cuFFT implementation requires output array to be on GPU")

    cufft_type = _types_to_cufft(indesc.dtype, outdesc.dtype)
    init_code = ""
    exit_code = ""
    callsite_code = ""

    # Make a unique name for this plan
    if not is_inverse:
        plan_name = f"fwdplan{cuFFTFFTExpansion.plan_uid}"
        cuFFTFFTExpansion.plan_uid += 1
        direction = "CUFFT_FORWARD"
        tasklet_prefix = ""
    else:
        plan_name = f"invplan{cuFFTIFFTExpansion.plan_uid}"
        cuFFTIFFTExpansion.plan_uid += 1
        direction = "CUFFT_INVERSE"
        tasklet_prefix = "i"

    fields = [
        f"cufftHandle {plan_name};",
    ]
    plan_name = f"__state->{plan_name}"

    init_code += f"""
    cufftCreate(&{plan_name});
    """
    exit_code += f"""
    cufftDestroy({plan_name});
    """

    cdims = ", ".join([cpp.sym2cpp(s) for s in indesc.shape])
    make_plan = f"""
    {{
        size_t __work_size = 0;
        cufftMakePlan{len(indesc.shape)}d({plan_name}, {cdims}, {cufft_type}, /*batch=*/1, &__work_size);
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
    cufftSetStream({plan_name}, __dace_current_stream);
    cufftXtExec({plan_name}, _inp, _out, {direction});
    """

    return nodes.Tasklet(
        f"cufft_{tasklet_prefix}fft",
        {"_inp"},
        {"_out"},
        callsite_code,
        language=dtypes.Language.CPP,
        state_fields=fields,
        code_init=init_code,
        code_exit=exit_code,
    )


##################################################################################################
# Helper functions
##################################################################################################


def transformed_axes(node: nodes.LibraryNode, ndim: int) -> list[int]:
    """
    Returns the normalized (non-negative, sorted) axes that an (I)FFT node transforms.

    :param node: The FFT or IFFT library node.
    :param ndim: The number of dimensions of the transformed array.
    :return: The list of axes.
    """
    if node.axes is None:
        return list(range(ndim))
    axes = sorted({axis % ndim for axis in node.axes})
    if len(axes) != len(node.axes) or any(not -ndim <= axis < ndim for axis in node.axes):
        raise ValueError(f"Invalid FFT axes {node.axes} for a {ndim}-dimensional array")
    return axes


def real_transformed_axes(node: nodes.LibraryNode, ndim: int) -> list[int]:
    """
    Returns the normalized axes that a real (I)FFT node transforms: the complex axes in ascending order, followed by
    the real axis (the last of the node's axes).

    :param node: The RFFT or IRFFT library node.
    :param ndim: The number of dimensions of the transformed array.
    :return: The list of axes.
    """
    if node.axes is None:
        return list(range(ndim))
    if any(not -ndim <= axis < ndim for axis in node.axes):
        raise ValueError(f"Invalid FFT axes {node.axes} for a {ndim}-dimensional array")
    axes = [axis % ndim for axis in node.axes]
    if len(set(axes)) != len(axes):
        raise ValueError(f"Repeated FFT axes are not supported, got {node.axes}")
    return sorted(axes[:-1]) + [axes[-1]]


def _in_map_scope(state: SDFGState, node: nodes.Node) -> bool:
    """
    Returns True if the node is inside a map scope, directly or through the nested SDFGs that contain it.
    """
    while True:
        if state.entry_node(node) is not None:
            return True
        nsdfg_node = state.sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            return False
        state, node = state.sdfg.parent, nsdfg_node


def _connected_descriptors(state: SDFGState, sdfg: SDFG, node: nodes.LibraryNode) -> tuple[data.Array, data.Array]:
    """
    Returns descriptors of the input and output of an (I)FFT node, shaped like the subsets the node is connected to
    (e.g., one row of a matrix for a node inside a map).
    """
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn)
    out_edge = next(e for e in state.out_edges(node) if e.src_conn)
    result = []
    for memlet in (in_edge.data, out_edge.data):
        desc = sdfg.arrays[memlet.data]
        result.append(data.Array(desc.dtype, memlet.subset.size(), strides=desc.strides, storage=desc.storage))
    return result[0], result[1]


def _get_input_and_output(state: SDFGState, node: nodes.LibraryNode):
    """
    Helper function that returns the input and output arrays of the library node
    """
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn)
    out_edge = next(e for e in state.out_edges(node) if e.src_conn)
    return in_edge.data.data, out_edge.data.data


def _types_to_cufft(indtype: dtypes.typeclass, outdtype: dtypes.typeclass):
    typedict = {
        dtypes.float32: "R",
        dtypes.float64: "D",
        dtypes.complex64: "C",
        dtypes.complex128: "Z",
    }
    return f"CUFFT_{typedict[indtype]}2{typedict[outdtype]}"
