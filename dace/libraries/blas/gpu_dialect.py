# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How cuBLAS spells a handle, an error check, a routine and an enum. A node's GPU expansion builds its call from
these; the hipBLAS expansion is the same call renamed (:mod:`dace.libraries.hip_dialect`)."""

from collections.abc import Callable
from typing import NamedTuple

from dace import dtypes, library
from dace.libraries import hip_dialect
from dace.libraries.blas import blas_helpers, environments
from dace.transformation.transformation import ExpandTransformation

#: The handle variable the environment's ``handle_setup_code`` puts in scope.
HANDLE = "__dace_cublas_handle"


def op(trans: bool) -> str:
    return "CUBLAS_OP_T" if trans else "CUBLAS_OP_N"


def fill(upper: bool) -> str:
    return "CUBLAS_FILL_MODE_UPPER" if upper else "CUBLAS_FILL_MODE_LOWER"


def side(right: bool) -> str:
    return "CUBLAS_SIDE_RIGHT" if right else "CUBLAS_SIDE_LEFT"


class GpuCall(NamedTuple):
    """The pieces a node's GPU call is written from, for one data type."""

    #: The BLAS letter of the data type (``D`` for ``double``).
    letter: str
    #: Cast for a pointer operand, empty for a real type.
    cast: str

    def routine(self, name: str) -> str:
        return f"cublas{self.letter}{name}"

    def pointer(self, connector: str) -> str:
        return f"{self.cast}{connector}"


def gpu_call(dtype: dtypes.typeclass) -> GpuCall | None:
    """The call pieces for ``dtype``, or None if the vendor library has no routines for it."""
    letter = blas_helpers.blas_prefix(dtype)
    if letter is None:
        return None
    cuda_type = blas_helpers.cublas_type_metadata(dtype)[1]
    cast = f"({cuda_type}*)" if dtype.is_complex() else ""
    return GpuCall(letter, cast)


def scalar_pointers(call: GpuCall, dtype: dtypes.typeclass, **values) -> tuple[str, dict[str, str]]:
    """Declares each of ``values`` as a host scalar of ``dtype``; returns the declarations and, per name, the pointer
    argument passing it to the vendor routine, which takes every scalar by address."""
    declarations, arguments = blas_helpers.scalar_arguments(dtype, **values)
    return declarations, {name: call.pointer(f"&__{name}") for name in values}


def required_layout(node, *matrices: blas_helpers.Operand) -> blas_helpers.MatrixLayout:
    """The common row- or column-major layout of ``matrices``. The vendor libraries take nothing else, so any other
    layout is an error rather than a silent fallback."""
    layout = blas_helpers.matrix_layout(*matrices)
    if layout is None:
        strides = [list(m.strides) for m in matrices]
        raise ValueError(
            f"{node.label}: cuBLAS/hipBLAS need every matrix in one row- or column-major layout, got strides {strides}"
        )
    return layout


def call_code(call: GpuCall, environment: type, node, scalars: str, routine_call: str) -> str:
    """The tasklet body running ``routine_call``: handle setup, then the call with its scalars read from the host.

    The handle is created in device pointer mode (``dace_cublas.h``), so a call passing a host address would have the
    GPU dereference it.
    """
    check = "dace::blas::CheckCublasError"
    return (
        environment.handle_setup_code(node)
        + f"{check}(cublasSetPointerMode({HANDLE}, CUBLAS_POINTER_MODE_HOST));\n"
        + scalars
        + f"{check}({routine_call});\n"
        + f"{check}(cublasSetPointerMode({HANDLE}, CUBLAS_POINTER_MODE_DEVICE));\n"
    )


def vendor_expansions(node_name: str, call: Callable) -> dict[str, type[ExpandTransformation]]:
    """The OpenBLAS, MKL, cuBLAS and hipBLAS expansions of a node, each validating the node and returning
    ``call(node, state, sdfg, gpu_environment)``; the CPU libraries get ``None``, as both are CBLAS. hipBLAS is the
    cuBLAS expansion renamed."""
    backends = {
        "OpenBLAS": (environments.openblas.OpenBLAS, False),
        "MKL": (environments.intel_mkl.IntelMKL, False),
        "cuBLAS": (environments.cublas.cuBLAS, True),
    }
    expansions = {}
    for implementation, (environment, gpu) in backends.items():

        def expansion(node, parent_state, parent_sdfg, *, bound_environment=environment if gpu else None, **kwargs):
            node.validate(parent_sdfg, parent_state)
            return call(node, parent_state, parent_sdfg, bound_environment)

        name = f"Expand{node_name}{implementation[0].upper()}{implementation[1:]}"
        attributes = {
            "environments": [environment],
            "expansion": staticmethod(expansion),
            "__module__": call.__module__,
        }
        expansions[implementation] = library.expansion(type(name, (ExpandTransformation,), attributes))
    expansions["hipBLAS"] = hip_dialect.hip_expansion(
        expansions["cuBLAS"], environments.hipblas.hipBLAS, hip_dialect.BLAS
    )
    return expansions


# The cuBLAS vocabulary the GEMM-family expansions spell their calls in; hipBLAS renames them.


class GpuBlasDialect(NamedTuple):
    """How one vendor GPU BLAS spells the things an expansion emits."""

    #: Human name, for a fallback warning that has to say which backend refused.
    name: str
    #: The handle variable the environment's ``handle_setup_code`` puts in scope.
    handle: str
    #: The ``__state`` field holding it, for the preallocated alpha/beta constants.
    handle_field: str
    #: Error-check function wrapping every call.
    check_error: str
    #: Pointer-mode setter and its two modes, for a node that takes alpha/beta by pointer.
    set_pointer_mode: str
    pointer_host: str
    pointer_device: str
    #: ``('D', 'gemv') -> 'cublasDgemv'``.
    func: Callable[[str, str], str]
    #: Strided-batched GEMM: ``cublasDgemmStridedBatched``.
    strided_batched: Callable[[str], str]
    #: For a node that already holds the COMPLETE routine name including its type letter, because
    #: the letter is not a plain prefix there -- ``Idamax``, ``Scnrm2``, ``Dzasum``. Passing such a
    #: name to :attr:`func` would prepend the letter a second time (``cublasDDnrm2``, measured).
    routine: Callable[[str], str]
    #: The C type a pointer operand is cast to, from the CUDA spelling
    #: :func:`~dace.libraries.blas.blas_helpers.cublas_type_metadata` returns. Only the complex
    #: types differ.
    ctype: Callable[[str], str]
    #: Enum spellings. ``op`` takes 'N' or 'T'; the rest take the boolean the node stores.
    op: Callable[[str], str]
    fill: Callable[[bool], str]
    side: Callable[[bool], str]
    diag: Callable[[bool], str]


CUBLAS = GpuBlasDialect(
    name="cuBLAS",
    handle="__dace_cublas_handle",
    handle_field="cublas_handle",
    check_error="dace::blas::CheckCublasError",
    set_pointer_mode="cublasSetPointerMode",
    pointer_host="CUBLAS_POINTER_MODE_HOST",
    pointer_device="CUBLAS_POINTER_MODE_DEVICE",
    func=lambda letter, routine: f"cublas{letter}{routine}",
    routine=lambda name: f"cublas{name}",
    strided_batched=lambda name: f"cublas{name}StridedBatched",
    ctype=lambda cuda_type: cuda_type,
    op=lambda mode: f"CUBLAS_OP_{mode}",
    fill=lambda upper: "CUBLAS_FILL_MODE_UPPER" if upper else "CUBLAS_FILL_MODE_LOWER",
    side=lambda right: "CUBLAS_SIDE_RIGHT" if right else "CUBLAS_SIDE_LEFT",
    diag=lambda unit: "CUBLAS_DIAG_UNIT" if unit else "CUBLAS_DIAG_NON_UNIT",
)


def host_scalar_mode(dialect: GpuBlasDialect, body: str) -> str:
    """Wrap ``body`` so the vendor reads its scalar arguments from the HOST.

    The handle is created in DEVICE pointer mode (``dace_cublas.h``), so a
    call passing ``&alpha`` from the host stack has the GPU dereference a host address and fault.
    """
    return (
        f"{dialect.check_error}({dialect.set_pointer_mode}({dialect.handle}, {dialect.pointer_host}));\n"
        f"{body}\n"
        f"{dialect.check_error}({dialect.set_pointer_mode}({dialect.handle}, {dialect.pointer_device}));\n"
    )
