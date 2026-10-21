# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How cuBLAS and rocBLAS spell a handle, an error check, a routine and an enum, one object per backend.

The two are not a prefix substitution of each other: cuBLAS is camel-case with upper-case enum constants
(``cublasDsymm``, ``CUBLAS_SIDE_LEFT``), rocBLAS is snake_case with lower-case enum values (``rocblas_dsymm``,
``rocblas_side_left``). A node's GPU expansion builds one call and asks the dialect for each spelling.
"""

from collections.abc import Callable
from typing import NamedTuple

from dace import dtypes
from dace.libraries.blas import blas_helpers


class GpuBlasDialect(NamedTuple):
    """How one vendor GPU BLAS spells the things an expansion emits."""

    #: The handle variable the environment's ``handle_setup_code`` puts in scope.
    handle: str
    #: Error-check function wrapping every call.
    check_error: str
    #: Pointer-mode setter and its two modes.
    set_pointer_mode: str
    pointer_host: str
    pointer_device: str
    #: ``('D', 'symm') -> 'cublasDsymm'`` / ``'rocblas_dsymm'``.
    func: Callable[[str, str], str]
    #: The C type a complex pointer operand is cast to, from the CUDA spelling of
    #: :func:`~dace.libraries.blas.blas_helpers.cublas_type_metadata`.
    ctype: Callable[[str], str]
    #: Enum spellings: ``op`` takes the transposition, ``fill`` the upper triangle, ``side`` the right side.
    op: Callable[[bool], str]
    fill: Callable[[bool], str]
    side: Callable[[bool], str]


CUBLAS = GpuBlasDialect(
    handle="__dace_cublas_handle",
    check_error="dace::blas::CheckCublasError",
    set_pointer_mode="cublasSetPointerMode",
    pointer_host="CUBLAS_POINTER_MODE_HOST",
    pointer_device="CUBLAS_POINTER_MODE_DEVICE",
    func=lambda letter, routine: f"cublas{letter}{routine}",
    ctype=lambda cuda_type: cuda_type,
    op=lambda trans: "CUBLAS_OP_T" if trans else "CUBLAS_OP_N",
    fill=lambda upper: "CUBLAS_FILL_MODE_UPPER" if upper else "CUBLAS_FILL_MODE_LOWER",
    side=lambda right: "CUBLAS_SIDE_RIGHT" if right else "CUBLAS_SIDE_LEFT",
)

ROCBLAS = GpuBlasDialect(
    handle="__dace_rocblas_handle",
    check_error="dace::blas::CheckRocblasError",
    set_pointer_mode="rocblas_set_pointer_mode",
    pointer_host="rocblas_pointer_mode_host",
    pointer_device="rocblas_pointer_mode_device",
    func=lambda letter, routine: f"rocblas_{letter.lower()}{routine.lower()}",
    ctype=blas_helpers.rocblas_type,
    op=lambda trans: "rocblas_operation_transpose" if trans else "rocblas_operation_none",
    fill=lambda upper: "rocblas_fill_upper" if upper else "rocblas_fill_lower",
    side=lambda right: "rocblas_side_right" if right else "rocblas_side_left",
)


class GpuCall(NamedTuple):
    """The pieces a node's GPU call is written from, for one dialect and data type."""

    dialect: GpuBlasDialect
    #: The BLAS letter of the data type (``D`` for ``double``).
    letter: str
    #: Cast for a pointer operand, empty for a real type: rocBLAS declares complex parameters as its own type.
    cast: str

    def routine(self, name: str) -> str:
        return self.dialect.func(self.letter, name)

    def pointer(self, connector: str) -> str:
        return f"{self.cast}{connector}"


def gpu_call(dialect: GpuBlasDialect, dtype: dtypes.typeclass) -> GpuCall | None:
    """The call pieces for ``dtype``, or None if the vendor library has no routines for it."""
    letter = blas_helpers.blas_prefix(dtype)
    if letter is None:
        return None
    cuda_type = blas_helpers.cublas_type_metadata(dtype)[1]
    cast = f"({dialect.ctype(cuda_type)}*)" if dtype.is_complex() else ""
    return GpuCall(dialect, letter, cast)


def scalar_pointers(call: GpuCall, dtype: dtypes.typeclass, **values) -> tuple[str, dict[str, str]]:
    """Declares each of ``values`` as a host scalar of ``dtype``; returns the declarations and, per name, the pointer
    argument passing it to the vendor routine, which takes every scalar by address."""
    declarations = "".join(f"{dtype.ctype} __{name} = {dtype.ctype}({value});\n" for name, value in values.items())
    return declarations, {name: call.pointer(f"&__{name}") for name in values}


def required_layout(node, *matrices: blas_helpers.Operand) -> blas_helpers.MatrixLayout:
    """The common row- or column-major layout of ``matrices``. The vendor libraries take nothing else, so any other
    layout is an error rather than a silent fallback."""
    layout = blas_helpers.matrix_layout(*matrices)
    if layout is None:
        strides = [list(m.strides) for m in matrices]
        raise ValueError(
            f"{node.label}: cuBLAS/rocBLAS need every matrix in one row- or column-major layout, got strides {strides}"
        )
    return layout


def call_code(call: GpuCall, environment: type, node, scalars: str, routine_call: str) -> str:
    """The tasklet body running ``routine_call``: handle setup, then the call with its scalars read from the host.

    Both handles are created in device pointer mode (``dace_cublas.h`` / ``dace_rocblas.h``), so a call passing a
    host address would have the GPU dereference it.
    """
    d = call.dialect
    return (
        environment.handle_setup_code(node)
        + f"{d.check_error}({d.set_pointer_mode}({d.handle}, {d.pointer_host}));\n"
        + scalars
        + f"{d.check_error}({routine_call});\n"
        + f"{d.check_error}({d.set_pointer_mode}({d.handle}, {d.pointer_device}));\n"
    )
