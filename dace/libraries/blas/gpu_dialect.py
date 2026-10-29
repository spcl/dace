# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How cuBLAS spells a handle, an error check, a routine and an enum. A node's GPU expansion builds its call from
these; the hipBLAS expansion is the same call renamed (:mod:`dace.libraries.hip_dialect`)."""

from typing import NamedTuple

from dace import dtypes
from dace.libraries.blas import blas_helpers

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
    declarations, _ = blas_helpers.scalar_arguments(dtype, **values)
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
