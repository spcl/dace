# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vendor vocabulary a GPU BLAS expansion needs, in one object per backend.

Every ``Expand<X>GPUBLAS`` body is identical across cuBLAS and rocBLAS -- same operands, same
leading dimensions, same fallbacks -- and differs only in how the two spell a handle, an error
check, a routine and an enum. Carrying those five or six strings on each node's pair of subclasses
put the same table in fifteen files; when a spelling turned out to be wrong it had to be corrected
in all of them. They live here instead, so a node's backend subclass is the environment plus one
``dialect =`` line.

The two are NOT interchangeable, which is why this is a table and not a prefix substitution:
cuBLAS is camel-case with upper-case enum constants (``cublasDgemv``, ``CUBLAS_OP_T``) while
rocBLAS is snake_case throughout with lower-case enum values (``rocblas_dgemv``,
``rocblas_operation_transpose``), and the transpose flag is not even the same shape of token.
"""
from typing import Callable, NamedTuple


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
    #: ``('D', 'gemv') -> 'cublasDgemv'`` / ``'rocblas_dgemv'``.
    func: Callable[[str, str], str]
    #: Strided-batched GEMM. Not derivable from :attr:`routine`: cuBLAS suffixes the camel-case
    #: name (``cublasDgemmStridedBatched``) while rocBLAS extends the snake_case one
    #: (``rocblas_dgemm_strided_batched``), so the two are different SHAPES, not different cases.
    strided_batched: Callable[[str], str]
    #: For a node that already holds the COMPLETE routine name including its type letter, because
    #: the letter is not a plain prefix there -- ``Idamax``, ``Scnrm2``, ``Dzasum``. Passing such a
    #: name to :attr:`func` would prepend the letter a second time (``cublasDDnrm2``, measured).
    routine: Callable[[str], str]
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
    op=lambda mode: f"CUBLAS_OP_{mode}",
    fill=lambda upper: "CUBLAS_FILL_MODE_UPPER" if upper else "CUBLAS_FILL_MODE_LOWER",
    side=lambda right: "CUBLAS_SIDE_RIGHT" if right else "CUBLAS_SIDE_LEFT",
    diag=lambda unit: "CUBLAS_DIAG_UNIT" if unit else "CUBLAS_DIAG_NON_UNIT",
)

ROCBLAS = GpuBlasDialect(
    name="rocBLAS",
    handle="__dace_rocblas_handle",
    handle_field="rocblas_handle",
    check_error="dace::blas::CheckRocblasError",
    set_pointer_mode="rocblas_set_pointer_mode",
    pointer_host="rocblas_pointer_mode_host",
    pointer_device="rocblas_pointer_mode_device",
    func=lambda letter, routine: f"rocblas_{letter.lower()}{routine.lower()}",
    routine=lambda name: f"rocblas_{name.lower()}",
    strided_batched=lambda name: f"rocblas_{name.lower()}_strided_batched",
    op=lambda mode: "rocblas_operation_transpose" if mode == "T" else "rocblas_operation_none",
    fill=lambda upper: "rocblas_fill_upper" if upper else "rocblas_fill_lower",
    side=lambda right: "rocblas_side_right" if right else "rocblas_side_left",
    diag=lambda unit: "rocblas_diagonal_unit" if unit else "rocblas_diagonal_non_unit",
)


def host_scalar_mode(dialect: GpuBlasDialect, body: str) -> str:
    """Wrap ``body`` so the vendor reads its scalar arguments from the HOST.

    Both handles are created in DEVICE pointer mode (``dace_cublas.h`` / ``dace_rocblas.h``), so a
    call passing ``&alpha`` from the host stack has the GPU dereference a host address and fault.
    """
    return (f"{dialect.check_error}({dialect.set_pointer_mode}({dialect.handle}, {dialect.pointer_host}));\n"
            f"{body}\n"
            f"{dialect.check_error}({dialect.set_pointer_mode}({dialect.handle}, {dialect.pointer_device}));\n")
