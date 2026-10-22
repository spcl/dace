# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import dtypes


def MPI_DDT(dtype):
    mpi_dtype_str = "MPI_BYTE"
    if dtype == dace.dtypes.float32:
        mpi_dtype_str = "MPI_FLOAT"
    elif dtype == dace.dtypes.float64:
        mpi_dtype_str = "MPI_DOUBLE"
    elif dtype == dace.dtypes.complex64:
        mpi_dtype_str = "MPI_C_FLOAT_COMPLEX"
    elif dtype == dace.dtypes.complex128:
        mpi_dtype_str = "MPI_C_DOUBLE_COMPLEX"
    elif dtype == dace.dtypes.bool_:
        mpi_dtype_str = "MPI_C_BOOL"
    elif dtype == dace.dtypes.int8:
        mpi_dtype_str = "MPI_SIGNED_CHAR"
    elif dtype == dace.dtypes.uint8:
        mpi_dtype_str = "MPI_UNSIGNED_CHAR"
    elif dtype == dace.dtypes.int16:
        mpi_dtype_str = "MPI_SHORT"
    elif dtype == dace.dtypes.int32:
        mpi_dtype_str = "MPI_INT"
    elif dtype == dace.dtypes.int64:
        mpi_dtype_str = "MPI_LONG_LONG"
    elif dtype == dace.dtypes.uint16:
        mpi_dtype_str = "MPI_UNSIGNED_SHORT"
    elif dtype == dace.dtypes.uint32:
        mpi_dtype_str = "MPI_UNSIGNED"
    elif dtype == dace.dtypes.uint64:
        mpi_dtype_str = "MPI_UNSIGNED_LONG_LONG"
    else:
        raise ValueError("DDT of " + str(dtype) + " not supported yet.")
    return mpi_dtype_str


def is_access_contiguous(memlet, data):
    """Whether ``memlet`` accesses a single contiguous run of ``data``'s memory. Thin adapter over
    :meth:`dace.subsets.Subset.is_contiguous_subset` (the shared, stride-aware contiguity check --
    correct for C, Fortran, and packed permutations), plus the restriction that a reshaping send
    (``other_subset``) is unsupported."""
    if memlet.other_subset is not None:
        raise ValueError("Other subset must be None, reshape in send not supported")
    return memlet.subset.is_contiguous_subset(data)


def is_column_major(desc) -> bool:
    """Whether ``desc`` is a packed column-major (Fortran) array that is not also row-major (a vector is both)."""
    return isinstance(desc, dace.data.Array) and desc.is_packed_fortran_strides() and not desc.is_packed_c_strides()


def create_vector_ddt(memlet, data):
    from dace.codegen.targets.cpp import sym2cpp  # avoid import loop

    if is_access_contiguous(memlet, data):
        return None
    if len(data.shape) != 2:
        raise ValueError("Dimensionality of access not supported atm.")
    # Rows run along the dimension of unit stride: the last of a row-major array, the first of a column-major one
    inner, outer = (0, 1) if is_column_major(data) else (1, 0)
    ddt = dict()
    sizes = memlet.subset.size_exact()
    ddt["blocklen"] = sym2cpp(sizes[inner])
    ddt["oldtype"] = str(MPI_DDT(data.dtype))
    ddt["count"] = sym2cpp(memlet.subset.num_elements_exact() / sizes[inner])
    ddt["stride"] = sym2cpp(data.strides[outer])
    return ddt


#: Canonical reduction operator names, from MPI operator names and :class:`dace.dtypes.ReductionType`. The average
#: (``avg``) has no MPI operator: MPI expansions sum and divide by the number of processes (see :func:`is_average`).
_REDUCTION_OPS = {
    "MPI_SUM": "sum",
    "MPI_PROD": "prod",
    "MPI_MAX": "max",
    "MPI_MIN": "min",
    "MPI_LAND": "land",
    "MPI_LOR": "lor",
    "MPI_LXOR": "lxor",
    "MPI_BAND": "band",
    "MPI_BOR": "bor",
    "MPI_BXOR": "bxor",
    dtypes.ReductionType.Sum.name: "sum",
    dtypes.ReductionType.Product.name: "prod",
    dtypes.ReductionType.Max.name: "max",
    dtypes.ReductionType.Min.name: "min",
    dtypes.ReductionType.Logical_And.name: "land",
    dtypes.ReductionType.Logical_Or.name: "lor",
    dtypes.ReductionType.Logical_Xor.name: "lxor",
    dtypes.ReductionType.Bitwise_And.name: "band",
    dtypes.ReductionType.Bitwise_Or.name: "bor",
    dtypes.ReductionType.Bitwise_Xor.name: "bxor",
    "Avg": "avg",
}

_NCCL_OPS = {"sum": "ncclSum", "prod": "ncclProd", "max": "ncclMax", "min": "ncclMin", "avg": "ncclAvg"}

_NCCL_TYPES = {
    dtypes.int8: "ncclInt8",
    dtypes.uint8: "ncclUint8",
    dtypes.int32: "ncclInt32",
    dtypes.uint32: "ncclUint32",
    dtypes.int64: "ncclInt64",
    dtypes.uint64: "ncclUint64",
    dtypes.float16: "ncclFloat16",
    dtypes.bfloat16: "ncclBfloat16",
    dtypes.float32: "ncclFloat32",
    dtypes.float64: "ncclFloat64",
}

#: Complex types and the real type of their components, which NCCL sums componentwise.
_COMPLEX_COMPONENTS = {dtypes.complex64: dtypes.float32, dtypes.complex128: dtypes.float64}


def reduction_op(op: str) -> str:
    """The canonical name (``sum``, ``prod``, ``max``, ``min``, ``avg``, ``land``, ...) of a reduction
    operator given by its MPI name (``MPI_SUM``), :class:`dace.dtypes.ReductionType` name (``Sum``), or
    canonical name."""
    if op in _REDUCTION_OPS:
        return _REDUCTION_OPS[op]
    if op in _REDUCTION_OPS.values():
        return op
    raise ValueError(f"Unknown reduction operator {op}")


def is_average(op: str) -> bool:
    """Whether ``op`` is the average, which MPI expansions compute as a sum divided by the number of processes."""
    return not op.startswith("MPI_") and reduction_op(op) == "avg"


def mpi_op(op: str) -> str:
    """The MPI operator of a reduction operator; ``MPI_SUM`` for the average (see :func:`is_average`). MPI
    operators without a portable equivalent (e.g., ``MPI_MAXLOC``) are passed through."""
    if op.startswith("MPI_") and op not in _REDUCTION_OPS:
        return op
    canonical = reduction_op(op)
    if canonical == "avg":
        return "MPI_SUM"
    return next(name for name, value in _REDUCTION_OPS.items() if value == canonical and name.startswith("MPI_"))


def nccl_op(op: str) -> str:
    """The NCCL operator of a reduction operator."""
    canonical = reduction_op(op)
    if canonical not in _NCCL_OPS:
        raise NotImplementedError(f"NCCL does not support the reduction operator {op}")
    return _NCCL_OPS[canonical]


def nccl_transfer_type(dtype: dtypes.typeclass) -> tuple[str, int]:
    """The NCCL type that moves elements of ``dtype``, and how many of them make up one element. Types
    without an NCCL equivalent move as bytes."""
    if dtype in _NCCL_TYPES:
        return _NCCL_TYPES[dtype], 1
    return "ncclUint8", dtype.bytes


def nccl_reduction_type(dtype: dtypes.typeclass, op: str) -> tuple[str, int]:
    """The NCCL type that reduces elements of ``dtype`` with ``op``, and how many of them make up one
    element. Complex numbers are summed (and averaged) as pairs of reals."""
    if dtype in _NCCL_TYPES:
        return _NCCL_TYPES[dtype], 1
    if dtype in _COMPLEX_COMPONENTS and reduction_op(op) in ("sum", "avg"):
        return _NCCL_TYPES[_COMPLEX_COMPONENTS[dtype]], 2
    raise NotImplementedError(f"NCCL cannot reduce {dtype} with {op}")
