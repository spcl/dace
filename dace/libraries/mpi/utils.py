import dace


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
    """Whether ``memlet`` accesses a single contiguous run of ``data``'s memory. Thin MPI-side adapter
    over :meth:`dace.subsets.Subset.is_contiguous_subset` (the shared, stride-aware contiguity check --
    correct for C, Fortran, and packed permutations), plus the MPI restriction that a reshaping send
    (``other_subset``) is unsupported."""
    if memlet.other_subset is not None:
        raise ValueError("Other subset must be None, reshape in send not supported")
    return memlet.subset.is_contiguous_subset(data)


def create_vector_ddt(memlet, data):
    if is_access_contiguous(memlet, data):
        return None
    if len(data.shape) != 2:
        raise ValueError("Dimensionality of access not supported atm.")
    ddt = dict()
    ddt["blocklen"] = str(memlet.subset.size_exact()[-1])
    ddt["oldtype"] = str(MPI_DDT(data.dtype))
    ddt["count"] = "(" + str(memlet.subset.num_elements_exact()) + ")" + "/" + str(ddt['blocklen'])
    ddt["stride"] = str(data.strides[0])
    return ddt
