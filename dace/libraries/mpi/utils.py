import dace


def MPI_DDT(dtype):
    mpi_dtype_str = "MPI_BYTE"
    if dtype == dace.dtypes.float32:
        mpi_dtype_str = "MPI_FLOAT"
    elif dtype == dace.dtypes.float64:
        mpi_dtype_str = "MPI_DOUBLE"
    elif dtype == dace.dtypes.complex64:
        mpi_dtype_str = "MPI_COMPLEX"
    elif dtype == dace.dtypes.complex128:
        mpi_dtype_str = "MPI_COMPLEX_DOUBLE"
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
    if memlet.other_subset is not None:
        raise ValueError("Other subset must be None, reshape in send not supported")
    # to be contiguous, in every dimension the memlet range must have the same size
    # than the data, except in the last dim, iff all other dims are only one element

    matching = []
    single = []
    for m, d in zip(memlet.subset.size_exact(), data.sizes()):
        if (str(m) == str(d)):
            matching.append(True)
        else:
            matching.append(False)
        if (m == 1):
            single.append(True)
        else:
            single.append(False)

    # if all dims are matching we are contiguous
    if all(x is True for x in matching):
        return True

    # remove last dim, check if all remaining access a single dim
    matching = matching[:-1]
    single = single[:-1]
    if all(x is True for x in single):
        return True

    return False


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
