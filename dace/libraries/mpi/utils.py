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
    elif dtype == dace.dtypes.int32:
        mpi_dtype_str = "MPI_INT"
    else:
        raise ValueError("DDT of "+str(dtype)+" not supported yet.")
    return mpi_dtype_str
