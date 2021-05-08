import dace.dtypes as dtypes

def LAPACK_DTYPE_CHR(dtype):
    lapack_dtype = "X"
    if dtype == dtypes.float32:
        lapack_dtype = "s"
    elif dtype == dtypes.float64:
        lapack_dtype = "d"
    elif dtype == dtypes.complex64:
        lapack_dtype = "c"
    elif dtype == dtypes.complex128:
        lapack_dtype = "z"
    else:
        print("The datatype " + str(dtype) + " is not supported!")
        raise (NotImplementedError)
    return lapack_dtype
