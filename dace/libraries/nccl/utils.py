import dace


def NCCL_DDT(dtype):
    nccl_dtype_str = ""
    if dtype == dace.dtypes.float16:
        nccl_dtype_str = "ncclFloat16"
    elif dtype == dace.dtypes.float32:
        nccl_dtype_str = "ncclFloat32"
    elif dtype == dace.dtypes.float64:
        nccl_dtype_str = "ncclFloat64"
    elif dtype == dace.dtypes.int8:
        nccl_dtype_str = "ncclInt8"
    elif dtype == dace.dtypes.int32:
        nccl_dtype_str = "ncclInt32"
    elif dtype == dace.dtypes.int32:
        nccl_dtype_str = "ncclInt32"
    elif dtype == dace.dtypes.int64:
        nccl_dtype_str = "ncclInt64"
    elif dtype == dace.dtypes.uint8:
        nccl_dtype_str = "ncclUint8"
    elif dtype == dace.dtypes.uint32:
        nccl_dtype_str = "ncclUint32"
    elif dtype == dace.dtypes.uint64:
        nccl_dtype_str = "ncclUint64"
    else:
        raise ValueError("DDT of " + str(dtype) + " not supported yet.")
    return nccl_dtype_str


def is_access_contiguous(memlet, data):
    if memlet.other_subset is not None:
        raise ValueError(
            "Other subset must be None, reshape in send not supported")
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
    ddt["oldtype"] = str(NCCL_DDT(data))
    ddt["count"] = "(" + str(
        memlet.subset.num_elements_exact()) + ")" + "/" + str(ddt['blocklen'])
    ddt["stride"] = str(data.strides[0])
    return ddt

# def data_accessible_by_gpu(lib_node: dace.nodes.LibraryNode, state: dace.SDFGState):
