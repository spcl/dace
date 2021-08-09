import dace
import aenum
from dace import dtypes
from dace.registry import extensible_enum, undefined_safe_enum


@undefined_safe_enum
@extensible_enum
class NcclReductionType(aenum.AutoNumberEnum):
    """ Reduction types supported by NCCL. """
    ncclSum = ()  #: Sum
    ncclProd = ()  #: Product
    ncclMin = ()  #: Minimum value
    ncclMax = ()  #: Maximum value


NCCL_SUPPORTED_OPERATIONS = {
    None: NcclReductionType.ncclSum,
    dtypes.ReductionType.Sum: NcclReductionType.ncclSum,
    dtypes.ReductionType.Product: NcclReductionType.ncclProd,
    dtypes.ReductionType.Min: NcclReductionType.ncclMin,
    dtypes.ReductionType.Max: NcclReductionType.ncclMax
}


def Nccl_dtypes(dtype):
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
