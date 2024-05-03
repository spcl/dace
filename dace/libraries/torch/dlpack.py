"""
Interface for integrating with DLPack.

Some of the following code is derived from the following resources:
https://github.com/dmlc/dlpack/blob/main/apps/from_numpy/main.py
https://github.com/vadimkantorov/pydlpack/blob/master/dlpack.py
"""

import ctypes

import dace
import torch
import torch.utils.dlpack
from dace import data, dtypes


class DLDeviceType(ctypes.c_int):
    kDLCPU = 1
    kDLGPU = 2
    kDLCPUPinned = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLExtDev = 12


class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4


class DLDataType(ctypes.Structure):
    _fields_ = [('type_code', DLDataTypeCode), ('bits', ctypes.c_uint8),
                ('lanes', ctypes.c_uint16)]


dace_to_dldtype_dict = {
    dace.float32: DLDataType(DLDataTypeCode.kDLFloat, 32, 1),
    dace.float64: DLDataType(DLDataTypeCode.kDLFloat, 64, 1),
    dace.uint8: DLDataType(DLDataTypeCode.kDLUInt, 8, 1),
    dace.uint16: DLDataType(DLDataTypeCode.kDLUInt, 16, 1),
    dace.uint32: DLDataType(DLDataTypeCode.kDLUInt, 32, 1),
    dace.uint64: DLDataType(DLDataTypeCode.kDLUInt, 64, 1),
    dace.int8: DLDataType(DLDataTypeCode.kDLInt, 8, 1),
    dace.int16: DLDataType(DLDataTypeCode.kDLInt, 16, 1),
    dace.int32: DLDataType(DLDataTypeCode.kDLInt, 32, 1),
    dace.int64: DLDataType(DLDataTypeCode.kDLInt, 64, 1),
}


class DLContext(ctypes.Structure):
    _fields_ = [('device_type', DLDeviceType), ('device_id', ctypes.c_int)]


class DLTensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('ctx', DLContext),
                ('ndim', ctypes.c_int), ('dtype', DLDataType),
                ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('byte_offset', ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    pass


DLManagedTensorHandle = ctypes.POINTER(DLManagedTensor)

DeleterFunc = ctypes.CFUNCTYPE(None, DLManagedTensorHandle)

DLManagedTensor._fields_ = [("dl_tensor", DLTensor),
                            ("manager_ctx", ctypes.c_void_p),
                            ("deleter", DeleterFunc)]


def make_manager_ctx(obj):
    pyobj = ctypes.py_object(obj)
    void_p = ctypes.c_void_p.from_buffer(pyobj)
    ctypes.pythonapi.Py_IncRef(pyobj)
    return void_p


@DeleterFunc
def dl_managed_tensor_deleter(dl_managed_tensor_handle):
    # do nothing: the data is freed in the state struct
    pass


class PyCapsule:
    New = ctypes.pythonapi.PyCapsule_New
    New.restype = ctypes.py_object
    New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)

    SetContext = ctypes.pythonapi.PyCapsule_SetContext
    SetContext.restype = ctypes.c_int
    SetContext.argtypes = (ctypes.py_object, ctypes.c_void_p)

    GetContext = ctypes.pythonapi.PyCapsule_GetContext
    GetContext.restype = ctypes.c_void_p
    GetContext.argtypes = (ctypes.py_object, )

    GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    GetPointer.restype = ctypes.c_void_p
    GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)

    Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)

    SetDestructor = ctypes.pythonapi.PyCapsule_SetDestructor
    SetDestructor.argtypes = (ctypes.py_object, Destructor)
    SetDestructor.restype = ctypes.c_int


def array_to_torch_tensor(ptr: ctypes.c_void_p,
                          desc: data.Array) -> torch.Tensor:
    """ Convert a dace array descriptor to a torch tensor that points to the same data.

        :param ptr: the pointer the the memory of the array.
        :param desc: the dace array descriptor.
        :return: the tensor.
    """

    if desc.storage is dtypes.StorageType.GPU_Global:
        device_type = 2
    elif desc.storage in [
            dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default
    ]:
        device_type = 1
    else:
        raise ValueError(f"Unsupported storage type {desc.storage}")

    context = DLContext(device_type=device_type, device_id=0)

    if desc.dtype not in dace_to_dldtype_dict:
        raise ValueError(f"Unsupported dtype {desc.dtype}")
    dtype = dace_to_dldtype_dict[desc.dtype]

    shape = (ctypes.c_int64 * len(desc.shape))()
    for i, s in enumerate(desc.shape):
        shape[i] = s

    strides = (ctypes.c_int64 * len(desc.shape))()
    for i, s in enumerate(desc.strides):
        strides[i] = s

    dltensor = DLTensor(data=ptr,
                        ctx=context,
                        ndim=len(desc.shape),
                        dtype=dtype,
                        shape=shape,
                        strides=strides,
                        byte_offset=0)

    c_obj = DLManagedTensor()
    c_obj.dl_tensor = dltensor
    c_obj.manager_ctx = ctypes.c_void_p(0)
    c_obj.deleter = dl_managed_tensor_deleter

    # the capsule must be used in the same stackframe, otherwise it will be deallocated an the capsule will
    # point to invalid data.
    capsule = PyCapsule.New(ctypes.byref(c_obj), b"dltensor", None)
    tensor: torch.Tensor = torch.utils.dlpack.from_dlpack(capsule)

    # store the dltensor as an attribute of the tensor so that the tensor takes ownership
    tensor._dace_dlpack = c_obj
    return tensor
