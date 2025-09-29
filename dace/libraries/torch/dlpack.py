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
    """DLPack device type enumeration."""
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
    """DLPack data type code enumeration."""
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4


class DLDataType(ctypes.Structure):
    """DLPack data type structure."""
    _fields_ = [('type_code', DLDataTypeCode), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


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
    """DLPack context structure for device information."""
    _fields_ = [('device_type', DLDeviceType), ('device_id', ctypes.c_int)]


class DLTensor(ctypes.Structure):
    """DLPack tensor structure."""
    _fields_ = [('data', ctypes.c_void_p), ('ctx', DLContext), ('ndim', ctypes.c_int), ('dtype', DLDataType),
                ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('byte_offset', ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    """DLPack managed tensor structure."""
    pass


DLManagedTensorHandle = ctypes.POINTER(DLManagedTensor)

DeleterFunc = ctypes.CFUNCTYPE(None, DLManagedTensorHandle)

DLManagedTensor._fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", DeleterFunc)]


def make_manager_ctx(obj) -> ctypes.c_void_p:
    """
    Create a manager context from a Python object.

    This function wraps a Python object in a ctypes void pointer and increments
    its reference count to prevent garbage collection while in use by DLPack.

    Args:
        obj: The Python object to create a context for.

    Returns:
        A ctypes void pointer to the object.
    """
    pyobj = ctypes.py_object(obj)
    void_p = ctypes.c_void_p.from_buffer(pyobj)
    ctypes.pythonapi.Py_IncRef(pyobj)
    return void_p


@DeleterFunc
def dl_managed_tensor_deleter(_dl_managed_tensor_handle) -> None:
    """
    Deleter function for DLPack managed tensors.

    This is a no-op deleter because the underlying data is managed by DaCe
    and will be freed when the SDFG state struct is deallocated.

    Args:
        _dl_managed_tensor_handle: Handle to the managed tensor (unused).
    """
    # Do nothing: the data is freed in the state struct
    pass


class PyCapsule:
    """Python capsule interface for DLPack integration."""
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


def array_to_torch_tensor(ptr: ctypes.c_void_p, desc: data.Array) -> torch.Tensor:
    """
    Convert a DaCe array descriptor to a PyTorch tensor that points to the same data.

    This function performs zero-copy conversion using the DLPack protocol,
    allowing PyTorch to access DaCe arrays without data duplication.

    Args:
        ptr: The pointer to the memory of the array.
        desc: The DaCe array descriptor containing shape, strides, and dtype information.

    Returns:
        A PyTorch tensor that shares memory with the DaCe array.

    Raises:
        ValueError: If the storage type or dtype is unsupported.
    """

    if desc.storage is dtypes.StorageType.GPU_Global:
        device_type = DLDeviceType.kDLGPU
    elif desc.storage in [dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default]:
        device_type = DLDeviceType.kDLCPU
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

    # The capsule must be used in the same stack frame, otherwise it will be deallocated and the capsule will
    # point to invalid data.
    capsule = PyCapsule.New(ctypes.byref(c_obj), b"dltensor", None)
    tensor: torch.Tensor = torch.utils.dlpack.from_dlpack(capsule)

    # Store the dltensor as an attribute of the tensor so that the tensor takes ownership
    tensor._dace_dlpack = c_obj
    return tensor
