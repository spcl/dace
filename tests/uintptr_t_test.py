# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import ctypes
import platform
import numpy as np


@dace.program
def test_dace(pointer: dace.uintp, value: dace.data.Array(dtype=dace.int64, shape=[1])):
    with dace.tasklet(dace.Language.CPP):
        p << pointer
        val >> value[0]
        """
        val = *reinterpret_cast<int*>(p);
        """


def test_uintptr_t():
    # According to https://en.cppreference.com/w/cpp/types/integer
    # uintptr_t: unsigned integer type capable of holding a pointer to void

    # c_void_p: C type -> void*
    size = ctypes.sizeof(ctypes.c_void_p)
    # numpy.uintp: Unsigned integer large enough to fit pointer, compatible with C uintptr_t
    size_of_np_uintp = np.uintp().itemsize
    # Dace uintptr_t representation
    size_of_dace_uintp = dace.uintp.bytes

    assert size == size_of_np_uintp == size_of_dace_uintp

    ###############################################################################

    cpp_code = """
    extern "C" {
        void* get_pointer() {
            int* ptr = new int(42);
            return static_cast<void*>(ptr);
        }
    }
    """

    with open("/tmp/temp.cpp", "w") as file:
        file.write(cpp_code)

    import os
    os.system("g++ -shared -o /tmp/test_uintptr_t.out -fPIC /tmp/temp.cpp")

    lib = ctypes.CDLL("/tmp/test_uintptr_t.out")
    lib.get_pointer.restype = ctypes.c_void_p
    pointer = lib.get_pointer()

    value_ctypes = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value
    value_dace = np.empty(shape=1, dtype=np.intc)
    test_dace(pointer, value_dace)

    assert value_ctypes == value_dace[0], f"Expected {value_ctypes}, got {value_dace[0]}"
    
    os.remove("/tmp/temp.cpp")
    os.remove("/tmp/test_uintptr_t.out")


if __name__ == '__main__':
    test_uintptr_t()
