# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import ctypes
import numpy as np


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


if __name__ == '__main__':
    test_uintptr_t()
