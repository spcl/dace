# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import ctypes
import numpy as np


def test_uintp_size():
    # c_void_p: C type -> void*
    size = ctypes.sizeof(ctypes.c_void_p)
    # numpy.uintp: Unsigned integer large enough to fit pointer, compatible with C uintptr_t
    size_of_np_uintp = np.uintp().itemsize
    # Dace uintptr_t representation
    size_of_dace_uintp = dace.uintp.bytes

    assert size == size_of_np_uintp == size_of_dace_uintp


def test_uintp_use():

    @dace.program
    def tester(arr: dace.float64[20], pointer: dace.uintp[1]):
        with dace.tasklet(dace.Language.CPP):
            a << arr(-1)
            """
            out = decltype(out)(a);
            """
            out >> pointer[0]

    ptr = np.empty([1], dtype=np.uintp)
    arr = np.random.rand(20)
    tester(arr, ptr)
    assert arr.__array_interface__['data'][0] == ptr[0]


if __name__ == '__main__':
    test_uintp_size()
    test_uintp_use()
