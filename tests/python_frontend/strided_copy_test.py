import dace
import pytest


def test_strided_copy():

    @dace.program
    def strided_copy(dst: dace.uint32[20], src: dace.uint32[40]):
        dst[0:20:2] = src[0:40:4]

    base_sdfg = strided_copy.to_sdfg(simplify=False)
    base_sdfg.validate()
    base_sdfg.simplify()
    base_sdfg.validate()


if __name__ == "__main__":
    test_strided_copy()
