# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest
import numpy as np

N = dace.symbol('N')


@dace.program
def imgcpy(img1: dace.float64[N, N], img2: dace.float64[N, N], coefficient: dace.float64):
    img1[:, :] = img2[:, :] * coefficient


def test_extra_args():
    with pytest.raises(TypeError):
        imgcpy([[1, 2], [3, 4]], [[4, 3], [2, 1]], 0.0, 1.0)


def test_missing_arguments_regression():

    def nester(a, b, T):
        for i, j in dace.map[0:20, 0:20]:
            start = 0
            end = min(T, 6)

            elem: dace.float64 = 0
            for ii in range(start, end):
                if ii % 2 == 0:
                    elem += b[ii]

            a[j, i] = elem

    @dace.program
    def tester(x: dace.float64[20, 20]):
        gdx = np.ones((10, ), dace.float64)
        for T in range(2):
            nester(x, gdx, T)

    tester.to_sdfg().compile()


if __name__ == '__main__':
    test_extra_args()
    test_missing_arguments_regression()
