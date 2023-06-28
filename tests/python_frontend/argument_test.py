# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest

N = dace.symbol('N')


@dace.program
def imgcpy(img1: dace.float64[N, N], img2: dace.float64[N, N], coefficient: dace.float64):
    img1[:, :] = img2[:, :] * coefficient


def test_extra_args():
    with pytest.raises(TypeError):
        imgcpy([[1, 2], [3, 4]], [[4, 3], [2, 1]], 0.0, 1.0)


if __name__ == '__main__':
    test_extra_args()
