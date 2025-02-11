# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

from dace.properties import make_properties


@make_properties
class ArraySubclass(dace.data.Array):
    pass


def test_replacement_subclass():
    @dace.program
    def tester(a: ArraySubclass(dace.float64, [20])):
        return a + 1

    aa = np.random.rand(20)
    assert np.allclose(tester(aa), aa + 1)


if __name__ == '__main__':
    test_replacement_subclass()
