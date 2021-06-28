# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import os


@dace.program
def customprog(A: dace.float64[20]):
    return A + 1


def test_custom_build_folder():
    with dace.config.set_temporary('default_build_folder', value='.mycache'):
        A = np.random.rand(20)
        assert np.allclose(customprog(A), A + 1)

        # Ensure the folder was created
        assert os.path.isdir('.mycache')

        # Ensure build folder matches
        sdfg = customprog.to_sdfg()
        assert '.mycache' in sdfg.build_folder


if __name__ == '__main__':
    test_custom_build_folder()
