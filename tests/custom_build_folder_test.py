# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import os
import tempfile


@dace.program
def customprog(A: dace.float64[20]):
    return A + 1


def test_custom_build_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        with dace.config.set_temporary('default_build_folder', value=tmpdir):
            # Ensure build folder matches
            sdfg = customprog.to_sdfg()
            assert tmpdir in sdfg.build_folder
            csdfg = sdfg.compile()

            # Ensure files were generated in the right folder
            assert os.path.isfile(os.path.join(sdfg.build_folder, 'program.sdfg'))

            # Ensure file is closed so it can be deleted
            del csdfg


if __name__ == '__main__':
    test_custom_build_folder()
