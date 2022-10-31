# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import re
import os
import shutil


def _program_name(function) -> str:
    """ Replicates the behavior of DaCe in determining the SDFG label."""
    result = ''
    if function.__module__ is not None and function.__module__ != '__main__':
        result += function.__module__.replace('.', '_') + '_'
    return result + function.__name__


def test_recreate_sdfg():
    # Get the program name, regardless of running directly or through pytest
    def very_unique_program_321():
        pass

    program_name = _program_name(very_unique_program_321)

    build_folder = os.path.join(dace.config.Config.get('default_build_folder'), program_name)

    # Ensure that the build folder is empty
    if os.path.exists(build_folder):
        shutil.rmtree(build_folder)

    @dace.program(recreate_sdfg=False)
    def very_unique_program_321(A: dace.float64[10]):
        return A + 1

    a = np.random.rand(10)
    assert np.allclose(a + 1, very_unique_program_321(a))

    assert os.path.exists(os.path.join(build_folder, 'program.sdfg'))
    sdfg = dace.SDFG.from_file(os.path.join(build_folder, 'program.sdfg'))

    # Replace the SDFG with the one we just created
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.Tasklet):
            node.code.as_string = re.sub(r'\b1\b', '2', node.code.as_string)

    sdfg.save(os.path.join(build_folder, 'program.sdfg'))

    # Now run the same program again, but this time with the SDFG already created
    @dace.program(recreate_sdfg=False)
    def very_unique_program_321(A: dace.float64[10]):
        return A + 1

    a = np.random.rand(10)
    assert np.allclose(a + 2, very_unique_program_321(a))


def test_regenerate_code():
    # Get the program name, regardless of running directly or through pytest
    def very_unique_program_432():
        pass

    program_name = _program_name(very_unique_program_432)

    build_folder = os.path.join(dace.config.Config.get('default_build_folder'), program_name)

    # Ensure that the build folder is empty
    if os.path.exists(build_folder):
        shutil.rmtree(build_folder)

    @dace.program(regenerate_code=False)
    def very_unique_program_432(A: dace.float64[10]):
        return A + 3

    a = np.random.rand(10)
    assert np.allclose(a + 3, very_unique_program_432(a))

    # Source code
    source_filename = os.path.join(build_folder, 'src', 'cpu', program_name + '.cpp')
    assert os.path.exists(source_filename)

    # Rewrite source code
    with open(source_filename, 'r') as f:
        source = f.read()
        source = re.sub(r'\b3\b', '4', source)

    with open(source_filename, 'w') as f:
        f.write(source)

    # Now run the same program again, but this time with the modified code (ensures it is recompiled)
    @dace.program(regenerate_code=False)
    def very_unique_program_432(A: dace.float64[10]):
        return A + 3

    a = np.random.rand(10)
    assert np.allclose(a + 4, very_unique_program_432(a))


if __name__ == '__main__':
    test_recreate_sdfg()
    test_regenerate_code()
