# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import pathlib

import dace
from dace.frontend.python.parser import DaceProgram
from dace.codegen.exceptions import CompilationError
from dace.codegen.compiler import load_precompiled_sdfg
import numpy as np


# Dynamically creates DaCe programs with the same name
def program_generator(size: int, factor: float) -> DaceProgram:

    @dace.program
    def lib_reuse(input: dace.float64[size], output: dace.float64[size]):

        @dace.map(_[0:size])
        def tasklet(i):
            a << input[i]
            b >> output[i]
            b = a * factor

    return lib_reuse


def test_reload():
    array_one = np.random.rand(10).astype(np.float64)
    array_two = np.random.rand(20).astype(np.float64)
    output_one = np.zeros(10, dtype=np.float64)
    output_two = np.zeros(20, dtype=np.float64)

    prog_one = program_generator(10, 2.0)
    prog_two = program_generator(20, 4.0)

    # This should create two libraries for the two SDFGs, as they compile over the same folder
    func1 = prog_one.compile()
    try:
        func2 = prog_two.compile()
    except CompilationError:
        # On some systems (e.g., Windows), the file will be locked, so compilation will fail
        pytest.skip('Compilation failed due to locked file. Skipping test.')

    lib1_path = pathlib.Path(func1.filename)
    lib2_path = pathlib.Path(func2.filename)
    assert lib1_path != lib2_path
    assert lib1_path.parent == lib2_path.parent

    func1(input=array_one, output=output_one)
    func2(input=array_two, output=output_two)

    diff1 = np.linalg.norm(2.0 * array_one - output_one) / 10.0
    diff2 = np.linalg.norm(4.0 * array_two - output_two) / 20.0
    print("Differences:", diff1, diff2)
    assert (diff1 < 1e-5 and diff2 < 1e-5)


def test_load_precompiled():
    for folder_version in ["development", "production"]:
        with dace.config.temporary_config() as conf:
            conf.set('compiler', 'build_folder_version', value=folder_version)
            _load_precompiled_impl(
                test_name="test_load_precompiled",
                folder_version=folder_version,
            )


def _load_precompiled_impl(test_name: str, folder_version: str) -> None:
    prog = program_generator(10, 2.0)
    sdfg = prog.to_sdfg()
    sdfg.name = f'{test_name}_{sdfg.name}_{folder_version}'

    func1 = sdfg.compile()

    if folder_version == "production":
        func2 = load_precompiled_sdfg(sdfg.build_folder, sdfg=sdfg)
    elif folder_version == "development":
        func2 = load_precompiled_sdfg(sdfg.build_folder)

    inp = np.random.rand(10).astype(np.float64)
    output_one = np.zeros(10, dtype=np.float64)
    output_two = np.zeros(10, dtype=np.float64)

    func1(input=inp, output=output_one)
    func2(input=inp, output=output_two)

    lib1_path = pathlib.Path(func1.filename)
    lib2_path = pathlib.Path(func2.filename)
    assert lib1_path != lib2_path
    assert lib1_path.parent == lib2_path.parent

    assert (np.allclose(output_one, output_two))

    del func1
    del func2


if __name__ == '__main__':
    test_reload()
    test_load_precompiled()
