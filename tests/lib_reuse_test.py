# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.frontend.python.parser import DaceProgram
from dace.codegen.exceptions import CompilationError
from dace.sdfg.utils import load_precompiled_sdfg
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
    print('Reloadable DaCe program test')

    array_one = np.random.rand(10).astype(np.float64)
    array_two = np.random.rand(20).astype(np.float64)
    output_one = np.zeros(10, dtype=np.float64)
    output_two = np.zeros(20, dtype=np.float64)

    prog_one = program_generator(10, 2.0)
    prog_two = program_generator(20, 4.0)

    # This should create two libraries for the two SDFGs, as they compile over
    # the same folder
    func1 = prog_one.compile()
    try:
        func2 = prog_two.compile()
    except CompilationError:
        # On some systems (e.g., Windows), the file will be locked, so
        # compilation will fail
        print('Compilation failed due to locked file. Skipping test.')
        return

    func1(input=array_one, output=output_one)
    func2(input=array_two, output=output_two)

    diff1 = np.linalg.norm(2.0 * array_one - output_one) / 10.0
    diff2 = np.linalg.norm(4.0 * array_two - output_two) / 20.0
    print("Differences:", diff1, diff2)
    assert (diff1 < 1e-5 and diff2 < 1e-5)


def test_load_precompiled():
    prog = program_generator(10, 2.0)
    sdfg = prog.to_sdfg()
    func1 = sdfg.compile()
    func2 = load_precompiled_sdfg(sdfg.build_folder)

    inp = np.random.rand(10).astype(np.float64)
    output_one = np.zeros(10, dtype=np.float64)
    output_two = np.zeros(10, dtype=np.float64)

    func1(input=inp, output=output_one)
    func2(input=inp, output=output_two)

    assert (np.allclose(output_one, output_two))

    del func1
    del func2


if __name__ == '__main__':
    test_reload()
    test_load_precompiled()
