import dace
from dace.frontend.python.parser import DaceProgram
from dace.codegen.compiler import load_precompiled_sdfg


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


def main():
    prog = program_generator(10, 2.0)
    sdfg = prog.to_sdfg()
    sdfg.name = f'test_sdfg'

    csdfg1 = sdfg.compile()
    csdfg2 = load_precompiled_sdfg("/home/quint_essent/git/1_CSCS/__cycle__/dace/tests/.dacecache/test_sdfg")
    sdfg.compile()
    print(f"CSDFG1: {csdfg1.filename}")
    print(f"CSDFG2: {csdfg2.filename}")


if __name__ == "__main__":
    main()
