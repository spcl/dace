# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests f-strings in dace programs. """
import dace


def test_static_fstring():
    N = 5

    @dace.program
    def fprog_static():
        with dace.tasklet:
            printf(f'hi {N}\n')

    fprog_static()


def test_partial_fstring():
    N = 5

    @dace.program
    def fprog_partial():
        with dace.tasklet:
            i = 2
            printf(f'hi {N} {i}\n')

    fprog_partial()


if __name__ == '__main__':
    test_static_fstring()
    test_partial_fstring()
