# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A build folder reused by a second program must not run the first one's code.

The default cache policy names the build folder after the program, so two ``@dace.program``
functions that share a name -- in one module, or in two modules of the same name -- share a folder.
What then keeps them apart is the build system's timestamp comparison, and that is not sound on a
filesystem stamping mtimes at one-second granularity (Lustre and NFS do): a program regenerated
within the same second as the previous link reads as up to date, the build is skipped, and the
PREVIOUS program's library is what gets loaded. The symptom is a plausible wrong answer rather than
a build error, and it comes and goes with how fast the two compiles happen, so it reads as test
flakiness.

Separately, ``is_library_loaded`` asked its question with ``dlopen(RTLD_NOLOAD)``, which takes a
reference on a library that IS loaded, and never gave it back -- one probe pinned the library for
the life of the process.
"""
import ctypes
import pathlib

import numpy as np

import dace
from dace.codegen.compiled_sdfg import ReloadableDLL


@dace.program
def scale(A: dace.float64[8], B: dace.float64[8]):
    for i in dace.map[0:8]:
        B[i] = A[i] * 2.0


def test_probing_whether_a_library_is_loaded_does_not_pin_it():
    """One load and one unload must balance, however many times the library was asked about."""
    csdfg = scale.to_sdfg().compile()
    library = pathlib.Path(csdfg.filename).resolve()
    del csdfg

    dll = ReloadableDLL(str(library))
    dll.load()
    name = ctypes.c_char_p(dll._library_filename.encode())
    assert dll._stub.is_library_loaded(name) == 1, 'the library did not load, so nothing is tested'

    for _ in range(5):
        dll._stub.is_library_loaded(name)
    stub = dll._stub  # unload() drops the handle, and the question outlives it
    dll.unload()

    assert stub.is_library_loaded(name) == 0, \
        'the library is still mapped after its one load was undone: each probe took a reference'


def test_a_folder_reused_by_another_program_runs_the_new_code():
    """Two programs, one name, one build folder -- and the second one has to be the one that runs."""

    @dace.program
    def first(A: dace.float64[8], B: dace.float64[8]):
        for i in dace.map[0:8]:
            B[i] = A[i] + 1.0

    @dace.program
    def second(A: dace.float64[8], B: dace.float64[8]):
        for i in dace.map[0:8]:
            B[i] = A[i] + 100.0

    shared_name = 'same_name_two_programs'
    a = first.to_sdfg()
    a.name = shared_name
    b = second.to_sdfg()
    b.name = shared_name
    assert a.build_folder == b.build_folder, 'the premise of this test is that they share a folder'

    A = np.arange(8, dtype=np.float64)
    out_a = np.zeros(8)
    a(A=A, B=out_a)
    assert np.allclose(out_a, A + 1.0), 'the first program did not run, so nothing is tested'

    out_b = np.zeros(8)
    b(A=A, B=out_b)
    assert np.allclose(out_b, A + 100.0), \
        f"the second program ran the first one's code: got {out_b}, wanted {A + 100.0}"


if __name__ == '__main__':
    test_probing_whether_a_library_is_loaded_does_not_pin_it()
    test_a_folder_reused_by_another_program_runs_the_new_code()
