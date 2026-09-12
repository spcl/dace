# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The loader stub loads without an OpenMP runtime of its own, and still keeps the one a program links alive.

A stub compiled with ``-fopenmp`` but linked without the runtime referenced ``omp_get_max_threads``
strongly, so ``dlopen`` failed with "undefined symbol: omp_get_max_threads" and no program could load.
The stub now finds the runtime through the program. It must still keep that runtime mapped past the
program's ``dlclose``, which a runtime on the stub's own link line used to do: unmapped under its worker
threads, it crashes the process.
"""
import ctypes
import pathlib
import subprocess

import dace

STUB_SOURCE = pathlib.Path(dace.__file__).parent / "codegen" / "tools" / "dacestub.cpp"

#: A stand-in program that exports its own ``omp_get_max_threads`` and reports each call into ``flag``.
FAKE_PROGRAM = """
static int *flag = nullptr;
extern "C" void watch(int *target) { flag = target; }
extern "C" int omp_get_max_threads() { if (flag) *flag += 1; return 1; }
"""


def shared_library(directory: pathlib.Path, name: str, source: pathlib.Path, compile_flags: list[str],
                   link_flags: list[str]) -> pathlib.Path:
    """Compile ``source`` and link it into ``lib<name>.so`` in two steps, so the link line is exactly ``link_flags``."""
    obj = directory / f"{name}.o"
    library = directory / f"lib{name}.so"
    subprocess.run(["c++", "-std=c++17", "-fPIC", *compile_flags, "-c", str(source), "-o", str(obj)], check=True)
    subprocess.run(["c++", "-shared", "-o", str(library), str(obj), *link_flags], check=True)
    return library


def fake_program(directory: pathlib.Path) -> pathlib.Path:
    source = directory / "fake_program.cpp"
    source.write_text(FAKE_PROGRAM)
    return shared_library(directory, "fake_program", source, [], [])


def stub_without_runtime(directory: pathlib.Path) -> ctypes.CDLL:
    """The stub as a build that compiles it with ``-fopenmp`` but links no OpenMP runtime produces it."""
    stub = ctypes.CDLL(str(shared_library(directory, "stub", STUB_SOURCE, ["-fopenmp"], ["-pthread", "-ldl"])))
    stub.load_library.restype = ctypes.c_void_p
    stub.load_library.argtypes = [ctypes.c_char_p]
    stub.get_symbol.restype = ctypes.c_void_p
    stub.get_symbol.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    stub.unload_library.argtypes = [ctypes.c_void_p]
    stub.is_library_loaded.argtypes = [ctypes.c_char_p]
    return stub


def test_a_stub_built_with_openmp_but_linked_without_its_runtime_loads(tmp_path: pathlib.Path):
    sut = stub_without_runtime(tmp_path)

    assert sut["unload_library"]


def test_unloading_a_program_calls_the_openmp_runtime_that_program_links(tmp_path: pathlib.Path):
    program = fake_program(tmp_path)
    sut = stub_without_runtime(tmp_path)
    handle = sut.load_library(str(program).encode())
    watch = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int))(sut.get_symbol(handle, b"watch"))
    calls = ctypes.c_int(0)
    watch(ctypes.byref(calls))

    sut.unload_library(handle)

    assert calls.value == 1


def test_the_openmp_runtime_a_program_links_stays_mapped_after_the_program_unloads(tmp_path: pathlib.Path):
    program = fake_program(tmp_path)
    sut = stub_without_runtime(tmp_path)
    handle = sut.load_library(str(program).encode())

    sut.unload_library(handle)

    assert sut.is_library_loaded(str(program).encode()) == 1
