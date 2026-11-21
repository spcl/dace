# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.cpu.heap_allocator``: CPU heap arrays from mimalloc, or from aligned ``operator new``."""

import ctypes.util
import os
import subprocess
import sys

import numpy as np
import pytest

import dace
from dace.libraries.standard.environments.mimalloc import CXX_RUNTIME, MiMalloc

N = dace.symbol("N")


def heap_transient_sdfg(name: str) -> dace.SDFG:
    """``B = (A + 1) * 2`` through a heap transient of ``N`` elements."""

    @dace.program
    def through_heap(A: dace.float64[N], B: dace.float64[N]):
        tmp = np.empty([N], dtype=np.float64)
        tmp[:] = A + 1.0
        B[:] = tmp * 2.0

    sdfg = through_heap.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def linked_libraries(compiled) -> str:
    """The libraries the program loads, its own dependencies only: a launcher's ``LD_PRELOAD`` (the judge containers
    preload mimalloc) would list in every program."""
    environment = {name: value for name, value in os.environ.items() if name != "LD_PRELOAD"}
    return subprocess.run(
        ["ldd", compiled.filename], capture_output=True, text=True, check=True, env=environment
    ).stdout


@pytest.mark.skipif(MiMalloc.library_path() is None, reason="mimalloc is not installed")
def test_heap_arrays_come_from_mimalloc():
    with dace.config.set_temporary("compiler", "cpu", "heap_allocator", value="mimalloc"):
        compiled = heap_transient_sdfg("heap_mimalloc").compile()
    assert "mimalloc" in linked_libraries(compiled)
    A = np.random.default_rng(1).random(1000)
    B = np.zeros(1000)
    compiled(A=A, B=B, N=1000)
    np.testing.assert_allclose(B, (A + 1.0) * 2.0)


def needed_libraries(compiled) -> list[str]:
    """The shared libraries ``compiled`` lists (``DT_NEEDED``), in the order the loader searches them."""
    dynamic = subprocess.run(["readelf", "-d", compiled.filename], capture_output=True, text=True, check=True).stdout
    return [line.split("[", 1)[1].split("]", 1)[0] for line in dynamic.splitlines() if "(NEEDED)" in line]


@pytest.mark.skipif(MiMalloc.library_path() is None, reason="mimalloc is not installed")
def test_a_mimalloc_program_binds_operator_delete_to_libstdcxx_not_mimalloc():
    """libmimalloc exports ``operator new`` / ``delete`` too: listed before libstdc++, a program's ``delete`` of
    memory libstdc++ allocated went to ``mi_free`` and crashed (a later program's instrumentation report)."""
    with dace.config.set_temporary("compiler", "cpu", "heap_allocator", value="mimalloc"):
        compiled = heap_transient_sdfg("heap_mimalloc_order").compile()
    needed = needed_libraries(compiled)
    cxx = next(i for i, name in enumerate(needed) if name.startswith("libstdc++"))
    mimalloc = next(i for i, name in enumerate(needed) if name.startswith("libmimalloc"))
    assert cxx < mimalloc, needed


def test_the_system_allocator_links_no_mimalloc():
    with dace.config.set_temporary("compiler", "cpu", "heap_allocator", value="system"):
        compiled = heap_transient_sdfg("heap_system").compile()
    assert "mimalloc" not in linked_libraries(compiled)
    A = np.random.default_rng(2).random(1000)
    B = np.zeros(1000)
    compiled(A=A, B=B, N=1000)
    np.testing.assert_allclose(B, (A + 1.0) * 2.0)


def test_a_runtime_only_mimalloc_is_linked_by_its_file_name(monkeypatch):
    """A runtime-only install has ``libmimalloc.so.3`` but no ``libmimalloc.so``: ``-lmimalloc`` failed to link
    every program (``cannot find -lmimalloc``) once ``auto`` had found the library."""
    monkeypatch.delenv("MIMALLOC_ROOT", raising=False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libmimalloc.so.3")

    runtime = [CXX_RUNTIME] if sys.platform.startswith("linux") else []
    assert MiMalloc.cmake_libraries() == [*runtime, "-l:libmimalloc.so.3"]


if __name__ == "__main__":
    if MiMalloc.library_path() is not None:
        test_heap_arrays_come_from_mimalloc()
        test_a_mimalloc_program_binds_operator_delete_to_libstdcxx_not_mimalloc()
    test_the_system_allocator_links_no_mimalloc()
    with pytest.MonkeyPatch.context() as patch:
        test_a_runtime_only_mimalloc_is_linked_by_its_file_name(patch)
