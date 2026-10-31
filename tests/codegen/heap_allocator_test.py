# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.cpu.heap_allocator``: CPU heap arrays from mimalloc, or from aligned ``operator new``."""

import ctypes.util
import subprocess

import numpy as np
import pytest

import dace
from dace.libraries.standard.environments.mimalloc import MiMalloc

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
    return subprocess.run(["ldd", compiled.filename], capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(MiMalloc.library_path() is None, reason="mimalloc is not installed")
def test_heap_arrays_come_from_mimalloc():
    with dace.config.set_temporary("compiler", "cpu", "heap_allocator", value="mimalloc"):
        compiled = heap_transient_sdfg("heap_mimalloc").compile()
    assert "mimalloc" in linked_libraries(compiled)
    A = np.random.default_rng(1).random(1000)
    B = np.zeros(1000)
    compiled(A=A, B=B, N=1000)
    np.testing.assert_allclose(B, (A + 1.0) * 2.0)


def test_the_system_allocator_links_no_mimalloc():
    with dace.config.set_temporary("compiler", "cpu", "heap_allocator", value="system"):
        compiled = heap_transient_sdfg("heap_system").compile()
    assert "mimalloc" not in linked_libraries(compiled)
    A = np.random.default_rng(2).random(1000)
    B = np.zeros(1000)
    compiled(A=A, B=B, N=1000)
    np.testing.assert_allclose(B, (A + 1.0) * 2.0)


if __name__ == "__main__":
    if MiMalloc.library_path() is not None:
        test_heap_arrays_come_from_mimalloc()
    test_the_system_allocator_links_no_mimalloc()


def test_a_runtime_only_mimalloc_is_linked_by_its_file_name(monkeypatch):
    """A runtime-only install has ``libmimalloc.so.3`` but no ``libmimalloc.so``: ``-lmimalloc`` failed to link
    every program (``cannot find -lmimalloc``) once ``auto`` had found the library."""
    monkeypatch.delenv("MIMALLOC_ROOT", raising=False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libmimalloc.so.3")

    assert MiMalloc.cmake_libraries() == ["-l:libmimalloc.so.3"]
