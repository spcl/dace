# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness of the memory microbenchmarks that parametrize the LogP memory model.

These are CORRECTNESS tests, not measurements. A timing number cannot be a CI assertion on a shared
box -- pytest workers contend with the very thing being measured -- so nothing here asserts a
latency or a bandwidth. What IS asserted is everything that decides whether a later measurement
means anything: the chain is one Hamiltonian cycle, the chase compiles to a bare dependent load, the
triad vectorizes and is not turned into memcpy, and the triad computes the right answer."""
import ctypes
import os
import subprocess

import numpy
import pytest

from dace.transformation.layout.cost_model import membench

pytestmark = pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="Linux-only benchmark")


@pytest.fixture(scope="module")
def lib():
    return membench.build()


@pytest.mark.parametrize("stride", [64, 256, 4096])
def test_chain_is_one_hamiltonian_cycle_per_stride(lib, stride):
    """The decisive gate. A chain that fragments into sub-cycles traps the chase in a footprint that
    fits in cache, so a 512 MiB arena silently reports an L2 latency labelled DRAM -- and looks
    perfectly healthy doing it. chase_verify walks the cycle and requires it to close after exactly
    nr_elts hops, never earlier."""
    handle = lib.chase_setup(8 << 20, stride, 1, 1)
    assert handle, "chase_setup failed"
    try:
        assert lib.chase_verify(handle) == 0
    finally:
        lib.chase_teardown(handle)


@pytest.mark.parametrize("nchain", [1, 2, 8, 32])
def test_chains_are_disjoint_cycles_for_every_concurrency(lib, nchain):
    """The MLP sweep partitions the arena into nchain independent chains; each must still be a single
    cycle over its own elements, or the concurrency curve measures nothing."""
    handle = lib.chase_setup(8 << 20, 256, nchain, 1)
    assert handle, "chase_setup failed"
    try:
        assert lib.chase_verify(handle) == 0
    finally:
        lib.chase_teardown(handle)


def test_chase_setup_rejects_unusable_parameters(lib):
    """A stride below the 64B line would put two elements on one line (the second access would be an
    L1 hit, not a miss), and nchain above the compiled bound would overrun the starts array."""
    assert not lib.chase_setup(8 << 20, 32, 1, 1)  # stride < cache line
    assert not lib.chase_setup(8 << 20, 256, 0, 1)  # no chains
    assert not lib.chase_setup(8 << 20, 256, 64, 1)  # nchain > CHASE_MAX_CHAINS
    assert not lib.chase_setup(1 << 12, 256, 32, 1)  # arena too small to split


def test_chase_hops_and_faults_are_reported(lib):
    """The window reports the hops actually performed and any major fault taken -- a window that
    faulted is rejected rather than averaged in."""
    handle = lib.chase_setup(8 << 20, 256, 1, 1)
    assert handle
    try:
        ns = ctypes.c_uint64(0)
        majflt = ctypes.c_uint64(0)
        done = lib.chase_run_timed(handle, 10000, ctypes.byref(ns), ctypes.byref(majflt))
        assert done == 10000
        assert ns.value > 0  # the window took real time
        assert majflt.value == 0  # a pre-faulted arena must not fault in the timed window
    finally:
        lib.chase_teardown(handle)


def _disassemble(symbol=None):
    out = os.path.join(os.path.dirname(membench.SOURCE), "build", "libmembench.so")
    command = ["objdump", "-d", out]
    if symbol:
        command.append(f"--disassemble={symbol}")
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def test_chase_body_is_a_bare_dependent_load(lib):
    """`mov (%rax),%rax` and nothing else. This is the whole benchmark: the address of the next load
    IS the value of the previous one, which is what pins MLP to 1. If the pointer goes through a
    stack slot instead, every hop pays a store-forward round trip on the critical path; a scaled-index
    addressing mode costs 5 cycles instead of 4. Either silently rewrites the number."""
    assert "mov    (%rax),%rax" in _disassemble("chase_run_timed")


def test_triad_vectorizes_and_is_not_rewritten_to_memcpy(lib):
    """-march=native must actually engage AVX-512 (plain -O3 gives SSE2), and -O3 must not turn the
    loop into a memcpy call -- glibc's memcpy switches to non-temporal stores past a size threshold,
    which would make the benchmark measure glibc's strategy rather than the memory system."""
    text = _disassemble()
    assert "%zmm" in text, "triad did not vectorize to AVX-512"
    assert "call" not in text or "memcpy" not in text, "loop was rewritten into memcpy"


def test_triad_computes_the_right_answer(lib):
    """a[i] = b[i] + q*c[i], checked against numpy. A benchmark that computes the wrong thing measures
    the wrong thing."""
    n = 4096
    q = 3.0
    rng = numpy.random.default_rng(0)
    b = rng.random(n)
    c = rng.random(n)
    a = numpy.zeros(n)
    ptr = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ns = ctypes.c_uint64(0)
    lib.triad_run_timed(ptr(a), ptr(b), ptr(c), ctypes.c_double(q), n, ctypes.byref(ns))
    assert numpy.allclose(a, b + q * c)
    assert ns.value > 0


def test_check_environment_reports_reasons_not_a_verdict():
    """The gate returns REASONS, so a caller cannot silently record a number taken on a box that
    cannot produce one. On an idle machine it is empty; the contents depend on the host, so only the
    shape is asserted here."""
    reasons = membench.check_environment(512 << 20)
    assert isinstance(reasons, list)
    assert all(isinstance(reason, str) for reason in reasons)


if __name__ == "__main__":
    library = membench.build()
    test_chain_is_one_hamiltonian_cycle_per_stride(library, 256)
    test_chains_are_disjoint_cycles_for_every_concurrency(library, 8)
    test_chase_setup_rejects_unusable_parameters(library)
    test_chase_hops_and_faults_are_reported(library)
    test_chase_body_is_a_bare_dependent_load(library)
    test_triad_vectorizes_and_is_not_rewritten_to_memcpy(library)
    test_triad_computes_the_right_answer(library)
    test_check_environment_reports_reasons_not_a_verdict()
    print("membench tests PASS")
