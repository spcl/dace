# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stack or heap placement of register arrays, chosen by ``StorageType.Register(dynamic=...)``."""

import re
import contextlib
import json
import os
import sys
import tempfile

import numpy as np
import pytest

import dace
from dace.codegen.targets.cpu import register_array_on_stack
from dace.dtypes import AllocationLifetime, StorageType, is_dynamic_register

N = dace.symbol("N", dtype=dace.int64)

#: A heap allocation of ``tmp``: plain ``new`` or, aligned, main's ``dace::aligned_new_array``.
HEAP_ALLOC = re.compile(r"tmp = (?:new\b|dace::aligned_new_array<)")


def register_scratch_sdfg(
    name: str, size, placement: StorageType, lifetime: AllocationLifetime = AllocationLifetime.Scope
) -> dace.SDFG:
    """``b = a + 1`` through a register transient ``tmp`` that two states share."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", (size,), dace.float64)
    sdfg.add_array("b", (size,), dace.float64)
    sdfg.add_transient("tmp", (size,), dace.float64, storage=placement, lifetime=lifetime)
    rng = f"0:{size}"
    first = sdfg.add_state("first", is_start_block=True)
    first.add_mapped_tasklet(
        "fill",
        {"i": rng},
        {"inp": dace.Memlet("a[i]")},
        "o = inp + 1.0",
        {"o": dace.Memlet("tmp[i]")},
        external_edges=True,
    )
    second = sdfg.add_state_after(first, "second")
    second.add_mapped_tasklet(
        "drain", {"i": rng}, {"inp": dace.Memlet("tmp[i]")}, "o = inp", {"o": dace.Memlet("b[i]")}, external_edges=True
    )
    sdfg.validate()
    return sdfg


def run_scratch(sdfg: dace.SDFG, size: int, **symbols):
    rng = np.random.default_rng(42)
    a = rng.random(size)
    b = np.zeros(size)
    sdfg(a=a, b=b, **symbols)
    np.testing.assert_allclose(b, a + 1.0, rtol=0, atol=0)


AUTO = StorageType.Register
DYNAMIC = StorageType.Register(dynamic=True)
STATIC = StorageType.Register(dynamic=False)

PLACEMENT_CASES = [
    (16, AUTO, True),
    (8192, AUTO, True),
    (8193, AUTO, False),
    (N, AUTO, False),
    (16, STATIC, True),
    (N, STATIC, False),
    (16, DYNAMIC, True),
    (8193, DYNAMIC, True),
    (N, DYNAMIC, True),
]


@contextlib.contextmanager
def captured_stderr_fd():
    """Collects what compiled code writes to file descriptor 2 into the one-item list it yields."""
    captured = [""]
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+") as sink:
        os.dup2(sink.fileno(), 2)
        try:
            yield captured
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            sink.seek(0)
            captured[0] = sink.read()


def on_stack(sdfg: dace.SDFG) -> bool:
    desc = sdfg.arrays["tmp"]
    return register_array_on_stack(sdfg, desc, desc.total_size, desc.lifetime, declared=False)


@pytest.mark.parametrize("size, placement, stack", PLACEMENT_CASES)
def test_constant_sizes_follow_the_size_limit_and_only_dynamic_puts_a_symbolic_size_on_the_stack(
    size, placement, stack
):
    assert on_stack(register_scratch_sdfg("placement", size, placement)) is stack


def test_a_large_constant_array_respects_max_stack_array_size():
    """Lowering the byte limit moves a constant-sized array to the heap."""
    sdfg = register_scratch_sdfg("placement_bytes", 1024, AUTO)
    with dace.config.set_temporary("compiler", "max_stack_array_size", value=1024):
        assert not on_stack(sdfg)


@pytest.mark.parametrize("value, stack", [(16, True), (8193, False)])
def test_a_size_given_by_an_sdfg_constant_follows_the_size_limit(value, stack):
    """The constant's dtype differs from the symbol's, as for the Polybench sizes."""
    sdfg = register_scratch_sdfg(f"placement_constant_{value}", dace.symbol("NC", dace.int32), AUTO)
    sdfg.add_constant("NC", np.int64(value))
    assert on_stack(sdfg) is stack


def test_a_symbolic_stack_array_is_a_variable_length_array():
    sdfg = register_scratch_sdfg("vla_stack", N, DYNAMIC)
    code = sdfg.generate_code()[0].clean_code
    assert "double tmp[Max(1, N)];" in code, code
    assert not HEAP_ALLOC.search(code), code
    run_scratch(sdfg, 32, N=32)


def test_a_zero_extent_stack_array_has_a_positive_bound():
    """A zero-length VLA is undefined behaviour, and Fortran automatic arrays are often empty."""
    sdfg = register_scratch_sdfg("vla_zero", N, DYNAMIC)
    args = dace.Config.get("compiler", "cpu", "args")
    with (
        dace.config.set_temporary("compiler", "cpu", "args", value=f"{args} -fsanitize=vla-bound"),
        dace.config.set_temporary("compiler", "cpu", "libs", value="ubsan"),
        captured_stderr_fd() as stderr,
    ):
        run_scratch(sdfg, 0, N=0)
    assert "runtime error" not in stderr[0], stderr[0]


def test_a_zeroed_symbolic_stack_array_is_cleared_by_memset():
    """A VLA cannot take a brace initializer."""
    sdfg = register_scratch_sdfg("vla_setzero", N, DYNAMIC)
    for node in sdfg.start_block.data_nodes():
        if node.data == "tmp":
            node.setzero = True
    code = sdfg.generate_code()[0].clean_code
    assert "memset(tmp, 0, sizeof(double)*(Max(1, N)));" in code, code


def test_an_auto_symbolic_register_array_stays_on_the_heap():
    sdfg = register_scratch_sdfg("vla_auto", N, AUTO)
    with pytest.warns(UserWarning, match="Variable-length array tmp"):
        code = sdfg.generate_code()[0].clean_code
    assert HEAP_ALLOC.search(code), code
    assert "double tmp[" not in code, code


def test_a_small_constant_register_array_stays_aligned_on_the_stack():
    sdfg = register_scratch_sdfg("stack_constant", 16, AUTO)
    code = sdfg.generate_code()[0].clean_code
    assert "double tmp[16]  DACE_ALIGN(64);" in code, code
    assert not HEAP_ALLOC.search(code), code
    run_scratch(sdfg, 16)


def test_a_large_constant_register_array_moves_to_the_heap():
    sdfg = register_scratch_sdfg("heap_constant", 8193, AUTO)
    with pytest.warns(
        UserWarning, match="Register array tmp with 8193 elements was allocated on the heap instead of the stack"
    ):
        code = sdfg.generate_code()[0].clean_code
        run_scratch(sdfg, 8193)
    assert HEAP_ALLOC.search(code), code


def test_a_global_lifetime_keeps_a_symbolic_stack_array_on_the_heap():
    """A VLA dies with its block, while a Global array is declared outside it. Persistent and External
    register arrays are rejected by validation."""
    sdfg = register_scratch_sdfg("vla_global", N, DYNAMIC, AllocationLifetime.Global)
    with pytest.warns(UserWarning, match="Variable-length array tmp"):
        code = sdfg.generate_code()[0].clean_code
        run_scratch(sdfg, 8, N=8)
    assert "double tmp[" not in code, code


def test_a_split_declaration_keeps_a_symbolic_stack_array_on_the_heap():
    """A size that only an interstate edge defines is declared in one block and allocated in another,
    and a VLA cannot bridge that split."""
    K = dace.symbol("K", dtype=dace.int64)
    sdfg = dace.SDFG("vla_split_declaration")
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_array("a", (N,), dace.float64)
    sdfg.add_array("b", (N,), dace.float64)
    sdfg.add_transient("tmp", (K,), dace.float64, storage=DYNAMIC)
    init = sdfg.add_state("init", is_start_block=True)
    first = sdfg.add_state("first")
    second = sdfg.add_state("second")
    sdfg.add_edge(init, first, dace.InterstateEdge(assignments={"K": "N"}))
    sdfg.add_edge(first, second, dace.InterstateEdge())
    first.add_mapped_tasklet(
        "fill",
        {"i": "0:K"},
        {"inp": dace.Memlet("a[i]")},
        "o = inp + 1.0",
        {"o": dace.Memlet("tmp[i]")},
        external_edges=True,
    )
    second.add_mapped_tasklet(
        "drain",
        {"i": "0:K"},
        {"inp": dace.Memlet("tmp[i]")},
        "o = inp",
        {"o": dace.Memlet("b[i]")},
        external_edges=True,
    )
    sdfg.validate()

    with pytest.warns(UserWarning, match="Variable-length array tmp"):
        code = sdfg.generate_code()[0].clean_code
        run_scratch(sdfg, 8, N=8)
    assert "double tmp[" not in code, code


@pytest.mark.parametrize("placement", [DYNAMIC, STATIC])
def test_stack_placement_survives_clone_and_serialization(placement):
    sdfg = register_scratch_sdfg("placement_roundtrip", N, placement)
    assert sdfg.arrays["tmp"].clone().storage is placement
    reloaded = dace.SDFG.from_json(json.loads(json.dumps(sdfg.to_json())))
    assert reloaded.arrays["tmp"].storage is placement


def test_the_bare_register_template_is_unchanged():
    """The template equals every instance, leaves the placement open and is stored by its name, as before."""
    assert AUTO == DYNAMIC and AUTO == STATIC and DYNAMIC != STATIC
    assert is_dynamic_register(AUTO) is None
    sdfg = register_scratch_sdfg("register_template_roundtrip", 16, AUTO)
    stored = json.loads(json.dumps(sdfg.to_json()))
    assert stored["attributes"]["_arrays"]["tmp"]["attributes"]["storage"] == "Register"
    assert dace.SDFG.from_json(stored).arrays["tmp"].storage is AUTO


if __name__ == "__main__":
    for case in PLACEMENT_CASES:
        test_constant_sizes_follow_the_size_limit_and_only_dynamic_puts_a_symbolic_size_on_the_stack(*case)
    test_a_large_constant_array_respects_max_stack_array_size()
    test_a_size_given_by_an_sdfg_constant_follows_the_size_limit(16, True)
    test_a_size_given_by_an_sdfg_constant_follows_the_size_limit(8193, False)
    test_a_symbolic_stack_array_is_a_variable_length_array()
    test_a_zero_extent_stack_array_has_a_positive_bound()
    test_a_zeroed_symbolic_stack_array_is_cleared_by_memset()
    test_an_auto_symbolic_register_array_stays_on_the_heap()
    test_a_small_constant_register_array_stays_aligned_on_the_stack()
    test_a_large_constant_register_array_moves_to_the_heap()
    test_a_global_lifetime_keeps_a_symbolic_stack_array_on_the_heap()
    test_a_split_declaration_keeps_a_symbolic_stack_array_on_the_heap()
    test_stack_placement_survives_clone_and_serialization(DYNAMIC)
    test_stack_placement_survives_clone_and_serialization(STATIC)
    test_the_bare_register_template_is_unchanged()
