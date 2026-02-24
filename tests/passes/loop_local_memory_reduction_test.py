# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests LoopLocalMemoryReduction transformation."""

import numpy as np
import copy
import dace
from dace.sdfg.state import LoopRegion, CodeBlock
from dace.transformation.passes import LoopLocalMemoryReduction
from typing import Any, Dict


# Checks if LoopLocalMemoryReduction applied at least N times and if memory footprint was reduced for a specific option
def check_transformation_option(orig_sdfg: dace.SDFG, N: int, options: Dict[str, Any]):
    # Apply and validate
    orig_sdfg.validate()
    llmr_sdfg = copy.deepcopy(orig_sdfg)

    llmr = LoopLocalMemoryReduction()
    llmr.bitmask_indexing = options["bitmask_indexing"]
    llmr.next_power_of_two = options["next_power_of_two"]
    llmr.assume_positive_symbols = options["assume_positive_symbols"]
    llmr.apply_pass(llmr_sdfg, {})
    apps = llmr.num_applications

    # We can stop here if we don't expect any transformations
    if N == 0:
        assert apps == 0, f"Expected 0 applications, got {apps} with options {options}"
        return

    assert apps >= N, f"Expected at least {N} applications, got {apps} with options {options}"
    llmr_sdfg.validate()

    # Execute both SDFGs
    input_data_orig = {}
    sym_data = {}
    for sym in orig_sdfg.symbols:
        if sym not in orig_sdfg.constants:
            sym_data[sym] = 32
            input_data_orig[sym] = 32

    for argName, argType in orig_sdfg.arglist().items():
        if isinstance(argType, dace.data.Scalar):
            input_data_orig[argName] = argType.dtype.type(32)
            continue

        shape = []
        for entry in argType.shape:
            shape.append(dace.symbolic.evaluate(entry, {**orig_sdfg.constants, **sym_data}))
        shape = tuple(shape)
        arr = dace.ndarray(shape=shape, dtype=argType.dtype)
        arr[:] = np.random.rand(*arr.shape).astype(arr.dtype)
        input_data_orig[argName] = arr

    input_data_llmr = copy.deepcopy(input_data_orig)
    orig_sdfg(**input_data_orig)
    llmr_sdfg(**input_data_llmr)

    # No difference should be observable
    assert (sum(not np.array_equal(input_data_orig[argName], input_data_llmr[argName])
                for argName, argType in llmr_sdfg.arglist().items()) == 0
            ), f"Output differs after transformation! Options: {options}"

    # Memory footprint should be reduced
    orig_mem = sum(np.prod(arrType.shape) for arrName, arrType in orig_sdfg.arrays.items())
    llmr_mem = sum(np.prod(arrType.shape) for arrName, arrType in llmr_sdfg.arrays.items())
    orig_mem = dace.symbolic.evaluate(orig_mem, {**orig_sdfg.constants, **sym_data})
    llmr_mem = dace.symbolic.evaluate(llmr_mem, {**llmr_sdfg.constants, **sym_data})
    assert llmr_mem < orig_mem, f"Memory not reduced: {orig_mem} >= {llmr_mem}"


# Checks if LoopLocalMemoryReduction applied at least N times and if memory footprint was reduced for all options
def check_transformation(sdfg: dace.SDFG, N: int, aps: bool = False):
    for bitmask in [False, True]:
        for np2 in [False, True]:
            check_transformation_option(sdfg,
                                        N,
                                        options={
                                            "bitmask_indexing": bitmask,
                                            "next_power_of_two": np2,
                                            "assume_positive_symbols": aps
                                        })


def test_simple():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_no_offsets():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        for i in range(0, 32):
            b[i] = a[i] + a[i]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_window_one():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for i in range(0, 32):
            a[i] = c[i] * 2
            b[i] = a[i] + 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_window_one2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[:] = 0
        for i in range(0, 32):
            b[i] = a[i] + 2
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_self_dependency():

    @dace.program
    def tester(b: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            a[i] = a[i - 1] + a[i - 2]
            b[i] = a[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_self_dependency2():

    @dace.program
    def tester(b: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            a[i] = a[i - 1] + a[i - 2]
            b[i] = a[i - 1] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_non_transient():

    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_multiple_writes():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 31):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

            b[i + 1] = a[i] + a[i - 1]
            a[i + 1] = c[i + 1] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_gaps():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        a[2] = 2
        a[3] = 3
        for i in range(3, 32 - 1):
            b[i] = a[i - 1] + a[i - 3]
            a[i + 1] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_interleaved():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        a[2] = 2
        a[3] = 3
        a[4] = 4
        for i in range(3, 32 - 2):
            b[i] = a[i + 1] + a[i - 3]
            a[i - 1] = c[i] + 2
            a[i + 2] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_nonlinear_index():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        a[2] = 2
        a[3] = 3
        for i in range(2, 5):
            b[i] = a[i - 1] + a[i - 2]
            a[i * i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_nonlinear_step():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(32):
            a[j] = j

        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)

    for node, p in sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion):
            node.update_statement = CodeBlock("i = i * 2")

    check_transformation(sdfg, 0)


def test_nonconstant_step():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(32):
            a[j] = j

        step = 2
        for i in range(2, 32, step):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2
            step += 1

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_indirect_access():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(2):
            a[j] = j

        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            idx = int(b[i]) % 3
            a[idx] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_indirect_access2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(2):
            a[j] = j

        for i in range(2, 32):
            idx = int(b[i]) % 3
            b[i] = a[idx - 1] + a[idx - 2]
            a[idx] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_indirect_access3():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(2):
            a[j] = j

        idx = int(b[0]) % 3
        for i in range(2, 32):
            b[i] = a[idx + 1] + a[idx]
            a[idx + 2] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_indirect_access4():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(2):
            a[j] = j

        idx = int(b[0]) % 3
        for i in range(2, 32):
            b[i] = a[i - idx - 1] + a[i - idx - 2]
            a[i - idx] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_constant_index():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        for j in range(32):
            a[j] = j

        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[0] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_constant_index2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[3] = 3
        a[5] = 5

        for i in range(2, 32):
            b[i] = a[5] + a[3]
            a[7] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_constant_index3():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1

        for i in range(2, 32):
            b[i] = a[1] + a[0]
            a[0] = c[i] * 2
            a[1] = c[i] * 3

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_larger_step():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[4] = 4

        for i in range(2, 32, 4):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_larger_step2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[4] = 4

        for i in range(8, 32, 4):
            b[i] = a[i - 4] + a[i - 8]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_larger_index():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[4] = 4

        for i in range(2, 7):
            b[i] = a[4 * i - 1] + a[4 * i - 2]
            a[4 * i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_larger_index2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[4] = 4

        for i in range(2, 7):
            b[i] = a[4 * i - 4] + a[4 * i - 8]
            a[4 * i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_reverse_step():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([64], dace.float64)
        a[31] = 0
        a[30] = 1

        for i in range(29, 0, -1):
            b[i] = a[i + 1] + a[i + 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_reverse_index():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([64], dace.float64)
        a[31] = 0
        a[30] = 1

        for i in range(2, 31):
            b[i] = a[-i + 31 + 1] + a[-i + 31 + 2]
            a[-i + 31] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_reverse_index_step():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([64], dace.float64)
        a[0] = 0
        a[1] = 1

        for i in range(29, 0, -1):
            b[i] = a[-i + 31 - 1] + a[-i + 31 - 2]
            a[-i + 31] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_used_values():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2
        c[0] = a[0] + a[1]

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_used_values2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        c[0] = a[0] + a[1]
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_used_values3():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2
        a[0] = c[0]
        a[1] = c[1]
        c[0] = a[0] + a[1]

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_multidimensional():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(2):
            for jj in range(2):
                a[ii, jj] = ii + jj

        for i in range(2, 16):
            b[i, i] = a[i - 1, i - 1] + a[i - 2, i - 2]
            a[i, i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_multidimensional2():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(2):
            for jj in range(4):
                a[ii, jj] = ii + jj

        # TODO (later): In theory, in this case, LLMR could be applied
        for i in range(2, 14):
            b[i, i] = a[i - 1, i] + a[i - 2, i - 1]
            a[i, i + 1] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_multidimensional3():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(3):
            for jj in range(3):
                a[ii, jj] = ii + jj

        for i in range(2, 8):
            b[i, i] = a[2 * i - 2, 2 * i - 2] + a[2 * i - 4, 2 * i - 4]
            a[2 * i, 2 * i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_multidimensional_mixed():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(2):
            for jj in range(4):
                a[ii, jj] = ii + jj

        for i in range(2, 14):
            b[i, i] = a[i - 1, 0] + a[i - 2, i - 1]
            a[i, i + 1] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_multidimensional_constant():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for jj in range(2):
            a[5, jj] = jj

        for i in range(2, 16):
            b[i, i] = a[5, i - 2] + a[5, i - 1]
            a[5, i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_multidimensional_constant2():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for jj in range(2):
            a[3, jj] = jj
            a[4, jj] = jj
            a[5, jj] = jj

        for i in range(2, 16):
            b[i, i] = a[4, i - 2] + a[3, i - 1]
            a[5, i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_multidimensional_no_overwrite():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(11):
            for jj in range(12):
                a[ii, jj] = ii + jj

        for i in range(2, 13):
            b[i, i] = a[i - 1, i + 1] + a[i - 2, i - 1]
            a[i, i + 2] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_nested():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(2):
            for jj in range(4):
                a[ii, jj] = ii + jj

        for i in range(2, 14):
            for j in range(2, 14):
                b[i, j] = a[i - 1, j - 1] + a[i - 2, j - 2]
                a[i, j] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_nested2():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(2):
            for jj in range(4):
                a[ii, jj] = ii + jj

        for i in range(2, 14):
            for j in range(2, 14):
                b[i, j] = a[i - 1, j] + a[i - 2, j - 1]
                a[i, j + 1] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_nested3():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(16):
            for jj in range(3):
                a[ii, jj] = ii + jj

        # TODO (later): In theory, in this case, LLMR could be applied
        for i in range(0, 16):
            for j in range(2, 14):
                b[i, j] = a[i, j - 1] + a[i, j - 2]
                a[i, j] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_nested4():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for i in range(0, 16):
            for j in range(2, 14):
                a[i, j] = c[i] * 2
                b[i, j] = a[i, j] + a[i, j]

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_nested_mixed():

    @dace.program
    def tester(b: dace.float64[16, 16], c: dace.float64[16]):
        a = dace.define_local([16, 16], dace.float64)
        for ii in range(2):
            for jj in range(4):
                a[ii, jj] = ii + jj

        for i in range(2, 14):
            for j in range(2, 14):
                b[i, j] = a[i - 1, j - 1] + a[j - 2, i - 2]
                a[i, j] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_conditional():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            if i % 2 == 0:
                b[i] = a[i - 1] + a[i - 2]
            else:
                a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_conditional2():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            if i % 2 == 0:
                b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_conditional3():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            if i % 2 == 0:
                b[i] = a[i - 1] + a[i - 2]
            else:
                a[i] = c[i] * 2
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_conditional4():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            if i % 2 == 0:
                b[i] = a[i - 1] + a[i - 2]
            else:
                a[i * 2] = c[i] * 2
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0)


def test_conditional5():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            if i % 2 == 0:
                a[i] = c[i] + 2
            else:
                a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_conditional6():

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            if i % 2 == 0:
                a[i] = c[i] + 2
            elif i % 3 == 0:
                a[i] = c[i] - 2
            else:
                a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_symbolic_offset():

    N = dace.symbol('N')

    @dace.program
    def tester(b: dace.float64[32], c: dace.float64[32], d: dace.int64[1]):
        a = dace.define_local([32], dace.float64)
        a[0] = 0
        a[1] = 1
        N = d[0]
        for i in range(2, 32):
            b[i] = a[i - 1 + N] + a[i - 2 + N]
            a[i + N] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_symbolic_sizes():

    N = dace.symbol('N')

    @dace.program
    def tester(b: dace.float64[N], c: dace.float64[N]):
        a = dace.define_local([N], dace.float64)
        a[0] = 0
        a[1] = 1
        for i in range(2, N):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


def test_symbolic_k():

    N = dace.symbol('N')

    @dace.program
    def tester(b: dace.float64[64], c: dace.float64[64]):
        a = dace.define_local([64], dace.float64)
        for j in range(32):
            a[j] = j
        for i in range(32, 64):
            b[i] = a[i - N] + a[i - N]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 0, aps=False)
    check_transformation(sdfg, 1, aps=True)


def test_cloudsc():

    @dace.program
    def tester(pt: dace.float64[100000], tendency_tmp_t: dace.float64[100000], ptsphy: dace.float64, r2es: dace.float64,
               rtice: dace.float64, rtwat: dace.float64, rtwat_rtice_r: dace.float64, r3les: dace.float64,
               r4les: dace.float64, r3ies: dace.float64, r4ies: dace.float64, rtt: dace.float64,
               pap: dace.float64[100000], zqx: dace.float64, zsolqa: dace.float64[100000]):
        ztp1 = dace.define_local([100000], dace.float64)
        zfoeewmt = dace.define_local([100000], dace.float64)
        zlevap = dace.define_local([100000], dace.float64)

        zfoeewmt[1] = 0.0
        zlevap[0] = pap[0]
        zlevap[1] = pap[1]

        for jk in range(2, 100000):
            ztp1[jk] = pt[jk] + ptsphy * tendency_tmp_t[jk]
            zfoeewmt[jk] = min(
                ((r2es * ((min(1.0, ((max(rtice, min(rtwat, ztp1[jk])) - rtice) * rtwat_rtice_r)**2)) *
                          ((r3les * (ztp1[jk] - rtt)) / (ztp1[jk] - r4les)) +
                          (1.0 - (min(1.0, ((max(rtice, min(rtwat, ztp1[jk])) - rtice) * rtwat_rtice_r)**2))) *
                          ((r3ies * (ztp1[jk] - rtt)) / (ztp1[jk] - r4ies))))) / pap[jk], 0.5)
            zlevap[jk] = zfoeewmt[jk - 1] + max(zqx, 0.0)
            zsolqa[jk] = zsolqa[jk] + zlevap[jk - 1] * zlevap[jk - 2]

    sdfg = tester.to_sdfg(simplify=True)
    check_transformation(sdfg, 1)


if __name__ == "__main__":
    test_simple()
    test_no_offsets()
    test_window_one()
    test_window_one2()
    test_self_dependency()
    test_self_dependency2()
    test_non_transient()
    test_multiple_writes()
    test_gaps()
    test_interleaved()
    test_nonlinear_index()
    test_nonlinear_step()
    test_nonconstant_step()
    test_indirect_access()
    test_indirect_access2()
    test_indirect_access3()
    test_indirect_access4()
    test_constant_index()
    test_constant_index2()
    test_constant_index3()
    test_larger_step()
    test_larger_step2()
    test_larger_index()
    test_larger_index2()
    test_reverse_step()
    test_reverse_index()
    test_reverse_index_step()
    test_used_values()
    test_used_values2()
    test_used_values3()
    test_multidimensional()
    test_multidimensional2()
    test_multidimensional3()
    test_multidimensional_mixed()
    test_multidimensional_constant()
    test_multidimensional_constant2()
    test_multidimensional_no_overwrite()
    test_nested()
    test_nested2()
    test_nested3()
    test_nested4()
    test_nested_mixed()
    test_conditional()
    test_conditional2()
    test_conditional3()
    test_conditional4()
    test_conditional5()
    test_conditional6()
    test_symbolic_offset()
    test_symbolic_sizes()
    test_symbolic_k()
    test_cloudsc()
