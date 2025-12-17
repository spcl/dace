import dace
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
            input_data_orig[argName] = 32
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
    for bitmask in [False]:
        for np2 in [False]:
            check_transformation_option(sdfg,
                                        N,
                                        options={
                                            "bitmask_indexing": bitmask,
                                            "next_power_of_two": np2,
                                            "assume_positive_symbols": aps
                                        })
            sdfg.save("transformed.sdfg")

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
