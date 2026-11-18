# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ReadableNames gives generated containers short free names, also when their stem starts with a reserved prefix."""

import numpy as np
import pytest

import dace
from dace.ordered import OrderedSet
from dace.transformation.passes.readable_names import ReadableNames, fresh, is_free


@pytest.mark.parametrize("stem", ["cpf_tmp", "omp_buf", "dace_acc", "__x"])
def test_a_reserved_stem_gets_a_free_name(stem: str):
    taken = OrderedSet(["tmp"])
    name = fresh(stem, taken)
    assert name == "tmp1" and name in taken and is_free("tmp2", taken)


def test_a_transient_with_a_reserved_stem_is_renamed_and_computes_the_same():
    sdfg = dace.SDFG("reserved_stem")
    sdfg.add_array("a", [8], dace.float64)
    sdfg.add_array("b", [8], dace.float64)
    sdfg.add_transient("__cpf_stage", [8], dace.float64)
    state = sdfg.add_state()
    staged = state.add_access("__cpf_stage")
    state.add_mapped_tasklet(
        "scale",
        {"i": "0:8"},
        {"x": dace.Memlet("a[i]")},
        "y = 2 * x",
        {"y": dace.Memlet("__cpf_stage[i]")},
        output_nodes={"__cpf_stage": staged},
        external_edges=True,
    )
    state.add_nedge(staged, state.add_write("b"), dace.Memlet("__cpf_stage[0:8]"))
    ReadableNames().apply_pass(sdfg, {})
    assert "__cpf_stage" not in sdfg.arrays and "tmp" in sdfg.arrays
    a, b = np.arange(8.0), np.zeros(8)
    sdfg(a=a, b=b)
    np.testing.assert_array_equal(b, 2 * a)
