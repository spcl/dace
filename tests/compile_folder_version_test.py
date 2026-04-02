# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import uuid

import dace


def _make_test_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("test_sdfg_" + str(uuid.uuid1()).replace("-", "_"))
    state = sdfg.add_state()
    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )

    state.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("a[__i]"),
            "__in2": dace.Memlet("b[__i]"),
        },
        outputs={
            "__out": dace.Memlet("c[__i]"),
        },
        code="__out = __in1 + __in2",
        external_edges=True,
    )
    sdfg.validate()

    return sdfg


def _perform_compile_test_for(version: str):
    a, b, c = (np.array(np.random.rand(10), copy=True, dtype=dace.float64.as_numpy_dtype()) for _ in range(3))
    ref = a + b
    with dace.config.temporary_config() as Config:
        Config.set('compiler', 'build_folder_version', value=version)
        sdfg = _make_test_sdfg()
        csdfg = sdfg.compile()
        # PERFROM INSPECTION OF THE FOLDER

        csdfg(a=a, b=b, c=c)
        assert np.allclose(c, ref)


def test_w():
    _perform_compile_test_for("full")
