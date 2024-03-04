# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import warnings
import dace
import numpy as np
import pytest


def test_memlet_range_not_overlap_ranges():
    sdfg = dace.SDFG('memlet_range_not_overlap_ranges')
    state = sdfg.add_state()
    N = dace.symbol("N", dtype=dace.int32)
    sdfg.add_array("A", (N//2,), dace.int32)
    A = state.add_access("A")
    sdfg.add_array("B", (N,), dace.int32)
    B = state.add_access("B")
    state.add_mapped_tasklet(
        name="first_tasklet",
        code="b = a + 10",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "0:N//2"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )
    state.add_mapped_tasklet(
        name="second_tasklet",
        code="b = a - 20",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k+N//2")},
        map_ranges={"k": "0:N//2"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )

    N = 6
    A = np.arange(N//2, dtype=np.int32)
    B = np.zeros((N,), dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with dace.config.set_temporary("check_race_conditions", value=True):
            sdfg(N=N, A=A, B=B)


def test_memlet_range_overlap_ranges():
    sdfg = dace.SDFG('memlet_range_overlap_ranges')
    state = sdfg.add_state()
    N = dace.symbol("N", dtype=dace.int32)
    sdfg.add_array("A", (N,), dace.int32)
    A = state.add_access("A")
    sdfg.add_array("B", (N,), dace.int32)
    B = state.add_access("B")
    state.add_mapped_tasklet(
        name="first_tasklet",
        code="b = a + 10",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "0:N"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )
    state.add_mapped_tasklet(
        name="second_tasklet",
        code="b = a - 20",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "0:N"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )

    N = 6
    A = np.arange(N, dtype=np.int32)
    B = np.zeros((N,), dtype=np.int32)
    with pytest.warns(UserWarning):
        with dace.config.set_temporary("check_race_conditions", value=True):
            sdfg(N=N, A=A, B=B)


if __name__ == '__main__':
    test_memlet_range_not_overlap_ranges()
    test_memlet_range_overlap_ranges()
