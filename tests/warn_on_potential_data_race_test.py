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
        with dace.config.set_temporary("experimental.check_race_conditions", value=True):
            sdfg(N=N, A=A, B=B)


def test_memlet_range_write_write_overlap_ranges():
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
        with dace.config.set_temporary("experimental.check_race_conditions", value=True):
            sdfg(N=N, A=A, B=B)

def test_memlet_range_write_read_overlap_ranges():
    sdfg = dace.SDFG('memlet_range_write_read_overlap_ranges')
    state = sdfg.add_state()
    N = dace.symbol("N", dtype=dace.int32)
    sdfg.add_array("A", (N,), dace.int32)
    A_read = state.add_read("A")
    A_write = state.add_write("A")
    sdfg.add_array("B", (N,), dace.int32)
    B = state.add_access("B")
    sdfg.add_array("C", (N,), dace.int32)
    C = state.add_access("C")
    state.add_mapped_tasklet(
        name="first_tasklet",
        code="b = a + 10",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "0:N"},
        external_edges=True,
        input_nodes={"A": A_read},
        output_nodes={"B": B}
    )
    state.add_mapped_tasklet(
        name="second_tasklet",
        code="a = c - 20",
        inputs={"c": dace.Memlet(data="C", subset="k")},
        outputs={"a": dace.Memlet(data="A", subset="k")},
        map_ranges={"k": "0:N"},
        external_edges=True,
        input_nodes={"C": C},
        output_nodes={"A": A_write}
    )

    N = 6
    A = np.arange(N, dtype=np.int32)
    B = np.zeros((N,), dtype=np.int32)
    C = 20 * A

    with pytest.warns(UserWarning):
        with dace.config.set_temporary('experimental', 'check_race_conditions', value=True):
            sdfg(N=N, A=A, B=B, C=C)

def test_memlet_overlap_ranges_two_access_nodes():
    sdfg = dace.SDFG('memlet_range_write_read_overlap_ranges')
    state = sdfg.add_state()
    N = dace.symbol("N", dtype=dace.int32)
    sdfg.add_array("A", (N,), dace.int32)
    A1 = state.add_access("A")
    A2 = state.add_access("A")
    sdfg.add_array("B", (N,), dace.int32)
    B1 = state.add_access("B")
    B2 = state.add_access("B")

    state.add_mapped_tasklet(
        name="first_tasklet",
        code="b = a + 10",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "0:N"},
        external_edges=True,
        input_nodes={"A": A1},
        output_nodes={"B": B1}
    )
    state.add_mapped_tasklet(
        name="second_tasklet",
        code="b = a - 20",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "0:N"},
        external_edges=True,
        input_nodes={"A": A2},
        output_nodes={"B": B2}
    )

    N = 6
    A = np.arange(N, dtype=np.int32)
    B = np.zeros((N,), dtype=np.int32)
    C = 20 * A

    with pytest.warns(UserWarning):
        with dace.config.set_temporary('experimental', 'check_race_conditions', value=True):
            sdfg(N=N, A=A, B=B, C=C)

def test_memlet_overlap_symbolic_ranges():
    sdfg = dace.SDFG('memlet_overlap_symbolic_ranges')
    state = sdfg.add_state()
    N = dace.symbol("N", dtype=dace.int32)
    sdfg.add_array("A", (2*N,), dace.int32)
    A = state.add_access("A")
    sdfg.add_array("B", (2*N,), dace.int32)
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
        map_ranges={"k": "0:2*N"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )

    N = 6
    A = np.arange(2*N, dtype=np.int32)
    B = np.zeros((2*N,), dtype=np.int32)
    C = 20 * A

    with pytest.warns(UserWarning):
        with dace.config.set_temporary('experimental', 'check_race_conditions', value=True):
            sdfg(N=N, A=A, B=B, C=C)

def test_constant_memlet_overlap():
    sdfg = dace.SDFG('constant_memlet_overlap')
    state = sdfg.add_state()
    sdfg.add_array("A", (12,), dace.int32)
    A = state.add_access("A")
    sdfg.add_array("B", (12,), dace.int32)
    B = state.add_access("B")

    state.add_mapped_tasklet(
        name="first_tasklet",
        code="b = a + 10",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "3:10"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )
    state.add_mapped_tasklet(
        name="second_tasklet",
        code="b = a - 20",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "6:12"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )

    A = np.arange(12, dtype=np.int32)
    B = np.zeros((12,), dtype=np.int32)

    with pytest.warns(UserWarning):
        with dace.config.set_temporary('experimental', 'check_race_conditions', value=True):
            sdfg(A=A, B=B)

def test_constant_memlet_almost_overlap():
    sdfg = dace.SDFG('constant_memlet_almost_overlap')
    state = sdfg.add_state()
    sdfg.add_array("A", (20,), dace.int32)
    A = state.add_access("A")
    sdfg.add_array("B", (20,), dace.int32)
    B = state.add_access("B")

    state.add_mapped_tasklet(
        name="first_tasklet",
        code="b = a + 10",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "3:10"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )
    state.add_mapped_tasklet(
        name="second_tasklet",
        code="b = a - 20",
        inputs={"a": dace.Memlet(data="A", subset="k")},
        outputs={"b": dace.Memlet(data="B", subset="k")},
        map_ranges={"k": "10:20"},
        external_edges=True,
        input_nodes={"A": A},
        output_nodes={"B": B}
    )

    A = np.arange(20, dtype=np.int32)
    B = np.zeros((20,), dtype=np.int32)

    with pytest.warns(UserWarning):
        warnings.simplefilter("error", UserWarning)
        with dace.config.set_temporary('experimental', 'check_race_conditions', value=True):
            sdfg(A=A, B=B)


if __name__ == '__main__':
    test_memlet_range_not_overlap_ranges()
    test_memlet_range_write_write_overlap_ranges()
    test_memlet_range_write_read_overlap_ranges()
    test_memlet_overlap_ranges_two_access_nodes()
    test_memlet_overlap_symbolic_ranges()
    test_constant_memlet_overlap()
    test_constant_memlet_almost_overlap()
