# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MapFusionVertical`` fuses an intermediate the first Map reads and writes other than point-wise (a row recurrence)
in place, without privatizing it, when no other iteration of the first Map writes what the second reads."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.dataflow import MapFusionVertical
from dace.transformation.helpers import nest_state_subgraph

N = dace.symbol("N")
SIZE = 24


def top_level_maps(sdfg: dace.SDFG) -> int:
    return sum(
        1 for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None
    )


@dace.program
def recurrence_then_offset_reader(p0: dace.float64[N], q: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    for i in dace.map[0:N]:
        p[i, 0] = p0[i]
        for j in range(1, N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0
    for k in dace.map[0:N]:
        for j in range(1, N):
            q[k, j] = q[k, j - 1] + p[k, j - 1]


@dace.program
def reader_of_the_row_above(p0: dace.float64[N], q: dace.float64[N, N]):
    p = np.empty((N, N), dtype=np.float64)
    for i in dace.map[0:N]:
        p[i, 0] = p0[i]
        for j in range(1, N):
            p[i, j] = 0.5 * p[i, j - 1] + 1.0
    for i in dace.map[1:N]:
        for j in range(1, N):
            q[i, j] = q[i, j - 1] + p[i - 1, j]


def rows_reference(p0: np.ndarray, q: np.ndarray) -> None:
    p = np.empty((len(p0), len(p0)))
    p[:, 0] = p0
    for j in range(1, len(p0)):
        p[:, j] = 0.5 * p[:, j - 1] + 1.0
    for j in range(1, len(p0)):
        q[:, j] = q[:, j - 1] + p[:, j - 1]


def test_a_row_recurrence_and_its_reader_fuse_in_place():
    sdfg = recurrence_then_offset_reader.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionVertical) == 1
    assert top_level_maps(sdfg) == 1
    # Not privatized: the fused body reads and writes the original ``p``
    assert not any(name.startswith("__map_fusion_") for name in sdfg.arrays)
    sdfg.validate()
    rng = np.random.default_rng(0)
    p0, q = rng.random(SIZE), rng.random((SIZE, SIZE))
    expected = q.copy()
    rows_reference(p0, expected)
    sdfg(p0=p0, q=q, N=SIZE)
    assert np.allclose(q, expected)


def test_a_reader_of_another_iterations_row_is_refused():
    sdfg = reader_of_the_row_above.to_sdfg(simplify=True)
    assert sdfg.apply_transformations_repeated(MapFusionVertical) == 0
    assert top_level_maps(sdfg) == 2


@dace.program
def scale_in_place_then_read(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] * 2
    for i in dace.map[0:N]:
        B[i] = A[i] + 1


def test_maps_whose_bodies_are_nested_sdfgs_fuse_through_an_array_the_first_updates_point_wise():
    """A nested body reading and writing ``A[i]`` updates one element per iteration, not the whole array."""
    sdfg = scale_in_place_then_read.to_sdfg(simplify=True)
    state = sdfg.start_block
    for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
        body = [n for n in state.scope_children()[entry] if n is not state.exit_node(entry)]
        nest_state_subgraph(sdfg, state, SubgraphView(state, body))
    propagate_memlets_sdfg(sdfg)
    A, B = np.random.rand(SIZE), np.zeros(SIZE)
    expected = A * 2 + 1

    assert sdfg.apply_transformations_repeated(MapFusionVertical) == 1

    assert top_level_maps(sdfg) == 1
    sdfg(A=A, B=B, N=SIZE)
    assert np.allclose(B, expected)


def updates_of_one_column(name: str, shift: int) -> dace.SDFG:
    """Two Maps updating ``a[i, 1]``, the second reading ``a[i + shift, 1]``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [SIZE + 1, 3], dace.float64)
    sdfg.add_array("c", [SIZE], dace.float64)
    state = sdfg.add_state()
    chain = [state.add_access("a") for _ in range(3)]
    state.add_mapped_tasklet(
        "first",
        {"i": f"0:{SIZE}"},
        {"p": dace.Memlet("a[i, 1]"), "q": dace.Memlet("c[i]")},
        "o = p * q + 1",
        {"o": dace.Memlet("a[i, 1]")},
        input_nodes={"a": chain[0]},
        output_nodes={"a": chain[1]},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "second",
        {"j": f"0:{SIZE}"},
        {"p": dace.Memlet(f"a[j + {shift}, 1]")},
        "o = p * 2",
        {"o": dace.Memlet("a[j, 1]")},
        input_nodes={"a": chain[1]},
        output_nodes={"a": chain[2]},
        external_edges=True,
    )
    return sdfg


def relaxed_fusion() -> MapFusionVertical:
    return MapFusionVertical(relaxed_dataflow=True)


def test_two_updates_of_one_container_fuse_in_place():
    sdfg = updates_of_one_column("two_updates", 0)
    assert sdfg.apply_transformations_repeated(relaxed_fusion()) == 1
    sdfg.validate()
    assert top_level_maps(sdfg) == 1
    array, factor = np.arange(3.0 * (SIZE + 1)).reshape(SIZE + 1, 3), np.arange(1.0, SIZE + 1)
    expected = array.copy()
    expected[:-1, 1] = (expected[:-1, 1] * factor + 1) * 2
    sdfg(a=array, c=factor)
    assert np.array_equal(array, expected)


def test_an_update_reading_another_iteration_stays_apart():
    sdfg = updates_of_one_column("shifted_updates", 1)
    assert sdfg.apply_transformations_repeated(relaxed_fusion()) == 0


def test_a_piece_computed_from_an_earlier_piece_fuses_without_a_cycle():
    """Three Maps write columns 0, 1, 2 of ``a``, each from the columns before it: fused in place, the second fusion
    finds two pieces of ``a`` in the body, one computed from the other."""
    sdfg = dace.SDFG("pieces_of_one_container")
    sdfg.add_transient("a", [SIZE, 3], dace.float64)
    sdfg.add_array("out", [SIZE, 3], dace.float64)
    sdfg.add_array("c", [SIZE], dace.float64)
    state = sdfg.add_state()
    chain = [state.add_access("a") for _ in range(3)]
    state.add_mapped_tasklet(
        "first", {"i": f"0:{SIZE}"}, {"q": dace.Memlet("c[i]")}, "o = q + 1", {"o": dace.Memlet("a[i, 0]")},
        output_nodes={"a": chain[0]}, external_edges=True,
    )  # fmt: skip
    state.add_mapped_tasklet(
        "second", {"i": f"0:{SIZE}"}, {"p": dace.Memlet("a[i, 0]")}, "o = -p", {"o": dace.Memlet("a[i, 1]")},
        input_nodes={"a": chain[0]}, output_nodes={"a": chain[1]}, external_edges=True,
    )  # fmt: skip
    state.add_mapped_tasklet(
        "third", {"i": f"0:{SIZE}"}, {"p": dace.Memlet("a[i, 0]"), "r": dace.Memlet("a[i, 1]")}, "o = p - 2 * r",
        {"o": dace.Memlet("a[i, 2]")}, input_nodes={"a": chain[1]}, output_nodes={"a": chain[2]}, external_edges=True,
    )  # fmt: skip
    state.add_nedge(chain[2], state.add_access("out"), dace.Memlet(f"a[0:{SIZE}, 0:3]"))
    sdfg.validate()
    assert sdfg.apply_transformations_repeated(relaxed_fusion(), validate_all=True) == 2
    assert top_level_maps(sdfg) == 1
    array, values = np.zeros((SIZE, 3)), np.arange(1.0, SIZE + 1)
    sdfg(out=array, c=values)
    assert np.array_equal(array, np.stack([values + 1, -(values + 1), 3 * (values + 1)], axis=1))


if __name__ == "__main__":
    test_a_row_recurrence_and_its_reader_fuse_in_place()
    test_a_reader_of_another_iterations_row_is_refused()
    test_maps_whose_bodies_are_nested_sdfgs_fuse_through_an_array_the_first_updates_point_wise()
    test_two_updates_of_one_container_fuse_in_place()
    test_an_update_reading_another_iteration_stays_apart()
    test_a_piece_computed_from_an_earlier_piece_fuses_without_a_cycle()
