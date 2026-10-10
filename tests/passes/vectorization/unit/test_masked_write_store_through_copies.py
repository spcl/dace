# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The store a masked write ``_o = IT(cond, val)`` gates is found through plain tile copies.

CloudSC's ``bn_ite_zqxfg`` writes a staging tile that is copied whole into the tile the store reads
(``stage -> tile_out -> store``); looking one access node past the tasklet missed the store and the vectorizer
refused the kernel. A staging tile something else also reads, or a copy into a non-transient, still hides it.
"""

import dace
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.tileops import MaskedCopyLibraryNode
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import ConvertTaskletsToTileOps

WIDTH = 8


def staged_store(second_reader: bool = False, tile_out_transient: bool = True):
    """``write -> stage -> tile_out -> store -> A``; returns the state, the tasklet's out edge and the store."""
    sdfg = dace.SDFG("staged_masked_store")
    sdfg.add_array("A", [WIDTH], dace.float64)
    sdfg.add_array("stage", [WIDTH], dace.float64, transient=True, storage=dace.StorageType.Register)
    sdfg.add_array("tile_out", [WIDTH], dace.float64, transient=tile_out_transient, storage=dace.StorageType.Register)
    state = sdfg.add_state()
    write = state.add_tasklet("write", {}, {"_o"}, "_o = 1.0")
    stage, tile_out = state.add_access("stage"), state.add_access("tile_out")
    out_edge = state.add_edge(write, "_o", stage, None, dace.Memlet(f"stage[0:{WIDTH}]"))
    state.add_edge(stage, None, tile_out, None, dace.Memlet(f"tile_out[0:{WIDTH}]"))
    store = MaskedCopyLibraryNode("store", (WIDTH,), has_mask=False)
    state.add_node(store)
    state.add_edge(tile_out, None, store, INPUT_CONNECTOR_NAME, dace.Memlet(f"tile_out[0:{WIDTH}]"))
    state.add_edge(store, OUTPUT_CONNECTOR_NAME, state.add_write("A"), None, dace.Memlet(f"A[0:{WIDTH}]"))
    if second_reader:
        sdfg.add_array("B", [WIDTH], dace.float64)
        state.add_edge(stage, None, state.add_write("B"), None, dace.Memlet(f"B[0:{WIDTH}]"))
    return state, out_edge, store


def test_the_store_behind_a_tile_copy_is_found():
    state, out_edge, store = staged_store()
    assert ConvertTaskletsToTileOps((WIDTH,))._find_downstream_store(state, out_edge) is store


def test_a_staging_tile_with_a_second_reader_hides_the_store():
    state, out_edge, _ = staged_store(second_reader=True)
    assert ConvertTaskletsToTileOps((WIDTH,))._find_downstream_store(state, out_edge) is None


def test_a_copy_into_a_non_transient_hides_the_store():
    state, out_edge, _ = staged_store(tile_out_transient=False)
    assert ConvertTaskletsToTileOps((WIDTH,))._find_downstream_store(state, out_edge) is None


if __name__ == "__main__":
    test_the_store_behind_a_tile_copy_is_found()
    test_a_staging_tile_with_a_second_reader_hides_the_store()
    test_a_copy_into_a_non_transient_hides_the_store()
