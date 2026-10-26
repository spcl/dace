# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Copy memlets are named by the container outside the scope they cross (spcl/dace#2037)."""

import numpy as np
import pytest

import dace
from dace.sdfg.utils import normalize_copy_memlets
from dace.sdfg.validation import InvalidSDFGEdgeError

N = 1024
TILE = 256


def staged(name: str, gpu: bool) -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.AccessNode]:
    """spcl/dace#2037: ``C[i + j] = A[i + j] * 2`` staging tile ``A[i:i+256]`` through ``loc``, with an inner
    access node of ``A`` that only renames it across the outer map."""
    sdfg = dace.SDFG(name)
    glob = dace.StorageType.GPU_Global if gpu else dace.StorageType.Default
    sdfg.add_array("A", [N], dace.float64, storage=glob, transient=gpu)
    sdfg.add_array("C", [N], dace.float64, storage=glob, transient=gpu)
    loc = dace.StorageType.GPU_Shared if gpu else dace.StorageType.Default
    sdfg.add_transient("loc", [TILE], dace.float64, storage=loc)
    state = sdfg.add_state("compute")
    outer = dace.ScheduleType.GPU_Device if gpu else dace.ScheduleType.Default
    inner = dace.ScheduleType.GPU_ThreadBlock if gpu else dace.ScheduleType.Sequential
    ome, omx = state.add_map("tiles", {"i": f"0:{N}:{TILE}"}, schedule=outer)
    ime, imx = state.add_map("elems", {"j": f"0:{TILE}"}, schedule=inner)
    alias, tile = state.add_access("A"), state.add_access("loc")
    tasklet = state.add_tasklet("scale", {"x": None}, {"y": None}, "y = x * 2")
    state.add_memlet_path(state.add_read("A"), ome, alias, memlet=dace.Memlet(f"A[i:i+{TILE}]"))
    state.add_edge(alias, None, tile, None, dace.Memlet(data="loc", subset=f"0:{TILE}", other_subset=f"i:i+{TILE}"))
    state.add_memlet_path(tile, ime, tasklet, dst_conn="x", memlet=dace.Memlet("loc[j]"))
    state.add_memlet_path(tasklet, imx, omx, state.add_write("C"), src_conn="y", memlet=dace.Memlet("C[i + j]"))
    if gpu:
        sdfg.add_array("hA", [N], dace.float64)
        sdfg.add_array("hC", [N], dace.float64)
        up, down = sdfg.add_state_before(state, "up"), sdfg.add_state_after(state, "down")
        up.add_nedge(up.add_read("hA"), up.add_write("A"), dace.Memlet(f"A[0:{N}]"))
        down.add_nedge(down.add_read("C"), down.add_write("hC"), dace.Memlet(f"hC[0:{N}]"))
    return sdfg, state, alias


def test_scope_alias_moves_no_data_on_cpu():
    """The renaming access node of ``A`` is dropped, not copied onto ``A``'s origin."""
    sdfg, _, _ = staged("copy_form_alias_cpu", gpu=False)
    sdfg.validate()
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert "copy_A_to_A" not in code and "A + i, A" not in code
    a = np.arange(N, dtype=np.float64)
    c = np.zeros(N)
    sdfg(A=a, C=c)
    np.testing.assert_array_equal(c, a * 2)
    np.testing.assert_array_equal(a, np.arange(N, dtype=np.float64))


@pytest.mark.gpu
def test_scope_alias_moves_no_data_on_gpu():
    """spcl/dace#2037: one block-collective load per tile, no ``GlobalToGlobal1D(A + i, 1, A)``."""
    sdfg, _, _ = staged("copy_form_alias_gpu", gpu=True)
    cuda = next(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cu")
    assert "GlobalToGlobal1D" not in cuda and "copy_A_to_A" not in cuda
    assert cuda.count("GlobalToShared1D") == 1
    # The collective spans the block the thread-block map launches, not the configured default
    assert f"GlobalToShared1D<double, {TILE}, 1, 1, {TILE}," in cuda
    a = np.arange(N, dtype=np.float64)
    c = np.zeros(N)
    sdfg(hA=a, hC=c)
    np.testing.assert_array_equal(c, a * 2)


def test_direct_copy_must_name_its_destination():
    """``A -> B`` named by ``A`` is refused; ``normalize_copy_memlets`` names ``B`` and swaps the subsets."""
    sdfg = dace.SDFG("copy_form_direct")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    state = sdfg.add_state()
    edge = state.add_edge(
        state.add_read("A"), None, state.add_write("B"), None, dace.Memlet(data="A", subset="2:6", other_subset="0:4")
    )
    with pytest.raises(InvalidSDFGEdgeError, match='must name "B"'):
        sdfg.validate()
    assert normalize_copy_memlets(sdfg) == 1
    sdfg.validate()
    assert (edge.data.data, str(edge.data.subset), str(edge.data.other_subset)) == ("B", "0:4", "2:6")


def test_stage_in_copy_must_name_the_outer_container():
    """A stage-in path is named by its source on every edge, as the propagated outer memlet must be."""
    sdfg = dace.SDFG("copy_form_stage_in")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_transient("loc", [TILE], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("tiles", {"i": f"0:{N}:{TILE}"})
    loc = state.add_access("loc")
    state.add_memlet_path(
        state.add_read("A"), me, loc, memlet=dace.Memlet(data="loc", subset=f"0:{TILE}", other_subset=f"i:i+{TILE}")
    )
    state.add_memlet_path(loc, mx, state.add_write("C"), memlet=dace.Memlet(data="C", subset=f"i:i+{TILE}"))
    with pytest.raises(InvalidSDFGEdgeError, match='must name "A"'):
        sdfg.validate()
    assert normalize_copy_memlets(sdfg) == 2  # the edge into the map and the edge out of it
    sdfg.validate()
    inner = state.in_edges(loc)[0]
    assert (inner.data.data, str(inner.data.subset), str(inner.data.other_subset)) == (
        "A",
        f"i:i + {TILE}",
        f"0:{TILE}",
    )


def test_stage_out_copy_must_name_the_outer_container():
    """A stage-out path is named by its destination on every edge, not by the local buffer."""
    sdfg = dace.SDFG("copy_form_stage_out")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_transient("loc", [TILE], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("tiles", {"i": f"0:{N}:{TILE}"})
    loc = state.add_access("loc")
    state.add_memlet_path(state.add_read("A"), me, loc, memlet=dace.Memlet(data="A", subset=f"i:i+{TILE}"))
    state.add_memlet_path(
        loc, mx, state.add_write("C"), memlet=dace.Memlet(data="loc", subset=f"0:{TILE}", other_subset=f"i:i+{TILE}")
    )
    with pytest.raises(InvalidSDFGEdgeError, match='must name "C"'):
        sdfg.validate()
    assert normalize_copy_memlets(sdfg) == 2  # the edge into the map exit and the edge out of it
    sdfg.validate()
    inner = state.out_edges(loc)[0]
    assert (inner.data.data, str(inner.data.subset), str(inner.data.other_subset)) == (
        "C",
        f"i:i + {TILE}",
        f"0:{TILE}",
    )


def test_same_container_copy_subset_must_be_the_owner_side():
    """``A[0:4] -> A[4:8]`` written as ``A[0:4] -> [4:8]`` names ``A`` but puts the source region in ``subset``."""
    sdfg = dace.SDFG("copy_form_same_container")
    sdfg.add_array("A", [8], dace.float64)
    state = sdfg.add_state()
    edge = state.add_nedge(state.add_read("A"), state.add_write("A"), dace.Memlet("A[0:4] -> [4:8]"))
    with pytest.raises(InvalidSDFGEdgeError, match='"subset" must be the region of the "A" end'):
        sdfg.validate()
    assert normalize_copy_memlets(sdfg) == 1
    sdfg.validate()
    assert (str(edge.data.subset), str(edge.data.other_subset)) == ("4:8", "0:4")
    assert (str(edge.data.get_src_subset(edge, state)), str(edge.data.get_dst_subset(edge, state))) == ("0:4", "4:8")
    a = np.arange(8, dtype=np.float64)
    sdfg(A=a)
    np.testing.assert_array_equal(a, [0, 1, 2, 3, 0, 1, 2, 3])


if __name__ == "__main__":
    test_scope_alias_moves_no_data_on_cpu()
    test_scope_alias_moves_no_data_on_gpu()
    test_direct_copy_must_name_its_destination()
    test_stage_in_copy_must_name_the_outer_container()
    test_stage_out_copy_must_name_the_outer_container()
    test_same_container_copy_subset_must_be_the_owner_side()
