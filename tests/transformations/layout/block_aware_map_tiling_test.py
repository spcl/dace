# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import numpy
import dace

from dace.transformation.layout.block_aware_map_tiling import BlockAwareMapTiling
from dace.transformation.layout.split_dimensions import SplitDimensions

N = dace.symbol("N")


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])


def _tasklet_access_has_int_floor(sdfg, arr):
    """int_floor on the innermost COMPUTE access (tasklet edge). The full-array map-boundary
    memlet legitimately over-approximates with int_floor; only the compute access must be clean
    for a perfect block match."""
    for state in sdfg.all_states():
        for e in state.edges():
            if e.data is None or e.data.data != arr or e.data.subset is None:
                continue
            if isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet):
                if "int_floor" in str(e.data.subset):
                    return True
    return False


def test_tile_then_block_is_perfect_match():
    """Auto-tiling the flat map by the block factors makes SplitDimensions take its
    perfect-block-match path: clean tile/offset accesses (no int_floor) and bit-exact."""
    original = madd.to_sdfg()

    sdfg = copy.deepcopy(original)
    sdfg.name = "madd_tiled_blocked"
    BlockAwareMapTiling(tile_sizes=(16, 4), divides_evenly=True).apply_pass(sdfg, {})
    sdfg.validate()

    split_map = {
        "A": ([True, True], [16, 4]),
        "B": ([True, True], [16, 4]),
    }
    SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()

    # Perfect match => the compute accesses use tile/offset directly, no int_floor residue.
    assert not _tasklet_access_has_int_floor(sdfg, "A"), "A compute access has int_floor -> not perfect match"
    assert not _tasklet_access_has_int_floor(sdfg, "B"), "B compute access has int_floor -> not perfect match"

    _N = 16 * 4 * 2
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))

    original(A=A.copy(), B=B.copy(), C=C0, N=_N)

    # Physically blocked inputs (mirror split_dimension_test): [N/16,16,N/4,4] -> transpose.
    A2 = A.reshape(_N // 16, 16, _N // 4, 4).transpose(0, 2, 1, 3).copy()
    B2 = B.reshape(_N // 16, 16, _N // 4, 4).transpose(0, 2, 1, 3).copy()
    sdfg(A=A2, B=B2, C=C1, N=_N)

    assert numpy.allclose(C1, C0)


if __name__ == "__main__":
    test_tile_then_block_is_perfect_match()
    print("block-aware map tiling test PASS")
