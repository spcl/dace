# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop-invariant Scalar producer feeding a tile bridge is broadcast, never copied.

``WidenAccesses`` deliberately keeps a loop-invariant read from a lane-dependent source a
``Scalar`` (widen_accesses.py:514): the value is identical across lanes, so it is a broadcast
operand. ``InsertTileLoadStore`` used to override that -- ``_rewire_producers_to_bridge`` attached
the WHOLE-bridge memlet ``bridge[0:W]`` to whatever producer it found, the Scalar included.
``validate()`` let the resulting AN-to-AN edge through (it checks ``subset`` against the memlet's
own ``data``, and ``other_subset`` is ``None``), and the ``TileStore`` then wrote W lanes out of a
buffer whose only defined element is lane 0 -- a silent wrong answer, surfacing only once
``InsertExplicitCopies`` derived the matching source subset and refused it out-of-bounds.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest

import dace
from dace import data as dd
from dace.libraries.tileops import TileLoad, TileStore
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg.nodes import AccessNode
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')
WIDTH = 8


def vectorized_invariant_scalar_into_lane_indexed_write(name: str) -> dace.SDFG:
    """``buf[i] = 2*scale[0]`` (loop-invariant) then ``res[i] = buf[i]*A[i]``, vectorized at 8.

    The read-back of ``buf`` makes it an RMW bridge, which ``StageGlobalArrayThroughScalars``
    routes through a ``Scalar``; the lane-indexed write makes ``InsertTileLoadStore`` stage a
    ``(8,)`` tile bridge in front of it. That pairing is the CloudSC ``zldifdt`` idiom.
    """

    @dace.program
    def kernel(A: dace.float64[N], scale: dace.float64[1], buf: dace.float64[N], res: dace.float64[N]):
        for i in dace.map[0:N]:
            buf[i] = 2.0 * scale[0]
            res[i] = buf[i] * A[i]

    sdfg = kernel.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(WIDTH, ), target_isa=detect_host_isa())).apply_pass(sdfg, {})
    sdfg.name = name
    return sdfg


def single_element_producers_with_a_multi_element_memlet(sdfg: dace.SDFG):
    """Every edge whose source holds one element yet moves more than one.

    Read off the memlet's own region rather than its source side: an over-wide bridge memlet
    names the DESTINATION and leaves ``other_subset`` unset, so the source side reports the
    descriptor's own ``[0:1]`` and hides the mismatch -- which is exactly why ``validate()``
    accepted the edge.
    """
    offenders = []
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for edge in state.edges():
                if not isinstance(edge.src, AccessNode):
                    continue
                desc = nested.arrays.get(edge.src.data)
                if desc is None:
                    continue
                if not (isinstance(desc, dd.Scalar) or tuple(str(s) for s in desc.shape) == ('1', )):
                    continue
                subset = edge.data.subset
                if subset is not None and subset.num_elements() != 1:
                    offenders.append(f"{state.label}: {edge.src} -> {edge.dst} [{edge.data}]")
    return offenders


def count_nodes(sdfg: dace.SDFG, node_type) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, node_type))


def test_the_invariant_scalar_reaches_its_tile_bridge_through_a_broadcast():
    """Structure: a ``TileLoad(src_kind='Scalar')`` splat stands between Scalar and bridge."""
    sdfg = vectorized_invariant_scalar_into_lane_indexed_write('splat_structure')

    # The map really was tiled -- a refused kernel is restored un-tiled and would satisfy every
    # assertion below by having no tile chain at all.
    assert count_nodes(sdfg, TileStore) > 0
    splats = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileLoad) and n.src_kind == 'Scalar']
    assert splats, 'the Scalar producer was wired to the tile bridge without a broadcast'
    for splat in splats:
        assert tuple(splat.widths) == (WIDTH, )


def test_no_single_element_producer_claims_a_full_tile():
    """The Scalar no longer carries the whole-bridge memlet, and the graph validates."""
    sdfg = vectorized_invariant_scalar_into_lane_indexed_write('splat_no_overwide')

    assert count_nodes(sdfg, TileStore) > 0
    assert single_element_producers_with_a_multi_element_memlet(sdfg) == []
    sdfg.validate()


@pytest.mark.parametrize('length', [40, 37])
def test_every_lane_receives_the_invariant_value(length):
    """Numerics: all ``length`` entries hold the broadcast value, not just lane 0.

    ``37`` also drives the masked tail map, which carried the same Scalar-to-bridge edge, so this
    is not a remainder-only guard.
    """
    sdfg = vectorized_invariant_scalar_into_lane_indexed_write(f'splat_numeric_{length}')
    assert count_nodes(sdfg, TileStore) > 0

    rng = np.random.default_rng(seed=20260911)
    a = rng.random(length)
    scale = np.array([3.5])
    buf = np.full(length, np.nan)
    res = np.full(length, np.nan)
    sdfg(A=a, scale=scale, buf=buf, res=res, N=length)

    assert np.array_equal(buf, np.full(length, 7.0))
    assert np.allclose(res, 7.0 * a, rtol=0.0, atol=0.0)


if __name__ == '__main__':
    test_the_invariant_scalar_reaches_its_tile_bridge_through_a_broadcast()
    test_no_single_element_producer_claims_a_full_tile()
    for n in (40, 37):
        test_every_lane_receives_the_invariant_value(n)
