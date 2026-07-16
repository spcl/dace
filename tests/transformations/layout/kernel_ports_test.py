# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the ported SC26 reference kernels (k01, k02, k03, k06, k07, k09, k10, k13, k14) through the
layout sweep, verifying every candidate against the numpy oracle.

Each kernel exercises a different primitive family: k01 STREAM triad (Block, the fragmentation
instrument), k02 transpose (Block + Permute), k03 complex conjugate (Zip SoA/AoS/AoSoA), k06
gather-accumulate-scatter (Shuffle, complex128), k07 ICON stencil (Permute 3D), k09 particle fields
(Zip AoSoA), k10 OMEN windowed contraction (cross-phase Permute of a transient, complex128), k13 1x1
conv (Permute + nChw16c Block), k14 Eytzinger search (Shuffle). For the kernels with a large
candidate set (k02: 9, k13: 25) a representative SUBSET is swept so the test stays fast -- the
module's ``candidates()`` still enumerates the full family for a real (timed) sweep. Correctness is
the invariant; timing is not asserted."""
import numpy

from dace.transformation.layout.brute_force import sweep, best
from tests.transformations.layout.kernels import (k01_stream_triad, k02_transpose_blocked,
                                                  k03_complex_conjugate_zip, k06_gather_accumulate_scatter,
                                                  k07_icon_stencil, k09_particle_field_affinity,
                                                  k10_omen_windowed_contraction, k13_conv1x1_channel_layouts,
                                                  k14_eytzinger_search)


def _subset(program, candidate_dict, pick):
    """``{name: make_sdfg}`` for the named subset -- build a fresh SDFG and apply the layout."""
    out = {}
    for name in pick:
        apply = candidate_dict[name]

        def make(apply=apply):
            sdfg = program.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        out[name] = make
    return out


def _assert_all_correct(candidates, run, reference):
    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None
    return results


def test_k01_stream_triad_block():
    inp = k01_stream_triad.make_inputs(64)
    cand_dict = k01_stream_triad.candidates()
    cands = _subset(k01_stream_triad.triad, cand_dict, list(cand_dict))
    _assert_all_correct(cands, k01_stream_triad.run_closure(inp, 64),
                        k01_stream_triad.oracle(inp["b"], inp["c"]))


def test_k02_transpose_block_and_permute():
    inp = k02_transpose_blocked.make_inputs(32)
    cands = _subset(k02_transpose_blocked.transpose, k02_transpose_blocked.candidates(),
                    ["noblock_A", "block_A_d0_8", "block_A_d1_8", "permute_A_10"])
    _assert_all_correct(cands, k02_transpose_blocked.run_closure(inp, 32), k02_transpose_blocked.oracle(inp["A"]))


def test_k03_complex_conjugate_zip():
    inp = k03_complex_conjugate_zip.make_inputs(32)
    cands = _subset(k03_complex_conjugate_zip.conjugate, k03_complex_conjugate_zip.candidates(),
                    ["unzipped_soa", "zipped_aos", "aosoa_4", "aosoa_8"])
    _assert_all_correct(cands, k03_complex_conjugate_zip.run_closure(inp, 32),
                        k03_complex_conjugate_zip.oracle(inp["re"], inp["im"]))


def test_k06_gather_accumulate_scatter_shuffle():
    inp = k06_gather_accumulate_scatter.make_inputs(16)
    cands = _subset(k06_gather_accumulate_scatter.gather_accumulate_scatter, k06_gather_accumulate_scatter.candidates(),
                    ["noshuffle_x", "shuffle_x_gas_xor", "shuffle_x_gas_cyc"])
    _assert_all_correct(cands, k06_gather_accumulate_scatter.run_closure(inp, 16),
                        k06_gather_accumulate_scatter.oracle(inp["x"], inp["y"]))


def test_k07_icon_stencil_permute():
    inp = k07_icon_stencil.make_inputs(6, 6, 6)
    cands = _subset(k07_icon_stencil.stencil, k07_icon_stencil.candidates(),
                    ["permute_A_012", "permute_A_210", "permute_A_120"])
    _assert_all_correct(cands, k07_icon_stencil.run_closure(inp, 6, 6, 6), k07_icon_stencil.oracle(inp["A"]))


def test_k09_particle_field_affinity_aosoa():
    inp = k09_particle_field_affinity.make_inputs(32)
    cands = _subset(k09_particle_field_affinity.particle_step, k09_particle_field_affinity.candidates(),
                    ["soa", "aos", "aosoa_4", "aosoa_8"])
    fields = ["x", "y", "z", "vx", "vy", "vz", "q", "m"]
    _assert_all_correct(cands, k09_particle_field_affinity.run_closure(inp, 32),
                        k09_particle_field_affinity.oracle(*[inp[f] for f in fields]))


def test_k10_omen_windowed_contraction_permute():
    inp = k10_omen_windowed_contraction.make_inputs(3, 8)
    cand_dict = k10_omen_windowed_contraction.candidates()
    cands = _subset(k10_omen_windowed_contraction.omen, cand_dict, list(cand_dict))
    _assert_all_correct(cands, k10_omen_windowed_contraction.run_closure(inp, 3, 8),
                        k10_omen_windowed_contraction.oracle(inp["H"], inp["X"], inp["D"]))


def test_k13_conv1x1_channel_layouts():
    inp = k13_conv1x1_channel_layouts.make_inputs(2, 16, 4, 4, 8)
    cand_dict = k13_conv1x1_channel_layouts.candidates()
    pick = [k for k in ("permute_X_0123", "permute_X_0231", "block_X_nChw16c") if k in cand_dict]
    cands = _subset(k13_conv1x1_channel_layouts.conv1x1, cand_dict, pick)
    _assert_all_correct(cands, k13_conv1x1_channel_layouts.run_closure(inp, 2, 16, 4, 4, 8),
                        k13_conv1x1_channel_layouts.oracle(inp["X"], inp["Wt"]))


def test_k14_eytzinger_search_shuffle():
    inp = k14_eytzinger_search.make_inputs(15)
    cand_dict = k14_eytzinger_search.candidates()
    cands = _subset(k14_eytzinger_search.eytzinger_search, cand_dict, list(cand_dict))
    _assert_all_correct(cands, k14_eytzinger_search.run_closure(inp, 15), k14_eytzinger_search.oracle(inp["A"]))


if __name__ == "__main__":
    test_k01_stream_triad_block()
    test_k02_transpose_block_and_permute()
    test_k03_complex_conjugate_zip()
    test_k06_gather_accumulate_scatter_shuffle()
    test_k07_icon_stencil_permute()
    test_k09_particle_field_affinity_aosoa()
    test_k10_omen_windowed_contraction_permute()
    test_k13_conv1x1_channel_layouts()
    test_k14_eytzinger_search_shuffle()
    print("kernel ports tests PASS")
