# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Masked-write lowering: a frontend boolean-masked assignment ``A[mask] = value``
lowers (``newast.py``) to a bare-if tasklet ``if __in_cond: __out = value``.

``NormalizeMaskedWriteTasklets`` rewrites that (in the tiled bodies only) to the
first-class write-only conditional-write function ``__out = IT(__in_cond, value)``
-- write ``value`` where the condition holds, else leave the destination unchanged,
with NO old-value read (unlike ``ITE(c, t, e)``). ``ConvertTaskletsToTileOps`` then
lowers ``IT`` to a masked ``TileStore`` (the ``cond`` gates the store; inactive lanes
are left untouched). This test pins both the structural rewrite and the end-to-end
numerics against NumPy.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import ast

import numpy as np
import pytest

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes as nd
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import BranchMode
from dace.transformation.passes.vectorization.normalize_masked_write_tasklets import NormalizeMaskedWriteTasklets
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

from tests.passes.vectorization.tile_assertions import assert_tiled

N = dace.symbol('N')
#: The host's best runnable SIMD ISA; vectorization enforces arch-native, so a hardcoded AVX-512
#: would SIGILL-refuse on an AVX2-only or ARM host.
HOST_ISA = detect_host_isa()


def _bare_if_tasklets(sdfg):
    hits = []
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nd.Tasklet) and n.code.language == dace.dtypes.Language.Python:
            try:
                body = ast.parse(n.code.as_string).body
            except (SyntaxError, ValueError):
                continue
            if any(isinstance(s, ast.If) for s in body):
                hits.append(n)
    return hits


@dace.program
def masked_zero(A: dace.float64[N], thresh: dace.float64):
    A[A > thresh] = 0.0


@dace.program
def masked_val(A: dace.float64[N], x: dace.float64[N], m: dace.bool[N]):
    A[m] = x


def _base(prog):
    sdfg = prog.to_sdfg(simplify=False)
    sdfg.simplify(validate=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify(validate=True)
    return sdfg


def test_masked_write_bare_if_becomes_it():
    """The frontend bare-if masked-write tasklet is rewritten to the write-only
    ``IT(cond, value)`` form -- no bare-if survives and NO ``*_old`` self-read
    connector is added (``IT`` never reads the destination's prior value)."""
    sdfg = masked_zero.to_sdfg(simplify=True)
    assert len(_bare_if_tasklets(sdfg)) >= 1, "test setup: expected a frontend bare-if tasklet"
    n = NormalizeMaskedWriteTasklets().apply_pass(sdfg, {})
    assert n and n >= 1
    survivors = _bare_if_tasklets(sdfg)
    assert not survivors, f"bare-if tasklets survived: {[t.code.as_string for t in survivors]}"
    it_tasklets = [
        t for t, _ in sdfg.all_nodes_recursive()
        if isinstance(t, nd.Tasklet) and t.code.language == dace.dtypes.Language.Python and "IT(" in t.code.as_string
    ]
    assert it_tasklets, "expected an ``IT(...)`` conditional-write tasklet"
    for t in it_tasklets:
        # Write-only: the IT rewrite must NOT introduce an old-value self-read connector.
        assert not any(c.endswith("_old") for c in t.in_connectors), \
            f"IT tasklet must not read the destination's old value: {sorted(t.in_connectors)}"
        assert t.code.as_string.strip().startswith(next(iter(t.out_connectors)) + " = IT(")
    sdfg.validate()


def test_normalize_skips_scalar_tail():
    """Normalize (run standalone) rewrites every masked write to ``IT`` -- but inside
    the full pipeline it skips scalar-tail scopes, which keep the bare-if. Here, with no
    tail markers present, all masked writes are rewritten (structural precondition)."""
    sdfg = masked_zero.to_sdfg(simplify=True)
    NormalizeMaskedWriteTasklets().apply_pass(sdfg, {})
    assert not _bare_if_tasklets(sdfg)


@pytest.mark.parametrize("isa", ["SCALAR", HOST_ISA])
@pytest.mark.parametrize("remainder", ["scalar_postamble", "masked_tail"])
def test_masked_const_write_matches_numpy(isa, remainder):
    """``A[A > thresh] = 0`` lowers through the tile pipeline (interior masked store +
    scalar/masked tail) bit-exact vs NumPy, at a non-tile-divisible size."""
    sdfg = _base(masked_zero)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8, ),
                        target_isa=isa,
                        remainder_strategy=remainder,
                        branch_mode=BranchMode.MERGE,
                        validate_all=True)).apply_pass(sdfg, {})
    assert_tiled(sdfg, _base(masked_zero))
    rng = np.random.default_rng(0)
    Nval = 37
    A = rng.random(Nval)
    thresh = 0.5
    ref = A.copy()
    ref[ref > thresh] = 0.0
    work = A.copy()
    sdfg(A=work, thresh=thresh, N=Nval)
    assert np.array_equal(work, ref), f"{work[:6]} != {ref[:6]}"


@pytest.mark.parametrize("isa", ["SCALAR", HOST_ISA])
@pytest.mark.parametrize("remainder", ["scalar_postamble", "masked_tail"])
def test_masked_value_write_matches_numpy(isa, remainder):
    """``A[m] = x`` (value tile, not a constant) lowers bit-exact vs NumPy. The
    ``masked_tail`` config exercises the AND-combine of ``cond`` with the tile
    iteration mask on the remainder store."""
    sdfg = _base(masked_val)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8, ),
                        target_isa=isa,
                        remainder_strategy=remainder,
                        branch_mode=BranchMode.MERGE,
                        validate_all=True)).apply_pass(sdfg, {})
    assert_tiled(sdfg, _base(masked_val))
    rng = np.random.default_rng(1)
    Nval = 37
    A = rng.random(Nval)
    x = rng.random(Nval)
    m = rng.random(Nval) > 0.5
    ref = A.copy()
    ref[m] = x[m]
    work = A.copy()
    sdfg(A=work, x=x.copy(), m=m.copy(), N=Nval)
    assert np.array_equal(work, ref), f"{work[:6]} != {ref[:6]}"


@dace.program
def overwrite_then_accumulate(total: dace.float64[N], cust: dace.float64[N], acc: dace.float64[N], eps: dace.float64):
    for jl in range(N):
        if total[jl] < eps:
            cust[jl] = 0.0
        acc[jl] = acc[jl] + cust[jl]


def canonical_overwrite_then_accumulate() -> dace.SDFG:
    sdfg = overwrite_then_accumulate.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    return sdfg


def test_a_masked_overwrite_read_again_keeps_the_old_value_on_unwritten_lanes():
    """CloudSC's ``if zlfinalsum < zepsec: zacust = 0; zsolac = zsolac + zacust``: the accumulate reads the
    overwritten element, so the lanes the condition leaves unwritten must still see the old value."""
    sdfg = canonical_overwrite_then_accumulate()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=HOST_ISA, validate=True)).apply_pass(sdfg, {})
    assert_tiled(sdfg, canonical_overwrite_then_accumulate())
    rng = np.random.default_rng(4)
    Nval = 37
    total, cust, acc = rng.random(Nval), rng.random(Nval), rng.random(Nval)
    expected_cust = np.where(total < 0.5, 0.0, cust)
    expected_acc = acc + expected_cust
    sdfg(total=total, cust=cust, acc=acc, eps=0.5, N=Nval)
    assert np.array_equal(cust, expected_cust), f"{cust[:6]} != {expected_cust[:6]}"
    assert np.array_equal(acc, expected_acc), f"{acc[:6]} != {expected_acc[:6]}"


def blend_over_an_element_the_body_already_updated() -> dace.SDFG:
    """``out[i] = out[i] + 2 x[i]`` into an access node, then ``out[i] = ITE(c[i] < -0.5, 3 x[i], <that node>)``."""
    sdfg = dace.SDFG('blend_over_an_updated_element')
    for name in ('c', 'x', 'out'):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:N'))
    x = state.add_read('x')
    first = state.add_tasklet('first', {'a', 'o'}, {'r'}, 'r = o + a * 2.0')
    state.add_memlet_path(x, me, first, dst_conn='a', memlet=dace.Memlet('x[i]'))
    state.add_memlet_path(state.add_read('out'), me, first, dst_conn='o', memlet=dace.Memlet('out[i]'))
    updated = state.add_access('out')
    state.add_edge(first, 'r', updated, None, dace.Memlet('out[i]'))
    blend = state.add_tasklet('blend', {'_c', '_t', '_e'}, {'_o'}, '_o = ITE(_c, _t, _e)')
    state.add_memlet_path(state.add_read('c'), me, blend, dst_conn='_c', memlet=dace.Memlet('c[i]'))
    state.add_memlet_path(x, me, blend, dst_conn='_t', memlet=dace.Memlet('x[i]'))
    state.add_edge(updated, None, blend, '_e', dace.Memlet('out[i]'))
    state.add_memlet_path(blend, mx, state.add_write('out'), src_conn='_o', memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def test_a_blend_over_an_element_the_body_already_updated_stays_a_blend():
    """Two guarded updates of ``out[i]`` in one map body: demoting the second to ``IT(cond, value)`` leaves the
    unwritten lanes at the value in memory, which a tiled body has not stored the first update into, so the
    first update vanished on every lane the second condition left unwritten."""
    sdfg = blend_over_an_element_the_body_already_updated()
    NormalizeMaskedWriteTasklets().apply_pass(sdfg, {})
    sdfg.validate()

    blend = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.Tasklet) and n.label == 'blend')
    assert 'ITE(' in blend.code.as_string, blend.code.as_string
    state = next(s for s in sdfg.all_states() if blend in s.nodes())
    assert [e.dst_conn for e in state.in_edges(blend) if e.dst_conn == '_e'] == ['_e']


def blend_on_a_symbol_condition() -> dace.SDFG:
    """``out[i] = ITE(flag, x[i], out[i])`` with ``flag`` a bool symbol, CloudSC's ``llfall_index_2_0`` select."""
    sdfg = dace.SDFG('blend_on_a_symbol_condition')
    for name in ('x', 'out'):
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_symbol('flag', dace.bool_)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:N'))
    blend = state.add_tasklet('blend', {'_t', '_e'}, {'_o'}, '_o = ITE(flag, _t, _e)')
    state.add_memlet_path(state.add_read('x'), me, blend, dst_conn='_t', memlet=dace.Memlet('x[i]'))
    state.add_memlet_path(state.add_read('out'), me, blend, dst_conn='_e', memlet=dace.Memlet('out[i]'))
    state.add_memlet_path(blend, mx, state.add_write('out'), src_conn='_o', memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def test_a_blend_on_a_symbol_condition_stays_a_blend():
    """Demoting ``ITE(flag, value, out)`` to ``IT(flag, value)`` leaves a masked write with no condition connector to
    gate the store on, so the tile converter refused the whole SDFG (CloudSC vectorized to zero tile nodes)."""
    sdfg = blend_on_a_symbol_condition()
    NormalizeMaskedWriteTasklets().apply_pass(sdfg, {})
    sdfg.validate()

    blend = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.Tasklet) and n.label == 'blend')
    assert blend.code.as_string.strip() == '_o = ITE(flag, _t, _e)', blend.code.as_string


@pytest.mark.parametrize('flag', [False, True])
def test_a_blend_on_a_symbol_condition_tiles_and_matches_numpy(flag):
    """The symbol-condition blend tiles (``TileITE`` with an inline predicate) and keeps the NumPy result."""
    sdfg = blend_on_a_symbol_condition()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=HOST_ISA, validate=True)).apply_pass(sdfg, {})
    assert_tiled(sdfg, blend_on_a_symbol_condition())
    rng = np.random.default_rng(5)
    Nval = 37
    x, out = rng.random(Nval), rng.random(Nval)
    expected = x.copy() if flag else out.copy()
    sdfg(x=x, out=out, flag=flag, N=Nval)
    assert np.array_equal(out, expected), f"{out[:6]} != {expected[:6]}"


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
