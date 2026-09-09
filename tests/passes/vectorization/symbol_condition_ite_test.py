# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An ITE whose CONDITION is a symbol is a uniform select, and ``TileITE`` already renders one.

``TileITE`` has carried ``kind_mask='Symbol'`` -- predicate embedded inline, no ``_mask``
connector -- since the arm-inlining work, but ``_detect_ite`` required the condition to be an
in-connector and returned ``None`` otherwise. CloudSC reads a Fortran ``LOGICAL`` at a constant
index into an interstate symbol (``llfall_index_2_0 = llfall[0]``), so its arm select comes out as
``ITE(llfall_index_2_0, _new, _old)`` -- five of them, each left a scalar tasklet beside widened
arms, which the orchestrator can only answer by refusing the whole SDFG.

Inlining is only sound for a LOOP-INVARIANT predicate: a per-lane one would splat lane 0's answer
across the tile, so both the spelled-out iter_var case and the one an interstate assignment hides
are refused instead.
"""
import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.libraries.tileops.nodes.tile_ite import TileITE
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import ConvertTaskletsToTileOps
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = 64
WIDTHS = (8, )


def symbol_cond_tasklet() -> nodes.Tasklet:
    return nodes.Tasklet('sel', {'_t', '_e'}, {'_o'}, '_o = ITE(flag_sym, _t, _e)')


def test_a_symbol_condition_is_reported_rather_than_declined():
    detected = ConvertTaskletsToTileOps(widths=WIDTHS)._detect_ite(symbol_cond_tasklet())
    assert detected is not None, 'a symbol-conditioned ITE was not recognised at all'
    out_conn, cond, t, e, t_sym, e_sym, cond_sym = detected
    assert (out_conn, cond, t, e) == ('_o', 'flag_sym', '_t', '_e')
    assert cond_sym is True, 'the condition was reported as a connector'
    assert (t_sym, e_sym) == (False, False), 'both arms ARE connectors here'


def test_a_connector_condition_still_reports_as_one():
    """The control: the ordinary shape must keep its old classification."""
    tasklet = nodes.Tasklet('sel', {'_c', '_t', '_e'}, {'_o'}, '_o = ITE(_c, _t, _e)')
    detected = ConvertTaskletsToTileOps(widths=WIDTHS)._detect_ite(tasklet)
    assert detected is not None
    assert detected[6] is False, 'a connector condition was reported as a symbol'


@pytest.mark.parametrize('isa', ['SCALAR', detect_host_isa()])
def test_a_uniform_flag_select_vectorizes_and_matches_numpy(isa):
    """End-to-end on CloudSC's shape: a loop-invariant flag choosing between two per-lane values."""

    @dace.program
    def pick(flags: dace.int32[4], a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
        for i in dace.map[0:N]:
            out[i] = a[i] if flags[0] else b[i]

    sdfg = pick.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=WIDTHS, target_isa=isa, validate=True)).apply_pass(sdfg, {})

    ites = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileITE)]
    assert ites, 'the flag select did not reach the tile pipeline'

    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(N), rng.standard_normal(N)
    for flag in (0, 1):
        flags = np.array([flag, 0, 0, 0], dtype=np.int32)
        out = np.zeros(N)
        sdfg(flags=flags, a=a, b=b, out=out)
        assert np.array_equal(out, a if flag else b), f'flag={flag} selected the wrong arm'
