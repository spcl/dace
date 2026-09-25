# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`FuseChainedScalarReductions`."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import List, Optional, Tuple

import numpy as np
import pytest

import dace
from dace.sdfg import SDFGState, nodes
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import (FuseChainedScalarReductions,
                                                                                    _binop_op)

N = dace.symbol('N')


@dace.program
def s319(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N],
         out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        a[i] = c[i] + d[i]
        s = s + a[i]
        b[i] = c[i] + e[i]
        s = s + b[i]
    out[0] = s


def chain_state(sdfg: dace.SDFG) -> Optional[Tuple[SDFGState, List[nodes.Tasklet]]]:
    """The loop-body state holding the chained accumulations, with its foldable binops."""
    for state in sdfg.states():
        binops = [n for n in state.nodes() if isinstance(n, nodes.Tasklet) and _binop_op(n) is not None]
        if len(binops) >= 2:
            return state, binops
    return None


def reference(c: np.ndarray, d: np.ndarray, e: np.ndarray) -> float:
    return float((c + d).sum() + (c + e).sum())


def run(sdfg: dace.SDFG, n: int) -> float:
    rng = np.random.default_rng(319)
    c, d, e = rng.random(n), rng.random(n), rng.random(n)
    out = np.zeros(1)
    sdfg(a=np.zeros(n), b=np.zeros(n), c=c, d=d, e=e, out=out, N=n)
    assert np.allclose(out[0], reference(c, d, e)), f'got {out[0]}, want {reference(c, d, e)}'
    return out[0]


def test_chained_accumulations_fold():
    sdfg = s319.to_sdfg(simplify=True)
    assert chain_state(sdfg) is not None, 'fixture must produce the chained-accumulator shape'
    assert FuseChainedScalarReductions().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    run(sdfg, 24)


def nest_inner_sdfg(inner_sdfg: dace.SDFG) -> dace.SDFG:
    """Wrap inner_sdfg as a NestedSDFG in a fresh outer host (seissol_tensor_contraction shape)."""
    outer_sdfg = dace.SDFG('outer_scalar_reduction_host')
    for name, desc in inner_sdfg.arrays.items():
        outer_sdfg.add_datadesc(name, desc.clone())

    outer_state = outer_sdfg.add_state('call_inner', is_start_block=True)
    non_transient = [name for name, desc in inner_sdfg.arrays.items() if not desc.transient]
    read_write = [name for name in non_transient if name in ('a', 'b')]
    inputs = dict.fromkeys(name for name in non_transient if name != 'out')
    outputs = dict.fromkeys(name for name in non_transient if name in ('a', 'b', 'out'))
    nsdfg_node = outer_state.add_nested_sdfg(inner_sdfg, inputs, outputs, symbol_mapping={'N': N})

    for name in non_transient:
        if name in read_write:
            outer_state.add_edge(outer_state.add_access(name), None, nsdfg_node, name,
                                 dace.Memlet.from_array(name, outer_sdfg.arrays[name]))
            outer_state.add_edge(nsdfg_node, name, outer_state.add_access(name), None,
                                 dace.Memlet.from_array(name, outer_sdfg.arrays[name]))
        elif name == 'out':
            outer_state.add_edge(nsdfg_node, name, outer_state.add_access(name), None,
                                 dace.Memlet.from_array(name, outer_sdfg.arrays[name]))
        else:
            outer_state.add_edge(outer_state.add_access(name), None, nsdfg_node, name,
                                 dace.Memlet.from_array(name, outer_sdfg.arrays[name]))
    return outer_sdfg


def test_fold_inside_nested_sdfg_scopes_descriptor_to_the_inner_sdfg():
    """The fold inside a NestedSDFG must register its new descriptor on the inner SDFG, not the outer host."""
    inner_sdfg = s319.to_sdfg(simplify=True)
    found = chain_state(inner_sdfg)
    assert found is not None, 'fixture must produce the chained-accumulator shape'
    inner_state, _ = found

    outer_sdfg = nest_inner_sdfg(inner_sdfg)
    outer_sdfg.validate()

    fused = FuseChainedScalarReductions().apply_pass(outer_sdfg, {})
    assert fused == 1, 'exactly one chain must fold, inside the nested SDFG state'
    outer_sdfg.validate()

    fused_nodes = [n for n in inner_state.data_nodes() if n.data.startswith('_fused_inc')]
    assert len(fused_nodes) == 1, 'the fold must run exactly once on the nested loop body'
    fused_name = fused_nodes[0].data
    assert fused_name in inner_sdfg.arrays, f'{fused_name} must be declared on the SDFG owning its access node'
    assert fused_name not in outer_sdfg.arrays, 'the fold must not register its descriptor on the outer host SDFG'

    run(outer_sdfg, 24)


def test_ordering_memlet_on_a_chain_node_refuses_the_fold():
    """The fold deletes the downstream chain nodes; an ordering memlet on one of them would go
    with it, so the fold must decline."""
    sdfg = s319.to_sdfg(simplify=True)
    found = chain_state(sdfg)
    assert found is not None
    state, binops = found
    anchor = next(iter(state.data_nodes()))
    state.add_edge(anchor, None, binops[-1], None, dace.Memlet())
    sdfg.validate()

    assert FuseChainedScalarReductions().apply_pass(sdfg, {}) is None, 'the fold must decline'
    assert any(e.data.is_empty() for e in state.edges()), 'the ordering memlet must survive'
    sdfg.validate()
    run(sdfg, 24)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
