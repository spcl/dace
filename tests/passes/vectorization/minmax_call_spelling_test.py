# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Min`` / ``Max`` reach the converter capitalized, and the matchers have to know it.

sympy derives ``Min`` / ``Max`` from ``Application`` rather than ``Function``, so DaCe's symbolic
printer emits them under the runtime's own variadic ``Min`` / ``Max`` spelling. Every function-form
matcher in the tile converter built its candidate strings from the lowercase op label, so a body
like CloudSC's ``_out_zanew_2 = Min(1.0, __t0)`` matched nothing and stayed a scalar tasklet beside
widened operands -- which the orchestrator can only answer by refusing the whole SDFG.
"""
import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.libraries.tileops.nodes.tile_binop import TileBinop
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import (_FUNCTION_FORM_BINOPS,
                                                                                   _call_spellings)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = 64
WIDTHS = (8, )


def test_min_and_max_carry_their_capitalized_spelling():
    assert _call_spellings('min') == ('min', 'Min')
    assert _call_spellings('max') == ('max', 'Max')


def test_every_other_function_form_op_keeps_one_spelling():
    """Only Min/Max are printed capitalized; inventing aliases for the rest would match noise."""
    for op in _FUNCTION_FORM_BINOPS:
        if op in ('min', 'max'):
            continue
        assert _call_spellings(op) == (op, ), f'{op} gained an unexpected alias'


@pytest.mark.parametrize('isa', ['SCALAR', detect_host_isa()])
def test_a_clamped_kernel_vectorizes_and_matches_numpy(isa):
    """End-to-end on the shape CloudSC has: a literal operand against a per-lane value."""

    @dace.program
    def clamp(a: dace.float64[N], out: dace.float64[N]):
        for i in dace.map[0:N]:
            out[i] = min(1.0, a[i] * 2.0)

    sdfg = clamp.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=WIDTHS, target_isa=isa, validate=True)).apply_pass(sdfg, {})

    assert [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop) and n.op == 'min'], \
        'the clamp did not reach the tile pipeline as a min TileBinop'
    leftover = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet) and 'Min(' in (n.code.as_string or '')
    ]
    assert not leftover, f'a scalar Min tasklet survived: {[n.code.as_string for n in leftover]}'

    rng = np.random.default_rng(0)
    a = rng.standard_normal(N)
    out = np.zeros(N)
    sdfg(a=a, out=out)
    assert np.allclose(out, np.minimum(1.0, a * 2.0))
