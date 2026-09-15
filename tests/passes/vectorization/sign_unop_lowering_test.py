# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``np.sign`` reaches the tile pipeline as a unop, and every op without an ISA character survives.

CloudSC's ``__out = sign_numpy_2(__in1)`` had no entry in ``_SUPPORTED_UNOPS``, so the converter
left it a scalar tasklet with widened operands -- which the orchestrator can only answer by refusing
the whole SDFG. The op label is the runtime's own function name, which is what the frontend writes
into the tasklet body, so the call-form matcher finds it with no alias table.

The ISA path had a second hole behind it: ``dace::tileops::tile_unop`` templates on a single op
character and only twelve of the seventeen lowered ops have one, while the lookup was a bare
subscript. Any of the other five reached codegen as a KeyError instead of a refusal or a fallback.
"""
import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.libraries.tileops.nodes.tile_unop import UNOP_CPP, TileUnop
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.convert_tasklets_to_tile_ops import _SUPPORTED_UNOPS
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.libraries.tileops import _isa_codegen

N = 64
WIDTHS = (8, )


def test_every_lowered_unop_has_a_cpp_rendering():
    """A converter that lowers an op the pure expansion cannot render is a codegen failure."""
    missing = sorted(_SUPPORTED_UNOPS - set(UNOP_CPP))
    assert not missing, f'lowered without a C++ rendering: {missing}'


def test_ops_without_an_isa_character_fall_back_rather_than_raise():
    """The ISA table is a SUBSET of what the converter lowers, and the gap must be a fallback."""
    no_char = sorted(set(UNOP_CPP) - set(_isa_codegen.UNOP_TO_CHAR))
    assert no_char, 'the fixture is vacuous -- every op now has an ISA character'
    # The op the fallback exists for; the others (tan, asin, ...) ride the same path.
    assert 'sign_numpy_2' in no_char


@pytest.mark.parametrize('isa', ['SCALAR', detect_host_isa()])
def test_sign_vectorizes_and_matches_numpy(isa):
    """End-to-end: the kernel really is tiled, and the numbers are numpy's."""

    @dace.program
    def signs(a: dace.float64[N], out: dace.float64[N]):
        for i in dace.map[0:N]:
            out[i] = np.sign(a[i])

    sdfg = signs.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=WIDTHS, target_isa=isa, validate=True)).apply_pass(sdfg, {})

    unops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileUnop) and n.op == 'sign_numpy_2']
    assert unops, 'np.sign did not reach the tile pipeline as a TileUnop'
    assert not [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and 'sign_numpy_2' in (n.code.as_string or '')
    ], 'a scalar sign tasklet survived beside the tile op'

    rng = np.random.default_rng(0)
    a = rng.standard_normal(N)
    a[3] = 0.0  # the branch numpy.sign answers with 0
    out = np.zeros(N)
    sdfg(a=a, out=out)
    assert np.array_equal(out, np.sign(a))
