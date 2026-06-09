# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import numpy
from tests.passes.vectorization.helpers.harness import (
    run_vectorization_test,
    _get_dependency_edge_to_unary_symbol_sdfg,
)


def test_dependency_edge_to_unary_symbol(emission_style):
    sdfg = _get_dependency_edge_to_unary_symbol_sdfg()
    N = 64
    A = numpy.random.random((N, )).astype(numpy.float64)
    B = numpy.random.random((N, )).astype(numpy.float64)

    run_vectorization_test(
        dace_func=sdfg,
        arrays={
            'int_array': A,
            'int_array2': B
        },
        params={
            'klon': N,
        },
        vector_width=8,
        sdfg_name="dependency_edge_to_unary_symbol",
        from_sdfg=True,
        no_inline=True,
        emission_style=emission_style,
    )
