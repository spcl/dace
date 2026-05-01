# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import math
import copy
from typing import Tuple
import dace
import pytest
import numpy
from dace import InterstateEdge
from dace import Union
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    ReplaceSTDExpWithDaCeExp, ReplaceSTDLogWithDaCeLog, ReplaceSTDPowWithDaCePow,
)
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    N, S, S1, S2, klev, kidia, kfdia, n, m, nnz,
    KLON, KLEV, NCLDQL, NCLDQI, ssym, X, Y, C,
    log, exp, pow,
    _get_disjoint_chain_sdfg, _get_disjoint_chain_sdfg_two,
    _get_cloudsc_snippet_three, _get_cloudsc_snippet_four,
    _get_map_inside_nested_map,
    _get_dependency_edge_to_unary_symbol_sdfg,
    _get_unstructured_access_cloudsc_sdfg,
)

def test_dependency_edge_to_unary_symbol():
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
        save_sdfgs=True,
        sdfg_name="dependency_edge_to_unary_symbol",
        from_sdfg=True,
        no_inline=True,
    )

