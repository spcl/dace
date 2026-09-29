# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that the frontend types map ranges and memlets with the declared symbol dtype. """
from typing import Dict, List

import numpy as np

import dace
from dace.sdfg import nodes

N = dace.symbol('N', dtype=dace.int64)
K = dace.symbol('K', dtype=dace.int64)


def dtypes_of(sdfg: dace.SDFG, name: str) -> List[dace.typeclass]:
    found: List[dace.typeclass] = []
    exprs = []
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    exprs.extend(bound for rng in node.map.range for bound in rng)
            for edge in state.edges():
                for subset in (edge.data.subset, edge.data.other_subset):
                    if subset is not None:
                        exprs.extend(bound for rng in subset.ndrange() for bound in rng)
    for expr in exprs:
        found.extend(sym.dtype for sym in getattr(expr, 'free_symbols', ()) if sym.name == name)
    return found


def assert_declared_dtype(sdfg: dace.SDFG, declared: Dict[str, dace.typeclass]):
    for name, dtype in declared.items():
        found = dtypes_of(sdfg, name)
        assert found, f'no occurrence of {name} in map ranges or memlets'
        assert all(d == dtype for d in found), f'{name}: {found} != {dtype}'


@dace.program
def map_and_elementwise(a: dace.float64[N], b: dace.float64[N]):
    for j in dace.map[0:N]:
        b[j] = a[j] + 1
    b[:] = a + b


@dace.program
def body_only_symbol(a: dace.float64[N], b: dace.float64[N]):
    for j in dace.map[0:K]:
        b[j] = a[j] + 1


@dace.program
def nested_callee(a: dace.float64[N], b: dace.float64[N]):
    for j in dace.map[0:N]:
        b[j] = a[j] * 2


@dace.program
def nested_caller(a: dace.float64[N], b: dace.float64[N]):
    nested_callee(a, b)
    b[:] = a + b


def test_map_and_elementwise_ranges_keep_declared_dtype():
    assert_declared_dtype(map_and_elementwise.to_sdfg(simplify=False), {'N': dace.int64})


def test_body_only_symbol_keeps_declared_dtype():
    assert_declared_dtype(body_only_symbol.to_sdfg(simplify=False), {'K': dace.int64})


def test_nested_program_call_keeps_declared_dtype():
    assert_declared_dtype(nested_caller.to_sdfg(simplify=False), {'N': dace.int64})


M = dace.symbol('M', dtype=dace.int64)


@dace.program
def copy_callee(a: dace.float64[M]):
    b = np.ndarray((M, ), dtype=np.float64)
    for j in dace.map[0:M]:
        b[j] = a[j]
    return b


@dace.program
def prefix_caller(a: dace.float64[N], out: dace.float64[N]):
    for k in range(1, N):
        out[:k] += copy_callee(a[:k])


def test_a_callee_symbol_mapped_to_a_loop_bound_is_renamed_everywhere():
    """The callee's ``M`` is renamed through a temporary; a memlet volume holding it must follow (durbin)."""

    @dace.program
    def outer(a: dace.float64[12], out: dace.float64[12]):
        prefix_caller(a, out)

    outer.to_sdfg(simplify=True).validate()


if __name__ == '__main__':
    test_map_and_elementwise_ranges_keep_declared_dtype()
    test_body_only_symbol_keeps_declared_dtype()
    test_nested_program_call_keeps_declared_dtype()
    test_a_callee_symbol_mapped_to_a_loop_bound_is_renamed_everywhere()
