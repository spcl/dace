# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A symbol named like a sympy function parses as a symbol."""

import dace
from dace import symbolic


def test_a_symbol_named_poly_parses_and_serializes():
    """crc16's ``(c >> 1) ^ poly`` made ``poly`` sympy's function, so the edge did not serialize."""
    expr = symbolic.pystr_to_symbolic("(c >> 1) ^ poly")
    assert {str(s) for s in expr.free_symbols} == {"c", "poly"}
    edge = dace.InterstateEdge(assignments={"c": "(c >> 1) ^ poly"})
    assert "poly" in edge.to_json()["attributes"]["assignments"]["c"]
