# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``StateScan`` expansion declares scratch containers next to the node's own operands, so the scratch names must
not be names a program can give its arrays. An operand called ``full`` used to raise
``FileExistsError: Data descriptor "full" already exists in SDFG`` at the first ``add_array``.
"""

import pytest

import dace
from dace.libraries.standard.nodes.state_scan import StateScan
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, NS, NA = (dace.symbol(name, dace.int64) for name in ("N", "NS", "NA"))


@dace.program
def crc16_full(full: dace.uint8[N], crc: dace.int64[1], polynomial: dace.int64):
    """A GF(2) scan whose data array has the name of a block power of the expansion."""
    c = 0xFFFF
    for b in range(N):
        cur_byte = 0xFF & full[b]
        for bit in range(0, 8):
            if (c & 0x0001) ^ (cur_byte & 0x0001):
                c = (c >> 1) ^ polynomial
            else:
                c = c >> 1
            cur_byte = cur_byte >> 1
    crc[0] = (c ^ 0xFFFF) & 0xFFFF


@dace.program
def crc16_raw(raw: dace.uint8[N], crc: dace.int64[1], polynomial: dace.int64):
    """A GF(2) scan whose data array has the name of the expansion's basis images."""
    c = 0xFFFF
    for b in range(N):
        cur_byte = 0xFF & raw[b]
        for bit in range(0, 8):
            if (c & 0x0001) ^ (cur_byte & 0x0001):
                c = (c >> 1) ^ polynomial
            else:
                c = c >> 1
            cur_byte = cur_byte >> 1
    crc[0] = (c ^ 0xFFFF) & 0xFFFF


@dace.program
def dfa_table(table: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    """A finite scan whose transition table has the name of the expansion's table of end states."""
    state = 0
    for i in range(N):
        state = table[state, symbols[i]]
        counts[state] += 1


def expand_pure(program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    scans = [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, StateScan)]
    assert len(scans) == 1, "the fixture is meant to lift its loop to one StateScan"
    scans[0].implementation = "pure"
    sdfg.expand_library_nodes()
    return sdfg


@pytest.mark.parametrize(("program", "operand"), [(crc16_full, "full"), (crc16_raw, "raw"), (dfa_table, "table")])
def test_a_scan_expands_when_an_operand_is_named_like_its_scratch(program, operand):
    sdfg = expand_pure(program)

    sdfg.validate()
    assert operand in sdfg.arrays


if __name__ == "__main__":
    for program, operand in [(crc16_full, "full"), (crc16_raw, "raw"), (dfa_table, "table")]:
        test_a_scan_expands_when_an_operand_is_named_like_its_scratch(program, operand)
