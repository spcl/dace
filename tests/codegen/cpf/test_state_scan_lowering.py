# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A host ``StateScan`` renders through its parallel ``CPU`` expansion, not the sequential ``pure`` one.

Both expansions are the same SDFG: blocks summarize themselves, one thread composes the summaries, the
blocks rerun from their start states. ``CPU`` puts the two block maps on ``CPU_Multicore``; ``pure`` pins
them ``Sequential``. Rendered ``pure``, the unit does the scan's extra work -- ``(count + 1) * N`` steps for
a finite scan, ``2 * N`` for a GF(2) one -- on one thread, slower than the loop the scan replaced. The
``CPU`` expansion reaches no runtime symbol (maps, nested SDFGs and plain tasklets), so the standalone unit
needs nothing past OpenMP.
"""

import re

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from dace.libraries.standard.nodes.state_scan import StateScan
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N, NS, NA = (dace.symbol(name, dace.int64) for name in ("N", "NS", "NA"))


@dace.program
def dfa(trans: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    state = 0
    for i in range(N):
        state = trans[state, symbols[i]]
        counts[state] += 1


@dace.program
def crc16(data: dace.uint8[N], crc: dace.int64[1], polynomial: dace.int64):
    c = 0xFFFF
    for b in range(N):
        cur_byte = 0xFF & data[b]
        for bit in range(0, 8):
            if (c & 0x0001) ^ (cur_byte & 0x0001):
                c = (c >> 1) ^ polynomial
            else:
                c = c >> 1
            cur_byte = cur_byte >> 1
    crc[0] = (c ^ 0xFFFF) & 0xFFFF


def reference_dfa(trans: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    counts = np.zeros(trans.shape[0], np.int64)
    state = 0
    for symbol in symbols:
        state = trans[state, symbol]
        counts[state] += 1
    return counts


def reference_crc16(data: np.ndarray, polynomial: int) -> int:
    c = 0xFFFF
    for byte in data:
        cur = int(byte)
        for _ in range(8):
            c = (c >> 1) ^ polynomial if (c ^ cur) & 1 else c >> 1
            cur >>= 1
    return (c ^ 0xFFFF) & 0xFFFF


def canonical(program, name: str) -> dace.SDFG:
    """``program`` through the CPU canonicalize pipeline and its tail, with exactly one lifted ``StateScan``."""
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    scans = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, StateScan)]
    assert len(scans) == 1, "the fixture is meant to lift its loop to one StateScan"
    sdfg.name = name
    return sdfg


def parallel_block_loops(code: str) -> int:
    """How many loops over the block index ``__c`` an OpenMP worksharing pragma opens."""
    return len(re.findall(r"#pragma omp parallel for[^\n]*\n\s*for \(int64_t __c = 0;", code))


def test_a_host_state_scan_is_offered_its_cpu_expansion_first():
    sdfg = canonical(dfa, "cpf_state_scan_offer")
    node, state = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, StateScan))

    offered = [name for name in cpf.renderable_implementations(node, state) if name in StateScan.implementations]

    assert offered[0] == "CPU"


@pytest.mark.parametrize("language", ["c", "c++"])
def test_a_finite_state_scan_runs_its_block_maps_in_parallel_and_counts_what_the_loop_counted(language):
    sdfg = canonical(dfa, f"cpf_state_scan_dfa_{'c' if language == 'c' else 'cpp'}")
    rng = np.random.default_rng(23)
    trans = rng.integers(0, 16, (16, 4)).astype(np.int64)
    symbols = rng.integers(0, 4, 5000).astype(np.int64)
    counts = np.zeros(16, np.int64)

    rendering = cpf.render(sdfg, language=language)
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language=language),
        rendering.sdfg,
        {"trans": trans, "symbols": symbols, "counts": counts, "N": symbols.size, "NS": 16, "NA": 4},
    )

    # The block-table map and the block-states map; the compose between them is one thread.
    assert parallel_block_loops(rendering.code) == 2, rendering.code
    np.testing.assert_array_equal(counts, reference_dfa(trans, symbols))


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize("length", [0, 7, 5000])
def test_a_gf2_state_scan_runs_its_block_maps_in_parallel_and_equals_the_bytewise_crc(language, length):
    sdfg = canonical(crc16, f"cpf_state_scan_crc16_{'c' if language == 'c' else 'cpp'}_{length}")
    data = np.random.default_rng(length).integers(0, 256, length).astype(np.uint8)
    crc = np.zeros(1, np.int64)

    rendering = cpf.render(sdfg, language=language)
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language=language),
        rendering.sdfg,
        {"data": data, "crc": crc, "polynomial": 0x8408, "N": length},
    )

    # The block-constant map and the block-states map; the basis-step map runs over ``__j``.
    assert parallel_block_loops(rendering.code) == 2, rendering.code
    assert crc[0] == reference_crc16(data, 0x8408)


if __name__ == "__main__":
    test_a_host_state_scan_is_offered_its_cpu_expansion_first()
    for lang in ("c", "c++"):
        test_a_finite_state_scan_runs_its_block_maps_in_parallel_and_counts_what_the_loop_counted(lang)
        for size in (0, 7, 5000):
            test_a_gf2_state_scan_runs_its_block_maps_in_parallel_and_equals_the_bytewise_crc(lang, size)
