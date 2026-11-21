# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every automata corpus kernel is lifted to exactly one ``StateScan`` of its kind, and nothing it carried is left
as a sequential loop. ``canonicalize_numerical_corpus_test`` checks the values; this checks the structure."""

import numpy as np
import pytest

from dace.libraries.standard.nodes.state_scan import StateScan, StateScanKind
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.automata import automata

KINDS = {
    "dfa": StateScanKind.FINITE,
    "kmp": StateScanKind.FINITE,
    "crc16": StateScanKind.GF2_AFFINE,
    "huffman_decode": StateScanKind.FINITE,
    "aho_corasick": StateScanKind.FINITE,
    "utf8_validate": StateScanKind.FINITE,
    "csv_tokenize": StateScanKind.FINITE,
}


def test_every_automata_kernel_has_a_kind():
    assert sorted(KINDS) == sorted(automata.collect())


@pytest.mark.parametrize("name", automata.collect())
def test_an_automata_kernel_becomes_one_state_scan_and_no_top_level_loop(name):
    sdfg = automata.PROGRAMS[name].to_sdfg(simplify=True)

    canonicalize(sdfg, validate=True)

    scans = [node for node, parent in sdfg.all_nodes_recursive() if isinstance(node, StateScan)]
    assert [node.kind for node in scans] == [KINDS[name]]
    assert not any(isinstance(block, LoopRegion) for block in sdfg.all_control_flow_regions())


@pytest.mark.parametrize("name", automata.collect())
def test_an_automata_kernel_input_drives_its_outputs_off_zero(name):
    """An input that never reaches an accepting state would pass any scan; every output must move."""
    made = automata.inputs(name, 1234)
    outputs = {key: value.copy() for key, value in made.arrays.items()}

    made.reference(**outputs, **made.scalars)

    written = [key for key in outputs if not np.array_equal(outputs[key], made.arrays[key])]
    assert written and all(np.count_nonzero(outputs[key]) > 0 for key in written)
    assert all(isinstance(value, (int, np.integer)) for value in made.sizes.values())
