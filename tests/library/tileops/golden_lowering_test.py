# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The lowering of every tile-op library node is what the committed snapshot says."""
import json

import pytest

from golden_lowering import CASES, DIGEST_FILE, digest, lowerings


@pytest.mark.parametrize("node_type", sorted(CASES))
def test_the_lowering_of_a_tile_node_is_unchanged(node_type):
    """A restructuring of the tile-op library must not change the tasklet, implementation or environment any
    configuration lowers to; ``python golden_lowering.py dump <file>`` before and after names the one that moved."""
    with open(DIGEST_FILE) as digests:
        expected = json.load(digests)[node_type]
    assert digest(lowerings(node_type)) == expected


if __name__ == "__main__":
    for tile_node_type in sorted(CASES):
        test_the_lowering_of_a_tile_node_is_unchanged(tile_node_type)
