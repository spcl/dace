# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The lowering of every tile-op library node is what the committed snapshot says."""

import json

import pytest
from golden_lowering import DIGEST_FILE, ShardKey, digest, lowerings, shard_keys


@pytest.mark.parametrize("key", shard_keys(), ids=str)
def test_the_lowering_of_a_tile_node_is_unchanged(key: ShardKey):
    """A restructuring of the tile-op library must not change the tasklet, implementation or environment any
    configuration lowers to; ``python golden_lowering.py dump <file>`` before and after names the one that moved."""
    with open(DIGEST_FILE) as digests:
        expected = json.load(digests)[str(key)]
    assert digest(lowerings(key)) == expected


if __name__ == "__main__":
    for shard_key in shard_keys():
        test_the_lowering_of_a_tile_node_is_unchanged(shard_key)
