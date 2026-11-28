# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An if/else-if chain whose arms all store one element is one select (ICON's CFL clip of ``z_w_con_c``):
``z[c] = A if c1 else (B if c2 else z[c])``, the old value kept when no arm runs."""

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.select_same_element_stores import SelectSameElementStores
from tests.codegen.cpf.conftest import assert_matches

NC = dace.symbol("NC")


@dace.program
def clip_chain(z: dace.float64[NC], dd: dace.float64[NC], dtime: dace.float64):
    for c in range(NC):
        vcfl = z[c] * dtime / dd[c]
        if vcfl < -0.85:
            z[c] = -0.85 * dd[c] / dtime
        elif vcfl > 0.85:
            z[c] = 0.85 * dd[c] / dtime


def test_an_if_else_if_chain_storing_one_element_is_one_select():
    sdfg = clip_chain.to_sdfg(simplify=True)
    sdfg.name = "guarded_clip_chain"
    assert SelectSameElementStores().apply_pass(sdfg, {}) == 1
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks()), sdfg
    rng = np.random.default_rng(2)
    z, dd = 3.0 * rng.normal(size=64), rng.random(64) + 0.5
    got = z.copy()
    sdfg(z=got, dd=dd, dtime=2.0, NC=64)
    vcfl = z * 2.0 / dd
    want = np.where(vcfl < -0.85, -0.85 * dd / 2.0, np.where(vcfl > 0.85, 0.85 * dd / 2.0, z))
    assert_matches({"z": want}, {"z": got})


@dace.program
def guarded_store(z: dace.float64[NC], g: dace.bool_[NC]):
    for c in range(NC):
        if g[c]:
            z[c] = 1.0


def test_a_lone_guarded_store_is_left_to_the_masked_lifts():
    sdfg = guarded_store.to_sdfg(simplify=True)
    assert SelectSameElementStores().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_an_if_else_if_chain_storing_one_element_is_one_select()
    test_a_lone_guarded_store_is_left_to_the_masked_lifts()
