# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace


def test_cycles():
    with pytest.raises(ValueError, match="Found cycles.*"):
        sdfg = dace.SDFG("hello")
        state = sdfg.add_state()

        sdfg.add_array("A", shape=(1, ), dtype=dace.float32)
        access = state.add_access("A")

        state.add_edge(access, None, access, None, dace.Memlet.simple("A", "0"))
        sdfg.validate()


def test_cycles_memlet_path():
    with pytest.raises(ValueError, match="Found cycles.*"):
        sdfg = dace.SDFG("foo")
        state = sdfg.add_state()
        sdfg.add_array("bla", shape=(10, ), dtype=dace.float32)
        mentry_3, _ = state.add_map("map_3", dict(i="0:9"))
        mentry_3.add_in_connector("IN_0")
        mentry_3.add_out_connector("OUT_0")
        state.add_edge(mentry_3, "OUT_0", mentry_3, "IN_0", dace.Memlet(data="bla", subset='0:9'))

        sdfg.validate()


if __name__ == '__main__':
    test_cycles()
    test_cycles_memlet_path()
