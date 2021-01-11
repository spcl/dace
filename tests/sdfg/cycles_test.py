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
