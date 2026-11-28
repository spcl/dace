# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A promoted integer division keeps C's truncation (vexx's block count ``(ngm + 255) / 256``)."""

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion


def block_count_sdfg(language: dace.Language) -> dace.SDFG:
    """``nb = (ngm + 255) / 256`` in a tasklet, then ``out[i - 1] = 1`` for ``i = 1 .. nb``."""
    sdfg = dace.SDFG("block_count_" + language.name)
    sdfg.add_scalar("ngm", dace.int64)
    sdfg.add_array("out", [8], dace.int64)
    sdfg.add_scalar("nb", dace.int64, transient=True)
    init = sdfg.add_state("init", is_start_block=True)
    code = "o = (a + 255) / 256" + (";" if language == dace.Language.CPP else "")
    tasklet = init.add_tasklet("count", {"a"}, {"o"}, code, language=language)
    init.add_edge(init.add_read("ngm"), None, tasklet, "a", dace.Memlet("ngm[0]"))
    init.add_edge(tasklet, "o", init.add_write("nb"), None, dace.Memlet("nb[0]"))
    loop = LoopRegion("blocks", "i <= nb", "i", "i = 1", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state("body", is_start_block=True)
    mark = body.add_tasklet("mark", {}, {"m"}, "m = 1")
    body.add_edge(mark, "m", body.add_write("out"), None, dace.Memlet("out[i - 1]"))
    return sdfg


@pytest.mark.parametrize("language", [dace.Language.Python, dace.Language.CPP])
def test_a_promoted_integer_division_truncates(language: dace.Language):
    sdfg = block_count_sdfg(language)
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    (assignment,) = [e.data.assignments["nb"] for e in sdfg.all_interstate_edges() if "nb" in e.data.assignments]
    count = symbolic.pystr_to_symbolic(assignment)
    ngm = symbolic.symbol("ngm")
    # C truncates toward zero: 511 / 256 == 1 and -45 / 256 == 0, never a fraction.
    assert [count.subs(ngm, value) for value in (256, 300, -300)] == [1, 2, 0]

    # The bound survives the loop becoming a map: 256 entries fill one block, not two.
    assert sdfg.apply_transformations_repeated(LoopToMap) == 1
    out = np.zeros(8, np.int64)
    sdfg(ngm=np.int64(256), out=out)
    assert out.sum() == 1


if __name__ == "__main__":
    test_a_promoted_integer_division_truncates(dace.Language.Python)
    test_a_promoted_integer_division_truncates(dace.Language.CPP)
