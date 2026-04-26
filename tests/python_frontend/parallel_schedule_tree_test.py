# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest


def test_parallel_schedule_tree_statement_nodes_raise_on_to_sdfg():

    class AttrHolder:

        def __init__(self):
            self.arr = np.zeros(4, dtype=np.float64)

    attr_holder = AttrHolder()

    @dace.program
    def prog(A: dace.float64[4], out: dace.float64[4]):
        attr_holder.arr = A
        out[:] = attr_holder.arr

    with pytest.raises(RuntimeError, match=r'StatementNode'):
        prog.to_sdfg(simplify=False)


def test_parallel_schedule_tree_warns_for_refsets_and_pythonclasses():

    class Holder:
        scalar: dace.float64

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder, A: dace.float64[4], out: dace.float64[4]):
        holder.new_data = A
        out[:] = holder.new_data[:]

    with pytest.warns(UserWarning) as captured:
        sdfg = prog.to_sdfg(simplify=False)

    messages = [str(record.message) for record in captured]
    assert any('RefSetNode target "holder.new_data"' in message for message in messages)
    assert any('PythonClass container "holder"' in message for message in messages)
    assert any(
        isinstance(descriptor, dace.data.Reference)
        for _, _, descriptor in sdfg.arrays_recursive(include_nested_data=True))
