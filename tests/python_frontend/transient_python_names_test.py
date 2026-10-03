# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transients take their Python names at the end of parsing, in one walk of the SDFG.

A reshape view is created under a generated name and renamed to the Python name it is bound to, so a
chain of views is the program shape that exercises the renaming.
"""
import numpy as np

import dace
from dace import SDFG

N = dace.symbol('N')


@dace.program
def three_reshaped_views(x: dace.float64[N], out: dace.float64[N]):
    b0 = np.copy(x)
    b1 = b0.reshape((N, ))
    b2 = b1.reshape((N, ))
    b3 = b2.reshape((N, ))
    out[:] = b3


@dace.program
def twelve_reshaped_views(x: dace.float64[N], out: dace.float64[N]):
    b0 = np.copy(x)
    b1 = b0.reshape((N, ))
    b2 = b1.reshape((N, ))
    b3 = b2.reshape((N, ))
    b4 = b3.reshape((N, ))
    b5 = b4.reshape((N, ))
    b6 = b5.reshape((N, ))
    b7 = b6.reshape((N, ))
    b8 = b7.reshape((N, ))
    b9 = b8.reshape((N, ))
    b10 = b9.reshape((N, ))
    b11 = b10.reshape((N, ))
    b12 = b11.reshape((N, ))
    out[:] = b12


def test_reshaped_views_carry_their_python_names_after_parsing():
    """A view bound to a Python name is renamed to that name; codegen and the CPF form read it."""
    sdfg = three_reshaped_views.to_sdfg(simplify=False)
    assert {'b1', 'b2', 'b3'} <= set(sdfg.arrays), sorted(sdfg.arrays)


def test_naming_transients_walks_the_sdfg_once_however_many_there_are(monkeypatch):
    """Each SDFG.replace_dict walks every node, edge and descriptor. One walk per named transient was
    40% of sw4_rhs4sg's parse, so the number of walks must not grow with the number of names."""
    calls = []
    original = SDFG.replace_dict

    def counting(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SDFG, 'replace_dict', counting)
    three_reshaped_views.to_sdfg(simplify=False)
    few = len(calls)
    calls.clear()
    twelve_reshaped_views.to_sdfg(simplify=False)
    many = len(calls)
    assert many == few, (few, many)


def test_the_renamed_view_chain_computes_what_numpy_computes():
    x = np.arange(8, dtype=np.float64)
    out = np.zeros(8)
    twelve_reshaped_views(x, out, N=8)
    assert np.array_equal(out, x), out


if __name__ == '__main__':
    test_reshaped_views_carry_their_python_names_after_parsing()
    test_the_renamed_view_chain_computes_what_numpy_computes()
