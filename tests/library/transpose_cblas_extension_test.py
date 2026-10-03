# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Transpose`` must decline OpenBLAS when the linked CBLAS has no ``?omatcopy``.

``cblas_domatcopy`` is an OpenBLAS/MKL extension, not part of CBLAS. Where the BLAS that gets
linked is a reference netlib build, every gemm in the graph resolves and this one symbol does not,
so the kernel configures, compiles, and dies at link time with "undefined reference to
`cblas_domatcopy'". That took banded_mmt's whole canonicalized CPU form down while its GPU form
(a vendor ``geam``) ran, so the CPU expansion asks whether the symbol exists before using it.
"""
import dace
from dace.libraries.blas.environments import openblas
from dace.libraries.linalg import Transpose

ROWS, COLUMNS = 6, 4


def transpose_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [ROWS, COLUMNS], dace.float64)
    sdfg.add_array('out', [COLUMNS, ROWS], dace.float64)
    state = sdfg.add_state()
    node = Transpose('t', dace.float64)
    node.implementation = 'OpenBLAS'
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_inp', dace.Memlet.from_array('a', sdfg.arrays['a']))
    state.add_edge(node, '_out', state.add_write('out'), None, dace.Memlet.from_array('out', sdfg.arrays['out']))
    return sdfg


def expanded_code(sdfg: dace.SDFG) -> str:
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg.generate_code()[0].clean_code


def test_openblas_transpose_falls_back_when_the_extension_is_absent(monkeypatch):
    """The numerics of the fallback are pinned in ``transpose_unit_extent_test``; this pins the
    CHOICE, which is what a machine without the extension needs and cannot link to check."""
    monkeypatch.setattr(openblas, 'exports_symbol', lambda symbol: False)
    assert 'cblas_domatcopy' not in expanded_code(transpose_sdfg('transpose_no_extension'))


def test_openblas_transpose_keeps_the_call_when_the_symbol_resolves(monkeypatch):
    """An unprobeable or present symbol leaves the BLAS call in place."""
    monkeypatch.setattr(openblas, 'exports_symbol', lambda symbol: True)
    assert 'cblas_domatcopy' in expanded_code(transpose_sdfg('transpose_extension_present'))
    monkeypatch.setattr(openblas, 'exports_symbol', lambda symbol: None)
    assert 'cblas_domatcopy' in expanded_code(transpose_sdfg('transpose_unprobeable'))


def test_probe_reports_an_unprobeable_library_set_as_unknown(monkeypatch):
    """``exports_symbol`` answers ``None`` only when no library could be loaded at all."""
    openblas.exports_symbol.cache_clear()
    monkeypatch.setattr(openblas.OpenBLAS, 'cmake_libraries', staticmethod(lambda: []))
    assert openblas.exports_symbol('cblas_domatcopy') is None
    openblas.exports_symbol.cache_clear()


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__]))
