# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memlet propagation through a strided map keeps the dtype of the range's bound symbols."""
import dace
from dace.sdfg.propagation import propagate_memlets_sdfg


def test_a_strided_range_keeps_the_dtype_of_its_bound_symbols():
    """The last index of a strided map range is built from the bound symbols, never re-parsed from text:
    a re-parse mints them at the default dtype, and ``S: uint32`` beside ``S: int32`` never cancels."""
    S = dace.symbol('S', dace.uint32)
    E = dace.symbol('E', dace.uint32)
    sdfg = dace.SDFG('strided_uint_bounds')
    sdfg.add_symbol('S', dace.uint32)
    sdfg.add_symbol('E', dace.uint32)
    sdfg.add_array('A', [E * E], dace.float64)
    sdfg.add_array('B', [E], dace.float64)
    state = sdfg.add_state()
    square = dace.symbol('i', dace.uint32)**2
    state.add_mapped_tasklet('gather', {'i': dace.subsets.Range([(S, E - 1, 4)])},
                             {'a': dace.Memlet(data='A', subset=dace.subsets.Range([(square, square, 1)]))},
                             'b = a', {'b': dace.Memlet('B[i]')},
                             external_edges=True)
    propagate_memlets_sdfg(sdfg)
    outer = next(e.data for e in state.edges() if isinstance(e.dst, dace.sdfg.nodes.MapEntry))
    dtypes_seen = {(s.name, s.dtype) for dim in outer.subset.ranges for x in dim for s in x.free_symbols}
    assert dtypes_seen and all(dtype == dace.uint32 for name, dtype in dtypes_seen if name in ('S', 'E')), dtypes_seen
    sdfg.validate()


if __name__ == '__main__':
    test_a_strided_range_keeps_the_dtype_of_its_bound_symbols()
