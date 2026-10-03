# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vendor TensorDot expansions spell symbolic extents and strides as C++.

They formatted each extent and stride with ``str()``, which prints a cast as Python does:
``int64(la_max_index) + int64(lb_max_index) + 1``. No C++ scope declares a bare ``int64``, so
``cp2k_grid_integrate`` on ``dace_gpu_canonicalize`` (hipTensor in the ROCm 7.2 judge image) failed
to build with ``'int64' was not declared in this scope``. The emitted code must use ``dace::int64``.
"""
import re

import dace
from dace import dtypes
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.symbolic import pystr_to_symbolic

#: A contracted extent in the shape the canon pipeline produces: a sum of int64-cast symbols.
EXTENT = pystr_to_symbolic('int64(la) + int64(lb) + 1')

#: ``int64(`` that is NOT the ``dace::int64(`` spelling.
BARE_INT64 = re.compile(r'(?<!::)\bint64\(')


def symbolic_contraction(implementation: str, storage: dtypes.StorageType) -> dace.SDFG:
    """``C[i, k] = sum_j A[i, j] * B[j, k]`` with ``j`` over :data:`EXTENT`, so both an extent
    and a stride of each operand are the cast expression."""
    sdfg = dace.SDFG(f'tensordot_symbolic_{implementation}')
    for name, shape in (('A', [4, EXTENT]), ('B', [EXTENT, 6]), ('C', [4, 6])):
        sdfg.add_array(name, shape, dace.float64, storage=storage)
    state = sdfg.add_state()
    node = TensorDot('contract', left_axes=[1], right_axes=[0])
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_left_tensor', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(state.add_read('B'), None, node, '_right_tensor', dace.Memlet.from_array('B', sdfg.arrays['B']))
    state.add_edge(node, '_out_tensor', state.add_write('C'), None, dace.Memlet('C[0:4, 0:6]'))
    sdfg.validate()
    return sdfg


def expanded_code(sdfg: dace.SDFG) -> str:
    """Every tasklet's code after expanding the library nodes."""
    sdfg.expand_library_nodes()
    return '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def test_hiptensor_extents_and_strides_are_cpp():
    """The extent and stride vectors carry ``dace::int64``, never a bare ``int64``."""
    code = expanded_code(symbolic_contraction('hipTENSOR', dtypes.StorageType.GPU_Global))
    assert 'hiptensor' in code.lower(), code[:400]
    assert 'dace::int64(la)' in code, code
    assert not BARE_INT64.search(code), BARE_INT64.search(code)


def test_tblis_lengths_and_strides_are_cpp():
    """The TBLIS length and stride arrays take the same C++ spelling."""
    code = expanded_code(symbolic_contraction('TBLIS', dtypes.StorageType.Default))
    assert 'tblis' in code.lower(), code[:400]
    assert 'dace::int64(la)' in code, code
    assert not BARE_INT64.search(code), BARE_INT64.search(code)


if __name__ == '__main__':
    test_hiptensor_extents_and_strides_are_cpp()
    test_tblis_lengths_and_strides_are_cpp()
