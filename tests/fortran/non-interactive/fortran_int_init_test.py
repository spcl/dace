# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

from dace import nodes
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string
from tests.fortran.fortran_test_helper import SourceCodeBuilder


@pytest.mark.skip('Cannot pass until internal AST handles `kind`s correctly.')
def test_fortran_frontend_int_init():
    """
    Tests that the power intrinsic is correctly parsed and translated to DaCe. (should become a*a)
    """
    sources, main = SourceCodeBuilder().add_file("""
subroutine main(d)
  integer, parameter :: i8 = selected_int_kind(15)
  integer(kind=i8) d(2)
  d(1) = int(z'000000ffffffffff', i8)
end subroutine main
""").check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main')
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            if node.sdfg is not None:
                if 'test_function' in node.sdfg.name:
                    sdfg = node.sdfg
                    break
    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()
    sdfg.simplify()
    d = np.full([2], 42, order="F", dtype=np.int64)
    sdfg(d=d)
    assert d[0] == int("000000ffffffffff", 16)


if __name__ == "__main__":
    test_fortran_frontend_int_init()
