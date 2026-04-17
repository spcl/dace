# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace import dtypes, symbolic
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import AccessNode

from dace.sdfg.state import LoopRegion


def test_fortran_frontend_array_access():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM access_test
                    implicit none
                    double precision d(4)
                    CALL array_access_test_function(d)
                    end

                    SUBROUTINE array_access_test_function(d)
                    double precision d(4)

                    d(2)=5.5

                    END SUBROUTINE array_access_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "array_access_test")
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 5.5)
    assert (a[2] == 42)


def test_fortran_frontend_array_ranges():
    """
    Tests that the Fortran frontend can parse multidimenstional arrays with vectorized ranges and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM ranges_test
                    implicit none
                    double precision d(3,4,5)
                    CALL array_ranges_test_function(d)
                    end

                    SUBROUTINE array_ranges_test_function(d)
                    double precision d(3,4,5),e(3,4,5),f(3,4,5)

                    e(:,:,:)=1.0
                    f(:,:,:)=2.0
                    f(:,2:4,:)=3.0
                    f(1,1,:)=4.0
                    d(:,:,:)=e(:,:,:)+f(:,:,:)
                    d(1,2:4,1)=e(1,2:4,1)*10.0
                    d(1,1,1)=SUM(e(:,1,:))

                    END SUBROUTINE array_ranges_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "array_access_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 15)
    assert (d[0, 1, 0] == 10)
    assert (d[1, 0, 0] == 3)
    assert (d[2, 3, 3] == 4)
    assert (d[0, 0, 2] == 5)


def test_fortran_frontend_array_3dmap():
    """
    Tests that the normalization of multidimensional array indices works correctly.
    """
    test_string = """
                    PROGRAM array_3dmap_test
                    implicit none
                    double precision d(4,4,4)
                    CALL array_3dmap_test_function(d)
                    end

                    SUBROUTINE array_3dmap_test_function(d)
                    double precision d(4,4,4)

                    d(:,:,:)=7

                    END SUBROUTINE array_3dmap_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "array_3dmap_test")
    sdfg.simplify(verbose=True)
    sdutil.normalize_offsets(sdfg)
    from dace.transformation.auto import auto_optimize as aopt
    aopt.auto_optimize(sdfg, dtypes.DeviceType.CPU)
    a = np.full([4, 4, 4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0, 0] == 7)
    assert (a[3, 3, 3] == 7)


def test_fortran_frontend_twoconnector():
    """
    Tests that the multiple connectors to one array are handled correctly.
    """
    test_string = """
                    PROGRAM twoconnector_test
                    implicit none
                    double precision d(4)
                    CALL twoconnector_test_function(d)
                    end

                    SUBROUTINE twoconnector_test_function(d)
                    double precision d(4)

                    d(2)=d(1)+d(3)

                    END SUBROUTINE twoconnector_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "twoconnector_test")
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 84)
    assert (a[2] == 42)


def test_fortran_frontend_input_output_connector():
    """
    Tests that the presence of input and output connectors for the same array is handled correctly.
    """
    test_string = """
                    PROGRAM ioc_test
                    implicit none
                    double precision d(2,3)
                    CALL ioc_test_function(d)
                    end

                    SUBROUTINE ioc_test_function(d)
                    double precision d(2,3)
                    integer a,b

                    a=1
                    b=2
                    d(:,:)=0.0
                    d(a,b)=d(1,1)+5

                    END SUBROUTINE ioc_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "ioc_test")
    sdfg.simplify(verbose=True)
    a = np.full([2, 3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 0)
    assert (a[0, 1] == 5)
    assert (a[1, 2] == 0)


def test_fortran_frontend_memlet_in_map_test():
    """
    Tests that no assumption is made where the iteration variable is inside a memlet subset
    """
    test_string = """
        PROGRAM memlet_range_test
        implicit None
        REAL INP(100, 10)
        REAL OUT(100, 10)
        CALL memlet_range_test_routine(INP, OUT)
        END PROGRAM

        SUBROUTINE memlet_range_test_routine(INP, OUT)
            REAL INP(100, 10)
            REAL OUT(100, 10)
            DO I=1,100
                CALL inner_loops(INP(I, :), OUT(I, :))
            ENDDO
        END SUBROUTINE memlet_range_test_routine

        SUBROUTINE inner_loops(INP, OUT)
            REAL INP(10)
            REAL OUT(10)
            DO J=1,10
                OUT(J) = INP(J) + 1
            ENDDO
        END SUBROUTINE inner_loops

    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "memlet_range_test")
    sdfg.simplify()
    # Expect that the start block is a loop
    loop = sdfg.nodes()[0]
    assert isinstance(loop, LoopRegion)
    iter_var = symbolic.pystr_to_symbolic(loop.loop_variable)

    for state in sdfg.states():
        if len(state.nodes()) > 1:
            for node in state.nodes():
                if isinstance(node, AccessNode) and node.data in ['INP', 'OUT']:
                    edges = [*state.in_edges(node), *state.out_edges(node)]
                    # There should be only one edge in/to the access node
                    assert len(edges) == 1
                    memlet = edges[0].data
                    # Check that the correct memlet has the iteration variable
                    subset = memlet.subset if memlet.data in ('INP', 'OUT') else memlet.other_subset
                    assert subset[0] == (iter_var, iter_var, 1)
                    assert subset[1] == (1, 10, 1)


if __name__ == "__main__":
    test_fortran_frontend_array_3dmap()
    test_fortran_frontend_array_access()
    test_fortran_frontend_input_output_connector()
    test_fortran_frontend_array_ranges()
    test_fortran_frontend_twoconnector()
    test_fortran_frontend_memlet_in_map_test()
