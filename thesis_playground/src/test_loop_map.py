import numpy as np
from numpy import f2py

from dace.sdfg import SDFG
from dace.frontend.fortran import fortran_parser
from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate import LoopToMap

from utils.general import save_graph, reset_graph_files, get_fortran

nblocks = 5
klev = 7


def test_parallel_sdfg(sdfg: SDFG, expected_output: np.ndarray):
    inp = np.asfortranarray(np.zeros((nblocks, klev), dtype=np.float32))
    sdfg(INP1=inp, NBLOCKS=nblocks, KLEV=klev)
    np.set_printoptions(formatter={'all': lambda x: f"{x:.4f}"})
    if np.allclose(inp, expected_output):
        print("SUCCESS")
    else:
        print("FAIL")
        print("Output")
        print(inp)
        print("Expected")
        print(expected_output)


def test_fortran_code(code: str, name: str, expected_output: np.ndarray, force: bool = False):
    sdfg = fortran_parser.create_sdfg_from_string(code, "test_loop_map_parallel")

    sdfg.simplify()
    save_graph(sdfg, "test_loop_map", f"{name}_initial")
    test_parallel_sdfg(sdfg, expected_output)

    sdfg.apply_transformations_repeated([LoopToMap])
    if force:
        sdfg.apply_transformations_repeated([LoopToMap], permissive=True)
    save_graph(sdfg, "test_loop_map", f"{name}_after_loop_to_map")
    # sdfg.simplify()
    save_graph(sdfg, "test_loop_map", f"{name}_after_simplify")

    sdfg.apply_transformations_repeated([MapToForLoop])
    # sdfg.simplify()
    save_graph(sdfg, "test_loop_map", f"{name}_after_map_to_for_loop")

    test_parallel_sdfg(sdfg, expected_output)


def main():
    fortran_code_parallel = """
    PROGRAM foo
        IMPLICIT NONE
        REAL INP1(NBLOCKS, KLEV)
        INTEGER, PARAMETER  :: KLEV = 137
        INTEGER, PARAMETER  :: NBLOCKS = 8


        CALL foo_test_function(NBLOCKS, KLEV, INP1)

    END PROGRAM

    SUBROUTINE foo_test_function(NBLOCKS, KLEV, INP1)
        INTEGER, PARAMETER  :: KLEV = 137
        INTEGER, PARAMETER  :: NBLOCKS = 1
        REAL INP1(NBLOCKS, KLEV)

        DO JN=1,NBLOCKS
            DO JK=1,KLEV
                INP1(JN, JK) = (JN-1) * KLEV + (JK-1)
            ENDDO
        ENDDO
    END SUBROUTINE foo_test_function
    """

    fortran_code_dependency = """
    PROGRAM foo
        IMPLICIT NONE
        REAL INP1(NBLOCKS, KLEV)
        INTEGER, PARAMETER  :: KLEV = 137
        INTEGER, PARAMETER  :: NBLOCKS = 8


        CALL foo_test_function(NBLOCKS, KLEV, INP1)

    END PROGRAM

    SUBROUTINE foo_test_function(NBLOCKS, KLEV, INP1)
        INTEGER, PARAMETER  :: KLEV = 137
        INTEGER, PARAMETER  :: NBLOCKS = 1
        REAL INP1(NBLOCKS, KLEV)

        DO JN=1,NBLOCKS
            INP1(JN, 1) = (JN-1) * KLEV
            DO JK=2,KLEV
                INP1(JN, JK) = INP1(JN, JK-1) + 1
            ENDDO
        ENDDO
    END SUBROUTINE foo_test_function
    """

    expected_array = np.reshape(np.arange(0, nblocks*klev, dtype=np.float32), (nblocks, klev))
    reset_graph_files("test_loop_map")

    print("Parallel")
    test_fortran_code(fortran_code_parallel, "parallel", expected_output=expected_array)

    print("Dependency")
    test_fortran_code(fortran_code_dependency, "dependency", expected_output=expected_array, force=True)


if __name__ == '__main__':
    main()
