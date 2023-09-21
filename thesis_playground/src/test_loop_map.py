import numpy as np

from dace.sdfg import SDFG
from dace.frontend.fortran import fortran_parser
from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate import LoopToMap

from utils.general import save_graph, reset_graph_files


def test_parallel_sdfg(sdfg: SDFG):
    nblocks = 8
    klev = 15
    inp = np.asfortranarray(np.zeros((nblocks, klev), dtype=np.float32))
    sdfg(INP1=inp, NBLOCKS=nblocks, KLEV=klev)
    np.set_printoptions(formatter={'all': lambda x: f"{x:.4f}"})
    inp_expected = np.reshape(np.arange(0, nblocks*klev, like=inp), (nblocks, klev))
    print(f"PASS: {np.allclose(inp, inp_expected)}")


def test_fortran_code(code: str, name: str):
    sdfg = fortran_parser.create_sdfg_from_string(code, "test_loop_map_parallel")
    sdfg.simplify()
    save_graph(sdfg, "test_loop_map", f"{name}_initial")

    sdfg.apply_transformations_repeated([LoopToMap])
    # sdfg.simplify()
    save_graph(sdfg, "test_loop_map", f"{name}_loop_to_map")

    sdfg.apply_transformations_repeated([MapToForLoop])
    # sdfg.simplify()
    save_graph(sdfg, "test_loop_map", f"{name}_after_map_to_for_loop")

    test_parallel_sdfg(sdfg)


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
    reset_graph_files("test_loop_map")
    test_fortran_code(fortran_code_parallel, "parallel")


if __name__ == '__main__':
    main()
