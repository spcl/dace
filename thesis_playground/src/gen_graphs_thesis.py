"""
Generate sample graphs to be used in the thesis
"""
from typing import Optional
from argparse import ArgumentParser
import os
import dace
from dace.dtypes import ScheduleType
from dace.memlet import Memlet
from dace.sdfg.sdfg import InterstateEdge, SDFG
from dace.transformation.helpers import nest_state_subgraph
from dace.sdfg.graph import SubgraphView
from dace.properties import CodeBlock
from dace.frontend.fortran import fortran_parser
from dace.transformation.interstate import StateFusion, LoopToMap, RefineNestedAccess
from dace.transformation.dataflow import MapExpansion, MapCollapse, MapToForLoop
from dace.transformation.passes.simplify import SimplifyPass

from execute.my_auto_opt import change_strides, auto_optimize_phase_1
from utils.log import setup_logging


def save(sdfg: SDFG, folder: str, filename: Optional[str] = None):
    if filename is None:
        filename = sdfg.name
    path = os.path.join(folder, f"{filename}.sdfg")
    print(f"Save SDFG into {path}")
    sdfg.save(path)


def simple_map(folder: str):
    sdfg = dace.SDFG('simple_map')
    sdfg.add_array('A', [5], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    state = sdfg.add_state()

    # Nodes
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name="sample_map",
            map_ranges={'i': '0:5'},
            inputs={'a': Memlet(data='A', subset='i')},
            outputs={'b': Memlet(data='B', subset='i')},
            code='b = a + 1',
            external_edges=True
            )
    sdfg.validate()
    save(sdfg, folder, 'simple_map')


def extend_collapse_map(folder: str):
    sdfg = dace.SDFG('extend_collapse_map')
    state = sdfg.add_state()
    m1_entry, m1_exit = state.add_map('map1', {'i': '0:10', 'j': '0:100'}, schedule=ScheduleType.GPU_Device)
    m2_entry, m2_exit = state.add_map('map2', {'i': '0:10'}, schedule=ScheduleType.GPU_Device)
    m2_inner_entry, m2_inner_exit = state.add_map('map2_inner', {'j': '0:100'}, schedule=ScheduleType.Sequential)
    tasklet1 = state.add_tasklet('work1', {}, {}, '')
    tasklet2 = state.add_tasklet('work2', {}, {}, '')

    state.add_memlet_path(m1_entry, tasklet1, m1_exit, memlet=Memlet())
    state.add_memlet_path(m2_entry, m2_inner_entry, tasklet2, m2_inner_exit, m2_exit, memlet=Memlet())

    save(sdfg, folder, 'extend_collapse_map')


def access_nodes(folder: str):
    sdfg = dace.SDFG('access_nodes')
    state = sdfg.add_state()
    A = state.add_array('A', [5], dace.float64)
    B = state.add_array('B', [5], dace.float64, transient=True)
    tasklet = state.add_tasklet('tasklet', {'A'}, {'B'}, 'B = A')
    state.add_memlet_path(A, tasklet, memlet=Memlet('A', '0'), dst_conn='A')
    state.add_memlet_path(tasklet, B, memlet=Memlet('B', '0'), src_conn='B')
    save(sdfg, folder, 'access_nodes')


def states(folder: str):
    sdfg = dace.SDFG('states')
    state1 = sdfg.add_state('init_state', is_start_state=True)
    state2 = sdfg.add_state('state_1')
    state3 = sdfg.add_state('state_2')
    sdfg.add_edge(state1, state2, InterstateEdge(CodeBlock("a>1")))
    sdfg.add_edge(state1, state3, InterstateEdge(CodeBlock("a<=1")))
    save(sdfg, folder)


def states_from_program(folder: str):
    fortran_code = """
PROGRAM main
    IMPLICIT NONE
    INTEGER A
    INTEGER B(2)
    IF (A > 1) THEN
        B(0) = 10
    ELSE
        B(0) = 5
    ENDIF
END
    """
    sdfg = fortran_parser.create_sdfg_from_string(fortran_code, "states_from_program")
    # sdfg.simplify(skip={'ConstantPropagation', 'DeadDataflowElimination'})
    SimplifyPass(skip={'ConstantPropagation', 'DeadDataflowElimination'}).apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated([StateFusion])
    save(sdfg, folder)


def nsdfg(folder: str):
    sdfg = dace.SDFG('outer_sdfg')
    outer_state = sdfg.add_state('outer_state')
    sdfg.add_array('A', [5], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    read_a = outer_state.add_read('A')
    write_a = outer_state.add_write('B')

    inner_sdfg = dace.SDFG('inner_sdfg')
    inner_state_1 = inner_sdfg.add_state('inner_state_1')
    inner_state_2 = inner_sdfg.add_state('inner_state_2')
    inner_sdfg.add_edge(inner_state_1, inner_state_2, InterstateEdge())
    inner_sdfg.add_array('inner_A', [5], dace.float64)
    inner_sdfg.add_array('B', [5], dace.float64)
    tasklet, map_entry, map_exit = inner_state_2.add_mapped_tasklet(
            name='map',
            map_ranges={'i': '0:4'},
            inputs={'a1': Memlet(data='inner_A', subset='i'), 'a2': Memlet(data='inner_A', subset='i+1')},
            outputs={'b': Memlet(data='B', subset='i')},
            code='b = a1 + a2', external_edges=True)

    nsdfg = outer_state.add_nested_sdfg(inner_sdfg, sdfg, {'inner_A'}, {'B'})
    outer_state.add_memlet_path(read_a, nsdfg, memlet=Memlet('A', '0:5'), dst_conn='inner_A')
    outer_state.add_memlet_path(nsdfg, write_a, memlet=Memlet('B', '0:5'), src_conn='B')
    save(sdfg, folder, 'nsdfg')


def change_strides_example(folder: str):
    sdfg = dace.SDFG('work')
    NBLOCKS = dace.symbol('NBLOCKS')
    sdfg.add_array('A', [5, NBLOCKS], dace.float64)
    state = sdfg.add_state()

    # Nodes
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name="sample_map",
            map_ranges={'i': '0:5', 'j': '0:NBLOCKS'},
            inputs={'a': Memlet(data='A', subset='i, j')},
            outputs={'a': Memlet(data='A', subset='i, j')},
            code='a = a + 1',
            external_edges=True
            )

    save(sdfg, folder, filename="before_changed_strides")
    new_sdfg = change_strides(sdfg, ['NBLOCKS'], dace.ScheduleType.Default)
    save(new_sdfg, folder, filename="after_changed_strides")

def refine_memlets(folder: str):
    sdfg = dace.SDFG('refine_dataflow')
    sdfg.add_array('A', [5, 5], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)

    inner_sdfg = dace.SDFG('inner_sdfg')
    inner_sdfg.add_array('A', [5, 5], dace.int32)
    inner_sdfg.add_array('B', [5, 5], dace.int32)
    inner_state = inner_sdfg.add_state('inner_state')
    tasklet = inner_state.add_tasklet('work', {'a'}, {'b'}, 'b = a + 1')
    inner_state.add_memlet_path(inner_state.add_access('A'), tasklet, dst_conn='a', memlet=Memlet.simple('A', 'i,j'))
    inner_state.add_memlet_path(tasklet, inner_state.add_access('B'), src_conn='b', memlet=Memlet.simple('B', 'i,j'))

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('m', dict(i='0:5', j='0:1'))
    # inner_sdfg = inner_sdfg.to_sdfg(simplify=False)
    nsdfg = state.add_nested_sdfg(inner_sdfg, sdfg, {'A'}, {'B'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(A, me, nsdfg, dst_conn='A', memlet=dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_memlet_path(nsdfg, mx, B, src_conn='B', memlet=dace.Memlet.from_array('B', sdfg.arrays['B']))

    save(sdfg, folder, filename='before_refine')
    sdfg.apply_transformations_repeated([RefineNestedAccess])
    save(sdfg, folder, filename='after_refine')

def foo(folder: str):
    fortran_code = """

PROGRAM main
    IMPLICIT NONE
    REAL INP1(KLON, KLEV, NCLV)
    REAL INP2(KLON, KLEV, NCLV)
    INTEGER  RLMIN = 1

    INTEGER, PARAMETER  :: NCLV = 5
    INTEGER, PARAMETER  :: KLEV = 137
    INTEGER, PARAMETER  :: KIDIA = 1
    INTEGER, PARAMETER  :: KFDIA = 1
    INTEGER, PARAMETER  :: KLON = 1


    CALL work(NCLV, KLEV, KIDIA, KFDIA, KLON, RLMIN, INP1, INP2)

END PROGRAM

SUBROUTINE work(NCLV, KLEV, KIDIA, KFDIA, KLON, RLMIN, INP1, INP2)
    INTEGER, PARAMETER  :: NCLV = 5
    INTEGER, PARAMETER  :: KLEV = 137
    INTEGER, PARAMETER  :: KIDIA = 1
    INTEGER, PARAMETER  :: KFDIA = 1
    INTEGER, PARAMETER  :: KLON = 1
    REAL INP1(KLON, KLEV, NCLV)
    REAL INP2(KLON, KLEV, NCLV)
    INTEGER  RLMIN

    DO JM=1,NCLV-1
      DO JK=1,KLEV
        DO JL=KIDIA,KFDIA
          IF (INP2(JL,JK,JM)<RLMIN) THEN
            INP1(JL, JK, JM) = 10
          ENDIF
        ENDDO
      ENDDO
    ENDDO
END SUBROUTINE work
"""
    setup_logging(level="DEBUG")
    sdfg = fortran_parser.create_sdfg_from_string(fortran_code, "foo")
    # sdfg.simplify()
    # sdfg.apply_transformations_repeated([LoopToMap, RefineNestedAccess])
    # sdfg.apply_transformations_repeated([MapCollapse ])
    auto_optimize_phase_1(sdfg, program="foo")
    save(sdfg, folder)


def loop_to_map(folder: str):
    sdfg = dace.SDFG('loop_to_map')
    state = sdfg.add_state()
    KLEV = dace.symbol('KLEV')
    sdfg.add_array('A', [KLEV], dace.int32)
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name="map",
            map_ranges={'i': '0:KLEV'},
            inputs={'a_in': Memlet(data='A', subset='i')},
            outputs={'a_out': Memlet(data='A', subset='i')},
            code='a_out = a_in + 1',
            external_edges=True
            )
    sdfg.apply_transformations_repeated([MapToForLoop])
    sdfg.name = "loop_to_map_before"
    save(sdfg, folder)
    sdfg.apply_transformations_repeated([LoopToMap])
    sdfg.simplify()
    sdfg.name = "loop_to_map_after"
    save(sdfg, folder)


def main():
    parser = ArgumentParser()
    parser.add_argument('graph_folder', help='Path to where all the SDFGs should be saved')
    args = parser.parse_args()

    simple_map(args.graph_folder)
    extend_collapse_map(args.graph_folder)
    access_nodes(args.graph_folder)
    states(args.graph_folder)
    nsdfg(args.graph_folder)
    states_from_program(args.graph_folder)
    change_strides_example(args.graph_folder)
    refine_memlets(args.graph_folder)
    loop_to_map(args.graph_folder)


if __name__ == '__main__':
    main()
