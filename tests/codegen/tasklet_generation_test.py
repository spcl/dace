import dace
import pytest


def _gen_sdfg_with_a_print_tasklet_between_map_exits() -> dace.SDFG:
    sdfg = dace.SDFG('test_sdfg')
    sdfg.add_array('A', [100], dace.float32)
    sdfg.add_array('B', [100], dace.float32)
    state = sdfg.add_state('main')

    outer_map_entry, outer_map_exit = state.add_map('map1', {'i0': '0:100:10'}, schedule=dace.ScheduleType.Default)
    inner_map_entry, inner_map_exit = state.add_map('map1', {'i1': 'i0:i0+10:1'}, schedule=dace.ScheduleType.Default)
    in_A = state.add_read('A')
    in_B = state.add_read('B')
    out_B = state.add_write('B')

    t1 = state.add_tasklet('add', {'_in_a', '_in_b'}, {'_out_b'}, '_out_b = _in_a + _in_b')

    state.add_edge(in_A, None, outer_map_entry, "IN_A", dace.memlet.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(in_B, None, outer_map_entry, "IN_B", dace.memlet.Memlet.from_array('B', sdfg.arrays['B']))
    state.add_edge(outer_map_entry, "OUT_A", inner_map_entry, "IN_A", dace.Memlet("A[i0:i0+10]"))
    state.add_edge(outer_map_entry, "OUT_B", inner_map_entry, "IN_B", dace.Memlet("B[i0:i0+10]"))
    state.add_edge(inner_map_entry, "OUT_A", t1, "_in_a", dace.Memlet("A[i1]"))
    state.add_edge(inner_map_entry, "OUT_B", t1, "_in_b", dace.Memlet("B[i1]"))
    state.add_edge(t1, "_out_b", inner_map_exit, "IN_B", dace.Memlet("B[i1]"))
    state.add_edge(inner_map_exit, "OUT_B", outer_map_exit, "IN_B", dace.Memlet("B[i0:i0+10]"))
    state.add_edge(outer_map_exit, "OUT_B", out_B, None, dace.Memlet("B[0:100]"))

    t2 = state.add_tasklet(name='printf',
                           inputs={},
                           outputs={},
                           code_global='#include <stdio.h>',
                           code='printf("At iteration %d\\n", i0);',
                           language=dace.Language.CPP)
    state.add_edge(inner_map_exit, None, t2, None, dace.Memlet())
    state.add_edge(t2, None, outer_map_exit, None, dace.Memlet())

    outer_map_entry.add_in_connector("IN_A")
    outer_map_entry.add_in_connector("IN_B")
    outer_map_entry.add_out_connector("OUT_A")
    outer_map_entry.add_out_connector("OUT_B")
    inner_map_entry.add_in_connector("IN_A")
    inner_map_entry.add_in_connector("IN_B")
    inner_map_entry.add_out_connector("OUT_A")
    inner_map_entry.add_out_connector("OUT_B")
    inner_map_exit.add_in_connector("IN_B")
    inner_map_exit.add_out_connector("OUT_B")
    outer_map_exit.add_in_connector("IN_B")
    outer_map_exit.add_out_connector("OUT_B")

    return sdfg


def test_print_tasklet_between_map_exits():
    sdfg = _gen_sdfg_with_a_print_tasklet_between_map_exits()
    sdfg.validate()
    sdfg.compile()


if __name__ == "__main__":
    test_print_tasklet_between_map_exits()
