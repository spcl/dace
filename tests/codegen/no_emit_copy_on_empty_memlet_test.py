import dace
import pytest

def _gen_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('test_no_emit_copy_on_empty_memlet')
    state = sdfg.add_state()


    nameA, arrA = sdfg.add_array('A', [2], dace.float32)
    nameB, arrB = sdfg.add_array('B', [2], dace.float32)


    tasklet1 = state.add_tasklet('tasklet1', {}, {}, 'prinf("Hello World Before Copy\n");', language=dace.Language.CPP, code_global="#include <stdio.h>")
    tasklet2 = state.add_tasklet('tasklet2', {}, {}, 'prinf("Hello World After Copy\n");', language=dace.Language.CPP, code_global="#include <stdio.h>")

    anA = state.add_access(nameA)
    anB = state.add_access(nameB)
    state.add_edge(tasklet1, None, anA, None, dace.Memlet())
    state.add_edge(anA, None, anB, None, dace.Memlet.from_array(name=nameA, datadesc=sdfg.arrays[nameA]))
    state.add_edge(anB, None, tasklet2, None, dace.Memlet())

    return sdfg

def test_no_emit_copy_on_empty_memlet():
    sdfg = _gen_sdfg()
    sdfg.validate()
    sdfg.compile()