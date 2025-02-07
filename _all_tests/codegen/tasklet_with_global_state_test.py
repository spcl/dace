# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_tasklet_with_global_state():

    sdfg = dace.SDFG("test_tasklet_with_global_state")
    state = sdfg.add_state()

    sdfg.add_array("output", [1], dace.int32)
    tasklet = state.add_tasklet(
        "print_global_str",
        {},
        {"out"},
        "out = *__state->global_int;",
        language=dace.dtypes.Language.CPP,
        state_fields=["int *global_int;"],
        code_init='__state->global_int = new int; *__state->global_int = 42;',
        code_exit='delete __state->global_int;',
    )

    state.add_edge(tasklet, "out", state.add_write("output"), None, dace.Memlet("output[0]"))

    output = np.zeros((1, ), dtype=np.int32)
    sdfg(output=output)

    assert output[0] == 42
