# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.optimization import MapPermutationTuner
import dace.dtypes as dtypes
import numpy as np
import copy


@dace.program
def program_autotuned(arr_A, arr_B):
    arr_A[:, :, :] = np.log(arr_B[:, :, :])


def test_map_permutation_tuner():
    shape = (3, 3, 100)
    A = np.ones(shape)
    B = np.ones(shape)

    # program_autotuned(A, B)

    sdfg = program_autotuned.to_sdfg(arr_A=A, arr_B=B)
    sdfg_pre_tune = copy.deepcopy(sdfg)
    for state, _node in sdfg_pre_tune.all_nodes_recursive():
        if isinstance(state, dace.nodes.MapEntry) and state.label == "_numpy_log__map":
            assert state.map.params[0] == "__i0"

    tuner = MapPermutationTuner(sdfg)
    tuner.dry_run(sdfg, arr_A=A, arr_B=B)
    report = tuner.optimize(apply=True, measurements=10)
    print(report)

    sdfg.save("post_tune.sdfg")
    sdfg_post_tune = copy.deepcopy(sdfg)
    for state, _node in sdfg_post_tune.all_nodes_recursive():
        if isinstance(state, dace.nodes.MapEntry) and state.label == "_numpy_log__map":
            assert state.map.params[0] == "__i2"


if __name__ == '__main__':
    test_map_permutation_tuner()
