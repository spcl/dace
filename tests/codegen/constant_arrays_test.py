# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_nsdfg_input():
    """ Tests constexpr array passed as input argument to a NestedSDFG. """

    @dace.program
    def constexpr_nsdfg():
        a = np.array([1.,2.,3.])
        b = np.max(a)

    with dace.config.set_temporary('compiler', 'inline_sdfgs', value=False):
        constexpr_nsdfg()


def test_tasklet_input_cpu():
    """ Tests constexpr array passed as input argument to a Tasklet (CPU)."""

    @dace.program
    def constexpr_tasklet_cpu():
        a = np.array([1.,2.,3.])
        b = np.max(a)

    with dace.config.set_temporary('optimizer', 'autooptimize', value=True):
        constexpr_tasklet_cpu()


if __name__ == "__main__":
    test_nsdfg_input()
    test_tasklet_input_cpu()
