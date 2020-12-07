# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
"""
Various helper functions to initialize variables and arrays
"""

from dace.memlet import Memlet
from dace import dtypes


def fpga_init_array(state, array, length, value, unroll=False):
    """Initialize memory array in given state from 0 to length (excl.)
        with the given value on the FPGA; optional unrolling.
    """
    buf_write_init = state.add_write(array)

    init_tasklet, init_entry, init_exit = state.add_mapped_tasklet(
        'init_{}'.format(array),
        dict(j_init='0:{}'.format(length)),
        dict(),
        '''
out = {}
        '''.format(value),
        dict(out=Memlet.simple(buf_write_init.data, 'j_init')),
        schedule=dtypes.ScheduleType.FPGA_Device,
        unroll_map=unroll)

    state.add_edge(init_exit,
                   None,
                   buf_write_init,
                   None,
                   memlet=Memlet.simple(buf_write_init.data, 'j_init'))
