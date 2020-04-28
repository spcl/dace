"""Deep Learning ops. Functions should be similar to torch.nn.functional"""

from typing import Dict, Union, Tuple, List

import numpy as np
import dace
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements import _reduce, _max, _sum
import dace.graph.nodes as nd
from dace.frontend.python.nested_call import NestedCall

@oprepo.replaces('torch.nn.functional.softmax')
def _softmax(sdfg: SDFG, state: SDFGState, inpname: str, dim: int):

    nest = NestedCall(sdfg, state)
    inparr = sdfg.arrays[inpname]

    tmp_max = nest(_max)(inpname, axis=dim)

    out_tmp_name, out_tmp_arr = sdfg.add_temp_transient(inparr.shape, inparr.dtype)
    nest.add_state().add_mapped_tasklet(
        "_softmax_exp_",
        map_ranges={
            "__i" + str(i): "0:" + str(shape)
            for i, shape in enumerate(inparr.shape)
        },
        inputs={
            '__max':
            Memlet.simple(
                tmp_max, ','.join("__i" + str(i) for i in range(len(inparr.shape))
                                  if i != dim)),
            '__x':
            Memlet.simple(
                inpname, ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        code='__out = exp(__x - __max)',
        outputs={
            '__out':
            Memlet.simple(
                out_tmp_name,
                ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        external_edges=True)

    tmp_sum = nest(_sum)(out_tmp_name, axis=dim)

    out_name, out_arr = sdfg.add_temp_transient(inparr.shape, inparr.dtype)
    nest.add_state().add_mapped_tasklet(
        "_softmax_div_",
        map_ranges={
            "__i" + str(i): "0:" + str(shape)
            for i, shape in enumerate(inparr.shape)
        },
        inputs={
            '__sum':
            Memlet.simple(
                tmp_sum, ','.join("__i" + str(i) for i in range(len(inparr.shape))
                                  if i != dim)),
            '__exp':
            Memlet.simple(
                out_tmp_name, ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        },
        code='__out = __exp / __sum',
        outputs={
            '__out':
            Memlet.simple(
                out_name,
                ','.join("__i" + str(i) for i in range(len(inparr.shape))))
        }, external_edges=True)
    return out_name
