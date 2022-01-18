# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.dtypes import vector
import dace
from dace.sdfg.sdfg import SDFG
from dace.transformation.dataflow.vectorization import Vectorization


def vectorize(program):
    sdfg: SDFG = program.to_sdfg(simplify=True)
    sdfg.apply_transformations(Vectorization,
                               {"target": dace.ScheduleType.SVE_Map})
    return sdfg


def get_code(program):
    return vectorize(program).generate_code()[0].clean_code
