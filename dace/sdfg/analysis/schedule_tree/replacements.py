# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
from typing import Tuple


@oprepo.replaces('dace.tree.library')
def _library(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, ltype: str, label: str, inputs: Tuple[str], outputs: Tuple[str]):
    print(ltype)
    print(label)
    print(inputs)
    print(outputs)
