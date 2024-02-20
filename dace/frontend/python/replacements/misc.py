# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for miscellaneous functions.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import SDFG, SDFGState


@oprepo.replaces('slice')
def _slice(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *args, **kwargs):
    return (slice(*args, **kwargs), )
