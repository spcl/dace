# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements for miscellaneous functions.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace import SDFG, SDFGState, dtypes


@oprepo.replaces('slice')
def _slice(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, *args, **kwargs):
    return (slice(*args, **kwargs), )


@oprepo.replaces_operator('Array', 'MatMult', otherclass='StorageType')
def _cast_storage(visitor: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, arr: str, stype: dtypes.StorageType) -> str:
    desc = sdfg.arrays[arr]
    desc.storage = stype
    return arr
