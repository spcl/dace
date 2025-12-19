# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements of Python mathematical operations.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, complex_to_scalar, simple_call
from dace import dtypes, symbolic, SDFG, SDFGState

from numbers import Number
from typing import Union


@oprepo.replaces('exp')
@oprepo.replaces('dace.exp')
@oprepo.replaces('numpy.exp')
@oprepo.replaces('math.exp')
def _exp(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'exp')


@oprepo.replaces('sin')
@oprepo.replaces('dace.sin')
@oprepo.replaces('numpy.sin')
@oprepo.replaces('math.sin')
def _sin(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'sin')


@oprepo.replaces('cos')
@oprepo.replaces('dace.cos')
@oprepo.replaces('numpy.cos')
@oprepo.replaces('math.cos')
def _cos(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'cos')


@oprepo.replaces('sqrt')
@oprepo.replaces('dace.sqrt')
@oprepo.replaces('numpy.sqrt')
@oprepo.replaces('math.sqrt')
def _sqrt(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'sqrt')


@oprepo.replaces('log')
@oprepo.replaces('dace.log')
@oprepo.replaces('numpy.log')
@oprepo.replaces('math.log')
def _log(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'log')


@oprepo.replaces('log10')
@oprepo.replaces('dace.log10')
@oprepo.replaces('math.log10')
def _log10(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'log10')


@oprepo.replaces('math.floor')
def _floor(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'floor', restype=dtypes.typeclass(int))


@oprepo.replaces('math.ceil')
def _ceil(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'ceil', restype=dtypes.typeclass(int))


@oprepo.replaces('conj')
@oprepo.replaces('dace.conj')
@oprepo.replaces('numpy.conj')
def _conj(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    return simple_call(pv, sdfg, state, input, 'conj')


@oprepo.replaces('real')
@oprepo.replaces('dace.real')
@oprepo.replaces('numpy.real')
def _real(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return simple_call(pv, sdfg, state, input, 'real', complex_to_scalar(inptype))


@oprepo.replaces('imag')
@oprepo.replaces('dace.imag')
@oprepo.replaces('numpy.imag')
def _imag(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: str):
    inptype = sdfg.arrays[input].dtype
    return simple_call(pv, sdfg, state, input, 'imag', complex_to_scalar(inptype))


@oprepo.replaces_attribute('Array', 'real')
@oprepo.replaces_attribute('Scalar', 'real')
@oprepo.replaces_attribute('View', 'real')
def _ndarray_real(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _real(pv, sdfg, state, arr)


@oprepo.replaces_attribute('Array', 'imag')
@oprepo.replaces_attribute('Scalar', 'imag')
@oprepo.replaces_attribute('View', 'imag')
def _ndarray_imag(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    return _imag(pv, sdfg, state, arr)


@oprepo.replaces_method('Array', 'conj')
@oprepo.replaces_method('Scalar', 'conj')
@oprepo.replaces_method('View', 'conj')
def _ndarray_conj(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arr: str) -> str:
    from dace.frontend.python.replacements.ufunc import implement_ufunc
    return implement_ufunc(pv, None, sdfg, state, 'conj', [arr], {})[0]


@oprepo.replaces('abs')
def _abs(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: Union[str, Number, symbolic.symbol]):
    return simple_call(pv, sdfg, state, input, 'abs')


@oprepo.replaces('round')
def _round(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: Union[str, Number, symbolic.symbol]):
    return simple_call(pv, sdfg, state, input, 'round', dtypes.typeclass(int))
