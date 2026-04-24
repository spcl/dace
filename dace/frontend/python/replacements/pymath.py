# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements of Python mathematical operations.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, complex_to_scalar, simple_call
from dace import data, dtypes, symbolic, SDFG, SDFGState

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


# -------------------------------------------------------------------- #
#  Descriptor inference for math attributes (schedule-tree frontend)     #
# -------------------------------------------------------------------- #

from dace.frontend.common.op_repository import infers_attribute_descriptor, infers_descriptor, infers_method_descriptor
from dace.frontend.python.replacements.utils import complex_to_scalar as _complex_to_scalar
from dace.frontend.python.replacements.type_inference import _get_desc


def _clone_shape_preserving_descriptor(desc: data.Data, dtype=None):
    out_dtype = desc.dtype if dtype is None else dtype
    retval = desc.clone()
    retval.dtype = out_dtype
    retval.transient = True
    return retval


def _infer_shape_preserving_math_descriptor(input_descs, input, dtype=None, **_kw):
    desc = _get_desc(input_descs, input)
    if desc is None:
        return None
    return _clone_shape_preserving_descriptor(desc, dtype=dtype)


def _infer_attr_real(self_desc):
    out_dtype = _complex_to_scalar(self_desc.dtype)
    if isinstance(self_desc, data.Scalar):
        return data.Scalar(out_dtype)
    return data.Array(out_dtype, list(self_desc.shape), transient=True)


def _infer_attr_imag(self_desc):
    out_dtype = _complex_to_scalar(self_desc.dtype)
    if isinstance(self_desc, data.Scalar):
        return data.Scalar(out_dtype)
    return data.Array(out_dtype, list(self_desc.shape), transient=True)


for _name in ('exp', 'dace.exp', 'numpy.exp', 'math.exp', 'sin', 'dace.sin', 'numpy.sin', 'math.sin', 'cos', 'dace.cos',
              'numpy.cos', 'math.cos', 'sqrt', 'dace.sqrt', 'numpy.sqrt', 'math.sqrt', 'log', 'dace.log', 'numpy.log',
              'math.log', 'log10', 'dace.log10', 'math.log10', 'abs'):
    infers_descriptor(_name)(_infer_shape_preserving_math_descriptor)

for _name in ('math.floor', 'math.ceil', 'round'):
    infers_descriptor(_name)(lambda input_descs, input, **_kw: _infer_shape_preserving_math_descriptor(
        input_descs, input, dtype=dtypes.typeclass(int)))


@infers_descriptor('conj')
@infers_descriptor('dace.conj')
@infers_descriptor('numpy.conj')
def _infer_conj(input_descs, input, **_kw):
    return _infer_shape_preserving_math_descriptor(input_descs, input)


@infers_descriptor('real')
@infers_descriptor('dace.real')
@infers_descriptor('numpy.real')
def _infer_real(input_descs, input, **_kw):
    desc = _get_desc(input_descs, input)
    if desc is None:
        return None
    return _infer_attr_real(desc)


@infers_descriptor('imag')
@infers_descriptor('dace.imag')
@infers_descriptor('numpy.imag')
def _infer_imag(input_descs, input, **_kw):
    desc = _get_desc(input_descs, input)
    if desc is None:
        return None
    return _infer_attr_imag(desc)


for _cls in ('Array', 'Scalar', 'View'):
    infers_attribute_descriptor(_cls, 'real')(_infer_attr_real)
    infers_attribute_descriptor(_cls, 'imag')(_infer_attr_imag)
    infers_method_descriptor(_cls, 'conj')(lambda self_desc, **_kw: _clone_shape_preserving_descriptor(self_desc))
