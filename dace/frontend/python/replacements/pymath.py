# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains replacements of Python mathematical operations.
"""
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.replacements.utils import ProgramVisitor, complex_to_scalar, simple_call, step_state
from dace import dtypes, symbolic, SDFG, SDFGState

from numbers import Integral, Number
from typing import Any, Union

import numpy as np

COMPLEX_TYPES = (dtypes.complex64, dtypes.complex128)


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
    # ``abs`` of a complex value is real-valued (the magnitude). Without an explicit result type,
    # simple_call defaults it to the input dtype, leaving the output typed complex -- then a comparison
    # like ``abs(z) < 1.0`` fails to compile (no ``operator<`` on ``std::complex``). Reduce to the scalar
    # type for complex inputs, mirroring the np.abs ufunc path.
    restype = None
    if isinstance(input, str) and input in sdfg.arrays:
        restype = complex_to_scalar(sdfg.arrays[input].dtype)
    return simple_call(pv, sdfg, state, input, 'abs', restype)


@oprepo.replaces('round')
def _round(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, input: Union[str, Number, symbolic.symbol]):
    return simple_call(pv, sdfg, state, input, 'round', dtypes.typeclass(int))


@oprepo.replaces('pow')
@oprepo.replaces('dace.pow')
@oprepo.replaces('math.pow')
def _pow(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Union[str, Number, symbolic.symbol],
         y: Union[str, Number, symbolic.symbol]):
    """Two-argument ``pow(x, y)`` — delegates to ``np.power`` so the resulting tasklet body uses
    the same ``__out = __in1 ** __in2`` shape that the rest of the pipeline (in particular
    ``PowerOperatorExpansion``) is set up to rewrite."""
    from dace.frontend.python.replacements.ufunc import implement_ufunc
    return implement_ufunc(pv, None, sdfg, state, 'power', [x, y], {})[0]


########################################################################
# Element-wise NumPy predicates and repairs
########################################################################


def is_data(sdfg: SDFG, operand: Any) -> bool:
    """Whether an operand names a data container rather than a constant or a symbol."""
    return isinstance(operand, str) and operand in sdfg.arrays


def operand_dtype(sdfg: SDFG, operand: Any) -> dtypes.typeclass:
    """dtype of a replacement operand: the container's own, else the constant's."""
    if is_data(sdfg, operand):
        return sdfg.arrays[operand].dtype
    if symbolic.issymbolic(operand):
        return symbolic.symtype(operand)
    return dtypes.dtype_to_typeclass(type(operand))


def real_result_type(dtype: dtypes.typeclass) -> dtypes.typeclass:
    """The dtype NumPy reaches when it mixes the operand with a Python float: float32 stays
    float32 under NEP 50 weak promotion, integers and booleans widen to float64."""
    return dtypes.dtype_to_typeclass(np.result_type(dtype.type, 1.0).type)


def ufunc_call(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, name: str, args: list[Any]) -> str:
    """One native ufunc application, with NumPy broadcasting and result-type rules."""
    from dace.frontend.python.replacements.ufunc import implement_ufunc  # Avoid import loop
    return implement_ufunc(pv, None, sdfg, step_state(pv, state), name, args, {})[0]


def magnitude(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Any) -> Any:
    """|x|, folded in Python when x is a constant."""
    if isinstance(x, Number):
        return abs(x)
    return ufunc_call(pv, sdfg, state, 'absolute', [x])


def real_probe(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: str) -> str:
    """``isfinite``/``isnan`` have no complex overload in C++, so complex operands are tested
    through their magnitude, which is non-finite exactly when a component is."""
    if operand_dtype(sdfg, x) in COMPLEX_TYPES:
        return ufunc_call(pv, sdfg, state, 'absolute', [x])
    return x


def finite_mask(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Any) -> str | bool:
    if is_data(sdfg, x):
        return ufunc_call(pv, sdfg, state, 'isfinite', [real_probe(pv, sdfg, state, x)])
    return bool(np.isfinite(x)) if isinstance(x, Number) else True


def nan_mask(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Any) -> str | bool:
    if is_data(sdfg, x):
        return ufunc_call(pv, sdfg, state, 'isnan', [real_probe(pv, sdfg, state, x)])
    return bool(np.isnan(x)) if isinstance(x, Number) else False


def mask_and(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, left: str | bool, right: str | bool) -> str | bool:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is False or right is False:
            return False
        return right if left is True else left
    return ufunc_call(pv, sdfg, state, 'logical_and', [left, right])


def mask_or(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, left: str | bool, right: str | bool) -> str | bool:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is True or right is True:
            return True
        return right if left is False else left
    return ufunc_call(pv, sdfg, state, 'logical_or', [left, right])


def select(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, cond: str | bool, on_true: Any, on_false: Any) -> Any:
    """``numpy.where`` with a constant condition folded away."""
    from dace.frontend.python.replacements.filtering import _array_array_where  # Avoid import loop
    if isinstance(cond, bool):
        return on_true if cond else on_false
    return _array_array_where(pv, sdfg, step_state(pv, state), cond, on_true, on_false)


def all_true(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, mask: str) -> str:
    """AND-reduce a mask to a scalar. Spelled out here rather than delegated to ``numpy.all``,
    which reduces with identity 0 and therefore answers False for every input."""
    from dace.frontend.python.replacements.reduction import reduce  # Avoid import loop
    if list(sdfg.arrays[mask].shape) == [1]:
        return mask
    return reduce(pv, sdfg, step_state(pv, state), 'lambda x, y: x and y', mask, axis=None, identity=1)


@oprepo.replaces('numpy.isclose')
def isclose(pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            a: Any,
            b: Any,
            rtol: Number = 1e-05,
            atol: Number = 1e-08,
            equal_nan: bool = False) -> str | bool:
    """``numpy.isclose``: |a - b| <= atol + rtol*|b| wherever both operands are finite, and plain
    equality wherever they are not, since no tolerance band contains an infinity."""
    if not is_data(sdfg, a) and not is_data(sdfg, b):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    difference = magnitude(pv, sdfg, state, ufunc_call(pv, sdfg, state, 'subtract', [a, b]))
    scale = magnitude(pv, sdfg, state, b)
    if isinstance(scale, Number):
        bound = atol + rtol * scale
    else:
        bound = ufunc_call(pv, sdfg, state, 'add', [atol, ufunc_call(pv, sdfg, state, 'multiply', [rtol, scale])])
    close = ufunc_call(pv, sdfg, state, 'less_equal', [difference, bound])

    finite = mask_and(pv, sdfg, state, finite_mask(pv, sdfg, state, a), finite_mask(pv, sdfg, state, b))
    if finite is True:
        result = close
    else:
        result = select(pv, sdfg, state, finite, close, ufunc_call(pv, sdfg, state, 'equal', [a, b]))

    if equal_nan:
        both_nan = mask_and(pv, sdfg, state, nan_mask(pv, sdfg, state, a), nan_mask(pv, sdfg, state, b))
        result = mask_or(pv, sdfg, state, result, both_nan)
    return result


@oprepo.replaces('numpy.allclose')
def allclose(pv: ProgramVisitor,
             sdfg: SDFG,
             state: SDFGState,
             a: Any,
             b: Any,
             rtol: Number = 1e-05,
             atol: Number = 1e-08,
             equal_nan: bool = False) -> str | bool:
    mask = isclose(pv, sdfg, state, a, b, rtol, atol, equal_nan)
    return mask if isinstance(mask, bool) else all_true(pv, sdfg, state, mask)


@oprepo.replaces('numpy.array_equal')
def array_equal(pv: ProgramVisitor,
                sdfg: SDFG,
                state: SDFGState,
                a1: Any,
                a2: Any,
                equal_nan: bool = False) -> str | bool:
    """``numpy.array_equal``: the shape verdict is read off the descriptors, before any dataflow.

    A pair of shapes that is neither provably equal nor provably different is refused: the answer
    would hinge on symbol values that are not known here.
    """
    if is_data(sdfg, a1) and is_data(sdfg, a2):
        shape1, shape2 = sdfg.arrays[a1].shape, sdfg.arrays[a2].shape
        if len(shape1) != len(shape2):
            return False
        for dim1, dim2 in zip(shape1, shape2):
            if not symbolic.inequal_symbols(dim1, dim2):
                continue
            if isinstance(dim1, Integral) and isinstance(dim2, Integral):
                return False
            raise ValueError('numpy.array_equal cannot decide whether the symbolic shapes '
                             f'{tuple(shape1)} and {tuple(shape2)} match')

    equal = ufunc_call(pv, sdfg, state, 'equal', [a1, a2])
    if equal_nan:
        both_nan = mask_and(pv, sdfg, state, nan_mask(pv, sdfg, state, a1), nan_mask(pv, sdfg, state, a2))
        equal = mask_or(pv, sdfg, state, equal, both_nan)
    return equal if isinstance(equal, bool) else all_true(pv, sdfg, state, equal)


@oprepo.replaces('numpy.nan_to_num')
def nan_to_num(pv: ProgramVisitor,
               sdfg: SDFG,
               state: SDFGState,
               x: Any,
               copy: bool = True,
               nan: Number = 0.0,
               posinf: Number | None = None,
               neginf: Number | None = None) -> Any:
    """``numpy.nan_to_num``: NaN and the two infinities are replaced by finite stand-ins."""
    from dace.frontend.python.replacements.array_creation import _numpy_copy  # Avoid import loop

    if isinstance(x, Number):
        return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf).item()

    dtype = operand_dtype(sdfg, x)
    if dtype in COMPLEX_TYPES:
        raise ValueError('numpy.nan_to_num on complex input is unsupported: the repaired real and '
                         'imaginary parts have no native path back into a complex value')
    if not np.issubdtype(dtype.type, np.floating):
        return _numpy_copy(pv, sdfg, state, x) if copy else x

    # Constants carry the input dtype so that a float32 operand does not widen the whole result.
    info = np.finfo(dtype.type)
    positive_fill = dtype.type(info.max if posinf is None else posinf)
    negative_fill = dtype.type(-info.max if neginf is None else neginf)

    infinite = ufunc_call(pv, sdfg, state, 'isinf', [x])
    above = mask_and(pv, sdfg, state, infinite, ufunc_call(pv, sdfg, state, 'greater', [x, 0]))
    below = mask_and(pv, sdfg, state, infinite, ufunc_call(pv, sdfg, state, 'less', [x, 0]))
    result = select(pv, sdfg, state, above, positive_fill, x)
    result = select(pv, sdfg, state, below, negative_fill, result)
    return select(pv, sdfg, state, ufunc_call(pv, sdfg, state, 'isnan', [x]), dtype.type(nan), result)


@oprepo.replaces('numpy.i0')
def i0(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Any) -> str:
    """``numpy.i0``: the modified Bessel function of the first kind, order 0."""
    if operand_dtype(sdfg, x) in COMPLEX_TYPES:
        raise ValueError('numpy.i0 is not defined for complex input')
    return ufunc_call(pv, sdfg, state, 'i0', [x])


@oprepo.replaces('numpy.sinc')
def sinc(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Any) -> Any:
    """``numpy.sinc``: sin(pi*x)/(pi*x), with NumPy's own tiny-offset standing in for the removable
    singularity at zero so the quotient never divides by it."""
    if isinstance(x, Number):
        return float(np.sinc(x))
    restype = real_result_type(operand_dtype(sdfg, x))
    shifted = select(pv, sdfg, state, ufunc_call(pv, sdfg, state, 'equal', [x, 0]), restype.type(1.0e-20), x)
    scaled = ufunc_call(pv, sdfg, state, 'multiply', [restype.type(np.pi), shifted])
    return ufunc_call(pv, sdfg, state, 'divide', [ufunc_call(pv, sdfg, state, 'sin', [scaled]), scaled])


@oprepo.replaces('numpy.angle')
def angle(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, z: Any, deg: bool = False) -> str:
    """``numpy.angle``: the argument of z, which ``arctan2`` already computes with the right
    quadrant and the right sign on both zeros."""
    dtype = operand_dtype(sdfg, z)
    if dtype in COMPLEX_TYPES:
        parts = [_imag(pv, sdfg, state, z), _real(pv, sdfg, state, z)]
    else:
        parts = [real_result_type(dtype).type(0.0), z]
    result = ufunc_call(pv, sdfg, state, 'arctan2', parts)
    if deg:
        scale = operand_dtype(sdfg, result).type(180.0 / np.pi)
        result = ufunc_call(pv, sdfg, state, 'multiply', [result, scale])
    return result


@oprepo.replaces('numpy.iscomplexobj')
def iscomplexobj(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, x: Any) -> bool:
    """A predicate on the descriptor, not on the values: it resolves while parsing and emits no
    dataflow at all."""
    return operand_dtype(sdfg, x) in COMPLEX_TYPES


@oprepo.replaces('numpy.real_if_close')
def real_if_close(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: Any, tol: Number = 100) -> Any:
    """``numpy.real_if_close``: a non-complex operand is handed back untouched, as NumPy does.

    A complex operand is refused. Whether the result is the real part or the untouched complex
    array depends on the imaginary parts themselves, so its dtype cannot be put on a descriptor.
    """
    if operand_dtype(sdfg, a) not in COMPLEX_TYPES:
        return a
    raise ValueError('numpy.real_if_close on complex input is unsupported: the result dtype is '
                     'data-dependent (real when every imaginary part is within the tolerance, '
                     'complex otherwise)')
