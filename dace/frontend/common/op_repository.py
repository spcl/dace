# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import itertools
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from dace import symbolic
from dace.dtypes import paramdec

MethodType = Callable[..., Tuple[str]]
_INFERENCE_MISSING = object()


def _get_all_bases(class_or_name: Union[str, Type]) -> List[str]:
    """
    Returns a list of the current class name and all its base classes.

    :param class_or_name: A class type or a class name.
    :return: A list of strings representing class names if a type was given, or a list with a single
             string if a string was given. The list is given in reverse order, with subclasses preceding
             superclasses.
    """
    if isinstance(class_or_name, str):
        return [class_or_name]
    return [base.__name__ for base in class_or_name.__mro__]


class Replacements(object):
    """
    A management singleton for functions that replace existing function calls
    with either an SDFG subgraph.
    Used in the Python frontend to replace functions such as `numpy.ndarray`
    and operators such as `Array.__add__`.
    """

    _rep: Dict[str, MethodType] = {}
    _oprep: Dict[Tuple[str, str, str], MethodType] = {}
    _ufunc_rep: Dict[str, MethodType] = {}
    _method_rep: Dict[Tuple[str, str], MethodType] = {}
    _attr_rep: Dict[Tuple[str, str], MethodType] = {}
    _dtype_rep: Dict[str, Callable] = {}  # Lightweight descriptor inference (free functions)
    _dtype_method_rep: Dict[Tuple[str, str], Callable] = {}  # (classname, method) -> fn(self_desc, *a, **kw)
    _dtype_method_self_rep: Dict[Tuple[str, str], Callable] = {}  # (classname, method) -> fn(self_desc, *a, **kw)
    _dtype_attr_rep: Dict[Tuple[str, str], Callable] = {}  # (classname, attr) -> fn(self_desc)
    _dtype_ufunc_rep: Dict[str, Callable] = {}  # ufunc method -> fn(input_descs, ufunc_name, *a, **kw)
    _dtype_op_rep: Dict[Tuple[Optional[str], Optional[str], str], Callable] = {}

    @staticmethod
    def get(name: str):
        """ Returns an implementation of a function. """
        if name not in Replacements._rep:
            return None
        return Replacements._rep[name]

    @staticmethod
    def getop(class_or_name: Union[str, Type], optype: str, otherclass: Union[str, Type, None] = None):
        """ Returns an implementation of an operator. """
        all_op1_types = _get_all_bases(class_or_name)
        if otherclass is None:
            for classname in all_op1_types:
                if (classname, classname, optype) in Replacements._oprep:
                    return Replacements._oprep[(classname, classname, optype)]

            return None

        # If the two classes are defined, try all possible combinations
        all_op2_types = _get_all_bases(otherclass)
        for op1, op2 in itertools.product(all_op1_types, all_op2_types):
            if (op1, op2, optype) in Replacements._oprep:
                return Replacements._oprep[(op1, op2, optype)]

        return None

    @staticmethod
    def get_ufunc(ufunc_method: Optional[str] = None):
        """ Returns the implementation for NumPy universal functions. """
        if ufunc_method:
            if ufunc_method not in Replacements._ufunc_rep:
                return None
            return Replacements._ufunc_rep[ufunc_method]
        return Replacements._ufunc_rep['ufunc']

    @staticmethod
    def get_method(class_or_name: Union[str, Type], method_name: str):
        for classname in _get_all_bases(class_or_name):
            if (classname, method_name) in Replacements._method_rep:
                return Replacements._method_rep[(classname, method_name)]
        return None

    @staticmethod
    def get_attribute(class_or_name: Union[str, Type], attr_name: str):
        for classname in _get_all_bases(class_or_name):
            if (classname, attr_name) in Replacements._attr_rep:
                return Replacements._attr_rep[(classname, attr_name)]
        return None

    @staticmethod
    def get_descriptor_inference(name: str):
        """Returns a lightweight descriptor-inference function for a named call, or None."""
        return Replacements._dtype_rep.get(name, None)

    @staticmethod
    def get_method_descriptor_inference(class_or_name: Union[str, Type], method_name: str):
        """Returns a descriptor-inference function for a method call, or None."""
        for classname in _get_all_bases(class_or_name):
            if (classname, method_name) in Replacements._dtype_method_rep:
                return Replacements._dtype_method_rep[(classname, method_name)]
        return None

    @staticmethod
    def get_method_self_descriptor_inference(class_or_name: Union[str, Type], method_name: str):
        """Returns a self-mutating inference function for a method call, or None."""
        for classname in _get_all_bases(class_or_name):
            if (classname, method_name) in Replacements._dtype_method_self_rep:
                return Replacements._dtype_method_self_rep[(classname, method_name)]
        return None

    @staticmethod
    def get_attribute_descriptor_inference(class_or_name: Union[str, Type], attr_name: str):
        """Returns a descriptor-inference function for an attribute access, or None."""
        for classname in _get_all_bases(class_or_name):
            if (classname, attr_name) in Replacements._dtype_attr_rep:
                return Replacements._dtype_attr_rep[(classname, attr_name)]
        return None

    def get_ufunc_descriptor_inference(ufunc_method: Optional[str] = None):
        """Returns a descriptor-inference function for a NumPy ufunc call or method, or None."""
        key = ufunc_method or 'ufunc'
        return Replacements._dtype_ufunc_rep.get(key, None)

    @staticmethod
    def get_operator_descriptor_inference(optype: str,
                                          left_operand: Any = _INFERENCE_MISSING,
                                          right_operand: Any = _INFERENCE_MISSING):
        """Returns a descriptor-inference function for an operator, or None."""
        if left_operand is _INFERENCE_MISSING and right_operand is _INFERENCE_MISSING:
            return Replacements._dtype_op_rep.get((None, None, optype), None)

        left_types = _get_inference_operand_types(left_operand)
        if right_operand is _INFERENCE_MISSING:
            for left_type in left_types:
                if (left_type, None, optype) in Replacements._dtype_op_rep:
                    return Replacements._dtype_op_rep[(left_type, None, optype)]
            return Replacements._dtype_op_rep.get((None, None, optype), None)

        right_types = _get_inference_operand_types(right_operand)
        for left_type, right_type in itertools.product(left_types, right_types):
            if (left_type, right_type, optype) in Replacements._dtype_op_rep:
                return Replacements._dtype_op_rep[(left_type, right_type, optype)]

        return Replacements._dtype_op_rep.get((None, None, optype), None)


def _get_inference_operand_types(operand: Any) -> List[Optional[str]]:
    if operand is _INFERENCE_MISSING:
        return [None]
    if isinstance(operand, (bool, np.bool_)):
        return ['BoolConstant']
    if isinstance(operand, Number):
        return ['NumConstant']
    if symbolic.issymbolic(operand):
        return ['symbol']
    if isinstance(operand, list):
        return ['ListLiteral']
    if isinstance(operand, tuple):
        return ['TupleLiteral']
    if isinstance(operand, str):
        return ['StringLiteral']
    return _get_all_bases(type(operand))


@paramdec
def replaces(func: Callable[..., Tuple[str]], name: str):
    """ Registers a replacement sub-SDFG generator for a function.

        :param func: A function that receives an SDFG, SDFGState, and the original function
                     arguments, returning a tuple of array names to connect to the outputs.
        :param name: Full name (pydoc-compliant, including package) of function to replace.
    """
    Replacements._rep[name] = func
    return func


@paramdec
def replaces_operator(func: Callable[[Any, Any, str, str], Tuple[str]],
                      classname: str,
                      optype: str,
                      otherclass: str = None):
    """ Registers a replacement sub-SDFG generator for an operator.

        :param func: A function that receives an SDFG, SDFGState, and the two operand array names,
                     returning a tuple of array names to connect to the outputs.
        :param classname: The name of the class to implement the operator for (extends dace.Data).
        :param optype: The type (as string) of the operator to replace (extends ast.operator).
        :param otherclass: Optional argument defining operators for a second class that
                           differs from the first.
    """
    if otherclass is None:
        otherclass = classname
    Replacements._oprep[(classname, otherclass, optype)] = func
    return func


@paramdec
def replaces_ufunc(func: Callable[..., Tuple[str]], name: str):
    """ Registers a replacement sub-SDFG generator for NumPy universal functions
        and methods.

        :param func: A function that receives a ProgramVisitor, AST call node,
                     SDFG, SDFGState, ufunc name, and the original function
                     positional and keyword arguments, returning a tuple of
                     array names to connect to the outputs.
        :param name: 'ufunc' for NumPy ufunc or ufunc method name for replacing
                     the NumPy ufunc methods.
    """
    Replacements._ufunc_rep[name] = func
    return func


@paramdec
def replaces_method(func: Callable[..., Tuple[str]], classname: str, method_name: str):
    """
    Registers a replacement sub-SDFG generator for methods on objects.

    :param func: A function that receives an SDFG, SDFGState, and the original
                 function arguments, returning a tuple of array names to
                 connect to the outputs.
    :param classname: Full name (pydoc-compliant, including package) of the
                      object class.
    :param method_name: Name of the invoked method.
    """
    Replacements._method_rep[(classname, method_name)] = func
    return func


@paramdec
def replaces_attribute(func: Callable[..., Tuple[str]], classname: str, attr_name: str):
    """
    Registers a replacement sub-SDFG generator for object attributes.

    :param func: A function that receives an SDFG, SDFGState, and the original
                 function arguments, returning a tuple of array names to
                 connect to the outputs.
    :param classname: Full name (pydoc-compliant, including package) of the
                      object class.
    :param attr_name: Name of the attribute.
    """
    Replacements._attr_rep[(classname, attr_name)] = func
    return func


@paramdec
def infers_descriptor(func: Callable, name: str):
    """
    Registers a lightweight descriptor-inference function for a named call.

    The function receives ``(input_descriptors, *args, **kwargs)`` where
    *input_descriptors* maps array-argument names to their
    :class:`dace.data.Data` descriptors and the remaining arguments are
    compile-time values (numbers, symbolic expressions, strings, or
    ``None`` when static evaluation failed).  It may return a single
    :class:`dace.data.Data` descriptor, a tuple or list of descriptors
    for structured multi-result calls, or ``None`` if inference is not
    possible. Empty tuples or lists denote a successful zero-output
    inference.

    :param func: The inference function.
    :param name: Fully-qualified function name (e.g. ``'numpy.sum'``).
    """
    Replacements._dtype_rep[name] = func
    return func


@paramdec
def infers_method_descriptor(func: Callable, classname: str, method_name: str):
    """
    Registers descriptor inference for a method call (e.g. ``a.sum()``).

    The function receives ``(self_descriptor, *args, **kwargs)`` where
    *self_descriptor* is the :class:`dace.data.Data` descriptor of the
    object the method is called on.  It may return a single
    :class:`dace.data.Data` descriptor, a tuple or list of descriptors,
    or ``None``.

    :param func: The inference function.
    :param classname: Data-descriptor class name (e.g. ``'Array'``).
    :param method_name: Method name (e.g. ``'sum'``).
    """
    Replacements._dtype_method_rep[(classname, method_name)] = func
    return func


@paramdec
def infers_method_self_descriptor(func: Callable, classname: str, method_name: str):
    """
    Registers descriptor inference for a method call that mutates ``self``.

    The function receives ``(self_descriptor, *args, **kwargs)`` where
    *self_descriptor* is the :class:`dace.data.Data` descriptor of the
    object the method is called on.  It may return a single
    :class:`dace.data.Data` descriptor, a tuple or list of descriptors,
    or ``None``.

    :param func: The inference function.
    :param classname: Data-descriptor class name (e.g. ``'Array'``).
    :param method_name: Method name (e.g. ``'sum'``).
    """
    Replacements._dtype_method_self_rep[(classname, method_name)] = func
    return func


@paramdec
def infers_attribute_descriptor(func: Callable, classname: str, attr_name: str):
    """
    Registers descriptor inference for an attribute access (e.g. ``a.T``).

    The function receives ``(self_descriptor,)`` and returns either a
    single :class:`dace.data.Data` descriptor, a tuple or list of
    descriptors, or ``None``.

    :param func: The inference function.
    :param classname: Data-descriptor class name (e.g. ``'Array'``).
    :param attr_name: Attribute name (e.g. ``'T'``).
    """
    Replacements._dtype_attr_rep[(classname, attr_name)] = func
    return func


@paramdec
def infers_ufunc_descriptor(func: Callable, name: str):
    """
    Registers lightweight descriptor inference for a NumPy ufunc call or ufunc method.

    The function receives ``(input_descriptors, ufunc_name, *args, **kwargs)`` and may return a
    single :class:`dace.data.Data` descriptor, a tuple or list of descriptors, or ``None``.

    :param func: The inference function.
    :param name: ``'ufunc'`` for a direct ufunc call or the ufunc method name, such as ``'reduce'``.
    """
    Replacements._dtype_ufunc_rep[name] = func
    return func


@paramdec
def infers_operator_descriptor(func: Callable,
                               optype: str,
                               classname: Optional[str] = None,
                               otherclass: Optional[str] = None):
    """
    Registers descriptor inference for an operator (e.g. ``A @ B`` or ``-A``).

    The function receives one or more operand descriptors, depending on
    the AST operator form being inferred, and returns a
    :class:`dace.data.Data` descriptor for the result, or ``None``.

    :param func: The inference function.
    :param optype: AST operator name (e.g. ``'MatMult'``).
    :param classname: Optional left operand category name.
    :param otherclass: Optional right operand category name.
    """
    Replacements._dtype_op_rep[(classname, otherclass, optype)] = func
    return func
