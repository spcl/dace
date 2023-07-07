# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from dace.dtypes import paramdec, deduplicate

MethodType = Callable[..., Tuple[str]]


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

    classes = [class_or_name.__name__]
    for base in class_or_name.__bases__:
        classes.extend(_get_all_bases(base))

    return deduplicate(classes)


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
