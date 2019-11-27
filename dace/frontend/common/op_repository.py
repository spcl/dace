from typing import Any, Callable, Tuple
from dace.dtypes import paramdec


class Replacements(object):
    """ A management singleton for functions that replace existing function calls with either an SDFG or a node.
        Used in the Python frontend to replace functions such as `numpy.ndarray` and operators such
        as `Array.__add__`. """

    _rep = {}
    _oprep = {}

    @staticmethod
    def get(name, implementation='sdfg'):
        """ Returns an implementation of a function. """
        if (name, implementation) not in Replacements._rep:
            return None
        return Replacements._rep[(name, implementation)]

    @staticmethod
    def getop(classname: str,
              optype: str,
              implementation='sdfg',
              otherclass: str = None):
        """ Returns an implementation of an operator. """
        if otherclass is None:
            otherclass = classname
        if (classname, otherclass, optype,
                implementation) not in Replacements._oprep:
            return None
        return Replacements._oprep[(classname, otherclass, optype,
                                    implementation)]


@paramdec
def replaces(func: Callable[..., Tuple[str]], name: str,
             implementation='sdfg'):
    """ Registers a replacement sub-SDFG generator for a function.
        :param func: A function that receives an SDFG, SDFGState, and the original function
                     arguments, returning a tuple of array names to connect to the outputs.
        :param name: Full name (pydoc-compliant, including package) of function to replace.
        :param implementation: The default implementation to replace the SDFG with.
    """
    Replacements._rep[(name, implementation)] = func
    return func


@paramdec
def replaces_operator(func: Callable[[Any, Any, str, str], Tuple[str]],
                      classname: str,
                      optype: str,
                      implementation='sdfg',
                      otherclass: str = None):
    """ Registers a replacement sub-SDFG generator for an operator.
        :param func: A function that receives an SDFG, SDFGState, and the two operand array names,
                     returning a tuple of array names to connect to the outputs.
        :param classname: The name of the class to implement the operator for (extends dace.Data).
        :param optype: The type (as string) of the operator to replace (extends ast.operator).
        :param implementation: The default implementation to replace the SDFG with.
        :param otherclass: Optional argument defining operators for a second class that
                           differs from the first.
    """
    if otherclass is None:
        otherclass = classname
    Replacements._oprep[(classname, otherclass, optype, implementation)] = func
    return func
