# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from typing import Any, Dict, Optional, Sequence, Tuple
from dace.sdfg.sdfg import SDFG


class DaceSyntaxError(Exception):
    def __init__(self, visitor, node: ast.AST, message: str):
        self.visitor = visitor
        self.node = node
        self.message = message

    def __str__(self):
        # Try to recover line and column
        try:
            line = self.node.lineno
            col = self.node.col_offset
        except AttributeError:
            line = 0
            col = 0

        if self.visitor is not None:
            return (self.message + "\n  in File " + str(self.visitor.filename) +
                    ", line " + str(line) + ":" + str(col))
        else:
            return (self.message + "\n  in line " + str(line) + ":" + str(col))


def inverse_dict_lookup(dict: Dict[str, Any], value: Any):
    """ Finds the first key in a dictionary with the input value. """
    for k, v in dict.items():
        if v == value:
            return k
    return None


class SDFGConvertible(object):
    """ 
    A mixin that defines the interface to annotate SDFG-convertible objects.
    """
    def __sdfg__(self, *args, **kwargs) -> SDFG:
        """
        Returns an SDFG representation of this object.
        :param args: Arguments or argument types (given as DaCe data 
                     descriptors) that can be used for compilation.
        :param kwargs: Keyword arguments or argument types (given as DaCe data 
                       descriptors) that can be used for compilation.
        :return: A parsed SDFG object representing this object.
        """
        raise NotImplementedError

    def __sdfg_closure__(
            self,
            reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """ 
        Returns the closure arrays of the SDFG represented by this object
        as a mapping between array name and the corresponding value.
        :param reevaluate: If given, re-evaluates closure elements based on the
                           input mapping (keys: array names, values: expressions
                           to evaluate). Otherwise, re-evaluates default
                           argument names.
        :return: A dictionary mapping between a name in the closure and the 
                 currently evaluated value.
        """
        raise NotImplementedError

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Returns the SDFG signature represented by this object, as a sequence
        of all argument names that will be found in a call to this object
        (i.e., including regular and constant arguments, but excluding "self"
        for bound methods) and a sequence of the constant argument names from
        the first sequence.
        :return: A 2-tuple of (all arguments, constant arguments).
        """
        raise NotImplementedError
