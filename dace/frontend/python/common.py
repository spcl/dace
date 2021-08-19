# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from dace import data
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

    def closure_resolver(self, constant_args: Dict[str, Any]) -> 'SDFGClosure':
        """ 
        Returns an SDFGClosure object representing the closure of the
        object to be converted to an SDFG.
        :param constant_args: Arguments whose values are already resolved to
                              compile-time values.
        """
        return SDFGClosure()


@dataclass
class SDFGClosure:
    """
    Represents a reduced closure of a parsed DaCe program.
    A dace.program's closure is composed of its used constants, arrays, and
    other internal SDFG-convertible objects.
    """
    closure_constants: Dict[str, Any]
    closure_arrays: Dict[str, Tuple[str, data.Data]]
    closure_sdfgs: Dict[str, Union[SDFG, SDFGConvertible]]
    nested_closures: Dict[str, 'SDFGClosure']

    # Map same array objects (checked via python id) to the same name
    array_mapping: Dict[int, str]

    def __init__(self):
        self.closure_constants = {}
        self.closure_arrays = {}
        self.closure_sdfgs = {}
        self.nested_closures = {}
        self.array_mapping = {}

    def print_call_tree(self, name, indent=0):
        print('  ' * indent + name)
        for cname, child in self.nested_closures.items():
            child.print_call_tree(cname, indent + 1)

    def call_tree_length(self):
        value = 1
        for child in self.nested_closures.values():
            value += child.call_tree_length()
        return value