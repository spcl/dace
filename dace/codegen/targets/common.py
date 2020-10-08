# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, symbolic
from dace.sdfg import SDFG
from dace.properties import CodeBlock
from dace.codegen import cppunparse
import warnings


def find_incoming_edges(node, dfg):
    # If it's an entire SDFG, look in each state
    if isinstance(dfg, SDFG):
        result = []
        for state in dfg.nodes():
            result.extend(list(state.in_edges(node)))
        return result
    else:  # If it's one state
        return list(dfg.in_edges(node))


def find_outgoing_edges(node, dfg):
    # If it's an entire SDFG, look in each state
    if isinstance(dfg, SDFG):
        result = []
        for state in dfg.nodes():
            result.extend(list(state.out_edges(node)))
        return result
    else:  # If it's one state
        return list(dfg.out_edges(node))


def sym2cpp(s):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(s, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(s))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d)) for d in s]


def codeblock_to_cpp(cb: CodeBlock):
    """ Converts a CodeBlock object to a C++ string. """
    if cb.language == dtypes.Language.CPP:
        return cb.as_string
    elif cb.language == dtypes.Language.Python:
        return cppunparse.py2cpp(cb.code)
    else:
        warnings.warn('Unrecognized language %s in codeblock' % cb.language)
        return cb.as_string
