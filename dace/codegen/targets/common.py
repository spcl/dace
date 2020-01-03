from dace import symbolic
from dace.sdfg import SDFG
from dace.codegen import cppunparse


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
