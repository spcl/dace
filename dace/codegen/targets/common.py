# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from dace import data, dtypes, sdfg as sd, symbolic
from dace.sdfg import SDFG
from dace.properties import CodeBlock
from dace.codegen import cppunparse
from functools import lru_cache
from typing import List, Optional, Set, Union
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


@lru_cache(maxsize=16384)
def _sym2cpp(s, arrayexprs):
    return cppunparse.pyexpr2cpp(symbolic.symstr(s, arrayexprs))


def sym2cpp(s, arrayexprs: Optional[Set[str]] = None) -> Union[str, List[str]]:
    """ 
    Converts an array of symbolic variables (or one) to C++ strings. 
    :param s: Symbolic expression to convert.
    :param arrayexprs: Set of names of arrays, used to convert SymPy 
                       user-functions back to array expressions.
    :return: C++-compilable expression or list thereof.
    """
    if not isinstance(s, list):
        return _sym2cpp(s, None if arrayexprs is None else frozenset(arrayexprs))
    return [sym2cpp(d, arrayexprs) for d in s]


def codeblock_to_cpp(cb: CodeBlock):
    """
    Converts a CodeBlock object to a C++ string.
    """
    if cb.language == dtypes.Language.CPP:
        return cb.as_string
    elif cb.language == dtypes.Language.Python:
        return cppunparse.py2cpp(cb.code)
    else:
        warnings.warn('Unrecognized language %s in codeblock' % cb.language)
        return cb.as_string


def update_persistent_desc(desc: data.Data, sdfg: SDFG):
    """
    Replaces the symbols used in a persistent data descriptor according to NestedSDFG's symbol mapping.
    The replacement happens recursively up to the top-level SDFG.
    """
    if (desc.lifetime == dtypes.AllocationLifetime.Persistent and sdfg.parent and
            any(str(s) in sdfg.parent_nsdfg_node.symbol_mapping for s in desc.free_symbols)):
        newdesc = deepcopy(desc)
        csdfg = sdfg
        while csdfg.parent_sdfg:
            if any(str(s) not in csdfg.parent_nsdfg_node.symbol_mapping for s in newdesc.free_symbols):
                raise ValueError("Persistent data descriptor depends on symbols defined in NestedSDFG scope.")
            symbolic.safe_replace(csdfg.parent_nsdfg_node.symbol_mapping,
                                  lambda m: sd.replace_properties_dict(newdesc, m))
            csdfg = csdfg.parent_sdfg
        return newdesc
    return desc
