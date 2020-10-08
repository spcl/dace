# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to perform find-and-replace of symbols in SDFGs. """

from dace import dtypes, properties, symbolic
from dace.frontend.python.astutils import ASTFindReplace
import re
import sympy as sp
from typing import Any, Union
import warnings


def _replsym(symlist, symrepl):
    """ Helper function to replace symbols in various symbolic expressions. """
    if symlist is None:
        return None
    if isinstance(symlist, (symbolic.SymExpr, symbolic.symbol, sp.Basic)):
        return symlist.subs(symrepl)
    for i, dim in enumerate(symlist):
        try:
            symlist[i] = tuple(
                d.subs(symrepl) if symbolic.issymbolic(d) else d for d in dim)
        except TypeError:
            symlist[i] = (dim.subs(symrepl)
                          if symbolic.issymbolic(dim) else dim)
    return symlist


def replace(subgraph: 'dace.sdfg.state.StateGraphView', name: str,
            new_name: str):
    """ Finds and replaces all occurrences of a symbol or array in the given
        subgraph.
        :param subgraph: The given graph or subgraph to replace in.
        :param name: Name to find.
        :param new_name: Name to replace.
    """
    symrepl = {
        symbolic.symbol(name):
        symbolic.pystr_to_symbolic(new_name)
        if isinstance(new_name, str) else new_name
    }

    # Replace in node properties
    for node in subgraph.nodes():
        replace_properties(node, name, new_name)

    # Replace in memlets
    for edge in subgraph.edges():
        if edge.data.data == name:
            edge.data.data = new_name
        edge.data.subset = _replsym(edge.data.subset, symrepl)
        edge.data.other_subset = _replsym(edge.data.other_subset, symrepl)
        edge.data.num_accesses = _replsym(edge.data.num_accesses, symrepl)


def replace_properties(node: Any, name: str, new_name: str):
    if str(name) == str(new_name):
        return
    symrepl = {
        symbolic.symbol(name):
        symbolic.pystr_to_symbolic(new_name)
        if isinstance(new_name, str) else new_name
    }

    for propclass, propval in node.properties():
        if propval is None:
            continue
        pname = propclass.attr_name
        if isinstance(propclass, properties.SymbolicProperty):
            setattr(node, pname, propval.subs(symrepl))
        elif isinstance(propclass, properties.DataProperty):
            if propval == name:
                setattr(node, pname, new_name)
        elif isinstance(propclass,
                        (properties.RangeProperty, properties.ShapeProperty)):
            setattr(node, pname, _replsym(list(propval), symrepl))
        elif isinstance(propclass, properties.CodeProperty):
            if isinstance(propval.code, str):
                if str(name) != str(new_name):
                    lang = propval.language
                    newcode = propval.code
                    if not re.findall(r'[^\w]%s[^\w]' % name, newcode):
                        continue

                    if lang is dtypes.Language.CPP:  # Replace in C++ code
                        # Use local variables and shadowing to replace
                        replacement = 'auto %s = %s;\n' % (name, new_name)
                        propval.code = replacement + newcode
                    else:
                        warnings.warn('Replacement of %s with %s was not made '
                                      'for string tasklet code of language %s' %
                                      (name, new_name, lang))
            elif propval.code is not None:
                for stmt in propval.code:
                    ASTFindReplace({name: new_name}).visit(stmt)
        elif (isinstance(propclass, properties.DictProperty)
              and pname == 'symbol_mapping'):
            # Symbol mappings for nested SDFGs
            for symname, sym_mapping in propval.items():
                propval[symname] = sym_mapping.subs(symrepl)
