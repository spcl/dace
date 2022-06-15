# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to perform find-and-replace of symbols in SDFGs. """

from dace import dtypes, properties, symbolic
from dace.codegen import cppunparse
from dace.frontend.python.astutils import ASTFindReplace
import re
import sympy as sp
from typing import Any, Dict, Optional, Union
import warnings

tokenize_cpp = re.compile(r'\b\w+\b')

def _internal_replace(sym, symrepl):
    if not isinstance(sym, sp.Basic):
        return sym
    
    # Filter out only relevant replacements
    fsyms = map(str, sym.free_symbols)
    newrepl = {k: v for k, v in symrepl.items() if str(k) in fsyms}
    if not newrepl:
        return sym
    
    return sym.subs(newrepl)

def _replsym(symlist, symrepl):
    """ Helper function to replace symbols in various symbolic expressions. """
    if symlist is None:
        return None
    if isinstance(symlist, (symbolic.SymExpr, symbolic.symbol, sp.Basic)):
        return _internal_replace(symlist, symrepl)
    for i, dim in enumerate(symlist):
        try:
            symlist[i] = tuple(_internal_replace(d, symrepl) for d in dim)
        except TypeError:
            symlist[i] = _internal_replace(dim, symrepl)
    return symlist


def replace_dict(subgraph: 'dace.sdfg.state.StateGraphView',
                 repl: Dict[str, str],
                 symrepl: Optional[Dict[symbolic.SymbolicType, symbolic.SymbolicType]] = None):
    """ 
    Finds and replaces all occurrences of a set of symbols/arrays in the given subgraph.
    :param subgraph: The given graph or subgraph to replace in.
    :param repl: Dictionary of replacements (key -> value).
    :param symrepl: Optional cached dictionary of ``repl`` as symbolic expressions.
    """
    symrepl = symrepl or {
        symbolic.pystr_to_symbolic(symname):
        symbolic.pystr_to_symbolic(new_name) if isinstance(new_name, str) else new_name
        for symname, new_name in repl.items()
    }

    # Replace in node properties
    for node in subgraph.nodes():
        replace_properties_dict(node, repl, symrepl)

    # Replace in memlets
    for edge in subgraph.edges():
        if edge.data.data in repl:
            edge.data.data = str(repl[edge.data.data])
        if (edge.data.subset is not None and repl.keys() & edge.data.subset.free_symbols):
            edge.data.subset = _replsym(edge.data.subset, symrepl)
        if (edge.data.other_subset is not None and repl.keys() & edge.data.other_subset.free_symbols):
            edge.data.other_subset = _replsym(edge.data.other_subset, symrepl)
        if symrepl.keys() & edge.data.volume.free_symbols:
            edge.data.volume = _replsym(edge.data.volume, symrepl)


def replace(subgraph: 'dace.sdfg.state.StateGraphView', name: str, new_name: str):
    """
    Finds and replaces all occurrences of a symbol or array in the given subgraph.
    :param subgraph: The given graph or subgraph to replace in.
    :param name: Name to find.
    :param new_name: Name to replace.
    """
    if str(name) == str(new_name):
        return
    replace_dict(subgraph, {name: new_name})


def replace_properties_dict(node: Any,
                            repl: Dict[str, str],
                            symrepl: Dict[symbolic.SymbolicType, symbolic.SymbolicType] = None):
    symrepl = symrepl or {
        symbolic.pystr_to_symbolic(symname):
        symbolic.pystr_to_symbolic(new_name) if isinstance(new_name, str) else new_name
        for symname, new_name in repl.items()
    }

    for propclass, propval in node.properties():
        if propval is None:
            continue
        pname = propclass.attr_name
        if isinstance(propclass, properties.SymbolicProperty):
            setattr(node, pname, propval.subs(symrepl))
        elif isinstance(propclass, properties.DataProperty):
            if propval in repl:
                setattr(node, pname, repl[propval])
        elif isinstance(propclass, (properties.RangeProperty, properties.ShapeProperty)):
            setattr(node, pname, _replsym(list(propval), symrepl))
        elif isinstance(propclass, properties.CodeProperty):
            # Don't replace variables that appear as an input or an output
            # connector, as this should shadow the outer declaration.
            reduced_repl = set(repl.keys())
            if hasattr(node, 'in_connectors'):
                reduced_repl -= set(node.in_connectors.keys()) | set(node.out_connectors.keys())
            reduced_repl = {k: repl[k] for k in reduced_repl}
            code = propval.code
            if isinstance(code, str) and code:
                lang = propval.language
                if lang is dtypes.Language.CPP:  # Replace in C++ code
                    prefix = ''
                    tokenized = tokenize_cpp.findall(code)
                    for name, new_name in reduced_repl.items():
                        if name not in tokenized:
                            continue

                        # Use local variables and shadowing to replace
                        replacement = f'auto {name} = {cppunparse.pyexpr2cpp(new_name)};\n'
                        prefix = replacement + prefix
                    if prefix:
                        propval.code = prefix + code
                else:
                    warnings.warn('Replacement of %s with %s was not made '
                                    'for string tasklet code of language %s' % (name, new_name, lang))

            elif propval.code is not None:
                afr = ASTFindReplace(reduced_repl)
                for stmt in propval.code:
                    afr.visit(stmt)
        elif (isinstance(propclass, properties.DictProperty) and pname == 'symbol_mapping'):
            # Symbol mappings for nested SDFGs
            for symname, sym_mapping in propval.items():
                try:
                    propval[symname] = symbolic.pystr_to_symbolic(str(sym_mapping)).subs(symrepl)
                except AttributeError:  # If the symbolified value has no subs
                    pass


def replace_properties(node: Any, symrepl: Dict[symbolic.symbol, symbolic.SymbolicType], name: str, new_name: str):
    replace_properties_dict(node, {name: new_name}, symrepl)


def replace_datadesc_names(sdfg, repl: Dict[str, str]):
    """ Reduced form of replace which only replaces data descriptor names. """
    from dace.sdfg import SDFG  # Avoid import loop
    sdfg: SDFG = sdfg

    # Replace in descriptor repository
    for aname, aval in list(sdfg.arrays.items()):
        if aname in repl:
            del sdfg.arrays[aname]
            sdfg.arrays[repl[aname]] = aval
            if aname in sdfg.constants_prop:
                sdfg.constants_prop[repl[aname]] = sdfg.constants_prop[aname]
                del sdfg.constants_prop[aname]

    # Replace in interstate edges
    for e in sdfg.edges():
        e.data.replace_dict(repl, replace_keys=False)

    for state in sdfg.nodes():
        # Replace in access nodes
        for node in state.data_nodes():
            if node.data in repl:
                node.data = repl[node.data]

        # Replace in memlets
        for edge in state.edges():
            if edge.data.data in repl:
                edge.data.data = repl[edge.data.data]
