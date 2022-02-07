# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to perform find-and-replace of symbols in SDFGs. """

from dace import dtypes, properties, symbolic
from dace.frontend.python.astutils import ASTFindReplace
import re
import sympy as sp
from typing import Any, Dict, Union
import warnings


def _replsym(symlist, symrepl):
    """ Helper function to replace symbols in various symbolic expressions. """
    if symlist is None:
        return None
    if isinstance(symlist, (symbolic.SymExpr, symbolic.symbol, sp.Basic)):
        return symlist.subs(symrepl)
    for i, dim in enumerate(symlist):
        try:
            symlist[i] = tuple(d.subs(symrepl) if symbolic.issymbolic(d) else d for d in dim)
        except TypeError:
            symlist[i] = (dim.subs(symrepl) if symbolic.issymbolic(dim) else dim)
    return symlist


def replace(subgraph: 'dace.sdfg.state.StateGraphView', name: str, new_name: str):
    """ Finds and replaces all occurrences of a symbol or array in the given
        subgraph.
        :param subgraph: The given graph or subgraph to replace in.
        :param name: Name to find.
        :param new_name: Name to replace.
    """
    if str(name) == str(new_name):
        return
    symname = symbolic.symbol(name)
    symrepl = {symname: symbolic.pystr_to_symbolic(new_name) if isinstance(new_name, str) else new_name}

    # Replace in node properties
    for node in subgraph.nodes():
        replace_properties(node, symrepl, name, new_name)

    # Replace in memlets
    for edge in subgraph.edges():
        if edge.data.data == name:
            edge.data.data = new_name
        if (edge.data.subset is not None and name in edge.data.subset.free_symbols):
            edge.data.subset = _replsym(edge.data.subset, symrepl)
        if (edge.data.other_subset is not None and name in edge.data.other_subset.free_symbols):
            edge.data.other_subset = _replsym(edge.data.other_subset, symrepl)
        if symname in edge.data.volume.free_symbols:
            edge.data.volume = _replsym(edge.data.volume, symrepl)


def replace_properties(node: Any, symrepl: Dict[symbolic.symbol, symbolic.SymbolicType], name: str, new_name: str):
    for propclass, propval in node.properties():
        if propval is None:
            continue
        pname = propclass.attr_name
        if isinstance(propclass, properties.SymbolicProperty):
            setattr(node, pname, propval.subs(symrepl))
        elif isinstance(propclass, properties.DataProperty):
            if propval == name:
                setattr(node, pname, new_name)
        elif isinstance(propclass, (properties.RangeProperty, properties.ShapeProperty)):
            setattr(node, pname, _replsym(list(propval), symrepl))
        elif isinstance(propclass, properties.CodeProperty):
            # Don't replace variables that appear as an input or an output
            # connector, as this should shadow the outer declaration.
            if hasattr(node, 'in_connectors'):
                if name in node.in_connectors:
                    continue
            if hasattr(node, 'out_connectors'):
                if name in node.out_connectors:
                    continue
            if isinstance(propval.code, str):
                if str(name) != str(new_name):
                    lang = propval.language
                    newcode = propval.code
                    if not re.findall(r'[^\w]%s[^\w]' % name, newcode):
                        continue

                    if lang is dtypes.Language.CPP:  # Replace in C++ code
                        # Avoid import loop
                        from dace.codegen.targets.cpp import sym2cpp

                        # Use local variables and shadowing to replace
                        replacement = 'auto %s = %s;\n' % (name, sym2cpp(new_name))
                        propval.code = replacement + newcode
                    else:
                        warnings.warn('Replacement of %s with %s was not made '
                                      'for string tasklet code of language %s' % (name, new_name, lang))
            elif propval.code is not None:
                afr = ASTFindReplace({name: symbolic.symstr(new_name)})
                for stmt in propval.code:
                    afr.visit(stmt)
        elif (isinstance(propclass, properties.DictProperty) and pname == 'symbol_mapping'):
            # Symbol mappings for nested SDFGs
            for symname, sym_mapping in propval.items():
                try:
                    propval[symname] = symbolic.pystr_to_symbolic(str(sym_mapping)).subs(symrepl)
                except AttributeError:  # If the symbolified value has no subs
                    pass


def replace_properties_dict(node: Any, symrepl: Dict[symbolic.SymbolicType, symbolic.SymbolicType]):
    for k, v in symrepl.items():
        replace_properties(node, {symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v)}, str(k), str(v))


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
        for k, v in repl.items():
            e.data.replace(k, v, replace_keys=False)

    for state in sdfg.nodes():
        # Replace in access nodes
        for node in state.data_nodes():
            if node.data in repl:
                node.data = repl[node.data]

        # Replace in memlets
        for edge in state.edges():
            if edge.data.data in repl:
                edge.data.data = repl[edge.data.data]
