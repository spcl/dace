# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import math
import operator
import sys
from copy import copy
from typing import Optional, Tuple, List, Dict, Union, Set

import fparser.two.Fortran2003 as f03
import fparser.two.Fortran2008 as f08
import numpy as np
from fparser.two.utils import Base, walk, BinaryOpBase, UnaryOpBase

from . import types, utils, optimizations
from .. import ast_utils

# Namespace for anonymous interfaces
INTERFACE_NAMESPACE = '__interface__'


def ident_spec(node: utils.NAMED_STMTS_OF_INTEREST_TYPES) -> types.SPEC:
    """
    Constructs a unique specifier (a tuple of strings) for a named statement node.
    The spec is built by traversing up the AST and collecting the names of parent scopes,
    creating a fully qualified name for the node.

    :param node: The named statement node to generate a spec for.
    :return: A tuple of strings representing the unique specifier.
    """

    def _ident_spec(_node: utils.NAMED_STMTS_OF_INTEREST_TYPES) -> types.SPEC:
        """
        Recursively constructs the spec by prepending the names of ancestor scopes.
        """
        if isinstance(_node, f03.Interface_Stmt):
            ident_base = (INTERFACE_NAMESPACE, utils.find_name_of_stmt(_node))
        else:
            ident_base = (utils.find_name_of_stmt(_node), )
        # Find the next named ancestor.
        anc = utils.find_named_ancestor(_node.parent)
        if not anc:
            return ident_base
        assert isinstance(anc, utils.NAMED_STMTS_OF_INTEREST_CLASSES)
        return _ident_spec(anc) + ident_base

    spec = _ident_spec(node)
    # The last part of the spec cannot be nothing, because we cannot refer to the anonymous blocks.
    assert spec and spec[-1]
    # For the rest, the anonymous blocks puts their content onto their parents.
    spec = tuple(c for c in spec if c)
    return spec


def search_scope_spec(node: Base) -> Optional[types.SPEC]:
    """
    Finds the specifier for the scope that contains the given AST node.

    :param node: The node to find the scope for.
    :return: The spec of the containing scope, or None if not found.
    """
    # A basic check to make sure that it is not on the tail of a data-ref.
    if isinstance(node.parent, (f03.Part_Ref, f03.Data_Ref)):
        cnode, par = node, node.parent
        while par and isinstance(par, (f03.Part_Ref, f03.Data_Ref)):
            if par.children[0] is not cnode:
                return None
            cnode, par = par, par.parent

    scope = utils.find_scope_ancestor(node)
    if not scope:
        return None
    lin = utils.lineage(scope, node)
    assert lin

    par = node.parent
    if (isinstance(scope, f03.Derived_Type_Def) and any(
            isinstance(x, (f03.Explicit_Shape_Spec, f03.Component_Initialization, f03.Kind_Selector, f03.Char_Selector))
            for x in lin)):
        # We're using `node` to describe a shape, an initialization etc. inside a type def. So, `node`` must have been
        # defined earlier.
        return search_scope_spec(scope)
    elif isinstance(par, f03.Actual_Arg_Spec):
        kw, _ = par.children
        if kw.string == node.string:
            # We're describing a keyword, which is not really an identifiable object.
            return None
    if isinstance(scope, f03.Stmt_Function_Stmt):
        stmt = scope
    else:
        stmt = ast_utils.atmost_one(ast_utils.children_of_type(scope, utils.NAMED_STMTS_OF_INTEREST_CLASSES))
    if not utils.find_name_of_stmt(stmt):
        # If this is an anonymous object, the scope has to be outside.
        return search_scope_spec(scope.parent)
    return ident_spec(stmt)


def find_scope_spec(node: Base) -> types.SPEC:
    """
    A wrapper around `search_scope_spec` that asserts a scope is always found.

    :param node: The node to find the scope for.
    :return: The spec of the containing scope.
    """
    spec = search_scope_spec(node)
    assert spec, f"cannot find scope for: ```\n{node.tofortran()}```"
    return spec


def search_local_alias_spec(node: f03.Name) -> Optional[types.SPEC]:
    """
    Constructs a potential alias spec for a name node in its local scope.
    This spec represents how the name would be identified if it were defined in the current scope.

    :param node: The Name node.
    :return: The potential local alias spec, or None if it cannot be formed.
    """
    name, par = node.string, node.parent
    scope_spec = search_scope_spec(node)
    if scope_spec is None:
        return None
    if isinstance(par, (f03.Part_Ref, f03.Data_Ref, f03.Data_Pointer_Object)):
        # If we are in a data-ref then we need to get to the root.
        while isinstance(par.parent, f03.Data_Ref):
            par = par.parent
        while isinstance(par, (f03.Data_Ref, f03.Part_Ref, f03.Data_Pointer_Object)):
            par, _ = par.children[0], par.children[1:]
        assert isinstance(par, f03.Name)
        if par != node:
            # Components do not really have a local alias.
            return None
    elif isinstance(par, f03.Kind_Selector):
        # Reserved name in this context.
        if name.upper() == 'KIND':
            return None
    elif isinstance(par, f03.Char_Selector):
        # Reserved name in this context.
        if name.upper() in {'KIND', 'LEN'}:
            return None
    elif isinstance(par, f03.Actual_Arg_Spec):
        # Keywords cannot be aliased.
        kw, _ = par.children
        if kw.string == node.string:
            return None
    return scope_spec + (name, )


def search_real_local_alias_spec_from_spec(loc: types.SPEC, alias_map: types.SPEC_TABLE) -> Optional[types.SPEC]:
    """
    Given a potential local spec, this function finds the actual spec it maps to in the alias table.
    It searches the current scope and parent scopes.

    :param loc: The local spec to resolve.
    :param alias_map: The table mapping alias specs to their canonical defining nodes.
    :return: The resolved spec from the alias map, or None if not found.
    """
    while len(loc) > 1 and loc not in alias_map:
        # The name is not immediately available in the current scope, but may be it is in the parent's scope.
        iface_loc = loc[:-2] + (INTERFACE_NAMESPACE, loc[-1])
        if iface_loc in alias_map:
            return iface_loc
        loc = loc[:-2] + (loc[-1], )
    return loc if loc in alias_map else None


def search_real_local_alias_spec(node: f03.Name, alias_map: types.SPEC_TABLE) -> Optional[types.SPEC]:
    """
    Finds the canonical spec for a Name node by first finding its local alias spec
    and then resolving it using the alias map.

    :param node: The Name node to resolve.
    :param alias_map: The alias map.
    :return: The resolved canonical spec, or None if not found.
    """
    loc = search_local_alias_spec(node)
    if not loc:
        return None
    return search_real_local_alias_spec_from_spec(loc, alias_map)


def identifier_specs(ast: f03.Program) -> types.SPEC_TABLE:
    """
    Walks the AST and creates an initial mapping from a unique specifier (SPEC) to the
    node that defines it for all named entities of interest.

    :param ast: The root of the fparser AST.
    :return: A dictionary mapping canonical specs to their defining fparser nodes.
    """
    ident_map: types.SPEC_TABLE = {}
    for stmt in walk(ast, utils.NAMED_STMTS_OF_INTEREST_CLASSES):
        assert isinstance(stmt, utils.NAMED_STMTS_OF_INTEREST_CLASSES)
        if isinstance(stmt, f03.Interface_Stmt) and not utils.find_name_of_stmt(stmt):
            # There can be anonymous blocks, e.g., interface blocks, which cannot be identified.
            continue
        spec = ident_spec(stmt)
        if isinstance(stmt, f03.Stmt_Function_Stmt):
            # An exception is statement-functions, which must have a dummy variable already declared in the same scope.
            continue
        assert spec not in ident_map, f"{spec}"
        ident_map[spec] = stmt
    return ident_map


def alias_specs(ast: f03.Program) -> types.SPEC_TABLE:
    """
    Builds a comprehensive map of all identifiers, resolving `USE` statements, `ONLY` clauses,
    renames, and derived type inheritance (`EXTENDS`). This allows any name in any scope to be
    resolved to its original declaration.

    :param ast: The root of the fparser AST.
    :return: A complete alias map from any accessible name spec to its canonical declaration node.
    """
    ident_map = identifier_specs(ast)
    alias_map: types.SPEC_TABLE = {k: v for k, v in ident_map.items()}

    for stmt in walk(ast, f03.Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string
        mod_spec = (mod_name, )

        scope_spec = find_scope_spec(stmt)
        use_spec = scope_spec + (mod_name, )

        assert mod_spec in ident_map, mod_spec
        # The module's name cannot be used as an identifier in this scope anymore, so just point to the module.
        alias_map[use_spec] = ident_map[mod_spec]

        olist = ast_utils.atmost_one(ast_utils.children_of_type(stmt, f03.Only_List))
        if not olist:
            # If there is no only list, all the top level (public) symbols are considered aliased.
            alias_updates: types.SPEC_TABLE = {}
            for k, v in alias_map.items():
                if len(k) < len(mod_spec) + 1 or len(k) > len(mod_spec) + 2 or k[:len(mod_spec)] != mod_spec:
                    continue
                if len(k) == len(mod_spec) + 2 and k[len(mod_spec)] != INTERFACE_NAMESPACE:
                    continue
                alias_spec = scope_spec + k[-1:]
                if alias_spec in alias_updates and not isinstance(v, f03.Interface_Stmt):
                    continue
                alias_updates[alias_spec] = v
            alias_map.update(alias_updates)
        else:
            # Otherwise, only specific identifiers are aliased.
            for c in olist.children:
                assert isinstance(c, (f03.Name, f03.Rename, f03.Generic_Spec))
                if isinstance(c, f03.Name):
                    src, tgt = c, c
                elif isinstance(c, f03.Rename):
                    _, src, tgt = c.children
                elif isinstance(c, f03.Generic_Spec):
                    src, tgt = c, c
                src, tgt = f"{src}", f"{tgt}"
                src_spec, tgt_spec = scope_spec + (src, ), mod_spec + (tgt, )
                if mod_spec + (INTERFACE_NAMESPACE, tgt) in alias_map:
                    # If there is an interface and a subroutine of the same name, the interface is selected.
                    tgt_spec = mod_spec + (INTERFACE_NAMESPACE, tgt)
                # `tgt_spec` must have already been resolved if we have sorted the modules properly.
                assert tgt_spec in alias_map, f"{src_spec} => {tgt_spec}"
                alias_map[src_spec] = alias_map[tgt_spec]

    for dt in walk(ast, f03.Derived_Type_Stmt):
        attrs, name, _ = dt.children
        if not attrs:
            continue
        dtspec = ident_spec(dt)
        extends = ast_utils.atmost_one(a.children[1] for a in attrs.children
                                       if isinstance(a, f03.Type_Attr_Spec) and a.children[0] == 'EXTENDS')
        if not extends:
            continue
        scope_spec = find_scope_spec(dt)
        base_dtspec = find_real_ident_spec(extends.string, scope_spec, alias_map)
        updates = {}
        for k, v in alias_map.items():
            if k[:len(base_dtspec)] != base_dtspec:
                continue
            updates[dtspec + k[len(base_dtspec) - 1:]] = v
        alias_map.update(updates)

    assert set(ident_map.keys()).issubset(alias_map.keys())
    return alias_map


def search_real_ident_spec(ident: str, in_spec: types.SPEC, alias_map: types.SPEC_TABLE) -> Optional[types.SPEC]:
    """
    Searches for the canonical (real) specifier for an identifier string within a given scope.
    It traverses up the scope hierarchy until the identifier is found in the alias map.

    :param ident: The identifier string to search for.
    :param in_spec: The spec of the scope to start the search from.
    :param alias_map: The complete alias map.
    :return: The canonical spec of the identifier, or None if not found.
    """
    k = in_spec + (ident, )
    if k in alias_map:
        return ident_spec(alias_map[k])
    k = in_spec + (INTERFACE_NAMESPACE, ident)
    if k in alias_map:
        return ident_spec(alias_map[k])
    if not in_spec:
        return None
    return search_real_ident_spec(ident, in_spec[:-1], alias_map)


def find_real_ident_spec(ident: str, in_spec: types.SPEC, alias_map: types.SPEC_TABLE) -> types.SPEC:
    """
    A wrapper around `search_real_ident_spec` that asserts an identifier is always found.

    :param ident: The identifier string to search for.
    :param in_spec: The spec of the scope to start the search from.
    :param alias_map: The complete alias map.
    :return: The canonical spec of the identifier.
    """
    spec = search_real_ident_spec(ident, in_spec, alias_map)
    assert spec, f"cannot find {ident} / {in_spec}"
    return spec


def _find_type_decl_node(node: f03.Entity_Decl):
    anc = node.parent
    while anc and not ast_utils.atmost_one(
            ast_utils.children_of_type(anc, (f03.Intrinsic_Type_Spec, f03.Declaration_Type_Spec))):
        anc = anc.parent
    return anc


# --- Start of Constant Evaluation Logic ---
# This section contains functions to evaluate constant expressions at "compile time".
# It handles intrinsic functions, type kinds, and arithmetic/logical operations.


def _eval_selected_int_kind(p: np.int32) -> int:
    """Calculates the integer kind value from a given precision."""
    # Copied logic from `replace_int_kind()` elsewhere in the project.
    # avoid int overflow in numpy 2.0
    p = int(p)
    kind = int(math.ceil((math.log2(10**p) + 1) / 8))
    assert kind <= 8
    if kind <= 2:
        return kind
    elif kind <= 4:
        return 4
    return 8


def _eval_selected_real_kind(p: int, r: int) -> int:
    """Calculates the real kind value from a given precision and range."""
    # Copied logic from `replace_real_kind()` elsewhere in the project.
    if p >= 9 or r > 126:
        return 8
    elif p >= 3 or r > 14:
        return 4
    return 2


def _cdiv(x, y):
    """Performs integer or real division based on operand types."""
    return operator.floordiv(x, y) \
        if (isinstance(x, types.NUMPY_INTS) and isinstance(y, types.NUMPY_INTS)) \
        else operator.truediv(x, y)


UNARY_OPS = {
    '.NOT.': np.logical_not,
    '-': operator.neg,
}

BINARY_OPS = {
    '<': operator.lt,
    '>': operator.gt,
    '==': operator.eq,
    '/=': operator.ne,
    '<=': operator.le,
    '>=': operator.ge,
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': _cdiv,
    '.OR.': np.logical_or,
    '.AND.': np.logical_and,
    '.XOR.': np.logical_xor,
    '**': operator.pow,
}

# A mapping of Fortran intrinsic function names to their numpy equivalents.
# TODO: Bessel, Complex constructors, DiM, ErF, ErFC
INTR_FNS = {
    'ABS': np.fabs,
    'ACOS': np.arccos,
    'AIMAG': np.imag,
    'AINT': np.trunc,
    'AND': np.logical_and,
    'ANINT': np.round,
    'ASIN': np.arcsin,
    'ATAN': np.arctan,
    'ATAN2': np.arctan2,
    'CONJ': np.conj,
    'COS': np.cos,
    'COSH': np.cosh,
    'EXP': np.exp,
    'IMAG': np.imag,
    'IMAGPART': np.imag,
    'LOG': np.log,
    'LOG10': np.log10,
    'MAX': np.max,
    'MIN': np.min,
    'MOD': np.fmod,
    'NOT': np.logical_not,
    'OR': np.logical_or,
    'REAL': np.real,
    'REALPART': np.real,
    'SIGN': np.copysign,
    'SIN': np.sin,
    'SINH': np.sinh,
    'SQRT': np.sqrt,
    'TAN': np.tan,
    'TANH': np.tanh,
    'XOR': np.logical_xor,
}


def _eval_int_literal(x: Union[f03.Signed_Int_Literal_Constant, f03.Int_Literal_Constant],
                      alias_map: types.SPEC_TABLE) -> types.NUMPY_INTS_TYPES:
    """Evaluates an integer literal constant, resolving its kind if specified."""
    num, kind = x.children
    if kind is None:
        kind = 4
    if str(kind) == "c_int":
        kind = 4
    elif str(kind) == "c_double":
        kind = 8
    elif str(kind) == "c_float":
        kind = 4
    elif kind in {'1', '2', '4', '8'}:
        kind = np.int32(kind)
    else:
        kind_spec = search_real_local_alias_spec_from_spec(find_scope_spec(x) + (kind, ), alias_map)
        if kind_spec:
            kind_decl = alias_map[kind_spec]
            kind_node, _, _, _ = kind_decl.children
            kind = _const_eval_basic_type(kind_node, alias_map)
            assert isinstance(kind, np.int32)
    assert kind in {1, 2, 4, 8}
    if kind == 1: return np.int8(num)
    if kind == 2: return np.int16(num)
    if kind == 4: return np.int32(num)
    if kind == 8: return np.int64(num)


def _eval_real_literal(x: Union[f03.Signed_Real_Literal_Constant, f03.Real_Literal_Constant],
                       alias_map: types.SPEC_TABLE) -> types.NUMPY_REALS_TYPES:
    """Evaluates a real literal constant, resolving its kind if specified."""
    num, kind = x.children
    if isinstance(kind, f03.Name):
        kind = kind.string
    if kind is None:
        if 'D' in num.upper():
            num = num.upper().replace('D', 'e')
            kind = 8
        else:
            kind = 4
    else:
        kind_spec = search_real_local_alias_spec_from_spec(find_scope_spec(x) + (kind, ), alias_map)
        if kind_spec:
            kind_decl = alias_map[kind_spec]
            kind_node, _, _, _ = kind_decl.children
            kind = _const_eval_basic_type(kind_node, alias_map)
            assert isinstance(kind, np.int32)
    assert kind in {4, 8}
    if kind == 4: return np.float32(num)
    if kind == 8: return np.float64(num)


def _const_eval_basic_type(expr: Base, alias_map: types.SPEC_TABLE) -> Optional[types.NUMPY_TYPES]:
    """
    Recursively evaluates a constant expression tree.

    :param expr: The fparser expression node.
    :param alias_map: The alias map for resolving symbols.
    :return: The evaluated value as a numpy scalar, or None if the expression is not constant.
    """
    if isinstance(expr, (f03.Part_Ref, f03.Data_Ref)):
        return None
    elif isinstance(expr, f03.Name):
        spec = search_real_local_alias_spec(expr, alias_map)
        if not spec: return None
        decl = alias_map[spec]
        if not isinstance(decl, f03.Entity_Decl): return None
        typ = find_type_of_entity(decl, alias_map)
        if not typ or not typ.const or typ.shape: return None
        init = ast_utils.atmost_one(ast_utils.children_of_type(decl, f03.Initialization))
        _, iexpr = init.children
        if f"{iexpr}" == 'NULL()': return None
        val = _const_eval_basic_type(iexpr, alias_map)
        assert val is not None
        return optimizations._val_2_np_lit(val, typ.spec)
    elif isinstance(expr, f03.Intrinsic_Function_Reference):
        intr, args = expr.children
        args = args.children if args else tuple()
        if intr.string == 'EPSILON':
            a, = args
            a = _const_eval_basic_type(a, alias_map)
            if isinstance(a, types.NUMPY_REALS):
                return type(a)(sys.float_info.epsilon)
        elif intr.string in INTR_FNS:
            avals = tuple(_const_eval_basic_type(a, alias_map) for a in args)
            if all(isinstance(a, types.NUMPY_REALS) for a in avals):
                return INTR_FNS[intr.string](*avals)
        elif intr.string == 'SELECTED_REAL_KIND':
            p, r = args
            p, r = _const_eval_basic_type(p, alias_map), _const_eval_basic_type(r, alias_map)
            if p is None or r is None: return None
            return np.int32(_eval_selected_real_kind(int(p), int(r)))
        elif intr.string == 'SELECTED_INT_KIND':
            p, = args
            p = _const_eval_basic_type(p, alias_map)
            if p is None: return None
            return np.int32(_eval_selected_int_kind(int(p)))
        elif intr.string == 'INT':
            kind = 4
            if len(args) == 2:
                kind = _const_eval_basic_type(args[-1], alias_map)
            num = _const_eval_basic_type(args[0], alias_map)
            if num is None or kind is None: return None
            return _eval_int_literal(f03.Int_Literal_Constant(f"{int(num)}_{int(kind)}"), alias_map)
        elif intr.string == 'REAL':
            kind = 4
            if len(args) == 2:
                kind = _const_eval_basic_type(args[-1], alias_map)
            num = _const_eval_basic_type(args[0], alias_map)
            if num is None or kind is None: return None
            valstr = str(num)
            if kind == 8:
                valstr = valstr.replace('e', 'D') if 'e' in valstr else f"{valstr}D0"
            return _eval_real_literal(f03.Real_Literal_Constant(valstr), alias_map)
    elif isinstance(expr, (f03.Int_Literal_Constant, f03.Signed_Int_Literal_Constant)):
        return _eval_int_literal(expr, alias_map)
    elif isinstance(expr, f03.Logical_Literal_Constant):
        return np.bool_(expr.tofortran().upper() == '.TRUE.')
    elif isinstance(expr, (f03.Real_Literal_Constant, f03.Signed_Real_Literal_Constant)):
        return _eval_real_literal(expr, alias_map)
    elif isinstance(expr, BinaryOpBase):
        lv, op, rv = expr.children
        if op in BINARY_OPS:
            lv = _const_eval_basic_type(lv, alias_map)
            rv = _const_eval_basic_type(rv, alias_map)
            if op == '.AND.' and (lv is np.bool_(False) or rv is np.bool_(False)): return np.bool_(False)
            if op == '.OR.' and (lv is np.bool_(True) or rv is np.bool_(True)): return np.bool_(True)
            if lv is None or rv is None: return None
            return BINARY_OPS[op](lv, rv)
    elif isinstance(expr, UnaryOpBase):
        op, val = expr.children
        if op in UNARY_OPS:
            val = _const_eval_basic_type(val, alias_map)
            if val is None: return None
            return UNARY_OPS[op](val)
    elif isinstance(expr, f03.Parenthesis):
        _, x, _ = expr.children
        return _const_eval_basic_type(x, alias_map)
    elif isinstance(expr, f03.Hex_Constant):
        x = expr.string
        assert f"{x[:2]}{x[-1:]}" in {'Z""', "Z''"}
        x = x[2:-1]
        return np.int64(int(x, 16))

    return None


# --- End of Constant Evaluation Logic ---


def find_type_of_entity(node: Union[f03.Entity_Decl, f03.Component_Decl],
                        alias_map: types.SPEC_TABLE) -> Optional[types.TYPE_SPEC]:
    """
    Determines the type (as a TYPE_SPEC object) of a declared entity or component.

    :param node: The entity or component declaration node.
    :param alias_map: The alias map for resolving type names and kinds.
    :return: A TYPE_SPEC object representing the entity's type, or None if it cannot be determined.
    """
    anc = _find_type_decl_node(node)
    if not anc:
        return None
    node_name, _, _, _ = node.children
    typ, attrs, _ = anc.children
    assert isinstance(typ, (f03.Intrinsic_Type_Spec, f03.Declaration_Type_Spec))
    attrs = attrs.tofortran() if attrs else ''

    extra_dim = None
    if isinstance(typ, f03.Intrinsic_Type_Spec):
        ACCEPTED_TYPES = {'INTEGER', 'REAL', 'DOUBLE PRECISION', 'LOGICAL', 'CHARACTER'}
        typ_name, kind = typ.children
        assert typ_name in ACCEPTED_TYPES, typ_name

        if isinstance(kind, f03.Length_Selector):
            assert typ_name == 'CHARACTER'
            extra_dim = (':', )
        elif isinstance(kind, f03.Kind_Selector):
            assert typ_name in {'INTEGER', 'REAL', 'LOGICAL'}
            _, kind, _ = kind.children
            kind_val = _const_eval_basic_type(kind, alias_map) or 4
            typ_name = f"{typ_name}{int(kind_val)}"
        elif kind is None:
            if typ_name in {'INTEGER', 'REAL'}:
                typ_name = f"{typ_name}4"
            elif typ_name == 'DOUBLE PRECISION':
                typ_name = "REAL8"
        spec = (typ_name, )
    elif isinstance(typ, f03.Declaration_Type_Spec):
        _, typ_name_node = typ.children
        typ_name = typ_name_node.string if isinstance(typ_name_node, f03.Name) else str(typ_name_node)
        spec = find_real_ident_spec(typ_name, ident_spec(node), alias_map)

    is_arg = False
    scope_spec = find_scope_spec(node)
    assert scope_spec in alias_map
    if isinstance(alias_map[scope_spec], (f03.Function_Stmt, f03.Subroutine_Stmt)):
        _, _, dummy_args, _ = alias_map[scope_spec].children
        dummy_args = dummy_args.children if dummy_args else tuple()
        is_arg = any(a.string == node_name.string for a in dummy_args)

    attrs_list = [attrs] if attrs else []
    _, shape, _, _ = node.children
    if shape is not None:
        attrs_list.append(f"DIMENSION({shape.tofortran()})")
    attrs_str = ', '.join(attrs_list)
    tspec = types.TYPE_SPEC(spec, attrs_str, is_arg)
    if extra_dim:
        tspec.shape += extra_dim
    return tspec


def _dataref_root(dref: Union[f03.Name, f03.Data_Ref, f03.Data_Pointer_Object], scope_spec: types.SPEC,
                  alias_map: types.SPEC_TABLE):
    """
    Helper function to deconstruct a data reference (e.g., `a % b % c`) into its root variable
    and the list of subsequent component accesses.

    :param dref: The data reference node.
    :param scope_spec: The scope in which the reference appears.
    :param alias_map: The alias map for resolving the root variable.
    :return: A tuple of (root_node, root_type, list_of_component_nodes).
    """
    if isinstance(dref, f03.Name):
        root, rest = dref, []
    else:
        assert len(dref.children) >= 2
        root, rest = dref.children[0], dref.children[1:]
        rest = [r for r in rest if r != '%']

    if isinstance(root, f03.Name):
        root_spec = find_real_ident_spec(root.string, scope_spec, alias_map)
        assert root_spec in alias_map, f"cannot find: {root_spec} / {dref} in {scope_spec}"
        root_type = find_type_of_entity(alias_map[root_spec], alias_map)
    else:
        root_type = find_type_dataref(root, scope_spec, alias_map)
    assert root_type

    return root, root_type, rest


def find_dataref_component_spec(dref: Union[f03.Name, f03.Data_Ref], scope_spec: types.SPEC,
                                alias_map: types.SPEC_TABLE) -> types.SPEC:
    """
    Finds the canonical spec for the final component in a data reference.
    For `a % b % c`, this would return the spec for `c`.

    :param dref: The data reference node.
    :param scope_spec: The scope of the reference.
    :param alias_map: The alias map.
    :return: The canonical spec of the final component.
    """
    _, root_type, rest = _dataref_root(dref, scope_spec, alias_map)
    cur_type = root_type
    for comp in rest[:-1]:
        part_name = comp.children[0] if isinstance(comp, f03.Part_Ref) else comp
        comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
        assert comp_spec in alias_map, f"cannot find: {comp_spec} / {dref} in {scope_spec}"
        cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
        assert cur_type

    comp = rest[-1]
    part_name = comp.children[0] if isinstance(comp, f03.Part_Ref) else comp
    comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
    assert comp_spec in alias_map, f"cannot find: {comp_spec} / {dref} in {scope_spec}"
    return comp_spec


def find_type_dataref(dref: Union[f03.Name, f03.Part_Ref, f03.Data_Ref, f03.Data_Pointer_Object],
                      scope_spec: types.SPEC, alias_map: types.SPEC_TABLE) -> types.TYPE_SPEC:
    """
    Determines the type of a data reference by traversing its components.
    Handles array slicing by adjusting the shape of the returned type.

    :param dref: The data reference node.
    :param scope_spec: The scope of the reference.
    :param alias_map: The alias map.
    :return: A TYPE_SPEC object for the final component in the data reference.
    """
    _, root_type, rest = _dataref_root(dref, scope_spec, alias_map)
    cur_type = root_type

    def _subscripted_type(t: types.TYPE_SPEC, pref: f03.Part_Ref):
        pname, subs = pref.children
        if not t.shape:
            assert not subs, f"{t} / {pname}, {t.spec}, {dref}"
        elif subs:
            t.shape = tuple(s.tofortran() for s in subs.children if ':' in s.tofortran())
        return t

    if isinstance(dref, f03.Part_Ref):
        return _subscripted_type(cur_type, dref)
    for comp in rest:
        assert isinstance(comp, (f03.Name, f03.Part_Ref))
        part_name = comp.children[0] if isinstance(comp, f03.Part_Ref) else comp
        comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
        assert comp_spec in alias_map, f"cannot find {comp_spec} / {dref} in {scope_spec}"
        cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
        if isinstance(comp, f03.Part_Ref):
            cur_type = _subscripted_type(cur_type, comp)
        assert cur_type
    return cur_type


def procedure_specs(ast: f03.Program) -> Dict[types.SPEC, types.SPEC]:
    """
    Creates a map from a type-bound procedure's spec to the spec of the subroutine/function
    that implements it.

    :param ast: The root of the fparser AST.
    :return: A dictionary mapping binding specs to implementation specs.
    """
    proc_map: Dict[types.SPEC, types.SPEC] = {}
    for pb in walk(ast, f03.Specific_Binding):
        _, _, _, bname, pname = pb.children
        proc_name = bname.string
        subp_name = pname.string if pname else bname.string

        typedef = pb.parent.parent
        typedef_stmt = ast_utils.singular(ast_utils.children_of_type(typedef, f03.Derived_Type_Stmt))
        typedef_name = ast_utils.singular(ast_utils.children_of_type(typedef_stmt, f03.Type_Name)).string

        mod = typedef.parent.parent
        mod_stmt = ast_utils.singular(ast_utils.children_of_type(mod, (f03.Module_Stmt, f03.Program_Stmt)))
        _, mod_name = mod_stmt.children

        proc_spec = (mod_name.string, typedef_name, proc_name)
        subp_spec = (mod_name.string, subp_name)
        proc_map[proc_spec] = subp_spec
    return proc_map


def generic_specs(ast: f03.Program) -> Dict[types.SPEC, Tuple[types.SPEC, ...]]:
    """
    Creates a map from a generic type-bound procedure to the list of specific procedures it resolves to.

    :param ast: The root of the fparser AST.
    :return: A dictionary mapping generic binding specs to a tuple of specific procedure specs.
    """
    genc_map: Dict[types.SPEC, Tuple[types.SPEC, ...]] = {}
    for gb in walk(ast, f03.Generic_Binding):
        _, bname, plist = gb.children
        plist = plist.children if plist else []
        scope_spec = find_scope_spec(gb)
        genc_spec = scope_spec + (bname.string, )
        proc_specs = [scope_spec + (pname.string, ) for pname in plist]
        genc_map[genc_spec] = tuple(proc_specs)
    return genc_map


def interface_specs(ast: f03.Program, alias_map: types.SPEC_TABLE) -> Dict[types.SPEC, Tuple[types.SPEC, ...]]:
    """
    Creates a map from an interface spec to the list of procedures it exposes.
    Handles both named interfaces and anonymous interfaces for individual procedures.

    :param ast: The root of the fparser AST.
    :param alias_map: The alias map for resolving procedure names.
    :return: A dictionary mapping interface specs to a tuple of procedure specs.
    """
    iface_map: Dict[types.SPEC, Tuple[types.SPEC, ...]] = {}
    for ifs in walk(ast, f03.Interface_Stmt):
        name = utils.find_name_of_stmt(ifs)
        if not name: continue
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        ifspec = ident_spec(ifs)
        fns = []
        for fn in walk(ib, (f03.Function_Stmt, f03.Subroutine_Stmt, f03.Procedure_Stmt)):
            if isinstance(fn, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                fns.append(utils.find_name_of_stmt(fn))
            elif isinstance(fn, f03.Procedure_Stmt):
                fns.extend(nm.string for nm in walk(fn, f03.Name))
        fn_specs = tuple(find_real_ident_spec(f, scope_spec, alias_map) for f in fns)
        assert ifspec not in fn_specs
        iface_map[ifspec] = fn_specs

    for ifs in walk(ast, f03.Interface_Stmt):
        if utils.find_name_of_stmt(ifs): continue
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        assert not walk(ib, f03.Procedure_Stmt)
        for fn in walk(ib, (f03.Function_Stmt, f03.Subroutine_Stmt)):
            fn_name = utils.find_name_of_stmt(fn)
            ifspec = ident_spec(fn)
            fn_impl_spec = search_real_local_alias_spec_from_spec(scope_spec + (fn_name, ), alias_map)
            iface_map[ifspec] = (fn_impl_spec, ) if fn_impl_spec else tuple()
    return iface_map


def _compute_argument_signature(args, scope_spec: types.SPEC,
                                alias_map: types.SPEC_TABLE) -> Tuple[types.TYPE_SPEC, ...]:
    if not args:
        return tuple()

    # Define MATCH_ALL locally as it's only used for signature matching logic
    MATCH_ALL = types.TYPE_SPEC(('*', ), '')

    args_sig = []
    for c in args.children:

        def _deduct_type(x) -> types.TYPE_SPEC:
            if isinstance(x, (f03.Real_Literal_Constant, f03.Signed_Real_Literal_Constant)):
                return types.TYPE_SPEC('REAL')
            elif isinstance(x, (f03.Int_Literal_Constant, f03.Signed_Int_Literal_Constant)):
                val = _eval_int_literal(x, alias_map)
                return types.TYPE_SPEC(f"INTEGER{types._count_bytes(type(val))}")
            elif isinstance(x, f03.Char_Literal_Constant):
                return types.TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
            elif isinstance(x, f03.Logical_Literal_Constant):
                return types.TYPE_SPEC('LOGICAL')
            elif isinstance(x, f03.Name):
                x_spec = find_real_ident_spec(x.string, scope_spec, alias_map)
                return find_type_of_entity(alias_map[x_spec], alias_map)
            elif isinstance(x, f03.Data_Ref):
                return find_type_dataref(x, scope_spec, alias_map)
            elif isinstance(x, f03.Part_Ref):
                part_name, subsc = x.children
                orig_type = find_type_dataref(part_name, scope_spec, alias_map)
                if not orig_type.shape:
                    assert not subsc
                    return orig_type
                if not subsc:
                    return orig_type
                subsc_tuple = tuple(s.tofortran() for s in subsc.children if ':' in s.tofortran())
                orig_type.shape = subsc_tuple
                return orig_type
            elif isinstance(x, f03.Actual_Arg_Spec):
                kw, val = x.children
                t = _deduct_type(val)
                if isinstance(kw, f03.Name):
                    t.keyword = kw.string
                return t
            elif isinstance(x, f03.Intrinsic_Function_Reference):
                fname, _args = x.children
                _args = _args.children if _args else tuple()
                if fname.string in {'TRIM'}: return types.TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
                if fname.string in {'SIZE'}: return types.TYPE_SPEC('INTEGER')
                kind = 4
                if len(_args) == 2:
                    kind_val = _const_eval_basic_type(_args[-1], alias_map)
                    if kind_val is not None:
                        kind = int(kind_val)
                if fname.string in {'REAL'}: return types.TYPE_SPEC(f"REAL{kind}")
                if fname.string in {'INT'}: return types.TYPE_SPEC(f"INTEGER{kind}")
                return MATCH_ALL
            elif isinstance(x, (f03.Level_2_Unary_Expr, f03.And_Operand)):
                op, dref = x.children
                if op in {'+', '-', '.NOT.'}: return _deduct_type(dref)
                return MATCH_ALL
            elif isinstance(x, f03.Parenthesis):
                _, exp, _ = x.children
                return _deduct_type(exp)
            elif isinstance(x, (f03.Level_2_Expr, f03.Level_3_Expr)):
                lval, op, rval = x.children
                if op == '+':
                    tl, tr = _deduct_type(lval), _deduct_type(rval)
                    return tr if len(tl.shape) < len(tr.shape) else tl
                if op == '//': return types.TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
                return MATCH_ALL
            elif isinstance(x, f03.Array_Constructor):
                _, items, _ = x.children
                t = _deduct_type(items.children[0]) if items.children else MATCH_ALL
                t.shape += (':', )
                return t
            else:
                return MATCH_ALL

        c_type = _deduct_type(c)
        assert c_type, f"got: {c} / {type(c)}"
        args_sig.append(c_type)
    return tuple(args_sig)


def _compute_candidate_argument_signature(args, cand_spec: types.SPEC,
                                          alias_map: types.SPEC_TABLE) -> Tuple[types.TYPE_SPEC, ...]:
    cand_args_sig: List[types.TYPE_SPEC] = []
    for ca in args:
        ca_decl = alias_map[cand_spec + (ca.string, )]
        ca_type = find_type_of_entity(ca_decl, alias_map)
        ca_type.keyword = ca.string
        assert ca_type, f"got: {ca} / {type(ca)}"
        cand_args_sig.append(ca_type)
    return tuple(cand_args_sig)


def _find_real_ident_spec(node: f03.Name, alias_map: types.SPEC_TABLE) -> types.SPEC:
    loc = search_real_local_alias_spec(node, alias_map)
    assert loc
    return ident_spec(alias_map[loc])


def _lookup_dataref(dr: (f03.Data_Ref, f03.Data_Pointer_Object), alias_map: types.SPEC_TABLE) -> Optional[Tuple[
        f03.Name, types.SPEC]]:
    scope_spec = find_scope_spec(dr)
    root, root_tspec, rest = _dataref_root(dr, scope_spec, alias_map)
    while not isinstance(root, f03.Name):
        root, root_tspec, nurest = _dataref_root(root, scope_spec, alias_map)
        rest = nurest + rest
    return root, tuple(rest)


MATCH_ALL = types.TYPE_SPEC(('*', ), '')  # TODO: Hacky; `_does_type_signature_match()` will match anything with this.


def _does_part_matches(g: types.TYPE_SPEC, c: types.TYPE_SPEC) -> bool:
    if c == MATCH_ALL:
        # Consider them matched.
        return True
    if len(g.shape) != len(c.shape):
        # Both's ranks must match
        return False

    def _real_num_type(t: str) -> Tuple[str, int]:
        if t == 'DOUBLE PRECISION':
            return 'REAL', 8
        elif t == 'REAL':
            return 'REAL', 4
        elif t.startswith('REAL'):
            w = int(t.removeprefix('REAL'))
            return 'REAL', w
        elif t == 'INTEGER':
            return 'INTEGER', 4
        elif t.startswith('INTEGER'):
            w = int(t.removeprefix('INTEGER'))
            return 'INTEGER', w
        return t, 1

    def _subsumes(b: types.SPEC, s: types.SPEC) -> bool:
        """If `b` subsumes `s`."""
        if b == s:
            return True
        if len(b) != 1 or len(s) != 1:
            # TODO: We don't know how to evaluate this?
            return False
        b, s = b[0], s[0]
        b, bw = _real_num_type(b)
        s, sw = _real_num_type(s)
        return b == s and bw >= sw

    return _subsumes(c.spec, g.spec)


def _does_type_signature_match(got_sig: Tuple[types.TYPE_SPEC, ...], cand_sig: Tuple[types.TYPE_SPEC, ...]):
    # Assumptions (Fortran rules):
    # 1. `got_sig` will not have any positional argument after keyworded arguments start.
    # 2. `got_sig` may have keyworded arguments that are actually required arguments, and in different orders.
    # 3. `got_sig` will not have any repeated keywords.

    got_pos, got_kwd = tuple(x for x in got_sig if not x.keyword), {x.keyword: x for x in got_sig if x.keyword}
    if len(got_sig) > len(cand_sig):
        # Cannot have more arguments than needed.
        return False

    cand_pos, cand_kwd = cand_sig[:len(got_pos)], {x.keyword: x for x in cand_sig[len(got_pos):]}
    # Positional arguments are must all match in order.
    for c, g in zip(cand_pos, got_pos):
        if not _does_part_matches(g, c):
            return False
    # Now, we just need to check if `cand_kwd` matches `got_kwd`.

    # All the provided keywords must show up and match in the candidate list.
    for k, g in got_kwd.items():
        if k not in cand_kwd or not _does_part_matches(g, cand_kwd[k]):
            return False
    # All the required candidates must have been provided as keywords.
    for k, c in cand_kwd.items():
        if c.optional:
            continue
        if k not in got_kwd or not _does_part_matches(got_kwd[k], c):
            return False
    return True


def _const_eval_int(expr: Base, alias_map: types.SPEC_TABLE) -> Optional[int]:
    if isinstance(expr, f03.Name):
        scope_spec = find_scope_spec(expr)
        spec = find_real_ident_spec(expr.string, scope_spec, alias_map)
        decl = alias_map[spec]
        assert isinstance(decl, f03.Entity_Decl)
        # TODO: Verify that it is a constant expression.
        init = utils.atmost_one(utils.children_of_type(decl, f03.Initialization))
        # TODO: Add ref.
        _, iexpr = init.children
        return _const_eval_int(iexpr, alias_map)
    elif isinstance(expr, f03.Intrinsic_Function_Reference):
        intr, args = expr.children
        if args:
            args = args.children
        if intr.string == 'SELECTED_REAL_KIND':
            assert len(args) == 2
            p, r = args
            p, r = _const_eval_int(p, alias_map), _const_eval_int(r, alias_map)
            assert p is not None and r is not None
            return _eval_selected_real_kind(p, r)
        elif intr.string == 'SELECTED_INT_KIND':
            assert len(args) == 1
            p, = args
            p = _const_eval_int(p, alias_map)
            assert p is not None
            return _eval_selected_int_kind(p)
    elif isinstance(expr, f03.Int_Literal_Constant):
        return int(expr.tofortran())

    # TODO: Add other evaluations.
    return None


def _track_local_consts(node: Union[Base, List[Base]], alias_map: types.SPEC_TABLE,
                        plus: Optional[Dict[Union[types.SPEC, Tuple[types.SPEC, types.SPEC]], types.LITERAL_TYPES]] = None,
                        minus: Optional[Set[Union[types.SPEC, Tuple[types.SPEC, types.SPEC]]]] = None) \
        -> Tuple[Dict[types.SPEC, types.LITERAL_TYPES], Set[types.SPEC]]:
    plus: Dict[Union[types.SPEC, Tuple[types.SPEC, types.SPEC]], types.LITERAL_TYPES] = copy(plus) if plus else {}
    minus: Set[Union[types.SPEC, Tuple[types.SPEC, types.SPEC]]] = copy(minus) if minus else set()

    def _root_comp(dref: (f03.Data_Ref, f03.Data_Pointer_Object)):
        scope_spec = search_scope_spec(dref)
        assert scope_spec
        if walk(dref, f03.Part_Ref):
            # If we are dealing with any array subscript, we cannot get a "component spec", and should take the
            # pessimistic path.
            # TODO: Handle the `cfg % a(1:5) % b(1:5) % c` type cases better.
            return None
        root, _, _ = _dataref_root(dref, scope_spec, alias_map)
        loc = search_real_local_alias_spec(root, alias_map)
        assert loc
        root_spec = ident_spec(alias_map[loc])
        comp_spec = find_dataref_component_spec(dref, scope_spec, alias_map)
        return root_spec, comp_spec

    def _integrate_subresults(tp: Dict[types.SPEC, types.LITERAL_TYPES], tm: Set[types.SPEC]):
        assert not (tm & tp.keys())
        for k in tm:
            if k in plus:
                del plus[k]
            minus.add(k)
        for k, v in tp.items():
            if k in minus:
                minus.remove(k)
            plus[k] = v

    def _inject_knowns(x: Base, value: bool = True, pointer: bool = True):
        if isinstance(x, (*types.LITERAL_CLASSES, f03.Char_Literal_Constant, f03.Write_Stmt, f03.Close_Stmt,
                          f03.Goto_Stmt, f03.Cycle_Stmt)):
            pass
        elif isinstance(x, f03.Assignment_Stmt):
            lv, op, rv = x.children
            _inject_knowns(lv, value=False, pointer=True)
            _inject_knowns(rv)
        elif isinstance(x, f03.Name):
            loc = search_real_local_alias_spec(x, alias_map)
            if not loc:
                return
            spec = ident_spec(alias_map[loc])
            if spec not in plus:
                return
            assert spec not in minus
            xdecl = alias_map[loc]
            xtyp = find_type_of_entity(xdecl, alias_map) if isinstance(xdecl, f03.Entity_Decl) else None
            if (pointer and xtyp and xtyp.pointer) or value:
                par = x.parent
                utils.replace_node(x, utils.copy_fparser_node(plus[spec]))
                if isinstance(par, (f03.Data_Ref, f03.Part_Ref)):
                    utils.replace_node(par, f03.Data_Ref(par.tofortran()))
        elif isinstance(x, f03.Data_Ref):
            spec = _root_comp(x)
            if spec not in plus:
                for pr in x.children[1:]:
                    if isinstance(pr, f03.Part_Ref):
                        _, subsc = pr.children
                        if subsc:
                            subsc = subsc.children
                        for sc in subsc:
                            _inject_knowns(sc, value, pointer)
                return
            assert spec not in minus
            scope_spec = find_scope_spec(x)
            xtyp = find_type_dataref(x, scope_spec, alias_map)
            if (pointer and xtyp.pointer) or value:
                par = x.parent
                utils.replace_node(x, utils.copy_fparser_node(plus[spec]))
                if isinstance(par, (f03.Data_Ref, f03.Part_Ref)):
                    utils.replace_node(par, f03.Data_Ref(par.tofortran()))
        elif isinstance(x, f03.Part_Ref):
            par, subsc = x.children
            _inject_knowns(par, value=False, pointer=True)
            assert isinstance(subsc, f03.Section_Subscript_List)
            for c in subsc.children:
                _inject_knowns(c)
        elif isinstance(x, f03.Subscript_Triplet):
            for c in x.children:
                if c:
                    _inject_knowns(c)
        elif isinstance(x, f03.Parenthesis):
            _, y, _ = x.children
            _inject_knowns(y)
        elif isinstance(x, UnaryOpBase):
            op, val = x.children
            _inject_knowns(val)
        elif isinstance(x, BinaryOpBase):
            assert not isinstance(x, f03.Assignment_Stmt)
            lv, op, rv = x.children
            _inject_knowns(lv)
            _inject_knowns(rv)
        elif isinstance(x, (f03.Function_Reference, f03.Call_Stmt, f03.Intrinsic_Function_Reference)):
            _, args = x.children
            args = args.children if args else tuple()
            for a in args:
                # TODO: For now, we assume that all arguments are writable.
                if not isinstance(a, f03.Name):
                    _inject_knowns(a)
        elif isinstance(x, f03.Actual_Arg_Spec):
            _, val = x.children
            _inject_knowns(val)
        else:
            raise NotImplementedError(f"cannot handle {x} | {type(x)}")

    if isinstance(node, list):
        for c in node:
            tp, tm = _track_local_consts(c, alias_map, plus, minus)
            _integrate_subresults(tp, tm)
    elif isinstance(node, f03.Execution_Part):
        scpart = ast_utils.atmost_one(ast_utils.children_of_type(node.parent, f03.Specification_Part))
        knowns: Dict[types.SPEC, types.LITERAL_TYPES] = {}
        if scpart:
            for tdcls in scpart.children:
                if not isinstance(tdcls, f03.Type_Declaration_Stmt):
                    continue
                _, _, edcls = tdcls.children
                edcls = edcls.children if edcls else tuple()
                for var in edcls:
                    _, _, _, init = var.children
                    if init:
                        _, init = init.children
                    if init and isinstance(init, types.LITERAL_CLASSES):
                        knowns[ident_spec(var)] = init
        _integrate_subresults(knowns, set())
        for op in node.children:
            # TODO: We wouldn't need the exception handling once we implement for all node types.
            try:
                tp, tm = _track_local_consts(op, alias_map, plus, minus)
                _integrate_subresults(tp, tm)
            except NotImplementedError:
                plus, minus = {}, set()
    elif isinstance(node, f03.Assignment_Stmt):
        lv, op, rv = node.children
        _inject_knowns(lv, value=False, pointer=True)
        _inject_knowns(rv)
        lv, op, rv = node.children
        lspec, ltyp = None, None
        if isinstance(lv, f03.Name):
            loc = search_real_local_alias_spec(lv, alias_map)
            assert loc
            lspec = ident_spec(alias_map[loc])
            if isinstance(alias_map[lspec], f03.Entity_Decl):
                ltyp = find_type_of_entity(alias_map[lspec], alias_map)
        elif isinstance(lv, f03.Data_Ref):
            lspec = _root_comp(lv)
            scope_spec = find_scope_spec(lv)
            ltyp = find_type_dataref(lv, scope_spec, alias_map)
        if lspec and ltyp:
            rval = _const_eval_basic_type(rv, alias_map)
            if rval is None:
                _integrate_subresults({}, {lspec})
            elif not ltyp.shape:
                plus[lspec] = types.numpy_type_to_literal(rval)
                if lspec in minus:
                    minus.remove(lspec)
        tp, tm = _track_local_consts(rv, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, f03.Pointer_Assignment_Stmt):
        lv, _, rv = node.children
        _inject_knowns(rv, value=False, pointer=True)
        lv, _, rv = node.children
        lspec, ltyp = None, None
        if isinstance(lv, f03.Name):
            loc = search_real_local_alias_spec(lv, alias_map)
            assert loc
            lspec = ident_spec(alias_map[loc])
            if isinstance(alias_map[lspec], f03.Entity_Decl):
                ltyp = find_type_of_entity(alias_map[lspec], alias_map)
        elif isinstance(lv, (f03.Data_Ref, f03.Data_Pointer_Object)):
            lspec = _root_comp(lv)
            scope_spec = find_scope_spec(lv)
            ltyp = find_type_dataref(lv, scope_spec, alias_map)
        if lspec and ltyp and ltyp.pointer:
            plus[lspec] = rv
            if lspec in minus:
                minus.remove(lspec)
        tp, tm = _track_local_consts(rv, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, f03.If_Stmt):
        cond, body = node.children
        _inject_knowns(cond)
        _inject_knowns(body)
        cond, body = node.children
        tp, tm = _track_local_consts(cond, alias_map)
        _integrate_subresults(tp, tm)
        tp, tm = _track_local_consts(body, alias_map)
        _integrate_subresults({}, tm | tp.keys())
    elif isinstance(node, f03.If_Construct):
        for c in ast_utils.children_of_type(node, (f03.If_Then_Stmt, f03.Else_If_Stmt)):
            if isinstance(c, f03.If_Then_Stmt):
                cond, = c.children
            elif isinstance(c, f03.Else_If_Stmt):
                cond, _ = c.children
            _inject_knowns(cond)
        assert isinstance(node.children[-1], f03.End_If_Stmt)
        # Split the construct into blocks.
        blocks: List[List[Base]] = []
        for c in node.children[:-1]:
            if isinstance(c, (f03.If_Then_Stmt, f03.Else_If_Stmt, f03.Else_Stmt)):
                # Start a new block.
                blocks.append([])
            else:
                # Add to the running block.
                blocks[-1].append(c)
        if not ast_utils.atmost_one(ast_utils.children_of_type(node, f03.Else_Stmt)):
            # If we don't have an else branch, then assume an empty block, as it negates any knowns from other branches.
            blocks.append([])
        # We add to `tp_net` only if it was fixed to the same value after every block.
        # Otherwise, we add it to `tm_net`.
        tp_net, tm_net = None, set()
        for b in blocks:
            tp, tm = _track_local_consts(b, alias_map, plus, minus)
            if tp_net is None:
                tp_net = tp
            else:
                # The retained subset of known local consts.
                tp_net_nu = {k: tp_net[k] for k in tp_net.keys() & tp.keys() if tp_net[k] == tp[k]}
                # Chuck everything else into unknown bin.
                tm_net.update((tp_net.keys() | tp.keys()) - tp_net_nu.keys())
                tp_net = tp_net_nu
            tm_net.update(tm)
        _integrate_subresults(tp_net, tm_net)
    elif isinstance(node, (f03.Block_Nonlabel_Do_Construct, f03.Block_Label_Do_Construct)):
        do_stmt = node.children[0]
        assert isinstance(do_stmt, (f03.Label_Do_Stmt, f03.Nonlabel_Do_Stmt))
        assert isinstance(node.children[-1], f03.End_Do_Stmt)
        do_ops = node.children[1:-1]
        has_pointer_asgns = bool(walk(node, f03.Pointer_Assignment_Stmt))

        net_tpm = set()
        for op in do_ops:
            tp, tm = _track_local_consts(op, alias_map, {}, set())
            net_tpm.update(tp.keys())
            net_tpm.update(tm)
        loop_control = ast_utils.singular(ast_utils.children_of_type(do_stmt, f03.Loop_Control))
        _, cntexpr, _, _ = loop_control.children
        if cntexpr:
            loopvar, _ = cntexpr
            loopvar_spec = _find_real_ident_spec(loopvar, alias_map)
            net_tpm.add(loopvar_spec)
        for op in do_ops:
            # Inside the do-block don't assume anything is known (a pessimistic, but safe view).
            # TODO: One exception that we make is to resolve the pointers if there is no assignment inside the block.
            #  But this is not entirely a correct operation, since the pointers can get reassigned in a function call.
            if not has_pointer_asgns:
                # We're assuming that all the non-literals in the dictionary are pointers.
                pointers = {k: v for k, v in plus.items() if not isinstance(v, types.LITERAL_CLASSES)}
                tp, tm = _track_local_consts(op, alias_map, pointers, set())
                _integrate_subresults(tp, tm)
            tp, tm = _track_local_consts(op, alias_map, {
                k: v
                for k, v in plus.items() if k not in net_tpm
            }, net_tpm | minus)
            _integrate_subresults(tp, tm)

        _, loop_ctl = do_stmt.children
        _, loop_var, _, _ = loop_ctl.children
        if loop_var:
            loop_var, _ = loop_var
            assert isinstance(loop_var, f03.Name)
            loop_var_spec = search_real_local_alias_spec(loop_var, alias_map)
            assert loop_var_spec
            loop_var_spec = ident_spec(alias_map[loop_var_spec])
            _integrate_subresults({}, {loop_var_spec})
    elif isinstance(
            node, (f03.Name, *types.LITERAL_CLASSES, f03.Char_Literal_Constant, f03.Data_Ref, f03.Part_Ref,
                   f03.Return_Stmt, f03.Write_Stmt, f08.Error_Stop_Stmt, f03.Exit_Stmt, f03.Actual_Arg_Spec,
                   f03.Write_Stmt, f03.Close_Stmt, f03.Goto_Stmt, f03.Continue_Stmt, f03.Format_Stmt, f03.Cycle_Stmt)):
        # These don't modify variables or give any new information.
        pass
    elif isinstance(node, f03.Allocate_Stmt):
        _, allocs, _ = node.children
        allocs = allocs.children if allocs else tuple()
        shape_bounds = []
        for al in allocs:
            _, shape = al.children
            if shape:
                shape_bounds.extend(sb for sb in shape.children)
        for sb in shape_bounds:
            _inject_knowns(sb)
    elif isinstance(node, f03.Deallocate_Stmt):
        # These are not expected to exist in the pruned AST, so don't bother tracking them.
        pass
    elif isinstance(node, UnaryOpBase):
        _inject_knowns(node)
        op, val = node.children
        tp, tm = _track_local_consts(val, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, BinaryOpBase):
        assert not isinstance(node, f03.Assignment_Stmt)
        lv, op, rv = node.children
        _inject_knowns(lv)
        _inject_knowns(rv)
        lv, op, rv = node.children
        tp, tm = _track_local_consts(lv, alias_map)
        _integrate_subresults(tp, tm)
        tp, tm = _track_local_consts(rv, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, f03.Parenthesis):
        _, val, _ = node.children
        tp, tm = _track_local_consts(val, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, (f03.Function_Reference, f03.Call_Stmt, f03.Intrinsic_Function_Reference)):
        # TODO: For now, we assume that all arguments are writable.
        _, args = node.children
        args = args.children if args else tuple()
        for a in args:
            _inject_knowns(a)
        _, args = node.children
        args = args.children if args else tuple()
        for a in args:
            tp, tm = _track_local_consts(a, alias_map)
            _integrate_subresults({}, tm | tp.keys())
    else:
        raise NotImplementedError(f"cannot handle {node} | {type(node)}")

    return plus, minus
