# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import math
import operator
import sys
from typing import Optional, Tuple, List, Dict, Union, Type

import numpy as np
from fparser.two.Fortran2003 import Program, Use_Stmt, Rename, Part_Ref, Data_Ref, Intrinsic_Type_Spec, \
    Declaration_Type_Spec, Initialization, Intrinsic_Function_Reference, Int_Literal_Constant, Length_Selector, \
    Kind_Selector, Derived_Type_Def, Type_Name, Real_Literal_Constant, Signed_Real_Literal_Constant, \
    Signed_Int_Literal_Constant, \
    Char_Literal_Constant, Logical_Literal_Constant, Actual_Arg_Spec, Level_2_Unary_Expr, \
    And_Operand, Parenthesis, Level_2_Expr, Level_3_Expr, Array_Constructor, Enumerator_List, Actual_Arg_Spec_List, \
    Only_List, Data_Pointer_Object, Explicit_Shape_Spec, Component_Initialization, Char_Selector, Hex_Constant, \
    Specific_Binding, \
    Generic_Binding, Type_Attr_Spec, Stmt_Function_Stmt, Generic_Spec
from fparser.two.Fortran2008 import Procedure_Stmt
from fparser.two.utils import Base, walk, BinaryOpBase, UnaryOpBase

from . import types
from . import utils

# Namespace for anonymous interfaces
INTERFACE_NAMESPACE = '__interface__'


def ident_spec(node: utils.NAMED_STMTS_OF_INTEREST_TYPES) -> types.SPEC:

    def _ident_spec(_node: utils.NAMED_STMTS_OF_INTEREST_TYPES) -> types.SPEC:
        """
        Construct a list of identifier strings that can uniquely determine it through the entire AST.
        """
        if isinstance(_node, utils.Interface_Stmt):
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
    # A basic check to make sure that it is not on the tail of a data-ref.
    if isinstance(node.parent, (Part_Ref, Data_Ref)):
        cnode, par = node, node.parent
        while par and isinstance(par, (Part_Ref, Data_Ref)):
            if par.children[0] is not cnode:
                return None
            cnode, par = par, par.parent

    scope = utils.find_scope_ancestor(node)
    if not scope:
        return None
    lin = utils.lineage(scope, node)
    assert lin

    par = node.parent
    if (isinstance(scope, Derived_Type_Def) and any(
            isinstance(x, (Explicit_Shape_Spec, Component_Initialization, Kind_Selector, Char_Selector)) for x in lin)):
        # We're using `node` to describe a shape, an initialization etc. inside a type def. So, `node`` must have been
        # defined earlier.
        return search_scope_spec(scope)
    elif isinstance(par, Actual_Arg_Spec):
        kw, _ = par.children
        if kw.string == node.string:
            # We're describing a keyword, which is not really an identifiable object.
            return None
    if isinstance(scope, Stmt_Function_Stmt):
        stmt = scope
    else:
        stmt = utils.atmost_one(utils.children_of_type(scope, utils.NAMED_STMTS_OF_INTEREST_CLASSES))
    if not utils.find_name_of_stmt(stmt):
        # If this is an anonymous object, the scope has to be outside.
        return search_scope_spec(scope.parent)
    return ident_spec(stmt)


def find_scope_spec(node: Base) -> types.SPEC:
    spec = search_scope_spec(node)
    assert spec, f"cannot find scope for: ```\n{node.tofortran()}```"
    return spec


def search_local_alias_spec(node: utils.Name) -> Optional[types.SPEC]:
    name, par = node.string, node.parent
    scope_spec = search_scope_spec(node)
    if scope_spec is None:
        return None
    if isinstance(par, (Part_Ref, Data_Ref, Data_Pointer_Object)):
        # If we are in a data-ref then we need to get to the root.
        while isinstance(par.parent, Data_Ref):
            par = par.parent
        while isinstance(par, (Data_Ref, Part_Ref, Data_Pointer_Object)):
            par, _ = par.children[0], par.children[1:]
        assert isinstance(par, utils.Name)
        if par != node:
            # Components do not really have a local alias.
            return None
    elif isinstance(par, Kind_Selector):
        # Reserved name in this context.
        if name.upper() == 'KIND':
            return None
    elif isinstance(par, Char_Selector):
        # Reserved name in this context.
        if name.upper() in {'KIND', 'LEN'}:
            return None
    elif isinstance(par, Actual_Arg_Spec):
        # Keywords cannot be aliased.
        kw, _ = par.children
        if kw.string == node.string:
            return None
    return scope_spec + (name, )


def search_real_local_alias_spec_from_spec(loc: types.SPEC, alias_map: types.SPEC_TABLE) -> Optional[types.SPEC]:
    while len(loc) > 1 and loc not in alias_map:
        # The name is not immediately available in the current scope, but may be it is in the parent's scope.
        iface_loc = loc[:-2] + (INTERFACE_NAMESPACE, loc[-1])
        if iface_loc in alias_map:
            return iface_loc
        loc = loc[:-2] + (loc[-1], )
    return loc if loc in alias_map else None


def search_real_local_alias_spec(node: utils.Name, alias_map: types.SPEC_TABLE) -> Optional[types.SPEC]:
    loc = search_local_alias_spec(node)
    if not loc:
        return None
    return search_real_local_alias_spec_from_spec(loc, alias_map)


def identifier_specs(ast: Program) -> types.SPEC_TABLE:
    """
    Maps each identifier of interest in `ast` to its associated node that defines it.
    """
    ident_map: types.SPEC_TABLE = {}
    for stmt in walk(ast, utils.NAMED_STMTS_OF_INTEREST_CLASSES):
        assert isinstance(stmt, utils.NAMED_STMTS_OF_INTEREST_CLASSES)
        if isinstance(stmt, utils.Interface_Stmt) and not utils.find_name_of_stmt(stmt):
            # There can be anonymous blocks, e.g., interface blocks, which cannot be identified.
            continue
        spec = ident_spec(stmt)
        if isinstance(stmt, Stmt_Function_Stmt):
            # An exception is statement-functions, which must have a dummy variable already declared in the same scope.
            continue
        assert spec not in ident_map, f"{spec}"
        ident_map[spec] = stmt
    return ident_map


def alias_specs(ast: Program):
    """
    Maps each "alias-type" identifier of interest in `ast` to its associated node that defines it.
    """
    ident_map = identifier_specs(ast)
    alias_map: types.SPEC_TABLE = {k: v for k, v in ident_map.items()}

    for stmt in walk(ast, Use_Stmt):
        mod_name = utils.singular(utils.children_of_type(stmt, utils.Name)).string
        mod_spec = (mod_name, )

        scope_spec = find_scope_spec(stmt)
        use_spec = scope_spec + (mod_name, )

        assert mod_spec in ident_map, mod_spec
        # The module's name cannot be used as an identifier in this scope anymore, so just point to the module.
        alias_map[use_spec] = ident_map[mod_spec]

        olist = utils.atmost_one(utils.children_of_type(stmt, Only_List))
        if not olist:
            # If there is no only list, all the top level (public) symbols are considered aliased.
            alias_updates: types.SPEC_TABLE = {}
            for k, v in alias_map.items():
                if len(k) < len(mod_spec) + 1 or len(k) > len(mod_spec) + 2 or k[:len(mod_spec)] != mod_spec:
                    continue
                if len(k) == len(mod_spec) + 2 and k[len(mod_spec)] != INTERFACE_NAMESPACE:
                    continue
                alias_spec = scope_spec + k[-1:]
                if alias_spec in alias_updates and not isinstance(v, utils.Interface_Stmt):
                    continue
                alias_updates[alias_spec] = v
            alias_map.update(alias_updates)
        else:
            # Otherwise, only specific identifiers are aliased.
            for c in olist.children:
                assert isinstance(c, (utils.Name, Rename, Generic_Spec))
                if isinstance(c, utils.Name):
                    src, tgt = c, c
                elif isinstance(c, Rename):
                    _, src, tgt = c.children
                elif isinstance(c, Generic_Spec):
                    src, tgt = c, c
                src, tgt = f"{src}", f"{tgt}"
                src_spec, tgt_spec = scope_spec + (src, ), mod_spec + (tgt, )
                if mod_spec + (INTERFACE_NAMESPACE, tgt) in alias_map:
                    # If there is an interface and a subroutine of the same name, the interface is selected.
                    tgt_spec = mod_spec + (INTERFACE_NAMESPACE, tgt)
                # `tgt_spec` must have already been resolved if we have sorted the modules properly.
                assert tgt_spec in alias_map, f"{src_spec} => {tgt_spec}"
                alias_map[src_spec] = alias_map[tgt_spec]

    for dt in walk(ast, utils.Derived_Type_Stmt):
        attrs, name, _ = dt.children
        if not attrs:
            continue
        dtspec = ident_spec(dt)
        extends = utils.atmost_one(a.children[1] for a in attrs.children
                                   if isinstance(a, Type_Attr_Spec) and a.children[0] == 'EXTENDS')
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
    spec = search_real_ident_spec(ident, in_spec, alias_map)
    assert spec, f"cannot find {ident} / {in_spec}"
    return spec


def _find_type_decl_node(node: utils.Entity_Decl):
    anc = node.parent
    while anc and not utils.atmost_one(utils.children_of_type(anc, (Intrinsic_Type_Spec, Declaration_Type_Spec))):
        anc = anc.parent
    return anc


# --- Start of Constant Evaluation Logic ---


def _eval_selected_int_kind(p: np.int32) -> int:
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
    # Copied logic from `replace_real_kind()` elsewhere in the project.
    if p >= 9 or r > 126:
        return 8
    elif p >= 3 or r > 14:
        return 4
    return 2


def _cdiv(x, y):
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


def _eval_int_literal(x: Union[Signed_Int_Literal_Constant, Int_Literal_Constant],
                      alias_map: types.SPEC_TABLE) -> types.NUMPY_INTS_TYPES:
    num, kind = x.children
    if kind is None:
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


def _eval_real_literal(x: Union[Signed_Real_Literal_Constant, Real_Literal_Constant],
                       alias_map: types.SPEC_TABLE) -> types.NUMPY_REALS_TYPES:
    num, kind = x.children
    if isinstance(kind, utils.Name):
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
    if isinstance(expr, (Part_Ref, Data_Ref)):
        return None
    elif isinstance(expr, utils.Name):
        spec = search_real_local_alias_spec(expr, alias_map)
        if not spec: return None
        decl = alias_map[spec]
        if not isinstance(decl, utils.Entity_Decl): return None
        typ = find_type_of_entity(decl, alias_map)
        if not typ or not typ.const or typ.shape: return None
        init = utils.atmost_one(utils.children_of_type(decl, Initialization))
        _, iexpr = init.children
        if f"{iexpr}" == 'NULL()': return None
        val = _const_eval_basic_type(iexpr, alias_map)
        assert val is not None
        if typ.spec == ('INTEGER1', ):
            val = np.int8(val)
        elif typ.spec == ('INTEGER2', ):
            val = np.int16(val)
        elif typ.spec in (('INTEGER4', ), ('INTEGER', )):
            val = np.int32(val)
        elif typ.spec == ('INTEGER8', ):
            val = np.int64(val)
        elif typ.spec in (('REAL4', ), ('REAL', )):
            val = np.float32(val)
        elif typ.spec == ('REAL8', ):
            val = np.float64(val)
        elif typ.spec == ('LOGICAL', ):
            val = np.bool_(val)
        else:
            raise ValueError(f"{expr}/{typ.spec} is not a basic type")
        return val
    elif isinstance(expr, Intrinsic_Function_Reference):
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
            return _eval_int_literal(Int_Literal_Constant(f"{int(num)}_{int(kind)}"), alias_map)
        elif intr.string == 'REAL':
            kind = 4
            if len(args) == 2:
                kind = _const_eval_basic_type(args[-1], alias_map)
            num = _const_eval_basic_type(args[0], alias_map)
            if num is None or kind is None: return None
            valstr = str(num)
            if kind == 8:
                valstr = valstr.replace('e', 'D') if 'e' in valstr else f"{valstr}D0"
            return _eval_real_literal(Real_Literal_Constant(valstr), alias_map)
    elif isinstance(expr, (Int_Literal_Constant, Signed_Int_Literal_Constant)):
        return _eval_int_literal(expr, alias_map)
    elif isinstance(expr, Logical_Literal_Constant):
        return np.bool_(expr.tofortran().upper() == '.TRUE.')
    elif isinstance(expr, (Real_Literal_Constant, Signed_Real_Literal_Constant)):
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
    elif isinstance(expr, Parenthesis):
        _, x, _ = expr.children
        return _const_eval_basic_type(x, alias_map)
    elif isinstance(expr, Hex_Constant):
        x = expr.string
        assert f"{x[:2]}{x[-1:]}" in {'Z""', "Z''"}
        x = x[2:-1]
        return np.int64(int(x, 16))

    return None


# --- End of Constant Evaluation Logic ---


def find_type_of_entity(node: Union[utils.Entity_Decl, utils.Component_Decl],
                        alias_map: types.SPEC_TABLE) -> Optional[types.TYPE_SPEC]:
    anc = _find_type_decl_node(node)
    if not anc:
        return None
    node_name, _, _, _ = node.children
    typ, attrs, _ = anc.children
    assert isinstance(typ, (Intrinsic_Type_Spec, Declaration_Type_Spec))
    attrs = attrs.tofortran() if attrs else ''

    extra_dim = None
    if isinstance(typ, Intrinsic_Type_Spec):
        ACCEPTED_TYPES = {'INTEGER', 'REAL', 'DOUBLE PRECISION', 'LOGICAL', 'CHARACTER'}
        typ_name, kind = typ.children
        assert typ_name in ACCEPTED_TYPES, typ_name

        if isinstance(kind, Length_Selector):
            assert typ_name == 'CHARACTER'
            extra_dim = (':', )
        elif isinstance(kind, Kind_Selector):
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
    elif isinstance(typ, Declaration_Type_Spec):
        _, typ_name_node = typ.children
        typ_name = typ_name_node.string if isinstance(typ_name_node, utils.Name) else str(typ_name_node)
        spec = find_real_ident_spec(typ_name, ident_spec(node), alias_map)

    is_arg = False
    scope_spec = find_scope_spec(node)
    assert scope_spec in alias_map
    if isinstance(alias_map[scope_spec], (utils.Function_Stmt, utils.Subroutine_Stmt)):
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


def _dataref_root(dref: Union[utils.Name, Data_Ref, Data_Pointer_Object], scope_spec: types.SPEC,
                  alias_map: types.SPEC_TABLE):
    if isinstance(dref, utils.Name):
        root, rest = dref, []
    else:
        assert len(dref.children) >= 2
        root, rest = dref.children[0], dref.children[1:]
        rest = [r for r in rest if r != '%']

    if isinstance(root, utils.Name):
        root_spec = find_real_ident_spec(root.string, scope_spec, alias_map)
        assert root_spec in alias_map, f"cannot find: {root_spec} / {dref} in {scope_spec}"
        root_type = find_type_of_entity(alias_map[root_spec], alias_map)
    else:
        root_type = find_type_dataref(root, scope_spec, alias_map)
    assert root_type

    return root, root_type, rest


def find_dataref_component_spec(dref: Union[utils.Name, Data_Ref], scope_spec: types.SPEC,
                                alias_map: types.SPEC_TABLE) -> types.SPEC:
    _, root_type, rest = _dataref_root(dref, scope_spec, alias_map)
    cur_type = root_type
    for comp in rest[:-1]:
        part_name = comp.children[0] if isinstance(comp, Part_Ref) else comp
        comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
        assert comp_spec in alias_map, f"cannot find: {comp_spec} / {dref} in {scope_spec}"
        cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
        assert cur_type

    comp = rest[-1]
    part_name = comp.children[0] if isinstance(comp, Part_Ref) else comp
    comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
    assert comp_spec in alias_map, f"cannot find: {comp_spec} / {dref} in {scope_spec}"
    return comp_spec


def find_type_dataref(dref: Union[utils.Name, Part_Ref, Data_Ref, Data_Pointer_Object], scope_spec: types.SPEC,
                      alias_map: types.SPEC_TABLE) -> types.TYPE_SPEC:
    _, root_type, rest = _dataref_root(dref, scope_spec, alias_map)
    cur_type = root_type

    def _subscripted_type(t: types.TYPE_SPEC, pref: Part_Ref):
        pname, subs = pref.children
        if not t.shape:
            assert not subs, f"{t} / {pname}, {t.spec}, {dref}"
        elif subs:
            t.shape = tuple(s.tofortran() for s in subs.children if ':' in s.tofortran())
        return t

    if isinstance(dref, Part_Ref):
        return _subscripted_type(cur_type, dref)
    for comp in rest:
        assert isinstance(comp, (utils.Name, Part_Ref))
        part_name = comp.children[0] if isinstance(comp, Part_Ref) else comp
        comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
        assert comp_spec in alias_map, f"cannot find {comp_spec} / {dref} in {scope_spec}"
        cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
        if isinstance(comp, Part_Ref):
            cur_type = _subscripted_type(cur_type, comp)
        assert cur_type
    return cur_type


def procedure_specs(ast: Program) -> Dict[types.SPEC, types.SPEC]:
    proc_map: Dict[types.SPEC, types.SPEC] = {}
    for pb in walk(ast, Specific_Binding):
        _, _, _, bname, pname = pb.children
        proc_name = bname.string
        subp_name = pname.string if pname else bname.string

        typedef = pb.parent.parent
        typedef_stmt = utils.singular(utils.children_of_type(typedef, utils.Derived_Type_Stmt))
        typedef_name = utils.singular(utils.children_of_type(typedef_stmt, Type_Name)).string

        mod = typedef.parent.parent
        mod_stmt = utils.singular(utils.children_of_type(mod, (utils.Module_Stmt, utils.Program_Stmt)))
        _, mod_name = mod_stmt.children

        proc_spec = (mod_name.string, typedef_name, proc_name)
        subp_spec = (mod_name.string, subp_name)
        proc_map[proc_spec] = subp_spec
    return proc_map


def generic_specs(ast: Program) -> Dict[types.SPEC, Tuple[types.SPEC, ...]]:
    genc_map: Dict[types.SPEC, Tuple[types.SPEC, ...]] = {}
    for gb in walk(ast, Generic_Binding):
        _, bname, plist = gb.children
        plist = plist.children if plist else []
        scope_spec = find_scope_spec(gb)
        genc_spec = scope_spec + (bname.string, )
        proc_specs = [scope_spec + (pname.string, ) for pname in plist]
        genc_map[genc_spec] = tuple(proc_specs)
    return genc_map


def interface_specs(ast: Program, alias_map: types.SPEC_TABLE) -> Dict[types.SPEC, Tuple[types.SPEC, ...]]:
    iface_map: Dict[types.SPEC, Tuple[types.SPEC, ...]] = {}
    for ifs in walk(ast, utils.Interface_Stmt):
        name = utils.find_name_of_stmt(ifs)
        if not name: continue
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        ifspec = ident_spec(ifs)
        fns = []
        for fn in walk(ib, (utils.Function_Stmt, utils.Subroutine_Stmt, Procedure_Stmt)):
            if isinstance(fn, (utils.Function_Stmt, utils.Subroutine_Stmt)):
                fns.append(utils.find_name_of_stmt(fn))
            elif isinstance(fn, Procedure_Stmt):
                fns.extend(nm.string for nm in walk(fn, utils.Name))
        fn_specs = tuple(find_real_ident_spec(f, scope_spec, alias_map) for f in fns)
        assert ifspec not in fn_specs
        iface_map[ifspec] = fn_specs

    for ifs in walk(ast, utils.Interface_Stmt):
        if utils.find_name_of_stmt(ifs): continue
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        assert not walk(ib, Procedure_Stmt)
        for fn in walk(ib, (utils.Function_Stmt, utils.Subroutine_Stmt)):
            fn_name = utils.find_name_of_stmt(fn)
            ifspec = ident_spec(fn)
            fn_impl_spec = search_real_local_alias_spec_from_spec(scope_spec + (fn_name, ), alias_map)
            iface_map[ifspec] = (fn_impl_spec, ) if fn_impl_spec else tuple()
    return iface_map


def _count_bytes(t: Type[types.NUMPY_TYPES]) -> int:
    if t is np.int8: return 1
    if t is np.int16: return 2
    if t is np.int32: return 4
    if t is np.int64: return 8
    if t is np.float32: return 4
    if t is np.float64: return 8
    if t is np.bool_: return 1
    raise ValueError(f"{t} is not an expected type; expected {types.NUMPY_TYPES}")


def _compute_argument_signature(args, scope_spec: types.SPEC,
                                alias_map: types.SPEC_TABLE) -> Tuple[types.TYPE_SPEC, ...]:
    if not args:
        return tuple()

    # Define MATCH_ALL locally as it's only used for signature matching logic
    MATCH_ALL = types.TYPE_SPEC(('*', ), '')

    args_sig = []
    for c in args.children:

        def _deduct_type(x) -> types.TYPE_SPEC:
            if isinstance(x, (Real_Literal_Constant, Signed_Real_Literal_Constant)):
                return types.TYPE_SPEC('REAL')
            elif isinstance(x, (Int_Literal_Constant, Signed_Int_Literal_Constant)):
                val = _eval_int_literal(x, alias_map)
                return types.TYPE_SPEC(f"INTEGER{_count_bytes(type(val))}")
            elif isinstance(x, Char_Literal_Constant):
                return types.TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
            elif isinstance(x, Logical_Literal_Constant):
                return types.TYPE_SPEC('LOGICAL')
            elif isinstance(x, utils.Name):
                x_spec = find_real_ident_spec(x.string, scope_spec, alias_map)
                return find_type_of_entity(alias_map[x_spec], alias_map)
            elif isinstance(x, Data_Ref):
                return find_type_dataref(x, scope_spec, alias_map)
            elif isinstance(x, Part_Ref):
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
            elif isinstance(x, Actual_Arg_Spec):
                kw, val = x.children
                t = _deduct_type(val)
                if isinstance(kw, utils.Name):
                    t.keyword = kw.string
                return t
            elif isinstance(x, Intrinsic_Function_Reference):
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
            elif isinstance(x, (Level_2_Unary_Expr, And_Operand)):
                op, dref = x.children
                if op in {'+', '-', '.NOT.'}: return _deduct_type(dref)
                return MATCH_ALL
            elif isinstance(x, Parenthesis):
                _, exp, _ = x.children
                return _deduct_type(exp)
            elif isinstance(x, (Level_2_Expr, Level_3_Expr)):
                lval, op, rval = x.children
                if op == '+':
                    tl, tr = _deduct_type(lval), _deduct_type(rval)
                    return tr if len(tl.shape) < len(tr.shape) else tl
                if op == '//': return types.TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
                return MATCH_ALL
            elif isinstance(x, Array_Constructor):
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
