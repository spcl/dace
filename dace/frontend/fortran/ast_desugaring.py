import math
import operator
import re
import sys
from copy import copy
from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Iterable, Set, Type, Any

import networkx as nx
import numpy as np
from fparser.api import get_reader
from fparser.two.Fortran2003 import Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt, \
    Component_Decl, Entity_Decl, Specific_Binding, Generic_Binding, Interface_Stmt, Main_Program, Subroutine_Subprogram, \
    Function_Subprogram, Name, Program, Use_Stmt, Rename, Part_Ref, Data_Ref, Intrinsic_Type_Spec, \
    Declaration_Type_Spec, Initialization, Intrinsic_Function_Reference, Int_Literal_Constant, Length_Selector, \
    Kind_Selector, Derived_Type_Def, Type_Name, Module, Function_Reference, Structure_Constructor, Call_Stmt, \
    Intrinsic_Name, Access_Stmt, Enum_Def, Expr, Enumerator, Real_Literal_Constant, Signed_Real_Literal_Constant, \
    Signed_Int_Literal_Constant, Char_Literal_Constant, Logical_Literal_Constant, Section_Subscript, Actual_Arg_Spec, \
    Level_2_Unary_Expr, And_Operand, Parenthesis, Level_2_Expr, Level_3_Expr, Array_Constructor, Execution_Part, \
    Specification_Part, Interface_Block, Association, Procedure_Designator, Type_Bound_Procedure_Part, \
    Associate_Construct, Subscript_Triplet, End_Function_Stmt, End_Subroutine_Stmt, Module_Subprogram_Part, \
    Enumerator_List, Actual_Arg_Spec_List, Only_List, Dummy_Arg_List, Section_Subscript_List, Char_Selector, \
    Data_Pointer_Object, Explicit_Shape_Spec, Component_Initialization, Subroutine_Body, Function_Body, If_Then_Stmt, \
    Else_If_Stmt, Else_Stmt, If_Construct, Level_4_Expr, Level_5_Expr, Hex_Constant, Add_Operand, Mult_Operand, \
    Assignment_Stmt, Loop_Control, Equivalence_Stmt, If_Stmt, Or_Operand, End_If_Stmt, Save_Stmt, Contains_Stmt, \
    Implicit_Part, Component_Part, End_Module_Stmt, Data_Stmt, Data_Stmt_Set, Data_Stmt_Value, Do_Construct, \
    Block_Nonlabel_Do_Construct, Block_Label_Do_Construct, Label_Do_Stmt, Nonlabel_Do_Stmt, End_Do_Stmt, Return_Stmt, \
    Write_Stmt, Data_Component_Def_Stmt, Exit_Stmt, Allocate_Stmt, Deallocate_Stmt, Close_Stmt, Goto_Stmt, \
    Continue_Stmt, Format_Stmt
from fparser.two.Fortran2008 import Procedure_Stmt, Type_Declaration_Stmt, Error_Stop_Stmt
from fparser.two.utils import Base, walk, BinaryOpBase, UnaryOpBase

from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one

ENTRY_POINT_OBJECT_TYPES = Union[Main_Program, Subroutine_Subprogram, Function_Subprogram]
ENTRY_POINT_OBJECT_CLASSES = (Main_Program, Subroutine_Subprogram, Function_Subprogram)
SCOPE_OBJECT_TYPES = Union[
    Main_Program, Module, Function_Subprogram, Subroutine_Subprogram, Derived_Type_Def, Interface_Block,
    Subroutine_Body, Function_Body]
SCOPE_OBJECT_CLASSES = (
    Main_Program, Module, Function_Subprogram, Subroutine_Subprogram, Derived_Type_Def, Interface_Block,
    Subroutine_Body, Function_Body)
NAMED_STMTS_OF_INTEREST_TYPES = Union[
    Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt, Component_Decl, Entity_Decl,
    Specific_Binding, Generic_Binding, Interface_Stmt]
NAMED_STMTS_OF_INTEREST_CLASSES = (
    Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt, Component_Decl, Entity_Decl,
    Specific_Binding, Generic_Binding, Interface_Stmt)
SPEC = Tuple[str, ...]
SPEC_TABLE = Dict[SPEC, NAMED_STMTS_OF_INTEREST_TYPES]


class TYPE_SPEC:
    NO_ATTRS = ''

    def __init__(self,
                 spec: Union[str, SPEC],
                 attrs: str = NO_ATTRS,
                 is_arg: bool = False):
        if isinstance(spec, str):
            spec = (spec,)
        self.spec: SPEC = spec
        self.shape: Tuple[str, ...] = self._parse_shape(attrs)
        self.optional: bool = 'OPTIONAL' in attrs
        self.inp: bool = 'INTENT(IN)' in attrs or 'INTENT(INOUT)' in attrs
        self.out: bool = 'INTENT(OUT)' in attrs or 'INTENT(INOUT)' in attrs
        self.alloc: bool = 'ALLOCATABLE' in attrs
        self.const: bool = 'PARAMETER' in attrs
        self.keyword: Optional[str] = None
        if is_arg and not self.inp and not self.out:
            self.inp, self.out = True, True

    @staticmethod
    def _parse_shape(attrs: str) -> Tuple[str, ...]:
        if 'DIMENSION' not in attrs:
            return tuple()
        dims: re.Match = re.search(r'DIMENSION\(([^)]*)\)', attrs, re.IGNORECASE)
        assert dims
        dims: str = dims.group(1)
        return tuple(p.strip().lower() for p in dims.split(','))

    def __repr__(self):
        attrs = []
        if self.shape:
            attrs.append(f"shape={self.shape}")
        if self.optional:
            attrs.append("optional")
        if not attrs:
            return f"{self.spec}"
        return f"{self.spec}[{' | '.join(attrs)}]"


def find_name_of_stmt(node: NAMED_STMTS_OF_INTEREST_TYPES) -> Optional[str]:
    """Find the name of the statement if it has one. For anonymous blocks, return `None`."""
    if isinstance(node, Specific_Binding):
        # Ref: https://github.com/stfc/fparser/blob/8c870f84edbf1a24dfbc886e2f7226d1b158d50b/src/fparser/two/Fortran2003.py#L2504
        _, _, _, bname, _ = node.children
        name = bname
    elif isinstance(node, Interface_Stmt):
        name, = node.children
    else:
        # TODO: Test out other type specific ways of finding names.
        name = singular(children_of_type(node, Name))
    if name:
        assert isinstance(name, Name)
        name = name.string
    return name


def find_name_of_node(node: Base) -> Optional[str]:
    """Find the name of the general node if it has one. For anonymous blocks, return `None`."""
    if isinstance(node, NAMED_STMTS_OF_INTEREST_CLASSES):
        return find_name_of_stmt(node)
    stmt = atmost_one(children_of_type(node, NAMED_STMTS_OF_INTEREST_CLASSES))
    if not stmt:
        return None
    return find_name_of_stmt(stmt)


def find_scope_ancestor(node: Base) -> Optional[SCOPE_OBJECT_TYPES]:
    anc = node.parent
    while anc and not isinstance(anc, SCOPE_OBJECT_CLASSES):
        anc = anc.parent
    return anc


def find_named_ancestor(node: Base) -> Optional[NAMED_STMTS_OF_INTEREST_TYPES]:
    anc = find_scope_ancestor(node)
    if not anc:
        return None
    return atmost_one(children_of_type(anc, NAMED_STMTS_OF_INTEREST_CLASSES))


def lineage(anc: Base, des: Base) -> Optional[Tuple[Base, ...]]:
    if anc == des:
        return (anc,)
    if not des.parent:
        return None
    lin = lineage(anc, des.parent)
    if not lin:
        return None
    return lin + (des,)


def search_scope_spec(node: Base) -> Optional[SPEC]:
    scope = find_scope_ancestor(node)
    if not scope:
        return None
    lin = lineage(scope, node)
    assert lin
    par = node.parent
    # TODO: How many other such cases can there be?
    if (isinstance(scope, Derived_Type_Def)
            and any(
                isinstance(x, (Explicit_Shape_Spec, Component_Initialization, Kind_Selector, Char_Selector))
                for x in lin)):
        # We're using `node` to describe a shape, an initialization etc. inside a type def. So, `node`` must have been
        # defined earlier.
        return search_scope_spec(scope)
    elif isinstance(par, Actual_Arg_Spec):
        kw, _ = par.children
        if kw == node:
            # We're describing a keyword, which is not really an identifiable object.
            return None
    stmt = singular(children_of_type(scope, NAMED_STMTS_OF_INTEREST_CLASSES))
    if not find_name_of_stmt(stmt):
        # If this is an anonymous object, the scope has to be outside.
        return search_scope_spec(scope.parent)
    return ident_spec(stmt)


def find_scope_spec(node: Base) -> SPEC:
    spec = search_scope_spec(node)
    assert spec, f"cannot find scope for: ```\n{node.tofortran()}```"
    return spec


def ident_spec(node: NAMED_STMTS_OF_INTEREST_TYPES) -> SPEC:
    def _ident_spec(_node: NAMED_STMTS_OF_INTEREST_TYPES) -> SPEC:
        """
        Constuct a list of identifier strings that can uniquely determine it through the entire AST.
        """
        ident_base = (find_name_of_stmt(_node),)
        # Find the next named ancestor.
        anc = find_named_ancestor(_node.parent)
        if not anc:
            return ident_base
        assert isinstance(anc, NAMED_STMTS_OF_INTEREST_CLASSES)
        return _ident_spec(anc) + ident_base

    spec = _ident_spec(node)
    # The last part of the spec cannot be nothing, because we cannot refer to the anonymous blocks.
    assert spec and spec[-1]
    # For the rest, the anonymous blocks puts their content onto their parents.
    spec = tuple(c for c in spec if c)
    return spec


def search_local_alias_spec(node: Name) -> Optional[SPEC]:
    name, par = node.string, node.parent
    scope_spec = search_scope_spec(node)
    if scope_spec is None:
        return None
    if isinstance(par, (Part_Ref, Data_Ref, Data_Pointer_Object)):
        # If we are in a data-ref then we need to get to the root.
        while isinstance(par.parent, Data_Ref):
            par = par.parent
        while isinstance(par, Data_Ref):
            # TODO: Add ref.
            par, _ = par.children[0], par.children[1:]
        if isinstance(par, (Part_Ref, Data_Pointer_Object)):
            # TODO: Add ref.
            par, _ = par.children[0], par.children[1:]
        assert isinstance(par, Name)
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
        if kw == node:
            return None
    return scope_spec + (name,)


def search_real_local_alias_spec_from_spec(loc: SPEC, alias_map: SPEC_TABLE) -> Optional[SPEC]:
    while len(loc) > 1 and loc not in alias_map:
        # The name is not immediately available in the current scope, but may be it is in the parent's scope.
        loc = loc[:-2] + (loc[-1],)
    return loc if loc in alias_map else None


def search_real_local_alias_spec(node: Name, alias_map: SPEC_TABLE) -> Optional[SPEC]:
    loc = search_local_alias_spec(node)
    if not loc:
        return None
    return search_real_local_alias_spec_from_spec(loc, alias_map)


def identifier_specs(ast: Program) -> SPEC_TABLE:
    """
    Maps each identifier of interest in `ast` to its associated node that defines it.
    """
    ident_map: SPEC_TABLE = {}
    for stmt in walk(ast, NAMED_STMTS_OF_INTEREST_CLASSES):
        assert isinstance(stmt, NAMED_STMTS_OF_INTEREST_CLASSES)
        if isinstance(stmt, Interface_Stmt) and not find_name_of_stmt(stmt):
            # There can be anonymous blocks, e.g., interface blocks, which cannot be identified.
            continue
        spec = ident_spec(stmt)
        assert spec not in ident_map, f"{spec} / {stmt.parent.parent.parent.parent} / {ident_map[spec].parent.parent.parent.parent}"
        ident_map[spec] = stmt
    return ident_map


def alias_specs(ast: Program):
    """
    Maps each "alias-type" identifier of interest in `ast` to its associated node that defines it.
    """
    ident_map = identifier_specs(ast)
    alias_map: SPEC_TABLE = {k: v for k, v in ident_map.items()}

    for stmt in walk(ast, Use_Stmt):
        mod_name = singular(children_of_type(stmt, Name)).string
        mod_spec = (mod_name,)

        scope_spec = find_scope_spec(stmt)
        use_spec = scope_spec + (mod_name,)

        assert mod_spec in ident_map, mod_spec
        # The module's name cannot be used as an identifier in this scope anymore, so just point to the module.
        alias_map[use_spec] = ident_map[mod_spec]

        olist = atmost_one(children_of_type(stmt, Only_List))
        if not olist:
            # If there is no only list, all the top level (public) symbols are considered aliased.
            alias_updates: SPEC_TABLE = {}
            for k, v in alias_map.items():
                if len(k) != len(mod_spec) + 1 or k[:len(mod_spec)] != mod_spec:
                    continue
                alias_spec = scope_spec + k[-1:]
                alias_updates[alias_spec] = v
            alias_map.update(alias_updates)
        else:
            # Otherwise, only specific identifiers are aliased.
            for c in olist.children:
                assert isinstance(c, (Name, Rename))
                if isinstance(c, Name):
                    src, tgt = c, c
                elif isinstance(c, Rename):
                    _, src, tgt = c.children
                src, tgt = src.string, tgt.string
                src_spec, tgt_spec = scope_spec + (src,), mod_spec + (tgt,)
                # `tgt_spec` must have already been resolved if we have sorted the modules properly.
                assert tgt_spec in alias_map, f"{src_spec} => {tgt_spec}"
                alias_map[src_spec] = alias_map[tgt_spec]

    assert set(ident_map.keys()).issubset(alias_map.keys())
    return alias_map


def search_real_ident_spec(ident: str, in_spec: SPEC, alias_map: SPEC_TABLE) -> Optional[SPEC]:
    k = in_spec + (ident,)
    if k in alias_map:
        return ident_spec(alias_map[k])
    if not in_spec:
        return None
    return search_real_ident_spec(ident, in_spec[:-1], alias_map)


def find_real_ident_spec(ident: str, in_spec: SPEC, alias_map: SPEC_TABLE) -> SPEC:
    spec = search_real_ident_spec(ident, in_spec, alias_map)
    assert spec, f"cannot find {ident} / {in_spec}"
    return spec


def _find_type_decl_node(node: Entity_Decl):
    anc = node.parent
    while anc and not atmost_one(
            children_of_type(anc, (Intrinsic_Type_Spec, Declaration_Type_Spec))):
        anc = anc.parent
    return anc


def _eval_selected_int_kind(p: np.int32) -> int:
    # Copied logic from `replace_int_kind()` elsewhere in the project.
    # avoid int overflow in numpy 2.0
    p = int(p)
    kind = int(math.ceil((math.log2(10 ** p) + 1) / 8))
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


def _const_eval_int(expr: Base, alias_map: SPEC_TABLE) -> Optional[int]:
    if isinstance(expr, Name):
        scope_spec = find_scope_spec(expr)
        spec = find_real_ident_spec(expr.string, scope_spec, alias_map)
        decl = alias_map[spec]
        assert isinstance(decl, Entity_Decl)
        # TODO: Verify that it is a constant expression.
        init = atmost_one(children_of_type(decl, Initialization))
        # TODO: Add ref.
        _, iexpr = init.children
        return _const_eval_int(iexpr, alias_map)
    elif isinstance(expr, Intrinsic_Function_Reference):
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
    elif isinstance(expr, Int_Literal_Constant):
        return int(expr.tofortran())

    # TODO: Add other evaluations.
    return None


def _cdiv(x, y):
    return operator.floordiv(x, y) \
        if (isinstance(x, (np.int8, np.int16, np.int32, np.int64))
            and isinstance(y, (np.int8, np.int16, np.int32, np.int64))) \
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
    '**': operator.pow,
}

NUMPY_INTS_TYPES = Union[np.int8, np.int16, np.int32, np.int64]
NUMPY_INTS = (np.int8, np.int16, np.int32, np.int64)
NUMPY_REALS = (np.float32, np.float64)
NUMPY_REALS_TYPES = Union[np.float32, np.float64]
NUMPY_TYPES = Union[NUMPY_INTS_TYPES, NUMPY_REALS_TYPES, np.bool_]


def _count_bytes(t: Type[NUMPY_TYPES]) -> int:
    if t is np.int8:
        return 1
    elif t is np.int16:
        return 2
    elif t is np.int32:
        return 4
    elif t is np.int64:
        return 8
    elif t is np.float32:
        return 4
    elif t is np.float64:
        return 8
    elif t is np.bool_:
        return 1
    raise ValueError(f"{t} is not an expected type; expected {NUMPY_TYPES}")


def _eval_int_literal(x: Union[Signed_Int_Literal_Constant, Int_Literal_Constant],
                      alias_map: SPEC_TABLE) -> NUMPY_INTS_TYPES:
    num, kind = x.children
    if kind is None:
        kind = 4
    elif kind in {'1', '2', '4', '8'}:
        kind = np.int32(kind)
    else:
        kind_spec = search_real_local_alias_spec_from_spec(find_scope_spec(x) + (kind,), alias_map)
        if kind_spec:
            kind_decl = alias_map[kind_spec]
            kind_node, _, _, _ = kind_decl.children
            kind = _const_eval_basic_type(kind_node, alias_map)
            assert isinstance(kind, np.int32)
    assert kind in {1, 2, 4, 8}
    if kind == 1:
        return np.int8(num)
    elif kind == 2:
        return np.int16(num)
    elif kind == 4:
        return np.int32(num)
    elif kind == 8:
        return np.int64(num)


def _eval_real_literal(x: Union[Signed_Real_Literal_Constant, Real_Literal_Constant],
                       alias_map: SPEC_TABLE) -> NUMPY_REALS_TYPES:
    num, kind = x.children
    if kind is None:
        if 'D' in num:
            num = num.replace('D', 'e')
            kind = 8
        else:
            kind = 4
    else:
        kind_spec = search_real_local_alias_spec_from_spec(find_scope_spec(x) + (kind,), alias_map)
        if kind_spec:
            kind_decl = alias_map[kind_spec]
            kind_node, _, _, _ = kind_decl.children
            kind = _const_eval_basic_type(kind_node, alias_map)
            assert isinstance(kind, np.int32)
    assert kind in {4, 8}
    if kind == 4:
        return np.float32(num)
    elif kind == 8:
        return np.float64(num)


def _const_eval_basic_type(expr: Base, alias_map: SPEC_TABLE) -> Optional[NUMPY_TYPES]:
    if isinstance(expr, (Part_Ref, Data_Ref)):
        return None
    elif isinstance(expr, Name):
        spec = search_real_local_alias_spec(expr, alias_map)
        if not spec:
            # Does not even have a valid identifier.
            return None
        decl = alias_map[spec]
        if not isinstance(decl, Entity_Decl):
            # Is not even a data entity.
            return None
        typ = find_type_of_entity(decl, alias_map)
        if not typ or not typ.const or typ.shape:
            # Does not have a constant type.
            return None
        init = atmost_one(children_of_type(decl, Initialization))
        # TODO: Add ref.
        _, iexpr = init.children
        val = _const_eval_basic_type(iexpr, alias_map)
        assert val is not None
        if typ.spec == ('INTEGER1',):
            val = np.int8(val)
        elif typ.spec == ('INTEGER2',):
            val = np.int16(val)
        elif typ.spec == ('INTEGER4',) or typ.spec == ('INTEGER',):
            val = np.int32(val)
        elif typ.spec == ('INTEGER8',):
            val = np.int64(val)
        elif typ.spec == ('REAL4',) or typ.spec == ('REAL',):
            val = np.float32(val)
        elif typ.spec == ('REAL8',):
            val = np.float64(val)
        elif typ.spec == ('LOGICAL',):
            val = np.bool_(val)
        else:
            raise ValueError(f"{expr}/{typ.spec} is not a basic type")
        return val
    elif isinstance(expr, Intrinsic_Function_Reference):
        intr, args = expr.children
        if args:
            args = args.children
        if intr.string == 'EPSILON':
            a, = args
            a = _const_eval_basic_type(a, alias_map)
            assert isinstance(a, (np.float32, np.float64))
            return type(a)(sys.float_info.epsilon)
        elif intr.string == 'SELECTED_REAL_KIND':
            p, r = args
            p, r = _const_eval_basic_type(p, alias_map), _const_eval_basic_type(r, alias_map)
            assert isinstance(p, np.int32) and isinstance(r, np.int32)
            return np.int32(_eval_selected_real_kind(p, r))
        elif intr.string == 'SELECTED_INT_KIND':
            p, = args
            p = _const_eval_basic_type(p, alias_map)
            assert isinstance(p, np.int32)
            return np.int32(_eval_selected_int_kind(p))
        elif intr.string == 'INT':
            if len(args) == 1:
                num, = args
                kind = 4
            else:
                num, kind = args
                kind = _const_eval_basic_type(kind, alias_map)
                assert kind is not None
            num = _const_eval_basic_type(num, alias_map)
            if not num:
                return None
            return _eval_int_literal(Int_Literal_Constant(f"{num}_{kind}"), alias_map)
        elif intr.string == 'REAL':
            if len(args) == 1:
                num, = args
                kind = 4
            else:
                num, kind = args
                kind = _const_eval_basic_type(kind, alias_map)
                assert kind is not None
            num = _const_eval_basic_type(num, alias_map)
            if not num:
                return None
            valstr = str(num)
            if kind == 8:
                if 'e' in valstr:
                    valstr = valstr.replace('e', 'D')
                else:
                    valstr = f"{valstr}D0"
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
            if op == '.AND.' and (lv is np.bool_(False) or rv is np.bool_(False)):
                return np.bool_(False)
            elif op == '.OR.' and (lv is np.bool_(True) or rv is np.bool_(True)):
                return np.bool_(True)
            elif lv is None or rv is None:
                return None
            return BINARY_OPS[op](lv, rv)
    elif isinstance(expr, UnaryOpBase):
        op, val = expr.children
        if op in UNARY_OPS:
            val = _const_eval_basic_type(val, alias_map)
            if val is None:
                return None
            return UNARY_OPS[op](val)
    elif isinstance(expr, Parenthesis):
        _, x, _ = expr.children
        return _const_eval_basic_type(x, alias_map)
    elif isinstance(expr, Hex_Constant):
        x = expr.string
        assert x[:2] == 'Z"' and x[-1:] == '"'
        x = x[2:-1]
        return np.int32(int(x, 16))

    # TODO: Add other evaluations.
    return None


def find_type_of_entity(node: Entity_Decl, alias_map: SPEC_TABLE) -> Optional[TYPE_SPEC]:
    anc = _find_type_decl_node(node)
    if not anc:
        return None
    # TODO: Add ref.
    node_name, _, _, _ = node.children
    typ, attrs, _ = anc.children
    assert isinstance(typ, (Intrinsic_Type_Spec, Declaration_Type_Spec))
    attrs = attrs.tofortran() if attrs else ''

    extra_dim = None
    if isinstance(typ, Intrinsic_Type_Spec):
        ACCEPTED_TYPES = {'INTEGER', 'REAL', 'DOUBLE PRECISION', 'LOGICAL', 'CHARACTER'}
        typ_name, kind = typ.children
        assert typ_name in ACCEPTED_TYPES, typ_name

        # TODO: How should we handle character lengths? Just treat it as an extra dimension?
        if isinstance(kind, Length_Selector):
            assert typ_name == 'CHARACTER'
            extra_dim = (':',)
        elif isinstance(kind, Kind_Selector):
            assert typ_name in {'INTEGER', 'REAL', 'LOGICAL'}
            _, kind, _ = kind.children
            kind = _const_eval_basic_type(kind, alias_map) or 4
            typ_name = f"{typ_name}{kind}"
        elif kind is None:
            if typ_name in {'INTEGER', 'REAL'}:
                typ_name = f"{typ_name}4"
            elif typ_name in {'DOUBLE PRECISION'}:
                typ_name = f"REAL8"
        spec = (typ_name,)
    elif isinstance(typ, Declaration_Type_Spec):
        _, typ_name = typ.children
        spec = find_real_ident_spec(typ_name.string, ident_spec(node), alias_map)

    is_arg = False
    scope_spec = find_scope_spec(node)
    assert scope_spec in alias_map
    if isinstance(alias_map[scope_spec], (Function_Stmt, Subroutine_Stmt)):
        _, fn, dummy_args, _ = alias_map[scope_spec].children
        dummy_args = dummy_args.children if dummy_args else tuple()
        is_arg = any(a == node_name for a in dummy_args)

    # TODO: This `attrs` manipulation is a hack. We should design the type specs better.
    # TODO: Add ref.
    attrs = [attrs] if attrs else []
    _, shape, _, _ = node.children
    if shape is not None:
        attrs.append(f"DIMENSION({shape.tofortran()})")
    attrs = ', '.join(attrs)
    tspec = TYPE_SPEC(spec, attrs, is_arg)
    if extra_dim:
        tspec.shape += extra_dim
    return tspec


def _dataref_root(dref: Union[Name, Data_Ref], scope_spec: SPEC, alias_map: SPEC_TABLE):
    if isinstance(dref, Name):
        root, rest = dref, []
    else:
        assert len(dref.children) >= 2
        root, rest = dref.children[0], dref.children[1:]

    if isinstance(root, Name):
        root_spec = find_real_ident_spec(root.string, scope_spec, alias_map)
        assert root_spec in alias_map, f"canont find: {root_spec} / {dref} in {scope_spec}"
        root_type = find_type_of_entity(alias_map[root_spec], alias_map)
    elif isinstance(root, Data_Ref):
        root_type = find_type_dataref(root, scope_spec, alias_map)
    elif isinstance(root, Part_Ref):
        root_type = find_type_dataref(root, scope_spec, alias_map)
    assert root_type

    return root, root_type, rest


def find_dataref_component_spec(dref: Union[Name, Data_Ref], scope_spec: SPEC, alias_map: SPEC_TABLE) -> SPEC:
    # The root must have been a typed object.
    _, root_type, rest = _dataref_root(dref, scope_spec, alias_map)

    cur_type = root_type
    # All component shards except for the last one must have been type objects too.
    for comp in rest[:-1]:
        assert isinstance(comp, (Name, Part_Ref))
        if isinstance(comp, Part_Ref):
            part_name, _ = comp.children[0], comp.children[1:]
            comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
        elif isinstance(comp, Name):
            comp_spec = find_real_ident_spec(comp.string, cur_type.spec, alias_map)
        assert comp_spec in alias_map, f"canont find: {comp_spec} / {dref} in {scope_spec}"
        # So, we get the type spec for those component shards.
        cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
        assert cur_type

    # For the last one, we just need the component spec.
    comp = rest[-1]
    assert isinstance(comp, (Name, Part_Ref))
    if isinstance(comp, Part_Ref):
        part_name, _ = comp.children[0], comp.children[1:]
        comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
    elif isinstance(comp, Name):
        comp_spec = find_real_ident_spec(comp.string, cur_type.spec, alias_map)
    assert comp_spec in alias_map, f"canont find: {comp_spec} / {dref} in {scope_spec}"

    return comp_spec


def find_type_dataref(dref: Union[Name, Part_Ref, Data_Ref], scope_spec: SPEC, alias_map: SPEC_TABLE) -> TYPE_SPEC:
    _, root_type, rest = _dataref_root(dref, scope_spec, alias_map)
    cur_type = root_type

    def _subscripted_type(t: TYPE_SPEC, pref: Part_Ref):
        pname, subs = pref.children
        if not t.shape:
            # The object was not an array in the first place.
            assert not subs, f"{t} / {pname}, {t.spec}, {dref}"
        elif subs:
            # TODO: This is a hack to deduce a array type instead of scalar.
            # We may have subscripted away all the dimensions.
            t.shape = tuple(s.tofortran() for s in subs.children if ':' in s.tofortran())
        return t

    if isinstance(dref, Part_Ref):
        return _subscripted_type(cur_type, dref)
    for comp in rest:
        assert isinstance(comp, (Name, Part_Ref))
        if isinstance(comp, Part_Ref):
            # TODO: Add ref.
            part_name, subsc = comp.children
            comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
            assert comp_spec in alias_map, f"cannot find {comp_spec} / {dref} in {scope_spec}"
            cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
            cur_type = _subscripted_type(cur_type, comp)
        elif isinstance(comp, Name):
            comp_spec = find_real_ident_spec(comp.string, cur_type.spec, alias_map)
            assert comp_spec in alias_map, f"cannot find {comp_spec} / {dref} in {scope_spec}"
            cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
        assert cur_type
    return cur_type


def procedure_specs(ast: Program) -> Dict[SPEC, SPEC]:
    proc_map: Dict[SPEC, SPEC] = {}
    for pb in walk(ast, Specific_Binding):
        # Ref: https://github.com/stfc/fparser/blob/8c870f84edbf1a24dfbc886e2f7226d1b158d50b/src/fparser/two/Fortran2003.py#L2504
        iname, mylist, dcolon, bname, pname = pb.children

        proc_spec, subp_spec = [bname.string], [pname.string if pname else bname.string]

        typedef: Derived_Type_Def = pb.parent.parent
        typedef_stmt: Derived_Type_Stmt = singular(children_of_type(typedef, Derived_Type_Stmt))
        typedef_name: str = singular(children_of_type(typedef_stmt, Type_Name)).string
        proc_spec.insert(0, typedef_name)

        # TODO: Generalize.
        # We assume that the type is defined inside a module (i.e., not another subprogram).
        mod: Module = typedef.parent.parent
        mod_stmt: Module_Stmt = singular(children_of_type(mod, (Module_Stmt, Program_Stmt)))
        # TODO: Add ref.
        _, mod_name = mod_stmt.children
        proc_spec.insert(0, mod_name.string)
        subp_spec.insert(0, mod_name.string)

        # TODO: Is this assumption true?
        # We assume that the type and the bound function exist in the same scope (i.e., module, subprogram etc.).
        proc_map[tuple(proc_spec)] = tuple(subp_spec)
    return proc_map


def generic_specs(ast: Program) -> Dict[SPEC, Tuple[SPEC, ...]]:
    genc_map: Dict[SPEC, Tuple[SPEC, ...]] = {}
    for gb in walk(ast, Generic_Binding):
        # TODO: Add ref.
        aspec, bname, plist = gb.children
        if plist:
            plist = plist.children
        else:
            plist = []

        scope_spec = find_scope_spec(gb)
        genc_spec = scope_spec + (bname.string,)

        proc_specs = []
        for pname in plist:
            pspec = scope_spec + (pname.string,)
            proc_specs.append(pspec)

        # TODO: Is this assumption true?
        # We assume that the type and the bound function exist in the same scope (i.e., module, subprogram etc.).
        genc_map[tuple(genc_spec)] = tuple(proc_specs)
    return genc_map


def interface_specs(ast: Program, alias_map: SPEC_TABLE) -> Dict[SPEC, Tuple[SPEC, ...]]:
    iface_map: Dict[SPEC, Tuple[SPEC, ...]] = {}

    # First, we deal with named interface blocks.
    for ifs in walk(ast, Interface_Stmt):
        name = find_name_of_stmt(ifs)
        if not name:
            # Only named interfaces can be called.
            continue
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        ifspec = scope_spec + (name,)

        # Get the spec of all the callable things in this block that may end up as a resolution for this interface.
        fns: List[str] = []
        for fn in walk(ib, (Function_Stmt, Subroutine_Stmt, Procedure_Stmt)):
            if isinstance(fn, (Function_Stmt, Subroutine_Stmt)):
                fns.append(find_name_of_stmt(fn))
            elif isinstance(fn, Procedure_Stmt):
                for nm in walk(fn, Name):
                    fns.append(nm.string)

        fn_specs = tuple(find_real_ident_spec(f, scope_spec, alias_map) for f in fns)
        assert ifspec not in fn_specs
        iface_map[ifspec] = fn_specs

    # Then, we try to resolve anonymous interface blocks' content onto their parents' scopes.
    for ifs in walk(ast, Interface_Stmt):
        name = find_name_of_stmt(ifs)
        if name:
            # Only anonymous interface blocks.
            continue
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        assert not walk(ib, Procedure_Stmt)

        # Get the spec of all the callable things in this block that may end up as a resolution for this interface.
        for fn in walk(ib, (Function_Stmt, Subroutine_Stmt)):
            fn_name = find_name_of_stmt(fn)
            ifspec = ident_spec(fn)
            cscope = scope_spec
            fn_spec = find_real_ident_spec(fn_name, cscope, alias_map)
            # If we are resolving the interface back to itself, we need to search a level above.
            while ifspec == fn_spec:
                assert cscope
                cscope = cscope[:-1]
                fn_spec = find_real_ident_spec(fn_name, cscope, alias_map)
            assert ifspec != fn_spec
            iface_map[ifspec] = (fn_spec,)

    return iface_map


def set_children(par: Base, children: Iterable[Base]):
    assert hasattr(par, 'content') != hasattr(par, 'items')
    if hasattr(par, 'items'):
        par.items = tuple(children)
    elif hasattr(par, 'content'):
        par.content = list(children)
    _reparent_children(par)


def replace_node(node: Base, subst: Union[Base, Iterable[Base]]):
    # A lot of hacky stuff to make sure that the new nodes are not just the same objects over and over.
    par = node.parent
    only_child = bool([c for c in par.children if c == node])
    repls = []
    for c in par.children:
        if c != node:
            repls.append(c)
            continue
        if isinstance(subst, Base):
            subst = [subst]
        if not only_child:
            subst = [Base.__new__(type(t), t.tofortran()) for t in subst]
        repls.extend(subst)
    if isinstance(par, Loop_Control) and isinstance(subst, Base):
        _, cntexpr, _, _ = par.children
        if cntexpr:
            loopvar, looprange = cntexpr
            for i in range(len(looprange)):
                if looprange[i] == node:
                    looprange[i] = subst
                    subst.parent = par
    set_children(par, repls)


def append_children(par: Base, children: Union[Base, List[Base]]):
    if isinstance(children, Base):
        children = [children]
    set_children(par, list(par.children) + children)


def prepend_children(par: Base, children: Union[Base, List[Base]]):
    if isinstance(children, Base):
        children = [children]
    set_children(par, children + list(par.children))


def remove_children(par: Base, children: Union[Base, List[Base]]):
    if isinstance(children, Base):
        children = [children]
    repl = [c for c in par.children if c not in children]
    set_children(par, repl)


def remove_self(nodes: Union[Base, List[Base]]):
    if isinstance(nodes, Base):
        nodes = [nodes]
    for n in nodes:
        remove_children(n.parent, n)


def correct_for_function_calls(ast: Program):
    """Look for function calls that may have been misidentified as array access and fix them."""
    alias_map = alias_specs(ast)

    # TODO: Looping over and over is not ideal. But `Function_Reference(...)` sometimes generate inner `Part_Ref`s. We
    #  should figure out a way to avoid this clutter.
    changed = True
    while changed:
        changed = False
        for pr in walk(ast, Part_Ref):
            scope_spec = find_scope_spec(pr)
            if isinstance(pr.parent, Data_Ref):
                dref = pr.parent
                comp_spec = find_dataref_component_spec(dref, scope_spec, alias_map)
                comp_type_spec = find_type_of_entity(alias_map[comp_spec], alias_map)
                if not comp_type_spec:
                    # Cannot find a type, so it must be a function call.
                    replace_node(dref, Function_Reference(dref.tofortran()))
                    changed = True
            else:
                pr_name, _ = pr.children
                if isinstance(pr_name, Name):
                    pr_spec = search_real_local_alias_spec(pr_name, alias_map)
                    if pr_spec in alias_map and isinstance(alias_map[pr_spec], (Function_Stmt, Interface_Stmt)):
                        replace_node(pr, Function_Reference(pr.tofortran()))
                        changed = True
                elif isinstance(pr_name, Data_Ref):
                    pr_type_spec = find_type_dataref(pr_name, scope_spec, alias_map)
                    if not pr_type_spec:
                        # Cannot find a type, so it must be a function call.
                        replace_node(pr, Function_Reference(pr.tofortran()))
                        changed = True

    for sc in walk(ast, Structure_Constructor):
        scope_spec = find_scope_spec(sc)

        # TODO: Add ref.
        sc_type, _ = sc.children
        sc_type_spec = find_real_ident_spec(sc_type.string, scope_spec, alias_map)
        if isinstance(alias_map[sc_type_spec], (Function_Stmt, Interface_Stmt)):
            # Now we know that this identifier actually refers to a function.
            replace_node(sc, Function_Reference(sc.tofortran()))

    # These can also be intrinsic function calls.
    for fref in walk(ast, (Function_Reference, Call_Stmt)):
        scope_spec = find_scope_spec(fref)

        name, args = fref.children
        name = name.string
        if not Intrinsic_Name.match(name):
            # There is no way this is an intrinsic call.
            continue
        fref_spec = scope_spec + (name,)
        if fref_spec in alias_map:
            # This is already an alias, so intrinsic object is shadowed.
            continue
        if isinstance(fref, Function_Reference):
            # We need to replace with this exact node structure, and cannot rely on FParser to parse it right.
            repl = Intrinsic_Function_Reference(fref.tofortran())
            # Set the arguments ourselves, just in case the parser messes it up.
            repl.items = (Intrinsic_Name(name), args)
            _reparent_children(repl)
            replace_node(fref, repl)
        else:
            fref.items = (Intrinsic_Name(name), args)
            _reparent_children(fref)

    return ast


def remove_access_statements(ast: Program):
    """Look for public/private access statements and just remove them."""
    # TODO: This can get us into ambiguity and unintended shadowing.

    # We also remove any access statement that makes these interfaces public/private.
    for acc in walk(ast, Access_Stmt):
        # TODO: Add ref.
        kind, alist = acc.children
        assert kind.upper() in {'PUBLIC', 'PRIVATE'}
        spec = acc.parent
        remove_self(acc)
        if not spec.children:
            remove_self(spec)

    return ast


def sort_modules(ast: Program) -> Program:
    TOPLEVEL = '__toplevel__'

    def _get_module(n: Base) -> str:
        p = n
        while p and not isinstance(p, (Module, Main_Program)):
            p = p.parent
        if not p:
            return TOPLEVEL
        else:
            p = singular(children_of_type(p, (Module_Stmt, Program_Stmt)))
            return find_name_of_stmt(p)

    g = nx.DiGraph()  # An edge u->v means u should come before v, i.e., v depends on u.
    for c in ast.children:
        g.add_node(_get_module(c))

    for u in walk(ast, Use_Stmt):
        u_name = singular(children_of_type(u, Name)).string
        v_name = _get_module(u)
        g.add_edge(u_name, v_name)

    top_ord = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(g))}
    # We keep the top-level subroutines at the end. It is only a cosmetic choice and fortran accepts them anywhere.
    top_ord[TOPLEVEL] = len(top_ord) + 1
    assert all(_get_module(n) in top_ord for n in ast.children)
    ast.content = sorted(ast.children, key=lambda x: top_ord[_get_module(x)])

    return ast


def deconstruct_enums(ast: Program) -> Program:
    for en in walk(ast, Enum_Def):
        en_dict: Dict[str, Expr] = {}
        # We need to for automatic counting.
        next_val = '0'
        next_offset = 0
        for el in walk(en, Enumerator_List):
            for c in el.children:
                if isinstance(c, Name):
                    c_name = c.string
                elif isinstance(c, Enumerator):
                    # TODO: Add ref.
                    name, _, val = c.children
                    c_name = name.string
                    next_val = val.string
                    next_offset = 0
                en_dict[c_name] = Expr(f"{next_val} + {next_offset}")
                next_offset = next_offset + 1
        type_decls = [Type_Declaration_Stmt(f"integer, parameter :: {k} = {v}") for k, v in en_dict.items()]
        replace_node(en, type_decls)
    return ast


def _compute_argument_signature(args, scope_spec: SPEC, alias_map: SPEC_TABLE) -> Tuple[TYPE_SPEC, ...]:
    if not args:
        return tuple()

    args_sig = []
    for c in args.children:
        def _deduct_type(x) -> TYPE_SPEC:
            if isinstance(x, (Real_Literal_Constant, Signed_Real_Literal_Constant)):
                return TYPE_SPEC('REAL')
            elif isinstance(x, (Int_Literal_Constant, Signed_Int_Literal_Constant)):
                val = _eval_int_literal(x, alias_map)
                assert isinstance(val, NUMPY_INTS)
                return TYPE_SPEC(f"INTEGER{_count_bytes(type(val))}")
            elif isinstance(x, Char_Literal_Constant):
                str_typ = TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
                return str_typ
            elif isinstance(x, Logical_Literal_Constant):
                return TYPE_SPEC('LOGICAL')
            elif isinstance(x, Name):
                x_spec = find_real_ident_spec(x.string, scope_spec, alias_map)
                assert x_spec in alias_map, f"cannot find: {x_spec} / {x}"
                x_type = find_type_of_entity(alias_map[x_spec], alias_map)
                assert x_type, f"cannot find type for: {x_spec} / x"
                # TODO: This is a hack to make the array etc. types different.
                return x_type
            elif isinstance(x, Data_Ref):
                return find_type_dataref(x, scope_spec, alias_map)
            elif isinstance(x, Part_Ref):
                # TODO: Add ref.
                part_name, subsc = x.children
                orig_type = find_type_dataref(part_name, scope_spec, alias_map)
                if not orig_type.shape:
                    # The object was not an array in the first place.
                    assert not subsc, f"{orig_type} / {part_name}, {scope_spec}, {x}"
                    return orig_type
                if not subsc:
                    # No further subscription, so retain the original type of the object.
                    return orig_type
                # TODO: This is a hack to deduce a array type instead of scalar.
                # We may have subscripted away all the dimensions.
                subsc = subsc.children
                # TODO: Can we avoid padding the missing dimensions? This happens when the type itself is array-ish too.
                subsc = tuple([Section_Subscript(':')] * (len(orig_type.shape) - len(subsc))) + subsc
                assert len(subsc) == len(orig_type.shape)
                orig_type.shape = tuple(s.tofortran() for s in subsc if ':' in s.tofortran())
                return orig_type
            elif isinstance(x, Actual_Arg_Spec):
                kw, val = x.children
                t = _deduct_type(val)
                if isinstance(kw, Name):
                    t.keyword = kw.string
                return t
            elif isinstance(x, Intrinsic_Function_Reference):
                fname, args = x.children
                if args:
                    args = args.children
                if fname.string in {'TRIM'}:
                    return TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
                elif fname.string in {'SIZE'}:
                    return TYPE_SPEC('INTEGER')
                elif fname.string in {'REAL'}:
                    assert 1 <= len(args) <= 2
                    kind = None
                    if len(args) == 2:
                        kind = _const_eval_int(args[-1], alias_map)
                    if kind:
                        return TYPE_SPEC(f"REAL{kind}")
                    else:
                        return TYPE_SPEC('REAL')
                elif fname.string in {'INT'}:
                    assert 1 <= len(args) <= 2
                    kind = None
                    if len(args) == 2:
                        kind = _const_eval_int(args[-1], alias_map)
                    if kind:
                        return TYPE_SPEC(f"INTEGER{kind}")
                    else:
                        return TYPE_SPEC('INTEGER')
                # TODO: Figure out the actual type.
                return MATCH_ALL
            elif isinstance(x, (Level_2_Unary_Expr, And_Operand)):
                op, dref = x.children
                if op in {'+', '-', '.NOT.'}:
                    return _deduct_type(dref)
                # TODO: Figure out the actual type.
                return MATCH_ALL
            elif isinstance(x, Parenthesis):
                _, exp, _ = x.children
                return _deduct_type(exp)
            elif isinstance(x, (Level_2_Expr, Level_3_Expr)):
                lval, op, rval = x.children
                if op in {'+', '-'}:
                    tl, tr = _deduct_type(lval), _deduct_type(rval)
                    if len(tl.shape) < len(tr.shape):
                        return tr
                    else:
                        return tl
                elif op in {'//'}:
                    return TYPE_SPEC('CHARACTER', 'DIMENSION(:)')
                # TODO: Figure out the actual type.
                return MATCH_ALL
            elif isinstance(x, Array_Constructor):
                b, items, e = x.children
                items = items.children
                # TODO: We are assuming there is an element. What if there isn't?
                t = _deduct_type(items[0])
                t.shape += (':',)
                return t
            else:
                # TODO: Figure out the actual type.
                return MATCH_ALL

        c_type = _deduct_type(c)
        assert c_type, f"got: {c} / {type(c)}"
        args_sig.append(c_type)

    return tuple(args_sig)


def _compute_candidate_argument_signature(args, cand_spec: SPEC, alias_map: SPEC_TABLE) -> Tuple[TYPE_SPEC, ...]:
    cand_args_sig: List[TYPE_SPEC] = []
    for ca in args:
        ca_decl = alias_map[cand_spec + (ca.string,)]
        ca_type = find_type_of_entity(ca_decl, alias_map)
        ca_type.keyword = ca.string
        assert ca_type, f"got: {ca} / {type(ca)}"
        cand_args_sig.append(ca_type)
    return tuple(cand_args_sig)


def deconstruct_interface_calls(ast: Program) -> Program:
    SUFFIX, COUNTER = 'deconiface', 0

    alias_map = alias_specs(ast)
    iface_map = interface_specs(ast, alias_map)

    for fref in walk(ast, (Function_Reference, Call_Stmt)):
        scope_spec = find_scope_spec(fref)
        name, args = fref.children
        if isinstance(name, Intrinsic_Name):
            continue
        fref_spec = find_real_ident_spec(name.string, scope_spec, alias_map)
        assert fref_spec in alias_map, f"cannot find: {fref_spec}"
        if fref_spec not in iface_map:
            # We are only interested in calls to interfaces here.
            continue

        # Find the nearest execution and its correpsonding specification parts.
        execution_part = fref.parent
        while not isinstance(execution_part, Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = atmost_one(children_of_type(subprog, Specification_Part))

        ifc_spec = ident_spec(alias_map[fref_spec])
        args_sig: Tuple[TYPE_SPEC, ...] = _compute_argument_signature(args, scope_spec, alias_map)
        all_cand_sigs: List[Tuple[SPEC, Tuple[TYPE_SPEC, ...]]] = []

        conc_spec = None
        for cand in iface_map[ifc_spec]:
            assert cand in alias_map
            cand_stmt = alias_map[cand]
            assert isinstance(cand_stmt, (Function_Stmt, Subroutine_Stmt))

            # However, this candidate could be inside an interface block, and this be just another level of indirection.
            cand_spec = cand
            if isinstance(cand_stmt.parent.parent, Interface_Block):
                cand_spec = find_real_ident_spec(cand_spec[-1], cand_spec[:-2], alias_map)
                assert cand_spec in alias_map
                cand_stmt = alias_map[cand_spec]
                assert isinstance(cand_stmt, (Function_Stmt, Subroutine_Stmt))

            # TODO: Add ref.
            _, _, cand_args, _ = cand_stmt.children
            if cand_args:
                cand_args_sig = _compute_candidate_argument_signature(cand_args.children, cand_spec, alias_map)
            else:
                cand_args_sig = tuple()
            all_cand_sigs.append((cand_spec, cand_args_sig))

            if _does_type_signature_match(args_sig, cand_args_sig):
                conc_spec = cand_spec
                break
        if conc_spec not in alias_map:
            print(f"{ifc_spec}/{conc_spec} / {args_sig}")
            for c in all_cand_sigs:
                print(f"...> {c}")
        assert conc_spec and conc_spec in alias_map, f"[in: {fref_spec}] {ifc_spec}/{conc_spec} not found"

        # We are assumping that it's either a toplevel subprogram or a subprogram defined directly inside a module.
        assert 1 <= len(conc_spec) <= 2
        if len(conc_spec) == 1:
            mod, pname = None, conc_spec[0]
        else:
            mod, pname = conc_spec

        if mod is None or mod == scope_spec[0]:
            # Since `pname` must have been already defined at either the top level or the module level, there is no need
            # for aliasing.
            pname_alias = pname
        else:
            # If we are importing it from a different module, we should create an alias to avoid name collision.
            pname_alias, COUNTER = f"{pname}_{SUFFIX}_{COUNTER}", COUNTER + 1
            if not specification_part:
                append_children(subprog, Specification_Part(get_reader(f"use {mod}, only: {pname_alias} => {pname}")))
            else:
                prepend_children(specification_part, Use_Stmt(f"use {mod}, only: {pname_alias} => {pname}"))

        # For both function and subroutine calls, replace `bname` with `pname_alias`, and add `dref` as the first arg.
        replace_node(name, Name(pname_alias))

    # TODO: Figure out a way without rebuilding here.
    # Rebuild the maps because aliasing may have changed.
    alias_map = alias_specs(ast)

    # At this point, we must have replaced all the interface calls with concrete calls.
    for use in walk(ast, Use_Stmt):
        mod_name = singular(children_of_type(use, Name)).string
        mod_spec = (mod_name,)
        olist = atmost_one(children_of_type(use, Only_List))
        if not olist:
            # There is nothing directly referring to the interface.
            continue

        survivors = []
        for c in olist.children:
            assert isinstance(c, (Name, Rename))
            if isinstance(c, Name):
                src, tgt = c, c
            elif isinstance(c, Rename):
                _, src, tgt = c.children
            src, tgt = src.string, tgt.string
            tgt_spec = find_real_ident_spec(tgt, mod_spec, alias_map)
            assert tgt_spec in alias_map
            if tgt_spec not in iface_map:
                # Leave the non-interface usages alone.
                survivors.append(c)

        if survivors:
            olist.items = survivors
            _reparent_children(olist)
        else:
            remove_self(use)

    # At this point, we must have replaced all references to the interfaces.
    for k in iface_map.keys():
        assert k in alias_map
        ib = None
        if isinstance(alias_map[k], Interface_Stmt):
            ib = alias_map[k].parent
        elif isinstance(alias_map[k], (Function_Stmt, Subroutine_Stmt)):
            ib = alias_map[k].parent.parent
        assert isinstance(ib, Interface_Block)
        remove_self(ib)

    return ast


MATCH_ALL = TYPE_SPEC(('*',), '')  # TODO: Hacky; `_does_type_signature_match()` will match anything with this.


def _does_part_matches(g: TYPE_SPEC, c: TYPE_SPEC) -> bool:
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

    def _subsumes(b: SPEC, s: SPEC) -> bool:
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


def _does_type_signature_match(got_sig: Tuple[TYPE_SPEC, ...], cand_sig: Tuple[TYPE_SPEC, ...]):
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


def deconstruct_procedure_calls(ast: Program) -> Program:
    SUFFIX, COUNTER = 'deconproc', 0

    alias_map = alias_specs(ast)
    proc_map = procedure_specs(ast)
    genc_map = generic_specs(ast)
    # We should have removed all the `association`s by now.
    assert not walk(ast, Association), f"{walk(ast, Association)}"

    for pd in walk(ast, Procedure_Designator):
        # Ref: https://github.com/stfc/fparser/blob/master/src/fparser/two/Fortran2003.py#L12530
        dref, op, bname = pd.children

        callsite = pd.parent
        assert isinstance(callsite, (Function_Reference, Call_Stmt))

        # Find out the module name.
        cmod = callsite.parent
        while cmod and not isinstance(cmod, (Module, Main_Program)):
            cmod = cmod.parent
        if cmod:
            stmt, _, _, _ = _get_module_or_program_parts(cmod)
            cmod = singular(children_of_type(stmt, Name)).string.lower()
        else:
            subp = list(children_of_type(ast, Subroutine_Subprogram))
            assert subp
            stmt = singular(children_of_type(subp[0], Subroutine_Stmt))
            cmod = singular(children_of_type(stmt, Name)).string.lower()

        # Find the nearest execution and its correpsonding specification parts.
        execution_part = callsite.parent
        while not isinstance(execution_part, Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = atmost_one(children_of_type(subprog, Specification_Part))

        scope_spec = find_scope_spec(callsite)
        dref_type = find_type_dataref(dref, scope_spec, alias_map)
        fnref = pd.parent
        assert isinstance(fnref, (Function_Reference, Call_Stmt))
        _, args = fnref.children
        args_sig: Tuple[TYPE_SPEC, ...] = _compute_argument_signature(args, scope_spec, alias_map)
        all_cand_sigs: List[Tuple[SPEC, Tuple[TYPE_SPEC, ...]]] = []

        bspec = dref_type.spec + (bname.string,)
        if bspec in genc_map and genc_map[bspec]:
            for cand in genc_map[bspec]:
                cand_stmt = alias_map[proc_map[cand]]
                cand_spec = ident_spec(cand_stmt)
                # TODO: Add ref.
                _, _, cand_args, _ = cand_stmt.children
                if cand_args:
                    cand_args_sig = _compute_candidate_argument_signature(cand_args.children[1:], cand_spec, alias_map)
                else:
                    cand_args_sig = tuple()
                all_cand_sigs.append((cand_spec, cand_args_sig))

                if _does_type_signature_match(args_sig, cand_args_sig):
                    bspec = cand
                    break
        if bspec not in proc_map:
            print(f"{bspec} / {args_sig}")
            for c in all_cand_sigs:
                print(f"...> {c}")
        assert bspec in proc_map, f"[in mod: {cmod}/{callsite}] {bspec} not found"
        pname = proc_map[bspec]

        # We are assumping that it's a subprogram defined directly inside a module.
        assert len(pname) == 2
        mod, pname = pname

        if mod == cmod:
            # Since `pname` must have been already defined at the module level, there is no need for aliasing.
            pname_alias = pname
        else:
            # If we are importing it from a different module, we should create an alias to avoid name collision.
            pname_alias, COUNTER = f"{pname}_{SUFFIX}_{COUNTER}", COUNTER + 1
            if not specification_part:
                append_children(subprog, Specification_Part(get_reader(f"use {mod}, only: {pname_alias} => {pname}")))
            else:
                prepend_children(specification_part, Use_Stmt(f"use {mod}, only: {pname_alias} => {pname}"))

        # For both function and subroutine calls, replace `bname` with `pname_alias`, and add `dref` as the first arg.
        _, args = callsite.children
        if args is None:
            args = Actual_Arg_Spec_List(f"{dref}")
        else:
            args = Actual_Arg_Spec_List(f"{dref}, {args}")
        callsite.items = (Name(pname_alias), args)
        _reparent_children(callsite)

    for tbp in walk(ast, Type_Bound_Procedure_Part):
        remove_self(tbp)
    return ast


def _reparent_children(node: Base):
    """Make `node` a parent of all its children, in case it isn't already."""
    for c in node.children:
        if isinstance(c, Base):
            c.parent = node


def prune_unused_objects(ast: Program, keepers: List[SPEC]) -> Program:
    """
    Precondition: All the indirections have been taken out of the program.
    """
    # NOTE: Modules are not included here, because they are simply containers with no other direct use. Empty modules
    # should be pruned at the end separately.
    PRUNABLE_OBJECT_CLASSES = (Program_Stmt, Subroutine_Stmt, Function_Stmt, Derived_Type_Stmt, Entity_Decl,
                               Component_Decl)

    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)
    survivors: Set[SPEC] = set()
    keepers = [alias_map[k] for k in keepers]
    assert all(isinstance(k, PRUNABLE_OBJECT_CLASSES) for k in keepers)

    def _keep_from(node: Base):
        """
        Ensure that `node` is not pruned. Things defined in it can be pruned, but only if unused.
        """
        # Go over all the scoped identifiers available under `node`.
        for nm in walk(node, Name):
            loc = search_real_local_alias_spec(nm, alias_map)
            scope_spec = search_scope_spec(nm.parent)
            if not loc or not scope_spec:
                continue
            nm_spec = ident_spec(alias_map[loc])
            if isinstance(nm.parent, Entity_Decl) and nm == nm.parent.children[0]:
                fnargs = atmost_one(children_of_type(alias_map[scope_spec], Dummy_Arg_List))
                fnargs = fnargs.children if fnargs else tuple()
                if any(a.string == nm.string for a in fnargs):
                    # We cannot remove function arguments yet.
                    survivors.add(nm_spec)
                    continue
                # Otherwise, this is a declaration of the variable, which is not a use, and so a fair game for removal.
                continue
            if isinstance(nm.parent, Component_Decl) and nm == nm.parent.children[0]:
                # This is a declaration of the component, which is not a use, and so a fair game for removal.
                continue

            # All the scope ancestors of `nm` must live too.
            for j in reversed(range(len(scope_spec))):
                anc = scope_spec[:j + 1]
                if anc in survivors:
                    continue
                survivors.add(anc)
                anc_node = alias_map[anc]
                if isinstance(anc_node, PRUNABLE_OBJECT_CLASSES):
                    _keep_from(anc_node.parent)

            # We keep the definition of that `nm` is an alias of.
            if not nm_spec or nm_spec not in alias_map or nm_spec in survivors:
                # If we don't have a valid `to_keep` or `to_keep` is already kept, we move on.
                continue
            survivors.add(nm_spec)
            keep_node = alias_map[nm_spec]
            if isinstance(keep_node, PRUNABLE_OBJECT_CLASSES):
                _keep_from(keep_node.parent)
        # Go over all the data-refs available under `node`.
        for dr in walk(node, Data_Ref):
            root, rest = _lookup_dataref(dr, alias_map)
            scope_spec = find_scope_spec(dr)
            # All the data-ref ancestors of `dr` must live too.
            for upto in range(1, len(rest)+1):
                anc: Tuple[Name, ...] = (root,) + rest[:upto]
                ancref = Data_Ref('%'.join([c.tofortran() for c in anc]))
                ancspec = find_dataref_component_spec(ancref, scope_spec, alias_map)
                survivors.add(ancspec)

    for k in keepers:
        _keep_from(k.parent)

    # We keep them sorted so that the parent scopes are handled earlier.
    killed: Set[SPEC] = set()
    for ns in sorted(set(ident_map.keys()) - survivors):
        ns_node = ident_map[ns]
        if not isinstance(ns_node, PRUNABLE_OBJECT_CLASSES):
            continue
        for i in range(len(ns) - 1):
            anc_spec = ns[:i + 1]
            if anc_spec in killed:
                killed.add(ns)
                break
        if ns in killed:
            continue
        if isinstance(ns_node, Entity_Decl):
            elist = ns_node.parent
            remove_self(ns_node)
            # But there are many things to clean-up.
            # 1. If the variable was declared alone, then the entire line with type declaration must be gone too.
            elist_tdecl = elist.parent
            assert isinstance(elist_tdecl, Type_Declaration_Stmt)
            if not elist.children:
                remove_self(elist_tdecl)
            # 2. There is a case of "equivalence" statement, which is a very Fortran-specific feature to clean up too.
            elist_spart = elist_tdecl.parent
            assert isinstance(elist_spart, Specification_Part)
            for c in elist_spart.children:
                if not isinstance(c, Equivalence_Stmt):
                    continue
                _, eqvs = c.children
                eqvs = eqvs.children if eqvs else tuple()
                for eqv in eqvs:
                    eqa, eqbs = eqv.children
                    eqbs = eqbs.children if eqbs else tuple()
                    eqz = (eqa,) + eqbs
                    assert all(isinstance(z, Part_Ref) for z in eqz)
                    assert len(eqz) == 2
                    eqz = tuple(z for z in eqz if search_real_local_alias_spec(z.children[0], alias_map) != ns)
                    if len(eqz) < 2:
                        remove_self(eqv)
                # If there is no remaining equivalent list, remove the entire statement.
                _, eqvs = c.children
                eqvs = eqvs.children if eqvs else tuple()
                if not eqvs:
                    remove_self(c)
            # 3. If the entire specification part becomes empty, we have to remove it too.
            if not elist_spart.children:
                remove_self(elist_spart)
        elif isinstance(ns_node, Component_Decl):
            clist = ns_node.parent
            remove_self(ns_node)
            # But there are many things to clean-up.
            # 1. If the component was declared alone, then the entire line within type defintion must be gone too.
            tdef = clist.parent
            assert isinstance(tdef, Data_Component_Def_Stmt)
            if not clist.children:
                remove_self(tdef)
        else:
            remove_self(ns_node.parent)
        killed.add(ns)

    # Cleanup the empty modules.
    for m in walk(ast, Module):
        _, sp, ex, sub = _get_module_or_program_parts(m)
        empty_specification = not sp or all(isinstance(c, (Save_Stmt, Implicit_Part)) for c in sp.children)
        empty_execution = not ex or not ex.children
        empty_subprogram = not sub or all(isinstance(c, Contains_Stmt) for c in sub.children)
        if empty_specification and empty_execution and empty_subprogram:
            remove_self(m)

    consolidate_uses(ast, alias_map)

    return ast


def make_practically_constant_global_vars_constants(ast: Program) -> Program:
    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)

    # Start with everything that _could_ be a candidate.
    never_assigned: Set[SPEC] = {k for k, v in ident_map.items()
                                 if isinstance(v, Entity_Decl) and not find_type_of_entity(v, alias_map).const
                                 and search_scope_spec(v) and isinstance(alias_map[search_scope_spec(v)], Module_Stmt)}

    for asgn in walk(ast, Assignment_Stmt):
        lv, _, rv = asgn.children
        if not isinstance(lv, Name):
            # Everything else unsupported for now.
            continue
        loc = search_real_local_alias_spec(lv, alias_map)
        assert loc
        var = alias_map[loc]
        assert isinstance(var, Entity_Decl)
        var_spec = ident_spec(var)
        if var_spec in never_assigned:
            never_assigned.remove(var_spec)

    for fcall in walk(ast, (Function_Reference, Call_Stmt)):
        fn, args = fcall.children
        args = args.children if args else tuple()
        for a in args:
            if not isinstance(a, Name):
                # Everything else unsupported for now.
                continue
            loc = search_real_local_alias_spec(a, alias_map)
            assert loc
            var = alias_map[loc]
            assert isinstance(var, Entity_Decl)
            var_spec = ident_spec(var)
            if var_spec in never_assigned:
                never_assigned.remove(var_spec)

    for fixed in never_assigned:
        edcl = alias_map[fixed]
        assert isinstance(edcl, Entity_Decl)
        if not atmost_one(children_of_type(edcl, Initialization)):
            # Without an initialization, we cannot fix it.
            continue
        edclist = edcl.parent
        tdcl = edclist.parent
        assert isinstance(tdcl, Type_Declaration_Stmt)
        typ, attr, _ = tdcl.children
        if not attr:
            nuattr = 'parameter'
        elif 'PARAMETER' in f"{attr}":
            nuattr = f"{attr}"
        else:
            nuattr = f"{attr}, parameter"
        if len(edclist.children) == 1:
            replace_node(tdcl, Type_Declaration_Stmt(f"{typ}, {nuattr} :: {edclist}"))
        else:
            replace_node(tdcl, Type_Declaration_Stmt(f"{typ}, {nuattr} :: {edcl}"))
            remove_children(edclist, edcl)
            attr = f", {attr}" if attr else ''
            append_children(tdcl.parent, Type_Declaration_Stmt(f"{typ} {attr} :: {edclist}"))

    return ast


def make_practically_constant_arguments_constants(ast: Program, keepers: List[SPEC]) -> Program:
    alias_map = alias_specs(ast)

    # First, build a table to see what possible values a function argument may see.
    fnargs_possible_values: Dict[SPEC, Set[Optional[NUMPY_TYPES]]] = {}
    fnargs_undecidables: Set[SPEC] = set()
    fnargs_optional_presence: Dict[SPEC, Set[bool]] = {}
    for fcall in walk(ast, (Function_Reference, Call_Stmt)):
        fn, args = fcall.children
        if isinstance(fn, Intrinsic_Name):
            # Cannot do anything with intrinsic functions.
            continue
        args = args.children if args else tuple()
        kwargs = tuple(a.children for a in args if isinstance(a, Actual_Arg_Spec))
        kwargs = {k.string: v for k, v in kwargs}
        fnspec = search_real_local_alias_spec(fn, alias_map)
        assert fnspec
        fnstmt = alias_map[fnspec]
        fnspec = ident_spec(fnstmt)
        if fnspec in keepers:
            # The "entry-point" functions arguments are fair game for external usage.
            continue
        fnargs = atmost_one(children_of_type(fnstmt, Dummy_Arg_List))
        fnargs = fnargs.children if fnargs else tuple()
        assert len(args) <= len(fnargs), f"Cannot pass more arguments({len(args)}) than defined ({len(fnargs)})"
        for a in fnargs:
            aspec = search_real_local_alias_spec(a, alias_map)
            assert aspec
            adecl = alias_map[aspec]
            atype = find_type_of_entity(adecl, alias_map)
            assert atype
            if not args:
                # If we do not have supplied arguments anymore, the remaining arguments must be optional
                assert atype.optional
                continue
            kwargs_zone = isinstance(args[0], Actual_Arg_Spec)  # Whether we are in keyword args territory.
            if kwargs_zone:
                # This is an argument, so it must have been supplied as a keyworded value.
                assert a.string in kwargs or atype.optional
                v = kwargs.get(a.string)
            else:
                # Pop the next non-keywordd supplied value.
                v, args = args[0], args[1:]
            if atype.optional:
                # The presence should be noted even if it is a writable argument.
                if aspec not in fnargs_optional_presence:
                    fnargs_optional_presence[aspec] = set()
                fnargs_optional_presence[aspec].add(v is not None)
            if atype.out:
                # Writable arguments are not practically constants anyway.
                continue
            assert atype.inp
            if atype.shape:
                # TODO: Cannot handle non-scalar literals yet. So we just skip for it.
                continue
            if isinstance(v, LITERAL_CLASSES):
                v = _const_eval_basic_type(v, alias_map)
                assert v is not None
                if aspec not in fnargs_possible_values:
                    fnargs_possible_values[aspec] = set()
                fnargs_possible_values[aspec].add(v)
            elif v is None:
                assert atype.optional
                if aspec not in fnargs_possible_values:
                    fnargs_possible_values[aspec] = set()
                fnargs_possible_values[aspec].add(v)
            else:
                fnargs_undecidables.add(aspec)

    for aspec, vals in fnargs_optional_presence.items():
        if len(vals) > 1:
            continue
        assert len(vals) == 1
        presence, = vals

        arg = alias_map[aspec]
        atype = find_type_of_entity(arg, alias_map)
        assert atype.optional
        fn = find_named_ancestor(arg).parent
        assert isinstance(fn, (Subroutine_Subprogram, Function_Subprogram))
        fexec = atmost_one(children_of_type(fn, Execution_Part))
        if not fexec:
            continue

        for pcall in walk(fexec, Intrinsic_Function_Reference):
            fn, cargs = pcall.children
            cargs = cargs.children if cargs else tuple()
            if fn.string != 'PRESENT':
                continue
            assert len(cargs) == 1
            optvar = cargs[0]
            if find_name_of_node(arg) != optvar.string:
                continue
            replace_node(pcall, numpy_type_to_literal(np.bool_(presence)))

    for aspec, vals in fnargs_possible_values.items():
        if aspec in fnargs_undecidables or len(vals) > 1:
            # There are multiple possiblities for the argument: either some undecidables or multiple literals.
            continue
        fixed_val, = vals
        arg = alias_map[aspec]
        atype = find_type_of_entity(arg, alias_map)
        fn = find_named_ancestor(arg).parent
        assert isinstance(fn, (Subroutine_Subprogram, Function_Subprogram))
        fexec = atmost_one(children_of_type(fn, Execution_Part))
        if not fexec:
            continue

        if fixed_val is not None:
            for nm in walk(fexec, Name):
                nmspec = search_real_local_alias_spec(nm, alias_map)
                if nmspec != aspec:
                    continue
                replace_node(nm, numpy_type_to_literal(fixed_val))
        # TODO: We could also try removing the argument entirely from the function definition, but that's more work with
        #  little benefit, so maybe another time.

    return ast


LITERAL_TYPES = Union[
    Real_Literal_Constant, Signed_Real_Literal_Constant, Int_Literal_Constant, Signed_Int_Literal_Constant,
    Logical_Literal_Constant]
LITERAL_CLASSES = (
    Real_Literal_Constant, Signed_Real_Literal_Constant, Int_Literal_Constant, Signed_Int_Literal_Constant,
    Logical_Literal_Constant)


def _track_local_consts(node: Base, alias_map: SPEC_TABLE,
                        plus: Optional[Dict[Union[SPEC, Tuple[SPEC, SPEC]], LITERAL_TYPES]] = None,
                        minus: Optional[Set[Union[SPEC, Tuple[SPEC, SPEC]]]] = None) \
        -> Tuple[Dict[SPEC, LITERAL_TYPES], Set[SPEC]]:
    plus: Dict[Union[SPEC, Tuple[SPEC, SPEC]], LITERAL_TYPES] = copy(plus) if plus else {}
    minus: Set[Union[SPEC, Tuple[SPEC, SPEC]]] = copy(minus) if minus else set()

    def _root_comp(dref: Data_Ref):
        scope_spec = search_scope_spec(dref)
        assert scope_spec
        root, _, _ = _dataref_root(dref, scope_spec, alias_map)
        assert isinstance(root, Name)
        loc = search_real_local_alias_spec(root, alias_map)
        assert loc
        root_spec = ident_spec(alias_map[loc])
        comp_spec = find_dataref_component_spec(dref, scope_spec, alias_map)
        return root_spec, comp_spec

    def _integrate_subresults(tp: Dict[SPEC, LITERAL_TYPES], tm: Set[SPEC]):
        assert not (tm & tp.keys())
        for k in tm:
            if k in plus:
                del plus[k]
            minus.add(k)
        for k, v in tp.items():
            if k in minus:
                minus.remove(k)
            plus[k] = v

    def _inject_knowns(x: Base):
        if isinstance(x, (*LITERAL_CLASSES, Char_Literal_Constant, Write_Stmt, Close_Stmt, Goto_Stmt)):
            pass
        elif isinstance(x, Assignment_Stmt):
            lv, op, rv = x.children
            _inject_knowns(rv)
        elif isinstance(x, Name):
            loc = search_real_local_alias_spec(x, alias_map)
            if loc:
                spec = ident_spec(alias_map[loc])
                if spec in plus:
                    assert spec not in minus
                    replace_node(x, plus[spec])
        elif isinstance(x, Data_Ref):
            spec = _root_comp(x)
            if spec in plus:
                assert spec not in minus
                replace_node(x, plus[spec])
        elif isinstance(x, Part_Ref):
            par, subsc = x.children
            assert isinstance(subsc, Section_Subscript_List)
            for c in subsc.children:
                _inject_knowns(c)
        elif isinstance(x, Subscript_Triplet):
            for c in x.children:
                if c:
                    _inject_knowns(c)
        elif isinstance(x, Parenthesis):
            _, y, _ = x.children
            _inject_knowns(y)
        elif isinstance(x, UnaryOpBase):
            op, val = x.children
            _inject_knowns(val)
        elif isinstance(x, BinaryOpBase):
            assert not isinstance(x, Assignment_Stmt)
            lv, op, rv = x.children
            _inject_knowns(lv)
            _inject_knowns(rv)
        elif isinstance(x, (Function_Reference, Call_Stmt, Intrinsic_Function_Reference)):
            _, args = x.children
            args = args.children if args else tuple()
            for a in args:
                # TODO: For now, we assume that all arguments are writable.
                if not isinstance(a, Name):
                    _inject_knowns(a)
        elif isinstance(x, Actual_Arg_Spec):
            _, val = x.children
            _inject_knowns(val)
        else:
            raise NotImplementedError(f"cannot handle {x} | {type(x)}")

    if isinstance(node, Execution_Part):
        scpart = atmost_one(children_of_type(node.parent, Specification_Part))
        knowns: Dict[SPEC, LITERAL_TYPES] = {}
        if scpart:
            for tdcls in scpart.children:
                if not isinstance(tdcls, Type_Declaration_Stmt):
                    continue
                _, _, edcls = tdcls.children
                edcls = edcls.children if edcls else tuple()
                for var in edcls:
                    _, _, _, init = var.children
                    if init:
                        _, init = init.children
                    if init and isinstance(init, LITERAL_CLASSES):
                        knowns[ident_spec(var)] = init
        _integrate_subresults(knowns, set())
        for op in node.children:
            # TODO: We wouldn't need the exception handling once we implement for all node types.
            try:
                tp, tm = _track_local_consts(op, alias_map, plus, minus)
                _integrate_subresults(tp, tm)
            except NotImplementedError:
                plus, minus = {}, set()
    elif isinstance(node, Assignment_Stmt):
        lv, op, rv = node.children
        _inject_knowns(rv)
        lv, op, rv = node.children
        lspec = None
        if isinstance(lv, Name):
            loc = search_real_local_alias_spec(lv, alias_map)
            assert loc
            lspec = ident_spec(alias_map[loc])
        elif isinstance(lv, Data_Ref):
            lspec = _root_comp(lv)
        if lspec:
            rval = _const_eval_basic_type(rv, alias_map)
            if rval is None:
                _integrate_subresults({}, {lspec})
            else:
                plus[lspec] = numpy_type_to_literal(rval)
                if lspec in minus:
                    minus.remove(lspec)
        tp, tm = _track_local_consts(rv, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, If_Stmt):
        cond, body = node.children
        _inject_knowns(cond)
        _inject_knowns(body)
        cond, body = node.children
        tp, tm = _track_local_consts(cond, alias_map)
        _integrate_subresults(tp, tm)
        tp, tm = _track_local_consts(body, alias_map)
        _integrate_subresults({}, tm | tp.keys())
    elif isinstance(node, If_Construct):
        for c in children_of_type(node, (If_Then_Stmt, Else_If_Stmt)):
            if isinstance(c, If_Then_Stmt):
                cond, = c.children
            elif isinstance(c, Else_If_Stmt):
                cond, _ = c.children
            _inject_knowns(cond)
        for c in node.children:
            if isinstance(c, (If_Then_Stmt, Else_If_Stmt, Else_Stmt, End_If_Stmt)):
                continue
            tp, tm = _track_local_consts(c, alias_map)
            _integrate_subresults({}, tm | tp.keys())
    elif isinstance(node, (Block_Nonlabel_Do_Construct, Block_Label_Do_Construct)):
        do_stmt = node.children[0]
        assert isinstance(do_stmt, (Label_Do_Stmt, Nonlabel_Do_Stmt))
        assert isinstance(node.children[-1], End_Do_Stmt)
        do_ops = node.children[1:-1]
        for op in do_ops:
            tp, tm = _track_local_consts(op, alias_map, plus, minus)
            _integrate_subresults(tp, tm)

        _, loop_ctl = do_stmt.children
        _, loop_var, _, _ = loop_ctl.children
        if loop_var:
            loop_var, _ = loop_var
            assert isinstance(loop_var, Name)
            loop_var_spec = search_real_local_alias_spec(loop_var, alias_map)
            assert loop_var_spec
            loop_var_spec = ident_spec(alias_map[loop_var_spec])
            minus.add(loop_var_spec)
    elif isinstance(node, (
        Name, *LITERAL_CLASSES, Char_Literal_Constant, Data_Ref, Part_Ref, Return_Stmt, Write_Stmt, Error_Stop_Stmt,
        Exit_Stmt, Actual_Arg_Spec, Write_Stmt, Close_Stmt, Goto_Stmt, Continue_Stmt, Format_Stmt)):
        # These don't modify variables or give any new information.
        pass
    elif isinstance(node, (Allocate_Stmt, Deallocate_Stmt)):
        # These are not expected to exit in the pruned AST, so don't bother tracking them.
        pass
    elif isinstance(node, UnaryOpBase):
        _inject_knowns(node)
        op, val = node.children
        tp, tm = _track_local_consts(val, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, BinaryOpBase):
        assert not isinstance(node, Assignment_Stmt)
        lv, op, rv = node.children
        _inject_knowns(lv)
        _inject_knowns(rv)
        lv, op, rv = node.children
        tp, tm = _track_local_consts(lv, alias_map)
        _integrate_subresults(tp, tm)
        tp, tm = _track_local_consts(rv, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, Parenthesis):
        _, val, _ = node.children
        tp, tm = _track_local_consts(val, alias_map)
        _integrate_subresults(tp, tm)
    elif isinstance(node, (Function_Reference, Call_Stmt, Intrinsic_Function_Reference)):
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


def exploit_locally_constant_variables(ast: Program) -> Program:
    alias_map = alias_specs(ast)

    for expart in walk(ast, Execution_Part):
        _track_local_consts(expart, alias_map)

    return ast


def deconstruct_associations(ast: Program) -> Program:
    for assoc in walk(ast, Associate_Construct):
        # TODO: Add ref.
        stmt, rest, _ = assoc.children[0], assoc.children[1:-1], assoc.children[-1]
        # TODO: Add ref.
        kw, assoc_list = stmt.children[0], stmt.children[1:]
        if not assoc_list:
            continue

        # Keep track of what to replace in the local scope.
        local_map: Dict[str, Base] = {}
        for al in assoc_list:
            for a in al.children:
                # TODO: Add ref.
                src, _, tgt = a.children
                local_map[src.string] = tgt

        for node in rest:
            # Replace the data-ref roots as appropriate.
            for dr in walk(node, Data_Ref):
                # TODO: Add ref.
                root, dr_rest = dr.children[0], dr.children[1:]
                if root.string in local_map:
                    repl = local_map[root.string]
                    repl = type(repl)(repl.tofortran())
                    dr.items = (repl, *dr_rest)
                    _reparent_children(dr)
            # # Replace the part-ref roots as appropriate.
            for pr in walk(node, Part_Ref):
                if isinstance(pr.parent, (Data_Ref, Part_Ref)):
                    continue
                # TODO: Add ref.
                root, subsc = pr.children
                if root.string in local_map:
                    repl = local_map[root.string]
                    repl = type(repl)(repl.tofortran())
                    if isinstance(subsc, Section_Subscript_List) and isinstance(repl, (Data_Ref, Part_Ref)):
                        access = repl
                        while isinstance(access, (Data_Ref, Part_Ref)):
                            access = access.children[-1]
                        if isinstance(access, Section_Subscript_List):
                            # We cannot just chain accesses, so we need to combine them to produce a single access.
                            # TODO: Maybe `isinstance(c, Subscript_Triplet)` + offset manipulation?
                            free_comps = [(i, c) for i, c in enumerate(access.children) if c == Subscript_Triplet(':')]
                            assert len(free_comps) >= len(subsc.children), \
                                f"Free rank cannot increase, got {root}/{access} => {subsc}"
                            for i, c in enumerate(subsc.children):
                                idx, _ = free_comps[i]
                                free_comps[i] = (idx, c)
                            free_comps = {i: c for i, c in free_comps}
                            access.items = [free_comps.get(i, c) for i, c in enumerate(access.children)]
                            # Now replace the entire `pr` with `repl`.
                            replace_node(pr, repl)
                            continue
                    # Otherwise, just replace normally.
                    pr.items = (repl, subsc)
                    _reparent_children(pr)
            # Replace all the other names.
            for nm in walk(node, Name):
                # TODO: This is hacky and can backfire if `nm` is not a standalone identifier.
                par = nm.parent
                # Avoid data refs as we have just processed them.
                if isinstance(par, (Data_Ref, Part_Ref)):
                    continue
                if nm.string not in local_map:
                    continue
                replace_node(nm, local_map[nm.string])
        replace_node(assoc, rest)

    return ast


def assign_globally_unique_subprogram_names(ast: Program, keepers: Set[SPEC]) -> Program:
    """
    Update the functions (and interchangeably, subroutines) to have globally unique names.
    Precondition:
    1. All indirections are already removed from the program, except for the explicit renames.
    2. All public/private access statements were cleanly removed.
    TODO: Make structure names unique too.
    """
    SUFFIX, COUNTER = 'fn', 0

    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)

    name_collisions: Dict[str, int] = {k[-1]: 0 for k in ident_map.keys()}
    for k in ident_map.keys():
        name_collisions[k[-1]] += 1
    name_collisions: Set[str] = {k for k, v in name_collisions.items() if v > 1}

    # Make new unique names for the identifiers.
    uident_map: Dict[SPEC, str] = {}
    for k in ident_map.keys():
        if k in keepers:
            continue
        if k[-1] in name_collisions:
            uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        else:
            uname = k[-1]
        uident_map[k] = uname

    # PHASE 1.a: Remove all the places where any function is imported.
    for use in walk(ast, Use_Stmt):
        mod_name = singular(children_of_type(use, Name)).string
        mod_spec = (mod_name,)
        olist = atmost_one(children_of_type(use, Only_List))
        if not olist:
            continue
        survivors = []
        for c in olist.children:
            assert isinstance(c, (Name, Rename))
            if isinstance(c, Name):
                src, tgt = c, c
            elif isinstance(c, Rename):
                _, src, tgt = c.children
            src, tgt = src.string, tgt.string
            tgt_spec = find_real_ident_spec(tgt, mod_spec, alias_map)
            assert tgt_spec in ident_map
            if not isinstance(ident_map[tgt_spec], (Function_Stmt, Subroutine_Stmt)):
                survivors.append(c)
        if survivors:
            olist.items = survivors
            _reparent_children(olist)
        else:
            par = use.parent
            par.content = [c for c in par.children if c != use]
            _reparent_children(par)

    # PHASE 1.b: Replaces all the function callsites.
    for fref in walk(ast, (Function_Reference, Call_Stmt)):
        scope_spec = find_scope_spec(fref)

        # TODO: Add ref.
        name, _ = fref.children
        if not isinstance(name, Name):
            # Intrinsics are not to be renamed.
            assert isinstance(name, Intrinsic_Name), f"{fref}"
            continue
        fspec = find_real_ident_spec(name.string, scope_spec, alias_map)
        assert fspec in ident_map
        assert isinstance(ident_map[fspec], (Function_Stmt, Subroutine_Stmt))
        if fspec not in uident_map:
            # We have chosen to not rename it.
            continue
        uname = uident_map[fspec]
        ufspec = fspec[:-1] + (uname,)
        name.string = uname

        # Find the nearest execution and its correpsonding specification parts.
        execution_part = fref.parent
        while not isinstance(execution_part, Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = atmost_one(children_of_type(subprog, Specification_Part))

        # Find out the module name.
        cmod = fref.parent
        while cmod and not isinstance(cmod, (Module, Main_Program)):
            cmod = cmod.parent
        if cmod:
            stmt, _, _, _ = _get_module_or_program_parts(cmod)
            cmod = singular(children_of_type(stmt, Name)).string.lower()
        else:
            subp = list(children_of_type(ast, Subroutine_Subprogram))
            assert subp
            stmt = singular(children_of_type(subp[0], Subroutine_Stmt))
            cmod = singular(children_of_type(stmt, Name)).string.lower()

        assert 1 <= len(ufspec)
        if len(ufspec) == 1:
            # Nothing to do for the toplevel subprograms. They are already available.
            continue
        mod = ufspec[0]
        if mod == cmod:
            # Since this function is already defined at the current module, there is nothing to import.
            continue

        if not specification_part:
            append_children(subprog, Specification_Part(get_reader(f"use {mod}, only: {uname}")))
        else:
            prepend_children(specification_part, Use_Stmt(f"use {mod}, only: {uname}"))

    # PHASE 1.d: Replaces actual function names.
    for k, v in ident_map.items():
        if not isinstance(v, (Function_Stmt, Subroutine_Stmt)):
            continue
        if k not in uident_map:
            # We have chosen to not rename it.
            continue
        oname, uname = k[-1], uident_map[k]
        singular(children_of_type(v, Name)).string = uname
        # Fix the tail too.
        fdef = v.parent
        end_stmt = singular(children_of_type(fdef, (End_Function_Stmt, End_Subroutine_Stmt)))
        singular(children_of_type(end_stmt, Name)).string = uname
        # For functions, the function name is also available as a variable inside.
        if isinstance(v, Function_Stmt):
            vspec = atmost_one(children_of_type(fdef, Specification_Part))
            vexec = atmost_one(children_of_type(fdef, Execution_Part))
            for nm in walk([n for n in [vspec, vexec] if n], Name):
                if nm.string != oname:
                    continue
                local_spec = search_local_alias_spec(nm)
                # We need to do a bit of surgery, since we have the `oname` inide the scope ending with `uname`.
                local_spec = local_spec[:-2] + local_spec[-1:]
                local_spec = tuple(x.split('_deconglobalfn_')[0] for x in local_spec)
                assert local_spec in ident_map and ident_map[local_spec] == v
                nm.string = uname

    return ast


def add_use_to_specification(scdef: SCOPE_OBJECT_TYPES, clause: str):
    specification_part = atmost_one(children_of_type(scdef, Specification_Part))
    if not specification_part:
        append_children(scdef, Specification_Part(get_reader(clause)))
    else:
        prepend_children(specification_part, Use_Stmt(clause))


def assign_globally_unique_variable_names(ast: Program, keepers: Set[str]) -> Program:
    """
    Update the variable declarations to have globally unique names.
    Precondition:
    1. All indirections are already removed from the program, except for the explicit renames.
    2. All public/private access statements were cleanly removed.
    """
    SUFFIX, COUNTER = 'var', 0

    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)

    name_collisions: Dict[str, int] = {k[-1]: 0 for k in ident_map.keys()}
    for k in ident_map.keys():
        name_collisions[k[-1]] += 1
    name_collisions: Set[str] = {k for k, v in name_collisions.items() if v > 1}

    # Make new unique names for the identifiers.
    uident_map: Dict[SPEC, str] = {}
    for k in ident_map.keys():
        if k[-1] in keepers:
            continue
        if k[-1] in name_collisions:
            uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        else:
            uname = k[-1]
        uident_map[k] = uname

    # PHASE 1.a: Remove all the places where any variable is imported.
    for use in walk(ast, Use_Stmt):
        mod_name = singular(children_of_type(use, Name)).string
        mod_spec = (mod_name,)
        olist = atmost_one(children_of_type(use, Only_List))
        if not olist:
            continue
        survivors = []
        for c in olist.children:
            assert isinstance(c, (Name, Rename))
            if isinstance(c, Name):
                src, tgt = c, c
            elif isinstance(c, Rename):
                _, src, tgt = c.children
            src, tgt = src.string, tgt.string
            tgt_spec = find_real_ident_spec(tgt, mod_spec, alias_map)
            assert tgt_spec in ident_map
            if not isinstance(ident_map[tgt_spec], Entity_Decl):
                survivors.append(c)
        if survivors:
            olist.items = survivors
            _reparent_children(olist)
        else:
            par = use.parent
            par.content = [c for c in par.children if c != use]
            _reparent_children(par)

    # PHASE 1.b: Replaces all the keywords when calling the functions. This must be done earlier than resolving other
    # references, because otherwise we cannot distinguish the two `kw`s in `fn(kw=kw)`.
    for kv in walk(ast, Actual_Arg_Spec):
        fref = kv.parent.parent
        if not isinstance(fref, (Function_Reference, Call_Stmt)):
            # Not a user defined function, so we are not renaming its internal variables anyway.
            continue
        callee, _ = fref.children
        if isinstance(callee, Intrinsic_Name):
            # Not a user defined function, so we are not renaming its internal variables anyway.
            continue
        cspec = search_real_local_alias_spec(callee, alias_map)
        cspec = ident_spec(alias_map[cspec])
        assert cspec
        k, _ = kv.children
        assert isinstance(k, Name)
        kspec = find_real_ident_spec(k.string, cspec, alias_map)
        if kspec not in uident_map:
            # If we haven't planned to rename it, then skip.
            continue
        k.string = uident_map[kspec]

    # PHASE 1.c: Replaces all the direct references.
    for vref in walk(ast, Name):
        if isinstance(vref.parent, Entity_Decl):
            # Do not change the variable declarations themselves just yet.
            continue
        vspec = search_real_local_alias_spec(vref, alias_map)
        if not vspec:
            # It was not a valid alias (e.g., a sturcture component).
            continue
        if not isinstance(alias_map[vspec], Entity_Decl):
            # Does not refer to a variable.
            continue
        edcl = alias_map[vspec]
        fdef = find_scope_ancestor(edcl)
        if isinstance(fdef, Function_Subprogram) and find_name_of_node(fdef) == find_name_of_node(edcl):
            # Function return variables must retain their names.
            continue

        scope_spec = find_scope_spec(vref)
        vspec = find_real_ident_spec(vspec[-1], scope_spec, alias_map)
        assert vspec in ident_map
        if vspec not in uident_map:
            # We have chosen to not rename it.
            continue
        uname = uident_map[vspec]
        vref.string = uname

        if len(vspec) > 2:
            # If the variable is not defined in a toplevel object, so we're done.
            continue
        assert len(vspec) == 2
        mod, _ = vspec
        if not isinstance(alias_map[(mod,)], Module_Stmt):
            # We can only import modules.
            continue

        # Find the nearest specification part (or lack thereof).
        scdef = alias_map[scope_spec].parent
        # Find out the current module name.
        cmod = scdef
        while not isinstance(cmod.parent, Program):
            cmod = cmod.parent
        if find_name_of_node(cmod) == mod:
            # Since this variable is already defined at the current module, there is nothing to import.
            continue
        add_use_to_specification(scdef, f"use {mod}, only: {uname}")

    # PHASE 1.d: Replaces all the literals where a variable can be used as a "kind".
    for lit in walk(ast, Real_Literal_Constant):
        val, kind = lit.children
        if not kind:
            continue
        # Strangely, we get a plain `str` instead of a `Name`.
        assert isinstance(kind, str)
        scope_spec = find_scope_spec(lit)
        kind_spec = search_real_ident_spec(kind, scope_spec, alias_map)
        if not kind_spec or kind_spec not in uident_map:
            continue
        uname = uident_map[kind_spec]
        lit.items = (val, uname)

        if len(kind_spec) > 2:
            # If the variable is not defined in a toplevel object, so we're done.
            continue
        assert len(kind_spec) == 2
        mod, _ = kind_spec
        if not isinstance(alias_map[(mod,)], Module_Stmt):
            # We can only import modules.
            continue

        # Find the nearest specification part (or lack thereof).
        scdef = alias_map[scope_spec].parent
        # Find out the current module name.
        cmod = scdef
        while not isinstance(cmod.parent, Program):
            cmod = cmod.parent
        if find_name_of_node(cmod) == mod:
            # Since this variable is already defined at the current module, there is nothing to import.
            continue
        add_use_to_specification(scdef, f"use {mod}, only: {uname}")

    # PHASE 1.e: Replaces actual variable names.
    for k, v in ident_map.items():
        if not isinstance(v, Entity_Decl):
            continue
        if k not in uident_map:
            # We have chosen to not rename it.
            continue
        oname, uname = k[-1], uident_map[k]
        fdef = find_scope_ancestor(v)
        if isinstance(fdef, Function_Subprogram) and find_name_of_node(fdef) == oname:
            # Function return variables must retain their names.
            continue
        singular(children_of_type(v, Name)).string = uname

    return ast


def _get_module_or_program_parts(mod: Union[Module, Main_Program]) \
        -> Tuple[
            Union[Module_Stmt, Program_Stmt],
            Optional[Specification_Part],
            Optional[Execution_Part],
            Optional[Module_Subprogram_Part],
        ]:
    # There must exist a module statment.
    stmt = singular(children_of_type(mod, Module_Stmt if isinstance(mod, Module) else Program_Stmt))
    # There may or may not exist a specification part.
    spec = list(children_of_type(mod, Specification_Part))
    assert len(spec) <= 1, f"A module/program cannot have more than one specification parts, found {spec} in {mod}"
    spec = spec[0] if spec else None
    # There may or may not exist an execution part.
    exec = list(children_of_type(mod, Execution_Part))
    assert len(exec) <= 1, f"A module/program cannot have more than one execution parts, found {spec} in {mod}"
    exec = exec[0] if exec else None
    # There may or may not exist a subprogram part.
    subp = list(children_of_type(mod, Module_Subprogram_Part))
    assert len(subp) <= 1, f"A module/program cannot have more than one subprogram parts, found {subp} in {mod}"
    subp = subp[0] if subp else None
    return stmt, spec, exec, subp


def consolidate_uses(ast: Program, alias_map: Optional[SPEC_TABLE] = None) -> Program:
    alias_map = alias_map or alias_specs(ast)
    for sp in reversed(walk(ast, Specification_Part)):
        use_map: Dict[str, Set[str]] = {}
        # Build the table to keep the use statements only if they are actually necessary.
        for nm in walk(sp.parent, Name):
            if isinstance(nm.parent, (Only_List, Rename)):
                # The identifiers in the use statements themselves are not of concern.
                continue
            # Where did we _really_ import `nm` from? Find the definition module.
            sc_spec = search_scope_spec(nm.parent)
            if not sc_spec:
                continue
            spec = search_real_ident_spec(nm.string, sc_spec, alias_map)
            if not spec or spec not in alias_map:
                continue
            if len(spec) != 2:
                continue
            if not isinstance(alias_map[spec[:-1]], Module_Stmt):
                # Objects defined inside a free function cannot be imported; so we must already be in that function.
                continue
            nm_mod = spec[0]
            # And which module are we in right now?
            sp_mod = sp
            while sp_mod and not isinstance(sp_mod, (Module, Main_Program)):
                sp_mod = sp_mod.parent
            if sp_mod and nm_mod == find_name_of_node(sp_mod):
                # Nothing to do if the object is defined in the current scope and not imported.
                continue
            if nm.string == spec[-1]:
                u = nm.string
            else:
                u = f"{nm.string} => {spec[-1]}"
            if nm_mod not in use_map:
                use_map[nm_mod] = set()
            use_map[nm_mod].add(u)
        # Build new use statements.
        nuses: List[Use_Stmt] = [Use_Stmt(f"use {k}, only: {', '.join(sorted(use_map[k]))}") for k in use_map.keys()]
        # Remove the old ones, and prepend the new ones.
        sp.content = nuses + [c for c in sp.children if not isinstance(c, Use_Stmt)]
        _reparent_children(sp)
    return ast


def _prune_branches_in_ifblock(ib: If_Construct, alias_map: SPEC_TABLE):
    ifthen = ib.children[0]
    assert isinstance(ifthen, If_Then_Stmt)
    cond, = ifthen.children
    cval = _const_eval_basic_type(cond, alias_map)
    if cval is None:
        return
    assert isinstance(cval, np.bool_)

    elifat = [idx for idx, c in enumerate(ib.children) if isinstance(c, (Else_If_Stmt, Else_Stmt))]
    if cval:
        cut = elifat[0] if elifat else -1
        actions = ib.children[1:cut]
        replace_node(ib, actions)
        return
    elif not elifat:
        remove_self(ib)
        return

    cut = elifat[0]
    cut_cond = ib.children[cut]
    if isinstance(cut_cond, Else_Stmt):
        actions = ib.children[cut + 1:-1]
        replace_node(ib, actions)
        return

    isinstance(cut_cond, Else_If_Stmt)
    cut_cond, _ = cut_cond.children
    remove_children(ib, ib.children[1:(cut + 1)])
    set_children(ifthen, (cut_cond,))
    _prune_branches_in_ifblock(ib, alias_map)


def _prune_branches_in_ifstmt(ib: If_Stmt, alias_map: SPEC_TABLE):
    cond, actions = ib.children
    cval = _const_eval_basic_type(cond, alias_map)
    if cval is None:
        return
    assert isinstance(cval, np.bool_)
    if cval:
        replace_node(ib, actions)
    else:
        remove_self(ib)
    expart = ib.parent
    if isinstance(expart, Execution_Part) and not expart.children:
        remove_self(expart)


def prune_branches(ast: Program) -> Program:
    alias_map = alias_specs(ast)
    for ib in walk(ast, If_Construct):
        _prune_branches_in_ifblock(ib, alias_map)
    for ib in walk(ast, If_Stmt):
        _prune_branches_in_ifstmt(ib, alias_map)
    return ast


def numpy_type_to_literal(val: NUMPY_TYPES) -> Union[LITERAL_TYPES]:
    if isinstance(val, np.bool_):
        val = Logical_Literal_Constant('.true.' if val else '.false.')
    elif isinstance(val, NUMPY_INTS):
        bytez = _count_bytes(type(val))
        if val < 0:
            val = Signed_Int_Literal_Constant(f"{val}" if bytez == 4 else f"{val}_{bytez}")
        else:
            val = Int_Literal_Constant(f"{val}" if bytez == 4 else f"{val}_{bytez}")
    elif isinstance(val, NUMPY_REALS):
        bytez = _count_bytes(type(val))
        valstr = str(val)
        if bytez == 8:
            if 'e' in valstr:
                valstr = valstr.replace('e', 'D')
            else:
                valstr = f"{valstr}D0"
        if val < 0:
            val = Signed_Real_Literal_Constant(valstr)
        else:
            val = Real_Literal_Constant(valstr)
    return val


def const_eval_nodes(ast: Program) -> Program:
    EXPRESSION_CLASSES = (
        LITERAL_CLASSES, Expr, Add_Operand, Or_Operand, Mult_Operand, Level_2_Expr, Level_3_Expr, Level_4_Expr,
        Level_5_Expr, Intrinsic_Function_Reference)

    alias_map = alias_specs(ast)

    def _const_eval_node(n: Base) -> bool:
        val = _const_eval_basic_type(n, alias_map)
        if val is None:
            return False
        assert not np.isnan(val)
        val = numpy_type_to_literal(val)
        replace_node(n, val)
        return True

    for asgn in reversed(walk(ast, Assignment_Stmt)):
        lv, op, rv = asgn.children
        assert op == '='
        _const_eval_node(rv)
    for expr in reversed(walk(ast, EXPRESSION_CLASSES)):
        # Try to const-eval the expression.
        if _const_eval_node(expr):
            # If the node is successfully replaced, then nothing else to do.
            continue
        # Otherwise, try to at least replace the names with the literal values.
        for nm in reversed(walk(expr, Name)):
            _const_eval_node(nm)
    for knode in reversed(walk(ast, Kind_Selector)):
        _, kind, _ = knode.children
        _const_eval_node(kind)

    NON_EXPRESSION_CLASSES = (
        Explicit_Shape_Spec, Loop_Control, Call_Stmt, Function_Reference, Initialization, Component_Initialization)
    for node in reversed(walk(ast, NON_EXPRESSION_CLASSES)):
        for nm in reversed(walk(node, Name)):
            _const_eval_node(nm)

    return ast


@dataclass
class ConstTypeInjection:
    scope_spec: Optional[SPEC]  # Only replace within this scope object.
    type_spec: SPEC  # The root config derived type's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitue with. The injected literal's type will match the type of the original.


@dataclass
class ConstInstanceInjection:
    scope_spec: Optional[SPEC]  # Only replace within this scope object.
    root_spec: SPEC  # The root config object's spec (w.r.t. where it is defined)
    component_spec: Optional[SPEC]  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitue with. The injected literal's type will match the type of the original.


ConstInjection = Union[ConstTypeInjection, ConstInstanceInjection]


def _val_2_lit(val: str, type_spec: SPEC) -> LITERAL_TYPES:
    val = str(val).lower()
    if type_spec == ('INTEGER1',):
        val = np.int8(val)
    elif type_spec == ('INTEGER2',):
        val = np.int16(val)
    elif type_spec == ('INTEGER4',):
        val = np.int32(val)
    elif type_spec == ('INTEGER8',):
        val = np.int64(val)
    elif type_spec == ('REAL4',):
        val = np.float32(val)
    elif type_spec == ('REAL8',):
        val = np.float64(val)
    elif type_spec == ('LOGICAL',):
        assert val in {'true', 'false'}
        val = np.bool_(val == 'true')
    else:
        raise NotImplementedError(
            f"{val} cannot be parsed as the target literal type: {type_spec}")
    return numpy_type_to_literal(val)


def _find_real_ident_spec(node: Name, alias_map: SPEC_TABLE) -> SPEC:
    loc = search_real_local_alias_spec(node, alias_map)
    assert loc
    return ident_spec(alias_map[loc])


def _lookup_dataref(dr: Data_Ref, alias_map: SPEC_TABLE) -> Optional[Tuple[Name, SPEC]]:
    scope_spec = find_scope_spec(dr)
    root, root_tspec, rest = _dataref_root(dr, scope_spec, alias_map)
    while not isinstance(root, Name):
        root, root_tspec, nurest = _dataref_root(root, scope_spec, alias_map)
        rest = nurest + rest
    return root, rest


def _find_matching_item(items: List[ConstInjection], dr: Data_Ref, alias_map: SPEC_TABLE) -> Optional[ConstInjection]:
    root, rest = _lookup_dataref(dr, alias_map)
    # NOTE: We should replace only when it is not an output of the function. However, here we pass the responsibilty to
    # the user to provide valid injections.
    if not all(isinstance(c, Name) for c in rest):
        return None
    root_id_spec = _find_real_ident_spec(root, alias_map)
    scope_spec = find_scope_spec(dr)
    comp_tspec = find_type_dataref(dr, scope_spec, alias_map)
    for item in items:
        if isinstance(item, ConstTypeInjection):
            # `item.component_spec` must be a precise suffix in the data-ref, and everything before that must
            # precisely match `item.type_spec`.
            if len(item.component_spec) > len(rest):
                continue
            pre, c_rest = rest[:-len(item.component_spec)], rest[-len(item.component_spec):]
            comp_spec: SPEC = tuple(c.string for c in c_rest)
            if comp_spec[:-1] != item.component_spec[:-1]:
                # All but the last component must exactly match in all case.
                continue
            if comp_tspec.alloc:
                if f"{comp_spec[-1]}_a" != item.component_spec[-1]:
                    # Allocatable array's special variable didn't match either.
                    continue
            else:
                if comp_spec[-1] != item.component_spec[-1]:
                    # Otherwise the last component must exactly match too.
                    continue
            comp = Data_Ref(' % '.join(tuple(p.string for p in (root,) + pre)))
            ctspec = find_type_dataref(comp, scope_spec, alias_map)
            if ctspec.spec != item.type_spec:
                continue
        elif isinstance(item, ConstInstanceInjection):
            # This is a simpler case, where the object must match `item.root_spec`, and the entire component
            # parts of the data-ref must precisely match `item.component_spec`.
            comp_spec: SPEC = tuple(c.string for c in rest)
            if root_id_spec != item.root_spec or comp_spec != item.component_spec:
                continue

        return item
    return None


def inject_const_evals(ast: Program,
                       inject_consts: Optional[List[ConstInjection]] = None) -> Program:
    inject_consts = inject_consts or []
    alias_map = alias_specs(ast)

    TOPLEVEL_SPEC = ('*',)

    items_by_scopes = {}
    for item in inject_consts:
        scope_spec = item.scope_spec or TOPLEVEL_SPEC
        if scope_spec not in items_by_scopes:
            items_by_scopes[scope_spec] = []
        items_by_scopes[scope_spec].append(item)

        # Validations.
        if item.scope_spec:
            assert item.scope_spec in alias_map
        if isinstance(item, ConstTypeInjection):
            assert item.type_spec in alias_map
            tdef = alias_map[item.type_spec].parent
            assert isinstance(tdef, Derived_Type_Def)
        elif isinstance(item, ConstInstanceInjection):
            assert item.root_spec in alias_map
            rdef = alias_map[item.root_spec]
            assert isinstance(rdef, Entity_Decl)

    for scope_spec, items in items_by_scopes.items():
        if scope_spec == TOPLEVEL_SPEC:
            scope = ast
        else:
            scope = alias_map[scope_spec].parent

        drefs: List[Data_Ref] = [dr for dr in walk(scope, Data_Ref)
                                 if find_type_dataref(dr, find_scope_spec(dr), alias_map).spec != ('CHARACTER',)]
        names: List[Name] = walk(scope, Name)
        allocateds: List[Intrinsic_Function_Reference] = [c for c in walk(scope, Intrinsic_Function_Reference)
                                                          if c.children[0].string == 'ALLOCATED']

        # Ignore the special variables related to array dimensions, since we don't handle them here.
        alloc_items = [item for item in items if item.component_spec[-1].endswith('_a')]
        items = [item for item in items
                 if not (item.component_spec[-1].endswith('_s') or item.component_spec[-1].endswith('_a'))]
        item_inst_root_specs: Set[SPEC] = {item.root_spec for item in items
                                           if isinstance(item, ConstInstanceInjection)}  # For speedup later.

        for al in allocateds:
            _, args = al.children
            assert args and len(args.children) == 1
            arr, = args.children
            if not isinstance(arr, Data_Ref):
                # TODO: We don't support anything else for now.
                continue
            item = _find_matching_item(alloc_items, arr, alias_map)
            if not item:
                continue
            replace_node(al, _val_2_lit(item.value, ('LOGICAL',)))

        for dr in drefs:
            if isinstance(dr.parent, Assignment_Stmt):
                # We cannot replace on the LHS of an assignment.
                lv, _, _ = dr.parent.children
                if lv == dr:
                    continue
            item = _find_matching_item(items, dr, alias_map)
            if not item:
                continue
            replace_node(dr, _val_2_lit(item.value, find_type_dataref(dr, find_scope_spec(dr), alias_map).spec))

        for nm in names:
            # We can also directly inject variables' values with `ConstInstanceInjection`.
            if isinstance(nm.parent, (Entity_Decl, Only_List)):
                # We don't want to replace the values in their declarations or imports, but only where their
                # values are being used.
                continue
            loc = search_real_local_alias_spec(nm, alias_map)
            if not loc or not isinstance(alias_map[loc], Entity_Decl):
                continue
            spec = ident_spec(alias_map[loc])
            if spec not in item_inst_root_specs:
                continue
            for item in items:
                if (not isinstance(item, ConstInstanceInjection)
                        or item.component_spec is not None
                        or spec != item.root_spec):
                    # To inject variables' values, it has to be a `ConstInstanceInjection` without a component and point
                    # to the (scalar) variable identified by `nm`.
                    continue
                tspec = find_type_of_entity(alias_map[loc], alias_map)
                # NOTE: We should replace only when it is not an output of the function. However, here we pass the
                # responsibilty to the user to provide valid injections.
                replace_node(nm, _val_2_lit(item.value, tspec.spec))
                break
    return ast


def lower_identifier_names(ast: Program) -> Program:
    for nm in walk(ast, Name):
        nm.string = nm.string.lower()
    return ast


def create_global_initializers(ast: Program, entry_points: List[SPEC]) -> Program:
    # TODO: Ordering of the initializations may matter, but for that we need to find how Fortran's global initialization
    # works and then reorder the initialization calls appropriately.

    ident_map = identifier_specs(ast)
    GLOBAL_INIT_FN_NAME = 'global_init_fn'
    if (GLOBAL_INIT_FN_NAME,) in ident_map:
        # We already have the global initialisers.
        return ast
    alias_map = alias_specs(ast)

    created_init_fns: Set[str] = set()
    used_init_fns: Set[str] = set()

    def _make_init_fn(fn_name: str, inited_vars: List[SPEC], this: Optional[SPEC]):
        if this:
            assert this in ident_map and isinstance(ident_map[this], Derived_Type_Stmt)
            box = ident_map[this]
            while not isinstance(box, Specification_Part):
                box = box.parent
            box = box.parent
            assert isinstance(box, Module)
            sp_part = atmost_one(children_of_type(box, Module_Subprogram_Part))
            if not sp_part:
                rest, end_mod = box.children[:-1], box.children[-1]
                assert isinstance(end_mod, End_Module_Stmt)
                sp_part = Module_Subprogram_Part('contains')
                set_children(box, rest + [sp_part, end_mod])
            box = sp_part
        else:
            box = ast

        uses, execs = [], []
        for v in inited_vars:
            var = ident_map[v]
            mod = var
            while not isinstance(mod, Module):
                mod = mod.parent
            if not this:
                uses.append(f"use {find_name_of_node(mod)}, only: {find_name_of_stmt(var)}")
            var_t = find_type_of_entity(var, alias_map)
            if var_t.spec in type_defs:
                if var_t.shape:
                    # TODO: We need to create loops for this initialization.
                    continue
                var_init, _ = type_defs[var_t.spec]
                tmod = ident_map[var_t.spec]
                while not isinstance(tmod, Module):
                    tmod = tmod.parent
                uses.append(f"use {find_name_of_node(tmod)}, only: {var_init}")
                execs.append(f"call {var_init}({'this % ' if this else ''}{find_name_of_node(var)})")
                used_init_fns.add(var_init)
            else:
                name, _, _, init_val = var.children
                assert init_val
                execs.append(f"{'this % ' if this else ''}{name.tofortran()}{init_val.tofortran()}")
        fn_args = 'this' if this else ''
        uses_stmts = '\n'.join(uses)
        this_decl = f"type({this[-1]}) :: this" if this else ''
        execs_stmts = '\n'.join(execs)
        init_fn = f"""
subroutine {fn_name}({fn_args})
  {uses_stmts}
  implicit none
  {this_decl}
  {execs_stmts}
end subroutine {fn_name}
"""
        init_fn = Subroutine_Subprogram(get_reader(init_fn.strip()))
        append_children(box, init_fn)
        created_init_fns.add(fn_name)

    type_defs: List[SPEC] = [k for k in ident_map.keys() if isinstance(ident_map[k], Derived_Type_Stmt)]
    type_defs: Dict[SPEC, Tuple[str, List[SPEC]]] =\
        {k: (f"type_init_{k[-1]}_{idx}", []) for idx, k in enumerate(type_defs)}
    for k, v in ident_map.items():
        if not isinstance(v, Component_Decl) or not atmost_one(children_of_type(v, Component_Initialization)):
            continue
        td = k[:-1]
        assert td in ident_map and isinstance(ident_map[td], Derived_Type_Stmt)
        if td not in type_defs:
            type_init_fn = f"type_init_{td[-1]}_{len(type_defs)}"
            type_defs[td] = type_init_fn, []
        type_defs[td][1].append(k)
    for t, v in type_defs.items():
        init_fn_name, comps = v
        if comps:
            _make_init_fn(init_fn_name, comps, t)

    global_inited_vars: List[SPEC] = [
        k for k, v in ident_map.items()
        if isinstance(v, Entity_Decl) and not find_type_of_entity(v, alias_map).const
           and (find_type_of_entity(v, alias_map).spec in type_defs or atmost_one(children_of_type(v, Initialization)))
           and search_scope_spec(v) and isinstance(alias_map[search_scope_spec(v)], Module_Stmt)
    ]
    if global_inited_vars:
        _make_init_fn(GLOBAL_INIT_FN_NAME, global_inited_vars, None)
        for ep in entry_points:
            assert ep in ident_map
            fn = ident_map[ep]
            if not isinstance(fn, (Function_Stmt, Subroutine_Stmt)):
                # Not a function (or subroutine), so there is nothing to exectue here.
                continue
            ex = atmost_one(children_of_type(fn.parent, Execution_Part))
            if not ex:
                # The function does nothing. We could still initialize, but there is no point.
                continue
            init_call = Call_Stmt(f"call {GLOBAL_INIT_FN_NAME}")
            prepend_children(ex, init_call)
            used_init_fns.add(GLOBAL_INIT_FN_NAME)

    unused_init_fns = created_init_fns - used_init_fns
    for fn in walk(ast, Subroutine_Subprogram):
        if find_name_of_node(fn) in unused_init_fns:
            remove_self(fn)

    return ast


def convert_data_statements_into_assignments(ast: Program) -> Program:
    # TODO: Data statements have unusual syntax even within Fortran and not everything is covered here yet.
    alias_map = alias_specs(ast)

    for spart in walk(ast, Specification_Part):
        box = spart.parent
        xpart = atmost_one(children_of_type(box, Execution_Part))
        for dst in reversed(walk(spart, Data_Stmt)):
            repls: List[Assignment_Stmt] = []
            for ds in dst.children:
                assert isinstance(ds, Data_Stmt_Set)
                varz, valz = ds.children
                varz, valz = varz.children, valz.children
                assert len(varz) == len(valz)
                for k, v in zip(varz, valz):
                    scope_spec = find_scope_spec(k)
                    kroot, ktyp, rest = _dataref_root(k, scope_spec, alias_map)
                    if isinstance(v, Data_Stmt_Value):
                        repeat, elem = v.children
                        repeat = 1 if not repeat else int(_const_eval_basic_type(repeat, alias_map))
                        assert repeat
                    else:
                        elem = v
                    # TODO: Support other types of data expressions.
                    assert isinstance(elem, LITERAL_CLASSES),\
                        f"only supports literal values in data data statements: {elem}"
                    if ktyp.shape:
                        if rest:
                            assert len(rest) == 1 and isinstance(rest[0], Section_Subscript_List)
                            subsc = rest[0].tofortran()
                        else:
                            subsc = ','.join([':' for _ in ktyp.shape])
                        repls.append(Assignment_Stmt(f"{kroot.string}({subsc}) = {elem.tofortran()}"))
                    else:
                        assert isinstance(k, Name)
                        repls.append(Assignment_Stmt(f"{k.string} = {elem.tofortran()}"))
            remove_self(dst)
            if not xpart:
                # NOTE: Since the function does nothing at all (hence, no execution part), don't bother with the inits.
                continue
            prepend_children(xpart, repls)

    return ast
