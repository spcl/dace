import math
import re
from typing import Union, Tuple, Dict, Optional, List, Iterable, Set

import networkx as nx
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
    Enumerator_List, Actual_Arg_Spec_List, Section_Subscript_List
from fparser.two.Fortran2008 import Procedure_Stmt, Type_Declaration_Stmt
from fparser.two.utils import Base, walk

from dace.frontend.fortran import ast_utils as ast_utils

NAMED_STMTS_OF_INTEREST_TYPES = Union[
    Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt, Component_Decl, Entity_Decl,
    Specific_Binding, Generic_Binding, Interface_Stmt]
SPEC = Tuple[str, ...]
SPEC_TABLE = Dict[SPEC, NAMED_STMTS_OF_INTEREST_TYPES]


class TYPE_SPEC:
    NO_ATTRS = ''

    def __init__(self,
                 spec: Union[str, SPEC],
                 attrs: str = NO_ATTRS):
        if isinstance(spec, str):
            spec = (spec,)
        self.spec: SPEC = spec
        self.shape: Tuple[str, ...] = self._parse_shape(attrs)
        self.optional: bool = 'OPTIONAL' in attrs
        self.inp: bool = 'INTENT(IN)' in attrs or 'INTENT(INOUT)' in attrs
        self.out: bool = 'INTENT(OUT)' in attrs or 'INTENT(INOUT)' in attrs
        self.keyword: Optional[str] = None

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
        if not attrs:
            return f"{self.spec}"
        return f"{self.spec}[{' | '.join(attrs)}]"


ENTRY_POINT_OBJECT_TYPES = Union[Main_Program, Subroutine_Subprogram, Function_Subprogram]


def find_name_of_stmt(node: NAMED_STMTS_OF_INTEREST_TYPES) -> Optional[str]:
    """Find the name of the statement if it has one. For anonymous blocks, return `None`."""
    if isinstance(node, Specific_Binding):
        # Ref: https://github.com/stfc/fparser/blob/8c870f84edbf1a24dfbc886e2f7226d1b158d50b/src/fparser/two/Fortran2003.py#L2504
        iname, mylist, dcolon, bname, pname = node.children
        name = bname
    elif isinstance(node, Interface_Stmt):
        name, = node.children
    else:
        # TODO: Test out other type specific ways of finding names.
        name = ast_utils.singular(ast_utils.children_of_type(node, Name))
    if name:
        name = name.string
    return name


def find_name_of_node(node: Base) -> Optional[str]:
    """Find the name of the general node if it has one. For anonymous blocks, return `None`."""
    if isinstance(node, NAMED_STMTS_OF_INTEREST_TYPES):
        return find_name_of_stmt(node)
    stmt = ast_utils.atmost_one(ast_utils.children_of_type(node, NAMED_STMTS_OF_INTEREST_TYPES))
    if stmt:
        return find_name_of_stmt(stmt)
    return None


def find_named_ancester(node: Base) -> Optional[NAMED_STMTS_OF_INTEREST_TYPES]:
    NAMED_ANCESTOR_STMT_TYPES = Union[Program_Stmt, Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt]
    anc = node.parent
    while anc:
        stmt = ast_utils.atmost_one(ast_utils.children_of_type(anc, NAMED_ANCESTOR_STMT_TYPES))
        if stmt:
            return stmt
        anc = anc.parent
    return None


def ident_spec(node: NAMED_STMTS_OF_INTEREST_TYPES) -> SPEC:
    def _ident_spec(_node: NAMED_STMTS_OF_INTEREST_TYPES) -> SPEC:
        """
        Constuct a list of identifier strings that can uniquely determine it through the entire AST.
        """
        ident_base = (find_name_of_stmt(_node),)
        # Find the next named ancestor.
        anc = find_named_ancester(_node.parent)
        if not anc:
            return ident_base
        assert isinstance(anc, NAMED_STMTS_OF_INTEREST_TYPES)
        return _ident_spec(anc) + ident_base

    spec = _ident_spec(node)
    # The last part of the spec cannot be nothing, because we cannot refer to the anonymous blocks.
    assert spec and spec[-1]
    # For the rest, the anonymous blocks puts their contend onto their parents.
    spec = tuple(c for c in spec if c)
    return spec


def identifier_specs(ast: Program) -> SPEC_TABLE:
    """
    Maps each identifier of interest in `ast` to its associated node that defines it.
    """
    ident_map: SPEC_TABLE = {}
    for stmt in walk(ast, NAMED_STMTS_OF_INTEREST_TYPES):
        assert isinstance(stmt, NAMED_STMTS_OF_INTEREST_TYPES)
        if isinstance(stmt, Interface_Stmt) and not find_name_of_stmt(stmt):
            # There can be anonymous blocks, e.g., interface blocks, which cannot be identified.
            continue
        ident_map[ident_spec(stmt)] = stmt
    return ident_map


def alias_specs(ast: Program):
    """
    Maps each "alias-type" identifier of interest in `ast` to its associated node that defines it.
    """
    ident_map = identifier_specs(ast)
    alias_map: SPEC_TABLE = {k: v for k, v in ident_map.items()}

    for stmt in walk(ast, Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(stmt, Name)).string
        mod_spec = (mod_name,)

        scope_spec = find_scope_spec(stmt)
        use_spec = scope_spec + (mod_name,)

        assert mod_spec in ident_map
        # The module's name cannot be used as an identifier in this scope anymore, so just point to the module.
        alias_map[use_spec] = ident_map[mod_spec]

        olist = ast_utils.atmost_one(ast_utils.children_of_type(stmt, 'Only_List'))
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


def search_scope_spec(node: Base) -> Optional[SPEC]:
    scope = find_named_ancester(node.parent)
    if not scope:
        return None
    return ident_spec(scope)


def find_scope_spec(node: Base) -> SPEC:
    spec = search_scope_spec(node)
    assert spec, f"cannot find scope for: ```\n{node.tofortran()}```"
    return spec


def search_local_alias_spec(node: Name) -> Optional[SPEC]:
    name, par = node.string, node.parent
    scope_spec = search_scope_spec(node.parent)
    if not scope_spec:
        return None
    local_spec = scope_spec + (name,)
    if isinstance(par, (Part_Ref, Data_Ref)):
        # If we are in a data-ref then we need to get to the root.
        if isinstance(par.parent, Data_Ref):
            par = par.parent
        assert not isinstance(par.parent, Data_Ref)
        # TODO: Add ref.
        par, _ = par.children[0], par.children[1:]
        if isinstance(par, Part_Ref):
            # TODO: Add ref.
            par, _ = par.children[0], par.children[1:]
        assert isinstance(par, Name)
        if par != node:
            return None
    return local_spec


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
    while anc and not ast_utils.atmost_one(
            ast_utils.children_of_type(anc, (Intrinsic_Type_Spec, Declaration_Type_Spec))):
        anc = anc.parent
    return anc


def _eval_selected_int_kind(p: int) -> int:
    # Copied logic from `replace_int_kind()` elsewhere in the project.
    return int(math.ceil((math.log2(10 ** p) + 1) / 8))


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
        init = ast_utils.atmost_one(ast_utils.children_of_type(decl, Initialization))
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


def find_type_of_entity(node: Entity_Decl, alias_map: SPEC_TABLE) -> Optional[TYPE_SPEC]:
    anc = _find_type_decl_node(node)
    if not anc:
        return None
    # TODO: Add ref.
    typ, attrs, _ = anc.children
    assert isinstance(typ, (Intrinsic_Type_Spec, Declaration_Type_Spec))
    attrs = attrs.tofortran() if attrs else ''

    extra_dim = None
    if isinstance(typ, Intrinsic_Type_Spec):
        typ_name, kind = typ.children
        # TODO: How should we handle character lengths? Just treat it as an extra dimension?
        if isinstance(kind, Length_Selector):
            extra_dim = (':',)
        elif isinstance(kind, Kind_Selector):
            _, kind, _ = kind.children
            kind = _const_eval_int(kind, alias_map)
            if kind:
                # TODO: We should always be able to evlauate a kind. I.e., this should be an assert.
                # TODO: Perhaps not attach it as a string?
                # If not a default kind, attach it to the type.
                typ_name = f"{typ_name}{kind}"
        spec = (typ_name,)
    elif isinstance(typ, Declaration_Type_Spec):
        _, typ_name = typ.children
        spec = find_real_ident_spec(typ_name.string, ident_spec(node), alias_map)

    # TODO: This `attrs` manipulation is a hack. We should design the type specs better.
    # TODO: Add ref.
    attrs = [attrs] if attrs else []
    _, shape, _, _ = node.children
    if shape is not None:
        attrs.append(f"DIMENSION({shape.tofortran()})")
    attrs = ', '.join(attrs)
    tspec = TYPE_SPEC(spec, attrs)
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
    assert root_type
    return root_type, rest


def find_dataref_component_spec(dref: Union[Name, Data_Ref], scope_spec: SPEC, alias_map: SPEC_TABLE) -> SPEC:
    # The root must have been a typed object.
    root_type, rest = _dataref_root(dref, scope_spec, alias_map)

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
    root_type, rest = _dataref_root(dref, scope_spec, alias_map)
    cur_type = root_type
    for comp in rest:
        assert isinstance(comp, (Name, Part_Ref))
        if isinstance(comp, Part_Ref):
            # TODO: Add ref.
            part_name, subsc = comp.children
            comp_spec = find_real_ident_spec(part_name.string, cur_type.spec, alias_map)
            assert comp_spec in alias_map, f"cannot find {comp_spec} / {dref} in {scope_spec}"
            cur_type = find_type_of_entity(alias_map[comp_spec], alias_map)
            if not cur_type.shape:
                # The object was not an array in the first place.
                assert not subsc, f"{cur_type} / {part_name}, {cur_type.spec}, {comp}"
            elif subsc:
                # TODO: This is a hack to deduce a array type instead of scalar.
                # We may have subscripted away all the dimensions.
                cur_type.shape = tuple(s.tofortran() for s in subsc.children if ':' in s.tofortran())
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
        typedef_stmt: Derived_Type_Stmt = ast_utils.singular(ast_utils.children_of_type(typedef, Derived_Type_Stmt))
        typedef_name: str = ast_utils.singular(ast_utils.children_of_type(typedef_stmt, Type_Name)).string
        proc_spec.insert(0, typedef_name)

        # TODO: Generalize.
        # We assume that the type is defined inside a module (i.e., not another subprogram).
        mod: Module = typedef.parent.parent
        mod_stmt: Module_Stmt = ast_utils.singular(ast_utils.children_of_type(mod, (Module_Stmt, Program_Stmt)))
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


def interface_specs(ast: Program) -> Dict[SPEC, Tuple[SPEC, ...]]:
    iface_map: Dict[SPEC, Tuple[SPEC, ...]] = {}

    # First, we deal with named interface blocks.
    for ifs in walk(ast, Interface_Stmt):
        assert isinstance(ifs, Interface_Stmt)
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        name = find_name_of_stmt(ifs)
        if not name:
            # Only named interfaces can be called.
            continue
        ifspec = scope_spec + (name,)

        # Get the spec of all the callable things in this block that may end up as a resolution for this interface.
        fns: List[str] = []
        for fn in walk(ib, (Function_Stmt, Subroutine_Stmt, Procedure_Stmt)):
            if isinstance(fn, (Function_Stmt, Subroutine_Stmt)):
                fns.append(find_name_of_stmt(fn))
            elif isinstance(fn, Procedure_Stmt):
                for nm in walk(fn, Name):
                    fns.append(nm.string)

        fn_specs = tuple(scope_spec + (f,) for f in fns)
        iface_map[ifspec] = fn_specs

    # Then, we try to put anonymous interface blocks' content onto their parents' scopes, but only if that identifier
    # is not already taken.
    for ifs in walk(ast, Interface_Stmt):
        assert isinstance(ifs, Interface_Stmt)
        ib = ifs.parent
        scope_spec = find_scope_spec(ib)
        name = find_name_of_stmt(ifs)
        if name:
            # Only anonymous interface blocks.
            continue

        # Get the spec of all the callable things in this block that may end up as a resolution for this interface.
        for fn in walk(ib, (Function_Stmt, Subroutine_Stmt)):
            fn_spec = ident_spec(fn)
            if fn_spec in iface_map:
                continue
            iface_map[fn_spec] = (fn_spec,)

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
                    pr_spec = find_real_ident_spec(pr_name.string, scope_spec, alias_map)
                    if isinstance(alias_map[pr_spec], (Function_Stmt, Interface_Stmt)):
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
            p = ast_utils.singular(ast_utils.children_of_type(p, (Module_Stmt, Program_Stmt)))
            return find_name_of_stmt(p)

    g = nx.DiGraph()  # An edge u->v means u should come before v, i.e., v depends on u.
    for c in ast.children:
        g.add_node(_get_module(c))

    for u in walk(ast, Use_Stmt):
        u_name = ast_utils.singular(ast_utils.children_of_type(u, Name)).string
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
        replace_node(en, [Type_Declaration_Stmt(f"integer, parameter :: {k} = {v}") for k, v in en_dict.items()])
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
                return TYPE_SPEC('INTEGER')
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
    iface_map = interface_specs(ast)

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
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, Specification_Part))

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
        mod_name = ast_utils.singular(ast_utils.children_of_type(use, Name)).string
        mod_spec = (mod_name,)
        olist = ast_utils.atmost_one(ast_utils.children_of_type(use, 'Only_List'))
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

    # We also remove any access statement that makes these interfaces public/private.
    for acc in walk(ast, Access_Stmt):
        # TODO: Add ref.
        kind, alist = acc.children
        if not alist:
            continue
        scope_spec = find_scope_spec(acc)

        survivors = []
        for c in alist.children:
            assert isinstance(c, Name)
            c_spec = scope_spec + (c.string,)
            assert c_spec in alias_map
            if not isinstance(alias_map[c_spec], Interface_Stmt):
                # Leave the non-interface usages alone.
                survivors.append(c)

        if survivors:
            alist.items = survivors
            _reparent_children(alist)
        else:
            remove_self(acc)

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
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, Name)).string.lower()
        else:
            subp = list(ast_utils.children_of_type(ast, Subroutine_Subprogram))
            assert subp
            stmt = ast_utils.singular(ast_utils.children_of_type(subp[0], Subroutine_Stmt))
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, Name)).string.lower()

        # Find the nearest execution and its correpsonding specification parts.
        execution_part = callsite.parent
        while not isinstance(execution_part, Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, Specification_Part))

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


def prune_unused_objects(ast: Program,
                         keepers: List[Union[Module, Main_Program, Subroutine_Subprogram, Function_Subprogram]]) \
        -> Program:
    """
    Precondition: All the indirections have been taken out of the program.
    """
    PRUNABLE_OBJECT_TYPES = Union[Subroutine_Subprogram, Function_Subprogram, Derived_Type_Def]

    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)
    survivors: Set[SPEC] = set()

    def _keep_from(node: Base):
        for nm in walk(node, Name):
            ob = nm.parent
            sc_spec = search_scope_spec(ob)
            if not sc_spec:
                continue

            for j in reversed(range(len(sc_spec))):
                anc = sc_spec[:j + 1]
                if anc in survivors:
                    continue
                survivors.add(anc)
                anc_node = alias_map[anc].parent
                if isinstance(anc_node, PRUNABLE_OBJECT_TYPES):
                    _keep_from(anc_node)

            to_keep = search_real_ident_spec(nm.string, sc_spec, alias_map)
            if not to_keep or to_keep not in alias_map or to_keep in survivors:
                # If we don't have a valid `to_keep` or `to_keep` is already kept, we move on.
                continue
            survivors.add(to_keep)
            keep_node = alias_map[to_keep].parent
            if isinstance(keep_node, PRUNABLE_OBJECT_TYPES):
                _keep_from(keep_node)

    for k in keepers:
        _keep_from(k)

    # We keep them sorted so that the parent scopes are handled earlier.
    killed: Set[SPEC] = set()
    for ns in list(sorted(set(ident_map.keys()) - survivors)):
        ns_node = ident_map[ns].parent
        if not isinstance(ns_node, PRUNABLE_OBJECT_TYPES):
            continue
        for i in range(len(ns) - 1):
            anc_spec = ns[:i + 1]
            if anc_spec in killed:
                killed.add(ns)
                break
        if ns in killed:
            continue
        remove_self(ns_node)
        killed.add(ns)

    # We also remove any access statement that makes the killed objects public/private.
    for acc in walk(ast, Access_Stmt):
        # TODO: Add ref.
        kind, alist = acc.children
        if not alist:
            continue
        scope_spec = find_scope_spec(acc)
        good_children = []
        for c in alist.children:
            assert isinstance(c, Name)
            c_spec = find_real_ident_spec(c.string, scope_spec, alias_map)
            assert c_spec in ident_map
            if c_spec not in killed:
                good_children.append(c)
        if good_children:
            alist.items = good_children
            _reparent_children(alist)
        else:
            remove_self(acc)

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


def assign_globally_unique_names(ast: Program, keepers: Set[SPEC]) -> Program:
    """
    Precondition: All indirections are already removed from the program, except for the explicit renames.
    TODO: Make structure names unique too. And possibly variables?
    """
    SUFFIX, COUNTER = 'deconglobal', 0

    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)

    # Make new unique names for the identifiers.
    uident_map: Dict[SPEC, str] = {}
    for k in ident_map.keys():
        if k in keepers:
            continue
        uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        uident_map[k] = uname

    # PHASE 1: Update the callsites for functions (and interchangeably, subroutines).
    # PHASE 1.a: Remove all the places where any function is imported.
    for use in walk(ast, Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(use, Name)).string
        mod_spec = (mod_name,)
        olist = ast_utils.atmost_one(ast_utils.children_of_type(use, 'Only_List'))
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
            continue
        uname = uident_map[fspec]
        ufspec = fspec[:-1] + (uname,)
        name.string = uname

        # Find the nearest execution and its correpsonding specification parts.
        execution_part = fref.parent
        while not isinstance(execution_part, Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, Specification_Part))

        # Find out the module name.
        cmod = fref.parent
        while cmod and not isinstance(cmod, (Module, Main_Program)):
            cmod = cmod.parent
        if cmod:
            stmt, _, _, _ = _get_module_or_program_parts(cmod)
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, Name)).string.lower()
        else:
            subp = list(ast_utils.children_of_type(ast, Subroutine_Subprogram))
            assert subp
            stmt = ast_utils.singular(ast_utils.children_of_type(subp[0], Subroutine_Stmt))
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, Name)).string.lower()

        # We are assumping that it's either a toplevel subprogram or a subprogram defined directly inside a module.
        assert 1 <= len(ufspec) <= 2
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
    # PHASE 1.c: Replace any access statments that made these functions public/private.
    for acc in walk(ast, Access_Stmt):
        # TODO: Add ref.
        kind, alist = acc.children
        if not alist:
            continue
        scope_spec = find_scope_spec(acc)
        for c in alist.children:
            assert isinstance(c, Name)
            c_spec = find_real_ident_spec(c.string, scope_spec, alias_map)
            if not isinstance(alias_map[c_spec], (Function_Stmt, Subroutine_Stmt)):
                continue
            if c_spec in uident_map:
                c.string = uident_map[c_spec]
    # PHASE 1.d: Replaces actual function names.
    for k, v in ident_map.items():
        if not isinstance(v, (Function_Stmt, Subroutine_Stmt)) or k not in uident_map:
            continue
        oname, uname = k[-1], uident_map[k]
        ast_utils.singular(ast_utils.children_of_type(v, Name)).string = uname
        # Fix the tail too.
        fdef = v.parent
        end_stmt = ast_utils.singular(ast_utils.children_of_type(fdef, (End_Function_Stmt, End_Subroutine_Stmt)))
        ast_utils.singular(ast_utils.children_of_type(end_stmt, Name)).string = uname
        # For functions, the function name is also available as a variable inside.
        if isinstance(v, Function_Stmt):
            vspec = ast_utils.atmost_one(ast_utils.children_of_type(fdef, Specification_Part))
            vexec = ast_utils.atmost_one(ast_utils.children_of_type(fdef, Execution_Part))
            for nm in walk([n for n in [vspec, vexec] if n], Name):
                if nm.string != oname:
                    continue
                local_spec = search_local_alias_spec(nm)
                # We need to do a bit of surgery, since we have the `oname` inide the scope ending with `uname`.
                local_spec = local_spec[:-2] + local_spec[-1:]
                assert local_spec in ident_map and ident_map[local_spec] == v
                nm.string = uname

    return ast


def _get_module_or_program_parts(mod: Union[Module, Main_Program]) \
        -> Tuple[
            Union[Module_Stmt, Program_Stmt],
            Optional[Specification_Part],
            Optional[Execution_Part],
            Optional[Module_Subprogram_Part],
        ]:
    # There must exist a module statment.
    stmt = ast_utils.singular(ast_utils.children_of_type(mod, Module_Stmt if isinstance(mod, Module) else Program_Stmt))
    # There may or may not exist a specification part.
    spec = list(ast_utils.children_of_type(mod, Specification_Part))
    assert len(spec) <= 1, f"A module/program cannot have more than one specification parts, found {spec} in {mod}"
    spec = spec[0] if spec else None
    # There may or may not exist an execution part.
    exec = list(ast_utils.children_of_type(mod, Execution_Part))
    assert len(exec) <= 1, f"A module/program cannot have more than one execution parts, found {spec} in {mod}"
    exec = exec[0] if exec else None
    # There may or may not exist a subprogram part.
    subp = list(ast_utils.children_of_type(mod, Module_Subprogram_Part))
    assert len(subp) <= 1, f"A module/program cannot have more than one subprogram parts, found {subp} in {mod}"
    subp = subp[0] if subp else None
    return stmt, spec, exec, subp
