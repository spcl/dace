# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Optional, List, Iterable, Set, Tuple, Dict

import networkx as nx
import numpy as np
import fparser.two.Fortran2003 as f03
from fparser.two.utils import Base, walk

from . import utils
from . import types
from . import analysis
from .. import ast_utils


def consolidate_uses(ast: f03.Program, alias_map: Optional[types.SPEC_TABLE] = None) -> f03.Program:
    """
    Rewrites all `USE` statements in the program to be more explicit.
    It analyzes the whole scope of the `USE` statement and replaces it with a `USE ..., ONLY: ...`
    statement that lists only the symbols that are actually used in that scope.

    :param ast: The root of the fparser AST.
    :param alias_map: An optional pre-computed alias map.
    :return: The modified AST.
    """
    alias_map = alias_map or analysis.alias_specs(ast)
    for sp in reversed(walk(ast, f03.Specification_Part)):
        # First, we do a surgery to fix the kind parameter of the literals that refer to a variable, but FParser leaves
        # them as plain strings instead of making them `Name`s.
        for lit in walk(sp.parent,
                        (f03.Real_Literal_Constant, f03.Signed_Real_Literal_Constant, f03.Int_Literal_Constant,
                         f03.Signed_Int_Literal_Constant, f03.Logical_Literal_Constant)):
            val, kind = lit.children
            if not isinstance(kind, str):
                continue
            utils.set_children(lit, (val, f03.Name(kind)))

        use_map: Dict[str, Set[str]] = {}
        # Build the table to keep the use statements only if they are actually necessary.
        for nm in walk(sp.parent, f03.Name):
            if isinstance(nm.parent, (f03.Use_Stmt, f03.Only_List, f03.Rename)):
                # The identifiers in the use statements themselves are not of concern.
                continue
            # Where did we _really_ import `nm` from? Find the definition module.
            sc_spec = analysis.search_scope_spec(nm)
            if not sc_spec:
                continue
            box = alias_map[sc_spec].parent
            if box is not sp.parent and isinstance(
                    box, (f03.Function_Subprogram, f03.Subroutine_Subprogram, f03.Main_Program)):
                # If `nm` is imported, it should happen in a deeper subprogram.
                continue
            spec = analysis.search_real_ident_spec(nm.string, sc_spec, alias_map)
            if not spec or spec not in alias_map:
                continue
            if alias_map[spec].parent is sp.parent:
                # If `nm` is just referring to the subprogram that `sp` is a part of, then just leave it be.
                continue
            if len(spec) == 2:
                mod_spec = spec[:-1]
            elif len(spec) == 3 and spec[-2] == analysis.INTERFACE_NAMESPACE:
                mod_spec = spec[:-2]
            else:
                continue
            if not isinstance(alias_map[mod_spec], f03.Module_Stmt):
                # Objects defined inside a free function cannot be imported; so we must already be in that function.
                continue
            nm_mod = mod_spec[0]
            # And which module are we in right now?
            sp_mod = sp
            while sp_mod and not isinstance(sp_mod, (f03.Module, f03.Main_Program)):
                sp_mod = sp_mod.parent
            if sp_mod and nm_mod == utils.find_name_of_node(sp_mod):
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
        nuses: List[f03.Use_Stmt] = [
            f03.Use_Stmt(f"use {k}, only: {', '.join(sorted(use_map[k]))}") for k in use_map.keys()
        ]
        # Remove the old ones, and prepend the new ones.
        utils.set_children(sp, nuses + [c for c in sp.children if not isinstance(c, f03.Use_Stmt)])
    return ast


def keep_sorted_used_modules(ast: f03.Program, entry_points: Optional[Iterable[types.SPEC]] = None) -> f03.Program:
    """
    Prunes entire unused modules from the AST and sorts the remaining ones topologically.
    It builds a dependency graph between modules based on `USE` statements. Starting from
    the given entry points, it finds all transitively used modules, removes all others,
    and sorts the remaining modules so that a module is always defined before it is used.

    :param ast: The root of the fparser AST.
    :param entry_points: A set of specs for entry-point subroutines. All modules used by these
                         are preserved. If None, all modules are considered entry points.
    :return: The modified AST with unused modules removed and remaining ones sorted.
    """
    TOPLEVEL = '__toplevel__'

    def _get_module(n: Base) -> str:
        p = n
        while p and not isinstance(p, (f03.Module, f03.Main_Program)):
            p = p.parent
        if not p:
            return TOPLEVEL
        else:
            p_stmt = ast_utils.singular(ast_utils.children_of_type(p, (f03.Module_Stmt, f03.Program_Stmt)))
            return utils.find_name_of_stmt(p_stmt).lower()

    g = nx.DiGraph()  # An edge u->v means u should come before v, i.e., v depends on u.
    for c in ast.children:
        g.add_node(_get_module(c))
    g.add_node(TOPLEVEL)

    for u in walk(ast, f03.Use_Stmt):
        u_name = ast_utils.singular(ast_utils.children_of_type(u, f03.Name)).string.lower()
        v_name = _get_module(u)
        g.add_edge(u_name, v_name)

    entry_modules: Set[str]
    if entry_points is None:
        entry_modules = set(g.nodes) | {TOPLEVEL}
    else:
        entry_modules = {ep[0] for ep in entry_points if ep[0] in g.nodes} | {TOPLEVEL}

    assert all(g.has_node(em) for em in entry_modules)
    used_modules: Set[str] = {anc for em in entry_modules for anc in nx.ancestors(g, em)} | entry_modules
    h = g.subgraph(used_modules).to_directed()

    top_ord = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(h))}
    top_ord[TOPLEVEL] = g.number_of_nodes() + 1

    utils.set_children(ast, [n for n in ast.children if _get_module(n) in used_modules])
    assert all(_get_module(n) in top_ord for n in ast.children)
    utils.set_children(ast, sorted(ast.children, key=lambda x: top_ord[_get_module(x)]))

    return ast


def prune_coarsely(ast: f03.Program, keepers: Iterable[types.SPEC]) -> f03.Program:
    """
    Iteratively removes unused functions, types, interfaces, and variables from the AST.

    This function performs a coarse-grained dead code elimination. It works by first identifying
    all code objects that are directly or indirectly used by the specified `keepers`.
    It then removes anything not in this set of used objects. This process is repeated
    until no more code can be removed, ensuring that all dependencies are correctly handled.

    :param ast: The root of the fparser AST.
    :param keepers: An iterable of specs for objects to keep (e.g., entry-point subroutines).
    :return: The pruned AST.
    """
    removed_something = None
    while removed_something is None or removed_something:
        removed_something = False
        ast = consolidate_uses(ast)
        ast = keep_sorted_used_modules(ast, keepers)
        ident_map = analysis.identifier_specs(ast)
        alias_map = analysis.alias_specs(ast)
        iface_map = analysis.interface_specs(ast, alias_map)

        used_fns: Set[types.SPEC] = set(keepers)
        for k, v in ident_map.items():
            if len(k) < 2 or not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-2] if k[-2] == analysis.INTERFACE_NAMESPACE else k[:-1]].parent
            for nm in walk(box, f03.Name):
                if (nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)) or isinstance(
                        nm.parent,
                    (f03.Function_Stmt, f03.End_Function_Stmt, f03.Subroutine_Stmt, f03.End_Subroutine_Stmt))):
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k:
                    continue
                used_fns.add(k)
                break
        for k, v in alias_map.items():
            if not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                continue
            if k not in ident_map:
                used_fns.add(analysis.ident_spec(v))
        for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
            scope_spec = analysis.find_scope_spec(fref)
            name, _ = fref.children
            if isinstance(name, f03.Intrinsic_Name):
                continue
            fref_spec = analysis.search_real_ident_spec(name.string, scope_spec, alias_map)
            if fref_spec and len(fref_spec) == 1:
                used_fns.add(fref_spec)
        for k, vs in iface_map.items():
            for v in vs:
                used_fns.add(v)
        for k, v in ident_map.items():
            if not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                continue
            if k not in used_fns:
                utils.remove_self(v.parent)
                removed_something = True

        used_types: Set[types.SPEC] = set()
        for k, v in ident_map.items():
            if not isinstance(v, f03.Derived_Type_Stmt):
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-1]].parent
            for nm in walk(box, f03.Name):
                if nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)):
                    continue
                if isinstance(nm.parent, (f03.Derived_Type_Stmt, f03.End_Type_Stmt)) and nm.parent.parent is v.parent:
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k:
                    continue
                used_types.add(k)
                break
        for k, v in alias_map.items():
            if not isinstance(v, f03.Derived_Type_Stmt):
                continue
            if k not in ident_map:
                used_types.add(analysis.ident_spec(v))
        for k, v in ident_map.items():
            if not isinstance(v, f03.Derived_Type_Stmt):
                continue
            if k not in used_types:
                utils.remove_self(v.parent)
                removed_something = True

        used_ifaces: Set[types.SPEC] = set()
        for k, v in ident_map.items():
            if len(k) < 2 or k[-2] != analysis.INTERFACE_NAMESPACE:
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-2]].parent
            for nm in walk(box, f03.Name):
                if nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)):
                    continue
                if isinstance(nm.parent, (f03.Interface_Stmt, f03.End_Interface_Stmt)) and nm.parent.parent is v.parent:
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k or scope_spec == k[:-2] + k[-1:]:
                    continue
                used_ifaces.add(k)
                break
        for k, v in alias_map.items():
            vspec = analysis.ident_spec(v)
            if len(vspec) < 2 or vspec[-2] != analysis.INTERFACE_NAMESPACE:
                continue
            if k not in ident_map:
                used_ifaces.add(vspec)
        for k, v in ident_map.items():
            if len(k) < 2 or k[-2] != analysis.INTERFACE_NAMESPACE:
                continue
            if k not in used_ifaces:
                utils.remove_self(v.parent)
                removed_something = True

        used_vars: Set[types.SPEC] = set()
        for k, v in ident_map.items():
            if not isinstance(v, (f03.Entity_Decl, f03.Proc_Decl)):
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-1]].parent
            for nm in walk(box, f03.Name):
                if nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)) or nm.parent is v:
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k:
                    continue
                used_vars.add(k)
                break
        for k, v in alias_map.items():
            if not isinstance(v, (f03.Entity_Decl, f03.Proc_Decl)):
                continue
            if k not in ident_map:
                used_vars.add(analysis.ident_spec(v))
        for k, v in ident_map.items():
            if not isinstance(v, (f03.Entity_Decl, f03.Proc_Decl)):
                continue
            if k not in used_vars:
                elist = v.parent
                utils.remove_self(v)
                elist_tdecl = elist.parent
                assert isinstance(elist_tdecl, (f03.Type_Declaration_Stmt, f03.Procedure_Declaration_Stmt))
                if not elist.children:
                    utils.remove_self(elist_tdecl)
                removed_something = True

    for iface in walk(ast, f03.Interface_Stmt):
        name, = iface.children
        if name and name != 'ABSTRACT':
            continue
        idef = iface.parent
        if not idef.children[1:-1]:
            utils.remove_self(idef)

    ast = keep_sorted_used_modules(ast, keepers)
    return ast


def prune_unused_objects(ast: f03.Program, keepers: List[types.SPEC]) -> f03.Program:
    """
    Performs a fine-grained pruning of the AST by tracing all usages from a set of
    "keeper" objects. Any object (subroutine, function, type, variable, etc.) that is
    not reachable from the keepers is removed from the AST.

    Precondition: All indirections (such as interface calls) should be resolved before calling this.

    :param ast: The root of the fparser AST.
    :param keepers: A list of specs for objects that must be kept.
    :return: The pruned AST.
    """
    PRUNABLE_OBJECT_CLASSES = (f03.Program_Stmt, f03.Subroutine_Stmt, f03.Function_Stmt, f03.Derived_Type_Stmt,
                               f03.Entity_Decl, f03.Component_Decl)

    ident_map = analysis.identifier_specs(ast)
    alias_map = analysis.alias_specs(ast)
    survivors: Set[types.SPEC] = set(keepers)
    keeper_nodes = [alias_map[k] for k in keepers]
    assert all(isinstance(k, PRUNABLE_OBJECT_CLASSES) for k in keeper_nodes)

    def _keep_from(node: Base):
        for nm in walk(node, f03.Name):
            loc = analysis.search_real_local_alias_spec(nm, alias_map)
            scope_spec = analysis.search_scope_spec(nm.parent)
            if not loc or not scope_spec: continue
            nm_spec = analysis.ident_spec(alias_map[loc])
            if isinstance(nm.parent, f03.Entity_Decl) and nm is nm.parent.children[0]:
                fnargs = ast_utils.atmost_one(ast_utils.children_of_type(alias_map[scope_spec], f03.Dummy_Arg_List))
                fnargs = fnargs.children if fnargs else tuple()
                if any(a.string == nm.string for a in fnargs):
                    survivors.add(nm_spec)
                continue
            if isinstance(nm.parent, f03.Component_Decl) and nm is nm.parent.children[0]: continue
            if isinstance(nm.parent, f03.Pointer_Assignment_Stmt) and nm is nm.parent.children[0]: continue

            for j in reversed(range(len(scope_spec))):
                anc_spec = scope_spec[:j + 1]
                if anc_spec in survivors: continue
                survivors.add(anc_spec)
                anc_node = alias_map[anc_spec]
                if isinstance(anc_node, PRUNABLE_OBJECT_CLASSES):
                    _keep_from(anc_node.parent)

            if not nm_spec or nm_spec not in alias_map or nm_spec in survivors: continue
            survivors.add(nm_spec)
            keep_node = alias_map[nm_spec]
            if isinstance(keep_node, PRUNABLE_OBJECT_CLASSES):
                _keep_from(keep_node.parent)
        for dr in walk(node, f03.Data_Ref):
            root, rest = analysis._lookup_dataref(dr, alias_map)
            if rest and isinstance(rest[0], f03.Section_Subscript_List):
                root, rest = f03.Part_Ref(f"{root.tofortran()}({rest[0].tofortran()})"), rest[1:]
            scope_spec = analysis.find_scope_spec(dr)
            for upto in range(1, len(rest) + 1):
                anc_nodes: Tuple[f03.Name, ...] = (root, ) + rest[:upto]
                ancref = f03.Data_Ref('%'.join([c.tofortran() for c in anc_nodes]))
                ancspec = analysis.find_dataref_component_spec(ancref, scope_spec, alias_map)
                survivors.add(ancspec)

    for k in keeper_nodes:
        _keep_from(k.parent)

    killed: Set[types.SPEC] = set()
    for ns in sorted(set(ident_map.keys()) - survivors):
        ns_node = ident_map[ns]
        if not isinstance(ns_node, PRUNABLE_OBJECT_CLASSES): continue
        is_killed = False
        for i in range(len(ns) - 1):
            anc_spec = ns[:i + 1]
            if anc_spec in killed:
                killed.add(ns)
                is_killed = True
                break
        if is_killed: continue
        ns_typ = analysis.find_type_of_entity(ns_node, alias_map)
        if isinstance(ns_node, f03.Entity_Decl) and ns_typ.pointer:
            for pa in walk(ast, f03.Pointer_Assignment_Stmt):
                dst = pa.children[0]
                if not isinstance(dst, f03.Name): continue
                dst_spec = analysis.search_real_local_alias_spec(dst, alias_map)
                if dst_spec and alias_map[dst_spec] is ns_node:
                    utils.remove_self(pa)
        if isinstance(ns_node, f03.Entity_Decl):
            elist = ns_node.parent
            utils.remove_self(ns_node)
            elist_tdecl = elist.parent
            assert isinstance(elist_tdecl, f03.Type_Declaration_Stmt)
            if not elist.children:
                utils.remove_self(elist_tdecl)
            elist_spart = elist_tdecl.parent
            assert isinstance(elist_spart, f03.Specification_Part)
            for c in elist_spart.children:
                if not isinstance(c, f03.Equivalence_Stmt): continue
                _, eqvs = c.children
                eqvs = eqvs.children if eqvs else tuple()
                for eqv in eqvs:
                    eqa, eqbs = eqv.children
                    eqbs = eqbs.children if eqbs else tuple()
                    eqz = (eqa, ) + eqbs
                    assert all(isinstance(z, f03.Part_Ref) for z in eqz) and len(eqz) == 2
                    eqz = tuple(z for z in eqz if analysis.search_real_local_alias_spec(z.children[0], alias_map) != ns)
                    if len(eqz) < 2:
                        utils.remove_self(eqv)
                _, eqvs = c.children
                if not (eqvs.children if eqvs else tuple()):
                    utils.remove_self(c)
            if not elist_spart.children:
                utils.remove_self(elist_spart)
        elif isinstance(ns_node, f03.Component_Decl):
            clist = ns_node.parent
            utils.remove_self(ns_node)
            tdef = clist.parent
            assert isinstance(tdef, f03.Data_Component_Def_Stmt)
            if not clist.children:
                utils.remove_self(tdef)
        else:
            utils.remove_self(ns_node.parent)
        killed.add(ns)

    for m in walk(ast, f03.Module):
        _, sp, ex, subp = utils._get_module_or_program_parts(m)
        empty_spec = not sp or all(isinstance(c, (f03.Save_Stmt, f03.Implicit_Part)) for c in sp.children)
        empty_exec = not ex or not ex.children
        empty_subp = not subp or all(isinstance(c, f03.Contains_Stmt) for c in subp.children)
        if empty_spec and empty_exec and empty_subp:
            utils.remove_self(m)

    consolidate_uses(ast, alias_map)
    return ast


def prune_branches(ast: f03.Program) -> f03.Program:
    """
    Prunes dead branches from `If_Construct` and `If_Stmt` nodes by evaluating
    their conditional expressions at compile time.

    :param ast: The root of the fparser AST.
    :return: The modified AST with dead branches removed.
    """
    alias_map = analysis.alias_specs(ast)
    for ib in walk(ast, f03.If_Construct):
        _prune_branches_in_ifblock(ib, alias_map)
    for ib in walk(ast, f03.If_Stmt):
        _prune_branches_in_ifstmt(ib, alias_map)
    return ast


def _prune_branches_in_ifblock(ib: f03.If_Construct, alias_map: types.SPEC_TABLE):
    """Helper to prune an `If_Construct` (a multi-line IF block)."""
    ifthen = ib.children[0]
    assert isinstance(ifthen, f03.If_Then_Stmt)
    cond, = ifthen.children
    cval = analysis._const_eval_basic_type(cond, alias_map)
    if cval is None:
        return
    assert isinstance(cval, np.bool_)

    elifat = [idx for idx, c in enumerate(ib.children) if isinstance(c, (f03.Else_If_Stmt, f03.Else_Stmt))]
    if cval:
        cut = elifat[0] if elifat else -1
        actions = ib.children[1:cut]
        utils.replace_node(ib, actions)
        return
    elif not elifat:
        utils.remove_self(ib)
        return

    cut = elifat[0]
    cut_cond_node = ib.children[cut]
    if isinstance(cut_cond_node, f03.Else_Stmt):
        actions = ib.children[cut + 1:-1]
        utils.replace_node(ib, actions)
        return

    assert isinstance(cut_cond_node, f03.Else_If_Stmt)
    cut_cond, _ = cut_cond_node.children
    utils.remove_children(ib, ib.children[1:(cut + 1)])
    utils.set_children(ifthen, (cut_cond, ))
    _prune_branches_in_ifblock(ib, alias_map)


def _prune_branches_in_ifstmt(ib: f03.If_Stmt, alias_map: types.SPEC_TABLE):
    """Helper to prune an `If_Stmt` (a single-line IF statement)."""
    cond, actions = ib.children
    cval = analysis._const_eval_basic_type(cond, alias_map)
    if cval is None:
        return
    assert isinstance(cval, np.bool_)
    if cval:
        utils.replace_node(ib, actions)
    else:
        utils.remove_self(ib)
    expart = ib.parent
    if isinstance(expart, f03.Execution_Part) and not expart.children:
        utils.remove_self(expart)
