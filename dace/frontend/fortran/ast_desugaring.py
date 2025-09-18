# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import sys
from typing import Union, Tuple, Dict, Optional, List, Set

from fparser.api import get_reader
from fparser.two.Fortran2003 import Module_Stmt, Function_Stmt, Subroutine_Stmt, Derived_Type_Stmt, \
    Component_Decl, Entity_Decl, Interface_Stmt, Main_Program, Subroutine_Subprogram, \
    Function_Subprogram, Name, Program, Use_Stmt, Rename, Part_Ref, Data_Ref, Initialization, \
    Intrinsic_Function_Reference, Derived_Type_Def, Module, Function_Reference, Structure_Constructor, Call_Stmt, \
    Intrinsic_Name, Access_Stmt, Enum_Def, Expr, Enumerator, Real_Literal_Constant, Actual_Arg_Spec, \
    Execution_Part, \
    Specification_Part, Interface_Block, Association, Procedure_Designator, Type_Bound_Procedure_Part, \
    Associate_Construct, Subscript_Triplet, End_Function_Stmt, End_Subroutine_Stmt, Module_Subprogram_Part, \
    Enumerator_List, Actual_Arg_Spec_List, Only_List, Dummy_Arg_List, Dummy_Arg_Name_List, Data_Stmt_Object_List, \
    Data_Stmt_Value_List, Section_Subscript_List, Component_Initialization, If_Then_Stmt, Else_If_Stmt, Else_Stmt, \
    If_Construct, \
    Assignment_Stmt, If_Stmt, End_Module_Stmt, \
    Data_Stmt, Data_Stmt_Set, Data_Stmt_Value, Goto_Stmt, Continue_Stmt, Format_Stmt, Stmt_Function_Stmt, \
    Internal_Subprogram_Part, \
    Private_Components_Stmt, Language_Binding_Spec, Type_Attr_Spec, Suffix, Case_Value_Range_List
from fparser.two.Fortran2008 import Type_Declaration_Stmt
from fparser.two.utils import Base, walk, NumberBase

from dace.frontend.fortran.ast_desugaring_v2.analysis import interface_specs, generic_specs, \
    _compute_argument_signature, _compute_candidate_argument_signature, procedure_specs, \
    find_type_dataref, find_type_of_entity, find_dataref_component_spec, _const_eval_basic_type, _dataref_root, \
    find_real_ident_spec, alias_specs, search_real_ident_spec, \
    identifier_specs, search_real_local_alias_spec, search_local_alias_spec, find_scope_spec, search_scope_spec, \
    ident_spec, _does_type_signature_match
from dace.frontend.fortran.ast_desugaring_v2.pruning import consolidate_uses
from dace.frontend.fortran.ast_desugaring_v2.types import TYPE_SPEC, SPEC, LITERAL_CLASSES
from dace.frontend.fortran.ast_desugaring_v2.utils import NAMED_STMTS_OF_INTEREST_CLASSES, SCOPE_OBJECT_TYPES, \
    find_name_of_stmt, find_scope_ancestor, find_name_of_node, append_children, remove_self, replace_node, set_children, \
    prepend_children, \
    remove_children, copy_fparser_node, _get_module_or_program_parts
from dace.frontend.fortran.ast_utils import singular, children_of_type, atmost_one

INTERFACE_NAMESPACE = '__interface__'


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


def deconstruct_interface_calls(ast: Program) -> Program:
    SUFFIX, COUNTER = 'deconiface', 0

    # We need to temporarily rename the interface imports to avoid shadowing the implementation.
    alias_map = alias_specs(ast)
    for olist in walk(ast, Only_List):
        use = olist.parent
        assert isinstance(use, Use_Stmt)
        scope_spec = find_scope_spec(use)
        mod = singular(children_of_type(use, Name))
        assert isinstance(mod, Name)
        for c in children_of_type(olist, Name):
            tgt_spec = find_real_ident_spec(c.string, scope_spec, alias_map)
            if len(tgt_spec) < 2 or tgt_spec[-2] != INTERFACE_NAMESPACE:
                continue
            replace_node(c, Rename(f"{c.string}_{SUFFIX}_tmp => {c.string}"))

            for nm in walk(use.parent.parent, Name):
                if nm.string != c.string or isinstance(nm.parent, (Only_List, Rename)):
                    continue
                local_spec = search_real_local_alias_spec(nm, alias_map)
                if not local_spec:
                    continue
                real_spec = ident_spec(alias_map[local_spec])
                if real_spec == tgt_spec:
                    replace_node(nm, Name(f"{c.string}_{SUFFIX}_tmp"))

    alias_map = alias_specs(ast)
    iface_map = interface_specs(ast, alias_map)
    unused_ifaces = set(iface_map.keys())
    for k, v in alias_map.items():
        if k == ident_spec(v):
            # The definition itself doesn't count as usage.
            continue
        if isinstance(v, Interface_Stmt) or isinstance(v.parent.parent, Interface_Block):
            unused_ifaces.difference_update({ident_spec(v)})

    for fref in walk(ast, (Function_Reference, Call_Stmt)):
        scope_spec = find_scope_spec(fref)
        name, args = fref.children
        if isinstance(name, Intrinsic_Name):
            continue
        fref_spec = search_real_ident_spec(name.string, scope_spec, alias_map)
        if not fref_spec:
            print(
                f"Could not resolve the function `{fref}` in scope `{scope_spec}`; "
                f"parts of AST is missing, but moving on",
                file=sys.stderr)
            continue
        assert fref_spec in alias_map, f"cannot find: {fref_spec}"
        if fref_spec not in iface_map:
            # We are only interested in calls to interfaces here.
            continue
        if not iface_map[fref_spec]:
            # We cannot resolve this one, because there is no candidate.
            print(f"{fref_spec} does not have any candidate to resolve to; moving on", file=sys.stderr)
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
            print(f"[in: {fref_spec}] {ifc_spec}/{conc_spec} not found; moving on", file=sys.stderr)
            for c in all_cand_sigs:
                print(f"...> {c}", file=sys.stderr)
            continue

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

    for ui in unused_ifaces:
        assert ui in alias_map
        assert isinstance(alias_map[ui], Interface_Stmt) or isinstance(alias_map[ui].parent.parent, Interface_Block)
        remove_self(alias_map[ui].parent)

    ast = consolidate_uses(ast)
    return ast


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

        bspec = dref_type.spec + (bname.string, )
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
        set_children(callsite, (Name(pname_alias), args))

    for tbp in walk(ast, Type_Bound_Procedure_Part):
        remove_self(tbp)
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
                    set_children(dr, (repl, *dr_rest))
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
                            set_children(access, [free_comps.get(i, c) for i, c in enumerate(access.children)])
                            # Now replace the entire `pr` with `repl`.
                            replace_node(pr, repl)
                            continue
                    # Otherwise, just replace normally.
                    set_children(pr, (repl, subsc))
            # Replace all the other names.
            for nm in walk(node, Name):
                # TODO: This is hacky and can backfire if `nm` is not a standalone identifier.
                par = nm.parent
                # Avoid data refs as we have just processed them.
                if isinstance(par, (Data_Ref, Part_Ref)):
                    continue
                if nm.string not in local_map:
                    continue
                replace_node(nm, copy_fparser_node(local_map[nm.string]))
        replace_node(assoc, rest)

    return ast


GLOBAL_DATA_OBJ_NAME = 'global_data'
GLOBAL_DATA_TYPE_NAME = 'global_data_type'


def consolidate_global_data_into_arg(ast: Program, always_add_global_data_arg: bool = False) -> Program:
    """
    Move all the global data into one structure and use it from there.
    TODO: We will have circular dependency if there are global objects of derived type. How to handle that?
    """
    alias_map = alias_specs(ast)
    GLOBAL_DATA_MOD_NAME = 'global_mod'
    if (GLOBAL_DATA_MOD_NAME, ) in alias_map:
        # We already have the global initialisers.
        return ast

    all_derived_types, all_global_vars = [], []
    # Collect all the derived types into a global module.
    for dt in walk(ast, Derived_Type_Def):
        dtspec = ident_spec(singular(children_of_type(dt, Derived_Type_Stmt)))
        assert len(dtspec) == 2
        mod, dtname = dtspec
        all_derived_types.append(f"use {mod}, only : {dtname}")
    # Collect all the global variables into a single global data structure.
    for m in walk(ast, Module):
        spart = atmost_one(children_of_type(m, Specification_Part))
        if not spart:
            continue
        for tdecl in children_of_type(spart, Type_Declaration_Stmt):
            typ, attr, _ = tdecl.children
            if 'PARAMETER' in f"{attr}":
                # This is a constant which should have been propagated away already.
                continue
            all_global_vars.append(tdecl.tofortran())
    all_derived_types = '\n'.join(all_derived_types)
    all_global_vars = '\n'.join(all_global_vars)

    # Then, replace all the instances of references to global variables with corresponding data-refs.
    for nm in walk(ast, Name):
        par = nm.parent
        if isinstance(par, (Entity_Decl, Use_Stmt, Rename, Only_List)):
            continue
        if isinstance(par, (Part_Ref, Data_Ref)):
            while par and isinstance(par.parent, (Part_Ref, Data_Ref)):
                par = par.parent
            scope_spec = search_scope_spec(par)
            root, _, _ = _dataref_root(par, scope_spec, alias_map)
            if root is not nm:
                continue
        local_spec = search_real_local_alias_spec(nm, alias_map)
        if not local_spec:
            continue
        assert local_spec in alias_map
        if not isinstance(alias_map[local_spec], Entity_Decl):
            continue
        edecl_spec = ident_spec(alias_map[local_spec])
        assert len(edecl_spec) >= 2, \
            f"Fortran cannot possibly have a top-level global variable, outside any module; got {edecl_spec}"
        if len(edecl_spec) != 2:
            # We cannot possibly have a module level variable declaration.
            continue
        mod, var = edecl_spec
        assert (mod, ) in alias_map
        if not isinstance(alias_map[(mod, )], Module_Stmt):
            continue
        if isinstance(nm.parent, Part_Ref):
            _, subsc = nm.parent.children
            replace_node(nm.parent, Data_Ref(f"{GLOBAL_DATA_OBJ_NAME} % {var}({subsc})"))
        else:
            replace_node(nm, Data_Ref(f"{GLOBAL_DATA_OBJ_NAME} % {var}"))

    if all_global_vars or always_add_global_data_arg:
        # Make `global_data` an argument to every defined function.
        for fn in walk(ast, (Function_Subprogram, Subroutine_Subprogram)):
            stmt = singular(children_of_type(fn, NAMED_STMTS_OF_INTEREST_CLASSES))
            assert isinstance(stmt, (Function_Stmt, Subroutine_Stmt))
            prefix, name, dummy_args, whatever = stmt.children
            if dummy_args:
                prepend_children(dummy_args, Name(GLOBAL_DATA_OBJ_NAME))
            else:
                set_children(stmt, (prefix, name, Dummy_Arg_Name_List(GLOBAL_DATA_OBJ_NAME), whatever))
            spart = atmost_one(children_of_type(fn, Specification_Part))
            use_stmt = f"use {GLOBAL_DATA_MOD_NAME}, only : {GLOBAL_DATA_TYPE_NAME}"
            if spart:
                prepend_children(spart, Use_Stmt(use_stmt))
            else:
                set_children(fn, fn.children[:1] + [Specification_Part(get_reader(use_stmt))] + fn.children[1:])
            spart = atmost_one(children_of_type(fn, Specification_Part))
            decl_idx = [idx for idx, v in enumerate(spart.children) if isinstance(v, Type_Declaration_Stmt)]
            decl_idx = decl_idx[0] if decl_idx else len(spart.children)
            set_children(
                spart, spart.children[:decl_idx] +
                [Type_Declaration_Stmt(f"type({GLOBAL_DATA_TYPE_NAME}) :: {GLOBAL_DATA_OBJ_NAME}")] +
                spart.children[decl_idx:])
        for fcall in walk(ast, (Function_Reference, Call_Stmt)):
            fn, args = fcall.children
            fnspec = search_real_local_alias_spec(fn, alias_map)
            if not fnspec:
                continue
            fnstmt = alias_map[fnspec]
            assert isinstance(fnstmt, (Function_Stmt, Subroutine_Stmt))
            if args:
                prepend_children(args, Name(GLOBAL_DATA_OBJ_NAME))
            else:
                set_children(fcall, (fn, Actual_Arg_Spec_List(GLOBAL_DATA_OBJ_NAME)))
        # NOTE: We do not remove the variables themselves, and let them be pruned later on.

    global_mod = Module(
        get_reader(f"""
module {GLOBAL_DATA_MOD_NAME}
  {all_derived_types}

  type {GLOBAL_DATA_TYPE_NAME}
    {all_global_vars}
  end type {GLOBAL_DATA_TYPE_NAME}
end module {GLOBAL_DATA_MOD_NAME}
"""))
    prepend_children(ast, global_mod)

    ast = consolidate_uses(ast)
    return ast


def create_global_initializers(ast: Program, entry_points: List[SPEC]) -> Program:
    # TODO: Ordering of the initializations may matter, but for that we need to find how Fortran's global initialization
    #  works and then reorder the initialization calls appropriately.

    ident_map = identifier_specs(ast)
    GLOBAL_INIT_FN_NAME = 'global_init_fn'
    if (GLOBAL_INIT_FN_NAME, ) in ident_map:
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
                # TODO: FParser bug; A simple `Module_Subprogram_Part('contains') should work, but doesn't;
                #  hence the surgery.
                sp_part = Module(get_reader('module m\ncontains\nend module m')).children[1]
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
    type_defs: Dict[SPEC, Tuple[str, List[SPEC]]] = \
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
        if len(t) != 2:
            # Not a type that's globally accessible anyway.
            continue
        mod, _ = t
        assert (mod, ) in alias_map
        if not isinstance(alias_map[(mod, )], Module_Stmt):
            # Not a type that's globally accessible anyway.
            continue
        init_fn_name, comps = v
        _make_init_fn(init_fn_name, comps, t)

    global_inited_vars: List[SPEC] = [
        k for k, v in ident_map.items()
        if isinstance(v, Entity_Decl) and not find_type_of_entity(v, alias_map).const and (
            find_type_of_entity(v, alias_map).spec in type_defs or atmost_one(children_of_type(v, Initialization)))
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
                assert isinstance(varz, Data_Stmt_Object_List)
                assert isinstance(valz, Data_Stmt_Value_List)
                if len(varz.children) != len(valz.children):
                    assert len(varz.children) == 1
                    singular_varz, = varz.children
                    new_varz = [f"{singular_varz}({i + 1})" for i in range(len(valz.children))]
                    replace_node(varz, Data_Stmt_Object_List(', '.join(new_varz)))
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
                    assert isinstance(elem, LITERAL_CLASSES), \
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


def deconstruct_statement_functions(ast: Program) -> Program:
    alias_map = alias_specs(ast)
    all_stmt_fns: Set[SPEC] = {find_scope_spec(sf) + (sf.children[0].string, ) for sf in walk(ast, Stmt_Function_Stmt)}

    for sf in walk(ast, Stmt_Function_Stmt):
        scope_spec = find_scope_spec(sf)
        fn, args, expr = sf.children
        if args:
            args = args.children

        def _get_typ(var: Name):
            _spec = scope_spec + (var.string, )
            _decl = alias_map[_spec]
            assert isinstance(_decl, Entity_Decl)
            _tdecl = _decl.parent.parent
            _typ, _, _ = _tdecl.children
            return _typ

        ret_typ = _get_typ(fn)
        arg_typs = tuple(_get_typ(a) for a in args)
        dummy_args = [a.string for a in args]

        arg_decls = [f"{t}, intent(in) :: {a}" for t, a in zip(arg_typs, args)]
        carryovers = []
        for nm in walk(expr, Name):
            if nm.string in dummy_args:
                continue
            spec = scope_spec + (nm.string, )
            if spec not in alias_map:
                continue
            decl = alias_map[spec]
            if not isinstance(decl, Entity_Decl):
                continue
            tdecl = decl.parent.parent
            typ, _, _ = tdecl.children
            shape = find_type_of_entity(decl, alias_map).shape
            shape = f"({','.join(shape)})" if shape else ''
            if spec not in all_stmt_fns and nm.string not in carryovers:
                carryovers.append(nm.string)
                arg_decls.append(f"{typ}, intent(in) :: {nm}{shape}")

        dummy_args = ','.join(dummy_args + carryovers)
        arg_decls = '\n'.join(arg_decls)
        nufn = f"""
{ret_typ} function {fn}({dummy_args})
  implicit none
  {arg_decls}
  {fn} = {expr}
end function {fn}
"""
        sp = sf.parent
        assert isinstance(sp, Specification_Part)
        box = sp.parent

        # Fix the arguments on the call-sites.
        for fcall in walk(box, (Call_Stmt, Function_Reference, Part_Ref, Structure_Constructor)):
            tfn, targs = fcall.children
            tfnloc = search_real_local_alias_spec(tfn, alias_map)
            if tfnloc != scope_spec + (fn.string, ):
                continue
            nufcall = Function_Reference(fcall.tofortran())
            tfn, targs = nufcall.children
            targs = targs.children if targs else []
            targs = [t.string for t in targs]
            targs.extend(carryovers)
            targs = ','.join(targs)
            nufcall = Function_Reference(f"{tfn}({targs})")
            replace_node(fcall, nufcall)

        intsp = atmost_one(children_of_type(box, (Internal_Subprogram_Part, Module_Subprogram_Part)))
        if intsp:
            append_children(intsp, Function_Subprogram(get_reader(nufn)))
        else:
            if isinstance(box, Module):
                intsp = Module_Subprogram_Part(get_reader(f"contains\n{nufn}".strip()))
            else:
                intsp = Internal_Subprogram_Part(get_reader(f"contains\n{nufn}".strip()))
            endbox = box.children[-1]
            replace_node(endbox, (intsp, endbox))

        ret_decl = alias_map[scope_spec + (fn.string, )]
        ret_declist = ret_decl.parent
        remove_children(ret_declist, ret_decl)
        if not ret_declist.children:
            remove_self(ret_declist.parent)
        remove_self(sf)

    return ast


def deconstuct_goto_statements(ast: Program) -> Program:
    # TODO: Support `Compound_Goto_Stmt`.
    for node in walk(ast, Base):
        # Move any label on a non-continue statement onto one (except for format statement which require one).
        if not isinstance(node, (Continue_Stmt, Format_Stmt)) and node.item and node.item.label is not None:
            cont = Continue_Stmt("CONTINUE")
            cont.item = node.item
            node.item = None
            replace_node(node, (cont, node))

    labels: Dict[str, Base] = {}
    for node in walk(ast, Base):
        if node.item and node.item.label is not None and isinstance(node, Continue_Stmt):
            labels[str(node.item.label)] = node

    # TODO: We have a very limited supported pattern of GOTO here, and possibly need to expand.
    # Assumptions: Each GOTO goes only forward. The target's parent is same as either the parent or the grandparent of
    # the GOTO. If the GOTO and its target have different parents, then the GOTO's parent is a if-construct.

    COUNTER = 0
    for goto in walk(ast, Goto_Stmt):
        target, = goto.children
        target = target.string
        target = labels[target]
        if goto.parent == target.parent:
            raise NotImplementedError

        ifc = goto.parent
        ifc_par = ifc.parent
        assert isinstance(ifc, (If_Stmt, If_Construct)), \
            f"Everything but conditionals are unsupported for goto's parent; got: {ifc}"
        assert ifc.parent is target.parent
        ifc_pos, target_pos = None, None
        for pos, c in enumerate(ifc.parent.children):
            if c is ifc:
                ifc_pos = pos
            elif c is target:
                target_pos = pos
        assert target_pos > ifc_pos, f"Only forward-facing GOTOs are supported"

        ex = goto.parent
        while not isinstance(ex, Execution_Part):
            ex = ex.parent
        spec = atmost_one(children_of_type(ex.parent, Specification_Part))
        assert spec

        if not isinstance(ifc, (If_Stmt, If_Construct)):
            raise NotImplementedError

        goto_var, COUNTER = f"goto_{COUNTER}", COUNTER + 1
        append_children(spec, Type_Declaration_Stmt(f"LOGICAL :: {goto_var}"))
        replace_node(goto, Assignment_Stmt(f"{goto_var} = .true."))

        for else_op in ifc_par.children[ifc_pos + 1:target_pos]:
            if isinstance(else_op, Continue_Stmt):
                # Continue statements are no-op, but they may have label attached, so we leave them be.
                continue
            if isinstance(else_op, If_Stmt):
                # We merge the condition with existing if.
                cond, op = else_op.children
                nu_cond = Expr(f".not.({goto_var}) .and. {cond}")
                replace_node(cond, nu_cond)
            elif isinstance(else_op, If_Construct):
                # We merge the condition with existing if.
                for c in else_op.children:
                    if isinstance(c, If_Then_Stmt):
                        cond, = c.children
                        nu_cond = Expr(f".not.({goto_var}) .and. {cond}")
                        replace_node(cond, nu_cond)
                    elif isinstance(c, Else_If_Stmt):
                        cond, _ = c.children
                        nu_cond = Expr(f".not.({goto_var}) .and. {cond}")
                        replace_node(cond, nu_cond)
                    elif isinstance(c, Else_Stmt):
                        nu_else = Else_If_Stmt(f"else if (.not.({goto_var})) then")
                        replace_node(c, nu_else)
                    else:
                        continue
            else:
                nu_if = If_Stmt(f"if (.not.({goto_var})) call x")
                replace_node(else_op, nu_if)
                replace_node(singular(nm for nm in walk(nu_if, Call_Stmt)), else_op)

        replace_node(ifc, [Assignment_Stmt(f"{goto_var} = .false."), ifc])

    return ast
