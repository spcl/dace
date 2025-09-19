# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module implements the "desugaring" passes for the Fortran AST. Desugaring is
the process of translating higher-level, more complex, or "syntactic sugar"
constructs into simpler, more fundamental language constructs.

The passes in this module handle various Fortran features, such as:
- `ENUM`: Converted to constant integer variables.
- `INTERFACE`: Generic function calls are resolved to their specific implementations.
- Type-bound procedures: Calls are deconstructed into regular subroutine/function calls.
- `ASSOCIATE`: Replaced with direct variable references.
- `DATA` statements: Converted into standard assignment statements.
- Statement functions: Transformed into full internal function definitions.
- `GOTO`: Converted into structured control flow where possible.

The goal is to produce a simplified AST that is easier to analyze and from which
DaCe SDFGs can be generated.
"""
import sys
from typing import Tuple, Dict, List, Set

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.utils import Base, walk

from . import analysis, pruning, types, utils
from .. import ast_utils

INTERFACE_NAMESPACE = '__interface__'


def deconstruct_enums(ast: f03.Program) -> f03.Program:
    """
    Replaces `ENUM` definitions with `INTEGER, PARAMETER` declarations.

    This pass finds all `Enum_Def` nodes in the AST. For each enum, it iterates
    through its enumerators, assigning them consecutive integer values (starting
    from 0, unless explicitly specified). It then replaces the `Enum_Def` node
    with a list of `Type_Declaration_Stmt` nodes that declare each enumerator
    as a named constant (`INTEGER, PARAMETER`).

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    for en in walk(ast, f03.Enum_Def):
        en_dict: Dict[str, f03.Expr] = {}
        # We need to for automatic counting.
        next_val = '0'
        next_offset = 0
        for el in walk(en, f03.Enumerator_List):
            for c in el.children:
                if isinstance(c, f03.Name):
                    c_name = c.string
                elif isinstance(c, f03.Enumerator):
                    # TODO: Add ref.
                    name, _, val = c.children
                    c_name = name.string
                    next_val = val.string
                    next_offset = 0
                en_dict[c_name] = f03.Expr(f"{next_val} + {next_offset}")
                next_offset = next_offset + 1
        type_decls = [f03.Type_Declaration_Stmt(f"integer, parameter :: {k} = {v}") for k, v in en_dict.items()]
        utils.replace_node(en, type_decls)
    return ast


def deconstruct_interface_calls(ast: f03.Program) -> f03.Program:
    """
    Resolves calls to generic interfaces into direct calls to concrete subprograms.

    Fortran interfaces allow defining a generic procedure name that can map to
    different specific procedures based on the arguments provided. This pass
    replaces these generic calls with direct calls to the appropriate concrete
    subprogram.

    The process is as follows:
    1.  It identifies all call sites (`Function_Reference`, `Call_Stmt`) that
        refer to an interface.
    2.  For each call, it computes the type signature of the arguments.
    3.  It compares this signature against the signatures of all concrete
        subprograms associated with the interface.
    4.  Once a match is found, the generic procedure name at the call site is
        replaced with the name of the matched concrete subprogram.
    5.  `USE` statements are added as needed to import the concrete subprogram
        into the calling scope.
    6.  Unused interfaces are pruned from the AST.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    SUFFIX, COUNTER = 'deconiface', 0

    # We need to temporarily rename the interface imports to avoid shadowing the implementation.
    alias_map = analysis.alias_specs(ast)
    for olist in walk(ast, f03.Only_List):
        use = olist.parent
        assert isinstance(use, f03.Use_Stmt)
        scope_spec = analysis.find_scope_spec(use)
        mod = ast_utils.singular(ast_utils.children_of_type(use, f03.Name))
        assert isinstance(mod, f03.Name)
        for c in ast_utils.children_of_type(olist, f03.Name):
            tgt_spec = analysis.find_real_ident_spec(c.string, scope_spec, alias_map)
            if len(tgt_spec) < 2 or tgt_spec[-2] != INTERFACE_NAMESPACE:
                continue
            utils.replace_node(c, f03.Rename(f"{c.string}_{SUFFIX}_tmp => {c.string}"))

            for nm in walk(use.parent.parent, f03.Name):
                if nm.string != c.string or isinstance(nm.parent, (f03.Only_List, f03.Rename)):
                    continue
                local_spec = analysis.search_real_local_alias_spec(nm, alias_map)
                if not local_spec:
                    continue
                real_spec = analysis.ident_spec(alias_map[local_spec])
                if real_spec == tgt_spec:
                    utils.replace_node(nm, f03.Name(f"{c.string}_{SUFFIX}_tmp"))

    alias_map = analysis.alias_specs(ast)
    iface_map = analysis.interface_specs(ast, alias_map)
    unused_ifaces = set(iface_map.keys())
    for k, v in alias_map.items():
        if k == analysis.ident_spec(v):
            # The definition itself doesn't count as usage.
            continue
        if isinstance(v, f03.Interface_Stmt) or isinstance(v.parent.parent, f03.Interface_Block):
            unused_ifaces.difference_update({analysis.ident_spec(v)})

    for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        scope_spec = analysis.find_scope_spec(fref)
        name, args = fref.children
        if isinstance(name, f03.Intrinsic_Name):
            continue
        fref_spec = analysis.search_real_ident_spec(name.string, scope_spec, alias_map)
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
        while not isinstance(execution_part, f03.Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, f03.Specification_Part))

        ifc_spec = analysis.ident_spec(alias_map[fref_spec])
        args_sig: Tuple[types.TYPE_SPEC, ...] = analysis._compute_argument_signature(args, scope_spec, alias_map)
        all_cand_sigs: List[Tuple[types.SPEC, Tuple[types.TYPE_SPEC, ...]]] = []

        conc_spec = None
        for cand in iface_map[ifc_spec]:
            assert cand in alias_map
            cand_stmt = alias_map[cand]
            assert isinstance(cand_stmt, (f03.Function_Stmt, f03.Subroutine_Stmt))

            # However, this candidate could be inside an interface block, and this be just another level of indirection.
            cand_spec = cand
            if isinstance(cand_stmt.parent.parent, f03.Interface_Block):
                cand_spec = analysis.find_real_ident_spec(cand_spec[-1], cand_spec[:-2], alias_map)
                assert cand_spec in alias_map
                cand_stmt = alias_map[cand_spec]
                assert isinstance(cand_stmt, (f03.Function_Stmt, f03.Subroutine_Stmt))

            # TODO: Add ref.
            _, _, cand_args, _ = cand_stmt.children
            if cand_args:
                cand_args_sig = analysis._compute_candidate_argument_signature(cand_args.children, cand_spec, alias_map)
            else:
                cand_args_sig = tuple()
            all_cand_sigs.append((cand_spec, cand_args_sig))

            if analysis._does_type_signature_match(args_sig, cand_args_sig):
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
                utils.append_children(subprog,
                                      f03.Specification_Part(get_reader(f"use {mod}, only: {pname_alias} => {pname}")))
            else:
                utils.prepend_children(specification_part, f03.Use_Stmt(f"use {mod}, only: {pname_alias} => {pname}"))

        # For both function and subroutine calls, replace `bname` with `pname_alias`, and add `dref` as the first arg.
        utils.replace_node(name, f03.Name(pname_alias))

    for ui in unused_ifaces:
        assert ui in alias_map
        assert isinstance(alias_map[ui], f03.Interface_Stmt) or isinstance(alias_map[ui].parent.parent,
                                                                           f03.Interface_Block)
        utils.remove_self(alias_map[ui].parent)

    ast = pruning.consolidate_uses(ast)
    return ast


def deconstruct_procedure_calls(ast: f03.Program) -> f03.Program:
    """
    Resolves calls to type-bound procedures into standard subprogram calls.

    This pass handles calls to procedures that are bound to a derived type, similar
    to method calls in object-oriented programming (e.g., `my_obj%method(args)`).
    It transforms these into regular, direct function or subroutine calls.

    The transformation involves:
    1.  Identifying all `Procedure_Designator` nodes, which represent type-bound
        procedure calls.
    2.  Resolving the procedure name based on the type of the object it's called on.
        This may involve matching against generic procedure bindings.
    3.  Replacing the `Procedure_Designator` call with a standard `Function_Reference`
        or `Call_Stmt`.
    4.  The object itself (e.g., `my_obj`) is passed as the new first argument to the
        concrete subprogram.
    5.  `USE` statements are added as needed to import the concrete subprogram.
    6.  The now-unused `Type_Bound_Procedure_Part` definitions are removed from the AST.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    SUFFIX, COUNTER = 'deconproc', 0

    alias_map = analysis.alias_specs(ast)
    proc_map = analysis.procedure_specs(ast)
    genc_map = analysis.generic_specs(ast)
    # We should have removed all the `association`s by now.
    assert not walk(ast, f03.Association), f"{walk(ast, f03.Association)}"

    for pd in walk(ast, f03.Procedure_Designator):
        # Ref: https://github.com/stfc/fparser/blob/master/src/fparser/two/Fortran2003.py#L12530
        dref, op, bname = pd.children

        callsite = pd.parent
        assert isinstance(callsite, (f03.Function_Reference, f03.Call_Stmt))

        # Find out the module name.
        cmod = callsite.parent
        while cmod and not isinstance(cmod, (f03.Module, f03.Main_Program)):
            cmod = cmod.parent
        if cmod:
            stmt, _, _, _ = utils._get_module_or_program_parts(cmod)
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string.lower()
        else:
            subp = list(ast_utils.children_of_type(ast, f03.Subroutine_Subprogram))
            assert subp
            stmt = ast_utils.singular(ast_utils.children_of_type(subp[0], f03.Subroutine_Stmt))
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string.lower()

        # Find the nearest execution and its correpsonding specification parts.
        execution_part = callsite.parent
        while not isinstance(execution_part, f03.Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, f03.Specification_Part))

        scope_spec = analysis.find_scope_spec(callsite)
        dref_type = analysis.find_type_dataref(dref, scope_spec, alias_map)
        fnref = pd.parent
        assert isinstance(fnref, (f03.Function_Reference, f03.Call_Stmt))
        _, args = fnref.children
        args_sig: Tuple[types.TYPE_SPEC, ...] = analysis._compute_argument_signature(args, scope_spec, alias_map)
        all_cand_sigs: List[Tuple[types.SPEC, Tuple[types.TYPE_SPEC, ...]]] = []

        bspec = dref_type.spec + (bname.string, )
        if bspec in genc_map and genc_map[bspec]:
            for cand in genc_map[bspec]:
                cand_stmt = alias_map[proc_map[cand]]
                cand_spec = analysis.ident_spec(cand_stmt)
                # TODO: Add ref.
                _, _, cand_args, _ = cand_stmt.children
                if cand_args:
                    cand_args_sig = analysis._compute_candidate_argument_signature(cand_args.children[1:], cand_spec,
                                                                                   alias_map)
                else:
                    cand_args_sig = tuple()
                all_cand_sigs.append((cand_spec, cand_args_sig))

                if analysis._does_type_signature_match(args_sig, cand_args_sig):
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
                utils.append_children(subprog,
                                      f03.Specification_Part(get_reader(f"use {mod}, only: {pname_alias} => {pname}")))
            else:
                utils.prepend_children(specification_part, f03.Use_Stmt(f"use {mod}, only: {pname_alias} => {pname}"))

        # For both function and subroutine calls, replace `bname` with `pname_alias`, and add `dref` as the first arg.
        _, args = callsite.children
        if args is None:
            args = f03.Actual_Arg_Spec_List(f"{dref}")
        else:
            args = f03.Actual_Arg_Spec_List(f"{dref}, {args}")
        utils.set_children(callsite, (f03.Name(pname_alias), args))

    for tbp in walk(ast, f03.Type_Bound_Procedure_Part):
        utils.remove_self(tbp)
    return ast


def deconstruct_associations(ast: f03.Program) -> f03.Program:
    """
    Eliminates `ASSOCIATE` constructs by substituting the associated names.

    The `ASSOCIATE` construct in Fortran provides a way to create a short-hand
    alias for a complex variable expression (e.g., `ASSOCIATE (x => a%b(i)%c)`).
    This pass removes these constructs by directly substituting the original
    complex expression wherever the alias is used within the associate block.

    For each `Associate_Construct`, it traverses the code inside the block and
    replaces all occurrences of the associated name (e.g., `x`) with a copy of
    the full target expression (e.g., `a%b(i)%c`). The `Associate_Construct`
    node is then replaced by the modified block of code.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    for assoc in walk(ast, f03.Associate_Construct):
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
            for dr in walk(node, f03.Data_Ref):
                # TODO: Add ref.
                root, dr_rest = dr.children[0], dr.children[1:]
                if root.string in local_map:
                    repl = local_map[root.string]
                    repl = type(repl)(repl.tofortran())
                    utils.set_children(dr, (repl, *dr_rest))
            # # Replace the part-ref roots as appropriate.
            for pr in walk(node, f03.Part_Ref):
                if isinstance(pr.parent, (f03.Data_Ref, f03.Part_Ref)):
                    continue
                # TODO: Add ref.
                root, subsc = pr.children
                if root.string in local_map:
                    repl = local_map[root.string]
                    repl = type(repl)(repl.tofortran())
                    if isinstance(subsc, f03.Section_Subscript_List) and isinstance(repl, (f03.Data_Ref, f03.Part_Ref)):
                        access = repl
                        while isinstance(access, (f03.Data_Ref, f03.Part_Ref)):
                            access = access.children[-1]
                        if isinstance(access, f03.Section_Subscript_List):
                            # We cannot just chain accesses, so we need to combine them to produce a single access.
                            # TODO: Maybe `isinstance(c, Subscript_Triplet)` + offset manipulation?
                            free_comps = [(i, c) for i, c in enumerate(access.children)
                                          if c == f03.Subscript_Triplet(':')]
                            assert len(free_comps) >= len(subsc.children), \
                                f"Free rank cannot increase, got {root}/{access} => {subsc}"
                            for i, c in enumerate(subsc.children):
                                idx, _ = free_comps[i]
                                free_comps[i] = (idx, c)
                            free_comps = {i: c for i, c in free_comps}
                            utils.set_children(access, [free_comps.get(i, c) for i, c in enumerate(access.children)])
                            # Now replace the entire `pr` with `repl`.
                            utils.replace_node(pr, repl)
                            continue
                    # Otherwise, just replace normally.
                    utils.set_children(pr, (repl, subsc))
            # Replace all the other names.
            for nm in walk(node, f03.Name):
                # TODO: This is hacky and can backfire if `nm` is not a standalone identifier.
                par = nm.parent
                # Avoid data refs as we have just processed them.
                if isinstance(par, (f03.Data_Ref, f03.Part_Ref)):
                    continue
                if nm.string not in local_map:
                    continue
                utils.replace_node(nm, utils.copy_fparser_node(local_map[nm.string]))
        utils.replace_node(assoc, rest)

    return ast


def convert_data_statements_into_assignments(ast: f03.Program) -> f03.Program:
    """
    Converts `DATA` statements into standard `Assignment_Stmt` nodes.

    The `DATA` statement in Fortran is a way to initialize variables at compile
    time. This pass transforms these static initializations into explicit
    assignment statements that are executed at the beginning of the subprogram.

    For each `Data_Stmt`, it iterates through the variable-value pairs. It then
    creates a corresponding `Assignment_Stmt` for each pair (e.g., `DATA x / 1 /`
    becomes `x = 1`) and prepends these new assignment statements to the
    execution part of the enclosing scope. The original `Data_Stmt` is then removed.

    This pass helps to make all initializations explicit and executable.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    # TODO: Data statements have unusual syntax even within Fortran and not everything is covered here yet.
    alias_map = analysis.alias_specs(ast)

    for spart in walk(ast, f03.Specification_Part):
        box = spart.parent
        xpart = ast_utils.atmost_one(ast_utils.children_of_type(box, f03.Execution_Part))
        for dst in reversed(walk(spart, f03.Data_Stmt)):
            repls: List[f03.Assignment_Stmt] = []
            for ds in dst.children:
                assert isinstance(ds, f03.Data_Stmt_Set)
                varz, valz = ds.children
                assert isinstance(varz, f03.Data_Stmt_Object_List)
                assert isinstance(valz, f03.Data_Stmt_Value_List)
                if len(varz.children) != len(valz.children):
                    assert len(varz.children) == 1
                    singular_varz, = varz.children
                    new_varz = [f"{singular_varz}({i + 1})" for i in range(len(valz.children))]
                    utils.replace_node(varz, f03.Data_Stmt_Object_List(', '.join(new_varz)))
                varz, valz = ds.children
                varz, valz = varz.children, valz.children
                assert len(varz) == len(valz)
                for k, v in zip(varz, valz):
                    scope_spec = analysis.find_scope_spec(k)
                    kroot, ktyp, rest = analysis._dataref_root(k, scope_spec, alias_map)
                    if isinstance(v, f03.Data_Stmt_Value):
                        repeat, elem = v.children
                        repeat = 1 if not repeat else int(analysis._const_eval_basic_type(repeat, alias_map))
                        assert repeat
                    else:
                        elem = v
                    # TODO: Support other types of data expressions.
                    assert isinstance(elem, types.LITERAL_CLASSES), \
                        f"only supports literal values in data data statements: {elem}"
                    if ktyp.shape:
                        if rest:
                            assert len(rest) == 1 and isinstance(rest[0], f03.Section_Subscript_List)
                            subsc = rest[0].tofortran()
                        else:
                            subsc = ','.join([':' for _ in ktyp.shape])
                        repls.append(f03.Assignment_Stmt(f"{kroot.string}({subsc}) = {elem.tofortran()}"))
                    else:
                        assert isinstance(k, f03.Name)
                        repls.append(f03.Assignment_Stmt(f"{k.string} = {elem.tofortran()}"))
            utils.remove_self(dst)
            if not xpart:
                # NOTE: Since the function does nothing at all (hence, no execution part), don't bother with the inits.
                continue
            utils.prepend_children(xpart, repls)

    return ast


def deconstruct_statement_functions(ast: f03.Program) -> f03.Program:
    """
    Transforms single-line statement functions into full internal subprograms.

    Statement functions are single-line functions defined within the
specification
    part of a subprogram (e.g., f(x) = x * x). This pass converts them into
    regular, multi-line INTERNAL or MODULE functions.

    The process involves:
     1. Identifying all Stmt_Function_Stmt nodes.
     2. Creating a new Function_Subprogram node for each one.
     3. The body of the new function becomes an assignment to the function's name
        (e.g., f = x * x).
     4. Arguments are explicitly declared. Variables from the outer scope that are
        used in the statement function are "carried over" and passed as additional
        arguments to the new function and at all its call sites.
     5. The new function is placed in the Internal_Subprogram_Part or
        Module_Subprogram_Part of the enclosing scope.
     6. The original Stmt_Function_Stmt is removed.

    This regularizes the AST by making all functions conform to the same
structure.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    alias_map = analysis.alias_specs(ast)
    all_stmt_fns: Set[types.SPEC] = {
        analysis.find_scope_spec(sf) + (sf.children[0].string, )
        for sf in walk(ast, f03.Stmt_Function_Stmt)
    }

    for sf in walk(ast, f03.Stmt_Function_Stmt):
        scope_spec = analysis.find_scope_spec(sf)
        fn, args, expr = sf.children
        if args:
            args = args.children

        def _get_typ(var: f03.Name):
            _spec = scope_spec + (var.string, )
            _decl = alias_map[_spec]
            assert isinstance(_decl, f03.Entity_Decl)
            _tdecl = _decl.parent.parent
            _typ, _, _ = _tdecl.children
            return _typ

        ret_typ = _get_typ(fn)
        arg_typs = tuple(_get_typ(a) for a in args)
        dummy_args = [a.string for a in args]

        arg_decls = [f"{t}, intent(in) :: {a}" for t, a in zip(arg_typs, args)]
        carryovers = []
        for nm in walk(expr, f03.Name):
            if nm.string in dummy_args:
                continue
            spec = scope_spec + (nm.string, )
            if spec not in alias_map:
                continue
            decl = alias_map[spec]
            if not isinstance(decl, f03.Entity_Decl):
                continue
            tdecl = decl.parent.parent
            typ, _, _ = tdecl.children
            shape = analysis.find_type_of_entity(decl, alias_map).shape
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
        assert isinstance(sp, f03.Specification_Part)
        box = sp.parent

        # Fix the arguments on the call-sites.
        for fcall in walk(box, (f03.Call_Stmt, f03.Function_Reference, f03.Part_Ref, f03.Structure_Constructor)):
            tfn, targs = fcall.children
            tfnloc = analysis.search_real_local_alias_spec(tfn, alias_map)
            if tfnloc != scope_spec + (fn.string, ):
                continue
            nufcall = f03.Function_Reference(fcall.tofortran())
            tfn, targs = nufcall.children
            targs = targs.children if targs else []
            targs = [t.string for t in targs]
            targs.extend(carryovers)
            targs = ','.join(targs)
            nufcall = f03.Function_Reference(f"{tfn}({targs})")
            utils.replace_node(fcall, nufcall)

        intsp = ast_utils.atmost_one(
            ast_utils.children_of_type(box, (f03.Internal_Subprogram_Part, f03.Module_Subprogram_Part)))
        if intsp:
            utils.append_children(intsp, f03.Function_Subprogram(get_reader(nufn)))
        else:
            if isinstance(box, f03.Module):
                intsp = f03.Module_Subprogram_Part(get_reader(f"contains\n{nufn}".strip()))
            else:
                intsp = f03.Internal_Subprogram_Part(get_reader(f"contains\n{nufn}".strip()))
            endbox = box.children[-1]
            utils.replace_node(endbox, (intsp, endbox))

        ret_decl = alias_map[scope_spec + (fn.string, )]
        ret_declist = ret_decl.parent
        utils.remove_children(ret_declist, ret_decl)
        if not ret_declist.children:
            utils.remove_self(ret_declist.parent)
        utils.remove_self(sf)

    return ast


def deconstuct_goto_statements(ast: f03.Program) -> f03.Program:
    """
    Attempts to convert `GOTO` statements into structured `IF` constructs.

    This pass replaces `GOTO` statements with structured control flow by introducing
    boolean flag variables. This simplifies the control flow graph, making it more
    amenable to analysis.

    The current implementation supports a specific pattern:
    - It handles forward-facing `GOTO` statements that jump out of an `IF` block
      to a `CONTINUE` statement later in the same execution part.
    - A boolean variable (e.g., `goto_1`) is created for each `GOTO`.
    - The `GOTO` is replaced by an assignment setting the flag to `.true.`.
    - The code between the original `IF` block and the `GOTO`'s target label is
      wrapped in a new `IF` condition that checks if the flag is `.not. .true.`.

    This effectively transforms a jump into a series of conditional blocks.

    NOTE: This pass has limited support and will raise a `NotImplementedError`
    for more complex `GOTO` patterns (e.g., backward jumps).

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    # TODO: Support `Compound_Goto_Stmt`.
    for node in walk(ast, Base):
        # Move any label on a non-continue statement onto one (except for format statement which require one).
        if not isinstance(node, (f03.Continue_Stmt, f03.Format_Stmt)) and node.item and node.item.label is not None:
            cont = f03.Continue_Stmt("CONTINUE")
            cont.item = node.item
            node.item = None
            utils.replace_node(node, (cont, node))

    labels: Dict[str, Base] = {}
    for node in walk(ast, Base):
        if node.item and node.item.label is not None and isinstance(node, f03.Continue_Stmt):
            labels[str(node.item.label)] = node

    # TODO: We have a very limited supported pattern of GOTO here, and possibly need to expand.
    # Assumptions: Each GOTO goes only forward. The target's parent is same as either the parent or the grandparent of
    # the GOTO. If the GOTO and its target have different parents, then the GOTO's parent is a if-construct.

    COUNTER = 0
    for goto in walk(ast, f03.Goto_Stmt):
        target, = goto.children
        target = target.string
        target = labels[target]
        if goto.parent == target.parent:
            raise NotImplementedError

        ifc = goto.parent
        ifc_par = ifc.parent
        assert isinstance(ifc, (f03.If_Stmt, f03.If_Construct)), \
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
        while not isinstance(ex, f03.Execution_Part):
            ex = ex.parent
        spec = ast_utils.atmost_one(ast_utils.children_of_type(ex.parent, f03.Specification_Part))
        assert spec

        if not isinstance(ifc, (f03.If_Stmt, f03.If_Construct)):
            raise NotImplementedError

        goto_var, COUNTER = f"goto_{COUNTER}", COUNTER + 1
        utils.append_children(spec, f03.Type_Declaration_Stmt(f"LOGICAL :: {goto_var}"))
        utils.replace_node(goto, f03.Assignment_Stmt(f"{goto_var} = .true."))

        for else_op in ifc_par.children[ifc_pos + 1:target_pos]:
            if isinstance(else_op, f03.Continue_Stmt):
                # Continue statements are no-op, but they may have label attached, so we leave them be.
                continue
            if isinstance(else_op, f03.If_Stmt):
                # We merge the condition with existing if.
                cond, op = else_op.children
                nu_cond = f03.Expr(f".not.({goto_var}) .and. {cond}")
                utils.replace_node(cond, nu_cond)
            elif isinstance(else_op, f03.If_Construct):
                # We merge the condition with existing if.
                for c in else_op.children:
                    if isinstance(c, f03.If_Then_Stmt):
                        cond, = c.children
                        nu_cond = f03.Expr(f".not.({goto_var}) .and. {cond}")
                        utils.replace_node(cond, nu_cond)
                    elif isinstance(c, f03.Else_If_Stmt):
                        cond, _ = c.children
                        nu_cond = f03.Expr(f".not.({goto_var}) .and. {cond}")
                        utils.replace_node(cond, nu_cond)
                    elif isinstance(c, f03.Else_Stmt):
                        nu_else = f03.Else_If_Stmt(f"else if (.not.({goto_var})) then")
                        utils.replace_node(c, nu_else)
                    else:
                        continue
            else:
                nu_if = f03.If_Stmt(f"if (.not.({goto_var})) call x")
                utils.replace_node(else_op, nu_if)
                utils.replace_node(ast_utils.singular(nm for nm in walk(nu_if, f03.Call_Stmt)), else_op)

        utils.replace_node(ifc, [f03.Assignment_Stmt(f"{goto_var} = .false."), ifc])

    return ast
