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
from typing import Tuple, Dict, List, Set, Union
from io import StringIO
import warnings

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.utils import Base, BlockBase, UnaryOpBase, BinaryOpBase, walk
from fparser.common.readfortran import FortranReaderBase
from fparser.common.sourceinfo import FortranFormat

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


def deconstruct_goto_statements(ast: f03.Program) -> f03.Program:
    """
    Attempts to convert `GOTO` statements into structured `IF` or 'DO WHILE' construct(s).

    All `GOTO`-target pairs are first collated and classified as forward or backward jumps. Each 
    jump is then deconstructed individually by the corresponding forward/backward helper function. 

    NOTE: The order in which `GOTO`s are deconstructed affects correctness. All backward `GOTO`s 
    should be deconstructed before forward `GOTO`s. Backward `GOTO`s should also be processed in 
    descending order of the line number of the `GOTO` statement.
    (This ordering ensures correct control flow due to the addition of `DO WHILE` loop constructs 
    in backward `GOTO` deconstruction.)

    NOTE: Only GOTOs in the subtree from parent of target are currently implemented.
    TODO: Support complex nesting.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    for node in walk(ast, Base):
        # Move any label on a non-continue statement onto one (except for format statement which require one).
        if not isinstance(node, (f03.Continue_Stmt, f03.Format_Stmt)) and node.item and node.item.label is not None:
            cont = f03.Continue_Stmt("CONTINUE")
            cont.item = node.item
            node.item = None

            if isinstance(node, (f03.If_Then_Stmt, f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt)):
                node_pos = ast_utils.singular(iter([i for i, x in enumerate(node.parent.children) if x is node]))
                assert (node_pos == 0)
                utils.replace_node(node.parent, (cont, node.parent))
            else:
                utils.replace_node(node, (cont, node))

    # Process each module/function/subroutine, one at a time.
    # Upon moving to next node of type in SCOPE_OBJECT_TYPES, all `GOTO`s in current scope_ast should be deconstructed. 
    for scope_ast in walk(ast, Union[f03.Function_Subprogram, f03.Subroutine_Subprogram]):
        # Maintain list of forward- and backward-facing gotos. Each entry is a tuple of (goto_node, target_node)
        forward_gotos = []
        backward_gotos = []

        # Resolve `GOTO` statements, classify into forward- and backward-facing.
        for goto in walk(scope_ast, f03.Goto_Stmt):
            label, = goto.children
            label = label.string

            # Inefficient search of target by walking through scope of parent function/subroutine.
            # Required, as subroutines can contain other subroutines.
            ancestor = goto.parent
            while ancestor and not isinstance(ancestor, (f03.Function_Subprogram, f03.Subroutine_Subprogram)):
                ancestor = ancestor.parent
            assert ancestor, f"Ancestor not found for GOTO {goto.item}."
            ancestor_subroutine = ancestor

            target = None
            for cont_node in walk(ancestor_subroutine, f03.Continue_Stmt):
                if (str(cont_node.item.label) == label):
                    assert (target is None), f"Multiple instances of label {label} found in subroutine/function {ancestor_subroutine.content[0].items[1]}. Expected each label to be unique."
                    target = cont_node
            assert target, f"Target of GOTO {goto.item} not found in scope of its parent Subroutine/Function."

            # Check that `GOTO` is in subtree from parent of `CONTINUE`.
            goto_in_parent_ast = (utils.lineage(target.parent, goto) is not None)
            if not goto_in_parent_ast:
                raise NotImplementedError("Only GOTOs in the subtree from parent of target are supported.")

            # Note: If `CONTINUE` was created from non-continue label, the `CONTINUE` is prepended into original node.
            # In this case, ensure `GOTO` is in subtree from -grandparent- of `CONTINUE`.
            # goto_in_grandparent_ast = (target == target.parent.children[0]) and (utils.lineage(target.parent.parent, goto) is not None)
            # if not (goto_in_parent_ast or goto_in_grandparent_ast):
            #     raise NotImplementedError("Only GOTOs in the subtree from parent/grandparent of target are supported.")

            # Determine whether `GOTO` is forward- or backward-facing.
            # We use the `walk` method of fparser (DFS) - checking whether goto or target encountered first.
            for node in walk(ancestor_subroutine, Base):
                if (node is target):
                    backward_gotos.append((ancestor_subroutine, goto, target))
                    break
                elif (node is goto):
                    forward_gotos.append((ancestor_subroutine, goto, target))
                    break

        # Sort backward_gotos list in descending order of line number of `GOTO` 
        # (`list.reverse()` also works due to order of adding nodes by DFS).
        # Attempted to sort by line number using `goto.item.span[0]`, but oddly not all `f03.Goto_Stmt` nodes have the line number in item.
        backward_gotos.reverse()

        for (ancestor_subroutine, goto, target) in backward_gotos:
            ancestor_subroutine = deconstruct_backward_goto_statements(ancestor_subroutine, goto, target)

        for (ancestor_subroutine, goto, target) in forward_gotos:
            ancestor_subroutine = deconstruct_forward_goto_statements(ancestor_subroutine, goto, target)

    return ast


def deconstruct_forward_goto_statements(ancestor_subroutine : Union[f03.Function_Subprogram, f03.Subroutine_Subprogram], goto: f03.Goto_Stmt, target: f03.Continue_Stmt):
    """
    Replaces forward-facing `GOTO` statements (i.e. forward jumps) with structured control flow by
    introducing boolean flag variables.

    Current implementation:
    - A boolean variable (e.g., `goto_10`) is created for each `GOTO`, and initialized to `.false.` .
    - The `GOTO` is replaced by an assignment setting the flag to `.true.`.
    - The AST is traversed upwards from the replaced `GOTO` to the parent containing the target `CONTINUE`, 
      adding conditional execution/exits to code between `GOTO` and `CONTINUE` for the following cases:
      + If parent is a loop construct and does not contain target `CONTINUE`, add a conditional `EXIT` that 
        executes if flag is `.true.`.
      + Otherwise, conditional execution is added to succeeding code until `CONTINUE` statement, to execute 
        only if flag is `.false.`.

    NOTE: Only `GOTO`s in the subtree from parent of `CONTINUE` are currently implemented.

    :param ancestor_subroutine: The subroutine or function AST containing `GOTO` and target.
    :param goto: `GOTO` statement node.
    :param target: 'CONTINUE` statement node that is target of `GOTO`.
    :return: The modified subroutine/function containing deconstructed `GOTO`-target.
    """
    exec_part = ast_utils.singular(ast_utils.children_of_type(ancestor_subroutine, f03.Execution_Part))
    spec_part = ast_utils.singular(ast_utils.children_of_type(ancestor_subroutine, f03.Specification_Part))
    label = target.item.label
    goto_var = f"goto_{label}" # Only label used for readability. TODO: Can include COUNTER

    # Check whether target has been deconstructed before. If not, perform deconstruction.
    if goto_var not in spec_part.tostr():
        # Add boolean flag `goto_{label}` to scope.
        utils.append_children(spec_part, f03.Type_Declaration_Stmt(f"LOGICAL :: {goto_var}"))
        utils.prepend_children(exec_part, f03.Assignment_Stmt(f"{goto_var} = .false.")) # initialize flag
        
        # Unset flag after `CONTINUE` 
        utils.replace_node(target, (target, f03.Assignment_Stmt(f"{goto_var} = .false.")))

    # Replace `GOTO` with assignement of flag to .true.
    goto_replacement = f03.Assignment_Stmt(f"{goto_var} = .true.")
    utils.replace_node(goto, goto_replacement)
    goto = goto_replacement

    # Transform ancestors of `GOTO`. We traverse up the AST, from `goto.parent` to `target.parent`.
    child_w_goto = goto
    par = child_w_goto.parent

    for _n in range(len(utils.lineage(target.parent, goto))-1):
        # determine position of `GOTO`/ancestor of `GOTO`, and `CONTINUE` (if applicable)
        child_pos = ast_utils.singular(iter([i for i, x in enumerate(par.children) if x is child_w_goto]))
        target_pos = ast_utils.singular(iter([i for i, x in enumerate(par.children) if x is target])) if par is target.parent else None

        if isinstance(par, (f03.Block_Label_Do_Construct, f03.Block_Nonlabel_Do_Construct)) and (par is not target.parent):
            # if EXIT has been previously added, break loop
            if (len(par.children) > (child_pos + 1)) and isinstance(par.children[child_pos+1], f03.If_Stmt) and (f"IF ({goto_var}) EXIT" in par.children[child_pos+1].tostr()):
                break

            # otherwise, insert conditional EXIT after child with goto
            utils.replace_node(child_w_goto, (child_w_goto, f03.If_Stmt(f"IF ( {goto_var} ) EXIT")))

        else:
            # For operations between (1) `GOTO` and end of parent block, (2) an ancestor of `GOTO` and `CONTINUE`, or (3) `GOTO` and `CONTINUE`,
            # Wrap these operations in a new `IF` condition that executes if flag is not set (`.not. goto_{label}`). 
            cond_str = f".not. {goto_var}"
            add_condition_to_node_execution(cond_str, par.children[child_pos+1:target_pos])

        child_w_goto = par
        par = child_w_goto.parent

    return ancestor_subroutine

def deconstruct_backward_goto_statements(ancestor_subroutine : Union[f03.Function_Subprogram, f03.Subroutine_Subprogram], goto: f03.Goto_Stmt, target: f03.Continue_Stmt):
    """
    Replaces backward-facing `GOTO` statements (i.e. forward jumps) with `DO WHILE` loop and structured 
    control flow with an introduced boolean flag variables.

    If backward `GOTO` is already in a `DO WHILE` block implemented for the same target `CONTINUE`, the 
    existing `DO WHILE` and boolean flag is reused.

    Current implementation:
    - If not existing, a boolean flag (e.g., `goto_10`) is created for each `CONTINUE`.
    - If not existing, the `CONTINUE` is replaced with a statement setting boolean flag to `.true.`.
    - If not existing, the code between `CONTINUE` and the last `GOTO` targeting it (highest line number) 
      is wrapped in a `DO WHILE` block, which executes when the flag is `.true.`. Upon entry to the loop 
      block, the flag is set to `.false.`.
    - The `GOTO` is replaced by an assignment setting the flag to `.true.`. 
    - The AST is traversed upwards from the replaced `GOTO` to the added `DO WHILE`, adding conditional 
      execution/exits to the succeeding code after the `GOTO` (similar to foward `GOTO`s):
      + If parent is a loop construct that is NOT the added `DO WHILE`, add a conditional `EXIT` that 
        executes if flag is `.true.`.
      + Otherwise, conditional execution is added to succeeding code to execute if flag is `.false.`.

    NOTE: Only `GOTO`s in the subtree from parent of `CONTINUE` are currently implemented.

    :param ancestor_subroutine: The subroutine or function AST containing `GOTO` and target.
    :param goto: `GOTO` statement node.
    :param target: 'CONTINUE` statement node that is target of `GOTO`.
    :return: The modified subroutine/function containing deconstructed `GOTO`-target.
    """
    spec_part = ast_utils.singular(ast_utils.children_of_type(ancestor_subroutine, f03.Specification_Part))
    label = target.item.label
    goto_var = f"goto_{label}" # Only label used for readability. TODO: Can include COUNTER

    # Check whether target has already been deconstructed into DO WHILE loop. If not, perform deconstruction.
    if goto_var not in spec_part.tostr():
        # Add boolean flag `goto_{label}` to scope.
        utils.append_children(spec_part, f03.Type_Declaration_Stmt(f"LOGICAL :: {goto_var}"))

        # Find ancestor of GOTO at the same tree depth as `CONTINUE`
        anc = goto
        while anc.parent and (anc.parent is not target.parent):
            anc = anc.parent
        assert (anc.parent is target.parent)

        target_pos = ast_utils.singular(iter([i for i, x in enumerate(target.parent.children) if x is target]))
        anc_pos = ast_utils.singular(iter([i for i, x in enumerate(anc.parent.children) if x is anc]))
        children_to_wrap = target.parent.children[target_pos+1:anc_pos+1]

        # Wrap blocks between `CONTINUE` and `GOTO` in loop
        do_while_constr_buffer = StringIO(f"DO WHILE ({goto_var})\n {goto_var} = .false.\n CALL x\nEND DO")
        do_while_constr_reader = FortranReaderBase(do_while_constr_buffer, mode=FortranFormat(True, False), ignore_comments=True)
        do_while_constr = f03.Block_Nonlabel_Do_Construct(do_while_constr_reader)
        utils.remove_children(target.parent, children_to_wrap)
        utils.replace_node(ast_utils.singular(nm for nm in walk(do_while_constr, f03.Call_Stmt)), children_to_wrap)
        utils.replace_node(target, (target, do_while_constr))

        # Set flag to .true. before loop.
        utils.replace_node(target, (target, f03.Assignment_Stmt(f"{goto_var} = .true.")))

    # Replace `GOTO` with assignement of flag to .true.
    goto_replacement = f03.Assignment_Stmt(f"{goto_var} = .true.")
    utils.replace_node(goto, goto_replacement)
    goto = goto_replacement

    # Transform ancestors of `GOTO`. We traverse up the AST, from `goto.parent` to `DO WHILE` wrapper.
    child_w_goto = goto
    par = child_w_goto.parent

    for _n in range(len(utils.lineage(target.parent, goto))-2):
        # determine position of `GOTO`/ancestor of `GOTO`
        child_pos = ast_utils.singular(iter([i for i, x in enumerate(par.children) if x is child_w_goto]))

        if isinstance(par, f03.Block_Nonlabel_Do_Construct) and (f"DO WHILE ({goto_var})" in par.content[0].tostr()):
            # If parent is the `DO WHILE` wrapper we created, no action needed and deconstruction is complete.
            break

        elif isinstance(par, (f03.Block_Label_Do_Construct, f03.Block_Nonlabel_Do_Construct)):
            # if EXIT has been previously added, break loop
            if (len(par.children) > (child_pos + 1)) and isinstance(par.children[child_pos+1], f03.If_Stmt) and (f"IF ({goto_var}) EXIT" in par.children[child_pos+1].tostr()):
                break

            # otherwise, insert conditional EXIT after child with goto
            utils.replace_node(child_w_goto, (child_w_goto, f03.If_Stmt(f"IF ( {goto_var} ) EXIT")))

        else:
            # For operations between (1) `GOTO` and end of parent block, (2) an ancestor of `GOTO` and `CONTINUE`, or (3) `GOTO` and `CONTINUE`,
            # Wrap these operations in a new `IF` condition that executes if flag is not set (`.not. goto_{label}`). 
            cond_str = f".not. {goto_var}"
            add_condition_to_node_execution(cond_str, par.children[child_pos+1:])

        child_w_goto = par
        par = child_w_goto.parent

    return ancestor_subroutine

def add_condition_to_node_execution(cond: Union[str, UnaryOpBase, BinaryOpBase], nodes: Union[Base, List[Base]]):
    """
    Adds a condition to the execution of given nodes. Nodes are executed only if condition is evaluated
    as `.true.`.

    :param cond: The condition (logical expression) to add.
    :param nodes: Node or list of nodes to add conditional execution.
    """
    if isinstance(cond, (UnaryOpBase, BinaryOpBase)):
        cond = cond.tostr()
    
    if isinstance(nodes, Base):
        nodes = [nodes]

    for node in nodes:
        if isinstance(node, (f03.Continue_Stmt, f03.EndStmtBase)):
            # Continue statements are no-op, but they may have label attached, so we leave them be.
            continue
        elif isinstance(node, (f03.If_Stmt, f03.Else_If_Stmt)):
            # We merge the condition with existing if.
            own_cond = node.children[0]
            if cond in own_cond.tostr():
                continue
            new_cond = f03.Expr(f"{cond} .and. {own_cond.tostr()}")
            utils.replace_node(own_cond, new_cond)
        elif isinstance(node, f03.Else_Stmt):
            new_else = f03.Else_If_Stmt(f"else if ({cond}) then")
            utils.replace_node(node, new_else)
        elif isinstance(node, (f03.If_Construct, f03.Block_Label_Do_Construct, f03.Block_Nonlabel_Do_Construct, f03.Case_Construct)):
            # Wrap block in `If_Construct`
            if_constr_buffer = StringIO(f"IF ({cond}) THEN\n CALL x\nEND IF")
            if_constr_reader = FortranReaderBase(if_constr_buffer, mode=FortranFormat(True, False), ignore_comments=True)
            if_constr = f03.If_Construct(if_constr_reader)
            utils.replace_node(node, if_constr)
            utils.replace_node(ast_utils.singular(nm for nm in walk(if_constr, f03.Call_Stmt)), node)
        else:
            assert not isinstance(node, BlockBase), f"Encountered an unexpected BlockBase instance."
            new_if = f03.If_Stmt(f"if ({cond}) call x")
            utils.replace_node(node, new_if)
            utils.replace_node(ast_utils.singular(nm for nm in walk(new_if, f03.Call_Stmt)), node)


def deconstruct_external_statements(ast: f03.Program) -> f03.Program:
    """
    Attemps to convert `EXTERNAL` statements to `USE` statements.
    Raises warning if target module/subroutine/function is not found in program. In this case, the target 
    likely requires a library node that should be created separately.

    If target subroutine/function is in the program but not contained in a module, the subroutine/function 
    is wrapped in a new module with similar name (e.g. `foo_module`).

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST.
    """
    # Build a dictionary of subprogram name to subprogram node
    subprograms_wo_modules: Dict[str, Union[f03.Function_Subprogram, f03.Subroutine_Subprogram]] = {}
    # Build a dictionary of subprogram name to module
    new_modules_name_map: Dict[str, str] = {}
    # Keep track of functions/subroutines with unsupported features. Raise error only if `EXTERNAL` references them.
    not_supported : list[str] = []

    # (1) Populate subprograms_wo_modules
    for node in walk(ast, (f03.Function_Stmt, f03.Subroutine_Stmt)):
        subprog_name = node.items[1].tostr()
        subprogram = node.parent

        # Check whether already in Module. If yes, do nothing and continue to next subprogram.
        anc = subprogram
        while anc and not isinstance(anc, f03.Module):
            anc = anc.parent
        if isinstance(anc, f03.Module):
            continue

        # TODO: Check Fortran standard on referencing subroutines/functions contained inside other subroutine/function.
        # Only Function_Subprogram and Subroutine_Subprogram supported for now (not yet: Subroutine_Body, Function_Body). Raise exception later if referenced by `EXTERNAL`.
        if not isinstance(subprogram, (f03.Function_Subprogram, f03.Subroutine_Subprogram)):
            not_supported.append(subprog_name)

        # NOT supported: multiple subprograms with same name.
        # TODO: Support same names according to scope.
        if (subprog_name in subprograms_wo_modules):
            not_supported.append(subprog_name)
            # warnings.warn(f"When deconstructing `EXTERNAL` statements, found duplicate names of subprogram {subprog_name} (can ignore if not referenced).")

        subprograms_wo_modules[subprog_name] = subprogram

    # (2) Create new modules for subprograms referenced by `EXTERNAL` statements.
    for node in walk(ast, (f03.External_Stmt, f03.Type_Declaration_Stmt)):
        # Get external subprogram names ( EXTERNAL :: [ ext_subprogram_names:list ] )
        if isinstance(node, f03.External_Stmt):
            ext_subprogram_names = node.children[1].children
        elif isinstance(node, f03.Type_Declaration_Stmt):
            attr_spec_list = ast_utils.atmost_one(ast_utils.children_of_type(node, f08.Attr_Spec_List))
            if attr_spec_list and ('EXTERNAL' in attr_spec_list.tostr()):
                entity_decl_list = ast_utils.singular(ast_utils.children_of_type(node, f03.Entity_Decl_List))
                ext_subprogram_names = entity_decl_list.children
            else:
                continue

        # Process each external reference
        for name_node in ext_subprogram_names:
            name_str = name_node.tostr()

            if name_str in new_modules_name_map:
                continue

            if name_str in not_supported:
                raise NotImplementedError(f"Function/subroutine with name {name_str} referenced in EXTERNAL statement, but not yet supported. Possible causes: (1) name shared by several functions, (2) Subroutine_Body not in Module.")

            # If subprogram not in parsed code, a warning is raised for now. These may be calls to library nodes, to be added later.
            if (name_str not in subprograms_wo_modules):
                warnings.warn(f"Unresolved `EXTERNAL` reference: {name_str} (subprogram not found in parsed code)")
                continue

            # Wrap in module (default name: {name_str}_module)
            module_new_buffer = StringIO(f"""
                MODULE {name_str}_module
                IMPLICIT NONE
                CONTAINS
                {subprograms_wo_modules[name_str].tostr()}
                END MODULE {name_str}_module
            """)
            module_new_reader = FortranReaderBase(module_new_buffer, mode=FortranFormat(True, False), ignore_comments=True)
            module_new = f03.Module(module_new_reader)

            # add module to AST at root
            utils.append_children(ast, module_new)
            new_modules_name_map[name_str] = f"{name_str}_module"

            utils.replace_node(subprograms_wo_modules[name_str], None) # remove original subprogram from AST if newly wrapped in module.
            subprograms_wo_modules.pop(name_str)
        
    # (3) For each function/subroutine - convert `EXTERNAL` statements to `USE`, if present.
    for subprogram_node in walk(ast, (f03.Function_Subprogram, f03.Subroutine_Subprogram, f03.Module)):
        spec_part = ast_utils.atmost_one(ast_utils.children_of_type(subprogram_node, f03.Specification_Part))
        if spec_part is None:
            continue

        nodes_to_remove = []
        for ext_node in walk(spec_part, (f03.External_Stmt, f03.Type_Declaration_Stmt)):
            # Get external subprogram names ( EXTERNAL :: [ ext_subprogram_names:list ] )
            # This part is repetitive, but object management is cleaner by separating passes (2) and (3)
            if isinstance(ext_node, f03.External_Stmt):
                ext_subprogram_names = ext_node.children[1].children
            elif isinstance(ext_node, f03.Type_Declaration_Stmt):
                attr_spec_list = ast_utils.atmost_one(ast_utils.children_of_type(ext_node, f08.Attr_Spec_List))
                if attr_spec_list and ('EXTERNAL' in attr_spec_list.tostr()):
                    entity_decl_list = ast_utils.singular(ast_utils.children_of_type(ext_node, f03.Entity_Decl_List))
                    ext_subprogram_names = entity_decl_list.children
                else:
                    continue

            for name_node in ext_subprogram_names:
                name_str = name_node.tostr()
                if name_str in new_modules_name_map:
                    utils.prepend_children(spec_part, f03.Use_Stmt(f"USE {new_modules_name_map[name_str]}"))
            
            if all([(name_node.tostr() in new_modules_name_map) for name_node in ext_subprogram_names]):
                nodes_to_remove.append(ext_node)
            else:
                nodes_to_remove += [n for n in ext_subprogram_names if (n.tostr() in new_modules_name_map)]

        utils.remove_self(nodes_to_remove)
        # for node in nodes_to_remove:
        #     utils.remove_children(node.parent, node)

    return ast