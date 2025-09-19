"""
This module contains a collection of AST cleanup and canonicalization passes for the
Fortran frontend. These passes are designed to run after the initial parsing and
desugaring stages. They perform tasks such as disambiguating language constructs
(e.g., array access vs. function calls), standardizing identifier names, removing
constructs not relevant for dataflow analysis, and restructuring global data to
simplify subsequent processing.
"""
from typing import Union, Set, Dict, List, Optional, Tuple

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.utils import walk, NumberBase

from . import analysis
from . import pruning
from . import types
from . import utils
from .. import ast_utils


def correct_for_function_calls(ast: f03.Program):
    """
    A cleanup pass to disambiguate array accesses from function calls.

    In Fortran, array access `A(i)` and function calls `F(x)` have identical syntax.
    Fparser often defaults to parsing them as `Part_Ref` (array access). This function
    uses type analysis to correct these misidentifications.

    The process involves several steps:
    1.  Identify and convert statement functions (e.g., `f(x) = x*x`) which are initially
        parsed as assignment statements.
    2.  Iteratively walk the AST to find `Part_Ref` nodes. If the base of a `Part_Ref`
        does not resolve to a variable with a known type (i.e., an array or scalar),
        it is reclassified as a `Function_Reference`.
    3.  Do the same for `Structure_Constructor` nodes, which can also be disguised
        function calls.
    4.  Finally, identify calls to standard Fortran intrinsics (e.g., `SIN`, `MAX`) and
        replace the generic `Function_Reference` node with a specific
        `Intrinsic_Function_Reference` node.

    :param ast: The Fortran AST to modify.
    """
    alias_map = analysis.alias_specs(ast)

    # First, look for statement functions, which are parsed as assignments to array-like expressions.
    for asgn in walk(ast, f03.Assignment_Stmt):
        lv, _, _ = asgn.children
        if not isinstance(lv, (f03.Part_Ref, f03.Structure_Constructor, f03.Function_Reference)):
            continue
        if walk(lv, f03.Subscript_Triplet):
            # If the LHS contains a subscript triplet, it's an array section, not a statement function.
            continue
        lv, _ = lv.children
        lvloc = analysis.search_real_local_alias_spec(lv, alias_map)
        if not lvloc:
            continue
        lv = alias_map[lvloc]
        if not isinstance(lv, f03.Entity_Decl):
            continue
        lv_type = analysis.find_type_of_entity(lv, alias_map)
        if not lv_type or lv_type.shape:
            # If it has a shape, it's an array, not a statement function.
            continue

        # Now we know that this identifier actually refers to a statement function.
        stmt_fn = f03.Stmt_Function_Stmt(asgn.tofortran())
        ex = asgn.parent
        # Find the enclosing execution part to correctly place the statement function.
        while not isinstance(ex, f03.Execution_Part):
            ex = ex.parent
        sp = ast_utils.atmost_one(ast_utils.children_of_type(ex.parent, f03.Specification_Part))
        assert sp
        utils.append_children(sp, stmt_fn)
        utils.remove_self(asgn)

    alias_map = analysis.alias_specs(ast)

    # Iteratively disambiguate Part_Ref nodes. This loop runs until no more changes are made,
    # as fixing one Part_Ref might reveal another that needs fixing.
    # TODO: Looping over and over is not ideal. But `Function_Reference(...)` sometimes generate inner `Part_Ref`s. We
    #  should figure out a way to avoid this clutter.
    changed = None
    while changed is None or changed:
        changed = False
        for pr in walk(ast, f03.Part_Ref):
            if isinstance(pr.parent, f03.Data_Ref):
                dref = pr.parent
                scope_spec = analysis.find_scope_spec(dref)
                comp_spec = analysis.find_dataref_component_spec(dref, scope_spec, alias_map)
                comp_type_spec = analysis.find_type_of_entity(alias_map[comp_spec], alias_map)
                if not comp_type_spec:
                    # If no type can be found for the component, it's likely a function call.
                    utils.replace_node(dref, f03.Function_Reference(dref.tofortran()))
                    changed = True
            else:
                pr_name, _ = pr.children
                if isinstance(pr_name, f03.Name):
                    pr_spec = analysis.search_real_local_alias_spec(pr_name, alias_map)
                    if pr_spec in alias_map and isinstance(alias_map[pr_spec], (f03.Function_Stmt, f03.Interface_Stmt)):
                        # If the name resolves to a function/interface, it's a function reference.
                        utils.replace_node(pr, f03.Function_Reference(pr.tofortran()))
                        changed = True
                elif isinstance(pr_name, f03.Data_Ref):
                    scope_spec = analysis.find_scope_spec(pr_name)
                    pr_type_spec = analysis.find_type_dataref(pr_name, scope_spec, alias_map)
                    if not pr_type_spec:
                        # If no type can be found for the data reference, it's likely a function call.
                        utils.replace_node(pr, f03.Function_Reference(pr.tofortran()))
                        changed = True

    # Disambiguate Structure_Constructor nodes that might actually be function calls.
    for sc in walk(ast, f03.Structure_Constructor):
        scope_spec = analysis.find_scope_spec(sc)

        # TODO: Add ref.
        sc_type, _ = sc.children
        sc_type_spec = analysis.find_real_ident_spec(sc_type.string, scope_spec, alias_map)
        sc_decl = alias_map[sc_type_spec]
        if isinstance(sc_decl, (f03.Function_Stmt, f03.Interface_Stmt, f03.Stmt_Function_Stmt)):
            # If the constructor's name resolves to a function, it's a function reference.
            utils.replace_node(sc, f03.Function_Reference(sc.tofortran()))

    # Identify and convert generic Function_Reference/Call_Stmt nodes to Intrinsic_Function_Reference if applicable.
    for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        scope_spec = analysis.find_scope_spec(fref)

        name, args = fref.children
        name = name.string
        if not f03.Intrinsic_Name.match(name):
            # If the name does not match an intrinsic, skip.
            continue
        fref_spec = scope_spec + (name, )
        if fref_spec in alias_map:
            # If an intrinsic name is shadowed by a user-defined entity, it's not an intrinsic call.
            continue
        if isinstance(fref, f03.Function_Reference):
            # Replace with specific intrinsic node, ensuring arguments are preserved.
            repl = f03.Intrinsic_Function_Reference(fref.tofortran())
            utils.set_children(repl, (f03.Intrinsic_Name(name), args))
            utils.replace_node(fref, repl)
        else:
            # For Call_Stmt, just update the name to Intrinsic_Name.
            utils.set_children(fref, (f03.Intrinsic_Name(name), args))

    return ast


def remove_access_and_bind_statements(ast: f03.Program):
    """
    Removes access-control and language-binding statements from the AST.

    This pass simplifies the tree by removing:
    - `PUBLIC` and `PRIVATE` statements (`Access_Stmt`).
    - `PRIVATE` declarations within derived types (`Private_Components_Stmt`).
    - `BIND(C, ...)` specifications from function/subroutine definitions and type declarations.

    These are removed because they are not relevant for the dataflow analysis and code
    generation that follows.

    :param ast: The Fortran AST to modify.
    """
    # TODO: This can get us into ambiguity and unintended shadowing.

    # We also remove any access statement that makes these interfaces public/private.
    for acc in walk(ast, f03.Access_Stmt):
        # TODO: Add ref.
        kind, alist = acc.children
        assert kind.upper() in {"PUBLIC", "PRIVATE"}
        spec = acc.parent
        utils.remove_self(acc)
        if not spec.children:
            # If the parent (e.g., Specification_Part) becomes empty, remove it too.
            utils.remove_self(spec)

    for acc in walk(ast, f03.Private_Components_Stmt):
        utils.remove_self(acc)

    for bind in walk(ast, f03.Language_Binding_Spec):
        if isinstance(bind.parent, (f03.Suffix, f03.Subroutine_Stmt, f03.Function_Stmt)):
            # Since this is part of a tuple, we need to replace it with a `None` to maintain tuple structure.
            utils.replace_node(bind, None)
        else:
            par = bind.parent
            utils.remove_self(bind)
            if not par.children:
                # If the parent becomes empty, replace it with `None`.
                utils.replace_node(par, None)
    for bind in walk(ast, f03.Type_Attr_Spec):
        b, c = bind.children
        if b == 'BIND':
            par = bind.parent
            utils.remove_self(bind)
            if not par.children:
                # If the parent becomes empty, replace it with `None`.
                utils.replace_node(par, None)

    return ast


def assign_globally_unique_subprogram_names(ast: f03.Program, keepers: Set[types.SPEC]) -> f03.Program:
    """
    Renames all subprograms to ensure they have globally unique names.

    This pass prevents name collisions that can occur when different modules define
    functions or subroutines with the same name. It makes all subprogram names unique
    across the entire program, simplifying subsequent analysis and transformations.

    The process is as follows:
    1.  **Collision Detection**: Identifies all subprogram names that are duplicated
        across different scopes or that conflict with reserved keywords.
    2.  **Name Generation**: Creates a unique name (e.g., `original_name_fn_0`) for each
        conflicting subprogram that is not marked as a `keeper` (entry point).
    3.  **AST Update**: Traverses the AST to:
        a. Remove old `USE ... ONLY` imports for subprograms that will be renamed.
        b. Replace all call sites (`CALL` statements and function references) with the
           new unique names.
        c. Insert new `USE` statements where needed to import the renamed subprograms
           from their parent modules into the scopes where they are called.
        d. Rename the subprogram definitions themselves (`SUBROUTINE`/`FUNCTION` and
           `END SUBROUTINE`/`END FUNCTION` statements).
        e. Handle the special case of the implicit function return variable.

    :param ast: The Fortran AST to modify.
    :param keepers: A set of specifications for subprograms (entry points) that should
                    not be renamed.
    :return: The modified Fortran AST.
    """
    SUFFIX, COUNTER = 'fn', 0

    ident_map = analysis.identifier_specs(ast)
    alias_map = analysis.alias_specs(ast)

    # Collect all known subprogram names and identify those with collisions or reserved keywords.
    known_names: Set[str] = {k[-1] for k in ident_map.keys()}
    name_collisions: Dict[str, int] = {k: 0 for k in known_names}
    for k in ident_map.keys():
        name_collisions[k[-1]] += 1
    name_collisions: Set[str] = {k for k, v in name_collisions.items() if v > 1 or k.lower() in KEYWORDS_TO_AVOID}

    # Generate new unique names for subprograms that have collisions or are reserved.
    uident_map: Dict[types.SPEC, str] = {}
    for k in ident_map.keys():
        if k in keepers:
            # Keep entry-point subprograms with their original names.
            continue
        if k[-1] in name_collisions:
            # Generate a unique name by appending a suffix and counter.
            uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
            while uname in known_names:
                uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        else:
            # No collision, keep the original name.
            uname = k[-1]
        uident_map[k] = uname
    uident_map.update({k: k[-1] for k in keepers})

    # PHASE 1.a: Remove all the places where any to-be-renamed function is imported.
    # This is necessary because the new name will be imported later.
    for use in walk(ast, f03.Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(use, f03.Name)).string
        mod_spec = (mod_name, )
        olist = ast_utils.atmost_one(ast_utils.children_of_type(use, f03.Only_List))
        if not olist:
            continue
        survivors = []
        for c in olist.children:
            if isinstance(c, f03.Rename):
                # Renamed uses shouldn't survive, and should be replaced with direct uses.
                continue
            assert isinstance(c, f03.Name)
            tgt_spec = analysis.find_real_ident_spec(c.string, mod_spec, alias_map)
            assert tgt_spec in ident_map and tgt_spec in uident_map
            if not isinstance(ident_map[tgt_spec], (f03.Function_Stmt, f03.Subroutine_Stmt)):
                # We leave non-function uses alone.
                survivors.append(c)
        if survivors:
            utils.set_children(olist, survivors)
        else:
            utils.remove_self(use)

    # PHASE 1.b: Replace all the function callsites.
    for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        scope_spec = analysis.find_scope_spec(fref)

        # TODO: Add ref.
        name, _ = fref.children
        if not isinstance(name, f03.Name):
            # Intrinsics are not to be renamed, so skip.
            assert isinstance(name, f03.Intrinsic_Name), f"{fref}"
            continue
        fspec = analysis.find_real_ident_spec(name.string, scope_spec, alias_map)
        assert fspec in ident_map and fspec in uident_map
        assert isinstance(ident_map[fspec], (f03.Function_Stmt, f03.Subroutine_Stmt))
        uname = uident_map[fspec]
        ufspec = fspec[:-1] + (uname, )
        name.string = uname  # Apply the new unique name to the call site.

        # Find the nearest execution and its corresponding specification parts.
        execution_part = fref.parent
        while not isinstance(execution_part, f03.Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, f03.Specification_Part))

        # Determine the current module/program name to check if a USE statement is needed.
        cmod = fref.parent
        while cmod and not isinstance(cmod, (f03.Module, f03.Main_Program)):
            cmod = cmod.parent
        if cmod:
            stmt, _, _, _ = utils._get_module_or_program_parts(cmod)
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string.lower()
        else:
            # If not in a module/main program, it must be a nested subprogram.
            subp = list(ast_utils.children_of_type(ast, f03.Subroutine_Subprogram))
            assert subp
            stmt = ast_utils.singular(ast_utils.children_of_type(subp[0], f03.Subroutine_Stmt))
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string.lower()

        assert 1 <= len(ufspec)
        if len(ufspec) == 1:
            # Toplevel subprograms are globally visible, no USE statement needed.
            continue
        mod = ufspec[0]
        if mod == cmod:
            # If the function is defined in the current module, no USE statement needed.
            continue

        # Add a "use" statement for the renamed function. It will be consolidated later.
        if not specification_part:
            utils.append_children(subprog, f03.Specification_Part(get_reader(f"use {mod}, only: {uname}")))
        else:
            utils.prepend_children(specification_part, f03.Use_Stmt(f"use {mod}, only: {uname}"))

    # PHASE 1.d: Replaces actual function names in their definitions.
    for k, v in ident_map.items():
        if not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
            continue
        assert k in uident_map
        if uident_map[k] == k[-1]:
            # If the subprogram was not renamed, skip.
            continue
        oname, uname = k[-1], uident_map[k]
        # Update the name in the FUNCTION/SUBROUTINE statement.
        ast_utils.singular(ast_utils.children_of_type(v, f03.Name)).string = uname
        # Fix the name in the corresponding END FUNCTION/SUBROUTINE statement.
        fdef = v.parent
        end_stmt = ast_utils.singular(ast_utils.children_of_type(fdef,
                                                                 (f03.End_Function_Stmt, f03.End_Subroutine_Stmt)))
        kw, end_name = end_stmt.children
        utils.set_children(end_stmt, (kw, f03.Name(uname)))
        # For functions, the function name is also available as a variable inside (return value).
        if isinstance(v, f03.Function_Stmt):
            for nm in walk(tuple(ast_utils.children_of_type(fdef, (f03.Specification_Part, f03.Execution_Part))),
                           f03.Name):
                if nm.string != oname:
                    continue
                local_spec = analysis.search_local_alias_spec(nm)
                # Adjust the local spec to match the function's original spec for validation.
                # This handles cases where the function name itself is used as a variable.
                local_spec = local_spec[:-2] + local_spec[-1:]
                assert local_spec in ident_map, f"`{local_spec}` is not a valid identifier"
                assert ident_map[local_spec] is v, f"`{local_spec}` does not refer to `{v}`"
                nm.string = uname

    return ast


def add_use_to_specification(scdef: utils.SCOPE_OBJECT_TYPES, clause: str):
    """
    Adds a `USE` statement to the specification part of a scope.

    This is a utility function that finds the `Specification_Part` of a given scope
    (module, function, etc.) and prepends a `USE` statement to it. If the scope does
    not have a `Specification_Part`, one is created.

    :param scdef: The AST node of the scope to add the `USE` statement to.
    :param clause: The full `USE` statement to add, as a string (e.g., "use my_mod, only: var").
    """
    specification_part = ast_utils.atmost_one(ast_utils.children_of_type(scdef, f03.Specification_Part))
    if not specification_part:
        utils.append_children(scdef, f03.Specification_Part(get_reader(clause)))
    else:
        utils.prepend_children(specification_part, f03.Use_Stmt(clause))


KEYWORDS_TO_AVOID = {k.lower() for k in ('for', 'in', 'beta', 'input', 'this')}


def assign_globally_unique_variable_names(ast: f03.Program, keepers: Set[Union[str, types.SPEC]]) -> f03.Program:
    """
    Renames all variables to ensure they have globally unique names.

    This pass prevents name collisions that can occur when different modules define
    variables with the same name, or when variable names conflict with Fortran keywords.
    It makes all variable names unique across the entire program, simplifying subsequent
    analysis and transformations.

    The process is as follows:
    1.  **Collision Detection**: Identifies all variable names that are duplicated
        across different scopes, conflict with reserved keywords, or are entry-point
        arguments (unless explicitly kept).
    2.  **Name Generation**: Creates a unique name (e.g., `original_name_var_0`) for each
        conflicting variable that is not marked as a `keeper`.
    3.  **AST Update**: Traverses the AST to:
        a. Remove old `USE ... ONLY` imports for variables that will be renamed.
        b. Replace variable names used as keyword arguments in function calls.
        c. Replace all direct references to variables (`Name` nodes) with the new unique names.
        d. Replace variable names used as `KIND` specifiers in literal constants.
        e. Insert new `USE` statements where needed to import global variables from
           their parent modules into the scopes where they are used.
        f. Rename the variable declarations themselves (`Entity_Decl`).

    :param ast: The Fortran AST to modify.
    :param keepers: A set of specifications for variables (or variable names) that should
                    not be renamed. This can include entry-point arguments.
    :return: The modified Fortran AST.
    """
    SUFFIX, COUNTER = 'var', 0

    ident_map = analysis.identifier_specs(ast)
    alias_map = analysis.alias_specs(ast)

    # Collect all known variable names and identify those with collisions or reserved keywords.
    known_names: Set[str] = {k[-1].lower() for k in ident_map.keys()}
    name_collisions: Dict[str, int] = {k: 0 for k in known_names}
    for k in ident_map.keys():
        name_collisions[k[-1].lower()] += 1
    name_collisions: Set[str] = {k for k, v in name_collisions.items() if v > 1 or k in KEYWORDS_TO_AVOID}

    # Identify arguments of entry-point functions, which might be kept with their original names.
    entry_point_args: Set[types.SPEC] = set()
    for k in keepers:
        if k not in ident_map:
            continue
        fn = ident_map[k]
        if not isinstance(fn, (f03.Subroutine_Stmt, f03.Function_Stmt)):
            continue
        args = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Dummy_Arg_List))
        args = args.children if args else tuple()
        for a in args:
            entry_point_args.add(k + (a.string, ))

    # Generate new unique names for variables that have collisions, are reserved, or are not kept.
    uident_map: Dict[types.SPEC, str] = {}
    for k in ident_map.keys():
        if k[-1].lower() not in KEYWORDS_TO_AVOID and k in entry_point_args:
            # Keep the entry point arguments if possible, unless they conflict with keywords.
            continue
        if k in keepers:
            # Specific variable instances requested to keep.
            continue
        if k[-1] in keepers:
            # Specific variable _name_ (anywhere) requested to keep.
            continue
        if k[-1].lower() in name_collisions:
            # Generate a unique name by appending a suffix and counter.
            uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
            while uname in known_names:
                uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        else:
            # No collision, keep the original name.
            uname = k[-1]
        uident_map[k] = uname
    uident_map.update({k: k[-1] for k in keepers})

    # PHASE 1.a: Remove all the places where any to-be-renamed variable is imported.
    # This is necessary because the new name will be imported later.
    for use in walk(ast, f03.Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(use, f03.Name)).string
        mod_spec = (mod_name, )
        olist = ast_utils.atmost_one(ast_utils.children_of_type(use, f03.Only_List))
        if not olist:
            continue
        survivors = []
        for c in olist.children:
            if isinstance(c, f03.Rename):
                # Renamed uses shouldn't survive, and should be replaced with direct uses.
                continue
            assert isinstance(c, f03.Name)
            tgt_spec = analysis.find_real_ident_spec(c.string, mod_spec, alias_map)
            assert tgt_spec in ident_map and tgt_spec in uident_map
            if not isinstance(ident_map[tgt_spec], f03.Entity_Decl):
                # We leave non-variable uses alone.
                survivors.append(c)
        if survivors:
            utils.set_children(olist, survivors)
        else:
            utils.remove_self(use)

    # PHASE 1.b: Replace all variable names used as keywords in function calls.
    # This must be done early to avoid ambiguity (e.g., `fn(kw=kw)`).
    for kv in walk(ast, f03.Actual_Arg_Spec):
        fref = kv.parent.parent
        if not isinstance(fref, (f03.Function_Reference, f03.Call_Stmt)):
            # Only interested in user-defined function calls.
            continue
        callee, _ = fref.children
        if isinstance(callee, f03.Intrinsic_Name):
            # Intrinsics don't have their internal variables renamed.
            continue
        cspec = analysis.search_real_local_alias_spec(callee, alias_map)
        cspec = analysis.ident_spec(alias_map[cspec])
        assert cspec
        k, _ = kv.children
        assert isinstance(k, f03.Name)
        kspec = analysis.find_real_ident_spec(k.string, cspec, alias_map)
        assert kspec in ident_map and kspec in uident_map
        assert isinstance(ident_map[kspec], f03.Entity_Decl)
        k.string = uident_map[kspec]  # Apply the new unique name.

    # PHASE 1.c: Replace all direct references to variables.
    for vref in walk(ast, f03.Name):
        if isinstance(vref.parent, f03.Entity_Decl):
            # Do not change the variable declarations themselves just yet; only usages.
            continue
        vspec = analysis.search_real_local_alias_spec(vref, alias_map)
        if not vspec:
            # Not a valid alias (e.g., a structure component that is not a variable).
            continue
        if not isinstance(alias_map[vspec], f03.Entity_Decl):
            # Does not refer to a variable (e.g., a function name).
            continue
        edcl = alias_map[vspec]
        fdef = utils.find_scope_ancestor(edcl)
        if isinstance(fdef, f03.Function_Subprogram) and utils.find_name_of_node(fdef) == utils.find_name_of_node(edcl):
            # Function return variables must retain their names.
            continue

        scope_spec = analysis.find_scope_spec(vref)
        vspec = analysis.find_real_ident_spec(vspec[-1], scope_spec, alias_map)
        assert vspec in ident_map
        if vspec not in uident_map:
            # If the variable was not chosen for renaming, skip.
            # TODO: `vspec` **should** be in `uident_map` if it is a variable (whether we rename it or not).
            continue
        uname = uident_map[vspec]
        vref.string = uname  # Apply the new unique name.

        # If the variable is global (defined in a module), add a USE statement if necessary.
        if len(vspec) > 2:
            # If the variable is not defined in a toplevel object, so we're done.
            continue
        assert len(vspec) == 2
        mod, _ = vspec
        if not isinstance(alias_map[(mod, )], f03.Module_Stmt):
            # We can only import modules.
            continue

        # Find the nearest specification part (or lack thereof).
        scdef = alias_map[scope_spec].parent
        # Find out the current module name.
        cmod = scdef
        while not isinstance(cmod.parent, f03.Program):
            cmod = cmod.parent
        if utils.find_name_of_node(cmod) == mod:
            # If the variable is already defined in the current module, no import needed.
            continue
        add_use_to_specification(scdef, f"use {mod}, only: {uname}")

    # PHASE 1.d: Replace variable names used as "kind" specifiers in literal constants.
    for lit in walk(ast, f03.Real_Literal_Constant):
        val, kind = lit.children
        if not kind:
            continue
        # Fparser sometimes parses kind as a plain `str` instead of a `Name`.
        assert isinstance(kind, str)
        scope_spec = analysis.find_scope_spec(lit)
        kind_spec = analysis.search_real_ident_spec(kind, scope_spec, alias_map)
        if not kind_spec or kind_spec not in uident_map:
            continue
        uname = uident_map[kind_spec]
        utils.set_children(lit, (val, uname))  # Apply the new unique name.

        # If the kind variable is global, add a USE statement if necessary.
        if len(kind_spec) > 2:
            # If the variable is not defined in a toplevel object, so we're done.
            continue
        assert len(kind_spec) == 2
        mod, _ = kind_spec
        if not isinstance(alias_map[(mod, )], f03.Module_Stmt):
            # We can only import modules.
            continue

        # Find the nearest specification part (or lack thereof).
        scdef = alias_map[scope_spec].parent
        # Find out the current module name.
        cmod = scdef
        while not isinstance(cmod.parent, f03.Program):
            cmod = cmod.parent
        if utils.find_name_of_node(cmod) == mod:
            # If the variable is already defined in the current module, no import needed.
            continue
        add_use_to_specification(scdef, f"use {mod}, only: {uname}")

    # PHASE 1.e: Replace actual variable names in their declarations.
    for k, v in ident_map.items():
        if not isinstance(v, f03.Entity_Decl):
            continue
        if k not in uident_map or uident_map[k] == k[-1]:
            # If the variable was not chosen for renaming, skip.
            # TODO: `k` **should** be in `uident_map` if it is a variable (whether we rename it or not).
            continue
        oname, uname = k[-1], uident_map[k]
        fdef = utils.find_scope_ancestor(v)
        if isinstance(fdef, f03.Function_Subprogram) and utils.find_name_of_node(fdef) == oname:
            # Function return variables must retain their names.
            continue
        ast_utils.singular(ast_utils.children_of_type(v, f03.Name)).string = uname  # Apply the new unique name.

    return ast


def lower_identifier_names(ast: f03.Program) -> f03.Program:
    """
    Converts all Fortran identifiers and `KIND` specifiers to lowercase.

    Fortran is largely case-insensitive for identifiers. This pass normalizes all
    `Name` nodes and `KIND` specifiers in numeric literals to lowercase to ensure
    consistent handling throughout the AST.

    :param ast: The Fortran AST to modify.
    :return: The modified Fortran AST with all relevant names in lowercase.
    """
    for nm in walk(ast, f03.Name):
        nm.string = nm.string.lower()
    # Also lower-case kind specifiers in numeric literals (e.g., 1.0_REAL8).
    for num in walk(ast, NumberBase):
        val, kind = num.children
        if isinstance(kind, str):
            utils.set_children(num, (val, kind.lower()))
    return ast


GLOBAL_DATA_OBJ_NAME = 'global_data'
GLOBAL_DATA_TYPE_NAME = 'global_data_type'


def consolidate_global_data_into_arg(ast: f03.Program, always_add_global_data_arg: bool = False) -> f03.Program:
    """
    Consolidates all global variables into a single derived type and passes it as an argument.

    This pass simplifies the handling of global state by:
    1.  Identifying all variables declared at the module level (global variables).
    2.  Creating a new module named `global_mod`.
    3.  Within `global_mod`, defining a new derived type `global_data_type` that contains
        all the identified global variables as its components.
    4.  Adding a new argument (`global_data`) of type `global_data_type` to the
        signature of every function and subroutine.
    5.  Replacing all direct usages of the original global variables with accesses to the
        corresponding components of the `global_data` argument (e.g., `my_global_var`
        becomes `global_data % my_global_var`).
    6.  Adding the `global_data` argument to all call sites.

    This makes global data dependencies explicit and simplifies dataflow analysis.

    :param ast: The Fortran AST to modify.
    :param always_add_global_data_arg: If True, adds the global data argument even if
                                       no global variables are found.
    :return: The modified Fortran AST.
    """
    alias_map = analysis.alias_specs(ast)
    GLOBAL_DATA_MOD_NAME = 'global_mod'
    if (GLOBAL_DATA_MOD_NAME, ) in alias_map:
        # We already have the global initialisers.
        return ast

    all_derived_types, all_global_vars = [], []
    # Collect all the derived types into a global module.
    for dt in walk(ast, f03.Derived_Type_Def):
        dtspec = analysis.ident_spec(ast_utils.singular(ast_utils.children_of_type(dt, f03.Derived_Type_Stmt)))
        assert len(dtspec) == 2
        mod, dtname = dtspec
        all_derived_types.append(f"use {mod}, only : {dtname}")
    # Collect all the global variables into a single global data structure.
    for m in walk(ast, f03.Module):
        spart = ast_utils.atmost_one(ast_utils.children_of_type(m, f03.Specification_Part))
        if not spart:
            continue
        for tdecl in ast_utils.children_of_type(spart, f03.Type_Declaration_Stmt):
            typ, attr, _ = tdecl.children
            if 'PARAMETER' in f"{attr}":
                # This is a constant which should have been propagated away already.
                continue
            all_global_vars.append(tdecl.tofortran())
    all_derived_types = '\n'.join(all_derived_types)
    all_global_vars = '\n'.join(all_global_vars)

    # Then, replace all the instances of references to global variables with corresponding data-refs.
    for nm in walk(ast, f03.Name):
        par = nm.parent
        if isinstance(par, (f03.Entity_Decl, f03.Use_Stmt, f03.Rename, f03.Only_List)):
            continue
        if isinstance(par, (f03.Part_Ref, f03.Data_Ref)):
            while par and isinstance(par.parent, (f03.Part_Ref, f03.Data_Ref)):
                par = par.parent
            scope_spec = analysis.search_scope_spec(par)
            root, _, _ = analysis._dataref_root(par, scope_spec, alias_map)
            if root is not nm:
                continue
        local_spec = analysis.search_real_local_alias_spec(nm, alias_map)
        if not local_spec:
            continue
        assert local_spec in alias_map
        if not isinstance(alias_map[local_spec], f03.Entity_Decl):
            continue
        edecl_spec = analysis.ident_spec(alias_map[local_spec])
        assert len(edecl_spec) >= 2, \
            f"Fortran cannot possibly have a top-level global variable, outside any module; got {edecl_spec}"
        if len(edecl_spec) != 2:
            # We cannot possibly have a module level variable declaration.
            continue
        mod, var = edecl_spec
        assert (mod, ) in alias_map
        if not isinstance(alias_map[(mod, )], f03.Module_Stmt):
            continue
        if isinstance(nm.parent, f03.Part_Ref):
            _, subsc = nm.parent.children
            utils.replace_node(nm.parent, f03.Data_Ref(f"{GLOBAL_DATA_OBJ_NAME} % {var}({subsc})"))
        else:
            utils.replace_node(nm, f03.Data_Ref(f"{GLOBAL_DATA_OBJ_NAME} % {var}"))

    if all_global_vars or always_add_global_data_arg:
        # Make `global_data` an argument to every defined function.
        for fn in walk(ast, (f03.Function_Subprogram, f03.Subroutine_Subprogram)):
            stmt = ast_utils.singular(ast_utils.children_of_type(fn, utils.NAMED_STMTS_OF_INTEREST_CLASSES))
            assert isinstance(stmt, (f03.Function_Stmt, f03.Subroutine_Stmt))
            prefix, name, dummy_args, whatever = stmt.children
            if dummy_args:
                utils.prepend_children(dummy_args, f03.Name(GLOBAL_DATA_OBJ_NAME))
            else:
                utils.set_children(stmt, (prefix, name, f03.Dummy_Arg_Name_List(GLOBAL_DATA_OBJ_NAME), whatever))
            spart = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Specification_Part))
            use_stmt = f"use {GLOBAL_DATA_MOD_NAME}, only : {GLOBAL_DATA_TYPE_NAME}"
            if spart:
                utils.prepend_children(spart, f03.Use_Stmt(use_stmt))
            else:
                utils.set_children(fn,
                                   fn.children[:1] + [f03.Specification_Part(get_reader(use_stmt))] + fn.children[1:])
            spart = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Specification_Part))
            decl_idx = [idx for idx, v in enumerate(spart.children) if isinstance(v, f03.Type_Declaration_Stmt)]
            decl_idx = decl_idx[0] if decl_idx else len(spart.children)
            utils.set_children(
                spart, spart.children[:decl_idx] +
                [f03.Type_Declaration_Stmt(f"type({GLOBAL_DATA_TYPE_NAME}) :: {GLOBAL_DATA_OBJ_NAME}")] +
                spart.children[decl_idx:])
        for fcall in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
            fn, args = fcall.children
            fnspec = analysis.search_real_local_alias_spec(fn, alias_map)
            if not fnspec:
                continue
            fnstmt = alias_map[fnspec]
            assert isinstance(fnstmt, (f03.Function_Stmt, f03.Subroutine_Stmt))
            if args:
                utils.prepend_children(args, f03.Name(GLOBAL_DATA_OBJ_NAME))
            else:
                utils.set_children(fcall, (fn, f03.Actual_Arg_Spec_List(GLOBAL_DATA_OBJ_NAME)))
        # NOTE: We do not remove the variables themselves, and let them be pruned later on.

    global_mod = f03.Module(
        get_reader(f"""
module {GLOBAL_DATA_MOD_NAME}
  {all_derived_types}

  type {GLOBAL_DATA_TYPE_NAME}
    {all_global_vars}
  end type {GLOBAL_DATA_TYPE_NAME}
end module {GLOBAL_DATA_MOD_NAME}
"""))
    utils.prepend_children(ast, global_mod)

    ast = pruning.consolidate_uses(ast)
    return ast


def create_global_initializers(ast: f03.Program, entry_points: List[types.SPEC]) -> f03.Program:
    """
    Creates initializer subroutines for global variables and derived types.

    This pass identifies global variables and derived type components that have
    initialization values. It then generates subroutines to perform these
    initializations at runtime.

    The process is as follows:
    1.  For each derived type with initialized components, a `type_init_...` subroutine
        is created to initialize the components of an object of that type.
    2.  A single `global_init_fn` subroutine is created to handle the initialization
        of all module-level global variables. This function will call the appropriate
        `type_init_...` subroutines for derived type variables.
    3.  A call to `global_init_fn` is inserted at the beginning of the execution part
        of every subprogram marked as an entry point.
    4.  Any generated initializer functions that are not used are pruned from the AST.

    This ensures that all global state is explicitly initialized before it is used
    in the program's entry points.

    :param ast: The Fortran AST to modify.
    :param entry_points: A list of specifications for subprograms that are entry points.
    :return: The modified Fortran AST.
    """
    # TODO: Ordering of the initializations may matter, but for that we need to find how Fortran's global initialization
    #  works and then reorder the initialization calls appropriately.

    ident_map = analysis.identifier_specs(ast)
    GLOBAL_INIT_FN_NAME = 'global_init_fn'
    if (GLOBAL_INIT_FN_NAME, ) in ident_map:
        # We already have the global initialisers.
        return ast
    alias_map = analysis.alias_specs(ast)

    created_init_fns: Set[str] = set()
    used_init_fns: Set[str] = set()

    def _make_init_fn(fn_name: str, inited_vars: List[types.SPEC], this: Optional[types.SPEC]):
        if this:
            assert this in ident_map and isinstance(ident_map[this], f03.Derived_Type_Stmt)
            box = ident_map[this]
            while not isinstance(box, f03.Specification_Part):
                box = box.parent
            box = box.parent
            assert isinstance(box, f03.Module)
            sp_part = ast_utils.atmost_one(ast_utils.children_of_type(box, f03.Module_Subprogram_Part))
            if not sp_part:
                rest, end_mod = box.children[:-1], box.children[-1]
                assert isinstance(end_mod, f03.End_Module_Stmt)
                # TODO: FParser bug; A simple `Module_Subprogram_Part('contains') should work, but doesn't;
                #  hence the surgery.
                sp_part = f03.Module(get_reader('module m\ncontains\nend module m')).children[1]
                utils.set_children(box, rest + [sp_part, end_mod])
            box = sp_part
        else:
            box = ast

        uses, execs = [], []
        for v in inited_vars:
            var = ident_map[v]
            mod = var
            while not isinstance(mod, f03.Module):
                mod = mod.parent
            if not this:
                uses.append(f"use {utils.find_name_of_node(mod)}, only: {utils.find_name_of_stmt(var)}")
            var_t = analysis.find_type_of_entity(var, alias_map)
            if var_t.spec in type_defs:
                if var_t.shape:
                    # TODO: We need to create loops for this initialization.
                    continue
                var_init, _ = type_defs[var_t.spec]
                tmod = ident_map[var_t.spec]
                while not isinstance(tmod, f03.Module):
                    tmod = tmod.parent
                uses.append(f"use {utils.find_name_of_node(tmod)}, only: {var_init}")
                execs.append(f"call {var_init}({'this % ' if this else ''}{utils.find_name_of_node(var)})")
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
        init_fn = f03.Subroutine_Subprogram(get_reader(init_fn.strip()))
        utils.append_children(box, init_fn)
        created_init_fns.add(fn_name)

    type_defs: List[types.SPEC] = [k for k in ident_map.keys() if isinstance(ident_map[k], f03.Derived_Type_Stmt)]
    type_defs: Dict[types.SPEC, Tuple[str, List[types.SPEC]]] = \
        {k: (f"type_init_{k[-1]}_{idx}", []) for idx, k in enumerate(type_defs)}
    for k, v in ident_map.items():
        if not isinstance(v, f03.Component_Decl) or not ast_utils.atmost_one(
                ast_utils.children_of_type(v, f03.Component_Initialization)):
            continue
        td = k[:-1]
        assert td in ident_map and isinstance(ident_map[td], f03.Derived_Type_Stmt)
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
        if not isinstance(alias_map[(mod, )], f03.Module_Stmt):
            # Not a type that's globally accessible anyway.
            continue
        init_fn_name, comps = v
        _make_init_fn(init_fn_name, comps, t)

    global_inited_vars: List[types.SPEC] = [
        k for k, v in ident_map.items()
        if isinstance(v, f03.Entity_Decl) and not analysis.find_type_of_entity(v, alias_map).const and (
            analysis.find_type_of_entity(v, alias_map).spec in type_defs
            or ast_utils.atmost_one(ast_utils.children_of_type(v, f03.Initialization)))
        and analysis.search_scope_spec(v) and isinstance(alias_map[analysis.search_scope_spec(v)], f03.Module_Stmt)
    ]
    if global_inited_vars:
        _make_init_fn(GLOBAL_INIT_FN_NAME, global_inited_vars, None)
        for ep in entry_points:
            assert ep in ident_map
            fn = ident_map[ep]
            if not isinstance(fn, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                # Not a function (or subroutine), so there is nothing to exectue here.
                continue
            ex = ast_utils.atmost_one(ast_utils.children_of_type(fn.parent, f03.Execution_Part))
            if not ex:
                # The function does nothing. We could still initialize, but there is no point.
                continue
            init_call = f03.Call_Stmt(f"call {GLOBAL_INIT_FN_NAME}")
            utils.prepend_children(ex, init_call)
            used_init_fns.add(GLOBAL_INIT_FN_NAME)

    unused_init_fns = created_init_fns - used_init_fns
    for fn in walk(ast, f03.Subroutine_Subprogram):
        if utils.find_name_of_node(fn) in unused_init_fns:
            utils.remove_self(fn)

    return ast
