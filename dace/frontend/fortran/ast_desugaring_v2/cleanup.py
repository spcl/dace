from typing import Union, Set, Dict

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.utils import walk, NumberBase

from . import analysis
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
