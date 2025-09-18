# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import re
import sys
from typing import Union, Tuple, Dict, Optional, List, Iterable, Set, Generator

import numpy as np
import fparser.two.Fortran2003 as f03
from fparser.two.Fortran2008 import Type_Declaration_Stmt
from fparser.two.utils import Base, walk

from dace.frontend.fortran.ast_desugaring_v2.analysis import (
    identifier_specs, alias_specs, find_type_of_entity, search_scope_spec,
    search_real_local_alias_spec, ident_spec, _const_eval_basic_type, _lookup_dataref,
    find_type_dataref, find_dataref_component_spec, _track_local_consts, find_scope_spec,
    search_real_ident_spec, find_real_ident_spec, search_local_alias_spec, procedure_specs
)
from . import types
from . import utils
from .. import ast_utils


def make_practically_constant_global_vars_constants(ast: f03.Program) -> f03.Program:
    ident_map = identifier_specs(ast)
    alias_map = alias_specs(ast)

    # Start with everything that _could_ be a candidate.
    never_assigned: Set[types.SPEC] = {
        k
        for k, v in ident_map.items() if isinstance(v, f03.Entity_Decl) and not find_type_of_entity(v, alias_map).const
        and search_scope_spec(v) and isinstance(alias_map[search_scope_spec(v)], f03.Module_Stmt)
    }

    for asgn in walk(ast, f03.Assignment_Stmt):
        lv, _, rv = asgn.children
        if not isinstance(lv, f03.Name):
            # Everything else unsupported for now.
            continue
        loc = search_real_local_alias_spec(lv, alias_map)
        assert loc
        var = alias_map[loc]
        assert isinstance(var, (f03.Entity_Decl, f03.Function_Stmt))
        if not isinstance(var, f03.Entity_Decl):
            continue
        var_spec = ident_spec(var)
        if var_spec in never_assigned:
            never_assigned.remove(var_spec)

    for fcall in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        fn, args = fcall.children
        args = args.children if args else tuple()
        for a in args:
            if not isinstance(a, f03.Name):
                # Everything else unsupported for now.
                continue
            loc = search_real_local_alias_spec(a, alias_map)
            assert loc
            var = alias_map[loc]
            assert isinstance(var, f03.Entity_Decl)
            var_spec = ident_spec(var)
            if var_spec in never_assigned:
                never_assigned.remove(var_spec)

    for fixed in never_assigned:
        edcl = alias_map[fixed]
        assert isinstance(edcl, f03.Entity_Decl)
        if not ast_utils.atmost_one(ast_utils.children_of_type(edcl, f03.Initialization)):
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
            utils.replace_node(tdcl, Type_Declaration_Stmt(f"{typ}, {nuattr} :: {edclist}"))
        else:
            utils.replace_node(tdcl, Type_Declaration_Stmt(f"{typ}, {nuattr} :: {edcl}"))
            utils.remove_children(edclist, edcl)
            attr = f", {attr}" if attr else ''
            utils.append_children(tdcl.parent, Type_Declaration_Stmt(f"{typ} {attr} :: {edclist}"))

    return ast


def make_practically_constant_arguments_constants(ast: f03.Program, keepers: List[types.SPEC]) -> f03.Program:
    alias_map = alias_specs(ast)

    # First, build a table to see what possible values a function argument may see.
    fnargs_possible_values: Dict[types.SPEC, Set[Optional[types.NUMPY_TYPES]]] = {}
    fnargs_undecidables: Set[types.SPEC] = set()
    fnargs_optional_presence: Dict[types.SPEC, Set[bool]] = {}
    for fcall in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        fn, args = fcall.children
        if isinstance(fn, f03.Intrinsic_Name):
            # Cannot do anything with intrinsic functions.
            continue
        args = args.children if args else tuple()
        kwargs = tuple(a.children for a in args if isinstance(a, f03.Actual_Arg_Spec))
        kwargs = {k.string:v for k, v in kwargs}
        fnspec = search_real_local_alias_spec(fn, alias_map)
        assert fnspec, fn
        fnstmt = alias_map[fnspec]
        fnspec = ident_spec(fnstmt)
        if fnspec in keepers:
            # The "entry-point" functions arguments are fair game for external usage.
            continue
        fnargs = ast_utils.atmost_one(ast_utils.children_of_type(fnstmt, (f03.Dummy_Arg_List, f03.Dummy_Arg_Name_List)))
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
                # The absense should be noted even if it is a writable argument.
                if aspec not in fnargs_optional_presence:
                    fnargs_optional_presence[aspec] = set()
                fnargs_optional_presence[aspec].add(False)
                continue
            kwargs_zone = isinstance(args[0], f03.Actual_Arg_Spec)  # Whether we are in keyword args territory.
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
            if isinstance(v, types.LITERAL_CLASSES):
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
        fn = utils.find_named_ancestor(arg).parent
        assert isinstance(fn, (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        fexec = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Execution_Part))
        if not fexec:
            continue

        for pcall in walk(fexec, f03.Intrinsic_Function_Reference):
            fn, cargs = pcall.children
            cargs = cargs.children if cargs else tuple()
            if fn.string != 'PRESENT':
                continue
            assert len(cargs) == 1
            optvar = cargs[0]
            if utils.find_name_of_node(arg) != optvar.string:
                continue
            utils.replace_node(pcall, types.numpy_type_to_literal(np.bool_(presence)))

    for aspec, vals in fnargs_possible_values.items():
        if (aspec in fnargs_undecidables or len(vals) > 1
                or (aspec in fnargs_optional_presence and False in fnargs_optional_presence[aspec])):
            # There are multiple possiblities for the argument: either some undecidables or multiple literals.
            continue
        fixed_val, = vals
        arg = alias_map[aspec]
        atype = find_type_of_entity(arg, alias_map)
        fn = utils.find_named_ancestor(arg).parent
        assert isinstance(fn, (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        fexec = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Execution_Part))
        if not fexec:
            continue

        if fixed_val is not None:
            for nm in walk(fexec, f03.Name):
                nmspec = search_real_local_alias_spec(nm, alias_map)
                if nmspec != aspec:
                    continue
                utils.replace_node(nm, types.numpy_type_to_literal(fixed_val))
        # TODO: We could also try removing the argument entirely from the function definition, but that's more work with
        #  little benefit, so maybe another time.

    return ast


def exploit_locally_constant_variables(ast: f03.Program) -> f03.Program:
    alias_map = alias_specs(ast)

    for expart in walk(ast, f03.Execution_Part):
        _track_local_consts(expart, alias_map)

    return ast


def const_eval_nodes(ast: f03.Program) -> f03.Program:
    EXPRESSION_CLASSES = (types.LITERAL_CLASSES, f03.Expr, f03.Equiv_Operand, f03.Add_Operand, f03.Or_Operand,
                          f03.Mult_Operand, f03.Level_2_Expr, f03.Level_3_Expr, f03.Level_4_Expr, f03.Level_5_Expr,
                          f03.Intrinsic_Function_Reference)

    alias_map = alias_specs(ast)

    def _const_eval_node(n: Base) -> bool:
        val = _const_eval_basic_type(n, alias_map)
        if val is None:
            return False
        assert not np.isnan(val)
        val = types.numpy_type_to_literal(val)
        if val.tostr().startswith('-'):
            val = f03.Parenthesis(f'({val})')
        utils.replace_node(n, val)
        return True

    for asgn in reversed(walk(ast, f03.Assignment_Stmt)):
        lv, op, rv = asgn.children
        assert op == '='
        _const_eval_node(rv)
    for rngl in reversed(walk(ast, f03.Case_Value_Range_List)):
        rng = len(rngl.children)
        assert rng <= 2
        # We currently do not yet support two children in Case_Value_Range, 
        # but this implementation should be future-proof.
        for v in rngl.children:
            if _const_eval_node(v):
                continue
            for nm in reversed(walk(rngl, f03.Name)):
                _const_eval_node(nm)
    for expr in reversed(walk(ast, EXPRESSION_CLASSES)):
        # Try to const-eval the expression.
        if _const_eval_node(expr):
            # If the node is successfully replaced, then nothing else to do.
            continue
        # Otherwise, try to at least replace the names with the literal values.
        for nm in reversed(walk(expr, f03.Name)):
            _const_eval_node(nm)
    for knode in reversed(walk(ast, f03.Kind_Selector)):
        _, kind, _ = knode.children
        _const_eval_node(kind)

    NON_EXPRESSION_CLASSES = (f03.Explicit_Shape_Spec, f03.Loop_Control, f03.Call_Stmt, f03.Function_Reference,
                              f03.Initialization, f03.Component_Initialization, f03.Section_Subscript_List,
                              f03.Write_Stmt, f03.Allocate_Stmt)
    for node in reversed(walk(ast, NON_EXPRESSION_CLASSES)):
        for nm in reversed(walk(node, f03.Name)):
            _const_eval_node(nm)

    return ast


def _val_2_lit(val: str, type_spec: types.SPEC) -> types.LITERAL_TYPES:
    val = str(val).lower()
    if type_spec == ('INTEGER1', ):
        val = np.int8(val)
    elif type_spec == ('INTEGER2', ):
        val = np.int16(val)
    elif type_spec == ('INTEGER4', ):
        val = np.int32(val)
    elif type_spec == ('INTEGER8', ):
        val = np.int64(val)
    elif type_spec == ('REAL4', ):
        val = np.float32(val)
    elif type_spec == ('REAL8', ):
        val = np.float64(val)
    elif type_spec == ('LOGICAL', ):
        assert val in {'true', 'false', '0', '1'}
        val = np.bool_(val in {'true', '1'})
    else:
        raise NotImplementedError(f"{val} cannot be parsed as the target literal type: {type_spec}")
    return types.numpy_type_to_literal(val)


def _item_comp_matches_actual_comp(item_comp: str, actual_comp: str) -> bool:
    if actual_comp == item_comp:
        # Exactly matched the leaf.
        return True
    if f"{actual_comp}_a" == item_comp:
        # Matched the allocatable array's special variable.
        return True
    dims: re.Match = re.match(r'^__f2dace_SO?A_([a-zA-Z0-9_]+)_d_[0-9]+_s$', item_comp)
    if dims and dims.group(1) == actual_comp:
        # Matched the general array's special variable.
        return True
    return False


def _type_injection_applies_to_instance(item: types.ConstTypeInjection, defn_spec: types.SPEC, comp_spec: types.SPEC,
                                        alias_map: types.SPEC_TABLE) -> bool:
    # ASSUMPTION: `item.scope_spec` must have been taken care of already.

    if not comp_spec:
        # Type injection always requires a component.
        return False
    if item.type_spec not in alias_map or not isinstance(alias_map[item.type_spec], f03.Derived_Type_Stmt):
        # `item` does not really describe a type injection; potentially a bug.
        return False

    inst_typ = find_type_of_entity(alias_map[defn_spec], alias_map)
    # We need to traverse the components until the remaining components have the exact same length as the `item`'s
    # (i.e., exactly 1 for now).
    while len(comp_spec) > 1:
        if inst_typ.spec not in alias_map:
            return False
        tdef = alias_map[inst_typ.spec].parent
        if not isinstance(tdef, f03.Derived_Type_Def):
            return False
        comp: Optional[f03.Component_Decl] = ast_utils.atmost_one(
            c for c in walk(tdef, f03.Component_Decl) if utils.find_name_of_node(c) == comp_spec[0])
        if not comp:
            # Either not a valid component (possibly a bug), or could not proceed with the traversal.
            return False
        inst_typ = find_type_of_entity(comp, alias_map)
        comp_spec = comp_spec[1:]

    if inst_typ.spec != item.type_spec:
        # `item` does not apply to this type.
        return False
    if comp_spec[:-1] != item.component_spec[:-1]:
        # For what's left, everything until the leaf must exactly match.
        return False
    comp: str = comp_spec[-1]
    item_comp = item.component_spec[-1]
    return _item_comp_matches_actual_comp(item_comp, comp)


def _instance_injection_applies_to_instance(item: types.ConstInstanceInjection, defn_spec: types.SPEC,
                                            comp_spec: types.SPEC) -> bool:
    # ASSUMPTION: `item.scope_spec` must have been taken care of already.

    if len(defn_spec) != len(item.root_spec):
        # `local_spec` surely does not match the instance described by `item`: different length.
        return False
    if defn_spec[:-1] != item.root_spec[:-1]:
        # The last part may encode some extra information (to be checked later), but the remaining parts must match.
        return False
    if len(comp_spec) != len(item.component_spec):
        # `comp_spec` surely does not match the instance described by `item`: different length.
        return False
    if comp_spec[:-1] != item.component_spec[:-1]:
        # The last part may encode some extra information (to be checked later), but the remaining parts must match.
        return False

    if not comp_spec:
        comp, item_comp = defn_spec[-1], item.root_spec[-1]
    else:
        if defn_spec[-1] != item.root_spec[-1]:
            # Since we have components to look at, the last parts must match too in this case.
            return False
        comp, item_comp = comp_spec[-1], item.component_spec[-1]

    return _item_comp_matches_actual_comp(item_comp, comp)


def _find_items_applicable_to_instance(
        items: Iterable[types.ConstInjection],
        inst_ref: Union[f03.Name, f03.Part_Ref, f03.Data_Ref, f03.Entity_Decl],
        alias_map: types.SPEC_TABLE) -> Generator[types.ConstInjection, None, None]:
    # Find out if `inst_ref` can match any item at all.
    if isinstance(inst_ref, f03.Entity_Decl):
        defn_spec, comp_spec, local_spec = ident_spec(inst_ref), tuple(), None
    else:
        root, rest = _lookup_dataref(inst_ref, alias_map) or (None, None)
        if not root:
            return None

        # Find out if `inst_ref`'s root refers to a valid variable.
        local_spec = search_real_local_alias_spec(root, alias_map)
        if local_spec not in alias_map or not isinstance(alias_map[local_spec], f03.Entity_Decl):
            # `local_spec` does not really describe a target instance.
            return None

        # Now get the spec of the root variable as it is defined.
        defn_spec = ident_spec(alias_map[local_spec])
        comp_spec = tuple(f"{p}" for p in rest)
        local_spec = search_local_alias_spec(root)

    for it in items:
        if it.scope_spec and it.scope_spec != local_spec[:len(it.scope_spec)]:
            # If `item` is restricted to a scope, then local spec of the instance must start with that.
            continue
        if (isinstance(it, types.ConstTypeInjection)
                and _type_injection_applies_to_instance(it, defn_spec, comp_spec, alias_map)):
            yield it
        elif (isinstance(it, types.ConstInstanceInjection)
              and _instance_injection_applies_to_instance(it, defn_spec, comp_spec)):
            yield it


def _type_injection_applies_to_component(item: types.ConstTypeInjection, defn_spec: types.SPEC, comp: str) -> bool:
    assert len(item.component_spec) == 1, \
        (f"Unimplemented: type injection must have just one-level of component for now; "
         f"got {item.component_spec} to match against {comp}")
    item_comp = item.component_spec[-1]

    if item.type_spec != defn_spec:
        # `item` is not applicable on this type.
        return False

    return _item_comp_matches_actual_comp(item_comp, comp)


def _find_items_applicable_to_component(
        items: Iterable[types.ConstInjection],
        comp_ref: f03.Component_Decl) -> Generator[types.ConstTypeInjection, None, None]:
    # Find out if `inst_ref` can match any item at all.
    tstmt = utils.find_named_ancestor(comp_ref)
    assert isinstance(tstmt, f03.Derived_Type_Stmt)
    defn_spec = ident_spec(tstmt)
    comp = utils.find_name_of_node(comp_ref)
    assert comp

    for it in items:
        # Find out if `item` is even allowed to apply in this scope.
        if it.scope_spec:
            if it.scope_spec != defn_spec[:len(it.scope_spec)]:
                # If `item` is restricted to a scope, then local spec of the instance must start with that.
                continue
        if isinstance(it, types.ConstTypeInjection) and _type_injection_applies_to_component(it, defn_spec, comp):
            yield it


def inject_const_evals(ast: f03.Program, inject_consts: Optional[List[types.ConstInjection]] = None) -> f03.Program:
    inject_consts = inject_consts or []
    alias_map = alias_specs(ast)

    TOPLEVEL_SPEC = ('*', )

    items_by_scopes = {}
    for item in inject_consts:
        scope_spec = item.scope_spec or TOPLEVEL_SPEC
        if scope_spec not in items_by_scopes:
            items_by_scopes[scope_spec] = []
        items_by_scopes[scope_spec].append(item)

        # Validations.
        if item.scope_spec:
            if item.scope_spec not in alias_map:
                print(f"{item}/{item.scope_spec} does not refer to a valid object; moving on...", file=sys.stderr)
                continue
        if isinstance(item, types.ConstTypeInjection):
            if item.type_spec not in alias_map or not isinstance(alias_map[item.type_spec].parent,
                                                                 f03.Derived_Type_Def):
                print(f"{item}/{item.type_spec} does not refer to a valid type; moving on...", file=sys.stderr)
                continue
        elif isinstance(item, types.ConstInstanceInjection):
            root_spec = item.root_spec
            if not item.component_spec and root_spec[-1].endswith('_a'):
                root_spec = root_spec[:-1] + tuple(root_spec[-1].rsplit('_', maxsplit=2)[:1])
            elif not item.component_spec and root_spec[-1].endswith('_s'):
                root_spec = root_spec[:-1] + tuple(root_spec[-1].rsplit('_', maxsplit=3)[:1])
            if root_spec not in alias_map or not isinstance(alias_map[root_spec], f03.Entity_Decl):
                print(f"{item}/{root_spec} does not refer to a valid object; moving on...", file=sys.stderr)
                continue

    for scope_spec, items in items_by_scopes.items():
        if scope_spec == TOPLEVEL_SPEC:
            scope = ast
        else:
            scope = alias_map[scope_spec].parent

        drefs: List[f03.Data_Ref] = [
            dr for dr in walk(scope, f03.Data_Ref)
            if find_type_dataref(dr, find_scope_spec(dr), alias_map).spec != ('CHARACTER', )
        ]
        names: List[f03.Name] = walk(scope, f03.Name)
        allocateds: List[f03.Intrinsic_Function_Reference] = [
            c for c in walk(scope, f03.Intrinsic_Function_Reference) if c.children[0].string == 'ALLOCATED'
        ]
        allocatables: List[Union[f03.Entity_Decl, f03.Component_Decl]] = [
            c for c in walk(scope, (f03.Entity_Decl, f03.Component_Decl)) if find_type_of_entity(c, alias_map).alloc
        ]

        # Ignore the special variables related to array dimensions, since we don't handle them here.
        alloc_items = list(
            filter(
                lambda it: it.component_spec[-1].endswith('_a')
                if it.component_spec else it.root_spec[-1].endswith('_a'), items))
        size_items = list(
            filter(
                lambda it: it.component_spec[-1].endswith('_s')
                if it.component_spec else it.root_spec[-1].endswith('_s'), items))
        items = [it for it in items if it not in alloc_items and it not in size_items]

        for al in allocateds:
            _, args = al.children
            assert args and len(args.children) == 1
            arr, = args.children
            item = ast_utils.atmost_one(_find_items_applicable_to_instance(alloc_items, arr, alias_map))
            if not item:
                continue
            utils.replace_node(al, _val_2_lit(item.value, ('LOGICAL', )))

        for al in allocatables:
            name = utils.find_name_of_node(al)
            typ = find_type_of_entity(al, alias_map)
            assert typ.alloc
            shape = list(typ.shape)
            if isinstance(al, f03.Component_Decl):
                siz_or_off: List[types.ConstInjection] = list(_find_items_applicable_to_component(size_items, al))
            else:
                siz_or_off: List[types.ConstInjection] = list(
                    _find_items_applicable_to_instance(size_items, al, alias_map))
            if not siz_or_off:
                continue

            def _key_(z: types.ConstInjection) -> Optional[str]:
                if z.component_spec:
                    return z.component_spec[-1]
                if isinstance(z, types.ConstInstanceInjection):
                    return z.root_spec[-1]
                return None

            for idx in range(len(shape)):
                if shape[idx] != ':':
                    # It's already fixed anyway.
                    continue
                siz = ast_utils.atmost_one(z for z in siz_or_off if _key_(z) == f"__f2dace_SA_{name}_d_{idx}_s")
                off = ast_utils.atmost_one(z for z in siz_or_off if _key_(z) == f"__f2dace_SOA_{name}_d_{idx}_s")
                if not siz or not off:
                    # We cannot inject and fix the size.
                    continue
                siz, off = int(siz.value), int(off.value)
                shape[idx] = f"{off}:{siz + off - 1}"
            if typ.shape == tuple(shape):
                # Nothing changed, therefore, nothing to do.
                continue
            if ':' in shape:
                # The shape is not fully determined, so don't replace it
                continue

            # Time to replace.
            typ.shape = tuple(shape)
            if not any(s == ':' for s in typ.shape):
                typ.alloc = False
            nudecl = typ.to_decl(name)
            if isinstance(al, f03.Component_Decl):
                clist = al.parent
                decl = clist.parent
                cpart = decl.parent
                nudecl = f03.Data_Component_Def_Stmt(nudecl)
                if len(clist.children) == 1:
                    # Just a single element, so we replace the whole thing.
                    utils.replace_node(decl, nudecl)
                else:
                    # Otherwise, remove `al` and append it later.
                    utils.remove_self(al)
                    utils.append_children(cpart, nudecl)
            elif isinstance(al, f03.Entity_Decl):
                elist = al.parent
                decl = elist.parent
                spart = decl.parent
                nudecl = Type_Declaration_Stmt(nudecl)
                if len(elist.children) == 1:
                    # Just a single element, so we replace the whole thing.
                    utils.replace_node(decl, nudecl)
                else:
                    # Otherwise, remove `al` and append it later.
                    utils.remove_self(al)
                    utils.append_children(spart, nudecl)

        for dr in drefs:
            if isinstance(dr.parent, f03.Assignment_Stmt):
                # We cannot replace on the LHS of an assignment.
                lv, _, _ = dr.parent.children
                if lv == dr:
                    continue
            item = ast_utils.atmost_one(_find_items_applicable_to_instance(items, dr, alias_map))
            if not item:
                continue
            utils.replace_node(dr, _val_2_lit(item.value, find_type_dataref(dr, find_scope_spec(dr), alias_map).spec))

        for nm in names:
            # We can also directly inject variables' values with `ConstInstanceInjection`.
            if isinstance(nm.parent, (f03.Entity_Decl, f03.Only_List, f03.Dummy_Arg_List, f03.Dummy_Arg_Name_List)):
                # We don't want to replace the values in their declarations or imports, but only where their
                # values are being used.
                continue
            loc = search_real_local_alias_spec(nm, alias_map)
            if not loc or not isinstance(alias_map[loc], f03.Entity_Decl):
                continue
            item = ast_utils.atmost_one(_find_items_applicable_to_instance(items, nm, alias_map))
            if not item:
                # Found no direct-value item that applies to `nm`.
                continue
            tspec = find_type_of_entity(alias_map[loc], alias_map)
            # NOTE: We should replace only when it is not an output of the function. However, here we pass the
            # responsibilty to the user to provide valid injections.
            if isinstance(nm.parent, f03.Assignment_Stmt) and nm is nm.parent.children[0]:
                # We're violating the rule of valid injection already: If we are assigning anything to this variable, we
                # can just ignore it, since it has to be treated as a constant anyway.
                print(
                    f"`{nm} = {item.value}` is supposed to be a constant injection, yet found `{nm.parent}` ; "
                    f"dropping the assignment and moving on...",
                    file=sys.stderr)
                utils.remove_self(nm.parent)
                continue
            utils.replace_node(nm, _val_2_lit(item.value, tspec.spec))
    return ast
