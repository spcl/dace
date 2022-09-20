# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import ast

from collections import ChainMap
from ast import AST, NodeVisitor, Name, Store, copy_location
from typing import Callable, Dict, Iterable, List, Tuple

from .ssa_helpers import Counting_UID, EnclosingLoop
from .ssa_nodes import ForCFG, IfCFG, PhiAssign, SingleAssign, UniqueClass, UniqueName, Interface, WhileCFG


class LoopAnalyzer(NodeVisitor):
    """Node visitor that collects used and defined variables"""

    def __init__(self) -> None:

        self.used = set()
        self.defined = set()
        self.has_break = False

    def visit_block(self, block: Iterable[AST]) -> None:

        for stmt in block:
            self.visit(stmt)

    def visit_Name(self, node: Name) -> None:
        
        varname = node.id

        if isinstance(node.ctx, Store):
            self.defined.add(varname)
        else:
            self.used.add(varname)

    def visit_Break(self, node: ast.Break) -> None:

        self.has_break = True


Variables = Dict[str, Tuple[str, AST]]


class SSA_Transpiler(NodeVisitor):
    "Applies SSA conversion algorithm to Python AST"

    undefined_label = "__UNDEFINED_"

    def __init__(self, uid_func: Callable[[str], str] = Counting_UID(), make_interface: bool = False) -> None:

        self.uid_stack = []
        self.make_interface = make_interface
        self.uid_func = uid_func
    
    def get_unique_id(self, name):

        uid = self.uid_func(name)
        self.uid_stack.append(uid)
        return uid

    def visit(self, node: AST, variables: Variables) -> Variables:
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, variables)

    def generic_visit(self, node: AST, variables: Variables) -> Variables:
        """Called if no explicit visitor function exists for a node."""

        new_variables = {}

        for field, value in ast.iter_fields(node):
            
            if isinstance(value, list):

                for item in value:
                    if isinstance(item, AST):
                        defs = self.visit(item, variables)
                        if defs is not None:
                            new_variables.update(defs)
                        
            elif isinstance(value, AST):
                defs = self.visit(value, variables)
                if defs is not None:
                    new_variables.update(defs)
                
        return new_variables

    def visit_Constant(self, node: ast.Constant, variables: Variables) -> Variables:

        return {}

    #################
    # Block Nodes
    #################

    def visit_namespace(self, block: List[AST], variables: Variables, make_interface=True) -> Variables:

        uids_before = len(self.uid_stack)
        new_variables = self.visit_block(block, variables)
        new_uids = self.uid_stack[uids_before:]
        del self.uid_stack[uids_before:]

        # establish interface
        if make_interface:
            interface_node = Interface()
            interface_node.set_variables(new_variables)
            interface_node.set_uids(new_uids)
            copy_location(interface_node, block[-1])
            block.append(interface_node)

        return new_variables

    def visit_block(self, block: List[AST], variables: Variables) -> Variables:

        new_variables = {}

        for stmt in block:
            defs = self.visit(stmt, ChainMap(new_variables, variables))
            new_variables.update(defs)

        return new_variables

    def visit_Module(self, node: ast.Module, variables: Variables) -> Variables:

        body = node.body
        make_interface = self.make_interface
        new_variables = self.visit_namespace(body, variables, make_interface)

        return new_variables

    def visit_IfCFG(self, node: IfCFG, variables: Variables) -> Variables:
        
        test_exp = node.test
        body_block = node.body
        else_block = node.orelse

        test_vars = self.visit(test_exp, variables)
        head_vars = ChainMap(test_vars, variables)

        body_vars = self.visit_block(body_block, head_vars)
        else_vars = self.visit_block(else_block, head_vars)

        statements_after = []
        phi_variables = {}

        twice_redefined = body_vars.keys() & else_vars.keys()
        body_redefined = (body_vars.keys() - twice_redefined) & head_vars.keys()
        else_redefined = (else_vars.keys() - twice_redefined) & head_vars.keys()
        body_created = body_vars.keys() - twice_redefined - body_redefined
        else_created = else_vars.keys() - twice_redefined - else_redefined
        # variables in header which aren't redefined need no phi node

        for var in twice_redefined:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(
                target=new_uid,
                variable_name=var,
                operands=[body_vars[var], else_vars[var]],
                active=False,
            )
            copy_location(phi, node)
            statements_after.append(phi)
            phi_variables[var] = (new_uid, phi)

        for var in body_redefined:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(
                target=new_uid,
                variable_name=var,
                operands=[body_vars[var], head_vars[var]],
                active=False,
            )
            copy_location(phi, node)
            statements_after.append(phi)
            phi_variables[var] = (new_uid, phi)

        for var in else_redefined:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(
                target=new_uid,
                variable_name=var,
                operands=[head_vars[var], else_vars[var]],
                active=False,
            )
            copy_location(phi, node)
            statements_after.append(phi)
            phi_variables[var] = (new_uid, phi)

        for var in body_created:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(
                target=new_uid,
                variable_name=var,
                operands=[body_vars[var], (None, None)],
                active=False,
            )
            copy_location(phi, node)
            statements_after.append(phi)
            phi_variables[var] = (new_uid, phi)

        for var in else_created:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(
                target=new_uid,
                variable_name=var,
                operands=[(None, None), else_vars[var]],
                active=False,
            )
            copy_location(phi, node)
            statements_after.append(phi)
            phi_variables[var] = (new_uid, phi)

        new_variables = test_vars
        new_variables.update(phi_variables)

        node.exit = statements_after

        return new_variables

    def visit_WhileCFG(self, node: WhileCFG, variables: Variables) -> Variables:

        head_block = node.head
        if_cond = node.ifelse.test
        if_body = node.ifelse.body
        if_else = node.ifelse.orelse

        # 1: find variables that are defined in head, body and else blocks
        head_var_finder = LoopAnalyzer()
        head_var_finder.visit_block(head_block)
        body_var_finder = LoopAnalyzer()
        body_var_finder.visit_block(if_body)
        else_var_finder = LoopAnalyzer()
        else_var_finder.visit_block(if_else)

        # 2: place empty entry phi nodes for vars in body
        entry_phis = {}
        entry_phi_vars = body_var_finder.defined  | head_var_finder.defined

        for var in entry_phi_vars:
            new_uid = self.get_unique_id(var)
            operand = variables.get(var, (None, None))
            phi = PhiAssign.create(target=new_uid, variable_name=var, operands=[operand])
            copy_location(phi, node)
            entry_phis[var] = (new_uid, phi)
            node.entry_phi_lookup[var] = phi

        # 4: place empty exit phi nodes for vars in body & else block
        exit_phis = {}
        can_reach_end = else_var_finder.defined | body_var_finder.defined

        for var in can_reach_end:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(target=new_uid, variable_name=var, operands=[])
            copy_location(phi, node)
            exit_phis[var] = (new_uid, phi)
            node.exit_phi_lookup[var] = phi

        with EnclosingLoop(self, node):

            # 5: process header
            # print("header lookup: ", ChainMap(entry_phis, variables))
            header_vars = self.visit_block(head_block, ChainMap(entry_phis, variables))
            self.visit(if_cond, ChainMap(entry_phis, variables))

            # 6: process body & else, fill phi nodes
            #    for each continue -> fill all variables into entry phi nodes
            #    for each break    -> fill all variables into exit phi nodes

            post_header_vars = ChainMap(header_vars, entry_phis, variables)
            body_vars = self.visit_block(if_body, post_header_vars)
            body_contains_break = self.has_break
            else_vars = self.visit_block(if_else, post_header_vars)

            if body_contains_break:
                for var in else_vars:
                    node.exit_phi_lookup[var].add_operand(None, None)

        # 7: Merge new  definitions
        new_variables = {**entry_phis, **header_vars, **exit_phis}

        entry_phi_stmts = [phi for _, phi in entry_phis.values()]
        exit_phi_stmts =  [phi for _, phi in exit_phis.values()]

        node.head = entry_phi_stmts + node.head
        node.exit = node.exit + exit_phi_stmts

        return new_variables

    def visit_ForCFG(self, node: ForCFG, variables: Variables) -> Variables:

        head_block = node.head
        if_cond = node.ifelse.test
        if_body = node.ifelse.body
        if_else = node.ifelse.orelse

        # 1: find variables that are defined in head, body and else blocks
        body_var_finder = LoopAnalyzer()
        body_var_finder.visit_block(if_body)
        else_var_finder = LoopAnalyzer()
        else_var_finder.visit_block(if_else)

        # 2: place empty entry phi nodes for vars in body
        entry_phis = {}

        for var in body_var_finder.defined:
            new_uid = self.get_unique_id(var)
            operand = variables.get(var, (None, None))
            phi = PhiAssign.create(target=new_uid, variable_name=var, operands=[operand])
            copy_location(phi, node)
            entry_phis[var] = (new_uid, phi)
            node.entry_phi_lookup[var] = phi

        # 4: place empty exit phi nodes for vars in body & else block
        exit_phis = {}
        can_reach_end = else_var_finder.defined | body_var_finder.defined

        for var in can_reach_end:
            new_uid = self.get_unique_id(var)
            phi = PhiAssign.create(target=new_uid, variable_name=var, operands=[])
            copy_location(phi, node)
            exit_phis[var] = (new_uid, phi)
            node.exit_phi_lookup[var] = phi

        with EnclosingLoop(self, node):

            # 5: process header
            header_vars = self.visit_block(head_block, ChainMap(entry_phis, variables))
            self.visit(if_cond, ChainMap(entry_phis, variables))

            # 6: process body & else, fill phi nodes
            #    for each continue -> fill all variables into entry phi nodes
            #    for each break    -> fill all variables into exit phi nodes

            post_header_vars = ChainMap(header_vars, entry_phis, variables)
            body_vars = self.visit_block(if_body, post_header_vars)
            body_contains_break = self.has_break
            else_vars = self.visit_block(if_else, post_header_vars)

            if body_contains_break:
                for var in else_vars:
                    node.exit_phi_lookup[var].add_operand(None, None)

        # 7: Merge new  definitions
        new_variables = {**entry_phis, **header_vars, **exit_phis}

        entry_phi_stmts = [phi for _, phi in entry_phis.values()]
        exit_phi_stmts =  [phi for _, phi in exit_phis.values()]

        node.head = entry_phi_stmts + node.head
        node.exit = node.exit + exit_phi_stmts

        return new_variables

    ####################
    # Control Flow Nodes
    ####################

    def visit_Break(self, node: ast.Break, variables: Variables) -> Variables:

        self.has_break = True
        exit_phis = self.current_loop.exit_phi_lookup

        for var, (uid, def_) in variables.items():
            if var in exit_phis:
                exit_phis[var].add_operand(uid, def_)

        return {}

    def visit_Continue(self, node: ast.Continue, variables: Variables) -> Variables:

        entry_phis = self.current_loop.entry_phi_lookup

        for var, (uid, def_) in variables.items():
            if var in entry_phis:
                entry_phis[var].add_operand(uid, def_)

        return {}

    #################
    # SSA Nodes
    #################

    def visit_Name(self, node: Name, variables: Variables) -> Variables:

        var_name = node.id
        res = variables.get(var_name)

        # if res is None:
        #     undef_label = self.undefined_label + var_name
        #     res = (undef_label, None)
        
        if res is None:
            # default to not touching undefined variables
            res = (var_name, None)
        
        uid, definition = res

        if uid is None:
            raise NameError(f"Variable {var_name} was deleted")
        if isinstance(definition, PhiAssign):
            definition.activate()

        node.id = uid
        return {}

    def visit_assignment(self, target: AST, variables: Variables, value: AST) -> Variables:

        new_variables = {}

        # Names
        if isinstance(target, UniqueName):
            self.uid_stack.append(target.id)
        elif isinstance(target, Name):
            variable = target.id
            uid = self.get_unique_id(variable)
            new_variables[variable] = (uid, value)
            vars = self.visit_Name(target, ChainMap(new_variables, variables))
            new_variables.update(vars)
        # Partial access w/o new uid
        elif isinstance(target, ast.Subscript):
            vars = self.generic_visit(target, variables)
            new_variables.update(vars)
        elif isinstance(target, ast.Attribute):
            vars = self.generic_visit(target, variables)
            new_variables.update(vars)
        # Multiple assignments
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                vars = self.visit_assignment(elt, variables, None)
                new_variables.update(vars)
        elif isinstance(target, ast.List):
            for elt in target.elts:
                vars = self.visit_assignment(elt, variables, None)
                new_variables.update(vars)
        elif isinstance(target, ast.Starred):
            vars = self.visit_assignment(target.value, variables, None)
            new_variables.update(vars)
        else:
            raise NotImplementedError(f'Node type {type(target)}')

        return new_variables

    def visit_SingleAssign(self, node: SingleAssign, variables: Variables) -> Variables:

        value = node.value
        target = node.target
        annotation = node.annotation

        new_variables = {}

        if value:
            defs = self.visit(value, variables)
            new_variables.update(defs)

        defs = self.visit_assignment(target, ChainMap(new_variables, variables), value)
        new_variables.update(defs)

        if annotation:
            defs = self.visit(annotation, ChainMap(new_variables, variables))
            new_variables.update(defs)

        return new_variables

    def visit_Assign(self, node: ast.Assign, variables: Variables) -> Variables:

        value = node.value
        targets = node.targets

        new_variables = self.visit(value, variables)

        for target in reversed(targets):

            vars = self.visit_assignment(target, ChainMap(new_variables, variables), value)
            new_variables.update(vars)

        return new_variables

    def visit_AnnAssign(self, node: ast.AnnAssign, variables: Variables) -> Variables:

        value = node.value
        annotation = node.annotation
        target = node.target

        new_variables = self.visit(value, variables)

        if isinstance(target, Name):
            variable = target.id
            uid = self.get_unique_id(variable)
            new_variables[variable] = (uid, value)

        defs = self.visit_assignment(target, ChainMap(new_variables, variables), value)
        new_variables.update(defs)
        self.visit(annotation, ChainMap(new_variables, variables))

        return new_variables

    def visit_deletion(self, target: AST, variables: Variables) -> Variables:

        new_variables = {}

        # Names
        if isinstance(target, UniqueName):
            pass
        elif isinstance(target, Name):
            variable = target.id
            self.visit_Name(target, variables)
            new_variables[variable] = (None, None)
        # Partial access w/o new uid
        elif isinstance(target, ast.Subscript):
            vars = self.generic_visit(target, variables)
            new_variables.update(vars)
        elif isinstance(target, ast.Attribute):
            vars = self.generic_visit(target, variables)
            new_variables.update(vars)
        # Multiple assignments
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                vars = self.visit_deletion(elt, variables)
                new_variables.update(vars)
        elif isinstance(target, ast.List):
            for elt in target.elts:
                vars = self.visit_deletion(elt, variables)
                new_variables.update(vars)
        else:
            raise NotImplementedError(f'Node type {type(target)}')

        return new_variables

    def visit_Delete(self, node: ast.Delete, variables: Variables) -> Variables:

        new_variables = {}

        for target in node.targets:
            vars = self.visit_deletion(target, variables)
            new_variables.update(vars)
        
        return new_variables
        
    def visit_Import(self, node: ast.Import, variables: Variables) -> Variables:

        new_variables = {}

        for alias in node.names:

            import_chain = alias.name.split('.')
            top_module = import_chain[0]
            low_module = import_chain[-1]

            import_name = alias.asname if (alias.asname is not None) else low_module
            uid_name = self.get_unique_id(import_name)

            alias.asname = uid_name
            new_variables[top_module] = top_module, None
            new_variables[uid_name] = uid_name, None
        
        return new_variables

    def visit_ImportFrom(self, node: ast.ImportFrom, variables: Variables) -> Variables:

        new_variables = {}

        for alias in node.names:
            import_chain = alias.asname if (alias.asname is not None) else alias.name
            import_name = import_chain.split('.')[-1]
            uid_name = self.get_unique_id(import_name)
            alias.asname = uid_name
            new_variables[import_chain] = uid_name, None
        
        return new_variables


    #########################
    # Function & Class Nodes
    #########################

    def variables_of_arguments(self, arguments: ast.arguments) -> Variables:

        new_variables = {}

        for arg in arguments.posonlyargs:
            var = arg.arg
            new_variables[var] = var, None

        for arg in arguments.args:
            var = arg.arg
            new_variables[var] = var, None
        
        if arguments.vararg:
            var = arguments.vararg.arg
            new_variables[var] = var, None

        for arg in arguments.kwonlyargs:
            var = arg.arg
            new_variables[var] = var, None

        if arguments.kwarg:
            var = arguments.kwarg.arg
            new_variables[var] = var, None

        return new_variables

    def visit_SimpleFunction(self, node, variables: Variables) -> Variables:
        
        name = node.name
        new_variables = {name: (name, None)}

        # visit all possible arguments
        args = node.args

        for arg in args.posonlyargs:
            if arg.annotation:
                self.visit(arg.annotation, variables)
        
        for arg in args.args:
            if arg.annotation:
                self.visit(arg.annotation, variables)
        
        if args.vararg and args.vararg.annotation:
            self.visit(args.vararg.annotation, variables)
        
        for arg in args.kwonlyargs:
            if arg.annotation:
                self.visit(arg.annotation, variables)
        
        if args.kwarg and args.kwarg.annotation:
            self.visit(args.kwarg.annotation, variables)
        

        if node.returns:
            self.visit(node.returns, variables)

        # visit decorators
        # self.visit_block(node.decorator_list, variables)

        # create locals & visit body
        local_vars = self.variables_of_arguments(node.args)
        self.visit_namespace(node.body, ChainMap(local_vars, new_variables, variables), make_interface=False)

        return {}

    def visit_UniqueClass(self, node: UniqueClass, variables: Variables) -> Variables:

        name = node.name
        new_variables = {name: (name, None)}

        # visit fields
        self.visit_block(node.bases, variables)
        self.visit_block(node.decorator_list, variables)

        self.visit_namespace(node.body, ChainMap(new_variables, variables), make_interface=True)

        return {}

    def visit_With(self, node: ast.With, variables: Variables) -> Variables:

        new_variables = {}

        for withitem in node.items:
            if withitem.optional_vars is not None:
                target = withitem.optional_vars
                value = None
                print("With target: ", target)
                vars = self.visit_assignment(target, variables, value)
                new_variables.update(vars)
        
        self.visit_block(node.body, ChainMap(new_variables, variables))

        return new_variables
