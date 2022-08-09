
import ast
from typing import List

from ast import AST, Constant, NodeVisitor, NodeTransformer, Name, Load, Store, copy_location, iter_fields

from .ssa_nodes import SingleAssign, PhiAssign, UniqueName, SimpleFunction, UniqueClass, IfCFG, ForCFG, WhileCFG, is_simple_expr


def get_additional_statements(node: AST) -> List[AST]:

    add_stmts = []
    is_processed = getattr(node, 'processed', False)

    if not is_processed:
        for field, value in iter_fields(node):

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        stmts = get_additional_statements(item)
                        add_stmts.extend(stmts)

            elif isinstance(value, AST):
                stmts = get_additional_statements(value)
                add_stmts.extend(stmts)

        if hasattr(node, 'statements_before'):
            add_stmts.extend(getattr(node, 'statements_before'))
            delattr(node, 'statements_before')

        node.processed = True
        return add_stmts

    else:
        return []


class GeneratorCheck(NodeVisitor):

    def visit_NamedExpr(self, node):
        raise SyntaxError('Assignment expressions in generator'
                          ' expressions are not supported!')


class SSA_Preprocessor(NodeTransformer):

    counter = 0
    
    ifexp_varname = "ifexp_var"
    assnexp_varname = "assnexp"

    multi_assign_varname = "multi_assign"
    tuple_assign_varname = "tuple_assign"

    while_varname = "while_cond"
    for_iter_varname = "loop_iter"
    for_vals_varname = "loop_value"
    for_next_varname = "has_value"

    lcomp_varname = "list_comp"
    scomp_varname = "set_comp"
    dcomp_varname = "dict_comp"
    gen_varname = "generator_expr"

    lambda_varname = "lambda_expr"
    fstring_varname = "fstr_val"

    func_default_varname = "_default"
    call_varname = "call_var"
    
    nonleaf_stmts = set([
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Return, ast.Delete, ast.Assign,
        ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.While, ast.If, ast.With,
        ast.AsyncWith, ast.Raise, ast.Try, ast.Assert, ast.Expr
    ])

    def unique_id(self, name):

        self.counter += 1
        unique_id = name + "_" + str(self.counter)

        return unique_id
    
    def visit2(self, node):
        node = super().visit(node)
        
        if type(node) in self.nonleaf_stmts:
            stmts_before = []
            for stmt in get_additional_statements(node):
                result = self.visit(stmt)
                if isinstance(result, list):
                    stmts_before.extend(result)
                elif result is not None:
                    stmts_before.append(result)
            return stmts_before + [node]
            
        else:
            return node

    def visit(self, node):

        result = super().visit(node)

        if type(node) not in self.nonleaf_stmts:
            return result

        else:
            return_stmts = []

            if isinstance(result, AST):
                todo = get_additional_statements(result)
                append_result = True

            elif isinstance(result, list):
                todo = result
                append_result = False

            for stmt in todo:
                new_stmts = self.visit(stmt)
                if isinstance(new_stmts, list):
                    return_stmts.extend(new_stmts)
                else:
                    return_stmts.append(new_stmts)

            if append_result:
                return_stmts.append(result)

            return return_stmts

    ###########################
    # Replaced Expression Nodes
    ###########################

    def visit_Global(self, node):
        raise SyntaxError("Global statements are not supported!")

    def visit_Nonlocal(self, node):
        raise SyntaxError("Nonlocal statements are not supported!")

    def visit_IfExp(self, node):

        test = self.visit(node.test)
        body_exp = self.visit(node.body)
        else_exp = self.visit(node.orelse)

        ifexp_var = self.ifexp_varname
        uid_body = self.unique_id(ifexp_var)
        uid_else = self.unique_id(ifexp_var)
        uid_phi_ = self.unique_id(ifexp_var)

        body_stmt = SingleAssign.create(target=UniqueName(id=uid_body, ctx=Store()), value=body_exp)
        else_stmt = SingleAssign.create(target=UniqueName(id=uid_else, ctx=Store()), value=else_exp)

        test_befores = get_additional_statements(test)
        body_befores = get_additional_statements(body_exp)
        else_befores = get_additional_statements(else_exp)

        if_stmt = ast.If(
            test = test,
            body = body_befores + [body_stmt],
            orelse = else_befores + [else_stmt],
        )

        assn_stmt = PhiAssign.create(
            target=uid_phi_,
            operands=[(uid_body, body_exp), (uid_else, else_exp)],
            active=True,
        )

        new_node = UniqueName(id=uid_phi_, ctx=Load())
        copy_location(new_node, node)
        new_node.statements_before = test_befores + [if_stmt, assn_stmt]

        return new_node

    def visit_NamedExpr(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)

        assn_var = self.assnexp_varname
        uid = self.unique_id(assn_var)

        assn_stmt1 = SingleAssign.create(target=UniqueName(id=uid, ctx=Store()), value=value)
        assn_stmt2 = SingleAssign.create(target=target, value=UniqueName(id=uid, ctx=Load()))

        stmts_before1 = get_additional_statements(assn_stmt1)
        stmts_before2 = get_additional_statements(assn_stmt2)

        new_node = UniqueName(id=uid, ctx=Load())
        new_node.statements_before = stmts_before1 + stmts_before2 + [assn_stmt1, assn_stmt2]
        copy_location(new_node, node)
        
        return new_node

    def visit_ListComp(self, node):

        elt = self.visit(node.elt) # generators are handled via For

        comp_var = self.lcomp_varname
        uid = self.unique_id(comp_var)
        new_node = UniqueName(id=uid, ctx=Load())

        assn_stmt = SingleAssign.create(
            target=UniqueName(id=uid, ctx=Store()),
            value=ast.List(elts=[], ctx=Load()),
        )

        element = ast.Expr(ast.Call(
            func=ast.Attribute(value=new_node, attr='append', ctx=Load()),
            args=[elt], keywords=[]
        ))

        for_stmt = self.replace_comprehension(element, node.generators)

        new_node.statements_before = [assn_stmt, for_stmt]

        return new_node

    def visit_SetComp(self, node):

        elt = self.visit(node.elt) # generators are handled via For

        comp_var = self.scomp_varname
        uid = self.unique_id(comp_var)
        new_node = UniqueName(id=uid, ctx=Load())

        assn_stmt = SingleAssign.create(
            target=UniqueName(id=uid, ctx=Store()),
            value=ast.Call(func=Name(id='set', ctx=Load()), args=[], keywords=[]),
        )

        element = ast.Expr(ast.Call(
            func=ast.Attribute(value=new_node, attr='add', ctx=Load()),
            args=[elt], keywords=[]
        ))

        for_stmt = self.replace_comprehension(element, node.generators)

        new_node.statements_before = [assn_stmt, for_stmt]

        return new_node

    def visit_DictComp(self, node):

        key = self.visit(node.key) # generators are handled via For
        val = self.visit(node.value)

        comp_var = self.dcomp_varname
        uid = self.unique_id(comp_var)
        new_node = UniqueName(id=uid, ctx=Load())

        assn_stmt = SingleAssign.create(
            target=UniqueName(id=uid, ctx=Store()),
            value=ast.Dict(keys=[], values=[]),
        )

        element = SingleAssign.create(
            target=ast.Subscript(value=new_node, slice=key),
            value=val,
        )

        for_stmt = self.replace_comprehension(element, node.generators)

        new_node.statements_before = [assn_stmt, for_stmt]

        return new_node

    def visit_GeneratorExp(self, node):

        # ensure there are no NamedExpr nodes
        GeneratorCheck().visit(node)

        elt = self.visit(node.elt)

        generator_var = self.gen_varname
        uid = self.unique_id(generator_var)
        new_node = ast.Call(func=UniqueName(id=uid, ctx=Load()), args=[], keywords=[])
        element = ast.Expr(value=ast.Yield(value=elt))

        for_stmt = self.replace_comprehension(element, node.generators)

        generator_def = SimpleFunction(
            name=uid,
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[for_stmt], 
            decorator_list=[], returns=None, type_comment=None
        )

        new_node.statements_before = [generator_def]

        return new_node

    def replace_comprehension(self, elt, comprehensions):

        assert len(comprehensions) > 0

        new_node = elt

        for comp in reversed(comprehensions):

            for if_exp in comp.ifs:
                new_node = ast.If(
                    test=if_exp,
                    body=[new_node],
                    orelse=[]
                )

            new_node = ast.For(
                target=comp.target, 
                iter=comp.iter, 
                body=[new_node], 
                orelse=[],
            )

        return new_node

    ###########################
    # Replaced Assignment Nodes
    ###########################

    def visit_Assign(self, node):
        # Assign(expr* targets, expr value, string? type_comment)

        targets = node.targets
        value = self.visit(node.value)
        add_stmts = get_additional_statements(value)
        
        if len(targets) == 1:
            target, add_assigns = self.visit_target(targets[0])

            new_node = SingleAssign.create(target=target, value=value)

            if add_assigns:
                return add_stmts + [new_node] + add_assigns
            else:
                return add_stmts + [new_node]

        assign_var = self.multi_assign_varname
        uid_assign = self.unique_id(assign_var)

        new_assigns = [SingleAssign.create(target=UniqueName(id=uid_assign, ctx=Store()), value=value)]
        nested_assigns = []

        for target in targets:
            target, add_assigns = self.visit_target(target)
            new_assigns.append(SingleAssign.create(target=target, value=UniqueName(id=uid_assign, ctx=Load())))
            nested_assigns.extend(add_assigns)

        new_assigns.extend(nested_assigns)

        return add_stmts + new_assigns

    def visit_AnnAssign(self, node):
        # AnnAssign(expr target, expr annotation, expr? value, int simple)

        value = node.value
        value = self.visit(value) if value else value

        annot = self.visit(node.annotation)
        target, add_assigns = self.visit_target(node.target)

        new_node = SingleAssign.create(target=target, value=value, annotation=annot)

        if add_assigns:
            return [new_node] + add_assigns
        else:
            return new_node

    def visit_target(self, target):

        target = self.generic_visit(target)

        # Simple Assignments
        if isinstance(target, (UniqueName, Name, ast.Subscript, ast.Attribute)):
            return target, []

        # Tuple/List Assignments
        elif isinstance(target, (ast.Tuple, ast.List)):

            assert isinstance(target.ctx, Store), f"target.ctx is not ast.Store() for target {ast.unparse(target)}"

            assign_var = self.tuple_assign_varname
            uid_assign = self.unique_id(assign_var)
            new_target = UniqueName(id=uid_assign, ctx=Store())

            pos_stmts = []
            neg_stmts = []
            nested_pos = []
            nested_neg = []
            starred = None

            # Positive index targets (before starred)
            i_pos = -1
            for i_pos, element in enumerate(target.elts):

                if isinstance(element, ast.Starred):
                    starred = element.value
                    break

                element, nested_stmts = self.visit_target(element)
                nested_pos.extend(nested_stmts)
                
                subscript = ast.Subscript(value=UniqueName(id=uid_assign, ctx=Load()), 
                                          slice=ast.Constant(value=i_pos), ctx=Load())
                pos_stmts.append(SingleAssign.create(target=element, value=subscript))

            # Negative index targets (after starred)
            for i, element in enumerate(target.elts[i_pos+1:], 1):
                i_neg = -i

                if isinstance(element, ast.Starred):
                    raise SyntaxError('two starred expressions in assignment (SSA_Preprocessor')

                element, nested_stmts = self.visit_target(element)
                nested_neg.extend(reversed(nested_stmts))
                
                subscript = ast.Subscript(value=UniqueName(id=uid_assign, ctx=Load()), 
                                          slice=ast.Constant(value=i_neg), ctx=Load())
                neg_stmts.append(SingleAssign.create(target=element, value=subscript))

            additional_assigns = []

            # Starred index: all remaining elements
            if starred is not None:

                try:
                    star_slice = ast.Slice(lower=ast.Constant(value=i_pos), upper=ast.Constant(value=i_neg))
                except UnboundLocalError:
                    star_slice = ast.Slice(lower=ast.Constant(value=i_pos))

                element, nested_stmts = self.visit_target(starred)
                nested_pos.extend(nested_stmts)
                
                subscript = ast.Subscript(value=UniqueName(id=uid_assign, ctx=Load()), slice=star_slice, ctx=Load())
                pos_stmts.append(SingleAssign.create(target=element, value=subscript))

                additional_assigns.extend(pos_stmts)
                additional_assigns.extend(reversed(neg_stmts))
                additional_assigns.extend(nested_pos)
                additional_assigns.extend(reversed(nested_neg))

            else:
                # no Starred thus no negative indexing
                additional_assigns.extend(pos_stmts)
                additional_assigns.extend(nested_pos)

            for new_stmt in additional_assigns:
                copy_location(new_stmt, target)

            return new_target, additional_assigns

        # Illegal or unsupported Assignments
        elif isinstance(target, ast.Starred):
            raise SyntaxError('starred assignment target must be in a list or tuple (in SSA_Preprocessor)')
        else:
            raise NotImplementedError(f'Node type {type(target)} as assignment target')


    ###########################
    # Replaced Loop Nodes
    ###########################

    def visit_Try(self, node):

        import warnings
        warnings.warn("Exceptions are disabled! Except clauses are removed.")

        body_block = node.body
        else_block = node.orelse
        final_block = node.finalbody

        return body_block + else_block + final_block

    def visit_If(self, node):
        """turns an ast.If node into a IfCFG node where header, body and orelse fields are all blocks"""

        node = self.generic_visit(node)
        
        test_expr = node.test
        body_block = node.body
        else_block = node.orelse

        new_node = IfCFG(test=test_expr, body=body_block, orelse=else_block)
        new_node.statements_before = get_additional_statements(test_expr)
        copy_location(new_node, node)

        return new_node

    def visit_While(self, node):
        """turns an ast.While node into a WhileCFG node"""

        node = self.generic_visit(node)
        
        test_expr = node.test
        body_block = node.body
        else_block = node.orelse

        before_stmts = get_additional_statements(test_expr)

        head = before_stmts
        if_body = body_block + [ast.Continue()]
        if_else = else_block + [ast.Break()]

        ifelse = ast.If(test=test_expr, body=if_body, orelse=if_else)

        new_node = WhileCFG(head=head, ifelse=ifelse)
        copy_location(new_node, node)

        return new_node

    def visit_For(self, node):
        """turns an ast.For node into a ForCFG node"""
        #| For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)

        node = self.generic_visit(node)
        
        iter = node.iter
        target = node.target
        body_block = node.body
        else_block = node.orelse

        iter_var = self.for_iter_varname
        value_var = self.for_vals_varname
        next_var = self.for_next_varname
        uid_iter = self.unique_id(iter_var)
        uid_value = self.unique_id(value_var)
        uid_next = self.unique_id(next_var)

        before_stmts = get_additional_statements(iter)
        create_iter = SingleAssign.create(target=UniqueName(id=uid_iter, ctx=Store()),
                             value=ast.Call(func=Name(id='for_iter', ctx=Load()),
                                            args=[iter], keywords=[]))
        copy_location(create_iter, node)

        new_target, add_assigns = self.visit_target(target)
        target_assign = SingleAssign.create(target=new_target, value=UniqueName(id=uid_value, ctx=Load()))
        copy_location(target_assign, node)

        setup = before_stmts + [create_iter]
        head = []
        if_body = [target_assign, *add_assigns, *body_block, ast.Continue()]
        if_else = else_block + [ast.Break()]

        test_expr = UniqueName(id=uid_next, ctx=Load())
        ifelse = ast.If(test=test_expr, body=if_body, orelse=if_else)

        for_target = ast.Tuple(elts=[UniqueName(id=uid_value, ctx=Store()), UniqueName(id=uid_next, ctx=Store())], ctx=Store())
        for_iter = UniqueName(id=uid_iter, ctx=Load())

        new_node = ForCFG(target=for_target, iter=for_iter, head=head, ifelse=ifelse)
        new_node.statements_before = setup
        copy_location(new_node, node)

        return new_node

    # Async variants are the same as the normal ones
    visit_AsyncFor = visit_For
    visit_AsyncWhile = visit_While

    #################################
    # Function, Class & Module Nodes
    #################################

    def visit_Lambda(self, node):

        lambda_var = self.lambda_varname
        uid_lambda = self.unique_id(lambda_var)

        return_stmt = ast.Return(value=node.body)
        stmts_before = get_additional_statements(return_stmt)
        new_body = stmts_before + [return_stmt]

        lambda_func = SimpleFunction(
            name=uid_lambda,
            args=node.args,
            body=new_body,
            decorator_list=[], returns=None, type_comment=None
        )

        lambda_func = self.generic_visit(lambda_func)

        new_node = UniqueName(id=uid_lambda, ctx=Load())
        new_node.statements_before = [lambda_func]

        return new_node

    def visit_FunctionDef(self, node):

        func_name = node.name
        node = self.generic_visit(node)
        stmts_before = get_additional_statements(node)

        # remove complex default values
        args = node.args
        args_stmts = []

        new_defaults = []
        for default in args.defaults:
            if is_simple_expr(default):
                new_defaults.append(default)
            else:
                default_var = func_name + self.func_default_varname
                uid_default = self.unique_id(default_var)
                default_assn = SingleAssign.create(target=UniqueName(id=uid_default, ctx=Store()), value=default)
                args_stmts.append(default_assn)
                new_defaults.append(UniqueName(id=uid_default, ctx=Load()))
        args.defaults = new_defaults

        new_kwdefaults = []
        for default in args.kw_defaults:
            if is_simple_expr(default) or default is None:
                new_kwdefaults.append(default)
            else:
                default_var = func_name + self.func_default_varname
                uid_default = self.unique_id(default_var)
                default_assn = SingleAssign.create(target=UniqueName(id=uid_default, ctx=Store()), value=default)
                args_stmts.append(default_assn)
                new_kwdefaults.append(UniqueName(id=uid_default, ctx=Load()))
        args.kw_defaults = new_kwdefaults

        # remove decorators
        decorators = node.decorator_list
        new_value = self.replace_decorators(UniqueName(id=func_name, ctx=Load()), decorators)
        node.decorator_list = []

        new_node = SimpleFunction.from_FunctionDef(node)

        assn_stmt = SingleAssign.create(
            target=Name(id=func_name, ctx=Store()),
            value=new_value,
        )

        del_stmt = ast.Delete(targets=[UniqueName(id=func_name, ctx=ast.Del())])

        return stmts_before + args_stmts + [new_node, assn_stmt, del_stmt]

    def visit_ClassDef(self, node: ast.ClassDef) -> List[ast.AST]:

        class_name = node.name
        node = self.generic_visit(node)
        stmts_before = get_additional_statements(node)

        # remove decorators
        decorators = node.decorator_list
        node.decorator_list = []

        new_node = UniqueClass.from_ClassDef(node)

        new_value = self.replace_decorators(UniqueName(id=class_name, ctx=Load()), decorators)
        
        assn_stmt = SingleAssign.create(
            target=Name(id=class_name, ctx=Store()),
            value=new_value,
        )

        del_stmt = ast.Delete(targets=[UniqueName(id=class_name, ctx=ast.Del())])

        return stmts_before + [new_node, assn_stmt, del_stmt]

    def replace_decorators(self, element, decorators):

        for decorator in reversed(decorators):
            element = ast.Call(func=decorator, args=[element], keywords=[])
        
        return element

    #################################
    # Simplified Nodes
    #################################

    # def visit_Call(self, node: ast.Call) -> ast.Call:

    # TODO: maybe finish this

    #     node = self.generic_visit(node)
    #     stmts_before = get_additional_statements(node)
    #     call_target = node.func

    #     if not is_simple_expr(call_target):
    #         call_var = self.call_varname
    #         uid_call = self.unique_id(call_var)
    #         assn_stmt = SingleAssign.create(target=UniqueName(id=uid_call, ctx=Store()), value=call_target)
    #         new_target = UniqueName(id=uid_call, ctx=Load())
    #         node.func = new_target
    #         node.statements_before = [assn_stmt]
        
    #     return stmts_before + [node]

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:

        node = self.generic_visit(node)

        fval_nodes = [v for v in node.values if isinstance(v, ast.FormattedValue)]

        stmts_before = []
        extracted_vals = []
        fval_var = self.fstring_varname

        for fval_node in fval_nodes:

            add_stmts = get_additional_statements(fval_node)
            stmts_before.extend(add_stmts)

            value = fval_node.value

            if not is_simple_expr(value):
                
                uid_fval = self.unique_id(fval_var)
                assn_stmt = SingleAssign.create(target=UniqueName(id=uid_fval, ctx=Store()), value=value)
                new_value = UniqueName(id=uid_fval, ctx=Load())
                fval_node.value = new_value

                extracted_vals.append(assn_stmt)

        node.statements_before = stmts_before + extracted_vals

        return  node