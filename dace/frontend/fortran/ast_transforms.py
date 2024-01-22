# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from dace.frontend.fortran import ast_components, ast_internal_classes
from typing import Dict, List, Optional, Tuple, Set
import copy


def iter_fields(node: ast_internal_classes.FNode):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    if not hasattr(node, "_fields"):
        a = 1
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node: ast_internal_classes.FNode):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """

    for name, field in iter_fields(node):
        #print("NASME:",name)
        if isinstance(field, ast_internal_classes.FNode):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast_internal_classes.FNode):
                    yield item


class NodeVisitor(object):
    """
    A base node visitor class for Fortran ASTs.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """
    def visit(self, node: ast_internal_classes.FNode):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast_internal_classes.FNode):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value)


class NodeTransformer(NodeVisitor):
    """
    A base node visitor that walks the abstract syntax tree and allows
    modification of nodes.
    The `NodeTransformer` will walk the AST and use the return value of the
    visitor methods to replace old nodes. 
    """
    def as_list(self, x):
        if isinstance(x, list):
            return x
        if x is None:
            return []
        return [x]

    def generic_visit(self, node: ast_internal_classes.FNode):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_internal_classes.FNode):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_internal_classes.FNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_internal_classes.FNode):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class FindFunctionAndSubroutines(NodeVisitor):
    """
    Finds all function and subroutine names in the AST
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):
        ret=node.name
        ret.elemental=node.elemental
        self.nodes.append(ret)

    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):
        ret=node.name
        ret.elemental=node.elemental
        self.nodes.append(ret)


class FindInputs(NodeVisitor):
    """
    Finds all inputs (reads) in the AST node and its children
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        self.nodes.append(node)

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.nodes.append(node.name)
        for i in node.indices:
            self.visit(i)

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                pass
            elif isinstance(node.lval, ast_internal_classes.Array_Subscript_Node):
                for i in node.lval.indices:
                    self.visit(i)

        else:
            self.visit(node.lval)
        self.visit(node.rval)


class FindOutputs(NodeVisitor):
    """
    Finds all outputs (writes) in the AST node and its children
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, ast_internal_classes.Name_Node):
                self.nodes.append(node.lval)
            elif isinstance(node.lval, ast_internal_classes.Array_Subscript_Node):
                self.nodes.append(node.lval.name)
            elif isinstance(node.lval, ast_internal_classes.Data_Ref_Node):
                if isinstance(node.lval.parent_ref, ast_internal_classes.Name_Node):
                    self.nodes.append(node.lval.parent_ref)    
                elif isinstance(node.lval.parent_ref, ast_internal_classes.Array_Subscript_Node):
                    self.nodes.append(node.lval.parent_ref.name)

            self.visit(node.rval)


class FindFunctionCalls(NodeVisitor):
    """
    Finds all function calls in the AST node and its children
    :return: List of names
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Name_Node] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        self.nodes.append(node)
        for i in node.args:
            self.visit(i)


class StructLister(NodeVisitor):
    """
    Fortran does not differentiate between arrays and functions.
    We need to go over and convert all function calls to arrays.
    So, we create a closure of all math and defined functions and 
    create array expressions for the others.
    """
    def __init__(self):
        
        self.structs = []
        self.names= []

    def visit_Derived_Type_Def_Node(self, node: ast_internal_classes.Derived_Type_Def_Node):
        self.structs.append(node)
        self.names.append(node.name.name)

class StructDependencyLister(NodeVisitor):
    def __init__(self, names=None):
        self.names= names
        self.structs_used = []
        self.is_pointer=[]
        self.pointer_names=[]

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        if node.type in self.names:
            self.structs_used.append(node.type)
            self.is_pointer.append(node.alloc)
            self.pointer_names.append(node.name)


class StructMemberLister(NodeVisitor):
    def __init__(self):
        
        self.members = []
        self.is_pointer=[]
        self.pointer_names=[]

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
            self.members.append(node.type)
            self.is_pointer.append(node.alloc)
            self.pointer_names.append(node.name)


class FindStructDefs(NodeVisitor):
    def __init__(self, name=None):
        self.name= name
        self.structs= []

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):
        if node.type==self.name:
            self.structs.append(node.name)

class FindStructUses(NodeVisitor):
    def __init__(self, names=None,target=None):
        self.names= names
        self.target=target
        self.nodes= []

    def visit_Data_Ref_Node(self, node: ast_internal_classes.Data_Ref_Node):
        
        if isinstance(node.parent_ref, ast_internal_classes.Name_Node):
            parent_name=node.parent_ref.name
        elif isinstance(node.parent_ref, ast_internal_classes.Array_Subscript_Node):
            parent_name=node.parent_ref.name.name
        elif isinstance(node.parent_ref, ast_internal_classes.Data_Ref_Node):
            self.visit(node.parent_ref)
            parent_name=None
        else:

            raise NotImplementedError("Data ref node not implemented for not name or array")
        if isinstance(node.part_ref,ast_internal_classes.Name_Node):
            part_name=node.part_ref.name
        elif isinstance(node.part_ref,ast_internal_classes.Array_Subscript_Node):
            part_name=node.part_ref.name.name
        elif isinstance(node.part_ref, ast_internal_classes.Data_Ref_Node):
            self.visit(node.part_ref)    
            part_name=None
        else:    
            raise NotImplementedError("Data ref node not implemented for not name or array")
        if part_name== self.target and parent_name in self.names:
            self.nodes.append(node)

class StructPointerChecker(NodeVisitor):
    def __init__(self, parent_struct,pointed_struct,pointer_name):
        self.parent_struct=parent_struct
        self.pointed_struct=pointed_struct
        self.pointer_name=pointer_name
        self.nodes=[]

    def visit_Main_Program_Node(self, node: ast_internal_classes.Main_Program_Node):
        finder=FindStructDefs(self.parent_struct)
        finder.visit(node.specification_part)
        struct_names=finder.structs
        use_finder=FindStructUses(struct_names,self.pointer_name)
        use_finder.visit(node.execution_part)
        self.nodes+=use_finder.nodes


    def visit_Subroutine_Subprogram_Node(self, node: ast_internal_classes.Subroutine_Subprogram_Node):        
        finder=FindStructDefs(self.parent_struct)
        if node.specification_part is not None:
            finder.visit(node.specification_part)
        struct_names=finder.structs
        use_finder=FindStructUses(struct_names,self.pointer_name)
        if node.execution_part is not None:
            use_finder.visit(node.execution_part)
        self.nodes+=use_finder.nodes


class StructPointerEliminator(NodeTransformer):
    def __init__(self, parent_struct,pointed_struct,pointer_name):
        self.parent_struct=parent_struct
        self.pointed_struct=pointed_struct
        self.pointer_name=pointer_name

    def visit_Derived_Type_Def_Node(self, node: ast_internal_classes.Derived_Type_Def_Node):
        if node.name.name==self.parent_struct:
            newnode=ast_internal_classes.Derived_Type_Def_Node(name=node.name,parent=node.parent)
            component_part=ast_internal_classes.Component_Part_Node(component_def_stmts=[],parent=node.parent)
            for i in node.component_part.component_def_stmts:
                
                    vardecl=[]
                    for k in i.vars.vardecl:
                        if k.name==self.pointer_name and k.alloc==True and k.type==self.pointed_struct:
                            print("Eliminating pointer "+self.pointer_name+" of type "+ k.type +" in struct "+self.parent_struct)
                            continue
                        else:
                            vardecl.append(k)
                    if vardecl!=[]:        
                        component_part.component_def_stmts.append(ast_internal_classes.Data_Component_Def_Stmt_Node(vars=ast_internal_classes.Decl_Stmt_Node(vardecl=vardecl,parent=node.parent),parent=node.parent))
            newnode.component_part=component_part        
            return newnode
        else:
            return node
class CallToArray(NodeTransformer):
    """
    Fortran does not differentiate between arrays and functions.
    We need to go over and convert all function calls to arrays.
    So, we create a closure of all math and defined functions and 
    create array expressions for the others.
    """
    def __init__(self, funcs=None):
        if funcs is None:
            funcs = []
        self.funcs = funcs

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        self.excepted_funcs = [
            "malloc", "pow", "cbrt", "__dace_sign", "tanh", "atan2",
            "__dace_epsilon", *FortranIntrinsics.function_names()
        ]

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if isinstance(node.name, str):
            return node
        if node.name is None:
            return ast_internal_classes.Char_Literal_Node(value="Error!", type="CHARACTER")
        if node.name.name in self.excepted_funcs or node.name in self.funcs:
            processed_args = []
            for i in node.args:
                arg = CallToArray(self.funcs).visit(i)
                processed_args.append(arg)
            node.args = processed_args
            return node
        indices = [CallToArray(self.funcs).visit(i) for i in node.args]
        return ast_internal_classes.Array_Subscript_Node(name=node.name, indices=indices)


class CallExtractorNodeLister(NodeVisitor):
    """
    Finds all function calls in the AST node and its children that have to be extracted into independent expressions
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Call_Expr_Node] = []

    def visit_For_Stmt_Node(self, node: ast_internal_classes.For_Stmt_Node):
        return

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        stop = False
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                stop = True

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if not stop and node.name.name not in [
                "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            self.nodes.append(node)
        return self.generic_visit(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return



class ArgumentExtractorNodeLister(NodeVisitor):
    """
    Finds all arguments in function calls in the AST node and its children that have to be extracted into independent expressions
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Call_Expr_Node] = []

    def visit_For_Stmt_Node(self, node: ast_internal_classes.For_Stmt_Node):
        return

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        stop = False
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                stop = True

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if not stop and node.name.name not in [
                "malloc", "pow", "cbrt", "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()
        ]:
            for i in node.args:
                if isinstance(i, ast_internal_classes.Name_Node) or isinstance(i, ast_internal_classes.Literal) or isinstance(i, ast_internal_classes.Array_Subscript_Node):
                    continue
                else:
                    self.nodes.append(i)
        return self.generic_visit(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class ArgumentExtractor(NodeTransformer):
    """
    Uses the CallExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """
    def __init__(self, count=0):
        self.count = count

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["malloc", "pow", "cbrt",  "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()]:
            return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                return self.generic_visit(node)
        if not hasattr(self, "count"):
            self.count = 0
        tmp = self.count
        result=ast_internal_classes.Call_Expr_Node(type=node.type,
                                                   name=node.name,
                                                   args=[],
                                                   line_number=node.line_number)
        for i, arg in enumerate(node.args):
            # Ensure we allow to extract function calls from arguments
            if isinstance(arg, ast_internal_classes.Name_Node) or isinstance(arg, ast_internal_classes.Literal) or isinstance(arg, ast_internal_classes.Array_Subscript_Node):
                result.args.append(arg)
            else:
                result.args.append(ast_internal_classes.Name_Node(name="tmp_call_" + str(tmp)))
                tmp = tmp + 1    
        self.count = tmp
        return result

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = ArgumentExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            for i in res:
                if i == child:
                    res.pop(res.index(i))
            if res is not None:
                # Variables are counted from 0...end, starting from main node, to all calls nested
                # in main node arguments.
                # However, we need to define nested ones first.
                # We go in reverse order, counting from end-1 to 0.
                temp = self.count + len(res) - 1
                for i in reversed(range(0, len(res))):

                    newbody.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Var_Decl_Node(
                                name="tmp_call_" + str(temp),
                                type=res[i].type,
                                sizes=None
                            )
                        ]))
                    newbody.append(
                        ast_internal_classes.BinOp_Node(op="=",
                                                        lval=ast_internal_classes.Name_Node(name="tmp_call_" +
                                                                                            str(temp),
                                                                                            type=res[i].type),
                                                        rval=res[i],
                                                        line_number=child.line_number))
                    temp = temp - 1
                    
            
            newbody.append(self.visit(child))
            
        return ast_internal_classes.Execution_Part_Node(execution=newbody)

class FunctionCallTransformer(NodeTransformer):
    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node):
            if hasattr(node.rval, "subroutine"):
                if node.rval.subroutine is True:
                    return self.generic_visit(node)
            if node.rval.name.name.find("__dace_") != -1:
                return self.generic_visit(node)
            if node.op != "=":
                return self.generic_visit(node)
            args = node.rval.args
            lval = node.lval
            args.append(lval)
            return (ast_internal_classes.Call_Expr_Node(type=node.rval.type,
                                                        name=ast_internal_classes.Name_Node(name=node.rval.name.name+"_srt",type=node.rval.type),
                                                        args=args,
                                                        subroutine=True,
                                                        line_number=node.line_number))

        else:
            return self.generic_visit(node)
        
class NameReplacer(NodeTransformer):
    """
    Replaces all occurences of a name with another name
    """
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        if node.name == self.old_name:
            return ast_internal_classes.Name_Node(name=self.new_name,type=node.type)
        else:
            return self.generic_visit(node)        
        
class FunctionToSubroutineDefiner(NodeTransformer):
    """
    Transforms all function definitions into subroutine definitions
    """
    def visit_Function_Subprogram_Node(self, node: ast_internal_classes.Function_Subprogram_Node):
        if node.ret!=None:
            ret=node.ret
        if node.specification_part is not None:    
          for j in node.specification_part.specifications:
            
            for k in j.vardecl:
                if node.ret!=None:
                    if k.name == ret.name:
                        j.vardecl[j.vardecl.index(k)].name=node.name.name+"__ret"
                if k.name == node.name.name:
                    j.vardecl[j.vardecl.index(k)].name=node.name.name+"__ret"
                    
                    break
        execution_part=NameReplacer(node.name.name,node.name.name+"__ret").visit(node.execution_part)
        if node.ret!=None:
            
            execution_part=NameReplacer(ret.name.name,node.name.name+"__ret").visit(node.execution_part)
        args=node.args
        args.append(ast_internal_classes.Name_Node(name=node.name.name+"__ret",type=node.type))
        return ast_internal_classes.Subroutine_Subprogram_Node(name=ast_internal_classes.Name_Node(name=node.name.name+"_srt",type=node.type),
                                                                args=args,
                                                                specification_part=node.specification_part,
                                                                execution_part=execution_part,
                                                                subroutine=True,
                                                                line_number=node.line_number,
                                                                elemental=node.elemental)
    

class CallExtractor(NodeTransformer):
    """
    Uses the CallExtractorNodeLister to find all function calls
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the call with the variable.
    """
    def __init__(self, count=0):
        self.count = count

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["malloc", "pow", "cbrt",  "__dace_epsilon", *FortranIntrinsics.call_extraction_exemptions()]:
            return self.generic_visit(node)
        if hasattr(node, "subroutine"):
            if node.subroutine is True:
                return self.generic_visit(node)
        if not hasattr(self, "count"):
            self.count = 0
        else:
            self.count = self.count + 1
        tmp = self.count

        for i, arg in enumerate(node.args):
            # Ensure we allow to extract function calls from arguments
            node.args[i] = self.visit(arg)

        return ast_internal_classes.Name_Node(name="tmp_call_" + str(tmp - 1))

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = CallExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            for i in res:
                if i == child:
                    res.pop(res.index(i))
            if res is not None:
                # Variables are counted from 0...end, starting from main node, to all calls nested
                # in main node arguments.
                # However, we need to define nested ones first.
                # We go in reverse order, counting from end-1 to 0.
                temp = self.count + len(res) - 1
                for i in reversed(range(0, len(res))):

                    newbody.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Var_Decl_Node(
                                name="tmp_call_" + str(temp),
                                type=res[i].type,
                                sizes=None
                            )
                        ]))
                    newbody.append(
                        ast_internal_classes.BinOp_Node(op="=",
                                                        lval=ast_internal_classes.Name_Node(name="tmp_call_" +
                                                                                            str(temp),
                                                                                            type=res[i].type),
                                                        rval=res[i],
                                                        line_number=child.line_number))
                    temp = temp - 1
            if isinstance(child, ast_internal_classes.Call_Expr_Node):
                new_args = []
                if hasattr(child, "args"):
                    for i in child.args:
                        new_args.append(self.visit(i))
                new_child = ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                name=child.name,
                                                                args=new_args,
                                                                line_number=child.line_number)
                newbody.append(new_child)
            else:
                newbody.append(self.visit(child))

        return ast_internal_classes.Execution_Part_Node(execution=newbody)

class ParentScopeAssigner(NodeVisitor):
    """
        For each node, it assigns its parent scope - program, subroutine, function.

        If the parent node is one of the "parent" types, we assign it as the parent.
        Otherwise, we look for the parent of my parent to cover nested AST nodes within
        a single scope.
    """
    def __init__(self):
        pass

    def visit(self, node: ast_internal_classes.FNode, parent_node: Optional[ast_internal_classes.FNode] = None):

        parent_node_types = [
            ast_internal_classes.Subroutine_Subprogram_Node,
            ast_internal_classes.Function_Subprogram_Node,
            ast_internal_classes.Main_Program_Node,
            ast_internal_classes.Module_Node
        ]

        if parent_node is not None and type(parent_node) in parent_node_types:
            node.parent = parent_node
        elif parent_node is not None:
            node.parent = parent_node.parent

        # Copied from `generic_visit` to recursively parse all leafs
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast_internal_classes.FNode):
                        self.visit(item, node)
            elif isinstance(value, ast_internal_classes.FNode):
                self.visit(value, node)

        return node

class ModuleVarsDeclarations(NodeVisitor):
    """
        Creates a mapping (scope name, variable name) -> variable declaration.

        The visitor is used to access information on variable dimension, sizes, and offsets.
    """

    def __init__(self): #, module_name: str):

        self.scope_vars: Dict[Tuple[str, str], ast_internal_classes.FNode] = {}

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):

        var_name = node.name
        self.scope_vars[var_name] = node

class ScopeVarsDeclarations(NodeVisitor):
    """
        Creates a mapping (scope name, variable name) -> variable declaration.

        The visitor is used to access information on variable dimension, sizes, and offsets.
    """

    def __init__(self):

        self.scope_vars: Dict[Tuple[str, str], ast_internal_classes.FNode] = {}

    def get_var(self, scope: ast_internal_classes.FNode, variable_name: str) -> ast_internal_classes.FNode:
        return self.scope_vars[(self._scope_name(scope), variable_name)]

    def contains_var(self, scope: ast_internal_classes.FNode, variable_name: str) -> bool:
        return (self._scope_name(scope), variable_name) in self.scope_vars

    def visit_Var_Decl_Node(self, node: ast_internal_classes.Var_Decl_Node):

        parent_name = self._scope_name(node.parent)
        var_name = node.name
        self.scope_vars[(parent_name, var_name)] = node

    def _scope_name(self, scope: ast_internal_classes.FNode) -> str:
        if isinstance(scope, ast_internal_classes.Main_Program_Node):
            return scope.name.name.name
        else:
            return scope.name.name

class IndexExtractorNodeLister(NodeVisitor):
    """
    Finds all array subscript expressions in the AST node and its children that have to be extracted into independent expressions
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.Array_Subscript_Node] = []

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["pow", "atan2", "tanh", *FortranIntrinsics.retained_function_names()]:
            return self.generic_visit(node)
        else:
            return

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):
        self.nodes.append(node)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


class IndexExtractor(NodeTransformer):
    """
    Uses the IndexExtractorNodeLister to find all array subscript expressions
    in the AST node and its children that have to be extracted into independent expressions
    It then creates a new temporary variable for each of them and replaces the index expression with the variable.

    Before parsing the AST, the transformation first runs:
    - ParentScopeAssigner to ensure that each node knows its scope assigner.
    - ScopeVarsDeclarations to aggregate all variable declarations for each function.
    """
    def __init__(self, ast: ast_internal_classes.FNode, normalize_offsets: bool = False, count=0):

        self.count = count
        self.normalize_offsets = normalize_offsets

        if normalize_offsets:
            ParentScopeAssigner().visit(ast)
            self.scope_vars = ScopeVarsDeclarations()
            self.scope_vars.visit(ast)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        from dace.frontend.fortran.intrinsics import FortranIntrinsics
        if node.name.name in ["pow", "atan2", "tanh", *FortranIntrinsics.retained_function_names()]:
            return self.generic_visit(node)
        else:
            return node

    def visit_Array_Subscript_Node(self, node: ast_internal_classes.Array_Subscript_Node):

        tmp = self.count
        new_indices = []
        for i in node.indices:
            if isinstance(i, ast_internal_classes.ParDecl_Node):
                new_indices.append(i)
            else:
                new_indices.append(ast_internal_classes.Name_Node(name="tmp_index_" + str(tmp)))
                tmp = tmp + 1
        self.count = tmp
        return ast_internal_classes.Array_Subscript_Node(name=node.name, indices=new_indices)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []

        for child in node.execution:
            lister = IndexExtractorNodeLister()
            lister.visit(child)
            res = lister.nodes
            temp = self.count


            if res is not None:
                for j in res:
                    for idx, i in enumerate(j.indices):
                        if isinstance(i, ast_internal_classes.ParDecl_Node):
                            continue
                        else:
                            tmp_name = "tmp_index_" + str(temp)
                            temp = temp + 1
                            newbody.append(
                                ast_internal_classes.Decl_Stmt_Node(vardecl=[
                                    ast_internal_classes.Var_Decl_Node(name=tmp_name,
                                                                       type="INTEGER",
                                                                       sizes=None,
                                                                       line_number=child.line_number)
                                ],
                                                                    line_number=child.line_number))
                            if self.normalize_offsets:

                                # Find the offset of a variable to which we are assigning
                                var_name = ""
                                if isinstance(j, ast_internal_classes.Name_Node):
                                    var_name = j.name
                                else:
                                    var_name = j.name.name
                                variable = self.scope_vars.get_var(child.parent, var_name)
                                offset = variable.offsets[idx]

                                # it can be a symbol - Name_Node - or a value
                                if not isinstance(offset, ast_internal_classes.Name_Node):
                                    offset = ast_internal_classes.Int_Literal_Node(value=str(offset))

                                newbody.append(
                                    ast_internal_classes.BinOp_Node(
                                        op="=",
                                        lval=ast_internal_classes.Name_Node(name=tmp_name),
                                        rval=ast_internal_classes.BinOp_Node(
                                            op="-",
                                            lval=i,
                                            rval=offset,
                                            line_number=child.line_number),
                                        line_number=child.line_number))
                            else:
                                newbody.append(
                                    ast_internal_classes.BinOp_Node(
                                        op="=",
                                        lval=ast_internal_classes.Name_Node(name=tmp_name),
                                        rval=ast_internal_classes.BinOp_Node(
                                            op="-",
                                            lval=i,
                                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                            line_number=child.line_number),
                                        line_number=child.line_number))
            newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class SignToIf(NodeTransformer):
    """
    Transforms all sign expressions into if statements
    """
    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        if isinstance(node.rval, ast_internal_classes.Call_Expr_Node) and node.rval.name.name == "__dace_sign":
            args = node.rval.args
            lval = node.lval
            cond = ast_internal_classes.BinOp_Node(op=">=",
                                                   rval=ast_internal_classes.Real_Literal_Node(value="0.0"),
                                                   lval=args[1],
                                                   line_number=node.line_number)
            body_if = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.Call_Expr_Node(
                                                    name=ast_internal_classes.Name_Node(name="abs"),
                                                    type="DOUBLE",
                                                    args=[copy.deepcopy(args[0])],
                                                    line_number=node.line_number),
                                                line_number=node.line_number)
            ])
            body_else = ast_internal_classes.Execution_Part_Node(execution=[
                ast_internal_classes.BinOp_Node(lval=copy.deepcopy(lval),
                                                op="=",
                                                rval=ast_internal_classes.UnOp_Node(
                                                    op="-",
                                                    lval=ast_internal_classes.Call_Expr_Node(
                                                        name=ast_internal_classes.Name_Node(name="abs"),
                                                        type="DOUBLE",
                                                        args=[copy.deepcopy(args[0])],
                                                        line_number=node.line_number),
                                                    line_number=node.line_number),
                                                line_number=node.line_number)
            ])
            return (ast_internal_classes.If_Stmt_Node(cond=cond,
                                                      body=body_if,
                                                      body_else=body_else,
                                                      line_number=node.line_number))

        else:
            return self.generic_visit(node)


class RenameArguments(NodeTransformer):
    """
    Renames all arguments of a function to the names of the arguments of the function call
    Used when eliminating function statements
    """
    def __init__(self, node_args: list, call_args: list):
        self.node_args = node_args
        self.call_args = call_args

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        for i, j in zip(self.node_args, self.call_args):
            if node.name == j.name:
                return copy.deepcopy(i)
        return node


class ReplaceFunctionStatement(NodeTransformer):
    """
    Replaces a function statement with its content, similar to propagating a macro
    """
    def __init__(self, statement, replacement):
        self.name = statement.name
        self.content = replacement

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        if node.name == self.name:
            return ast_internal_classes.Parenthesis_Expr_Node(expr=copy.deepcopy(self.content))
        else:
            return self.generic_visit(node)


class ReplaceFunctionStatementPass(NodeTransformer):
    """
    Replaces a function statement with its content, similar to propagating a macro
    """
    def __init__(self, statefunc: list):
        self.funcs = statefunc

    def visit_Structure_Constructor_Node(self, node: ast_internal_classes.Structure_Constructor_Node):
        for i in self.funcs:
            if node.name.name == i[0].name.name:
                ret_node = copy.deepcopy(i[1])
                ret_node = RenameArguments(node.args, i[0].args).visit(ret_node)
                return ast_internal_classes.Parenthesis_Expr_Node(expr=ret_node)
        return self.generic_visit(node)

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):
        for i in self.funcs:
            if node.name.name == i[0].name.name:
                ret_node = copy.deepcopy(i[1])
                ret_node = RenameArguments(node.args, i[0].args).visit(ret_node)
                return ast_internal_classes.Parenthesis_Expr_Node(expr=ret_node)
        return self.generic_visit(node)

def optionalArgsHandleFunction(func):

    func.optional_args = []

    for spec in func.specification_part.specifications:
        for var in spec.vardecl:
            if var.optional:
                func.optional_args.append((var.name, var.type))

    vardecls = []
    new_args = []
    for arg in func.args:

        found = False
        for opt_arg in func.optional_args:
            if opt_arg[0] == arg.name:
                found = True
                break

        if found:

            name = f'__dace_OPTIONAL_{var.name}'
            var = ast_internal_classes.Var_Decl_Node(name=name,
                                            type='BOOL',
                                            alloc=False,
                                            sizes=None,
                                            offsets=None,
                                            kind=None,
                                            optional=False,
                                            line_number=func.line_number)
            new_args.append(ast_internal_classes.Name_Node(name=name))
            vardecls.append(var)

    if len(new_args) > 0:
        func.args.extend(new_args)

    if len(vardecls) > 0:
        func.specification_part.specifications.append(
            ast_internal_classes.Decl_Stmt_Node(
                vardecl=vardecls,
                line_number=func.line_number
            )
        )

    return len(new_args)

class OptionalArgsTransformer(NodeTransformer):
    def __init__(self, funcs_with_opt_args):
        self.funcs_with_opt_args = funcs_with_opt_args

    def visit_Call_Expr_Node(self, node: ast_internal_classes.Call_Expr_Node):

        if node.name.name not in self.funcs_with_opt_args:
            return node

        # Basic assumption for positioanl arguments
        # Optional arguments follow the mandatory ones
        # We use that to determine which optional arguments are missing
        func_decl = self.funcs_with_opt_args[node.name.name]
        optional_args = len(func_decl.optional_args)
        if optional_args == 0:
            return node

        should_be_args = len(func_decl.args)
        mandatory_args = should_be_args - optional_args*2

        present_args = len(node.args)
        # Remove the deduplicated variable entries acting as flags for optional args
        missing_args_count = should_be_args - present_args - optional_args
        present_optional_args = present_args - mandatory_args

        for i in range(missing_args_count):
            relative_position = i + present_optional_args
            dtype = func_decl.optional_args[relative_position][1]
            if dtype == 'INTEGER':
                node.args.append(ast_internal_classes.Int_Literal_Node(value='0'))
            elif dtype == 'BOOL':
                node.args.append(ast_internal_classes.Bool_Literal_Node(value='0'))
            elif dtype == 'DOUBLE':
                node.args.append(ast_internal_classes.Real_Literal_Node(value='0'))
            else:
                raise NotImplementedError()

        # Now pass the 'present' flags
        for i in range(optional_args - missing_args_count):
            node.args.append(ast_internal_classes.Bool_Literal_Node(value='1'))
        for i in range(missing_args_count):
            node.args.append(ast_internal_classes.Bool_Literal_Node(value='0'))

        return node

def optionalArgsExpander(node=ast_internal_classes.Program_Node):
    """
    Adds to each optional arg a logical value specifying its status.
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    :note Should only be used on the program node
    """

    modified_functions = {}

    for func in node.subroutine_definitions:
        if optionalArgsHandleFunction(func):
            modified_functions[func.name.name] = func
    for mod in node.modules:
        for func in mod.subroutine_definitions:
            if optionalArgsHandleFunction(func):
                modified_functions[func.name.name] = func

    node = OptionalArgsTransformer(modified_functions).visit(node)

    return node

def functionStatementEliminator(node=ast_internal_classes.Program_Node):
    """
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    :note Should only be used on the program node
    """
    main_program = localFunctionStatementEliminator(node.main_program)
    function_definitions = [localFunctionStatementEliminator(i) for i in node.function_definitions]
    subroutine_definitions = [localFunctionStatementEliminator(i) for i in node.subroutine_definitions]
    modules = []
    for i in node.modules:
        module_function_definitions = [localFunctionStatementEliminator(j) for j in i.function_definitions]
        module_subroutine_definitions = [localFunctionStatementEliminator(j) for j in i.subroutine_definitions]
        modules.append(
            ast_internal_classes.Module_Node(
                name=i.name,
                specification_part=i.specification_part,
                subroutine_definitions=module_subroutine_definitions,
                function_definitions=module_function_definitions,
            ))
    return ast_internal_classes.Program_Node(main_program=main_program,
                                             function_definitions=function_definitions,
                                             subroutine_definitions=subroutine_definitions,
                                             modules=modules)


def localFunctionStatementEliminator(node: ast_internal_classes.FNode):
    """
    Eliminates function statements from the AST
    :param node: The AST to be transformed
    :return: The transformed AST
    """
    if node is None:
        return None
    if hasattr(node,"specification_part"):
        spec = node.specification_part.specifications
    else:
        spec = []    
    exec = node.execution_part.execution
    new_exec = exec.copy()
    to_change = []
    for i in exec:
        if isinstance(i, ast_internal_classes.BinOp_Node):
            if i.op == "=":
                if isinstance(i.lval, ast_internal_classes.Call_Expr_Node) or isinstance(
                        i.lval, ast_internal_classes.Structure_Constructor_Node):
                    function_statement_name = i.lval.name
                    is_actually_function_statement = False
                    # In Fortran, function statement are defined as scalar values,
                    # but called as arrays, so by identifiying that it is called as
                    # a call_expr or structure_constructor, we also need to match
                    # the specification part and see that it is scalar rather than an array.
                    found = False
                    for j in spec:
                        if found:
                            break
                        for k in j.vardecl:
                            if k.name == function_statement_name.name:
                                if k.sizes is None:
                                    is_actually_function_statement = True
                                    function_statement_type = k.type
                                    j.vardecl.remove(k)
                                    found = True
                                    break
                    if is_actually_function_statement:
                        to_change.append([i.lval, i.rval])
                        new_exec.remove(i)

                    else:
                        #There are no function statements after the first one that isn't a function statement
                        break
    still_changing = True
    while still_changing:
        still_changing = False
        for i in to_change:
            rval = i[1]
            calls = FindFunctionCalls()
            calls.visit(rval)
            for j in to_change:
                for k in calls.nodes:
                    if k.name == j[0].name:
                        calls_to_replace = FindFunctionCalls()
                        calls_to_replace.visit(j[1])
                        #must check if it is recursive and contains other function statements
                        it_is_simple = True
                        for l in calls_to_replace.nodes:
                            for m in to_change:
                                if l.name == m[0].name:
                                    it_is_simple = False
                        if it_is_simple:
                            still_changing = True
                            i[1] = ReplaceFunctionStatement(j[0], j[1]).visit(rval)
    final_exec = []
    for i in new_exec:
        final_exec.append(ReplaceFunctionStatementPass(to_change).visit(i))
    node.execution_part.execution = final_exec
    node.specification_part.specifications = spec
    return node


class ArrayLoopNodeLister(NodeVisitor):
    """
    Finds all array operations that have to be transformed to loops in the AST
    """
    def __init__(self):
        self.nodes: List[ast_internal_classes.FNode] = []
        self.range_nodes: List[ast_internal_classes.FNode] = []

    def visit_BinOp_Node(self, node: ast_internal_classes.BinOp_Node):
        rval_pardecls = [i for i in mywalk(node.rval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        lval_pardecls = [i for i in mywalk(node.lval) if isinstance(i, ast_internal_classes.ParDecl_Node)]
        if len(lval_pardecls) > 0:
            if len(rval_pardecls) == 1:
                self.range_nodes.append(node)
                self.nodes.append(node)
                return
            elif len(rval_pardecls) > 1:
                for i in rval_pardecls[1:]:
                    if i != rval_pardecls[0] and i.type != 'ALL':
                        raise NotImplementedError("Only supporting one range in right expression")

                self.range_nodes.append(node)
                self.nodes.append(node)
                return
            else:
                self.nodes.append(node)
                return

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        return


def par_Decl_Range_Finder(node: ast_internal_classes.Array_Subscript_Node,
                          ranges: list,
                          rangepos: list,
                          rangeslen: list,
                          count: int,
                          newbody: list,
                          scope_vars: ScopeVarsDeclarations,
                          declaration=True):
    """
    Helper function for the transformation of array operations and sums to loops
    :param node: The AST to be transformed
    :param ranges: The ranges of the loop
    :param rangeslength: The length of ranges of the loop
    :param rangepos: The positions of the ranges
    :param count: The current count of the loop
    :param newbody: The new basic block that will contain the loop
    :param declaration: Whether the declaration of the loop variable is needed
    :param is_sum_to_loop: Whether the transformation is for a sum to loop
    :return: Ranges, rangepos, newbody
    """

    currentindex = 0
    indices = []

    offsets = scope_vars.get_var(node.parent, node.name.name).offsets

    for idx, i in enumerate(node.indices):
        if isinstance(i, ast_internal_classes.ParDecl_Node):

            if i.type == "ALL":

                lower_boundary = None
                if offsets[idx] != 1:
                    # support symbols and integer literals
                    if isinstance(offsets[idx], ast_internal_classes.Name_Node):
                        lower_boundary = offsets[idx]
                    else:
                        lower_boundary = ast_internal_classes.Int_Literal_Node(value=str(offsets[idx]))
                else:
                    lower_boundary = ast_internal_classes.Int_Literal_Node(value="1")

                upper_boundary = ast_internal_classes.Name_Range_Node(name="f2dace_MAX",
                                                        type="INTEGER",
                                                        arrname=node.name,
                                                        pos=currentindex)
                """
                    When there's an offset, we add MAX_RANGE + offset.
                    But since the generated loop has `<=` condition, we need to subtract 1.
                """
                if offsets[idx] != 1:

                    # support symbols and integer literals
                    if isinstance(offsets[idx], ast_internal_classes.Name_Node):
                        offset = offsets[idx]
                    else:
                        offset = ast_internal_classes.Int_Literal_Node(value=str(offsets[idx]))

                    upper_boundary = ast_internal_classes.BinOp_Node(
                        lval=upper_boundary,
                        op="+",
                        rval=offset
                    )
                    upper_boundary = ast_internal_classes.BinOp_Node(
                        lval=upper_boundary,
                        op="-",
                        rval=ast_internal_classes.Int_Literal_Node(value="1")
                    )
                ranges.append([lower_boundary, upper_boundary])
                rangeslen.append(-1)

            else:
                ranges.append([i.range[0], i.range[1]])

                start = 0
                if isinstance(i.range[0], ast_internal_classes.Int_Literal_Node):
                    start = int(i.range[0].value)
                else:
                    start = i.range[0]

                end = 0
                if isinstance(i.range[1], ast_internal_classes.Int_Literal_Node):
                    end = int(i.range[1].value)
                else:
                    end = i.range[1]

                if isinstance(end, int) and isinstance(start, int):
                    rangeslen.append(end - start + 1)
                else:
                    add = ast_internal_classes.BinOp_Node(
                        lval=start,
                        op="+",
                        rval=ast_internal_classes.Int_Literal_Node(value="1")
                    )
                    substr = ast_internal_classes.BinOp_Node(
                        lval=end,
                        op="-",
                        rval=add
                    )
                    rangeslen.append(substr)

            rangepos.append(currentindex)
            if declaration:
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(
                            name="tmp_parfor_" + str(count + len(rangepos) - 1), type="INTEGER", sizes=None, init=None)
                    ]))
            indices.append(ast_internal_classes.Name_Node(name="tmp_parfor_" + str(count + len(rangepos) - 1)))
        else:
            indices.append(i)
        currentindex += 1

    node.indices = indices


class ArrayToLoop(NodeTransformer):
    """
    Transforms the AST by removing array expressions and replacing them with loops
    """
    def __init__(self, ast):
        self.count = 0

        ParentScopeAssigner().visit(ast)
        self.scope_vars = ScopeVarsDeclarations()
        self.scope_vars.visit(ast)

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            lister = ArrayLoopNodeLister()
            lister.visit(child)
            res = lister.nodes
            res_range = lister.range_nodes
            if res is not None and len(res) > 0:

                current = child.lval
                val = child.rval
                ranges = []
                rangepos = []
                par_Decl_Range_Finder(current, ranges, rangepos, [], self.count, newbody, self.scope_vars, True)

                if res_range is not None and len(res_range) > 0:
                    rvals = [i for i in mywalk(val) if isinstance(i, ast_internal_classes.Array_Subscript_Node)]
                    for i in rvals:
                        rangeposrval = []
                        rangesrval = []

                        par_Decl_Range_Finder(i, rangesrval, rangeposrval, [], self.count, newbody, self.scope_vars, False)

                        for i, j in zip(ranges, rangesrval):
                            if i != j:
                                if isinstance(i, list) and isinstance(j, list) and len(i) == len(j):
                                    for k, l in zip(i, j):
                                        if k != l:
                                            if isinstance(k, ast_internal_classes.Name_Range_Node) and isinstance(
                                                    l, ast_internal_classes.Name_Range_Node):
                                                if k.name != l.name:
                                                    raise NotImplementedError("Ranges must be the same")
                                            else:
                                                raise NotImplementedError("Ranges must be the same")
                                else:
                                    raise NotImplementedError("Ranges must be identical")

                range_index = 0
                body = ast_internal_classes.BinOp_Node(lval=current, op="=", rval=val, line_number=child.line_number)
                for i in ranges:
                    initrange = i[0]
                    finalrange = i[1]
                    init = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="=",
                        rval=initrange,
                        line_number=child.line_number)
                    cond = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="<=",
                        rval=finalrange,
                        line_number=child.line_number)
                    iter = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                        op="=",
                        rval=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="tmp_parfor_" + str(self.count + range_index)),
                            op="+",
                            rval=ast_internal_classes.Int_Literal_Node(value="1")),
                        line_number=child.line_number)
                    current_for = ast_internal_classes.Map_Stmt_Node(
                        init=init,
                        cond=cond,
                        iter=iter,
                        body=ast_internal_classes.Execution_Part_Node(execution=[body]),
                        line_number=child.line_number)
                    body = current_for
                    range_index += 1

                newbody.append(body)

                self.count = self.count + range_index
            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


def mywalk(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        yield node

class RenameVar(NodeTransformer):
    def __init__(self, oldname: str, newname: str):
        self.oldname = oldname
        self.newname = newname

    def visit_Name_Node(self, node: ast_internal_classes.Name_Node):
        return ast_internal_classes.Name_Node(name=self.newname) if node.name == self.oldname else node


class ForDeclarer(NodeTransformer):
    """
    Ensures that each loop iterator is unique by extracting the actual iterator and assigning it to a uniquely named local variable
    """
    def __init__(self):
        self.count = 0

    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Map_Stmt_Node):
                newbody.append(self.visit(child))
                continue
            if isinstance(child, ast_internal_classes.For_Stmt_Node):
                newbody.append(
                    ast_internal_classes.Decl_Stmt_Node(vardecl=[
                        ast_internal_classes.Symbol_Decl_Node(
                            name="_for_it_" + str(self.count), type="INTEGER", sizes=None, init=None)
                    ]))
                final_assign = ast_internal_classes.BinOp_Node(lval=child.init.lval,
                                                               op="=",
                                                               rval=child.cond.rval,
                                                               line_number=child.line_number)
                newfor = RenameVar(child.init.lval.name, "_for_it_" + str(self.count)).visit(child)
                self.count += 1
                newfor = self.visit(newfor)
                newbody.append(newfor)

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)


class ElementalFunctionExpander(NodeTransformer):
    "Makes elemental functions into normal functions by creating a loop around thme if they are called with arrays"
    def __init__(self, func_list:list ):
        self.func_list = func_list
        self.count=0
    
    def visit_Execution_Part_Node(self, node: ast_internal_classes.Execution_Part_Node):
        newbody = []
        for child in node.execution:
            if isinstance(child, ast_internal_classes.Call_Expr_Node):
                arrays=False
                for i in self.func_list:
                    if child.name.name==i.name or child.name.name==i.name+"_srt":
                        if i.elemental is True:
                            if len(child.args)>0:
                                for j in child.args:
                                    #THIS Needs a proper check
                                    if j.name=="z":
                                        arrays=True
                if not arrays:       
                    newbody.append(self.visit(child))
                else:
                    newbody.append(
                        ast_internal_classes.Decl_Stmt_Node(vardecl=[
                            ast_internal_classes.Symbol_Decl_Node(
                                name="_for_elem_it_" + str(self.count), type="INTEGER", sizes=None, init=None)
                    ]))
                    newargs=[]
                    #The range must be determined! It's currently hard set to 10  
                    shape=["10"]
                    for i in child.args:
                        if isinstance(i,ast_internal_classes.Name_Node):
                            newargs.append(ast_internal_classes.Array_Subscript_Node(name=i,indices=[ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count))],line_number=child.line_number,type=i.type))
                            if i.name.startswith("tmp_call_"):
                                for j in newbody:
                                    if isinstance(j,ast_internal_classes.Decl_Stmt_Node):
                                        if j.vardecl[0].name==i.name:
                                            newbody[newbody.index(j)].vardecl[0].sizes=shape
                                            break
                        else:
                            raise NotImplementedError("Only name nodes are supported")    
                      
                    newbody.append(ast_internal_classes.For_Stmt_Node(
                        init=ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                                            op="=",
                                                            rval=ast_internal_classes.Int_Literal_Node(value="1"),
                                                            line_number=child.line_number),
                        cond=ast_internal_classes.BinOp_Node(lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                                                            op="<=",
                                                            rval=ast_internal_classes.Int_Literal_Node(value=shape[0]),
                                                            line_number=child.line_number),
                        body=ast_internal_classes.Execution_Part_Node(execution=[
                            ast_internal_classes.Call_Expr_Node(type=child.type,
                                                                name=child.name,
                                                                args=newargs,
                                                                line_number=child.line_number)
                        ]), line_number=child.line_number ,
                        iter = ast_internal_classes.BinOp_Node(
                        lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                        op="=",
                        rval=ast_internal_classes.BinOp_Node(
                            lval=ast_internal_classes.Name_Node(name="_for_elem_it_" + str(self.count)),
                            op="+",
                            rval=ast_internal_classes.Int_Literal_Node(value="1")),
                        line_number=child.line_number)                                                                
                    ))
                    self.count += 1
                    

            else:
                newbody.append(self.visit(child))
        return ast_internal_classes.Execution_Part_Node(execution=newbody)
    
