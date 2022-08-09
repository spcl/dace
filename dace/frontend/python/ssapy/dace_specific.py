
import ast
from ast import AST, NodeVisitor, NodeTransformer, Name
from typing import Dict, List

from .ssa_nodes import PhiAssign, SingleAssign, UniqueName
from .ssa_postprocess import SSA_Finisher


PHI_CREATOR_CALL = '__create_ref'



class DaCe_Finisher(NodeTransformer):

    def visit_WhileCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_While()

    def visit_SingleAssign(self, node):

        return node.to_assignment()
    
    def visit_Delete(self, node):

        return None


PhiAssigns = Dict[str, List[str]]


class PhiAssignError(Exception):
    ...


class PhiCollector(NodeVisitor):

    # Visitor Function Calls
    #############################

    def generic_visit(self, node: ast.AST) -> PhiAssigns:

        new_phis = {}

        for field, value in ast.iter_fields(node):
            
            if isinstance(value, list):

                for item in value:
                    if isinstance(item, AST):
                        defs = self.visit(item)
                        if defs is not None:
                            new_phis.update(defs)
                        
            elif isinstance(value, AST):
                defs = self.visit(value)
                if defs is not None:
                    new_phis.update(defs)
                
        return new_phis
    
    # Phi Nodes
    #############################

    def visit_PhiAssign(self, node: PhiAssign) -> PhiAssigns:

        target = node.target
        ops = node.operand_names

        if node.has_undefined:
            # Nicer error =)
            raise PhiAssignError(f"Phi node for target {target} contains None!"
                                 "(Path with no assignment to variable)")
        else:
            return {target: ops}

    # Namespace Nodes
    #############################

    def visit_Module(self, node: ast.Module) -> None:
        return self.visit_body_of(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return self.visit_body_of(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return self.visit_body_of(node)
        
    def visit_body_of(self, node: AST) -> None:

        new_phis = self.generic_visit(node)

        # 1) Create additional assignments to references
        replacements = {}

        for phi_target, phi_operands in new_phis.items():
            for op in phi_operands:
                phi_targets = replacements.get(op, [])
                phi_targets.append(phi_target)
                replacements[op] = phi_targets
        
        PhiImplementor(replacements=replacements).visit(node)

        # 2) Add references assigns for variables not defined in this scope
        ref_assigns = []
        for target_id, phi_assigns in replacements.items():
            for phi_target in phi_assigns:
                arg = Name(id=target_id, ctx=ast.Load())
                ref_call = ast.Call(func=Name(id=PHI_CREATOR_CALL, ctx=ast.Load()), args=[arg], keywords=[])
                ref_target = Name(id=phi_target, ctx=ast.Store())
                ref_assign = SingleAssign.create(target=ref_target, value=ref_call)
                ref_assigns.append(ref_assign)

        node.body[:] = ref_assigns + node.body

class PhiImplementor(NodeTransformer):

    def __init__(self, replacements: Dict[str, str]) -> None:

        self.replacements = replacements
    
    # Assignment Nodes
    #############################

    def visit_SingleAssign(self, node: SingleAssign) -> List[SingleAssign]:

        target = node.target

        if not isinstance(target, Name):
            return [node]
        
        ref_assigns = self.get_ref_assigns(target.id)

        return [node] + ref_assigns
    
    def visit_PhiAssign(self, node: SingleAssign) -> List[SingleAssign]:

        target: str = node.target
        ref_assigns = self.get_ref_assigns(target)

        return [node] + ref_assigns
    
    def get_ref_assigns(self, target_id: str) -> List[SingleAssign]:
    
        ref_assigns = []
        phi_targets = self.replacements.pop(target_id, None)

        if phi_targets is None:
            return ref_assigns

        for phi_target in phi_targets:
            arg = Name(id=target_id, ctx=ast.Load())
            ref_call = ast.Call(func=Name(id=PHI_CREATOR_CALL, ctx=ast.Load()), args=[arg], keywords=[])
            ref_target = Name(id=phi_target, ctx=ast.Store())
            ref_assign = SingleAssign.create(target=ref_target, value=ref_call)
            ref_assigns.append(ref_assign)

        return ref_assigns
    
    # Namespace Nodes
    #############################
    # stop traversing AST at nodes that create new namespaces

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return node
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node
