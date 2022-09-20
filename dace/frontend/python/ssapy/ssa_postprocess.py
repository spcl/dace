# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from ast import NodeTransformer


class SSA_Reducer(NodeTransformer):

    def __init__(self, clean_interface: bool = True) -> None:
        
        self.clean_interface = clean_interface
    
    #################
    # Parent Nodes
    #################

    def visit_IfCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_If()

    def visit_ForCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_WhileCFG()

    def visit_ForCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_SSAFor()

    def visit_SimpleFunction(self, node):

        node = self.generic_visit(node)
        
        return node.to_FunctionDef()

    def visit_UniqueClass(self, node):

        node = self.generic_visit(node)
        
        return node.to_ClassDef()

    def visit_Interface(self, node):

        return node.to_assignments(include_dels=self.clean_interface)

    #################
    # Leaf Nodes
    #################
    
    def visit_UniqueName(self, node):

        return node.to_Name()

    def visit_PhiAssign(self, node):

        return node.cleaned()


class SSA_Postprocessor(NodeTransformer):
    
    def visit_WhileCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_While()

    def visit_SingleAssign(self, node):

        return node.to_assignment()
    
    def visit_PhiAssign(self, node):

        return node.to_Assign()
