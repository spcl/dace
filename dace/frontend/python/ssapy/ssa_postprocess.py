
from ast import NodeTransformer


class SSA_Postprocessor(NodeTransformer):
    
    #################
    # Parent Nodes
    #################

    def visit_IfCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_If()

    def visit_ForCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_While()

    def visit_SimpleFunction(self, node):

        node = self.generic_visit(node)
        
        return node.to_FunctionDef()

    def visit_UniqueClass(self, node):

        node = self.generic_visit(node)
        
        return node.to_ClassDef()

    def visit_Interface(self, node):

        return node.to_assignments()

    #################
    # Leaf Nodes
    #################
    
    def visit_UniqueName(self, node):

        return node.to_Name()

    def visit_PhiAssign(self, node):

        return node.cleaned()


class SSA_Finisher(NodeTransformer):
    
    def visit_WhileCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_While()

    def visit_SingleAssign(self, node):

        return node.to_assignment()
    
    def visit_PhiAssign(self, node):

        return node.to_Assign()
