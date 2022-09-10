
from ast import NodeTransformer


class DaCe_Postprocessor(NodeTransformer):

    def visit_WhileCFG(self, node):

        node = self.generic_visit(node)
        
        return node.to_While()

    def visit_SingleAssign(self, node):

        return node.to_assignment()
    
    def visit_Delete(self, node):

        return None
