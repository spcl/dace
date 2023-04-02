# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Replaces certain conditions on inter-state edges."""

import ast

from dace import sdfg as sd
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.symbolic import pystr_to_symbolic


@make_properties
class ReplaceConstantsCloudSC(transformation.MultiStateTransformation):

    @classmethod
    def expressions(cls):
        return [sd.SDFG('_')]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, _, sdfg: sd.SDFG):
        
        replacements = {
            "IPHASE": {
                "0": 1,
                "1": 2,
                "2": 1,
                "3": 2,
                "4": 0
            },
            "LLFALL": {
                "0": False,
                "1": False,
                "2": True,
                "3": True,
                "4": False, 
            } 
            
            
        }
        
        class ArrayLoaderVisitor(ast.NodeTransformer):

            def visit_Subscript(self, node):
                
                if isinstance(node.value, ast.Name) and node.value.id in replacements:
                   
                    # Is there a way to obtain Sympy expression from AST, without going through stringification? 
                    idx_symbolic = pystr_to_symbolic(ast.unparse(node.slice))
                    if idx_symbolic.is_integer and idx_symbolic.is_constant():
                        return ast.Constant(value=replacements[node.value.id][str(idx_symbolic)])
                        
                return node
                    

        for edge in sdfg.edges():
            if isinstance(edge.data, sd.InterstateEdge):
                if edge.data.condition is not None:
                    
                    for code in edge.data.condition.code:
                        ArrayLoaderVisitor().visit(code)
        
        for nsdfg in sdfg.all_sdfgs_recursive(): 
            
            if nsdfg != sdfg:
                nsdfg.apply_transformations(ReplaceConstantsCloudSC)
        