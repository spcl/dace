# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy

from typing import Dict, Union


_body_attrs = ['body', 'orelse', 'finalbody']


class RenameNames(ast.NodeTransformer):
    """ Renames `ast.Name`s. """

    def __init__(self, repl_dict: Dict):
        self.repl_dict = repl_dict

    def visit_Name(self, node: ast.Name) -> ast.Name:
        self.generic_visit(node)
        if node.id in self.repl_dict:
            node.id = self.repl_dict[node.id]
        return node


class AddParentRef(ast.NodeTransformer):
    """ Adds to each node a `parent` attribute pointing to its parent node. """

    last_parent: ast.AST = None

    def visit(self, node):
        node.parent = self.last_parent
        self.last_parent = node
        node = super().visit(node)
        if isinstance(node, ast.AST):
            self.last_parent = node.parent
        return node

class ExpandIf(AddParentRef):

    def visit_If(self, node: ast.If) -> Union[ast.AST, None]:
        self.generic_visit(node)

        body = None
        node_idx = None
        for attr in _body_attrs:
            if not hasattr(node.parent, attr):
                continue
            body = getattr(node.parent, attr)
            node_idx = next((i for i, n in enumerate(body) if n is node), None)
            if node_idx is not None:
                break
        assert node_idx is not None
        node_idx += 1

        assignments = set()
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assignments.add(target.id)
        repl_dict = {}
        for child in node.orelse:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if target.id in assignments:
                            repl_dict[target.id] = f'{target.id}_else'
        
        node.orelse = [RenameNames(repl_dict).visit(n) for n in node.orelse]

        child_num = 0
        for child in body[node_idx:]:
            node.body.append(copy.deepcopy(child))
            node.orelse.append(RenameNames(repl_dict).visit(copy.deepcopy(child)))
            child_num += 1

        for i in range(child_num):
            del body[node_idx]

        return node
