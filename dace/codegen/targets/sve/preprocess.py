# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Preprocessing: This module is responsible for preprocessing a graph to detect some SVE-specific constructs.
    Currently, it is only used for fused operations.
"""

import ast
from dace.codegen.targets.sve import util as util
import dace
import dace.dtypes


class SVEBinOpFuser(ast.NodeTransformer):
    def __init__(self, defined_symbols):
        self.defined_symbols = defined_symbols

    def visit_BinOp(self, t):
        self.visit(t.left)
        self.visit(t.right)

        if util.only_scalars_involed(self.defined_symbols, t.left, t.right):
            return self.generic_visit(t)

        # Detect fused operations

        # MAD: These functions multiply the first two floating-point inputs and add the result to the third input.
        # MLA: These functions multiply the second and third floating-point inputs and add the result to the first input.
        # MSB: These functions multiply the first two floating-point inputs and subtract the result from the third input.
        # MLS: These functions multiply the second and third floating-point inputs and subtract the result from the first input.

        parent_op = t.op.__class__
        left_op = None
        right_op = None

        if isinstance(t.left, ast.BinOp):
            left_op = t.left.op.__class__
        if isinstance(t.right, ast.BinOp):
            right_op = t.right.op.__class__

        args = []
        name = None

        if parent_op == ast.Add:
            if left_op == ast.Mult:
                name = '__svmad_'
                args = [t.left.left, t.left.right, t.right]
            elif right_op == ast.Mult:
                name = '__svmla_'
                args = [t.left, t.right.left, t.right.right]
        elif parent_op == ast.Sub:
            if left_op == ast.Mult:
                name = '__svmsb_'
                args = [t.left.left, t.left.right, t.right]
            elif right_op == ast.Mult:
                name = '__svmls_'
                args = [t.left, t.right.left, t.right.right]

        # Fused ops need at least two of three arguments to be a vector
        if name:
            inferred = util.infer_ast(self.defined_symbols, *args)
            scalar_args = sum([util.is_scalar(tp) for tp in inferred])
            if scalar_args > 1:
                return self.generic_visit(t)
            # Add the type suffix for internal representation
            name += util.TYPE_TO_SVE_SUFFIX[util.get_base_type(dace.dtypes.result_type_of(*inferred))]
            return ast.copy_location(ast.Call(func=ast.Name(name, ast.Load()), args=args, keywords=[]), t)

        return self.generic_visit(t)
