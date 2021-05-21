# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    AST to SVE: This module is responsible for converting an AST into SVE code.
"""

import dace
import ast
import astunparse
from dace.codegen import cppunparse
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from typing import IO, Tuple, Union
from dace import registry, symbolic, dtypes
import dace.codegen.targets.sve.preprocess as preprocess
import dace.codegen.targets.sve.util as util
import dace.frontend.python.astutils as astutils
from dace.codegen.targets.sve.type_compatibility import assert_type_compatibility
import copy
import collections


class SVEUnparser(cppunparse.CPPUnparser):
    def __init__(self,
                 sdfg: SDFG,
                 tree: ast.AST,
                 file: IO[str],
                 code,
                 memlets,
                 pred_name,
                 counter_type,
                 defined_symbols=None,
                 stream_associations=dict()):

        self.sdfg = sdfg
        self.dtypes = {k: v[3] for k, v in memlets.items() if k is not None}
        for k, v in sdfg.constants.items():
            if k is not None:
                self.dtypes[k] = v.dtype

        # This defines the type of the loop param and is used whenever we work on predicates to find out the bit size
        self.counter_type = counter_type
        self.pred_bits = counter_type.bytes * 8

        # Keeps track of the currently used predicate
        # This can change during the unparsing (e.g. in an If block)
        self.pred_name = pred_name

        self.code = code

        # This value is used to determine whether we are in an If block, and how deep we are (to name the predicates)
        self.if_depth = 0

        # Stream associations keep track between the local stream variable name <-> underlying stream
        self.stream_associations = stream_associations

        # Detect fused operations first
        preprocessed = preprocess.SVEPreprocessor(defined_symbols).visit(tree)

        super().__init__(preprocessed,
                         0,
                         cppunparse.CPPLocals(),
                         file,
                         expr_semicolon=True,
                         type_inference=True,
                         defined_symbols=defined_symbols)

    def get_defined_symbols(self) -> collections.OrderedDict:
        sym = self.defined_symbols.copy()
        sym.update(self.locals.get_name_type_associations())
        return sym

    def infer(self, *args) -> tuple:
        return util.infer_ast(self.get_defined_symbols(), *args)

    def assert_type_compatibility(self, *args):
        return assert_type_compatibility(self.get_defined_symbols(), list(args))

    def reorder_vector_scalar(self, left: ast.AST, right: ast.AST) -> tuple:
        """
        SVE vector-scalar operations are only defined for LHS a vector and RHS a scalar.
        Therefore we might have to flip the sides.
        """
        lhs_type, rhs_type = self.infer(left, right)
        lhs_vec = util.is_vector(lhs_type)
        rhs_vec = util.is_vector(rhs_type)

        if not lhs_vec and not rhs_vec:
            raise TypeError('At least one operand must be a vector')

        if rhs_vec and not lhs_vec:
            return (right, left)

        return (left, right)

    def dispatch_scalar_dup(self, vec_type: dace.typeclass,
                            scalar_type: dace.typeclass, body: ast.AST):
        """ Some SVE instructions do not accept any scalar argument. In this case we must duplicate the vector. """

        # Duplicates the scalar to create a vector
        self.write(f'svdup_{util.TYPE_TO_SVE_SUFFIX[vec_type.type]}(')
        self.dispatch_scalar_cast(vec_type, scalar_type, body)
        self.write(')')

    def dispatch_scalar_cast(self, vec_type: dace.typeclass,
                             scalar_type: dace.typeclass, body: ast.AST):
        """ Some SVE instructions accept a scalar argument, but it must be the exact type, so we cast it. """
        required = util.get_base_type(vec_type)

        if scalar_type != required:
            # Cast the scalar because it is not of the required type
            self.write(f'({required.ctype}) ')

        self.dispatch(body)

    def generate_case_predicate(self, t: ast.If, acc_pred: str, id: int) -> str:
        test_pred = f'__pg_test_{self.if_depth}_{id}'

        # Compute the test predicate for the current case
        self.fill(f'svbool_t {test_pred} = ')
        self.pred_name = acc_pred
        self.dispatch(t.test)
        self.write(';')

        # Update the accumulator to exclude the test (the next case only occurs if we had failed)
        # BIC(A, B) = A AND NOT B
        self.fill(f'{acc_pred} = svbic_z({acc_pred}, {acc_pred}, {test_pred});')

        return test_pred

    def generate_case_body(self, t: ast.If, test_pred: str):
        # It is very important to remember that elif's rely on the previous elif's (they are sequential)
        # i.e. the first subcase that hits wins and all following ones lose

        self.fill('// Case ' + astunparse.unparse(t.test))
        self.enter()

        # Generate the case body, which will use the test predicate in the ops
        self.pred_name = test_pred

        # Allow for local definitions, so we backup all symbols
        sym = copy.deepcopy(self.defined_symbols)
        self.dispatch(t.body)
        self.defined_symbols = sym

        self.leave()

    def _If(self, t):
        self.if_depth += 1
        old_pred = self.pred_name

        # Accumulator predicate for the cases
        acc_pred = f'__pg_acc_{self.if_depth}'

        self.fill('// ==== If ===\n')
        self.enter()

        # It is very important to remember that elif's rely on the previous elif's (they are sequential)
        # i.e. the first subcase that hits wins and all following ones lose
        # The case accumulator keeps track of which elements of the current vector
        # would be still affected by the current case (since the tests are evaluated sequentially)
        self.fill(f'svbool_t {acc_pred} = {self.pred_name};')

        # Find all branches (except for the last else, because it is treated differently)
        branches = [t]
        while (t.orelse and len(t.orelse) == 1
            and isinstance(t.orelse[0], ast.If)):
            t = t.orelse[0]
            branches.append(t)

        # Precompute all case predicates
        predicates = [self.generate_case_predicate(b, acc_pred, i + 1) for i, b in enumerate(branches)]

        # Generate the cases
        for b, p in zip(branches, predicates): 
            self.generate_case_body(b, p)

        if t.orelse:
            self.fill('// Case (else)\n')
            # else case simply uses the accumulator (i.e. where other all cases failed)
            self.enter()
            self.pred_name = acc_pred
            for t_ in t.orelse:
                self.dispatch(t_)
            self.leave()

        self.leave()
        self.fill('// ==== End If ===')

        self.pred_name = old_pred
        self.if_depth -= 1

    def _Assign(self, t):
        if len(t.targets) > 1:
            raise util.NotSupportedError('Tuple output not supported')

        target = t.targets[0]

        lhs_type, rhs_type = self.infer(target, t.value)

        if lhs_type is None:
            self.assert_type_compatibility(rhs_type)
        else:
            self.assert_type_compatibility(lhs_type, rhs_type)

        is_new_variable = False

        if lhs_type is None:
            # The LHS could involve a variable name that was not declared (which is why inference fails)

            if not isinstance(
                    target,
                    ast.Name) or target.id in self.get_defined_symbols():
                # Either we don't assign to a name, or the variable name has already been declared (but infer still fails, i.e. something went wrong!)
                raise NotImplementedError('Can not infer LHS of assignment')

            # Declare it as `type name`

            lhs_type = rhs_type
            if isinstance(lhs_type, dace.dtypes.vector):
                # SVE register is possible
                self.fill(util.TYPE_TO_SVE[lhs_type.type])
                self.write(' ')
                self.defined_symbols.update({target.id: lhs_type})

            is_new_variable = True

        if rhs_type is None:
            raise NotImplementedError('Can not infer RHS of assignment')

        lhs_vec = util.is_vector(lhs_type)
        rhs_vec = util.is_vector(rhs_type)

        if not lhs_vec and not rhs_vec:
            # Only scalars involved
            super()._Assign(t)
            if isinstance(target, ast.Name):
                self.defined_symbols.update({target.id: rhs_type})
            return

        if not is_new_variable:
            self.fill()

        if not lhs_vec and rhs_vec:
            raise util.NotSupportedError('Cannot assign a vector to a scalar')

        if self.if_depth > 0 and not lhs_vec:
            raise util.NotSupportedError(
                'Assignments in an if block must be to a vector or stream (otherwise unvectorizable)'
            )

        if isinstance(target,
                      ast.Name) and target.id in self.stream_associations:
            # Assigning to a stream variable is equivalent to pushing into a stream

            target_stream = self.stream_associations[target.id]

            self.enter()
            self.fill('// === Stream push ===')

            stream_type = target_stream[1]

            rhs_base = util.get_base_type(rhs_type)

            # Create a temporary array on the heap, where we will copy the SVE register contents to
            self.fill('{} __tmp[{} / {}];'.format(stream_type.ctype,
                                                  util.REGISTER_BYTE_SIZE,
                                                  rhs_base.bytes))

            # Count the number of "to push" elements based on the current predicate
            self.fill('size_t __cnt = svcntp_b{}({}, {});'.format(
                self.pred_bits, self.pred_name, self.pred_name))

            # Store the contents of the SVE register in the temporary array
            self.fill(
                f'svst1(svwhilelt_b{self.pred_bits}(0, ({self.counter_type}) __cnt), __tmp, '
            )

            # The contents should be compacted (i.e. all elements where the predicate is true are aligned)
            self.write(f'svcompact({self.pred_name}, ')

            if rhs_vec:
                # We are pushing a vector
                self.dispatch(t.value)
            else:
                # If we push a scalar, we must first convert it into a vector
                self.dispatch_scalar_dup(stream_type, rhs_type, t.value)

            self.write('));')

            # Push the temporary array onto the stream using DaCe's push
            self.fill(f'{target_stream[0]}.push(&__tmp[0], __cnt);')
            self.leave()

            return

        self.dispatch(target)
        self.write(' = ')

        # Note, that if this variable is declared in the same line, we don't need to select at all (there is nothing to select from, because it just got declared)
        if self.if_depth > 0 and not is_new_variable:
            # If we are in an If block, we assign based on the predicate
            # In case of "a = b", we do:
            # a = select(if_pred, b, a)

            self.write(f'svsel({self.pred_name}, ')

        if lhs_vec and not rhs_vec:
            # Upcast scalar to vector by duplicating it

            self.dispatch_scalar_dup(lhs_type, rhs_type, t.value)
        else:
            self.dispatch(t.value)

        if self.if_depth > 0 and not is_new_variable:
            self.write(', ')
            self.dispatch(target)
            self.write(')')

        self.write(';')

    def _Call(self, t):
        name = None
        if isinstance(t.func, ast.Name):
            name = util.FUSED_OPERATION_TO_SVE.get(t.func.id)
            if name is None:
                raise NotImplementedError(
                    f'Function {t.func.id} is not implemented')
        elif isinstance(t.func, ast.Attribute):
            name = util.MATH_FUNCTION_TO_SVE.get(astutils.rname(t.func))
            if name is None:
                raise NotImplementedError(
                    f'Function {astutils.rname(t.func)} is not implemented')

        if util.only_scalars_involed(self.get_defined_symbols(), *t.args):
            # Replace default modules (e.g., math) with dace::math::
            attr_name = astutils.rname(t)
            module_name = attr_name[:attr_name.rfind(".")]
            func_name = attr_name[attr_name.rfind(".") + 1:]
            if module_name not in dtypes._ALLOWED_MODULES:
                raise NotImplementedError(
                    f'Module {module_name} is not implemented')
            cpp_mod_name = dtypes._ALLOWED_MODULES[module_name]
            name = cpp_mod_name + func_name

            self.write(name)
            self.write('(')

            comma = False
            for e in t.args:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.dispatch(e)
            self.write(')')
            return

        arg_types = self.infer(*t.args)
        self.assert_type_compatibility(*arg_types)

        res_type = dtypes.result_type_of(arg_types[0], *arg_types)

        # Vectorized function
        self.write('{}_x({}, '.format(name, self.pred_name))

        # Due to a very limited set of functions are available, the correctness of the reordering is ensured in the preprocessing
        arg_count = len(t.args)
        if arg_count >= 2:  # Binary ops need the scalar on the RHS
            t.args[0], t.args[1] = self.reorder_vector_scalar(
                t.args[0], t.args[1])
        if arg_count == 3:  # Fused ops (3 args) need the scalar at the end
            t.args[1], t.args[2] = self.reorder_vector_scalar(
                t.args[1], t.args[2])
        if arg_count > 3:
            raise NotImplementedError('Too many arguments')

        comma = False
        for e in t.args:
            if comma:
                self.write(", ")
            else:
                comma = True
            type = self.infer(e)[0]
            if util.is_scalar(type):
                self.dispatch_scalar_cast(res_type, type, e)
            else:
                self.dispatch(e)
        self.write(')')

    def _UnaryOp(self, t):
        self.assert_type_compatibility(self.infer(t.operand)[0])

        if util.only_scalars_involed(self.get_defined_symbols(), t.operand):
            return super()._UnaryOp(t)

        if isinstance(t.op, ast.UAdd):
            t.dispatch(t.operand)
            return

        if t.op.__class__ not in util.UN_OP_TO_SVE:
            raise NotImplementedError(
                f'Unary operation {t.op.__class__.__name__} not implemented')

        self.write('{}_x({}, '.format(util.UN_OP_TO_SVE[t.op.__class__],
                                      self.pred_name))

        self.dispatch(t.operand)
        self.write(')')

    def _AugAssign(self, t):
        # Break up statements like a += 5 into a = a + 5
        self._Assign(ast.Assign([t.target], ast.BinOp(t.target, t.op, t.value)))

    def _BinOp(self, t):
        if util.only_scalars_involed(self.get_defined_symbols(), t.left,
                                     t.right):
            return super()._BinOp(t)

        lhs_type, rhs_type = self.infer(t.left, t.right)
        self.assert_type_compatibility(lhs_type, rhs_type)

        lhs_vec = util.is_vector(lhs_type)
        rhs_vec = util.is_vector(rhs_type)

        if t.op.__class__ not in util.BIN_OP_TO_SVE:
            raise NotImplementedError(
                f'Binary operation {t.op.__class__.__name__} not implemented')

        op_name = util.BIN_OP_TO_SVE[t.op.__class__]

        # Special case: scalar / vector => svdivr (division reversed)
        if isinstance(t.op, ast.Div) and not lhs_vec and rhs_vec:
            op_name = 'svdivr'

        self.write('{}_x({}, '.format(op_name, self.pred_name))

        left, right = self.reorder_vector_scalar(t.left, t.right)

        lhs_type, rhs_type = self.infer(left, right)
        self.assert_type_compatibility(lhs_type, rhs_type)

        self.dispatch(left)
        self.write(', ')

        if right != t.right:
            self.dispatch_scalar_cast(lhs_type, rhs_type, right)
        else:
            self.dispatch(right)

        self.write(')')

    #
    # Predicate related stuff
    #

    def _IfExp(self, t):
        if util.only_scalars_involed(self.get_defined_symbols(), t.test, t.body,
                                     t.orelse):
            return super()._IfExp(t)

        if_type, else_type = self.infer(t.body, t.orelse)

        type = dtypes.result_type_of(if_type, else_type)
        self.assert_type_compatibility(type)

        if_vec = util.is_vector(if_type)
        else_vec = util.is_vector(else_type)

        # svsel() doesn't accept any scalar argument, so we must manually create a vector if we have a scalar (svdup())

        self.write('svsel(')
        self.dispatch(t.test)
        self.write(', ')

        if not if_vec:
            self.dispatch_scalar_dup(else_type, if_type, t.body)
        else:
            self.dispatch(t.body)

        self.write(', ')

        if not else_vec:
            self.dispatch_scalar_dup(if_type, else_type, t.orelse)
        else:
            self.dispatch(t.orelse)

        self.write(')')

    def _BoolOp(self, t):
        if util.only_scalars_involed(self.get_defined_symbols(), *t.values):
            return super()._BoolOp(t)

        types = self.infer(*t.values)

        # Bool ops are nested SVE instructions, so we must make sure they all act on vectors
        for type in types:
            if not isinstance(type, dtypes.vector):
                raise util.NotSupportedError('Unvectorizable boolean operation')

        # There can be many t.values, e.g. if
        # x or y or z
        for val in t.values:
            # The last entry doesn't need more nesting
            if val == t.values[-1]:
                self.dispatch(t.values[-1])
                break

            # Binary nesting
            self.write('{}_z({}, '.format(util.BOOL_OP_TO_SVE[t.op.__class__],
                                          self.pred_name))
            self.dispatch(val)
            self.write(', ')

        # Close all except the last bracket (because the last entry isn't nested)
        self.write(')' * (len(t.values) - 1))

    def _Compare(self, t):
        if len(t.ops) != 1:
            # This includes things like  a < b <= c
            raise NotImplementedError(
                'Multiple comparisions at once not implemented')

        lhs = t.left
        op = t.ops[0]
        rhs = t.comparators[0]

        self.assert_type_compatibility(*self.infer(lhs, rhs))

        if util.only_scalars_involed(self.get_defined_symbols(), lhs, rhs):
            return super()._Compare(t)

        if op.__class__ not in util.COMPARE_TO_SVE:
            raise NotImplementedError('Comparator not supported')

        new_lhs, new_rhs = self.reorder_vector_scalar(lhs, rhs)

        if lhs != new_lhs:
            # Invert inequality, because scalar must be at the end
            op = util.FLIP_INEQUALITY[op.__class__]()

        self.write('{}({}, '.format(util.COMPARE_TO_SVE[op.__class__],
                                    self.pred_name))

        lhs_type, rhs_type = self.infer(new_lhs, new_rhs)

        self.dispatch(new_lhs)
        self.write(', ')

        if util.is_scalar(rhs_type):
            self.dispatch_scalar_cast(lhs_type, rhs_type, new_rhs)
        else:
            self.dispatch(new_rhs)

        self.write(')')

    def _Subscript(self, t):
        type, slice = self.infer(t.value, t.slice.value)
        self.assert_type_compatibility(type)
        self.assert_type_compatibility(slice)

        if isinstance(type, dtypes.pointer) and isinstance(slice, dtypes.vector):
            # Indirect load
            self.write(f'svld1_gather_index({self.pred_name}, ')
            self.dispatch(t.value)
            self.write(', ')
            self.dispatch(t.slice)
            self.write(')')
            return

        raise NotImplementedError(
            'You should define memlets for array accesses')
