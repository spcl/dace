# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    AST to SVE: This module is responsible for converting an AST into SVE code.
"""

from dace.frontend.python.wrappers import stream
import dace
import ast
import astunparse
from dace.codegen import cppunparse
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from typing import IO, Tuple, Union
from dace import registry, symbolic, dtypes
from dace.codegen.targets.sve import preprocess as preprocess
from dace.codegen.targets.sve import util as util
import dace.frontend.python.astutils as astutils
from dace.codegen.targets.sve.type_compatibility import assert_type_compatibility
import copy
import collections
import numpy as np
from dace import data as data
from dace.frontend.operations import detect_reduction_type
from dace.codegen.targets.cpp import is_write_conflicted, cpp_ptr_expr, DefinedType, sym2cpp


class SVEUnparser(cppunparse.CPPUnparser):
    def __init__(self,
                 sdfg: SDFG,
                 dfg,
                 map,
                 cpu_codegen,
                 tree: ast.AST,
                 file: IO[str],
                 code,
                 memlets,
                 pred_name,
                 counter_type,
                 defined_symbols=None,
                 stream_associations=None,
                 wcr_associations=None):

        self.sdfg = sdfg
        self.dfg = dfg
        self.map = map

        self.cpu_codegen = cpu_codegen

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
        self.stream_associations = stream_associations or {}
        self.wcr_associations = wcr_associations or {}

        # Detect fused operations first (are converted into internal calls)
        preprocessed = preprocess.SVEBinOpFuser(defined_symbols).visit(tree)

        # Make sure all internal calls are defined for the inference
        defined_symbols.update(util.get_internal_symbols())

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

    def dispatch_expect(self, tree: ast.AST, expect: dtypes.typeclass):
        """
        This function is an extension to the dispatch() call and allows to pass
        the type that is expected when unparsing the tree and will take care of
        any casting that might be required. It is mainly used in SVE instructions
        in cases where an argument must be of some type (otherwise the
        compiler complains).
        """

        inf = self.infer(tree)[0]

        # Sanity check
        if not inf:
            raise util.NotSupportedError(
                f'Could not infer the expression type of `{astunparse.unparse(tree)}`'
            )

        if isinstance(inf, dtypes.vector):
            # Unparsing a vector
            if isinstance(expect, dtypes.vector):
                # A vector is expected
                if inf.vtype.type == expect.vtype.type:
                    # No cast required
                    self.dispatch(tree)
                else:
                    # TODO: Cast vectors (but only if same bitwidth)
                    raise NotImplementedError(
                        'Vector-vector casting not implemented')
            else:
                # A pointer or scalar is expected (incompatible)
                raise util.NotSupportedError(
                    'Given a vector, expected a scalar or pointer')
        elif isinstance(inf, dtypes.pointer):
            # Unparsing a pointer
            if isinstance(expect, dtypes.pointer):
                # Expecting a pointer
                if inf.base_type.type == expect.base_type.type:
                    # No cast required, expect for `long long` fix
                    if expect.base_type.type == np.int64:
                        self.write('(int_64_t*) ')
                    if expect.base_type.type == np.uint64:
                        self.write('(uint_64_t*) ')
                    self.dispatch(tree)
                else:
                    raise util.NotSupportedError('Inconsistent pointer types')
            else:
                # Expecting anything else
                raise util.NotSupportedError(
                    'Given a pointer, expected a scalar or vector')
        else:
            # Unparsing a scalar
            if isinstance(expect, dtypes.vector):
                # Expecting a vector: duplicate the scalar
                if expect.type in [np.bool, np.bool_, bool]:
                    # Special case for duplicating boolean into predicate
                    suffix = f'b{self.pred_bits}'
                    #self.write(f'svptrue_{suffix}()')
                    self.dispatch_expect(tree, expect.base_type)
                    self.write(f' ? svptrue_{suffix}() : svpfalse_b()')
                else:
                    self.write(f'svdup_{util.TYPE_TO_SVE_SUFFIX[expect.type]}(')
                    self.dispatch_expect(tree, expect.base_type)
                    self.write(')')

            elif isinstance(expect, dtypes.pointer):
                # Expecting a pointer
                raise util.NotSupportedError(
                    'Given a scalar, expected a pointer')
            else:
                # Expecting a scalar: cast if needed
                cast_ctype = None
                if inf.type != expect.type:
                    cast_ctype = expect.ctype

                # Special casting for `long long`
                if expect.type == np.int64:
                    cast_ctype = 'int64_t'
                elif expect.type == np.uint64:
                    cast_ctype = 'uint64_t'

                if cast_ctype:
                    self.write(f'({cast_ctype}) ')

                self.dispatch(tree)

    def generate_case_predicate(self, t: ast.If, acc_pred: str, id: int) -> str:
        test_pred = f'__pg_test_{self.if_depth}_{id}'

        # Compute the test predicate for the current case
        self.fill(f'svbool_t {test_pred} = ')
        self.pred_name = acc_pred
        self.dispatch_expect(t.test, dtypes.vector(dace.bool, -1))
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
        predicates = [
            self.generate_case_predicate(b, acc_pred, i + 1)
            for i, b in enumerate(branches)
        ]

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

    def push_to_stream(self, t, target):
        target_stream = self.stream_associations[target.id]
        stream_type = target_stream[1]

        self.enter()
        self.fill('\n// === Stream push ===')

        # Casting in case of `long long`
        stream_type = copy.copy(stream_type)
        if stream_type.type == np.int64:
            stream_type.ctype = 'int64_t'
        elif stream_type.type == np.uint64:
            stream_type.ctype = 'uint64_t'

        # Create a temporary array on the heap, where we will copy the SVE register contents to
        self.fill('{} __tmp[{} / {}];'.format(stream_type,
                                              util.REGISTER_BYTE_SIZE,
                                              stream_type.bytes))

        # Count the number of "to push" elements based on the current predicate
        self.fill('size_t __cnt = svcntp_b{}({}, {});'.format(
            self.pred_bits, self.pred_name, self.pred_name))

        # Store the contents of the SVE register in the temporary array
        self.fill(
            f'svst1(svwhilelt_b{self.pred_bits}(0, ({self.counter_type}) __cnt), __tmp, '
        )

        # The contents should be compacted (i.e. all elements where the predicate is true are aligned)
        self.write(f'svcompact({self.pred_name}, ')
        self.dispatch_expect(t.value, dtypes.vector(stream_type, -1))
        self.write('));')

        ptr_cast = ''
        # Special casting for int64_t back to `long long`
        if stream_type.type == np.int64:
            ptr_cast = '(long long*) '
        elif stream_type.type == np.uint64:
            ptr_cast = '(unsigned long long*) '

        # Push the temporary array onto the stream using DaCe's push
        self.fill(f'{target_stream[0]}.push({ptr_cast}&__tmp[0], __cnt);')
        self.leave()

    def vector_reduction_expr(self, edge, dtype, rhs):
        # Check whether it is a known reduction that is possible in SVE
        reduction_type = detect_reduction_type(edge.data.wcr)
        if reduction_type not in util.REDUCTION_TYPE_TO_SVE:
            raise util.NotSupportedError('Unsupported reduction in SVE')

        nc = not is_write_conflicted(self.dfg, edge)
        if not nc or not isinstance(edge.src.out_connectors[edge.src_conn],
                                    (dtypes.pointer, dtypes.vector)):
            # WCR on vectors works in two steps:
            # 1. Reduce the SVE register using SVE instructions into a scalar
            # 2. WCR the scalar to memory using DaCe functionality
            dst_node = self.dfg.memlet_path(edge)[-1].dst
            if (isinstance(dst_node, nodes.AccessNode) and 
                dst_node.desc(self.sdfg).storage == dtypes.StorageType.SVE_Register):
                return

            wcr = self.cpu_codegen.write_and_resolve_expr(self.sdfg,
                                                          edge.data,
                                                          not nc,
                                                          None,
                                                          '@',
                                                          dtype=dtype)
            self.fill(wcr[:wcr.find('@')])
            self.write(util.REDUCTION_TYPE_TO_SVE[reduction_type])
            self.write('(')
            self.write(self.pred_name)
            self.write(', ')
            self.dispatch_expect(rhs, dtypes.vector(dtype, -1))
            self.write(')')
            self.write(wcr[wcr.find('@') + 1:])
            self.write(';')
        else:
            ######################
            # Horizontal non-atomic reduction

            stride = edge.data.get_stride(self.sdfg, self.map)

            # long long fix
            ptr_cast = ''
            src_type = edge.src.out_connectors[edge.src_conn]

            if src_type.type == np.int64:
                ptr_cast = '(int64_t*) '
            elif src_type.type == np.uint64:
                ptr_cast = '(uint64_t*) '

            store_args = '{}, {}'.format(
                self.pred_name,
                ptr_cast +
                cpp_ptr_expr(self.sdfg, edge.data, DefinedType.Pointer),
            )

            red_type = util.REDUCTION_TYPE_TO_SVE[reduction_type][:-1] + '_x'
            if stride == 1:
                self.write(
                    f'svst1({store_args}, {red_type}({self.pred_name}, svld1({store_args}), '
                )
                self.dispatch_expect(rhs, dtypes.vector(dtype, -1))
                self.write('));')
            else:
                store_args = f'{store_args}, svindex_s{util.get_base_type(src_type).bytes * 8}(0, {sym2cpp(stride)})'
                self.write(
                    f'svst1_scatter_index({store_args}, {red_type}({self.pred_name}, svld1_gather_index({store_args}), '
                )
                self.dispatch_expect(rhs, dtypes.vector(dtype, -1))
                self.write('));')

    def resolve_conflict(self, t, target):
        dst_node, edge, base_type = self.wcr_associations[target.id]
        self.vector_reduction_expr(edge, base_type, t.value)

    def _Assign(self, t):
        if len(t.targets) > 1:
            raise util.NotSupportedError('Tuple output not supported')

        target = t.targets[0]
        if isinstance(target,
                      ast.Name) and target.id in self.stream_associations:
            # Assigning to a stream variable is equivalent to a push
            self.push_to_stream(t, target)
            return
        elif isinstance(target,
                        ast.Name) and target.id in self.wcr_associations:
            # Assigning to a WCR output
            self.resolve_conflict(t, target)
            return

        lhs_type, rhs_type = self.infer(target, t.value)

        if rhs_type is None:
            raise NotImplementedError(
                f'Can not infer RHS of assignment ({astunparse.unparse(t.value)})'
            )

        is_new_variable = False

        if lhs_type is None:
            # The LHS could involve a variable name that was not declared (which is why inference fails)
            if not isinstance(
                    target,
                    ast.Name) or target.id in self.get_defined_symbols():
                # Either we don't assign to a name, or the variable name has
                # already been declared (but infer still fails, i.e. something went wrong!)
                raise NotImplementedError('Can not infer LHS of assignment')

            # Declare it as `type name`
            lhs_type = rhs_type
            if isinstance(rhs_type, dtypes.vector):
                # SVE register is possible (declare it as svXXX_t)
                self.fill(util.TYPE_TO_SVE[rhs_type.type])
                self.write(' ')
                # Define the new symbol as vector
                self.defined_symbols.update({target.id: rhs_type})
            elif isinstance(rhs_type, dtypes.pointer):
                raise util.NotSupportedError(
                    'Defining pointers in Tasklet code not supported')

            # Otherwise, the fallback will grab the case of a scalar,
            # because the RHS is scalar, and the LHS is the same
            is_new_variable = True

        # LHS and RHS types are now both well defined
        lhs_vec = isinstance(lhs_type, dtypes.vector)
        rhs_vec = isinstance(rhs_type, dtypes.vector)

        # TODO: This is only bad if we assign to a variable from an outer scope
        """
        if self.if_depth > 0 and not lhs_vec:
            raise util.NotSupportedError(
                'Assignments in an if block must be to a vector or stream (otherwise not vectorizable)')
        """

        if not lhs_vec and not rhs_vec:
            # Simple scalar-scalar assign handled by fallback
            super()._Assign(t)
            if isinstance(target, ast.Name):
                self.defined_symbols.update({target.id: rhs_type})
            return

        if not is_new_variable:
            # Indentation fix
            self.fill()

        # Some vector assignment
        self.dispatch(target)
        self.write(' = ')

        # Note, that if this variable is declared in the same line, we
        # don't need to select at all (there is nothing to select from,
        # because it just got declared)
        if self.if_depth > 0 and not is_new_variable:
            # If we are in an If block, we assign based on the predicate
            # In case of "a = b", we do:
            # a = select(if_pred, b, a)
            self.write(f'svsel({self.pred_name}, ')

        self.dispatch_expect(t.value, lhs_type)

        if self.if_depth > 0 and not is_new_variable:
            # Close the select
            self.write(', ')
            self.dispatch(target)
            self.write(')')

        self.write(';')

    def _Call(self, t):
        res_type = self.infer(t)[0]
        if not res_type:
            raise util.NotSupportedError(f'Unsupported call')

        if not isinstance(res_type, dtypes.vector):
            # Call does not involve any vectors (to our knowledge)
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

        name = None
        if isinstance(t.func, ast.Name):
            # Could be an internal operation (provided by the preprocessor)
            if not util.is_sve_internal(t.func.id):
                raise NotImplementedError(
                    f'Function {t.func.id} is not implemented')
            name = util.internal_to_external(t.func.id)[0]
        elif isinstance(t.func, ast.Attribute):
            # Some module function (xxx.xxx), make sure it is available
            name = util.MATH_FUNCTION_TO_SVE.get(astutils.rname(t.func))
            if name is None:
                raise NotImplementedError(
                    f'Function {astutils.rname(t.func)} is not implemented')

        # Vectorized function
        self.write('{}_x({}, '.format(name, self.pred_name))
        comma = False
        for e in t.args:
            if comma:
                self.write(", ")
            else:
                comma = True
            self.dispatch_expect(e, res_type)
        self.write(')')

    def _UnaryOp(self, t):
        inf_type = self.infer(t.operand)[0]

        if util.is_scalar(inf_type):
            return super()._UnaryOp(t)

        if isinstance(t.op, ast.UAdd):
            # A + in front is just ignored
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
        lhs_type, rhs_type = self.infer(t.left, t.right)
        res_type = dtypes.result_type_of(lhs_type, rhs_type)

        if not isinstance(res_type, (dtypes.vector, dtypes.pointer)):
            return super()._BinOp(t)

        if t.op.__class__ not in util.BIN_OP_TO_SVE:
            raise NotImplementedError(
                f'Binary operation {t.op.__class__.__name__} not implemented')

        op_name = util.BIN_OP_TO_SVE[t.op.__class__]

        self.write('{}_x({}, '.format(op_name, self.pred_name))
        self.dispatch_expect(t.left, res_type)
        self.write(', ')
        self.dispatch_expect(t.right, res_type)
        self.write(')')

    #
    # Predicate related stuff
    #

    def _IfExp(self, t):
        if util.only_scalars_involed(self.get_defined_symbols(), t.test, t.body,
                                     t.orelse):
            return super()._IfExp(t)

        if_type, else_type = self.infer(t.body, t.orelse)
        res_type = dtypes.result_type_of(if_type, else_type)
        if not isinstance(res_type, dtypes.vector):
            res_type = dtypes.vector(res_type, -1)

        self.write('svsel(')
        self.dispatch_expect(t.test, dtypes.vector(dace.bool, -1))
        self.write(', ')
        self.dispatch_expect(t.body, res_type)
        self.write(', ')
        self.dispatch_expect(t.orelse, res_type)
        self.write(')')

    def _BoolOp(self, t):
        if util.only_scalars_involed(self.get_defined_symbols(), *t.values):
            return super()._BoolOp(t)

        types = self.infer(*t.values)

        # Bool ops are nested SVE instructions, so we must make sure they all act on vectors
        for type in types:
            if not isinstance(type, dtypes.vector):
                raise util.NotSupportedError(
                    'Non-vectorizable boolean operation')

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
                'Multiple comparisons at once not implemented')

        lhs = t.left
        op = t.ops[0]
        rhs = t.comparators[0]

        self.assert_type_compatibility(*self.infer(lhs, rhs))

        if not isinstance(self.infer(t)[0], (dtypes.vector, dtypes.pointer)):
            return super()._Compare(t)

        if op.__class__ not in util.COMPARE_TO_SVE:
            raise NotImplementedError('Comparator not supported')

        self.write('{}({}, '.format(util.COMPARE_TO_SVE[op.__class__],
                                    self.pred_name))

        lhs_type, rhs_type = self.infer(lhs, rhs)
        res_type = dtypes.result_type_of(lhs_type, rhs_type)

        self.dispatch_expect(lhs, res_type)
        self.write(', ')
        self.dispatch_expect(rhs, res_type)
        self.write(')')

    def _Subscript(self, t):
        type, slice = self.infer(t.value, t.slice)
        self.assert_type_compatibility(type)
        self.assert_type_compatibility(slice)

        if isinstance(type, dtypes.pointer) and isinstance(
                slice, dtypes.vector):
            # Indirect load
            self.write(f'svld1_gather_index({self.pred_name}, ')
            self.dispatch_expect(t.value, type)
            self.write(', ')
            self.dispatch(t.slice)
            self.write(')')
            return

        raise NotImplementedError(
            'You should define memlets for array accesses')
