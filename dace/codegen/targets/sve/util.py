# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Utility functions for SVE: Contains many datatype mappings (Python to SVE) and frequently used functions.
"""

import numpy as np
import dace
import dace.dtypes as dtypes
import ast
import dace.codegen.targets
from dace.codegen.targets.sve import infer as infer
import collections
import itertools
import numpy

# Translation of types to C++ types
_SVE_CTYPES = {
    None: "void",
    int: "dace::int32",
    float: "dace::float64",
    complex: "dace::complex64",
    bool: "dace::bool_",
    numpy.bool: "dace::bool_",
    numpy.bool_: "dace::bool_",
    numpy.int8: "dace::int8",
    numpy.int16: "dace::int16",
    numpy.int32: "dace::int32",
    numpy.int64: "dace::int64",
    numpy.uint8: "dace::uint8",
    numpy.uint16: "dace::uint16",
    numpy.uint32: "dace::uint32",
    numpy.uint64: "dace::uint64",
    numpy.float16: "dace::float16",
    numpy.float32: "dace::float32",
    numpy.float64: "dace::float64",
    numpy.complex64: "dace::complex64",
    numpy.complex128: "dace::complex128",
}

# Used as the unknown SVE vector size in the graph
SVE_LEN = dace.symbol('__dace_sve_len')


class NotSupportedError(Exception):
    def __init__(self, message):
        super().__init__(message)


REGISTER_BYTE_SIZE = '__SVE_REGISTER_BYTES'


def instr(name: str, type: dace.typeclass = None, suffix: bool = True, pred_mode: str = None) -> str:
    """
    Generates the name of the SVE instruction with possibly suffixes and flags.
    param name: Instruction name (without `sv` prefix)
    param type: The type of the involved operands
    param suffix: If True, the type suffix (e.g. _f32) is added
    param pred_mode: If not None, this is the behaviour in case of a zero predicate, (e.g. _z for zero fill, _x for ignore)
    """
    out = 'sv' + name
    if type:
        if suffix:
            out += '_' + TYPE_TO_SVE_SUFFIX[type]

    if pred_mode:
        out += '_' + pred_mode

    return out


# % is not supported
BIN_OP_TO_SVE = {
    ast.Add: 'svadd',
    ast.Mult: 'svmul',
    ast.Sub: 'svsub',
    ast.Div: 'svdiv',
    ast.BitXor: 'sveor',
    ast.BitAnd: 'svand',
    ast.BitOr: 'svorr',
    # Logical shifts
    ast.LShift: 'svlsl',
    ast.RShift: 'svlsr'
}

COMPARE_TO_SVE = {
    ast.Eq: 'svcmpeq',
    ast.NotEq: 'svcmpne',
    ast.Lt: 'svcmplt',
    ast.LtE: 'svcmple',
    ast.Gt: 'svcmpgt',
    ast.GtE: 'svcmpge',
    # Is, IsNot, In and NotIn not supported
}

BOOL_OP_TO_SVE = {ast.And: 'svand', ast.Or: 'svorr'}

# This is needed, because in SVE, the scalar is always at the end.
# If the scalar is on the LHS, we must switch LHS and RHS and also flip the sign.
FLIP_INEQUALITY = {
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt
}

# UAdd is ignored
UN_OP_TO_SVE = {ast.USub: 'svneg', ast.Not: 'svnot'}

# Used when a suffix for an SVE instruction is required.
TYPE_TO_SVE_SUFFIX = {
    int: 's32',
    float: 'f32',
    np.int8: 's8',
    np.int16: 's16',
    np.int32: 's32',
    np.int64: 's64',
    np.uint8: 'u8',
    np.uint16: 'u16',
    np.uint32: 'u32',
    np.uint64: 'u64',
    np.float16: 'f16',
    np.float32: 'f32',
    np.float64: 'f64',
    dace.int8: 's8',
    dace.int16: 's16',
    dace.int32: 's32',
    dace.int64: 's64',
    dace.uint8: 'u8',
    dace.uint16: 'u16',
    dace.uint32: 'u32',
    dace.uint64: 'u64',
    dace.float16: 'f16',
    dace.float32: 'f32',
    dace.float64: 'f64'
}

SVE_SUFFIX_TO_TYPE = dict((v, k) for k, v in TYPE_TO_SVE_SUFFIX.items())

TYPE_TO_SVE = {
    int: 'svint32_t',
    float: 'svfloat32_t',
    np.int8: 'svint8_t',
    np.int16: 'svint16_t',
    np.int32: 'svint32_t',
    np.int64: 'svint64_t',
    np.uint8: 'svuint8_t',
    np.uint16: 'svuint16_t',
    np.uint32: 'svuint32_t',
    np.uint64: 'svuint64_t',
    np.float16: 'svfloat16_t',
    np.float32: 'svfloat32_t',
    np.float64: 'svfloat64_t',
    dace.int8: 'svint8_t',
    dace.int16: 'svint16_t',
    dace.int32: 'svint32_t',
    dace.int64: 'svint64_t',
    dace.uint8: 'svuint8_t',
    dace.uint16: 'svuint16_t',
    dace.uint32: 'svuint32_t',
    dace.uint64: 'svuint64_t',
    dace.float16: 'svfloat16_t',
    dace.float32: 'svfloat32_t',
    dace.float64: 'svfloat64_t'
}

REDUCTION_TYPE_TO_SVE = {
    # Note: tree-based reduction for FP
    dace.dtypes.ReductionType.Sum: 'svaddv',
    dace.dtypes.ReductionType.Max: 'svmaxv',
    dace.dtypes.ReductionType.Min: 'svminv'
}

MATH_FUNCTION_TO_SVE = {'math.min': 'svmin', 'math.max': 'svmax', 'math.abs': 'svabs', 'math.sqrt': 'svsqrt'}

FUSED_OPERATION_TO_SVE = {'__svmad': 'svmad', '__svmla': 'svmla', '__svmsb': 'svmsb', '__svmls': 'svmls'}


def get_internal_symbols() -> dict:
    """
    Generates all internal symbols by crossing the internal function names with all possible type suffixes.
    Then defines the symbol with the corresponding return type (based on the suffix).
    """
    res = {}

    for func, type in itertools.product(FUSED_OPERATION_TO_SVE, TYPE_TO_SVE_SUFFIX):
        res[f'{func}_{TYPE_TO_SVE_SUFFIX[type.type if isinstance(type, dace.dtypes.typeclass) else type]}'] = dtypes.vector(
            type if isinstance(type, dtypes.typeclass) else dtypes.typeclass(type), SVE_LEN)
    return res


def is_sve_internal(name: str) -> bool:
    return name.startswith('__sv')


def internal_to_external(name: str) -> tuple:
    """
    Converts the internal symbol (e.g. __svmad_f32) into the
    external symbol (e.g. svmad_f32) and returns a tuple of
    (external symbol, dtype).
    """
    und = name.rfind('_')
    meth = name[:und]
    ext = name[und + 1:]
    if meth not in FUSED_OPERATION_TO_SVE:
        raise NotSupportedError('Unknown internal function')
    return (FUSED_OPERATION_TO_SVE[meth] + '_' + ext, SVE_SUFFIX_TO_TYPE[ext])


def get_base_type(type: dace.typeclass) -> dace.typeclass:
    """ Returns the underlying type for any dtype. """
    if isinstance(type, dtypes.vector):
        return type.vtype
    elif isinstance(type, dtypes.pointer):
        return type.base_type
    else:
        return type


def is_vector(type: dace.typeclass) -> bool:
    return isinstance(type, dtypes.vector)


def is_pointer(type: dace.typeclass) -> bool:
    return isinstance(type, dtypes.pointer)


def is_scalar(type: dace.typeclass) -> bool:
    return not is_vector(type) and not is_pointer(type)


def infer_ast(defined_symbols: collections.OrderedDict, *args) -> tuple:
    """ Returns the inferred types of the arguments, which must be AST nodes, as tuples. """
    return tuple([infer.infer_expr_type(t, defined_symbols) for t in args])


def only_scalars_involed(defined_symbols: collections.OrderedDict, *terms) -> bool:
    """ Takes AST nodes and returns whether only scalars are involved in the subtrees. """
    return all([is_scalar(infer_ast(defined_symbols, t)[0]) for t in terms])


def get_sve_scope(sdfg: dace.sdfg.SDFG, state: dace.sdfg.SDFGState, node: dace.sdfg.nodes.Node) -> dace.nodes.Map:
    while sdfg is not None:
        sdict = state.scope_dict()
        scope = sdict[node]
        while scope is not None:
            if scope.schedule == dace.ScheduleType.SVE_Map:
                return scope
            scope = sdict[scope]
        if sdfg.parent is not None:
            parent = sdfg.parent_sdfg
            state = sdfg.parent
            node = sdfg.parent_nsdfg_node
            if node.schedule == dace.ScheduleType.SVE_Map:
                return node
        else:
            parent = sdfg.parent
        sdfg = parent
    return None


def get_loop_predicate(sdfg: dace.sdfg.SDFG, dfg: dace.sdfg.SDFGState, node: dace.sdfg.nodes.Node) -> str:
    scope = get_sve_scope(sdfg, dfg, node)
    if scope is None:
        raise NotSupportedError('Not in an SVE scope')
    return '__pg_' + scope.params[-1]
