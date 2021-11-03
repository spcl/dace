# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
'''
Helps extracting the relevant information from MLIR for CodeGen of an MLIR tasklet
Can handle MLIR in generic form or in the supported dialect of pyMLIR
Requires pyMLIR to run
'''
try:
    import mlir
except (ModuleNotFoundError, NameError, ImportError):
    raise ImportError(
        'To use MLIR tasklets, please install the "pymlir" package.')

import dace
from typing import Union

# Only these types and the vector version of them are supported
TYPE_DICT = {
    "ui8": dace.uint8,
    "ui16": dace.uint16,
    "ui32": dace.uint32,
    "ui64": dace.uint64,
    "si8": dace.int8,
    "si16": dace.int16,
    "si32": dace.int32,
    "si64": dace.int64,
    "i8": dace.int8,
    "i16": dace.int16,
    "i32": dace.int32,
    "i64": dace.int64,
    "f16": dace.float16,
    "f32": dace.float32,
    "f64": dace.float64
}


def get_ast(code: str):
    return mlir.parse_string(code).modules[0]


def is_generic(ast: Union[mlir.astnodes.Module, mlir.astnodes.GenericModule]):
    return isinstance(ast, mlir.astnodes.GenericModule)


def get_entry_func(ast: Union[mlir.astnodes.Module,
                              mlir.astnodes.GenericModule],
                   is_generic: bool,
                   func_uid: str = None):
    # mlir_entry is a reserved keyword for the entry function. In order to allow for multiple MLIR tasklets we append a UID
    entry_func_name = "mlir_entry"
    if func_uid is not None:
        entry_func_name = entry_func_name + func_uid

    # Iterating over every function in the body of every region to check if ast contains exactly one entry and saving the entry function
    entry_func = None
    for body in ast.region.body:
        for op in body.body:
            func = op.op
            func_name = get_func_name(func, is_generic)

            if func_name == entry_func_name:
                if entry_func is not None:
                    raise SyntaxError(
                        "Multiple entry function in MLIR tasklet.")
                entry_func = func

    if entry_func is None:
        raise SyntaxError(
            'No entry function in MLIR tasklet, please make sure a "mlir_entry()" function is present.'
        )

    return entry_func


def get_func_name(func: Union[mlir.astnodes.Function,
                              mlir.astnodes.GenericModule], is_generic: bool):
    if is_generic:
        # In generic ast the name can be found in ast->module[0]->region->body[0]->body[]->op->attributes->values[0]->value
        # The consecutive .values ensure to read the name as a string
        return func.attributes.values[0].value.value.value
    # In dialect ast the name can be found in ast->module[0]->region->body[0]->body[]->op->name->value
    return func.name.value


def get_entry_args(entry_func: Union[mlir.astnodes.Function,
                                     mlir.astnodes.GenericModule],
                   is_generic: bool):
    ret = []

    if is_generic:
        # In generic ast the list of arguments can be found in ast->module[0]->region->body[0]->body[]->op->region->body[0]->label->arg_ids
        arg_names = entry_func.region.body[0].label.arg_ids
        arg_types = entry_func.region.body[0].label.arg_types

        for idx in range(len(arg_names)):
            arg_name = arg_names[idx].value
            arg_type = arg_types[idx]
            ret.append((arg_name, arg_type))
        return ret

    if entry_func.args is None:
        return []

    for arg in entry_func.args:
        arg_name = arg.name.value
        arg_type = arg.type
        ret.append((arg_name, arg_type))
    return ret


def get_entry_result_type(entry_func: Union[mlir.astnodes.Function,
                                            mlir.astnodes.GenericModule],
                          is_generic: bool):
    if is_generic:
        generic_result_list = entry_func.attributes.values[
            1].value.value.result_types
        # Only one return value allowed as we can not match multiple return values
        if len(generic_result_list) != 1:
            raise SyntaxError(
                'Entry function in MLIR tasklet must return exactly one value.')

        return generic_result_list[0]

    dialect_result = entry_func.result_types
    # Only one return value allowed as we can not match multiple return values
    if isinstance(dialect_result, list):
        raise SyntaxError(
            'Entry function in MLIR tasklet must return exactly one value.')

    return dialect_result


def get_dace_type(node: Union[mlir.astnodes.IntegerType,
                              mlir.astnodes.FloatType,
                              mlir.astnodes.VectorType]):
    if isinstance(node, mlir.astnodes.IntegerType) or isinstance(
            node, mlir.astnodes.FloatType):
        return TYPE_DICT[node.dump()]

    if isinstance(node, mlir.astnodes.VectorType):
        result_dim = node.dimensions[0]
        result_subtype = get_dace_type(node.element_type)
        return dace.vector(result_subtype, result_dim)
