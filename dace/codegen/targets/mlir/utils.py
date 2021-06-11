# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
'''
Helps extracting the relevant information from MLIR for CodeGen of an MLIR tasklet
'''
import dace

class MLIRUtils():

    # Only these types and the vector version of them are supported
    TYPE_DICT = {
        "i8": dace.int8,
        "i16": dace.int16,
        "i32": dace.int32,
        "i64": dace.int64,
        "f16": dace.float16,
        "f32": dace.float32,
        "f64": dace.float64
    }

    def __init__(self, code, func_uid=""):
        # pyMLIR needed to generate and read the AST
        try:
            self.mlir =  __import__('mlir')
        except (ModuleNotFoundError, NameError, ImportError):
            raise ImportError('To use MLIR tasklets, please install the "pymlir" package.')

        self.code = code

        # mlir_entry is a reserved keyword for the entry function. In order to allow for multiple MLIR tasklets we append a UID
        self.entry_func_name = "mlir_entry" + func_uid

        self.ast = self.mlir.parse_string(self.code)
        self.is_generic = isinstance(self.ast, self.mlir.astnodes.GenericModule)

        # Iterating over every function in the body to check if ast contains exactly one entry and saving the entry function
        self.entry_func = None
        for func in self.ast.body: 
            func_name = self.__get_func_name(func)

            if func_name == self.entry_func_name:
                if self.entry_func is not None:
                    raise SyntaxError("Multiple entry function in MLIR tasklet.")
                self.entry_func = func

        if self.entry_func is None:
            raise SyntaxError('No entry function in MLIR tasklet, please make sure a "mlir_entry()" function is present.')

    def __get_func_name(self, func):
        if self.is_generic:
            # In generic ast the name can be found in ast->body->func[]->attributes->values[0]->value
            # The consecutive .values ensure to read the name as a string
            return func.attributes.values[0].value.value.value 
        # In dialect ast the name can be found in ast->body->func[]->name->value
        return func.name.value
        
    def get_entry_args(self):
        ret = []

        if self.is_generic:
            # In generic ast the list of arguments can be found in ast->body->func[]->body[0]->label->arg_ids
            arg_names = self.entry_func.body[0].label.arg_ids
            arg_types = self.entry_func.body[0].label.arg_types

            for idx in range( len(arg_names) ):
                arg_name = arg_names[idx].value
                arg_type = arg_types[idx]
                ret.append( (arg_name, arg_type) )
            return ret

        for arg in self.entry_func.args:
            arg_name = arg.name.value
            arg_type = arg.type
            ret.append( (arg_name, arg_type) )
        return ret

    def get_entry_result_type(self):
        if self.is_generic:
            generic_result_list = self.entry_func.attributes.values[1].value.value.result_types
            # Only one return value allowed as we can not match multiple return values
            if len(generic_result_list) != 1:
                raise SyntaxError('Entry function in MLIR tasklet must return exactly one value.')

            return generic_result_list[0]
        
        dialect_result = self.entry_func.result_types
        # Only one return value allowed as we can not match multiple return values
        if isinstance(dialect_result, list):
            raise SyntaxError('Entry function in MLIR tasklet must return exactly one value.')
            
        return dialect_result

    def get_dace_type(self, node):
        if type(node) == self.mlir.astnodes.IntegerType:
            result_width = node.width.value
            return MLIRUtils.TYPE_DICT["i" + result_width]
            
        if type(node) == self.mlir.astnodes.FloatType:
            return MLIRUtils.TYPE_DICT[node.type.name]

        if type(node) == self.mlir.astnodes.VectorType:
            result_dim = node.dimensions[0]
            result_subtype = self.get_dace_type(node.element_type)
            return dace.vector(result_subtype, result_dim)
