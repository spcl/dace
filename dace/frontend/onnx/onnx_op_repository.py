import inspect
import ast
from typing import Callable, List, Union

import yaml
import pkg_resources

import dace
from dace.dtypes import paramdec
from dace.graph.nodes import Node


class InvalidONNXGraphError(Exception):
    """ A class of exceptions thrown when ONNX validation fails. """
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class InvalidONNXOpError(Exception):
    """ A class of exceptions thrown when ONNX op validation fails. """
    def __init__(self, message: str, op_type: str, inputs: List[str]):
        self.message = message
        self.op_type = op_type
        self.inputs = inputs

    def __str__(self):
        return "{} at op of type {} with inputs {}".format(
            self.message, self.op_type, self.inputs)


class ONNXOps(object):
    """ A management singleton for ONNX ops."""
    _ops = {}
    _signatures = yaml.load(pkg_resources.resource_stream(
        __name__, "op_signatures.yaml"),
                            Loader=yaml.SafeLoader)

    @staticmethod
    def get(name):
        """ Returns an implementation of a function. """
        if name not in ONNXOps._ops:
            raise ValueError("Unknown op {}".format(name))
        return ONNXOps._ops[name]


def _register_onnx_op(func: Callable[
    [dace.SDFG, dace.SDFGState, List[str], List[str],
     type(...)], None], op_name: str):
    """ Registers a sub-SDFG generator for a onnx op.
    """
    signature = ONNXOps._signatures[op_name]

    def op_func(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
                outputs: List[str], **attributes):
        # Wrapper to verify the op's inputs
        inputs = list(inputs)
        outputs = list(outputs)

        if not (signature["required"] <= len(inputs) <= len(
                signature["inputs"])):
            raise InvalidONNXOpError("too little or too many inputs", op_name,
                                     inputs)

        if len(outputs) != len(signature["outputs"]):
            raise InvalidONNXOpError("expected {} outputs, got {}".format(
                len(signature["outputs"]), len(outputs), op_name, inputs))

        instantiated_typeclasses = {}

        # Check the dtypes of the inputs
        for i, inp in enumerate(inputs):

            type_class = signature["inputs"][i][1]

            assert type_class in signature["types"]  # not a user error

            if type_class not in instantiated_typeclasses:
                allowed_types = {
                    getattr(dace, t)
                    for t in signature["types"][type_class]
                }

                if sdfg.arrays[inp].dtype not in allowed_types:
                    raise InvalidONNXOpError(
                        "expected input {} to have dtype in {}, but it has dtype {}"
                        .format(inp, allowed_types, sdfg.arrays[inp].dtype,
                                op_name, inputs))

                instantiated_typeclasses[type_class] = sdfg.arrays[inp].dtype
            else:
                if sdfg.arrays[inp].dtype != instantiated_typeclasses[
                        type_class]:
                    raise InvalidONNXOpError(
                        "expected input {} to have dtype {}, but it has dtype {}"
                        .format(inp, instantiated_typeclasses[type_class],
                                sdfg.arrays[inp].dtype, op_name, inputs))

        if "attributes" in signature:
            all_attributes = set(signature["required_attributes"]) | set(
                signature["attributes"])

            # check that all required attributes are present
            if len(
                    set(signature["required_attributes"]).difference(
                        attributes)) != 0:
                raise InvalidONNXOpError(
                    "No value was passed for required attributes {}".format(
                        signature["required_attributes"].difference(
                            attributes)), op_name, inputs)

            # Check that there are no unknown attributes
            if len(set(attributes).difference(all_attributes)) != 0:
                raise InvalidONNXOpError(
                    "Unknown attributes {}".format(
                        set(attributes).difference(all_attributes), ), op_name,
                    inputs)

            # add default values
            missing = {
                attr: default_value
                for attr, default_value in signature["attributes"].items()
                if attr not in attributes
            }
            attributes.update(missing)

        func(sdfg, state, inputs, outputs, **attributes)

    ONNXOps._ops[op_name] = op_func
    return op_func


def onnx_op(func: Callable[
    [dace.SDFG, dace.SDFGState, List[str], List[str],
     type(...)], None]):
    return _register_onnx_op(func, func.__name__)


def onnx_op_with_name(name):
    def wrapper(func: Callable[
        [dace.SDFG, dace.SDFGState, List[str], List[str],
         type(...)], None]):
        return _register_onnx_op(func, name)

    return wrapper


def _parse_annotation(annot):
    try:
        subscript = ast.parse(annot).body[0].value
        array_type = subscript.value.id

        if type(subscript.slice.value) is ast.Tuple:
            dims = [elt.id for elt in subscript.slice.value.elts]
        else:
            dims = subscript.slice.value.id

        return array_type, dims
    except (AttributeError, IndexError):
        raise SyntaxError("Could not parse type annotation {}".format(annot))


def _instantiate_annotation(variable_instantiations, annot):
    array_type, dims = _parse_annotation(annot)
    if type(dims) is list:
        return variable_instantiations[array_type][[
            variable_instantiations[dim] for dim in dims
        ]]
    else:
        return variable_instantiations[array_type][
            variable_instantiations[dims]]


def _instantiate_and_add_transient(sdfg, name, variable_instantiations, annot):
    array_type, dims = _parse_annotation(annot)
    return sdfg.add_transient(
        name,
        shape=[variable_instantiations[dim] for dim in dims]
        if type(dims) is list else variable_instantiations[dims],
        dtype=variable_instantiations[array_type])


def onnx_op_program(program):
    """Register a @dace.program annotated function"""

    op_name = program.__name__
    op_signature = ONNXOps._signatures[op_name]

    if len(op_signature["attributes"]) + len(
            op_signature["required_attributes"]) != 0:
        raise NotImplementedError(
            "Using @onnx_op_program with ops that have attributes is not supported"
        )

    def op(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
           outputs: List[str]):  # attributes not allowed

        program_signature = inspect.signature(program)
        program_annotations = program.__annotations__
        params = program_signature.parameters
        input_names = [name for name, type_str in op_signature["inputs"]]
        output_names = [name for name, type_str in op_signature["outputs"]]

        input_arrs = [sdfg.arrays[name] for name in inputs]

        # do type pattern matching for the inputs and outputs
        input_annotations = [params[name].annotation for name in input_names]
        output_annotations = [params[name].annotation for name in output_names]

        assert all(type(annot) is str for annot in input_annotations)
        assert all(type(annot) is str for annot in output_annotations)

        variable_instantiations = {}

        # instantiate variables with values from input
        for input_arr, annot in zip(input_arrs, input_annotations):
            array_type, dims = _parse_annotation(annot)

            if array_type in variable_instantiations:
                # variable has already been instantiated;
                # check that it matches
                assert variable_instantiations[array_type] is input_arr.dtype
            else:
                # instantiate the variable
                variable_instantiations[array_type] = input_arr.dtype

            if type(dims) is list:
                for dim, arr_dim in zip(dims, input_arr.shape):
                    if dim in variable_instantiations:
                        # variable has already been instantiated;
                        # check that it matches

                        # == works for both ints and symbols
                        assert variable_instantiations[dim] == arr_dim
                    else:
                        # instantiate the variable
                        variable_instantiations[dim] = arr_dim
            else:
                if dims in variable_instantiations:
                    # variable has already been instantiated;
                    # check that it matches
                    assert variable_instantiations[dims] == list(
                        input_arr.shape)
                else:
                    # instantiate the variable
                    variable_instantiations[dims] = list(input_arr.shape)

        new_params = [
            param.replace(annotation=_instantiate_annotation(
                variable_instantiations, param.annotation))
            for _, param in params.items()
        ]

        program.__signature__ = program_signature.replace(
            parameters=new_params)
        program.__annotations__ = {
            name: _instantiate_annotation(variable_instantiations, annot)
            for name, annot in program.__annotations__.items()
        }

        # add output arrays using instantiated types
        output_arrs = [
            _instantiate_and_add_transient(sdfg, name, variable_instantiations,
                                           annot)[1]
            for name, annot in zip(outputs, output_annotations)
        ]

        # overwrite dtypes incase they are used in the program
        overwrite_dtypes = {
            '__dtype_' + name: value
            for name, value in variable_instantiations.items()
            if type(value) is dace.dtypes.typeclass
        }
        dace_program = dace.parser.DaceProgram(
            program, (), {}, overwrite_globals=overwrite_dtypes)

        nsdfg = dace_program.to_sdfg()

        # reset the signature back to the parameterized one
        program.__signature__ = program_signature
        program.__annotations__ = program_annotations

        nsdfg_node = state.add_nested_sdfg(nsdfg, None, set(input_names),
                                           set(output_names))

        read_inputs = [state.add_read(inp) for inp in inputs]
        write_outputs = [state.add_write(outp) for outp in outputs]

        # connect inputs
        for input_conn, read_input, input_arr_name, input_arr in zip(
                input_names, read_inputs, inputs, input_arrs):
            state.add_edge(read_input,
                           None,
                           nsdfg_node,
                           input_conn,
                           memlet=dace.Memlet.from_array(
                               input_arr_name, input_arr))

        # connect outputs
        for output_conn, write_output, output_arr_name, output_arr in zip(
                output_names, write_outputs, outputs, output_arrs):
            state.add_edge(nsdfg_node,
                           output_conn,
                           write_output,
                           None,
                           memlet=dace.Memlet.from_array(
                               output_arr_name, output_arr))

    return _register_onnx_op(op, op_name)


# this import will register all the ops
import dace.frontend.onnx.onnx_op_impl
