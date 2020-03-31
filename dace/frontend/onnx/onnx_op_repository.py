from typing import Callable, List, Union

import yaml
import pkg_resources

import dace
from dace.dtypes import paramdec
from dace.graph.nodes import Node


class InvalidONNXGraphError(Exception):
    """ A class of exceptions thrown when ONNX validation fails. """
    def __init__(self, message: str, graph):
        self.message = message
        self.graph = graph

    def __str__(self):
        return self.message


class InvalidONNXOpError(Exception):
    """ A class of exceptions thrown when ONNX op validation fails. """
    def __init__(self, message: str, graph, op_type: str, inputs: List[str]):
        self.message = message
        self.graph = graph
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
            _, type_class = map(str.strip, signature["inputs"][i].split(":"))

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
            # Check that there are no unknown/unsupported attributes
            if len(set(attributes).difference(signature["attributes"])) != 0:
                raise InvalidONNXOpError(
                    "Unsupported or Unknown attribute(s) {}".format(
                        set(attributes).difference(signature["attributes"]),
                        op_name, inputs))

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


# this import will register all the ops
import dace.frontend.onnx.onnx_op_impl
