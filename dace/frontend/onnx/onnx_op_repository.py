from typing import Callable, List, Union

import yaml
import pkg_resources

import dace
from dace.dtypes import paramdec
from dace.graph.nodes import Node


class OnnxOps(object):
    """ A management singleton for onnx ops."""
    _ops = {}
    _signatures = yaml.load(pkg_resources.resource_stream(
        __name__, "op_signatures.yaml"),
                            Loader=yaml.SafeLoader)

    @staticmethod
    def get(name):
        """ Returns an implementation of a function. """
        if name not in OnnxOps._ops:
            raise ValueError("Unknown op {}".format(name))
        return OnnxOps._ops[name]

def _register_onnx_op(func: Callable[
    [dace.SDFG, dace.SDFGState, List[str], List[str],
     type(...)], None], op_name: str):
    """ Registers a sub-SDFG generator for a onnx op.
    """
    signature = OnnxOps._signatures[op_name]

    def op_func(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
                outputs: List[str], **attributes):
        # TODO @orausch cleaner errors here
        # Wrapper to verify the op's inputs
        inputs = list(inputs)
        outputs = list(outputs)

        assert signature["required"] <= len(inputs) <= len(signature["inputs"])
        assert len(outputs) == len(signature["outputs"])

        instantiated_typeclasses = {}

        # Check the dtypes of the inputs
        for i, inp in enumerate(inputs):
            _, type_class = map(str.strip, signature["inputs"][i].split(":"))
            assert type_class in signature["types"]
            if type_class not in instantiated_typeclasses:
                allowed_types = {
                    getattr(dace, t)
                    for t in signature["types"][type_class]
                }
                assert sdfg.arrays[inp].dtype in allowed_types
                instantiated_typeclasses[type_class] = sdfg.arrays[inp].dtype
            else:
                assert sdfg.arrays[inp].dtype == instantiated_typeclasses[
                    type_class]

        if "attributes" in signature:
            # Check that there are no unknown/unsupported attributes
            assert len(set(attributes).difference(
                signature["attributes"])) == 0

            # add default values
            missing = {
                attr: default_value for attr, default_value in signature["attributes"].items()
                if attr not in attributes
            }
            attributes.update(missing)

        func(sdfg, state, inputs, outputs, **attributes)

    OnnxOps._ops[op_name] = op_func
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
