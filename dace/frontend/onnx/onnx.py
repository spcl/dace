import warnings

import numpy as np
import onnx
from onnx import numpy_helper
from google.protobuf.json_format import MessageToDict

import dace
from dace import SDFG, Memlet
from dace.frontend.onnx.onnx_op_repository import OnnxOps


class AttributeProtoConverter:
    """Converts an AttributeProto to the relevant python object.

       Supported AttributeTypes: FLOAT, INT, STRING, FLOATS, INTS, STRINGS"""
    def __init__(self):
        self.inv_dtype_map = {
            v: k
            for k, v in onnx.AttributeProto.AttributeType.items()
        }

    def __call__(self, attribute: onnx.AttributeProto):
        dtype = attribute.type
        if dtype not in self.inv_dtype_map:
            # shouldn't happen, but it doesn't hurt to check
            raise ValueError(
                "Model contains unsupported attribute type {}".format(dtype))

        dtype_str = self.inv_dtype_map[dtype]

        if not hasattr(self, "_conv_" + dtype_str):
            raise ValueError(
                "Model contains unsupported attribute type {}".format(
                    self.inv_dtype_map[dtype]))
        converter = getattr(self, "_conv_" + dtype_str)

        return converter(attribute)

    @staticmethod
    def _conv_FLOAT(attribute: onnx.AttributeProto):
        return attribute.f

    @staticmethod
    def _conv_INT(attribute: onnx.AttributeProto):
        return attribute.i

    @staticmethod
    def _conv_STRING(attribute: onnx.AttributeProto):
        return attribute.s

    @staticmethod
    def _conv_FLOATS(attribute: onnx.AttributeProto):
        return list(attribute.floats)

    @staticmethod
    def _conv_INTS(attribute: onnx.AttributeProto):
        return list(attribute.ints)

    @staticmethod
    def _conv_STRINGS(attribute: onnx.AttributeProto):
        return list(attribute.strings)


_attribute_converter = AttributeProtoConverter()


def _get_attributes(node: onnx.NodeProto) -> dict:
    """Extracts the attributes of a NodeProto to a python dict."""
    return {attr.name: _attribute_converter(attr) for attr in node.attribute}


class OnnxTensorDtypeToDaceConverter:
    """Converts the onnx TensorProto.DataType dtype ints to dace dtypes"""
    def __init__(self):
        # STRING is unsupported for now
        str_to_dace_map = {
            "FLOAT": dace.float32,
            "DOUBLE": dace.float64,
            "UINT8": dace.uint8,
            "INT8": dace.int8,
            "UINT16": dace.uint16,
            "INT16": dace.int16,
            "INT32": dace.int32,
            "INT64": dace.int64,
            "BOOL": dace.bool,
        }
        self.inv_dtype_map = {
            v: k
            for k, v in onnx.TensorProto.DataType.items()
        }
        self.map = {
            k: str_to_dace_map[v]
            for k, v in self.inv_dtype_map.items() if v in str_to_dace_map
        }

    def __getitem__(self, dtype: int):
        if dtype not in self.map:
            if dtype not in self.inv_dtype_map:
                # shouldn't happen, but it doesn't hurt to check
                raise ValueError(
                    "Model contains unsupported datatype {}".format(dtype))
            raise ValueError("Model contains unsupported datatype {}".format(
                self.inv_dtype_map[dtype]))
        return self.map[dtype]


_onnx_dtype_to_dace = OnnxTensorDtypeToDaceConverter()


def _nested_hasattr(obj, full_attr):
    """Performs a nested hasattr check, separating attr on dots."""
    attrs = full_attr.split(".")
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return False
    return True


def _add_value(sdfg: SDFG,
               value: onnx.ValueInfoProto,
               name: str,
               shape=None,
               dtype=None) -> tuple:
    """Add a onnx value to the SDFG. """
    if shape is None and not _nested_hasattr(value, "type.tensor_type.shape"):
        raise ValueError("Shape not provided for value {}".format(value.name))
    else:
        # read from the value
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    if dtype is None and not _nested_hasattr(value,
                                             "type.tensor_type.elem_type"):
        raise ValueError("dtype not provided for value {}".format(value.name))
    else:
        # read from the value
        dtype = _onnx_dtype_to_dace[value.type.tensor_type.elem_type]

    return sdfg.add_array(name, shape, dtype)


def _add_param_tensor(sdfg: SDFG, tensor: onnx.TensorProto,
                      name: str) -> tuple:
    """Add a onnx parameter tensor to the SDFG. """
    if hasattr(tensor, "dims"):
        # read from the value
        shape = list(tensor.dims)
    else:
        raise ValueError("Shape not provided for tensor {}".format(
            tensor.name))

    if hasattr(tensor, "data_type"):
        # read from the value
        dtype = _onnx_dtype_to_dace[tensor.data_type]
    else:
        raise ValueError("dtype not provided for tensor {}".format(
            tensor.name))

    return sdfg.add_array(name, shape, dtype)


class OnnxModel:
    def __init__(self, model: onnx.ModelProto):
        """Loads an onnx model
        """
        onnx.checker.check_model(model)

        # eventually (once we have more ops) we should check model.opset_import to conform to the onnx spec

        if model.ir_version != 4:
            raise ValueError("Unsupported model IR version {}".format(
                model.ir_version))  # for now

        graph: onnx.GraphProto = model.graph

        self.sdfg = SDFG("onnx")
        self.last_state = self.sdfg.add_state('init', is_start_state=True)
        # note that this set doesn't include the _IN_ prefix
        self.global_inputs = []
        # Add inputs
        for input in graph.input:
            # these should have shapes on them. If they don't (not sure if that's possible), this will fail.
            # TODO @orausch check how onnx handles variable size batches (is the first dim the batch dim?)
            self.global_inputs.append(input.name)
            _add_value(self.sdfg, input, self._get_trans_name(input.name))

        self.params = dict()
        for param in graph.initializer:
            if param.name in self.global_inputs:
                warnings.warn(
                    "Loaded model contains default values for some inputs, but this is currently not supported"
                )
            self.params[param.name] = numpy_helper.to_array(param)
            _add_param_tensor(self.sdfg, param,
                              self._get_trans_name(param.name))

        # add the ops
        op: onnx.NodeProto
        for i, op in enumerate(graph.node):
            inputs = [self._get_trans_name(inp) for inp in op.input]
            outputs = [self._get_trans_name(outp) for outp in op.output]

            attributes = _get_attributes(op)

            self._add_state(label='_read_outputs')
            # add the op to the graph
            OnnxOps.get(op.op_type)(self.sdfg, self.last_state, inputs,
                                    outputs, **attributes)

        # Add outputs
        self._add_state(label='_read_outputs')
        self.global_outputs = dict()
        for output in graph.output:
            trans_name = self._get_trans_name(output.name)
            trans_array = self.sdfg.arrays[trans_name]

            out_name = "_OUT_" + self._clean_array_name(output.name)
            _, output_array = self.sdfg.add_array(out_name, trans_array.shape,
                                                  trans_array.dtype)
            self.global_outputs[out_name] = output_array
            read = self.last_state.add_read(trans_name)
            write = self.last_state.add_write(out_name)
            self.last_state.add_edge(read,
                                     None,
                                     write,
                                     None,
                                     memlet=Memlet.from_array(
                                         trans_name, trans_array))

        self.sdfg.apply_strict_transformations()

    def _add_state(self, state=None, label=None):
        if state is None:
            state = self.sdfg.add_state(label)
        else:
            self.sdfg.add_node(state)

        if self.last_state is not None:
            self.sdfg.add_edge(self.last_state, state, dace.InterstateEdge())
        self.last_state = state
        return state

    def _get_trans_name(self, name):
        if name in self.global_inputs:
            return "_IN_" + self._clean_array_name(name)
        elif name in self.params:
            return "_PARAM_" + self._clean_array_name(name)
        else:
            return "_TRANS_" + self._clean_array_name(name)

    def __call__(self, *args, **kwargs):

        # convert the positional args to kwargs
        if len(args) > len(self.global_inputs):
            raise ValueError("Expected {} arguments, got {}".format(
                len(self.global_inputs), len(args)))

        kwargs.update(dict(zip(self.global_inputs, args)))

        # check that there are no missing inputs
        if len(set(self.global_inputs).difference(kwargs)) != 0:
            raise ValueError("Missing inputs {}".format(", ".join(
                set(self.global_inputs).difference(kwargs))))

        # check that there are no unknown inputs
        if len(set(kwargs).difference(self.global_inputs)) != 0:
            raise ValueError("Unknown inputs {}".format(", ".join(
                set(kwargs).difference(self.global_inputs))))

        kwargs = {
            self._get_trans_name(input): arr
            for input, arr in kwargs.items()
        }
        # grab the parameters
        params = {
            self._get_trans_name(name): arr
            for name, arr in self.params.items()
        }
        # create numpy arrays for the outputs
        outputs = {
            output: np.zeros(shape=arr.shape,
                             dtype=getattr(np, arr.dtype.to_string()))
            for output, arr in self.global_outputs.items()
        }

        self.sdfg(**kwargs, **params, **outputs)

        if len(outputs) == 1:
            return next(iter(outputs.values()))

        # TODO @orausch change this to make it respect the order of the outputs from onnx
        return tuple(outputs.values())

    @staticmethod
    def _clean_array_name(name: str):
        """Modifies a onnx array name that is potentially invalid in dace
           to make it valid"""
        return name.replace(".", "__")
