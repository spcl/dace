from collections import OrderedDict
from copy import deepcopy
from itertools import chain, repeat

import numpy as np
import numba.cuda
import onnx
from onnx import numpy_helper

import dace
from dace.frontend.python.parser import infer_symbols_from_shapes
from dace.sdfg import SDFG, SDFGState
from dace.libraries.onnx.converters import convert_attribute_proto, onnx_tensor_type_to_typeclass, clean_onnx_name
from dace.libraries.onnx import get_onnx_node, has_onnx_node, ONNXParameterType
from dace.dtypes import AccessType, StorageType, AllocationLifetime
import dace.sdfg.nodes as nd
from dace.symbolic import pystr_to_symbolic


def _nested_HasField(obj, full_attr):
    """Performs a nested hasattr check, separating attr on dots."""
    attrs = full_attr.split(".")
    for attr in attrs:
        if obj.HasField(attr):
            obj = getattr(obj, attr)
        else:
            return False
    return True


class ONNXModel:
    """Loads an ONNX model into an SDFG."""
    def __init__(self, name, model: onnx.ModelProto, cuda=False):
        """
        Constructs a new ONNXImporter.
        :param name: the name for the SDFG.
        :param model: the model to import.
        :param cuda: if `True`, weights will be passed as cuda arrays.
        """

        graph: onnx.GraphProto = model.graph

        self.sdfg = SDFG(name)
        self.cuda = cuda
        self.state = self.sdfg.add_state()

        # Add all values to the SDFG, check for unsupported ops
        ##########################################

        self.value_infos = {}

        self.inputs = []
        self.outputs = []

        for value, is_input in chain(zip(graph.input, repeat(True)),
                                     zip(graph.output, repeat(False))):
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if is_input:
                self.inputs.append(value.name)
            else:
                self.outputs.append(value.name)

            self.value_infos[value.name] = value
            self._add_value_info(value)

        for value in graph.value_info:
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if value.name not in self.value_infos:
                self.value_infos[value.name] = value

        # add weights
        self.weights = {}
        for init in graph.initializer:
            self._add_constant_tensor(init)

        access_nodes = {}
        self._idx_to_node = []
        for i, node in enumerate(graph.node):
            if not has_onnx_node(node.op_type):
                raise ValueError("Unsupported ONNX operator: '{}'".format(
                    node.op_type))

            # extract the op attributes

            op_attributes = {
                attribute_proto.name: convert_attribute_proto(attribute_proto)
                for attribute_proto in node.attribute
            }

            if node.HasField("name"):
                node_name = clean_onnx_name(node.name)
            else:
                node_name = node.op_type + "_" + str(i)

            # construct the dace node
            op_node = get_onnx_node(node.op_type)(node_name, **op_attributes)
            self.state.add_node(op_node)
            self._idx_to_node.append(op_node)

            for param_idx, (name, is_input) in chain(
                    enumerate(zip(node.input, repeat(True))),
                    enumerate(zip(node.output, repeat(False)))):
                if clean_onnx_name(name) not in self.sdfg.arrays:
                    if name not in self.value_infos:
                        raise ValueError(
                            "Could not find array with name '{}'".format(name))
                    self._add_value_info(self.value_infos[name])

                # get the access node
                if name in access_nodes:
                    access = access_nodes[name]
                    self._update_access_type(access, is_input)
                else:
                    access = nd.AccessNode(
                        clean_onnx_name(name), AccessType.ReadOnly
                        if is_input else AccessType.WriteOnly)
                    self.state.add_node(access)
                    access_nodes[name] = access

                # get the connector name
                params = op_node.schema.inputs if is_input else op_node.schema.outputs
                params_len = len(params)
                if param_idx >= params_len:
                    # this is a variadic parameter. Then the last parameter of the parameter must be variadic.
                    if params[-1].param_type != ONNXParameterType.Variadic:
                        raise ValueError(
                            "Expected the last {i_or_o} parameter to be variadic,"
                            " since the {i_or_o} with idx {param_idx} has more parameters than the schema ({params_len})"
                            .format(i_or_o="input" if is_input else "output",
                                    param_idx=param_idx,
                                    params_len=params_len))
                    conn_name = params[-1].name + "__" + str(param_idx -
                                                             params_len + 1)
                elif params[param_idx].param_type == ONNXParameterType.Variadic:
                    # this is a variadic parameter, and it is within the range of params, so it must be the first
                    # instance of a variadic parameter
                    conn_name = params[param_idx].name + "__0"
                else:
                    conn_name = params[param_idx].name

                data_desc = self.sdfg.arrays[clean_onnx_name(name)]

                # add the connector if required, and add an edge
                if is_input:
                    if conn_name not in op_node.in_connectors:
                        op_node.add_in_connector(conn_name)
                    self.state.add_edge(
                        access, None, op_node, conn_name,
                        dace.Memlet.from_array(clean_onnx_name(name),
                                               data_desc))
                else:
                    if conn_name not in op_node.out_connectors:
                        op_node.add_out_connector(conn_name)

                    self.state.add_edge(
                        op_node, conn_name, access, None,
                        dace.Memlet.from_array(clean_onnx_name(name),
                                               data_desc))

        if self.cuda:
            self.sdfg.apply_strict_transformations()
            self.sdfg.apply_gpu_transformations()
            self.sdfg.apply_strict_transformations()

            # set all gpu transients to be persistent
            for _, _, arr in self.sdfg.arrays_recursive():
                if arr.transient and arr.storage == StorageType.GPU_Global:
                    arr.lifetime = AllocationLifetime.Persistent

    @staticmethod
    def _update_access_type(node: dace.nodes.AccessNode, is_input: bool):
        if node.access == AccessType.ReadOnly and not is_input:
            node.access = AccessType.ReadWrite
        elif node.access == AccessType.WriteOnly and is_input:
            node.access = AccessType.ReadWrite

    def _add_constant_tensor(self, tensor: onnx.TensorProto):
        if not tensor.HasField("name"):
            raise ValueError("Got tensor without name")

        if not tensor.HasField("data_type"):
            raise ValueError("Initializer tensor '{}' has no type".format(
                tensor.name))

        name = clean_onnx_name(tensor.name)

        dtype = onnx_tensor_type_to_typeclass(tensor.data_type)

        if len(tensor.dims) == 0:
            # this is a scalar
            self.sdfg.add_scalar(name, dtype)
        else:
            dims = [d for d in tensor.dims]
            if name not in self.sdfg.arrays:
                self.sdfg.add_array(name, dims, dtype)
            else:
                existing_arr = self.sdfg.arrays[name]
                if existing_arr.dtype != dtype:
                    raise ValueError(
                        "Invalid ONNX model; found two values with name '{}', but different dtypes ({} and {})"
                        .format(name, existing_arr.dtype, dtype))
                if tuple(existing_arr.shape) != tuple(dims):
                    raise ValueError(
                        "Invalid ONNX model; found two values with name '{}', but different dimensions ({} and {})"
                        .format(name, existing_arr.shape, dims))

        self.weights[tensor.name] = numpy_helper.to_array(tensor)

    def _add_value_info(self, value_info: onnx.ValueInfoProto):
        if not value_info.HasField("name"):
            raise ValueError("Got value without name")

        name = value_info.name

        if not _nested_HasField(value_info, "type.tensor_type.shape"):
            raise ValueError(
                "Value '{}' does not have a shape in this graph."
                " Please run shape inference before importing.".format(name))

        tensor_type = value_info.type.tensor_type

        if not tensor_type.HasField("elem_type"):
            raise ValueError(
                "Value '{}' does not have a type in this graph."
                " Please run type inference before importing.".format(name))

        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                parsed = pystr_to_symbolic(d.dim_param)

                for sym in parsed.free_symbols:
                    if clean_onnx_name(str(sym)) not in self.sdfg.symbols:
                        self.sdfg.add_symbol(clean_onnx_name(str(sym)),
                                             stype=int)
                    parsed = parsed.subs(sym,
                                         dace.symbol(clean_onnx_name(str(sym))))

                shape.append(parsed)
            else:
                raise ValueError(
                    "Value '{}' does not have a shape in this graph."
                    " Please run shape inference before importing.".format(
                        name))
        transient = name not in self.inputs and name not in self.outputs
        if len(shape) == 0:
            self.sdfg.add_scalar(clean_onnx_name(name),
                                 dtype=onnx_tensor_type_to_typeclass(
                                     tensor_type.elem_type),
                                 transient=transient)
        else:
            self.sdfg.add_array(clean_onnx_name(name),
                                shape=shape,
                                dtype=onnx_tensor_type_to_typeclass(
                                    tensor_type.elem_type),
                                transient=transient)

    def __call__(self, *args, **inputs):
        sdfg = deepcopy(self.sdfg)

        # convert the positional args to kwargs
        if len(args) > len(self.inputs):
            raise ValueError("Expected {} arguments, got {}".format(
                len(self.inputs), len(args)))

        inputs.update(dict(zip(self.inputs, args)))

        # check that there are no missing inputs
        if len(set(self.inputs).difference(inputs)) != 0:
            raise ValueError("Missing inputs {}".format(", ".join(
                set(self.inputs).difference(inputs))))

        # check that there are no unknown inputs
        # NOTE symbols can only be passed as kwargs
        if len(
                set(inputs).difference(self.inputs).difference(
                    sdfg.free_symbols)) != 0:
            raise ValueError("Unknown inputs {}".format(", ".join(
                set(inputs).difference(self.inputs))))

        clean_inputs = {}
        for input, arr in inputs.items():
            if input in sdfg.free_symbols:
                clean_inputs[input] = arr
            else:
                clean_inputs[clean_onnx_name(input)] = arr

        # add the weights
        params = {}
        for name, arr in self.weights.items():
            if len(arr.shape) == 0:
                params[clean_onnx_name(name)] = arr[()]
            else:
                if self.cuda:
                    clean_name = clean_onnx_name(name)
                    sdfg.arrays[clean_name].storage = StorageType.GPU_Global
                    params[clean_name] = numba.cuda.to_device(arr)
                else:
                    params[clean_onnx_name(name)] = arr.copy()

        inferred_symbols = infer_symbols_from_shapes(sdfg, {
            **clean_inputs,
            **params
        })
        # TODO @orausch if this is removed the SDFG complains
        # TypeError: Type mismatch for argument ONNX_unk__493: expected scalar type, got <class 'sympy.core.numbers.Integer'>
        # fix this better
        inferred_symbols = {k: int(v) for k, v in inferred_symbols.items()}

        def eval_dim(dim):
            for sym in dim.free_symbols:
                dim = dim.subs(sym, inferred_symbols[sym.name])
            return dim

        outputs = OrderedDict()
        # create numpy arrays for the outputs
        for output in self.outputs:
            clean_name = clean_onnx_name(output)
            arr = sdfg.arrays[clean_name]

            # TODO @orausch add error handling for evalf
            shape = [
                eval_dim(d) if type(d) is dace.symbol else d for d in arr.shape
            ]
            outputs[clean_name] = np.empty(shape,
                                           dtype=arr.dtype.as_numpy_dtype())

        sdfg.expand_library_nodes()
        #sdfg.apply_strict_transformations()

        sdfg(**clean_inputs, **params, **outputs, **inferred_symbols)

        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return tuple(outputs.values())
