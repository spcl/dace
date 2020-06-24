# TODO do checking on if type exists for non required fields (maybe automate this?)
from collections import Iterable
from itertools import chain, repeat, count
from functools import reduce

import onnx

import dace
import dace.sdfg.nodes as nd
from dace import SDFG, SDFGState, ScheduleType
from dace.properties import make_properties, Property, ListProperty
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.libraries.onnx import ONNXRuntime
from dace.libraries.onnx.converters import ONNX_DTYPES_TO_DACE_TYPE_CLASS
from dace.libraries.onnx.schema import ONNXSchema, ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXParameterType, ONNXAttribute
from dace.sdfg import InvalidSDFGNodeError


def get_position(schema: ONNXSchema, is_input: bool, parameter_name: str):
    """Get the position that the parameter has in the onnx op"""
    if "__" in parameter_name:
        parameter_name, variadic_number = parse_variadic_param(parameter_name)
    else:
        variadic_number = None

    matches = [
        (i, param)
        for i, param in enumerate(schema.inputs if is_input else schema.outputs)
        if param.name == parameter_name
    ]
    if len(matches) != 1:
        raise ValueError(
            "Error in schema: found more or less than one parameter with name {}"
            .format(parameter_name))

    index, param = matches[0]

    if variadic_number is not None and param.param_type != ONNXParameterType.Variadic:
        raise ValueError(
            "Got variadic index for non variadic parameter {}".format(
                parameter_name))

    if variadic_number is None and param.param_type == ONNXParameterType.Variadic:
        raise ValueError(
            "Did not get variadic index for variadic parameter {}. "
            "Specify a variadic index by renaming the parameter to {}__i, where i is a number"
            .format(parameter_name, parameter_name))

    if variadic_number is not None:
        return variadic_number + index
    else:
        return index


def get_missing_arguments_message(function_name, missing_arguments,
                                  argument_type):
    names = list(map(lambda x: "'" + x + "'", missing_arguments))

    if len(missing_arguments) == 1:
        arglist = names[0]
    else:
        arglist = ", ".join(names[:-1]) + ", and " + names[-1]

    return "{function_name} missing {num_missing} required {argument_type}{s}: {arglist}".format(
        function_name=function_name,
        num_missing=len(missing_arguments),
        argument_type=argument_type,
        s='' if len(missing_arguments) == 1 else 's',
        arglist=arglist)


def parse_variadic_param(param):
    split = param.split('__')
    if len(split) != 2:
        raise ValueError(
            "Unable to parse variadic parameter '{}'".format(param))
    name = split[0]
    number = split[1]

    if number[0] == '0' and len(number) > 1:
        raise ValueError(
            "Variadic parameters must not be numbered with leading zeroes, got: '{}'"
            .format(number))

    number = int(number)
    if number < 0:
        raise ValueError(
            "Variadic parameters numberings must be greater than zero, got: '{}'"
            .format(number))
    return name, number


# this counter is used to get unique names for the Node protos
GLOBAL_COUNTER = count()


def _gen_attr_init_code(node_proto: str, attr: ONNXAttribute, value) -> str:
    """ Get the code to setup an attribute on an onnx::NodeProto
        :param node_proto: the variable name of the node protobuf
        :param attr: the attribute to setup
    """
    if value is None:
        return ""

    def assert_type(val, expected_type):
        if not isinstance(val, expected_type):
            raise ValueError(
                "Expected value of attribute '{}' to have type {}, got {} (type {})"
                .format(attr.name, expected_type, val, type(val)))

    init_code = """{{
    // Setup attribute {name}
    onnx::AttributeProto* attribute = {node_proto}.add_attribute();
    attribute->set_name("{name}");
    """.format(name=attr.name, node_proto=node_proto)

    attr_type_to_type = {
        ONNXAttributeType.Int: int,
        ONNXAttributeType.Float: float,
        ONNXAttributeType.String: str,  # TODO @orausch: change to bytes
        ONNXAttributeType.Ints: int,
        ONNXAttributeType.Floats: float,
        ONNXAttributeType.Strings: str,
    }
    type_to_str = {
        ONNXAttributeType.Int: "int",
        ONNXAttributeType.Float: "float",
        ONNXAttributeType.String: "string",
        ONNXAttributeType.Ints: "int",
        ONNXAttributeType.Floats: "float",
        ONNXAttributeType.Strings: "string",
    }

    if attr.type in [
            ONNXAttributeType.Int, ONNXAttributeType.Float,
            ONNXAttributeType.String
    ]:
        assert_type(value, attr_type_to_type[attr.type])
        init_code += """
        attribute->set_type(onnx::AttributeProto::{type_str});
        attribute->set_{short_type}({value});
        """.format(type_str=type_to_str[attr.type].upper(),
                   short_type=type_to_str[attr.type][0],
                   value='"{}"'.format(value)
                   if attr.type == ONNXAttributeType.String else value)
    elif attr.type in [
            ONNXAttributeType.Ints, ONNXAttributeType.Floats,
            ONNXAttributeType.Strings
    ]:
        init_code += "attribute->set_type(onnx::AttributeProto::{}S);\n".format(
            type_to_str[attr.type].upper())

        if not isinstance(value, Iterable):
            raise ValueError(
                "Expected iterable value for attribute '{}', got {}".format(
                    attr.name, value))

        for i in value:
            assert_type(i, attr_type_to_type[attr.type])
            init_code += "attribute->add_{type_str}s({value});\n".format(
                type_str_decl='std::string' if attr.type
                == ONNXAttributeType.String else type_to_str[attr.type],
                type_str=type_to_str[attr.type],
                value='"{}"'.format(i)
                if attr.type == ONNXAttributeType.String else i)
    else:
        raise NotImplementedError(
            "Got unsupported attribute type {} for '{}'".format(
                attr.dtype, attr.name))
    init_code += "}\n"
    return init_code


class ONNXOp(nd.LibraryNode):
    """ Abstract superclass for all ONNX ops"""

    # Global properties
    # these two are filled out in the generated constructor
    implementations = {}
    default_implementation = None

    # Object fields
    schema = Property(dtype=ONNXSchema,
                      desc="The operator's ONNX OpSchema",
                      allow_none=True)

    def validate(self, sdfg: SDFG, state: SDFGState):
        # TODO @orausch change errors to InvalidSDFGNodeError
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)

        # check that we have all required in_edges
        ##########################################
        required_inputs = {
            inp.name
            for inp in self.schema.inputs
            if inp.param_type == ONNXParameterType.Single
        }
        passed_inputs = {
            inp.dst_conn
            for inp in in_edges if '__' not in inp.dst_conn
        }  # we will test variadic inputs separately
        known_inputs = {inp.name for inp in self.schema.inputs}

        missing_inputs = required_inputs.difference(passed_inputs)
        if len(missing_inputs) > 0:
            raise ValueError(
                get_missing_arguments_message(self.schema.name, missing_inputs,
                                              "input"))

        # check that we have all required out_edges
        ##########################################
        required_outputs = {
            outp.name
            for outp in self.schema.outputs
            if outp.param_type == ONNXParameterType.Single
        }
        passed_outputs = {
            outp.src_conn
            for outp in out_edges if '__' not in outp.src_conn
        }  # we will test variadic inputs separately
        known_outputs = {outp.name for outp in self.schema.outputs}

        missing_outputs = required_outputs.difference(passed_outputs)
        if len(missing_outputs) > 0:
            raise ValueError(
                get_missing_arguments_message(self.schema.name, missing_outputs,
                                              "output"))

        # check that we have no unknown in edges
        ##########################################
        unknown_inputs = passed_inputs.difference(known_inputs)
        if len(unknown_inputs) > 0:
            raise TypeError("Got an unexpected argument '{}'".format(
                list(unknown_inputs)[0]))

        # check that we have no unknown out edges
        ##########################################
        unknown_outputs = passed_outputs.difference(known_outputs)
        if len(unknown_outputs) > 0:
            raise TypeError("Got an unexpected argument '{}'".format(
                list(unknown_outputs)[0]))

        # check variadic params
        ##########################################
        variadic_inputs = {
            inp.name
            for inp in self.schema.inputs
            if inp.param_type == ONNXParameterType.Variadic
        }
        passed_variadic_inputs = {
            edge.dst_conn
            for edge in in_edges if '__' in edge.dst_conn
        }

        seen_variadic_numbers = set()
        for param in passed_variadic_inputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_inputs:
                raise ValueError(
                    "Got an unexpected variadic argument '{}'".format(param))
            if number in seen_variadic_numbers:
                raise ValueError(
                    "Got two variadic inputs with index {}, expected at most one"
                    .format(number))
            seen_variadic_numbers.add(number)

        # check that we have seen every number
        for i in range(len(seen_variadic_numbers)):
            if i not in seen_variadic_numbers:
                raise ValueError(
                    "Since {} variadic inputs were passed, expected variadic parameter with number {}"
                    .format(len(seen_variadic_numbers), i))

        variadic_outputs = {
            outp.name
            for outp in self.schema.outputs
            if outp.param_type == ONNXParameterType.Variadic
        }
        passed_variadic_outputs = {
            edge.src_conn
            for edge in out_edges if '__' in edge.src_conn
        }
        seen_variadic_numbers = set()
        for param in passed_variadic_outputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_outputs:
                raise ValueError(
                    "Got an unexpected variadic argument '{}'".format(param))
            if number in seen_variadic_numbers:
                raise ValueError(
                    "Got two variadic outputs with index {}, expected at most one"
                    .format(number))
            seen_variadic_numbers.add(number)

        # check that we have seen every number
        for i in range(len(seen_variadic_numbers)):
            if i not in seen_variadic_numbers:
                raise ValueError(
                    "Since {} variadic outputs were passed, expected variadic parameter with number {}"
                    .format(len(seen_variadic_numbers), i))

        # check that type params solve
        ##########################################

        assigned_params = {}
        for edge, is_input in chain(zip(in_edges, repeat(True)),
                                    zip(out_edges, repeat(False))):
            conn_name = edge.dst_conn if is_input else edge.src_conn

            if '__' in conn_name:
                parsed_name, number = parse_variadic_param(conn_name)
            else:
                parsed_name = conn_name

            matching = [
                inp for inp in (
                    self.schema.inputs if is_input else self.schema.outputs)
                if inp.name == parsed_name
            ]

            if len(matching) != 1:
                raise ValueError(
                    "Expected to find one {} parameter in schema with name '{}', but found {}"
                    .format("input" if is_input else "output", parsed_name,
                            len(matching)))
            matched = matching[0]

            if '__' in conn_name and matched.param_type != ONNXParameterType.Variadic:
                raise ValueError(
                    "Got variadic argument '{}' for non-variadic parameter '{}'."
                    " Ensure that non-variadic args do not contain '__'".format(conn_name, matched.name))

            if '__' not in conn_name and matched.param_type == ONNXParameterType.Variadic:
                raise ValueError(
                    "Expected variadic argument for variadic parameter '{}', got '{}'. Use '{}__i' as the connector"
                    " name, where i is the desired index of the variadic parameter.".format(matched.name, conn_name, conn_name))


            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            if matched.param_type == ONNXParameterType.Variadic and not matched.homogeneous:
                # non homogeneous parameters don't need to be consistent
                pass
            elif matched.type_str in assigned_params and assigned_params[
                    matched.type_str] != edge_dtype:
                raise ValueError(
                    "Could not solve type constraints;"
                    " excepted type '{expected}' for {param_type} '{conn_name}', got type '{actual}'"
                    .format(expected=assigned_params[matched.type_str],
                            param_type="input" if is_input else "output",
                            conn_name=matched.name,
                            actual=edge_dtype))

            # otherwise, matched.type_str was not assigned a type yet: try to assign it
            cons = self.schema.type_constraints[matched.type_str]
            if edge_dtype not in cons.types:
                raise ValueError(
                    "Expected type in '{possible}' for {param_type} '{conn_name}', got type '{actual}'"
                    .format(possible=cons.types,
                            param_type="input" if is_input else "output",
                            conn_name=matched.name,
                            actual=edge_dtype))
            assigned_params[matched.type_str] = edge_dtype

        # check that we have all required attributes
        ##########################################
        required_attrs = {
            name
            for name, attr in dace_schema.attributes.items() if attr.required
        }
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(
                    "Expected value for required attribute '{}', got None".
                    format(attr))

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        # Extract input and output array views (as generated by memlets)
        inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
        # Generate the appropriate code
        # Replace this node with a C++ tasklet
        unique_id = "{}_{}".format(node.schema.name, next(GLOBAL_COUNTER))
        if "OrtKernelSession" not in sdfg.global_code['frame'].as_string:
            sdfg.append_global_code("""
            // Start global ORT setup
            const OrtApi* __ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

            // helper function to check for status
            void __ort_check_status(OrtStatus* status)
            {
                if (status != NULL) {
                    const char* msg = __ort_api->GetErrorMessage(status);
                    fprintf(stderr, "%s\\n", msg);
                    __ort_api->ReleaseStatus(status);
                    exit(1);
                }
            }
            OrtEnv* __ort_env;
            OrtKernelSession* __ort_session;
            OrtSessionOptions* __ort_session_options;

            // TODO check what this does. Is it fine to use the CpuMemoryInfo for CUDA?
            OrtMemoryInfo* __ort_mem_info;

            // End global ORT setup
            """)

            sdfg.append_init_code("""
            __ort_check_status(__ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &__ort_mem_info)); 
            __ort_check_status(__ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dace_graph", &__ort_env));
            __ort_check_status(__ort_api->CreateSessionOptions(&__ort_session_options));
            __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CPU(__ort_session_options, /*use_arena=*/0));
            __ort_check_status(__ort_api->CreateKernelSession(__ort_session_options, &__ort_session));
            """)

            session_cleanup_code = """
            __ort_api->ReleaseMemoryInfo(__ort_mem_info);
            __ort_api->ReleaseKernelSession(__ort_session);
            __ort_api->ReleaseSessionOptions(__ort_session_options);
            __ort_api->ReleaseEnv(__ort_env);
            """
            sdfg.prepend_exit_code(session_cleanup_code)

        if node.schedule == ScheduleType.GPU_Device and "OrtSessionOptionsAppendExecutionProvider_CUDA" not in sdfg.global_code[
                'frame'].as_string:
            # the first time we expand a GPU node, we need to add the execution provider (only once)
            sdfg.append_init_code("""
            __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CUDA(__ort_session_options, /*device=*/0));
            """)

        sdfg.append_global_code(
            "OrtExecutableKernelContext *__ort_context_{};\n".format(unique_id))

        sdfg.append_init_code("""
        {{
        // Setup for context_{id}
        onnx::NodeProto proto;
        proto.set_op_type("{name}");
        proto.set_name("{id}");
        std::unordered_map<std::string, onnx::TypeProto> type_map;
        """.format(id=unique_id, name=node.schema.name))

        tasklet_setup_code = ""
        tasklet_code = ""
        tasklet_cleanup_code = ""

        reversed_onnx_dtype_map = {
            v: k
            for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()
        }
        for edge, is_input in chain(zip(state.in_edges(node), repeat(True)),
                                    zip(state.out_edges(node), repeat(False))):
            parameter_name = edge.dst_conn if is_input else edge.src_conn
            input_output_string = "input" if is_input else "output"
            memlet = edge.data
            arr = sdfg.arrays[memlet.data]
            sdfg.append_init_code("""
            {{
                // Add parameter {parameter_name}
                onnx::TypeProto type_proto;
                onnx::TypeProto::Tensor *tensor_type = type_proto.mutable_tensor_type();
                tensor_type->set_elem_type(onnx::TensorProto::{type_string});
                type_map["{unique_id}_{parameter_name}"] = type_proto;

                std::string* {input_output_string} = proto.add_{input_output_string}();
                *{input_output_string} = "{unique_id}_{parameter_name}";
            }}
            """.format(unique_id=unique_id,
                       type_string=reversed_onnx_dtype_map[arr.dtype].upper(),
                       parameter_name=parameter_name,
                       input_output_string=input_output_string))

            tasklet_setup_code += """
            int64_t {input_output_string}_{parameter_name}_dims[{dims_size}] = {{{dims}}};
            """.format(input_output_string=input_output_string,
                       parameter_name=parameter_name,
                       dims_size=len(arr.shape),
                       dims=", ".join(str(s) for s in arr.shape))

            ort_value_name = "ort_value_{input_output_string}_{parameter_name}".format(
                input_output_string=input_output_string,
                parameter_name=parameter_name)
            tasklet_setup_code += """
            OrtValue* {ort_value_name};
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
                __ort_mem_info,
                const_cast<void*>(reinterpret_cast<const void*>({parameter_name})),
                {data_size} * sizeof({ctype}),
                {input_output_string}_{parameter_name}_dims,
                {dims_size},
                ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str},
                &{ort_value_name}
            ));
            """.format(input_output_string=input_output_string,
                       parameter_name=parameter_name,
                       data_size=reduce(lambda x, y: x * y, arr.shape),
                       ctype=arr.dtype.ctype,
                       dims_size=len(arr.shape),
                       type_str=reversed_onnx_dtype_map[arr.dtype].upper(),
                       ort_value_name=ort_value_name)

            tasklet_code += "__ort_check_status(__ort_api->ExecutableKernelContext_Set{input_output_string_capital}(" \
                            "__ort_context_{unique_id}, {position}, {ort_value_name}));\n".format(
                input_output_string_capital=input_output_string.
                    capitalize(),
                ort_value_name=ort_value_name,
                unique_id=unique_id,
                position=get_position(node.schema, is_input,
                                      parameter_name))

            tasklet_cleanup_code += "__ort_api->ReleaseValue(ort_value_{input_output_string}_{parameter_name});\n".format(
                input_output_string=input_output_string,
                parameter_name=parameter_name)

        sdfg.append_init_code("// Setup attributes\n")

        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                sdfg.append_init_code(
                    _gen_attr_init_code("proto", node.schema.attributes[name],
                                        getattr(node, name)))

        if node.schedule == ScheduleType.CPU_Multicore or node.schedule == ScheduleType.Default:
            provider_index = 0
        elif node.schedule == ScheduleType.GPU_Device:
            provider_index = 1
        else:
            raise NotImplementedError(
                "ORT expansion for schedule '{}' is not implemented".format(
                    node.schedule))

        sdfg.append_init_code(
            "__ort_check_status(__ort_api->CreateExecutableKernelContext("
            "__ort_session, /*provider_index=*/{provider_index}, &proto, &type_map, &__ort_context_{id}));\n"
            .format(provider_index=provider_index, id=unique_id))
        sdfg.append_init_code(
            "}} // end setup for context_{}".format(unique_id))

        sdfg.prepend_exit_code(
            "__ort_api->ReleaseExecutableKernelContext(__ort_context_{});\n".
            format(unique_id))

        tasklet_code += "__ort_check_status(__ort_api->ExecutableKernelContext_Compute(__ort_context_{}));\n".format(
            unique_id)

        tasklet_code = tasklet_setup_code + tasklet_code + tasklet_cleanup_code
        return nd.Tasklet('onnx_code',
                          set(inputs.keys()),
                          set(outputs.keys()),
                          tasklet_code,
                          language=dace.dtypes.Language.CPP)


_ONNX_OPS_BY_NAME = {}
# Generate all of the Op Nodes
for schema in onnx.defs.get_all_schemas():
    try:
        dace_schema = ONNXSchema.from_onnx_proto(schema)
    except Exception as e:
        print("Import of {} failed: {}".format(schema.name, e))
        continue

    docstring = dace_schema.doc
    attrs = {}
    attrs['__doc__'] = docstring
    attrs['schema'] = dace_schema

    # add properties for each op attribute
    for name, attr in dace_schema.attributes.items():
        if attr.type in [
                ONNXAttributeType.Int, ONNXAttributeType.String,
                ONNXAttributeType.Float
        ]:
            attrs[name] = Property(dtype=_ATTR_TYPE_TO_PYTHON_TYPE[attr.type],
                                   desc=attr.description,
                                   allow_none=True,
                                   default=None if attr.default_value is None
                                   else attr.default_value)
        elif attr.type in [
                ONNXAttributeType.Ints, ONNXAttributeType.Strings,
                ONNXAttributeType.Floats
        ]:
            attrs[name] = ListProperty(
                element_type=_ATTR_TYPE_TO_PYTHON_TYPE[attr.type],
                desc=attr.description,
                allow_none=True,
                default=None
                if attr.default_value is None else attr.default_value)
        else:
            raise NotImplementedError(
                "Got unsupported ONNXAttributeType: {}".format(attr.type))

    required_attrs = {
        name
        for name, attr in dace_schema.attributes.items() if attr.required
    }

    def __init__(self, name, *args, location=None, **op_attributes):
        super(ONNXOp, self).__init__(
            name,
            location=location,
            # add required parameters as in/out connectors
            inputs={
                inp.name
                for inp in self.schema.inputs
                if inp.param_type == ONNXParameterType.Single
            },
            outputs={
                out.name
                for out in self.schema.outputs
                if out.param_type == ONNXParameterType.Single
            })

        if len(args) > 0:
            raise TypeError(
                "__init__() takes 2 positional arguments but {} were given".
                format(2 + len(args)))

        missing_arguments = required_attrs.difference(op_attributes)
        if len(missing_arguments) > 0:

            raise TypeError(
                get_missing_arguments_message("__init__()", missing_arguments,
                                              "keyword-only argument"))

        unknown_attrs = set(op_attributes).difference(self.schema.attributes)
        if len(unknown_attrs) > 0:
            raise TypeError(
                "__init__() got an unexpected keyword argument '{}'".format(
                    list(unknown_attrs)[0]))

        for name, attr in op_attributes.items():
            setattr(self, name, attr)

        # Inline the class such that "self" is included in the expansion
        @dace.library.expansion
        class Expansion(ExpandTransformation):
            environments = [ONNXRuntime]

            @staticmethod
            def expansion(node, state: SDFGState, sdfg: SDFG):
                try:
                    node.validate(sdfg, state)
                except Exception as ex:
                    raise ValueError(
                        "Node validation failed: {} (at ONNX Operator {})".
                        format(str(ex), self.schema.name))

                return self.expansion(node, state, sdfg)

        self.implementations['default'] = Expansion
        Expansion._match_node = self
        self.implementation = 'default'

    attrs['__init__'] = __init__

    cls_name = "ONNX" + dace_schema.name
    cls = type(cls_name, (ONNXOp, ), attrs)

    cls = dace.library.node(cls)
    globals()[cls_name] = cls
    _ONNX_OPS_BY_NAME[cls_name] = cls

del cls


def has_onnx_node(name: str):
    """ Check if an ONNX operator is supported
        :param name: the operator name
    """
    return ("ONNX" + name) in _ONNX_OPS_BY_NAME


def get_onnx_node(name: str):
    """ Get the ONNX Operator node for an operator by name
        :param name: the operator name
    """
    return _ONNX_OPS_BY_NAME["ONNX" + name]
