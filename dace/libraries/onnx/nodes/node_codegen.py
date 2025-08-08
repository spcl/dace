import logging
from collections.abc import Iterable
from copy import deepcopy
import functools
from typing import Dict, Tuple, List, Optional

import dace
import dace.data as dt
import dace.library
import dace.sdfg.nodes as nd
import numpy as np
from dace import dtypes, SDFGState, SDFG
from dace.codegen import cppunparse
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs

from dace.libraries.onnx.binary_utilities.op_checker import check_op
from dace.libraries.onnx.converters import clean_onnx_name, typeclass_to_onnx_str
from dace.libraries.onnx.environments import ONNXRuntime, ONNXRuntimeCUDA
from dace.libraries.onnx.nodes.node_utils import get_position
from dace.libraries.onnx.schema import ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXAttribute
from dace.libraries.ort_api import ORTAPIError

from dace.util import prod

import onnx
from onnx import helper
from onnx import TensorProto

log = logging.getLogger(__name__)

def numpy_dtype_to_onnx_dtype(np_dtype):
    """Convert numpy dtype to ONNX TensorProto data type integer."""
    dtype_map = {
        np.float32: TensorProto.FLOAT,
        np.float64: TensorProto.DOUBLE,
        np.int32: TensorProto.INT32,
        np.int64: TensorProto.INT64,
        np.uint32: TensorProto.UINT32,
        np.uint64: TensorProto.UINT64,
        np.int16: TensorProto.INT16,
        np.uint16: TensorProto.UINT16,
        np.int8: TensorProto.INT8,
        np.uint8: TensorProto.UINT8,
        np.bool_: TensorProto.BOOL,
        np.float16: TensorProto.FLOAT16,
        # Add more mappings as needed
    }
    return dtype_map.get(np_dtype, TensorProto.FLOAT)  # Default to FLOAT

def _get_onnx_shape_from_desc(desc: dt.Data) -> Optional[List]:
    """Get the appropriate ONNX shape from a DaCe data descriptor.
    
    Args:
        desc: The DaCe data descriptor
        
    Returns:
        The shape as a list, or None for scalars (0-dimensional tensors)
    """
    if isinstance(desc, dt.Scalar):
        # Scalars have shape [1] but should be represented as 0-dimensional in ONNX
        return None
    elif isinstance(desc, dt.Array):
        # Arrays have their actual shape
        return list(desc.shape)
    else:
        # For other types (Stream, etc.), use the shape property
        return list(desc.shape) if desc.shape else None


def check_required_copies(
        node: nd.Node, state: SDFGState, sdfg: SDFG, outputs_on_host: List[bool],
        inputs_on_host: List[bool]) -> Tuple[Dict[str, dtypes.StorageType], Dict[str, dtypes.StorageType]]:
    """ Check whether copies are required for all parameters.
        :param node: the node.
        :param state: the state.
        :param sdfg: the sdfg.
        :param outputs_on_host: boolean list, where the ith bool indicates if the ith output should be on host.
        :param inputs_on_host: boolean list, where the ith bool indicates if the ith input should be on host.
        :return: two dicts containing storage types for each of the connectors that require copies. The first
                 dict is for the inputs, the second is for the outputs.
    """

    # maps the connectors for which a copy will be required to the storage type required to be connected to the tasklet
    input_copy_required: Dict[str, dtypes.StorageType] = {}
    output_copy_required: Dict[str, dtypes.StorageType] = {}

    assert len(node.iter_outputs_in_onnx_order(state)) == len(outputs_on_host)
    assert len(node.iter_inputs_in_onnx_order(state)) == len(inputs_on_host)

    # check outputs
    for edge, output_on_host in zip(node.iter_outputs_in_onnx_order(state), outputs_on_host):
        # get the memlet for this output
        array = sdfg.arrays[edge.data.data]

        if output_on_host:
            is_device_mismatch = not dtypes.can_access(dtypes.ScheduleType.Default, array.storage)
        else:
            is_device_mismatch = not dtypes.can_access(dtypes.ScheduleType.GPU_Device, array.storage)

        if is_device_mismatch:
            # we need to insert a copy
            storage = dtypes.StorageType.CPU_Heap if output_on_host else dtypes.StorageType.GPU_Global
            output_copy_required[edge.src_conn] = storage

    # check inputs (same thing again)
    for edge, input_on_host in zip(node.iter_inputs_in_onnx_order(state), inputs_on_host):
        array = sdfg.arrays[edge.data.data]

        if input_on_host:
            is_device_mismatch = not dtypes.can_access(dtypes.ScheduleType.Default, array.storage)
        else:
            is_device_mismatch = not dtypes.can_access(dtypes.ScheduleType.GPU_Device, array.storage)

        if is_device_mismatch:
            # we need to insert a copy
            storage = dtypes.StorageType.CPU_Heap if input_on_host else dtypes.StorageType.GPU_Global
            input_copy_required[edge.dst_conn] = storage

    return input_copy_required, output_copy_required


def emit_setup_code_for_ortvalue(node: nd.CodeNode, parameter_name: str, edge_connector_name: str, desc: dt.Data,
                                 required_copy: Optional[dtypes.StorageType], is_input: bool, ort_value_name: str,
                                 connector_dict: dict) -> str:
    """ Emit the code that creates the OrtValue for a parameter. Also set the connector types on the parent node.

        :param node: the parent node that we are expanding
        :param parameter_name: the onnx name of the parameter.
        :param edge_connector_name: the name of the edge connector to the tasklet.
        :param desc: the dace input descriptor connected to this parameter.
        :param required_copy: the ``StorageType`` to copy to for this parameter, if a copy is required.
        :param is_input: whether the parameter is an input.
        :param ort_value_name: the name for the ort_value.
        :param connector_dict: either the input connector or output connector dict for the expanded node, depending on
                               whether this is an input or an output.
        :return: the code that creates the OrtValue for ``parameter_name``.
    """

    parent_connector_dict = node.in_connectors if is_input else node.out_connectors
    input_output_string = "input" if is_input else "output"
    code = ""

    if required_copy is not None:
        storage = required_copy
    else:
        storage = desc.storage

    if storage in [dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register]:
        mem_info = "ort_cpu_mem_info"
    elif storage is dtypes.StorageType.GPU_Global:
        mem_info = "ort_cuda_mem_info"
    elif storage is dtypes.StorageType.CPU_Pinned:
        mem_info = "ort_cuda_pinned_mem_info"
    else:
        raise ValueError("Unsupported storage type {} for input to ONNX node".format(desc.storage))

    if isinstance(desc, dt.Scalar):

        on_gpu = storage is dtypes.StorageType.GPU_Global

        code += """
        OrtValue* {ort_value_name};
        __ort_check_status(__state->ort_api, __state->ort_api->CreateTensorWithDataAsOrtValue(
            {mem_info},
            {maybe_ref}{edge_connector_name},
            {data_size} * sizeof({ctype}),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str},
            &{ort_value_name}
        ));
        """.format(mem_info=mem_info,
                   edge_connector_name=edge_connector_name,
                   data_size=cppunparse.pyexpr2cpp(str(prod(desc.shape))),
                   ctype=desc.dtype.ctype,
                   type_str=typeclass_to_onnx_str(desc.dtype).upper(),
                   ort_value_name=ort_value_name,
                   maybe_ref="" if on_gpu else "&")

        if on_gpu:
            connector_dict[edge_connector_name] = dace.pointer(desc.dtype)
            parent_connector_dict[parameter_name] = dace.pointer(desc.dtype)
        else:
            connector_dict[edge_connector_name] = desc.dtype
            parent_connector_dict[parameter_name] = desc.dtype
    elif isinstance(desc, dt.Array):

        # setup dims array
        code += """
        int64_t {input_output_string}_{parameter_name}_dims[{dims_size}] = {{{dims}}};
        """.format(input_output_string=input_output_string,
                   parameter_name=parameter_name,
                   dims_size=len(desc.shape),
                   dims=", ".join(str(s) for s in desc.shape))

        data = "const_cast < void * > (reinterpret_cast < const void * > ({}))".format(edge_connector_name)

        code += """
        OrtValue* {ort_value_name};
        __ort_check_status(__state->ort_api, __state->ort_api->CreateTensorWithDataAsOrtValue(
            {mem_info},
            {data},
            {data_size} * sizeof({ctype}),
            {input_output_string}_{parameter_name}_dims,
            {dims_size},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_{type_str},
            &{ort_value_name}
        ));
        """.format(input_output_string=input_output_string,
                   data=data,
                   mem_info=mem_info,
                   parameter_name=parameter_name,
                   data_size=cppunparse.pyexpr2cpp(str(prod(desc.shape))),
                   ctype=desc.dtype.ctype,
                   dims_size=len(desc.shape),
                   type_str=typeclass_to_onnx_str(desc.dtype).upper(),
                   ort_value_name=ort_value_name)
        connector_dict[edge_connector_name] = dace.pointer(desc.dtype)
        parent_connector_dict[parameter_name] = dace.pointer(desc.dtype)
    else:
        raise NotImplementedError("Data-descriptor type {} not supported for ONNX nodes".format(type(desc)))
    return code


def expand_node(node, state, sdfg):
    if not ONNXRuntime.is_installed():
        raise RuntimeError("ONNXRuntime is not installed, cannot expand node "
                           "{}. You can either install ONNX Runtime as described in the "
                           "docs, or add a pure node implementation for the {} op.".format(node, node.schema.name))

    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)

    unique_id = "{}_{}_{}_{}".format(clean_onnx_name(node.name), sdfg.sdfg_id, sdfg.node_id(state), state.node_id(node))

    # check if ORT supports CUDA for this node using the op checker
    ###############################################################

    # Default: all parameters are on CPU if we execute using cpu
    outputs_on_host = [True for _ in range(len(outputs))]
    inputs_on_host = [True for _ in range(len(inputs))]

    actual_node_schedule = node.schedule
    if node.schedule == dtypes.ScheduleType.CPU_Multicore or node.schedule == dtypes.ScheduleType.Default:
        provider_index = 0
    elif node.schedule in dtypes.GPU_SCHEDULES + [dtypes.ScheduleType.GPU_Default]:
        provider_index = 1
        try:
            # the ith position indicates whether the ith output is in host memory
            inputs_on_host, outputs_on_host = check_op(sdfg, state, node, cuda=True)

        except ORTAPIError as e:
            # fallback to CPU
            log.warning("Falling back to CPU for node {}. Reason:\n{}".format(node.name, str(e)))
            provider_index = 0
    else:
        raise NotImplementedError("ORT expansion for schedule '{}' is not implemented".format(node.schedule))

    # check if we need to insert device copies
    ##########################################

    input_copy_required, output_copy_required = check_required_copies(node, state, sdfg, outputs_on_host,
                                                                      inputs_on_host)

    # Collect input and output names for the node
    input_names = []
    output_names = []
    
    for edge, is_input in node.iter_edges(state):
        parameter_name = edge.dst_conn if is_input else edge.src_conn
        if is_input:
            input_names.append(parameter_name)
        else:
            output_names.append(parameter_name)
    
    # Collect attributes for the node
    node_attrs = {}
    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            value = getattr(node, name)
            if value is not None:
                # Handle tensor attributes specially since helper.make_node doesn't handle them automatically
                if attr.attribute_type == ONNXAttributeType.Tensor:
                    # Convert numpy array to ONNX tensor
                    tensor = helper.make_tensor(name, numpy_dtype_to_onnx_dtype(value.dtype.type), value.shape, value.flatten().tolist())
                    node_attrs[name] = tensor
                else:
                    node_attrs[name] = value
    
    # Create the ONNX node using helper.make_node with attributes
    node_proto = helper.make_node(node.schema.name, input_names, output_names, name=unique_id + "_node", **node_attrs)
    
    # Create input and output value infos
    input_value_infos = []
    output_value_infos = []
    
    for edge, is_input in node.iter_edges(state):
        parameter_name = edge.dst_conn if is_input else edge.src_conn
        memlet = edge.data
        desc = sdfg.arrays[memlet.data]
        
        value_info = helper.make_tensor_value_info(
            parameter_name, 
            getattr(onnx.TensorProto, typeclass_to_onnx_str(desc.dtype).upper()),
            _get_onnx_shape_from_desc(desc)
        )
        
        if is_input:
            input_value_infos.append(value_info)
        else:
            output_value_infos.append(value_info)
    
    # Create the graph using helper.make_graph
    graph_def = helper.make_graph(
        [node_proto],
        unique_id,
        input_value_infos,
        output_value_infos
    )
    
    # Create the model using helper.make_model with opset version from the original node
    onnx_model = helper.make_model(graph_def, producer_name="dace", opset_imports=[helper.make_opsetid("", node.schema.since_version)])
    

    # begin codegen
    ##########################################
    tasklet_setup_code = ""
    tasklet_code = ""
    tasklet_cleanup_code = ""

    env_init_code = ""

    # save constructed model
    model_bytes = onnx_model.SerializeToString()
    # save model to file
    # with open(f"{unique_id}.onnx", "wb") as f:
    #     f.write(model_bytes)
    # embed model as C byte string
    model_int_values = [str(b) for b in model_bytes]
    model_int_values_str = ", ".join(model_int_values)
    tasklet_setup_code += f"""
    unsigned char kernel_data_{unique_id}[{len(model_int_values)}] = {{ {model_int_values_str} }};
    """
    
    tasklet_setup_code += f"""
        OrtMemoryInfo* ort_cpu_mem_info;
        __ort_check_status(__state->ort_api, __state->ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, /*type=*/OrtMemTypeDefault, &ort_cpu_mem_info));
        OrtSessionOptions* ort_session_options;
        __ort_check_status(__state->ort_api, __state->ort_api->CreateSessionOptions(&ort_session_options));
        OrtSession* ort_session;
        __ort_check_status(__state->ort_api, __state->ort_api->CreateSessionFromArray(
            __state->ort_env, kernel_data_{unique_id}, {len(model_int_values)},
            ort_session_options, &ort_session
        ));
    """

    # emit code for inputs and outputs
    ##########################################
    in_connectors = {}
    out_connectors = {}

    input_names = []
    output_names = []
    input_values = []
    output_values = []

    for edge, is_input in node.iter_edges(state):
        parameter_name = edge.dst_conn if is_input else edge.src_conn

        input_output_string = "input" if is_input else "output"
        memlet = edge.data
        desc = sdfg.arrays[memlet.data]

        if is_input:
            input_names.append(parameter_name)
        else:
            output_names.append(parameter_name)

        ort_value_name = "ort_value_{input_output_string}_{parameter_name}".format(
            input_output_string=input_output_string, parameter_name=parameter_name)

        if is_input:
            input_values.append(ort_value_name)
        else:
            output_values.append(ort_value_name)

        # We always emit a NestedSDFG, so the edge connector names must be prefixed (otherwise there would be a conflict
        # of names).
        edge_connector_name = "__" + parameter_name

        copy_options_dict = input_copy_required if is_input else output_copy_required
        copy_options = copy_options_dict.get(parameter_name, None)

        tasklet_setup_code += emit_setup_code_for_ortvalue(node=node,
                                                           parameter_name=parameter_name,
                                                           edge_connector_name=edge_connector_name,
                                                           desc=desc,
                                                           required_copy=copy_options,
                                                           is_input=is_input,
                                                           ort_value_name=ort_value_name,
                                                           connector_dict=in_connectors if is_input else out_connectors)

        tasklet_cleanup_code += "__state->ort_api->ReleaseValue(" \
                                "ort_value_{input_output_string}_{parameter_name});\n".format(
            input_output_string=input_output_string,
            parameter_name=parameter_name)
    
    env_finalize_code = ""

    if logging.root.level <= logging.DEBUG:
        tasklet_code += 'fprintf(stderr, "Launching {}\\n");\n'.format(unique_id)

    tasklet_code += f"""
    const char* input_names[] = {{{", ".join(f'"{name}"' for name in input_names)}}};
    const char* output_names[] = {{{", ".join(f'"{name}"' for name in output_names)}}};
    OrtValue* input_values[] = {{{", ".join(input_values)}}};
    OrtValue* output_values[] = {{{", ".join(output_values)}}};
    __ort_check_status(__state->ort_api, __state->ort_api->Run(
        ort_session, NULL, input_names, input_values, {len(input_values)}, output_names, {len(output_values)}, output_values));
    """

    tasklet_cleanup_code += f"""
    __state->ort_api->ReleaseSession(ort_session);
    __state->ort_api->ReleaseSessionOptions(ort_session_options);
    __state->ort_api->ReleaseMemoryInfo(ort_cpu_mem_info);
    """

    tasklet_code = tasklet_setup_code + tasklet_code + tasklet_cleanup_code

    if ONNXRuntimeCUDA.use_streams:
        raise ValueError("Currently not supported anymore.")

    if env_init_code:
        env_init_code = "{\n" + env_init_code + "\n}"
    if env_finalize_code:
        env_finalize_code = "{\n" + env_finalize_code + "\n}"

    tasklet = nd.Tasklet(unique_id + '_onnx_code',
                         in_connectors,
                         out_connectors,
                         tasklet_code,
                         code_init=env_init_code,
                         code_exit=env_finalize_code,
                         language=dace.dtypes.Language.CPP)

    env = ONNXRuntimeCUDA if node.schedule in dtypes.GPU_SCHEDULES + [dtypes.ScheduleType.GPU_Default] else ONNXRuntime
    tasklet.environments = {env.full_class_path()}

    nsdfg = dace.SDFG("nested_{}".format(unique_id))
    nstate = nsdfg.add_state()
    ntasklet = deepcopy(tasklet)

    nstate.add_node(ntasklet)

    for edge, is_input in node.iter_edges(state):
        parameter_name = edge.dst_conn if is_input else edge.src_conn

        memlet = edge.data
        desc = sdfg.arrays[memlet.data]

        # add the original array
        original_desc = deepcopy(desc)
        original_desc.transient = False
        nsdfg.add_datadesc(parameter_name, original_desc)
        if not (isinstance(desc, dt.Array) or isinstance(desc, dt.Scalar)):
            raise ValueError("Unsupported data type {} connected to an ONNX tasklet".format(type(desc)))

        copy_options_dict = input_copy_required if is_input else output_copy_required
        # handle parameters for which no copies are required
        if parameter_name not in copy_options_dict:
            copied_memlet = deepcopy(memlet)
            copied_memlet.data = parameter_name
            if is_input:
                access = nstate.add_read(parameter_name)
                nstate.add_edge(access, None, ntasklet, "__" + parameter_name, copied_memlet)
            else:
                access = nstate.add_write(parameter_name)
                nstate.add_edge(ntasklet, "__" + parameter_name, access, None, copied_memlet)
            continue

        # add the copy of the descriptor
        copy_desc = deepcopy(desc)

        copy_desc.transient = True
        copy_desc.storage = copy_options_dict[parameter_name]
        # there can be name conflicts here if an input is given to multiple
        # connectors. We could technically share the copied result, but it's
        # likely not worth the effort since these are just scalars.
        copy_name = nsdfg.add_datadesc("copy_" + memlet.data, copy_desc, find_new_name=True)

        nmemlet = deepcopy(memlet)
        nmemlet_copy = deepcopy(memlet)
        nmemlet_copy.data = copy_name
        nmemlet.data = copy_name
        if is_input:
            access = nstate.add_read(parameter_name)
            access_copy = nstate.add_access(copy_name)
            nstate.add_edge(access, None, access_copy, None, nmemlet_copy)
            nstate.add_edge(access_copy, None, ntasklet, "__" + parameter_name, nmemlet)
        else:
            access = nstate.add_write(parameter_name)
            access_copy = nstate.add_access(copy_name)
            nstate.add_edge(ntasklet, "__" + parameter_name, access_copy, None, nmemlet)
            nstate.add_edge(access_copy, None, access, None, nmemlet_copy)

    return nsdfg
