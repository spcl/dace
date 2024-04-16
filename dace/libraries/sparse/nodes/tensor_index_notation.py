"""
General purpose Tensor Index Notation (TIN) library node.

Used as initial step 
"""

import subprocess
import re

from copy import deepcopy
from functools import reduce
from typing import List

from dace import SDFG, library, nodes, transformation, dtypes, Memlet
from dace.data import TensorIndexDense, TensorIndexCompressed, Tensor, Array
from dace.frontend.common.op_repository import replaces
from dace.properties import Property, ListProperty
from dace.sdfg import SDFG, SDFGState, nodes as nd


@library.node
class TensorIndexNotation(nodes.LibraryNode):
    implementations = {}
    default_implementation = "taco"

    tensor_index_notation = Property(
        dtype=str,
        default="",
        desc="The Tensor Index Notation (TIN) string that describes this operation",
    )

    extra_taco_args = ListProperty(
        element_type=str,
        desc="Extra arguments to pass to taco"
    )

    def __init__(self, name, expr, extra_taco_args: List[str] = [], *args, schedule=None, **kwargs):
        super().__init__(name, *args, schedule=schedule, **kwargs)
        self.tensor_index_notation = expr
        self.extra_taco_args = extra_taco_args


@library.register_expansion(TensorIndexNotation, "concretize")
class ConcretizeTIN(transformation.ExpandTransformation):
    # Define environments necessary for this expansion (optional, can be an empty list)
    environments = []

    @staticmethod
    def expansion(
        node: TensorIndexNotation, parent_state: SDFGState, parent_sdfg: SDFG
    ) -> SDFG:
        
        # Check if it can be reduced to CSRMM

        if len(parent_state.in_edges(node)) != 2:
            raise NotImplementedError("Only supports two input tensors at the moment")
        
        inputs = {e.dst_conn: parent_sdfg.arrays[e.data.data] for e in parent_state.in_edges(node)}

        print(f'DEBUG: {inputs = }')

        keys = list(inputs.keys())

        if isinstance(inputs[keys[0]], Array):
            array = inputs[keys[0]]
            array_name = keys[0]
            tensor = inputs[keys[1]]
            tensor_name = keys[1]
        else:
            array = inputs[keys[1]]
            array_name = keys[1]
            tensor = inputs[keys[0]]
            tensor_name = keys[0]

        output_array_name = parent_state.out_edges(node)[0].src_conn
        output_array = parent_sdfg.arrays[parent_state.out_edges(node)[0].data.data]
        
        print(f'DEBUG: {array = }; {tensor = }')
        
        if not (isinstance(array, Array) and isinstance(tensor, Tensor)):
            raise NotImplementedError("Only suports one array and one tensor")
        
        if not isinstance(tensor.indices[-1], TensorIndexCompressed):
            raise NotImplementedError("Only implemented if last dimension of tensor is compressed")
        
        if not all([isinstance(idx, TensorIndexDense) for idx in tensor.indices[:-1]]):
           raise NotImplementedError("Only implemented if all dimensions but last are dense")
        
        array_shape = list(array.shape)
        array_reduced_dim = reduce(lambda x,y: x*y, array_shape[:-1])

        tensor_shape = list(tensor.tensor_shape)
        tensor_reduced_dim =reduce(lambda x,y: x*y, tensor_shape[:-1]) 

        output_reduced_shape = [tensor_reduced_dim, array_reduced_dim]

        print(f"Will view array as {array_reduced_dim} by {array_shape[-1]}")
        print(f"Will view tensor as {tensor_reduced_dim} by {tensor_shape[-1]}")
        print(f"output will be {', '.join([str(x) for x in output_reduced_shape])}, but viewd as {tensor_shape[:-1] + array_shape[:-1]}")

        print(f"DEBUG: {tensor.members = }")
        
        sdfg = SDFG("CSRMM")
        state = sdfg.add_state()

        # Create the arrays
        sdfg.add_datadesc(tensor_name, tensor.clone())
        sdfg.add_array(array_name, array.shape, array.dtype)
        sdfg.add_array('_output', list(reversed(output_reduced_shape)), output_array.dtype, transient=True)
        sdfg.add_array(output_array_name, output_array.shape, output_array.dtype)

        sdfg.add_view('pos', [tensor_reduced_dim + 1], dtypes.int32)
        sdfg.add_view('crd', [tensor_reduced_dim + 1], dtypes.int32)
        sdfg.add_view('vals', tensor.members['values'].shape, tensor.value_dtype)
        sdfg.add_view('dense', (array_reduced_dim, array_shape[-1]), array.dtype)
        sdfg.add_view('output', output_reduced_shape, output_array.dtype)

        tensor_access = state.add_access(tensor_name)
        array_access = state.add_access(array_name)
        output_access = state.add_access(output_array_name)

        pos = state.add_access('pos')
        crd = state.add_access('crd')
        vals = state.add_access('vals')
        dense = state.add_access('dense')
        _output = state.add_access('_output')
        output = state.add_access('output')

        idx_num = len(tensor.indices) - 1

        state.add_edge(tensor_access, None, pos, 'views', Memlet.from_array(f'{tensor_name}.idx{idx_num}_pos', tensor.members[f'idx{idx_num}_pos']))
        state.add_edge(tensor_access, None, crd, 'views', Memlet.from_array(f'{tensor_name}.idx{idx_num}_crd', tensor.members[f'idx{idx_num}_crd']))
        state.add_edge(tensor_access, None, vals, 'views', Memlet.from_array(f'{tensor_name}.values', tensor.members['values']))
        state.add_edge(array_access, None, dense, 'views', Memlet.from_array(array_name, array))

        from dace.libraries.sparse import CSRMM
        csrmm = CSRMM('csrmm')
        state.add_node(csrmm)

        state.add_edge(pos, None, csrmm, '_a_rows', Memlet(data='pos'))
        state.add_edge(crd, None, csrmm, '_a_cols', Memlet(data='crd'))
        state.add_edge(vals, None, csrmm, '_a_vals', Memlet(data='vals'))
        state.add_edge(dense, None, csrmm, '_b', Memlet(data='dense'))
        state.add_edge(csrmm, '_c', _output, None, Memlet(data='_output'))

        from dace.libraries.standard import Transpose
        transpose = Transpose('transpose', dtype=array.dtype)
        state.add_node(transpose)

        state.add_edge(_output, None, transpose, '_inp', Memlet(data='_output'))
        state.add_edge(transpose, '_out', output, None, Memlet(data='output'))

        state.add_edge(output, 'views', output_access, None, Memlet.from_array(output_array_name, output_array))

        return sdfg

        


@library.register_expansion(TensorIndexNotation, "taco")
class TacoTIN(transformation.ExpandTransformation):
    # Define environments necessary for this expansion (optional, can be an empty list)
    environments = []

    @staticmethod
    def expansion(
        node: TensorIndexNotation, parent_state: SDFGState, parent_sdfg: SDFG
    ) -> SDFG:
        sdfg = SDFG("tensor_index_notation")
        state = sdfg.add_state()

        # print(f"DEBUG Tensor index expression: {node.tensor_index_notation}")

        args = [
            "taco",
            node.tensor_index_notation,
            "-print-kernels",
            "-print-nocolor",
        ]

        inputs = {}
        outputs = {}

        output_dense = False

        tensor_init_code = "// INIT\n"
        tensor_deinit_code = "// DE-INIT\n"

        PROFILE = False

        if PROFILE:
            tensor_init_code += "auto __tin_init = std::chrono::high_resolution_clock::now();\n"
            tensor_deinit_code+= "auto __tin_end = std::chrono::high_resolution_clock::now();\n"

        for e in parent_state.in_edges(node):
            desc = parent_sdfg.arrays[e.data.data]
            # print(f"DEBUG input {e.dst_conn} of type {desc} {type(desc)}")

            t_name = str(e.dst_conn)[4:]
            t = f"{t_name}_tensor"

            tensor_dtype = None

            # generating taco build flags
            if isinstance(desc, Tensor):
                for idx in desc.indices:
                    if type(idx) not in [TensorIndexDense, TensorIndexCompressed]:
                        raise NotImplementedError(
                            "Only supports Dense and Compressed indices at the moment"
                        )

                order = len(desc.tensor_shape)
                tensor_dtype = desc.value_dtype

                encoding = (
                    "".join([str(idx)[0] for idx in desc.indices])
                    .lower()
                    .replace("c", "s")
                )
                if desc.index_ordering == list(range(order)):
                    args.append(f"-f={t_name}:{encoding}")
                else:
                    args.append(
                        f"-f={t_name}:{encoding}:{','.join([str(s) for s in desc.index_ordering])}"
                    )

                idx_types = [
                    (
                        "taco_mode_dense"
                        if type(idx) is TensorIndexDense
                        else "taco_mode_sparse"
                    )
                    for idx in desc.indices
                ]
                tensor_init_code += (
                    f"// init potentially sparse input tensor {t_name}\n"
                    f"int32_t {t}_dims[] = {{{', '.join([str(s) for s in desc.tensor_shape])}}};\n"
                    f"int32_t {t}_ordering[] = {{{', '.join([str(s) for s in desc.index_ordering])}}};\n"
                    f"taco_mode_t {t}_types[] = {{{', '.join(idx_types)}}};\n"
                    f"taco_tensor_t* {t} = init_taco_tensor_t({order}, sizeof(double), {t}_dims, {t}_ordering, {t}_types);\n"
                    f"{t}->vals = (uint8_t *) tin_{t_name}_task->values;\n"
                )
                for i, idx in enumerate(desc.indices):
                    if type(idx) is TensorIndexCompressed:
                        tensor_init_code += (
                            f"{t}->indices[{i}][0] = (uint8_t *) tin_{t_name}_task->idx{i}_pos;\n"
                            f"{t}->indices[{i}][1] = (uint8_t *) tin_{t_name}_task->idx{i}_crd;\n"
                        )
                tensor_deinit_code += (
                    f"// deinit potentially sparse input tensor {t_name}\n"
                    f"deinit_taco_tensor_t({t});\n"
                )
            else:
                order = len(desc.shape)
                tensor_dtype = desc.dtype
                tensor_init_code += (
                    f"// init dense input tensor {t_name}\n"
                    f"int32_t {t}_dims[] = {{{', '.join([str(s) for s in desc.shape])}}};\n"
                    f"int32_t {t}_ordering[] = {{{', '.join([str(s) for s in range(order)])}}};\n"
                    f"taco_mode_t {t}_types[] = {{{', '.join(['taco_mode_dense'] * order)}}};\n"
                    f"taco_tensor_t* {t} = init_taco_tensor_t({order}, sizeof(double), {t}_dims, {t}_ordering, {t}_types);\n"
                    f"{t}->vals = (uint8_t *) tin_{t_name}_task;\n"
                )
                tensor_deinit_code += (
                    f"// deinit dense input tensor {t_name}\n"
                    f"deinit_taco_tensor_t({t});\n"
                )

            if tensor_dtype == dtypes.float32:
                args.append(f"-t={t_name}:float")
            elif tensor_dtype == dtypes.float64:
                args.append(f"-t={t_name}:double")
            else:
                assert False

            tensor_init_code += "\n"
            tensor_deinit_code += "\n"

            insubset = deepcopy(e.data.src_subset)
            isqdim = insubset.squeeze()
            sdfg.add_array(
                e.dst_conn,
                insubset.size(),
                desc.dtype,
                strides=[s for i, s in enumerate(desc.strides) if i in isqdim],
                storage=desc.storage,
            )
            inputs[e.dst_conn] = state.add_access(e.dst_conn)

        for e in parent_state.out_edges(node):
            desc = parent_sdfg.arrays[e.data.data]
            # print(f"DEBUG output {e.src_conn} of type {desc}")

            t_name = str(e.src_conn)[4:]
            t = f"{t_name}_tensor"

            tensor_dtype = None

            # generating taco build flags
            if isinstance(desc, Tensor):
                for idx in desc.indices:
                    if type(idx) not in [TensorIndexDense, TensorIndexCompressed]:
                        raise NotImplementedError(
                            "Only supports Dense and Compressed indices at the moment"
                        )

                order = len(desc.tensor_shape)
                tensor_dtype = desc.value_dtype

                encoding = (
                    "".join([str(idx)[0] for idx in desc.indices])
                    .lower()
                    .replace("c", "s")
                )
                if desc.index_ordering == list(range(order)):
                    args.append(f"-f={t_name}:{encoding}")
                else:
                    args.append(
                        f"-f={t_name}:{encoding}:{','.join([str(s) for s in desc.index_ordering])}"
                    )

                idx_types = [
                    (
                        "taco_mode_dense"
                        if type(idx) is TensorIndexDense
                        else "taco_mode_sparse"
                    )
                    for idx in desc.indices
                ]
                tensor_init_code += (
                    f"// init potentially sparse output tensor {t_name}\n"
                    f"int32_t {t}_dims[] = {{{', '.join([str(s) for s in desc.tensor_shape])}}};\n"
                    f"int32_t {t}_ordering[] = {{{', '.join([str(s) for s in desc.index_ordering])}}};\n"
                    f"taco_mode_t {t}_types[] = {{{', '.join(idx_types)}}};\n"
                    f"taco_tensor_t* {t} = init_taco_tensor_t({order}, sizeof(double), {t}_dims, {t}_ordering, {t}_types);\n"
                    f"{t}->vals = (uint8_t *) tin_{t_name}_task->values;\n"
                )
                for i, idx in enumerate(desc.indices):
                    if type(idx) is TensorIndexCompressed:
                        tensor_init_code += (
                            f"{t}->indices[{i}][0] = (uint8_t *) tin_{t_name}_task->idx{i}_pos;\n"
                            f"{t}->indices[{i}][1] = (uint8_t *) tin_{t_name}_task->idx{i}_crd;\n"
                        )
                tensor_deinit_code += (
                    f"// deinit potentially sparse output tensor {t_name}\n"
                    f"tin_{t_name}_task->values = ({desc.value_dtype.ctype} *) {t}->vals;\n"
                )
                for i, idx in enumerate(desc.indices):
                    if type(idx) is TensorIndexCompressed:
                        tensor_deinit_code += (
                            f"tin_{t_name}_task->idx{i}_pos = (int *) {t}->indices[{i}][0];\n"
                            f"tin_{t_name}_task->idx{i}_crd = (int *) {t}->indices[{i}][1];\n"
                        )
                tensor_deinit_code += f"deinit_taco_tensor_t({t});\n"
            else:
                output_dense = True
                order = len(desc.shape)
                tensor_dtype = desc.dtype
                tensor_init_code += (
                    f"// init dense output tensor {t_name}\n"
                    f"int32_t {t}_dims[] = {{{', '.join([str(s) for s in desc.shape])}}};\n"
                    f"int32_t {t}_ordering[] = {{{', '.join([str(s) for s in range(order)])}}};\n"
                    f"taco_mode_t {t}_types[] = {{{', '.join(['taco_mode_dense'] * order)}}};\n"
                    f"taco_tensor_t* {t} = init_taco_tensor_t({order}, sizeof(double), {t}_dims, {t}_ordering, {t}_types);\n"
                    f"{t}->vals = (uint8_t *) tin_{t_name}_task;\n"
                )
                tensor_deinit_code += (
                    f"// deinit dense output tensor {t_name}\n"
                    f"deinit_taco_tensor_t({t});\n"
                )

            if tensor_dtype == dtypes.float32:
                args.append(f"-t={t_name}:float")
            elif tensor_dtype == dtypes.float64:
                args.append(f"-t={t_name}:double")
            else:
                assert False

            tensor_init_code += "\n"
            tensor_deinit_code += "\n"

            outsubset = deepcopy(e.data.dst_subset)
            osqdim = outsubset.squeeze()
            sdfg.add_array(
                e.src_conn,
                outsubset.size(),
                desc.dtype,
                strides=[s for i, s in enumerate(desc.strides) if i in osqdim],
                storage=desc.storage,
            )
            outputs[e.src_conn] = state.add_access(e.src_conn)

        args.extend(node.extra_taco_args)

        # print(f"DEBUG {args}")

        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()

        # print(f"DEBUG TACO output:")
        # print(f"{output}")

        restrict_regex = re.compile(r" restrict ")
        glob_code = restrict_regex.sub(" __restrict__ ", output.decode("utf-8"))

        function_prefix_regex = re.compile(
            r"(?=(assemble|compute|evaluate|pack_|unpack))"
        )
        glob_code = function_prefix_regex.sub(f"{node.name}_", glob_code)

        calloc_regex = re.compile(r"(?=(calloc))")
        glob_code = calloc_regex.sub("(bool*)", glob_code)

        tensor_order = [line for line in glob_code.split("\n") if "compute(" in line][0]
        tensor_order = tensor_order.split("(")[1]
        tensor_order = tensor_order.split(")")[0]
        tensor_order = tensor_order.split(",")
        tensor_order = [f"{t.split('*')[1]}_tensor" for t in tensor_order]

        if PROFILE:
            tensor_init_code+= "auto __tin_begin = std::chrono::high_resolution_clock::now();\n"
            tensor_deinit_code+= "auto __tin_deinit = std::chrono::high_resolution_clock::now();\n"
            tensor_deinit_code += "const std::chrono::duration<double> _tin_init = __tin_begin - __tin_init;\n"
            tensor_deinit_code += "const std::chrono::duration<double> _tin_run = __tin_end - __tin_begin;\n"
            tensor_deinit_code += "const std::chrono::duration<double> _tin_deinit = __tin_deinit - __tin_end;\n"
            tensor_deinit_code+= f'printf("{node.name}: init: %f ms; run: %f ms; deinit: %f ms;\\n", _tin_init * 1000, _tin_run * 1000, _tin_deinit * 1000);\n'

        code = f"""{tensor_init_code}{node.name}_{"compute" if output_dense else "evaluate"}({', '.join(tensor_order)});
        
        {tensor_deinit_code}"""

        tasklet = state.add_tasklet(
            "taco_code",
            [f"{inp}_task" for inp in inputs.keys()],
            [f"{outp}_task" for outp in outputs.keys()],
            code=code,
            language=dtypes.Language.CPP,
            code_global=glob_code,
        )

        for name, input in inputs.items():
            state.add_edge(
                input, None, tasklet, f"{name}_task", memlet=Memlet(data=input.data)
            )
        for name, output in outputs.items():
            state.add_edge(
                tasklet, f"{name}_task", output, None, memlet=Memlet(data=output.data)
            )

        return sdfg


@replaces("TensorIndexNotation")
@replaces("dace.libraries.sparse.TensorIndexNotation")
def gemv_libnode(
    pv: "ProgramVisitor", sdfg: SDFG, state: SDFGState, name, tin_expr, extra_taco_args = [], **kwargs
):
    
    extra_taco_args = [arg.value for arg in extra_taco_args]

    tin_expr: str = tin_expr.value
    output_name = tin_expr.split("(", 1)[0]

    accesses = {k: (v, state.add_access(v)) for k, v in kwargs.items()}

    tin_node = TensorIndexNotation(name.value, tin_expr, extra_taco_args)
    state.add_node(tin_node)

    for name, (data, data_node) in accesses.items():
        if name == output_name:
            tin_node.add_out_connector(f"tin_{name}")
            state.add_edge(tin_node, f"tin_{name}", data_node, None, Memlet(data=data))
        else:
            tin_node.add_in_connector(f"tin_{name}")
            state.add_edge(data_node, None, tin_node, f"tin_{name}", Memlet(data=data))

    return []
