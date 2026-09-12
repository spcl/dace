# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TensorDot library node and its pure / TTGT / cuTENSOR expansions."""
import collections
import dace

from dace.libraries.linalg import environments
from dace import library, nodes, properties, symbolic
from dace.utils import prod as _prod
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.ordered import OrderedSet


@library.expansion
class ExpandPure(ExpandTransformation):
    """ Implements the pure expansion of TensorDot library node. """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor, left_ext, right_ext, out_ext = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        # Shape from the memlet, strides from the container: the connector sees the SUBSET the edge
        # carries, laid out the way the array it is cut from is laid out. Everything below reads
        # these connector descriptors, so the whole expansion follows the subset from here.
        _, left_arr = sdfg.add_array("_left_tensor",
                                     left_ext,
                                     left_tensor.dtype,
                                     left_tensor.storage,
                                     strides=left_tensor.strides)
        _, right_arr = sdfg.add_array("_right_tensor",
                                      right_ext,
                                      right_tensor.dtype,
                                      right_tensor.storage,
                                      strides=right_tensor.strides)
        _, out_arr = sdfg.add_array("_out_tensor",
                                    out_ext,
                                    out_tensor.dtype,
                                    out_tensor.storage,
                                    strides=out_tensor.strides)

        init_state = sdfg.add_state(f"{node.label}_init", is_start_block=True)
        init_state.add_mapped_tasklet(
            f"{node.label}_init_tasklet", {
                f"__i{i}": f"0:{symstr(s)}"
                for i, s in enumerate(out_arr.shape)
            }, {},
            '__out = 0',
            {'__out': dace.Memlet(expr=f"_out_tensor[{','.join(['__i%d' % i for i in range(len(out_arr.shape))])}]")},
            external_edges=True)

        state = sdfg.add_state(f"{node.label}_state")
        sdfg.add_edge(init_state, state, dace.InterstateEdge())

        outer_map_shape = list([s for i, s in enumerate(left_arr.shape) if i not in node.left_axes])
        outer_map_shape.extend([s for i, s in enumerate(right_arr.shape) if i not in node.right_axes])
        outer_map_params = [f"__oi{i}" for i in range(len(outer_map_shape))]
        outer_map_rng = {i: f"0:{symstr(s)}" for i, s in zip(outer_map_params, outer_map_shape)}
        inner_map_shape = list([left_arr.shape[i] for i in node.left_axes])
        inner_map_params = [f"__ii{i}" for i in range(len(inner_map_shape))]
        inner_map_rng = {i: f"0:{symstr(s)}" for i, s in zip(inner_map_params, inner_map_shape)}

        left_idx = outer_map_params[:len(left_arr.shape) - len(node.left_axes)]
        left_dict = {j: inner_map_params[i] for i, j in enumerate(node.left_axes)}
        left_sorted_dict = collections.OrderedDict(sorted(left_dict.items()))
        for k, v in left_sorted_dict.items():
            left_idx.insert(k, v)
        right_idx = outer_map_params[len(left_arr.shape) - len(node.left_axes):]
        right_dict = {j: inner_map_params[i] for i, j in enumerate(node.right_axes)}
        right_sorted_dict = collections.OrderedDict(sorted(right_dict.items()))
        for k, v in right_sorted_dict.items():
            right_idx.insert(k, v)
        out_idx = outer_map_params
        if node.permutation:
            out_idx = [outer_map_params[i] for i in node.permutation]

        left_mem = dace.Memlet(expr=f"_left_tensor[{','.join(left_idx)}]")
        right_mem = dace.Memlet(expr=f"_right_tensor[{','.join(right_idx)}]")
        out_mem = dace.Memlet(expr=f"_out_tensor[{','.join(out_idx)}]", wcr="lambda x, y: x + y")
        inputs = {"_left": left_mem, "_right": right_mem}
        outputs = {"_out": out_mem}
        code = f"_out = _left * _right"
        state.add_mapped_tasklet(f"{node.label}_tasklet", {
            **outer_map_rng,
            **inner_map_rng
        },
                                 inputs,
                                 code,
                                 outputs,
                                 external_edges=True)

        return sdfg


@library.expansion
class ExpandTTGT(ExpandTransformation):
    """
    Expands the TensorDot library node to TensorTranspose + GEMM operations.
    TTGT stands for Transpose-Transpose-GEMM-Transpose.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor, left_ext, right_ext, out_ext = node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        # Shape from the memlet, strides from the container: the connector sees the SUBSET the edge
        # carries, laid out the way the array it is cut from is laid out. Everything below reads
        # these connector descriptors, so the whole expansion follows the subset from here.
        _, left_arr = sdfg.add_array("_left_tensor",
                                     left_ext,
                                     left_tensor.dtype,
                                     left_tensor.storage,
                                     strides=left_tensor.strides)
        _, right_arr = sdfg.add_array("_right_tensor",
                                      right_ext,
                                      right_tensor.dtype,
                                      right_tensor.storage,
                                      strides=right_tensor.strides)
        _, out_arr = sdfg.add_array("_out_tensor",
                                    out_ext,
                                    out_tensor.dtype,
                                    out_tensor.storage,
                                    strides=out_tensor.strides)

        from dace.frontend.python.replacements.array_manipulation import _transpose
        # NOTE: We use the numpy.transpose replacement because:
        # (1) It will return the tensor itself if transposition is uncessary.
        # (2) It will use matrix transpose operation for 2-mode tensors.
        state = sdfg.add_state(f"{node.label}_inp_transpose_state", is_start_block=True)

        if node.left_axes == list(range(len(node.left_axes))):
            transA = True
        else:
            transA = False
        if node.right_axes == list(range(len(right_arr.shape) - len(node.right_axes), len(right_arr.shape))):
            transB = True
        else:
            transB = False

        if transA:
            left_tt = "_left_tensor"
            left_tt_arr = left_arr
        else:
            left_axes = [i for i in range(len(left_arr.shape)) if i not in node.left_axes]
            left_axes.extend(node.left_axes)
            left_tt = _transpose(None, sdfg, state, "_left_tensor", left_axes, outname="ttgt_left_transposed")
            left_tt_arr = sdfg.arrays[left_tt]

        if transB:
            right_tt = "_right_tensor"
            right_tt_arr = right_arr
        else:
            right_axes = list(node.right_axes)
            right_axes.extend([i for i in range(len(right_arr.shape)) if i not in node.right_axes])
            right_tt = _transpose(None, sdfg, state, "_right_tensor", right_axes, outname="ttgt_right_transposed")
            right_tt_arr = sdfg.arrays[right_tt]

        from dace.libraries.blas import Gemm  # Avoid import loop
        prv_state = state
        state = sdfg.add_state(f"{node.label}_gemm_state")
        sdfg.add_edge(prv_state, state, dace.InterstateEdge())

        if transA:
            left_shape = [
                _prod(left_tt_arr.shape[:len(node.left_axes)]),
                _prod(left_tt_arr.shape[len(node.left_axes):])
            ]
            left_strides = [left_tt_arr.strides[len(node.left_axes) - 1], left_tt_arr.strides[-1]]
        else:
            left_shape = [
                _prod(left_tt_arr.shape[:-len(node.left_axes)]),
                _prod(left_tt_arr.shape[len(left_tt_arr.shape) - len(node.left_axes):])
            ]
            left_strides = [left_tt_arr.strides[-len(node.left_axes) - 1], left_tt_arr.strides[-1]]
        left_vname, left_view = sdfg.add_view(left_tt,
                                              left_shape,
                                              left_tt_arr.dtype,
                                              left_tt_arr.storage,
                                              strides=left_strides,
                                              find_new_name=True)
        left_anode = state.add_read(left_tt)
        left_vnode = state.add_access(left_vname)
        state.add_edge(left_anode, None, left_vnode, 'views', dace.Memlet.from_array(left_tt, left_tt_arr))

        if transB:
            right_shape = [
                _prod(right_tt_arr.shape[:-len(node.right_axes)]),
                _prod(right_tt_arr.shape[len(right_tt_arr.shape) - len(node.right_axes):])
            ]
            right_strides = [right_tt_arr.strides[-len(node.right_axes) - 1], right_tt_arr.strides[-1]]
        else:
            right_shape = [
                _prod(right_tt_arr.shape[0:len(node.right_axes)]),
                _prod(right_tt_arr.shape[len(node.right_axes):])
            ]
            right_strides = [right_tt_arr.strides[len(node.right_axes) - 1], right_tt_arr.strides[-1]]
        right_vname, right_view = sdfg.add_view(right_tt,
                                                right_shape,
                                                right_tt_arr.dtype,
                                                right_tt_arr.storage,
                                                strides=right_strides,
                                                find_new_name=True)
        right_anode = state.add_read(right_tt)
        right_vnode = state.add_access(right_vname)
        state.add_edge(right_anode, None, right_vnode, 'views', dace.Memlet.from_array(right_tt, right_tt_arr))

        tasklet = Gemm('_GEMM_', cin=False, transA=transA, transB=transB)
        state.add_edge(left_vnode, None, tasklet, '_a', dace.Memlet.from_array(left_vname, left_view))
        state.add_edge(right_vnode, None, tasklet, '_b', dace.Memlet.from_array(right_vname, right_view))

        # Output handling
        out_shape = []
        if transA:
            out_shape.append(left_shape[1])
        else:
            out_shape.append(left_shape[0])
        if transB:
            out_shape.append(right_shape[0])
        else:
            out_shape.append(right_shape[1])
        if node.permutation and node.permutation != list(range(len(node.permutation))):
            dot_shape = [s for i, s in enumerate(left_arr.shape) if i not in node.left_axes]
            dot_shape.extend([s for i, s in enumerate(right_arr.shape) if i not in node.right_axes])
            dot_name, dot_arr = sdfg.add_temp_transient(dot_shape, out_arr.dtype, out_arr.storage)
            out_strides = [dot_arr.strides[len(left_tt_arr.shape) - len(node.left_axes) - 1], dot_arr.strides[-1]]
            dot_vname, dot_view = sdfg.add_view('__gemm_out',
                                                out_shape,
                                                dot_arr.dtype,
                                                dot_arr.storage,
                                                strides=out_strides,
                                                find_new_name=True)
            dot_anode = state.add_access(dot_name)
            dot_vnode = state.add_access(dot_vname)
            state.add_edge(tasklet, '_c', dot_vnode, None, dace.Memlet.from_array(dot_vname, dot_view))
            state.add_edge(dot_vnode, 'views', dot_anode, None, dace.Memlet.from_array(dot_name, dot_arr))
            out_node = state.add_write('_out_tensor')
            # Avoid import loop: TensorTranspose is a sibling node in dace.libraries.linalg
            from dace.libraries.linalg import TensorTranspose
            tasklet = TensorTranspose('_TensorTranspose', node.permutation)
            state.add_edge(dot_anode, None, tasklet, '_inp_tensor', dace.Memlet.from_array(dot_name, dot_arr))
            state.add_edge(tasklet, '_out_tensor', out_node, None, dace.Memlet.from_array('_out_tensor', out_arr))
        else:
            out_strides = [out_arr.strides[len(left_tt_arr.shape) - len(node.left_axes) - 1], out_arr.strides[-1]]
            out_vname, out_view = sdfg.add_view('__gemm_out',
                                                out_shape,
                                                out_arr.dtype,
                                                out_arr.storage,
                                                strides=out_strides,
                                                find_new_name=True)
            out_anode = state.add_access('_out_tensor')
            out_vnode = state.add_access(out_vname)
            state.add_edge(tasklet, '_c', out_vnode, None, dace.Memlet.from_array(out_vname, out_view))
            state.add_edge(out_vnode, 'views', out_anode, None, dace.Memlet.from_array('_out_tensor', out_arr))

        return sdfg


@library.expansion
class ExpandGPUTensorDot(ExpandTransformation):
    """
    Implements the TensorDot library node using the vendor tensor library (``Contract``)
    for CUDA-compatible GPUs. Requires cuTENSOR >= 2.0.

    The contraction expresses:
        D_{modesD} = alpha * A_{modesA} * B_{modesB} + beta * C_{modesC}
    where in this kernel C and D share the same buffer and modes
    (i.e. D = alpha * A * B with beta = 0).
    """

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor, left_ext, right_ext, out_ext = node.validate(parent_sdfg, parent_state)

        dtype = out_tensor.dtype.base_type
        if dtype not in cls.environments[0].TYPE_MAP:
            raise NotImplementedError(f"{cls.vendor} TensorDot does not support dtype {dtype}; supported: "
                                      f"{sorted(str(t) for t in cls.environments[0].TYPE_MAP)}")
        tensor_dtype, compute_desc, scalar_type = cls.environments[0].TYPE_MAP[dtype]

        alpha = f"({scalar_type})1.0"
        beta = f"({scalar_type})0.0"
        abtext = f"""
            {scalar_type} alpha = {alpha};
            {scalar_type} beta = {beta};
        """

        left_modes = list(range(len(left_ext)))
        right_modes = [
            node.left_axes[node.right_axes.index(i)] if i in node.right_axes else len(left_ext) + i
            for i in range(len(right_ext))
        ]
        out_modes = [i for i in left_modes if i not in node.left_axes]
        out_modes.extend([i for i in right_modes if i not in node.left_axes])
        if node.permutation and node.permutation != list(range(len(node.permutation))):
            out_modes = [out_modes[i] for i in node.permutation]

        modes = f"""
            std::vector<int32_t> modeA{{{','.join(str(m) for m in left_modes)}}};
            std::vector<int32_t> modeB{{{','.join(str(m) for m in right_modes)}}};
            std::vector<int32_t> modeC{{{','.join(str(m) for m in out_modes)}}};
        """

        # Modes are dense indices into the concatenated shapes, so a vector indexes them directly.
        extents = f"std::vector<int64_t> extent({len(left_ext) + len(right_ext)});\n"
        for i, s in zip(left_modes, left_ext):
            extents += f"extent[{i}] = {s};\n"
        for i, s in zip(right_modes, right_ext):
            if i in node.right_axes:
                continue
            extents += f"extent[{i}] = {s};\n"
        extents += f"""
            std::vector<int64_t> extentA;
            for (auto mode : modeA) extentA.push_back(extent[mode]);
            std::vector<int64_t> extentB;
            for (auto mode : modeB) extentB.push_back(extent[mode]);
            std::vector<int64_t> extentC;
            for (auto mode : modeC) extentC.push_back(extent[mode]);
        """

        extents += f"""
            std::vector<int64_t> stridesA{{{','.join(str(s) for s in left_tensor.strides)}}};
            std::vector<int64_t> stridesB{{{','.join(str(s) for s in right_tensor.strides)}}};
            std::vector<int64_t> stridesC{{{','.join(str(s) for s in out_tensor.strides)}}};
        """

        # cuTENSOR v2: descriptors take an alignment hint (bytes) instead of
        # a per-pointer query; 256 is safe for all CUDA allocations.
        tdesc = f"""
            {cls.vendor_lower}TensorDescriptor_t descA, descB, descC;
            {cls.check}({cls.vendor_lower}CreateTensorDescriptor(
                {cls.handle}, &descA, modeA.size(),
                extentA.data(), stridesA.data(), {tensor_dtype}, 256));
            {cls.check}({cls.vendor_lower}CreateTensorDescriptor(
                {cls.handle}, &descB, modeB.size(),
                extentB.data(), stridesB.data(), {tensor_dtype}, 256));
            {cls.check}({cls.vendor_lower}CreateTensorDescriptor(
                {cls.handle}, &descC, modeC.size(),
                extentC.data(), stridesC.data(), {tensor_dtype}, 256));
        """

        # Contraction descriptor: D = alpha * A * B + beta * C; here D == C.
        cdesc = f"""
            {cls.vendor_lower}OperationDescriptor_t opDesc;
            {cls.check}({cls.vendor_lower}CreateContraction(
                {cls.handle}, &opDesc,
                descA, modeA.data(), {cls.op_identity},
                descB, modeB.data(), {cls.op_identity},
                descC, modeC.data(), {cls.op_identity},
                descC, modeC.data(),
                {compute_desc}));
        """

        workspace = f"""
            {cls.vendor_lower}PlanPreference_t planPref;
            {cls.check}({cls.vendor_lower}CreatePlanPreference(
                {cls.handle}, &planPref,
                {cls.algo_default}, {cls.jit_default}));
            uint64_t worksize = 0;
            {cls.check}({cls.vendor_lower}EstimateWorkspaceSize(
                {cls.handle}, opDesc, planPref,
                {cls.workspace_default}, &worksize));
            void *work = nullptr;
            if (worksize > 0) gpuMalloc(&work, worksize);
        """

        execute = f"""
            {cls.vendor_lower}Plan_t plan;
            {cls.check}({cls.vendor_lower}CreatePlan(
                {cls.handle}, &plan, opDesc, planPref, worksize));
            {cls.vendor_lower}Status_t err = {cls.vendor_lower}Contract(
                {cls.handle}, plan,
                (const void*)&alpha, _left_tensor, _right_tensor,
                (const void*)&beta,  _out_tensor,  _out_tensor,
                work, worksize, __dace_current_stream);
            if (err != {cls.status_success}) {{
                printf("ERROR: %s\\n", {cls.vendor_lower}GetErrorString(err));
            }}
            {cls.vendor_lower}DestroyPlan(plan);
            {cls.vendor_lower}DestroyPlanPreference(planPref);
            {cls.vendor_lower}DestroyOperationDescriptor(opDesc);
            {cls.vendor_lower}DestroyTensorDescriptor(descC);
            {cls.vendor_lower}DestroyTensorDescriptor(descB);
            {cls.vendor_lower}DestroyTensorDescriptor(descA);
            if (work) gpuFree(work);
        """

        code = f"{cls.environments[0].handle_setup_code(node)}{abtext}{modes}{extents}{tdesc}{cdesc}{workspace}{execute}"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@library.expansion
class ExpandTBLIS(ExpandTransformation):
    """TensorDot via TBLIS ``tblis_tensor_mult`` -- native, transpose-free CPU contraction.

    Reuses the cuTENSOR mode assignment, then renders integer modes as single-char index
    labels (the einsum-style strings TBLIS consumes). ``C`` is initialized scaled by 0 so the
    call overwrites (``C = A*B``) rather than accumulating (``C = A*B + C``).
    """

    environments = [environments.TBLIS]

    # dace dtype -> (tblis init suffix, C element type)
    TYPE_MAP = {dace.float32: ("s", "float"), dace.float64: ("d", "double")}

    @staticmethod
    def contraction_labels(num_left, num_right, left_axes, right_axes, permutation=None):
        """Einsum-style index labels ``(idx_a, idx_b, idx_c)`` for the contraction.

        Contracted right modes borrow the matching left mode's label (shared index = summed);
        free modes get fresh labels; the output is the free-left ++ free-right modes, permuted.
        Correct iff ``np.einsum(f'{idx_a},{idx_b}->{idx_c}', A, B)`` equals the tensordot.
        """
        left_modes = list(range(num_left))
        right_modes = [left_axes[right_axes.index(i)] if i in right_axes else num_left + i for i in range(num_right)]
        out_modes = [i for i in left_modes if i not in left_axes]
        out_modes.extend([i for i in right_modes if i not in left_axes])
        if permutation and permutation != list(range(len(permutation))):
            out_modes = [out_modes[i] for i in permutation]
        label_of = {}
        for m in left_modes + right_modes:
            if m not in label_of:
                if len(label_of) >= 26:
                    raise NotImplementedError("TBLIS TensorDot: more than 26 distinct modes")
                label_of[m] = chr(ord('a') + len(label_of))
        return (''.join(label_of[m] for m in left_modes), ''.join(label_of[m] for m in right_modes),
                ''.join(label_of[m] for m in out_modes))

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        left_tensor, right_tensor, out_tensor, left_ext, right_ext, out_ext = node.validate(parent_sdfg, parent_state)

        dtype = out_tensor.dtype.base_type
        if dtype not in ExpandTBLIS.TYPE_MAP:
            raise NotImplementedError(f"TBLIS TensorDot does not support dtype {dtype}; supported: "
                                      f"{sorted(str(t) for t in ExpandTBLIS.TYPE_MAP)}")
        suffix, ctype = ExpandTBLIS.TYPE_MAP[dtype]

        idx_a, idx_b, idx_c = ExpandTBLIS.contraction_labels(len(left_ext), len(right_ext), node.left_axes,
                                                             node.right_axes, node.permutation)

        def carr(name, vals):
            if len(vals) == 0:
                return f"ptrdiff_t* {name} = nullptr;"
            return f"ptrdiff_t {name}[] = {{{', '.join(symstr(v) for v in vals)}}};"

        code = f"""
            {carr('lenA', list(left_ext))}
            {carr('strideA', list(left_tensor.strides))}
            {carr('lenB', list(right_ext))}
            {carr('strideB', list(right_tensor.strides))}
            {carr('lenC', list(out_ext))}
            {carr('strideC', list(out_tensor.strides))}
            using namespace tblis;
            tblis_tensor A, B, C;
            tblis_init_tensor_{suffix}(&A, {len(left_ext)}, lenA, ({ctype}*)_left_tensor, strideA);
            tblis_init_tensor_{suffix}(&B, {len(right_ext)}, lenB, ({ctype}*)_right_tensor, strideB);
            tblis_init_tensor_scaled_{suffix}(&C, ({ctype})0, {len(out_ext)}, lenC, ({ctype}*)_out_tensor, strideC);
            tblis_tensor_mult(NULL, NULL, &A, "{idx_a}", &B, "{idx_b}", &C, "{idx_c}");
        """

        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP,
                                       side_effects=True)


@dace.library.expansion
class ExpandCuTensor(ExpandGPUTensorDot):
    environments = [environments.cuTensor]
    vendor = "cuTENSOR"
    vendor_lower = "cutensor"
    handle = "__dace_cutensor_handle"
    check = "dace::linalg::CheckCuTensorError"
    op_identity = "CUTENSOR_OP_IDENTITY"
    algo_default = "CUTENSOR_ALGO_DEFAULT"
    jit_default = "CUTENSOR_JIT_MODE_DEFAULT"
    workspace_default = "CUTENSOR_WORKSPACE_DEFAULT"
    status_success = "CUTENSOR_STATUS_SUCCESS"


@dace.library.expansion
class ExpandHipTensorDot(ExpandGPUTensorDot):
    environments = [environments.hipTensor]
    vendor = "hipTensor"
    vendor_lower = "hiptensor"
    handle = "__dace_hiptensor_handle"
    check = "dace::linalg::CheckHipTensorError"
    op_identity = "HIPTENSOR_OP_IDENTITY"
    algo_default = "HIPTENSOR_ALGO_DEFAULT"
    #: NONE, not DEFAULT -- hipTensor accepts the DEFAULT name and then refuses the plan at
    #: execution, measured for every rank and dtype (see the ttranspose node).
    jit_default = "HIPTENSOR_JIT_MODE_NONE"
    workspace_default = "HIPTENSOR_WORKSPACE_DEFAULT"
    status_success = "HIPTENSOR_STATUS_SUCCESS"


@library.node
class TensorDot(nodes.LibraryNode):
    """ Implements tensor dot-product. """

    implementations = {
        "pure": ExpandPure,
        "TTGT": ExpandTTGT,
        "cuTENSOR": ExpandCuTensor,
        "hipTENSOR": ExpandHipTensorDot,
        "TBLIS": ExpandTBLIS
    }
    # Deliberately None: the ``library.linalg.default_implementation`` config knob must stay able to
    # select this node's TTGT / cuTENSOR / pure lowering, and a node-class default would shadow it.
    # The library-wide default (``OpenBLAS``, meant for the LAPACK-backed Cholesky/Solve/Inv) is not
    # one of this node's implementations; ``LibraryNode.expand`` falls back to ``pure`` for that case.
    #
    # TODO: no corpus kernel reaches TensorDot yet, so the fallback above is the only thing keeping
    # ``np.tensordot`` working on CPU and it is exercised by tests alone. Pick the lowering here on
    # merit (TTGT vs pure) once a corpus kernel actually measures it.
    default_implementation = None

    left_axes = properties.ListProperty(element_type=int, default=[], desc="Left tensor's contracting modes")
    right_axes = properties.ListProperty(element_type=int, default=[], desc="Right tensor's contracting modes")
    permutation = properties.ListProperty(element_type=int,
                                          allow_none=True,
                                          default=None,
                                          desc="Permutation of the output tensor")

    def __init__(self, name, left_axes=[], right_axes=[], permutation=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs=OrderedSet(('_left_tensor', '_right_tensor')),
                         outputs={"_out_tensor"},
                         **kwargs)
        self.left_axes = left_axes
        self.right_axes = right_axes
        self.permutation = permutation

    def validate(self, sdfg, state):
        """
        Validates the tensor dot-product operation.
        :return: A tuple (left_tensor, right_tensor, out_tensor, left_shape, right_shape, out_shape) -- the
                 data descriptors in the parent SDFG, and the extents the memlets actually move.
        """

        left_tensor, right_tensor, out_tensor = None, None, None
        left_shape, right_shape, out_shape = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_out_tensor":
                out_tensor = sdfg.arrays[e.data.data]
                # The extents come from the SUBSET, not the descriptor: a contraction may write into
                # a slice of a larger container, and comparing the dot-product shape against the
                # container rejects a write that is correct.
                out_shape = e.data.subset.size()
        for e in state.in_edges(self):
            if e.dst_conn == "_left_tensor":
                left_tensor = sdfg.arrays[e.data.data]
                left_shape = e.data.subset.size()
            elif e.dst_conn == "_right_tensor":
                right_tensor = sdfg.arrays[e.data.data]
                right_shape = e.data.subset.size()

        if not left_tensor or not right_tensor:
            raise ValueError("Missing the input tensors.")
        if not out_tensor:
            raise ValueError("Missing the output tensor.")

        if left_tensor.dtype != right_tensor.dtype or left_tensor.dtype != out_tensor.dtype:
            raise TypeError("The datatype of the input and output tensors must match.")
        # TODO: Check disabled due to causing issues with CUDA + MPI. Revisit in the future.
        # if left_tensor.storage != right_tensor.storage or left_tensor.storage != out_tensor.storage:
        #     raise ValueError("The storage of the input and output tensors must match.")

        if any(a >= len(left_shape) or a < 0 for a in self.left_axes):
            raise ValueError("Axes for left tensor are out-of-bounds.")
        if any(a >= len(right_shape) or a < 0 for a in self.right_axes):
            raise ValueError("Axes for right tensor are out-of-bounds.")
        if len(self.left_axes) != len(self.right_axes):
            raise ValueError("The input tensors must have the same number of contracting modes.")
        # Compared by NAME: one extent reaches the two sides through different rewrites and arrives
        # as two spellings that raw ``!=`` calls unequal, rejecting shapes that match.
        if any(
                symbolic.inequal_symbols(left_shape[l], right_shape[r])
                for l, r in zip(self.left_axes, self.right_axes)):
            raise ValueError("The input tensors' contracting modes must have the same length.")

        dot_shape = [s for i, s in enumerate(left_shape) if i not in self.left_axes]
        dot_shape.extend([s for i, s in enumerate(right_shape) if i not in self.right_axes])
        out_shape = list(out_shape)
        if len(dot_shape) != len(out_shape):
            raise ValueError("The intermediate (dot-product) and output tensors must have the same number of modes..")

        # # We check if the output shape is a permutation of a dot-product shape.
        # TODO: Check disabled due to causing issues with valid test cases. Revisit in the future.
        # # NOTE: Since the shapes may be symbolic, we cannot just sort and compare them.
        # for s in out_shape:
        #     try:
        #         idx = dot_shape.index(s)
        #         dot_shape.pop(idx)
        #     except ValueError:
        #         raise ValueError("The output tensor shape is not a permutation of the dot-product shape.")
        # if dot_shape:
        #     raise ValueError("The output tensor shape is not a permutation of the dot-product shape.")

        if not self.permutation:
            if not symbolic.shapes_equal(dot_shape, out_shape):
                raise ValueError("The shapes of the intermediate (dot-product) and output tensors must match.")
        else:
            # NOTE: If the output tensor is transposed, then the permutation must be given explicitely. The permutation
            # can only be inferred if each tensor mode has different length, which should never be assumed.
            if len(out_shape) != len(self.permutation):
                raise ValueError(
                    "The permutation list property must have as many elements as the number of output tensor modes.")
            if sorted(self.permutation) != list(range(len(out_shape))):
                raise ValueError("The permutation list property is not a perimutation of the output tensor's modes.")
            transposed_shape = [dot_shape[p] for p in self.permutation]
            if not symbolic.shapes_equal(transposed_shape, out_shape):
                raise ValueError(
                    "The permutation of the intermediate (dot-product) shape does not match the output shape.")

        return left_tensor, right_tensor, out_tensor, left_shape, right_shape, out_shape
