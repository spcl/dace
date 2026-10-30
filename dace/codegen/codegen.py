# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import json

import dace
from dace import config, data, dtypes
from dace.codegen import exceptions as exc
from dace.codegen.codeobject import CodeObject
from dace.codegen.instrumentation import InstrumentationProvider
from dace.codegen.targets import framecode
from dace.config import Config
from dace.sdfg import SDFG, infer_types
from dace.sdfg.state import SDFGState
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising


def generate_headers(sdfg: SDFG, frame: framecode.DaCeCodeGenerator) -> str:
    """Generate a header file for the SDFG"""
    proto = ""
    proto += "#include <dace/dace.h>\n"
    init_params = (sdfg.name, sdfg.name, sdfg.init_signature(free_symbols=frame.free_symbols(sdfg)))
    call_params = sdfg.signature(with_types=True, for_call=False, arglist=frame.arglist)
    if len(call_params) > 0:
        call_params = ", " + call_params
    params = (sdfg.name, sdfg.name, call_params)
    exit_params = (sdfg.name, sdfg.name)
    proto += f"typedef void * {sdfg.name}Handle_t;\n"
    proto += 'extern "C" {}Handle_t __dace_init_{}({});\n'.format(*init_params)
    proto += 'extern "C" int __dace_exit_{}({}Handle_t handle);\n'.format(*exit_params)
    proto += 'extern "C" void __program_{}({}Handle_t handle{});\n'.format(*params)
    return proto


def generate_dummy(sdfg: SDFG, frame: framecode.DaCeCodeGenerator) -> str:
    """Generates a C program calling this SDFG. Since we do not
    know the purpose/semantics of the program, we allocate
    the right types and and guess values for scalars.
    """
    al = frame.arglist
    init_params = sdfg.init_signature(for_call=True, free_symbols=frame.free_symbols(sdfg))
    params = sdfg.signature(with_types=False, for_call=True, arglist=frame.arglist)
    if len(params) > 0:
        params = ", " + params

    allocations = ""
    deallocations = ""

    # first find all scalars and set them to 42
    for argname, arg in al.items():
        if isinstance(arg, data.Scalar):
            allocations += "    " + str(arg.as_arg(name=argname, with_types=True)) + " = 42;\n"

    # allocate the array args using calloc
    for argname, arg in al.items():
        if isinstance(arg, data.Array):
            from dace.codegen.targets import cpp

            dims_mul = cpp.sym2cpp(functools.reduce(lambda a, b: a * b, arg.shape, 1))
            basetype = str(arg.dtype)
            allocations += (
                "    "
                + str(arg.as_arg(name=argname, with_types=True))
                + " = ("
                + basetype
                + "*) calloc("
                + dims_mul
                + ", sizeof("
                + basetype
                + ")"
                + ");\n"
            )
            deallocations += "    free(" + argname + ");\n"

    return f"""#include <cstdlib>
#include "../include/{sdfg.name}.h"

int main(int argc, char **argv) {{
    {sdfg.name}Handle_t handle;
    int err;
{allocations}

    handle = __dace_init_{sdfg.name}({init_params});
    __program_{sdfg.name}(handle{params});
    err = __dace_exit_{sdfg.name}(handle);

{deallocations}

    return err;
}}
"""


def _get_codegen_targets(sdfg: SDFG, frame: framecode.DaCeCodeGenerator):
    """
    Queries all code generation targets in this SDFG and all nested SDFGs,
    as well as instrumentation providers, and stores them in the frame code generator.
    """
    disp = frame._dispatcher
    provider_mapping = InstrumentationProvider.get_provider_mapping()
    disp.instrumentation[dtypes.InstrumentationType.No_Instrumentation] = None
    disp.instrumentation[dtypes.DataInstrumentationType.No_Instrumentation] = None
    for node, parent in sdfg.all_nodes_recursive():
        # Query nodes and scopes
        if isinstance(node, SDFGState):
            frame.targets.add(disp.get_state_dispatcher(node.sdfg, node))
        elif isinstance(node, dace.nodes.EntryNode):
            frame.targets.add(disp.get_scope_dispatcher(node.schedule))
        elif isinstance(node, dace.nodes.Node):
            state: SDFGState = parent
            nsdfg = state.parent
            frame.targets.add(disp.get_node_dispatcher(nsdfg, state, node))

        # Array allocation
        if isinstance(node, dace.nodes.AccessNode):
            state: SDFGState = parent
            nsdfg = state.parent
            desc = node.desc(nsdfg)
            frame.targets.add(disp.get_array_dispatcher(desc.storage))

        # Copies and memlets - via access nodes and tasklets
        # To avoid duplicate checks, only look at outgoing edges of access nodes and tasklets
        if isinstance(node, (dace.nodes.AccessNode, dace.nodes.Tasklet)):
            state: SDFGState = parent
            for e in state.out_edges(node):
                if e.data.is_empty():
                    continue
                mtree = state.memlet_tree(e)
                if mtree.downwards:
                    # Rooted at src_node
                    for leaf_e in mtree.leaves():
                        dst_node = leaf_e.dst
                        if leaf_e.data.is_empty():
                            continue
                        tgt = disp.get_copy_dispatcher(node, dst_node, leaf_e, state.parent, state)
                        if tgt is not None:
                            frame.targets.add(tgt)
                else:
                    # Rooted at dst_node
                    dst_node = mtree.root().edge.dst
                    tgt = disp.get_copy_dispatcher(node, dst_node, e, state.parent, state)
                    if tgt is not None:
                        frame.targets.add(tgt)

        # Instrumentation-related query
        if isinstance(node, SDFGState):
            disp.instrumentation[node.symbol_instrument] = provider_mapping[node.symbol_instrument]
        # MapEntry and ConsumeEntry forward ``instrument`` to their Map/Consume object
        if isinstance(
            node,
            (
                SDFGState,
                dace.nodes.AccessNode,
                dace.nodes.Tasklet,
                dace.nodes.NestedSDFG,
                dace.nodes.MapEntry,
                dace.nodes.ConsumeEntry,
            ),
        ):
            disp.instrumentation[node.instrument] = provider_mapping[node.instrument]
        elif isinstance(node, dace.nodes.ConsumeExit):
            disp.instrumentation[node.consume.instrument] = provider_mapping[node.consume.instrument]
        elif isinstance(node, dace.nodes.MapExit):
            disp.instrumentation[node.map.instrument] = provider_mapping[node.map.instrument]

    # Query instrumentation provider of SDFG
    if sdfg.instrument != dtypes.InstrumentationType.No_Instrumentation:
        disp.instrumentation[sdfg.instrument] = provider_mapping[sdfg.instrument]


def sdfg_uses_gpu(sdfg: SDFG) -> bool:
    """True when any descriptor lives in GPU storage or any scope/library node carries a GPU schedule."""
    from dace.sdfg import nodes

    if any(
        desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared)
        for _, _, desc in sdfg.arrays_recursive()
    ):
        return True
    return any(
        isinstance(node, (nodes.EntryNode, nodes.ExitNode, nodes.LibraryNode)) and node.schedule in dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )


def unselected_cuda_target() -> str:
    """The CUDA target not chosen in ``compiler.cuda.implementation``. Both share the GPU schedule types,
    so instantiating both would register duplicate dispatchers."""
    cuda_impl = config.Config.get("compiler", "cuda", "implementation")
    if cuda_impl not in ("legacy", "experimental"):
        raise ValueError(
            f"Invalid compiler.cuda.implementation: {cuda_impl!r}. Please select one of 'legacy' or 'experimental'."
        )
    return "experimental_cuda" if cuda_impl == "legacy" else "cuda"


def changes_symbol_kind(state: SDFGState, node: "dace.nodes.NestedSDFG") -> bool:
    """Whether a symbol the nested SDFG declares is mapped to a value of another scalar kind (an integer fed a float,
    a signed symbol an unsigned value): the call converts the value, inlining would substitute it as is."""
    from dace.sdfg.type_inference import infer_expr_type

    outer = None
    for name, value in node.symbol_mapping.items():
        declared = node.sdfg.symbols.get(name)
        if declared is None:
            continue
        if outer is None:
            outer = state.symbols_defined_at(node)
        mapped = infer_expr_type(value, outer)
        if mapped is not None and not (
            type(declared) is type(mapped) is dtypes.typeclass
            and declared.as_numpy_dtype().kind == mapped.as_numpy_dtype().kind
        ):
            return True
    return False


def inline_host_nested_sdfgs(sdfg: SDFG) -> int:
    """Flattens the nested SDFGs the CPU target emits and returns how many were inlined.

    A nest inside a GPU scope stays, as the CUDA target emits it as a device function, and so does one that
    :func:`changes_symbol_kind`.
    """
    from dace.sdfg.scope import is_devicelevel_gpu
    from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
    from dace.transformation.interstate.sdfg_nesting import InlineSDFG

    pinned = [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.NestedSDFG)
        and not node.no_inline
        and isinstance(parent, SDFGState)
        and (is_devicelevel_gpu(parent.sdfg, parent, node) or changes_symbol_kind(parent, node))
    ]
    for node in pinned:
        node.no_inline = True
    try:
        return sdfg.apply_transformations_repeated(InlineSDFG) + sdfg.apply_transformations_repeated(
            InlineMultistateSDFG
        )
    finally:
        for node in pinned:
            node.no_inline = False


def generate_code(sdfg: SDFG, validate=True) -> list[CodeObject]:
    """
    Generates code as a list of code objects for a given SDFG.

    :param sdfg: The SDFG to use
    :param validate: If True, validates the SDFG before generating the code.
    :return: List of code objects that correspond to files to compile.
    """
    # Lowering is not program history; recorded, the first expansion deep-copies the SDFG into ``orig_sdfg``
    with config.set_temporary("store_history", value=False):
        return lower_and_generate_code(sdfg, validate)


def lower_and_generate_code(sdfg: SDFG, validate: bool) -> list[CodeObject]:
    from dace.codegen.target import TargetCodeGenerator  # Avoid import loop

    # Before compiling, validate SDFG correctness
    if validate:
        sdfg.validate()

    if Config.get_bool("testing", "serialization"):
        import difflib
        import filecmp
        import shutil
        import tempfile

        from dace.sdfg import SDFG

        with tempfile.TemporaryDirectory() as tmp_dir:
            sdfg.save(f"{tmp_dir}/test.sdfg", hash=False)
            sdfg2 = SDFG.from_file(f"{tmp_dir}/test.sdfg")
            sdfg2.save(f"{tmp_dir}/test2.sdfg", hash=False)

            if not filecmp.cmp(f"{tmp_dir}/test.sdfg", f"{tmp_dir}/test2.sdfg"):
                with open(f"{tmp_dir}/test.sdfg") as f1:
                    with open(f"{tmp_dir}/test2.sdfg") as f2:
                        data1 = json.dumps(json.load(f1), indent=2).splitlines(keepends=True)
                        data2 = json.dumps(json.load(f2), indent=2).splitlines(keepends=True)
                        diff = difflib.unified_diff(
                            data1, data2, fromfile="test.sdfg  (first save)", tofile="test2.sdfg (after roundtrip)"
                        )
                diff = "".join(diff)
                shutil.move(f"{tmp_dir}/test.sdfg", "test.sdfg")
                shutil.move(f"{tmp_dir}/test2.sdfg", "test2.sdfg")
                raise RuntimeError(f"SDFG serialization failed - files do not match:\n{diff}")

    if config.Config.get_bool("optimizer", "detect_control_flow"):
        # NOTE: This should likely be done either earlier in the future, or changed entirely in modular codegen.
        # It is being done here to ensure that for now the semantics of the setting are preserved and legacy tests,
        # where explicit control flow was not used, continue to work as expected.
        FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})

    # Before generating the code, run type inference on the SDFG connectors
    infer_types.infer_connector_types(sdfg)

    # Set default storage/schedule types in SDFG
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # Dynamic thread-block maps not lowered before code generation become thread-block maps that drive a work queue
    from dace.transformation.passes.lower_dynamic_thread_block_maps import LowerDynamicThreadBlockMaps

    LowerDynamicThreadBlockMaps().apply_pass(sdfg, {})

    # Give every implicit copy a node of its own, before the expansion below lowers it. An implicit
    # copy is a write no node performs, so an empty memlet ordering that write has nothing to point
    # at and the copy is free to move ahead of a write it must follow. The readable generator lowers
    # its copies after inlining instead: an expanded copy inside a nest changes what InlineSDFG accepts.
    readable = config.Config.get("compiler", "cpu", "implementation") == "experimental_readable"
    if config.Config.get_bool("compiler", "cpu", "explicit_copy") and not readable:
        from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

        InsertExplicitCopies().apply_pass(sdfg, {})
        # The nodes just inserted carry no inferred connector types, and the expansion below reads
        # them to decide pointer vs. value. Storage defaults above already hold.
        infer_types.infer_connector_types(sdfg)

    # GPU device maps nested in kernels become guarded or grid-strided parts of their kernel
    from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})
    # Lower memlet access policies (e.g. loop-carried address cursors) to ordinary SDFG constructs on this
    # code-generation copy of the SDFG. This is a no-op unless a memlet carries a non-default policy. It must
    # follow InsertExplicitCopies (which moves the policy of a lifted copy onto the library node's operand
    # memlet) and precede library-node expansion (whose generated code addresses the rewritten operands).
    from dace.transformation.passes.memlet_access_policy_lowering import LowerMemletAccessPolicies

    LowerMemletAccessPolicies().apply_pass(sdfg, {})

    # Recursively expand library nodes that have not yet been expanded
    sdfg.expand_library_nodes()

    # After expansion, run another pass of connector/type inference
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # Experimental readable generator: flatten nested SDFGs (so the connector-free + index-function
    # lowering applies uniformly, not stopping at NSDFG boundaries), then mark write-once data
    # const/constexpr and inline tasklet connectors. After library expansion so post-expansion
    # tasklets are seen; affects CPU and GPU-kernel tasklets alike.
    if readable:
        from dace.transformation.passes.canonicalize_nested_index_names import CanonicalizeNestedIndexNames
        from dace.transformation.passes.inline_tasklet_connectors import InlineTaskletConnectors
        from dace.transformation.passes.promote_constant_transients import PromoteConstantTransients

        # Data instrumentation saves at the access nodes the user instrumented; inlining would replace them
        if not any(
            isinstance(node, dace.nodes.AccessNode)
            and node.instrument != dtypes.DataInstrumentationType.No_Instrumentation
            for node, _ in sdfg.all_nodes_recursive()
        ):
            inline_host_nested_sdfgs(sdfg)
        if config.Config.get_bool("compiler", "cpu", "explicit_copy"):
            from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

            InsertExplicitCopies().apply_pass(sdfg, {})
            infer_types.infer_connector_types(sdfg)
            sdfg.expand_library_nodes()
        infer_types.infer_connector_types(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        # GPU preprocessing, before the inlining below freezes accesses into tasklet code. Single-value transients
        # become Scalars, kernel outputs are widened back to length-1 arrays (a by-value Scalar cannot live in device
        # memory), and in-kernel arrays are lifted out of their kernels.
        if sdfg_uses_gpu(sdfg):
            from dace.transformation.pass_pipeline import Pipeline
            from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import (
                InferDefaultSchedulesAndStorages,
            )
            from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
            from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel
            from dace.transformation.passes.scalar_promotion import PromoteScalarOutputsToArrays

            ConvertLengthOneArraysToScalars(skip_gpu_outputs=True).apply_pass(sdfg, {})
            promote_gpu_scalars = PromoteScalarOutputsToArrays()
            promote_gpu_scalars.gpu = True
            Pipeline([InferDefaultSchedulesAndStorages(), promote_gpu_scalars]).apply_pass(sdfg, {})
            move_arrays = MoveArrayOutOfKernel()
            move_arrays.raise_on_unsupported = False
            move_arrays.apply_pass(sdfg, {})
            infer_types.infer_connector_types(sdfg)
            infer_types.set_default_schedule_and_storage_types(sdfg, None)
        # Pure readability rewrites over an already-valid SDFG; validate once afterwards.
        # Scalar fission (``PrivatizeScalars``) is deliberately NOT run here. It is an optimization pass and belongs in
        # the caller's pipeline, before WCR memlets exist: by codegen time an accumulator chain has been rewritten to
        # WCR, and fissioning a read-modify-write into a fresh SSA version drops the running value.
        # ``const_init``: transients that only store literals become ``constexpr`` SDFG constants.
        if config.Config.get("compiler", "cpu", "codegen_params", "const_init") == "on":
            PromoteConstantTransients().apply_pass(sdfg, {})
        InlineTaskletConnectors().apply_pass(sdfg, {})
        # Any nested SDFG that survived inlining (e.g. a library expansion) must not share a data name
        # with a differently-strided parent array, else its ``<name>_idx`` helper redefines the parent's.
        CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
        if validate:
            sdfg.validate()

        # The readable ``<arr>_idx`` / ``<arr>_size`` helpers are ``constexpr``; when they are reached
        # from device code, nvcc needs ``--expt-relaxed-constexpr`` to evaluate a host/device constexpr
        # in a constant context. Ensure it is on the CUDA flags (idempotent) so a GPU build under the
        # readable generator compiles without a manual config edit.
        cuda_args = config.Config.get("compiler", "cuda", "args")
        if "--expt-relaxed-constexpr" not in cuda_args:
            config.Config.set("compiler", "cuda", "args", value=(cuda_args + " --expt-relaxed-constexpr").strip())

    # Right before codegen, not in simplify: until here SymPy's power laws can still fold ``Pow``
    # (``R**i * R**(K-i-1) -> R**(K-1)``), which the opaque ``ipow`` would freeze.
    RelaxIntegerPowers().apply_pass(sdfg, {})

    frame = framecode.DaCeCodeGenerator(sdfg)

    # Test for undefined symbols in SDFG arguments
    if "?" in frame.arglist.keys():
        raise exc.CodegenError(
            f"SDFG '{sdfg.name}' has undefined symbols in its arguments. "
            "Please ensure all symbols are defined before generating code."
        )

    # Instantiate CPU first (as it is used by the other code generators)
    # TODO: Refactor the parts used by other code generators out of CPU
    from dace.codegen.targets import cpu

    default_target = cpu.CPUCodeGen
    for k, v in TargetCodeGenerator.extensions().items():
        # If another target has already been registered as CPU, use it instead
        if v["name"] == "cpu":
            default_target = k
    # The experimental readable CPU generator is opt-in and selected explicitly
    # (it is not registered with the target registry), so it wins over any 'cpu'
    # extension picked above.
    if config.Config.get("compiler", "cpu", "implementation") == "experimental_readable":
        from dace.codegen.targets import experimental_cpu

        default_target = experimental_cpu.ExperimentalCPUCodeGen
    targets = {"cpu": default_target(frame, sdfg)}

    disabled_cuda_target = unselected_cuda_target()

    # Instantiate the rest of the targets
    targets.update(
        {
            v["name"]: k(frame, sdfg)
            for k, v in TargetCodeGenerator.extensions().items()
            if v["name"] not in (*targets, disabled_cuda_target)
        }
    )

    # Query all code generation targets and instrumentation providers in SDFG
    _get_codegen_targets(sdfg, frame)

    # Preprocess SDFG
    for target in frame.targets:
        target.preprocess(sdfg)

    # Instantiate instrumentation providers
    frame._dispatcher.instrumentation = {
        k: v() if v is not None else None for k, v in frame._dispatcher.instrumentation.items()
    }

    # NOTE: THE SDFG IS ASSUMED TO BE FROZEN (not change) FROM THIS POINT ONWARDS

    # Generate frame code (and the rest of the code)
    (global_code, frame_code, used_targets, used_environments) = frame.generate_code(sdfg, None)
    target_objects = [
        CodeObject(
            sdfg.name,
            global_code + frame_code,
            "cpp",
            cpu.CPUCodeGen,
            "Frame",
            environments=used_environments,
            sdfg=sdfg,
        )
    ]

    # Create code objects for each target
    for tgt in used_targets:
        target_objects.extend(tgt.get_generated_codeobjects())

    # Ensure that no new targets were dynamically added
    assert frame._dispatcher.used_targets == (frame.targets - {frame})

    # add a header file for calling the SDFG
    dummy = CodeObject(
        sdfg.name,
        generate_headers(sdfg, frame),
        "h",
        cpu.CPUCodeGen,
        "CallHeader",
        target_type="../../include",
        linkable=False,
    )
    target_objects.append(dummy)

    for env in dace.library.get_environments_and_dependencies(used_environments):
        if hasattr(env, "codeobjects"):
            target_objects.extend(env.codeobjects)

    # add a dummy main function to show how to call the SDFG
    dummy = CodeObject(
        sdfg.name + "_main",
        generate_dummy(sdfg, frame),
        "cpp",
        cpu.CPUCodeGen,
        "SampleMain",
        target_type="../../sample",
        linkable=False,
    )
    target_objects.append(dummy)

    return target_objects


##################################################################
