# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import functools
from typing import List

import dace
from dace import dtypes
from dace import data
from dace import config
from dace.sdfg import SDFG
from dace.codegen.targets import framecode
from dace.codegen.codeobject import CodeObject
from dace.codegen import exceptions as exc
from dace.config import Config
from dace.sdfg import infer_types

from dace.codegen.instrumentation import InstrumentationProvider
from dace.sdfg.state import SDFGState
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising


def generate_headers(sdfg: SDFG, frame: framecode.DaCeCodeGenerator) -> str:
    """ Generate a header file for the SDFG """
    proto = ""
    proto += "#include <dace/dace.h>\n"
    init_params = (sdfg.name, sdfg.name, sdfg.init_signature(free_symbols=frame.free_symbols(sdfg)))
    call_params = sdfg.signature(with_types=True, for_call=False, arglist=frame.arglist)
    if len(call_params) > 0:
        call_params = ', ' + call_params
    params = (sdfg.name, sdfg.name, call_params)
    exit_params = (sdfg.name, sdfg.name)
    proto += 'typedef void * %sHandle_t;\n' % sdfg.name
    proto += 'extern "C" %sHandle_t __dace_init_%s(%s);\n' % init_params
    proto += 'extern "C" int __dace_exit_%s(%sHandle_t handle);\n' % exit_params
    proto += 'extern "C" void __program_%s(%sHandle_t handle%s);\n' % params

    return proto


def generate_dummy(sdfg: SDFG, frame: framecode.DaCeCodeGenerator) -> str:
    """ Generates a C program that calls this SDFG, guessing scalar values and allocating array args. """
    al = frame.arglist
    init_params = sdfg.init_signature(for_call=True, free_symbols=frame.free_symbols(sdfg))
    params = sdfg.signature(with_types=False, for_call=True, arglist=frame.arglist)
    if len(params) > 0:
        params = ', ' + params

    allocations = ''
    deallocations = ''

    for argname, arg in al.items():
        if isinstance(arg, data.Scalar):
            allocations += ("    " + str(arg.as_arg(name=argname, with_types=True)) + " = 42;\n")

    for argname, arg in al.items():
        if isinstance(arg, data.Array):
            from dace.codegen.targets import cpp
            dims_mul = cpp.sym2cpp(functools.reduce(lambda a, b: a * b, arg.shape, 1))
            basetype = str(arg.dtype)
            allocations += ("    " + str(arg.as_arg(name=argname, with_types=True)) + " = (" + basetype + "*) calloc(" +
                            dims_mul + ", sizeof(" + basetype + ")" + ");\n")
            deallocations += "    free(" + argname + ");\n"

    return f'''#include <cstdlib>
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
'''


def _get_codegen_targets(sdfg: SDFG, frame: framecode.DaCeCodeGenerator):
    """Collects code generation targets and instrumentation providers for the SDFG into the frame code generator."""
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

        # Copies/memlets: only check outgoing edges of access nodes/tasklets, to avoid duplicate checks.
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
        if hasattr(node, 'symbol_instrument'):
            disp.instrumentation[node.symbol_instrument] = provider_mapping[node.symbol_instrument]
        if hasattr(node, 'instrument'):
            disp.instrumentation[node.instrument] = provider_mapping[node.instrument]
        elif hasattr(node, 'consume'):
            disp.instrumentation[node.consume.instrument] = provider_mapping[node.consume.instrument]
        elif hasattr(node, 'map'):
            disp.instrumentation[node.map.instrument] = provider_mapping[node.map.instrument]

    if sdfg.instrument != dtypes.InstrumentationType.No_Instrumentation:
        disp.instrumentation[sdfg.instrument] = provider_mapping[sdfg.instrument]


def generate_code(sdfg: SDFG, validate=True) -> List[CodeObject]:
    """
    Generates code as a list of code objects for a given SDFG.

    :param sdfg: The SDFG to use
    :param validate: If True, validates the SDFG before generating the code.
    :return: List of code objects that correspond to files to compile.
    """
    from dace.codegen.target import TargetCodeGenerator  # Avoid import loop

    if validate:
        sdfg.validate()

    if Config.get_bool('testing', 'serialization'):
        from dace.sdfg import SDFG
        import difflib
        import filecmp
        import shutil
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix="_.sdfg", delete=False) as tmp1, \
            tempfile.NamedTemporaryFile(suffix="_.sdfg", delete=False) as tmp2:
            tmp1_path = tmp1.name
            tmp2_path = tmp2.name

        try:
            sdfg.save(tmp1_path, hash=False)
            sdfg2 = SDFG.from_file(tmp1_path)
            sdfg2.save(tmp2_path, hash=False)

            print('Testing SDFG serialization...')
            if not filecmp.cmp(tmp1_path, tmp2_path):
                with open(tmp1_path, 'r') as f1, open(tmp2_path, 'r') as f2:
                    diff = difflib.unified_diff(f1.readlines(),
                                                f2.readlines(),
                                                fromfile='test.sdfg  (first save)',
                                                tofile='test2.sdfg (after roundtrip)')
                diff = ''.join(diff)

                shutil.copy(tmp1_path, 'test.sdfg')
                shutil.copy(tmp2_path, 'test2.sdfg')
                raise RuntimeError(f'SDFG serialization failed - files do not match:\n{diff}')

        finally:
            try:
                os.remove(tmp1_path)
                os.remove(tmp2_path)
            except OSError:
                pass

    if config.Config.get_bool('optimizer', 'detect_control_flow'):
        # TODO: move earlier / into modular codegen; kept here for now to preserve legacy-test semantics.
        FixedPointPipeline([ControlFlowRaising()]).apply_pass(sdfg, {})

    infer_types.infer_connector_types(sdfg)

    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    sdfg.expand_library_nodes()

    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # Wrap top-level map-nests/loops into no_inline nested SDFGs (own .cpp each, via the do_split path
    # in _generate_NestedSDFG; and, for GPU nests, own standalone SDFG + .cu via the do_external path).
    # Must run before the readable generator's InlineSDFG sweep below (which would otherwise inline them
    # straight back) and after expand_library_nodes.
    if (config.Config.get_bool('compiler', 'cpu', 'codegen_params', 'split_nsdfg_translation_units')
            or config.Config.get_bool('compiler', 'cpu', 'codegen_params', 'external_translation_units')):
        from dace.transformation.passes.outline_top_level_nests import outline_top_level_nests
        outline_top_level_nests(sdfg)
        infer_types.infer_connector_types(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # Experimental readable generator: flatten nested SDFGs, mark write-once data const/constexpr, and
    # inline tasklet connectors. Runs after library expansion so post-expansion tasklets are seen too.
    if config.Config.get('compiler', 'cpu', 'implementation') == 'experimental_readable':
        from dace.transformation.pass_pipeline import Pipeline
        from dace.transformation.passes.mark_const_init import MarkConstInit
        from dace.transformation.passes.inline_tasklet_connectors import InlineTaskletConnectors
        from dace.transformation.passes.canonicalize_nested_index_names import CanonicalizeNestedIndexNames
        from dace.transformation.interstate.sdfg_nesting import InlineSDFG
        from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
        sdfg.apply_transformations_repeated(InlineSDFG)
        sdfg.apply_transformations_repeated(InlineMultistateSDFG)
        infer_types.infer_connector_types(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        # Normalize single-value transients to Scalar/len1-array (transient_only keeps the signature);
        # must run before explicit_copy so copy lowering sees the final descriptor form.
        scalar_emission = config.Config.get('compiler', 'cpu', 'codegen_params', 'scalar_emission_type')
        if scalar_emission == 'scalar':
            from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
            from dace.transformation.passes.promote_gpu_scalars_to_arrays import (InferDefaultSchedulesAndStorages,
                                                                                  PromoteGPUScalarsToArrays)
            ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
            # Widen GPU-storage scalars back: a by-value Scalar cannot live in device memory.
            Pipeline([InferDefaultSchedulesAndStorages(), PromoteGPUScalarsToArrays()]).apply_pass(sdfg, {})
            infer_types.infer_connector_types(sdfg)
            infer_types.set_default_schedule_and_storage_types(sdfg, None)
        elif scalar_emission == 'len1_array':
            from dace.transformation.passes.length_one_array_scalar_conversion import ConvertScalarsToLengthOneArrays
            ConvertScalarsToLengthOneArrays(transient_only=True).apply_pass(sdfg, {})
            infer_types.infer_connector_types(sdfg)
            infer_types.set_default_schedule_and_storage_types(sdfg, None)
        # Lift implicit copies to CopyLibraryNodes so ExpandAuto picks memcpy over dace::CopyND; expand
        # only these nodes -- RewriteCopyForLayout needs the shared pass's other callers unexpanded.
        if config.Config.get('compiler', 'cpu', 'codegen_params', 'explicit_copy') == 'on':
            from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
            from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
            InsertExplicitCopies().apply_pass(sdfg, {})
            sdfg.expand_library_nodes(predicate=lambda n: isinstance(n, CopyLibraryNode))
            infer_types.infer_connector_types(sdfg)
            infer_types.set_default_schedule_and_storage_types(sdfg, None)
        # Pure readability rewrites over an already-valid SDFG; validate once afterwards.
        # ssa_loop_scalars: SSA-versions reassigned scalars so each write-once version can be marked
        # const. Must run before MarkConstInit. Default 'off' keeps output byte-identical.
        if config.Config.get('compiler', 'cpu', 'codegen_params', 'ssa_loop_scalars') == 'on':
            from dace.transformation.passes.scalar_fission import PrivatizeScalars
            PrivatizeScalars().apply_pass(sdfg, {})
            infer_types.infer_connector_types(sdfg)
            infer_types.set_default_schedule_and_storage_types(sdfg, None)
        # const_init: classify write-once transients as constexpr/const. Default 'on' keeps output
        # byte-identical (unlike ssa_loop_scalars above, this already runs today).
        if config.Config.get('compiler', 'cpu', 'codegen_params', 'const_init') == 'on':
            Pipeline([MarkConstInit()]).apply_pass(sdfg, {})
        InlineTaskletConnectors().apply_pass(sdfg, {})
        # A nested SDFG surviving inlining (e.g. a library expansion) must not share a data name with a
        # differently-strided parent array, else its ``<name>_idx`` helper redefines the parent's.
        CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
        sdfg.validate()

        # Device code reaching the constexpr _idx/_size helpers needs nvcc's --expt-relaxed-constexpr;
        # ensure it's set (idempotent) so GPU builds don't need a manual config edit.
        cuda_args = config.Config.get('compiler', 'cuda', 'args')
        if '--expt-relaxed-constexpr' not in cuda_args:
            config.Config.set('compiler', 'cuda', 'args', value=(cuda_args + ' --expt-relaxed-constexpr').strip())

    # Lower base**exp to ipow where the exponent is a provable non-negative integer. Runs here (not in
    # simplify) so SymPy's power laws can still fold Pow expressions beforehand.
    from dace.transformation.passes.relax_integer_powers import RelaxIntegerPowers
    RelaxIntegerPowers().apply_pass(sdfg, {})

    frame = framecode.DaCeCodeGenerator(sdfg)

    if "?" in frame.arglist.keys():
        raise exc.CodegenError("SDFG '%s' has undefined symbols in its arguments. "
                               "Please ensure all symbols are defined before generating code." % sdfg.name)

    # Instantiate CPU first (as it is used by the other code generators)
    # TODO: Refactor the parts used by other code generators out of CPU
    from dace.codegen.targets import cpu
    default_target = cpu.CPUCodeGen
    for k, v in TargetCodeGenerator.extensions().items():
        # If another target has already been registered as CPU, use it instead
        if v['name'] == 'cpu':
            default_target = k
    # Readable CPU generator is opt-in (not in the target registry), so it wins over any 'cpu'
    # extension picked above.
    if config.Config.get('compiler', 'cpu', 'implementation') == 'experimental_readable':
        from dace.codegen.targets import experimental_cpu
        default_target = experimental_cpu.ExperimentalCPUCodeGen
    targets = {'cpu': default_target(frame, sdfg)}

    # Only the CUDA generator selected via compiler.cuda.implementation may be instantiated -- both
    # share GPU schedule types, so instantiating both raises a duplicate-dispatcher error.
    cuda_impl = config.Config.get('compiler', 'cuda', 'implementation')
    if cuda_impl not in ('legacy', 'experimental'):
        raise ValueError(f"Invalid compiler.cuda.implementation: {cuda_impl!r}. "
                         "Please select one of 'legacy' or 'experimental'.")
    disabled_cuda_target = 'experimental_cuda' if cuda_impl == 'legacy' else 'cuda'

    targets.update({
        v['name']: k(frame, sdfg)
        for k, v in TargetCodeGenerator.extensions().items()
        if v['name'] not in targets and v['name'] != disabled_cuda_target
    })

    _get_codegen_targets(sdfg, frame)

    for target in frame.targets:
        target.preprocess(sdfg)

    # Give the allocator a state at each loop/conditional boundary, so a transient whose shape depends
    # on a symbol assigned inside the block allocates after that symbol is defined.
    framecode.pad_control_flow_region_boundaries(sdfg)

    frame._dispatcher.instrumentation = {
        k: v() if v is not None else None
        for k, v in frame._dispatcher.instrumentation.items()
    }

    # NOTE: THE SDFG IS ASSUMED TO BE FROZEN (not change) FROM THIS POINT ONWARDS

    (global_code, frame_code, used_targets, used_environments) = frame.generate_code(sdfg, None)
    target_objects = [
        CodeObject(sdfg.name,
                   global_code + frame_code,
                   'cpp',
                   cpu.CPUCodeGen,
                   'Frame',
                   environments=used_environments,
                   sdfg=sdfg)
    ]

    for tgt in used_targets:
        target_objects.extend(tgt.get_generated_codeobjects())

    # Ensure that no new targets were dynamically added
    assert frame._dispatcher.used_targets == (frame.targets - {frame})

    dummy = CodeObject(sdfg.name,
                       generate_headers(sdfg, frame),
                       'h',
                       cpu.CPUCodeGen,
                       'CallHeader',
                       target_type='../../include',
                       linkable=False)
    target_objects.append(dummy)

    for env in dace.library.get_environments_and_dependencies(used_environments):
        if hasattr(env, "codeobjects"):
            target_objects.extend(env.codeobjects)

    dummy = CodeObject(sdfg.name + "_main",
                       generate_dummy(sdfg, frame),
                       'cpp',
                       cpu.CPUCodeGen,
                       'SampleMain',
                       target_type='../../sample',
                       linkable=False)
    target_objects.append(dummy)

    return target_objects


##################################################################
