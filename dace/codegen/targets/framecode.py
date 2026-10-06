# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
import itertools
import pathlib
import re
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import dace
from dace import config, data, dtypes
from dace.cli import progress
from dace.codegen import control_flow as cflow
from dace.codegen import dispatcher as disp
from dace.codegen import exceptions as cgx
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.common import codeblock_to_cpp, sym2cpp
from dace.codegen.target import TargetCodeGenerator
from dace.frontend.python.astutils import rname
from dace.sdfg.type_inference import infer_expr_type
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import scope as sdscope
from dace.sdfg import utils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, CodeGeneratorFunctionRegion, ContinueBlock,
                             ControlFlowBlock, ControlFlowRegion, LoopRegion, ReturnBlock, SymbolResolver)
from dace.transformation.passes.analysis import StateReachability, loop_analysis


def _reaches_state_struct(sdfg: SDFG, states: List[SDFGState]) -> bool:
    """
    Whether code in the given states may access data through the state struct other than by name: nested SDFGs and
    library nodes (their functions receive the state struct), and opaque code (see ``utils.calls_opaque_code``).
    """
    if any(isinstance(node, (nodes.NestedSDFG, nodes.LibraryNode)) for state in states for node in state.nodes()):
        return True
    return utils.calls_opaque_code(sdfg, states)


def _assigned_literal(edge) -> Optional[Union[bool, int, float]]:
    """
    The constant that the edge writes if it comes from a tasklet without inputs whose code assigns a numeric or
    boolean literal (possibly cast, e.g., ``float(2.0)``, or negated) to the edge's connector; None otherwise.
    """
    tasklet = edge.src
    if (not isinstance(tasklet, nodes.Tasklet) or tasklet.in_connectors or len(tasklet.out_connectors) != 1
            or edge.data.wcr is not None or tasklet.code.language != dtypes.Language.Python
            or len(tasklet.code.code) != 1):
        return None
    stmt = tasklet.code.code[0]
    if (not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name)
            or stmt.targets[0].id != edge.src_conn):
        return None
    value = stmt.value
    # A cast to a type, e.g., ``float(2.0)`` or ``dace.float32(2.0)``
    if isinstance(value, ast.Call):
        if (len(value.args) != 1 or value.keywords
                or rname(value.func).split('.')[-1] not in set(dtypes.TYPECLASS_STRINGS) | {'float', 'int', 'bool'}):
            return None
        value = value.args[0]
    negate = isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
    if negate:
        value = value.operand
    if not isinstance(value, ast.Constant) or not isinstance(value.value, (bool, int, float)):
        return None
    if negate:
        return None if isinstance(value.value, bool) else -value.value
    return value.value


_LABEL = re.compile(r'__state_(?:exit_)?\d+(?:_\w+)?')
_MARKER = re.compile(r'////__DACE:[\d:]*')


def _function_key(code: str, name: str) -> str:
    """
    The code of a generated function up to the names that differ between equal regions: the function's name, the
    labels of its control flow (numbered by appearance) and the comments mapping code to SDFG elements.
    """
    labels: Dict[str, str] = {}
    code = _MARKER.sub('', code.replace(name, '@'))
    return _LABEL.sub(lambda m: labels.setdefault(m.group(0), f'@{len(labels)}'), code)


def _inside_loop_of(block: ControlFlowBlock, region: ControlFlowRegion) -> bool:
    """ Whether a loop inside ``region`` encloses ``block`` (i.e., a break or continue there stays in the region). """
    graph = block.parent_graph
    while graph is not region:
        if isinstance(graph, LoopRegion):
            return True
        graph = graph.parent_graph
    return False


def _get_or_eval_sdfg_first_arg(func, sdfg):
    if callable(func):
        return func(sdfg)
    return func


class DaCeCodeGenerator(object):
    """ DaCe code generator class that writes the generated code for SDFG
        state machines, and uses a dispatcher to generate code for
        individual states based on the target. """

    def __init__(self, sdfg: SDFG):
        self._dispatcher = disp.TargetDispatcher(self)
        self._dispatcher.register_state_dispatcher(self)
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self.statestruct: List[str] = []
        self.environments: List[Any] = []
        self.targets: Set[TargetCodeGenerator] = set()
        self.to_allocate: DefaultDict[Union[SDFG, SDFGState, nodes.EntryNode],
                                      List[Tuple[SDFG, Optional[SDFGState], Optional[nodes.AccessNode], bool, bool,
                                                 bool]]] = collections.defaultdict(list)
        self.where_allocated: Dict[Tuple[SDFG, str], SDFG] = {}
        self.fsyms: Dict[int, Set[str]] = {}
        self._symbols_and_constants: Dict[int, Set[str]] = {}
        # The symbols visible in each state, shared by all nodes of the state (filled during code generation)
        self._symbol_resolver = SymbolResolver()
        # Code of the functions placed in separate translation units, by unit name
        self.translation_units: Dict[str, List[str]] = {}
        # The separate translation unit whose code is being generated, or None for the frame code's unit
        self.current_translation_unit: Optional[str] = None
        # The stream that receives the global code of the states of each SDFG (e.g., nested SDFG functions), which is
        # the unit's while the function of a region placed in a separate unit is generated
        self._global_streams: Dict[SDFG, CodeIOStream] = {}
        # The types of the symbols (including inter-state symbols) of each SDFG, filled during code generation
        self._symbol_types: Dict[SDFG, Dict[str, dtypes.typeclass]] = {}
        self._symbol_uses_cache: Dict[SDFG, Dict[Any, Set[str]]] = {}
        self._state_local_cache: Dict[SDFG, Set[str]] = {}
        self._literal_cache: Dict[SDFG, Dict[str, str]] = {}
        # The functions of regions in separate translation units, by their code up to names (see ``_function_key``)
        self._region_functions: Dict[str, str] = {}
        self._toplevel_sdfg = sdfg
        self._struct_types: Dict[SDFG, Dict[str, dtypes.struct]] = {}
        fsyms = self.free_symbols(sdfg)
        self.arglist = sdfg.arglist(scalars_only=False, free_symbols=fsyms)

        # resolve all symbols and constants
        # first handle root
        sdfg.reset_cfg_list()
        self._symbols_and_constants[sdfg.cfg_id] = sdfg.free_symbols.union(sdfg.constants_prop.keys())
        # then recurse
        for nested, state in sdfg.all_nodes_recursive():
            if isinstance(nested, nodes.NestedSDFG):
                state: SDFGState

                nsdfg = nested.sdfg

                # found a new nested sdfg: resolve symbols and constants
                result = nsdfg.free_symbols.union(nsdfg.constants_prop.keys())

                parent_constants = self._symbols_and_constants[nsdfg.parent_sdfg.cfg_id]
                result |= parent_constants

                # check for constant inputs
                for edge in state.in_edges(nested):
                    if edge.data.data in parent_constants:
                        # this edge is constant => propagate to nested sdfg
                        result.add(edge.dst_conn)

                self._symbols_and_constants[nsdfg.cfg_id] = result

    # Cached fields
    def symbols_and_constants(self, sdfg: SDFG):
        return self._symbols_and_constants[sdfg.cfg_id]

    def free_symbols(self, obj: Any):
        k = id(obj)
        if k in self.fsyms:
            return self.fsyms[k]
        if hasattr(obj, 'used_symbols'):
            result = obj.used_symbols(all_symbols=False)
        else:
            result = obj.free_symbols
        self.fsyms[k] = result
        return result

    def symbols_defined_at(self, state: SDFGState, node: nodes.Node) -> Dict[str, dtypes.typeclass]:
        """
        Returns the symbols available to a node, as ``SDFGState.symbols_defined_at``. The part of the result that
        depends only on the state (the SDFG symbols and those defined along the control flow leading to the state)
        is computed once per state, since the SDFG does not change while its code is generated.

        :param state: The state containing the node.
        :param node: The node.
        :return: A dictionary mapping symbol names to their types, which the caller may modify.
        """
        return self._symbol_resolver.defined_at(state, node)

    def struct_types(self, sdfg: SDFG) -> Dict[str, dtypes.struct]:
        """
        Returns the struct types of the data containers of an SDFG by name, computed once per SDFG during code
        generation.

        :param sdfg: The SDFG.
        :return: A dictionary mapping struct type names to the struct types.
        """
        result = self._struct_types.get(sdfg)
        if result is None:
            from dace.codegen.targets.cpp import StructInitializer  # Avoid circular import
            result = self._struct_types[sdfg] = StructInitializer.struct_types(sdfg)
        return result

    ##################################################################
    # Target registry

    @property
    def dispatcher(self):
        return self._dispatcher

    ##################################################################
    # Code generation

    def preprocess(self, sdfg: SDFG) -> None:
        """
        Called before code generation. Used for making modifications on the SDFG prior to code generation.

        :note: Post-conditions assume that the SDFG will NOT be changed after this point.
        :param sdfg: The SDFG to modify in-place.
        """
        pass

    def generate_constants(self, sdfg: SDFG, callsite_stream: CodeIOStream):
        # Write constants
        for cstname, (csttype, cstval) in sdfg.constants_prop.items():
            if isinstance(csttype, data.Array):
                const_str = "constexpr " + csttype.dtype.ctype + " " + cstname + "[" + str(cstval.size) + "] = {"
                it = np.nditer(cstval, order='C')
                for i in range(cstval.size - 1):
                    const_str += str(it[0]) + ", "
                    it.iternext()
                const_str += str(it[0]) + "};\n"
                callsite_stream.write(const_str, sdfg)
            else:
                callsite_stream.write("constexpr %s %s = %s;\n" % (csttype.dtype.ctype, cstname, sym2cpp(cstval)), sdfg)

    def add_to_translation_unit(self, unit: str, code: str) -> None:
        """
        Adds code (e.g., the definition of a nested SDFG function) to a separate translation unit.

        :param unit: The name of the translation unit, which is created on first use.
        :param code: The code to append to the unit.
        """
        self.translation_units.setdefault(unit, []).append(code)

    def generate_translation_unit(self, sdfg: SDFG, unit: str) -> str:
        """
        Generates the code of a separate translation unit: the same preamble as the frame code (includes, custom
        types, constants and the state struct), followed by the code added to the unit.

        :param sdfg: The top-level SDFG.
        :param unit: The name of the translation unit.
        :return: The code of the translation unit.
        """
        stream = CodeIOStream()
        stream.write('/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n#include <dace/dace.h>\n', sdfg)
        self.generate_fileheader(sdfg, stream, 'frame')
        for code in self.translation_units[unit]:
            stream.write(code)
        return stream.getvalue()

    def generate_fileheader(self, sdfg: SDFG, global_stream: CodeIOStream, backend: str = 'frame'):
        """ Generate a header in every output file that includes custom types
            and constants.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param backend: Whose backend this header belongs to.
        """
        from dace.codegen.targets.cpp import mangle_dace_state_struct_name  # Avoid circular import
        # Hash file include
        if backend == 'frame':
            global_stream.write('#include "../../include/hash.h"\n', sdfg)

        #########################################################
        # Target-based includes
        for target in self._dispatcher.used_targets:
            headers = target.get_includes()
            if backend in headers:
                global_stream.write("\n".join("#include \"" + h + "\"" for h in headers[backend]), sdfg)

        # Environment-based includes
        for env in self.environments:
            if len(env.headers) > 0:
                if not isinstance(env.headers, dict):
                    headers = {'frame': env.headers}
                else:
                    headers = env.headers
                if backend in headers:
                    global_stream.write("\n".join("#include \"" + h + "\"" for h in headers[backend]), sdfg)

        #########################################################
        # Custom types
        datatypes = set()
        # Types of this SDFG
        for _, arrname, arr in sdfg.arrays_recursive():
            if arr is not None:
                datatypes.add(arr.dtype)

        emitted = set()

        def _emit_definitions(dtype: dtypes.typeclass, wrote_something: bool) -> bool:
            if isinstance(dtype, dtypes.pointer):
                wrote_something = _emit_definitions(dtype._typeclass, wrote_something)
            elif isinstance(dtype, dtypes.struct):
                for field in dtype.fields.values():
                    wrote_something = _emit_definitions(field, wrote_something)
                if not wrote_something:
                    global_stream.write("", sdfg)
                if dtype not in emitted:
                    global_stream.write(dtype.emit_definition(), sdfg)
                    wrote_something = True
                    emitted.add(dtype)
            return wrote_something

        # Emit unique definitions
        wrote_something = False
        for typ in datatypes:
            wrote_something = _emit_definitions(typ, wrote_something)
        if wrote_something:
            global_stream.write("", sdfg)

        #########################################################
        # Write constants
        self.generate_constants(sdfg, global_stream)

        #########################################################
        # Write state struct
        structstr = '\n'.join(self.statestruct)
        global_stream.write(f'''
struct {mangle_dace_state_struct_name(sdfg)} {{
    {structstr}
}};

''', sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[None]), sd)
            if backend in sd.global_code:
                global_stream.write(codeblock_to_cpp(sd.global_code[backend]), sd)

    def generate_header(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        # Write frame code - header
        global_stream.write('/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n' + '#include <dace/dace.h>\n', sdfg)

        # Write header required by environments
        for env in self.environments:
            self.statestruct.extend(env.state_fields)

        # Instrumentation preamble
        # NOTE: Some instrumentation providers (e.g. GPU_TX_MARKERS) never write to
        # __state->report, so skip the report machinery unless at least one active provider does.
        if any(i is not None and i.writes_to_report() for i in self._dispatcher.instrumentation.values()):
            self.statestruct.append('dace::perf::Report report;')
            # Reset report if written every invocation
            if config.Config.get_bool('instrumentation', 'report_each_invocation'):
                callsite_stream.write('__state->report.reset();', sdfg)

        self.generate_fileheader(sdfg, global_stream, 'frame')

    def generate_footer(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        from dace.codegen.targets.cpp import mangle_dace_state_struct_name  # Avoid circular import
        fname = sdfg.name
        params = sdfg.signature(arglist=self.arglist)
        paramnames = sdfg.signature(False, for_call=True, arglist=self.arglist)
        initparams = sdfg.init_signature(free_symbols=self.free_symbols(sdfg))
        initparamnames = sdfg.init_signature(for_call=True, free_symbols=self.free_symbols(sdfg))

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_end(sdfg, callsite_stream, global_stream)

        # Instrumentation saving
        if (config.Config.get_bool('instrumentation', 'report_each_invocation')
                and any(i is not None and i.writes_to_report() for i in self._dispatcher.instrumentation.values())):
            callsite_stream.write(
                '__state->report.save("%s", __HASH_%s);' % (pathlib.Path(sdfg.build_folder) / "perf", sdfg.name), sdfg)

        # Write closing brace of program
        callsite_stream.write('}', sdfg)

        # Write awkward footer to avoid 'extern "C"' issues
        params_comma = (', ' + params) if params else ''
        initparams_comma = (', ' + initparams) if initparams else ''
        paramnames_comma = (', ' + paramnames) if paramnames else ''
        initparamnames_comma = (', ' + initparamnames) if initparamnames else ''
        # Drain per invocation, not just per state: contamination can arrive between any two
        # calls. Declared rather than included, since it lives in the generated .cu.
        gpu_drain_decl = ''
        gpu_drain_call = ''
        # getattr: a user-registered code generator need not define target_name.
        if any(getattr(target, 'target_name', None) == 'cuda' for target in self._dispatcher.used_targets):
            gpu_drain_decl = (f'DACE_EXPORTED void '
                              f'__dace_gpu_drain_error({mangle_dace_state_struct_name(fname)} *__state);\n')
            gpu_drain_call = '    __dace_gpu_drain_error(__state);\n'

        callsite_stream.write(
            f'''
{gpu_drain_decl}DACE_EXPORTED void __program_{fname}({mangle_dace_state_struct_name(fname)} *__state{params_comma})
{{
{gpu_drain_call}    __program_{fname}_internal(__state{paramnames_comma});
}}''', sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    f'DACE_EXPORTED int __dace_init_{target.target_name}({mangle_dace_state_struct_name(sdfg)} *__state{initparams_comma});\n',
                    sdfg)
            if target.has_finalizer:
                callsite_stream.write(
                    f'DACE_EXPORTED int __dace_exit_{target.target_name}({mangle_dace_state_struct_name(sdfg)} *__state);\n',
                    sdfg)

        callsite_stream.write(
            f"""
DACE_EXPORTED {mangle_dace_state_struct_name(sdfg)} *__dace_init_{sdfg.name}({initparams})
{{""", sdfg)

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_init_begin(sdfg, callsite_stream, global_stream)

        callsite_stream.write(
            f"""
    int __result = 0;
    {mangle_dace_state_struct_name(sdfg)} *__state = new {mangle_dace_state_struct_name(sdfg)}();""", sdfg)

        for target in self._dispatcher.used_targets:
            if target.has_initializer:
                callsite_stream.write(
                    '__result |= __dace_init_%s(__state%s);' % (target.target_name, initparamnames_comma), sdfg)
        # A failed target initializer leaves its part of the state struct unset, and everything below
        # allocates against it -- persistent GPU arrays dereference __state->gpu_context, which
        # __dace_init_cuda never constructs when it bails out on a missing device. Leave here first.
        callsite_stream.write(f"""
    if (__result) {{
        delete __state;
        return nullptr;
    }}
""", sdfg)
        for env in self.environments:
            init_code = _get_or_eval_sdfg_first_arg(env.init_code, sdfg)
            if init_code:
                callsite_stream.write("{  // Environment: " + env.__name__, sdfg)
                callsite_stream.write(init_code)
                callsite_stream.write("}")

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.init_code:
                callsite_stream.write(codeblock_to_cpp(sd.init_code[None]), sd)
            if 'frame' in sd.init_code:
                callsite_stream.write(codeblock_to_cpp(sd.init_code['frame']), sd)

        callsite_stream.write(self._initcode.getvalue(), sdfg)

        callsite_stream.write(f"""
    if (__result) {{
        delete __state;
        return nullptr;
    }}
""", sdfg)
        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_init_end(sdfg, callsite_stream, global_stream)
        callsite_stream.write(
            f"""
    return __state;
}}

DACE_EXPORTED int __dace_exit_{sdfg.name}({mangle_dace_state_struct_name(sdfg)} *__state)
{{
""", sdfg)
        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_exit_begin(sdfg, callsite_stream, global_stream)
        callsite_stream.write(f"""
    int __err = 0;
""", sdfg)

        # Instrumentation saving
        if (not config.Config.get_bool('instrumentation', 'report_each_invocation')
                and any(i is not None and i.writes_to_report() for i in self._dispatcher.instrumentation.values())):
            callsite_stream.write(
                '__state->report.save("%s", __HASH_%s);' % (pathlib.Path(sdfg.build_folder) / "perf", sdfg.name), sdfg)

        callsite_stream.write(self._exitcode.getvalue(), sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.exit_code:
                callsite_stream.write(codeblock_to_cpp(sd.exit_code[None]), sd)
            if 'frame' in sd.exit_code:
                callsite_stream.write(codeblock_to_cpp(sd.exit_code['frame']), sd)

        for target in self._dispatcher.used_targets:
            if target.has_finalizer:
                callsite_stream.write(
                    f'''
    int __err_{target.target_name} = __dace_exit_{target.target_name}(__state);
    if (__err_{target.target_name}) {{
        __err = __err_{target.target_name};
    }}
''', sdfg)
        for env in reversed(self.environments):
            finalize_code = _get_or_eval_sdfg_first_arg(env.finalize_code, sdfg)
            if finalize_code:
                callsite_stream.write("{  // Environment: " + env.__name__, sdfg)
                callsite_stream.write(finalize_code)
                callsite_stream.write("}")

        callsite_stream.write('delete __state;\n', sdfg)
        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_exit_end(sdfg, callsite_stream, global_stream)
        callsite_stream.write('return __err;\n}\n', sdfg)

    def generate_external_memory_management(self, sdfg: SDFG, callsite_stream: CodeIOStream):
        """
        If external data descriptors are found in the SDFG (or any nested SDFGs),
        this function will generate exported functions to (1) get the required memory size
        per storage location (``__dace_get_external_memory_size_<STORAGE>``, where ``<STORAGE>``
        can be ``CPU_Heap`` or any other ``dtypes.StorageType``); and (2) set the externally-allocated
        pointer to the generated code's internal state (``__dace_set_external_memory_<STORAGE>``).
        """
        from dace.codegen.targets.cpp import mangle_dace_state_struct_name  # Avoid circular import

        # Collect external arrays
        ext_arrays: Dict[dtypes.StorageType, List[Tuple[SDFG, str, data.Data]]] = collections.defaultdict(list)
        for subsdfg, aname, arr in sdfg.arrays_recursive():
            if arr.lifetime == dtypes.AllocationLifetime.External:
                ext_arrays[arr.storage].append((subsdfg, aname, arr))

        # Only generate functions as necessary
        if not ext_arrays:
            return

        initparams = sdfg.init_signature(free_symbols=self.free_symbols(sdfg))
        initparams_comma = (', ' + initparams) if initparams else ''

        for storage, arrays in ext_arrays.items():
            size = 0
            for subsdfg, aname, arr in arrays:
                size += arr.total_size_in_bytes

            # Size query functions
            callsite_stream.write(
                f'''
DACE_EXPORTED size_t __dace_get_external_memory_size_{storage.name}({mangle_dace_state_struct_name(sdfg)} *__state{initparams_comma})
{{
    return {sym2cpp(size)};
}}
''', sdfg)

            # Pointer set functions
            callsite_stream.write(
                f'''
DACE_EXPORTED void __dace_set_external_memory_{storage.name}({mangle_dace_state_struct_name(sdfg)} *__state, char *ptr{initparams_comma})
{{''', sdfg)

            offset = 0
            for subsdfg, aname, arr in arrays:
                allocname = f'__state->__{subsdfg.cfg_id}_{aname}'
                callsite_stream.write(f'{allocname} = decltype({allocname})(ptr + {sym2cpp(offset)});', subsdfg)
                offset += arr.total_size_in_bytes

            # Footer
            callsite_stream.write('}', sdfg)

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       global_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = True):
        sid = state.block_id

        # Emit internal transient array allocation
        self.allocate_arrays_in_scope(sdfg, cfg, state, global_stream, callsite_stream)

        callsite_stream.write('\n')

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_state_begin(sdfg, cfg, state, callsite_stream, global_stream)

        #####################
        # Create dataflow graph for state's children.

        # DFG to code scheme: Only generate code for nodes whose all
        # dependencies have been executed (topological sort).
        # For different connected components, run them concurrently.

        components = dace.sdfg.concurrent_subgraphs(state)

        if len(components) <= 1:
            self._dispatcher.dispatch_subgraph(sdfg,
                                               cfg,
                                               state,
                                               sid,
                                               global_stream,
                                               callsite_stream,
                                               skip_entry_node=False)
        else:
            if sdfg.openmp_sections:
                callsite_stream.write("#pragma omp parallel sections\n{")
            for c in components:
                if sdfg.openmp_sections:
                    callsite_stream.write("#pragma omp section\n{")
                self._dispatcher.dispatch_subgraph(sdfg,
                                                   cfg,
                                                   c,
                                                   sid,
                                                   global_stream,
                                                   callsite_stream,
                                                   skip_entry_node=False)
                if sdfg.openmp_sections:
                    callsite_stream.write("} // End omp section")
            if sdfg.openmp_sections:
                callsite_stream.write("} // End omp sections")

        #####################
        # Write state footer

        if generate_state_footer:
            # Emit internal transient array deallocation
            self.deallocate_arrays_in_scope(sdfg, state.parent_graph, state, global_stream, callsite_stream)

            # Invoke all instrumentation providers
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_state_end(sdfg, cfg, state, callsite_stream, global_stream)

    def generate_states(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream) -> Set[SDFGState]:
        states_generated = set()

        opbar = progress.OptionalProgressBar(len(sdfg.states()), title=f'Generating code (SDFG {sdfg.cfg_id})')

        # Create closure + function for state dispatcher
        self._global_streams[sdfg] = global_stream

        def dispatch_state(state: SDFGState) -> str:
            stream = CodeIOStream()
            self._dispatcher.dispatch_state(state, self._global_streams[sdfg], stream)
            opbar.next()
            states_generated.add(state)  # For sanity check
            return stream.getvalue()

        callsite_stream.write(cflow.control_flow_region_to_code(sdfg, dispatch_state, self, sdfg.symbols), sdfg)

        opbar.done()

        return states_generated

    def _symbol_uses(self, sdfg: SDFG) -> Dict[Any, Set[str]]:
        """
        The symbols that each state, inter-state edge and control flow block (by its own expressions, e.g., a loop
        header) of an SDFG reads, computed once per SDFG. The variables of the loops enclosing them (and of a loop's own
        header) are left out: they are assigned before they are read there.
        """
        uses = self._symbol_uses_cache.get(sdfg)
        if uses is not None:
            return uses

        loop_vars: Dict[ControlFlowRegion, Set[str]] = {}

        def enclosing_loop_vars(graph: ControlFlowRegion) -> Set[str]:
            if graph is sdfg or graph is None:
                return set()
            if graph not in loop_vars:
                own = {graph.loop_variable} if isinstance(graph, LoopRegion) and graph.loop_variable else set()
                loop_vars[graph] = own | enclosing_loop_vars(graph.parent_graph)
            return loop_vars[graph]

        uses = {}
        for block in sdfg.all_control_flow_blocks():
            if isinstance(block, SDFGState):
                syms = block.used_symbols(all_symbols=True)
            else:
                syms = block.used_symbols(all_symbols=True, with_contents=False)
            discounted = enclosing_loop_vars(block.parent_graph)
            if isinstance(block, LoopRegion) and block.loop_variable:
                discounted = discounted | {block.loop_variable}
            uses[block] = set(syms) - discounted
        for cfg in sdfg.all_control_flow_regions():
            for edge in cfg.edges():
                uses[edge] = set(edge.data.free_symbols) - enclosing_loop_vars(cfg)
        self._symbol_uses_cache[sdfg] = uses
        return uses

    def _state_local_scalars(self, sdfg: SDFG) -> Set[str]:
        """
        The transient scalars of an SDFG whose value never crosses a state boundary: every state that reads one writes
        it first (no read without a preceding write in the state, no write-conflict resolution, which reads the old
        value), and no inter-state edge or control flow expression reads it. Computed once per SDFG.
        """
        cached = self._state_local_cache.get(sdfg)
        if cached is not None:
            return cached
        candidates = {
            name
            for name, desc in sdfg.arrays.items()
            if desc.transient and isinstance(desc, data.Scalar) and desc.lifetime not in
            (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External, dtypes.AllocationLifetime.Global)
        }
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data not in candidates:
                    continue
                if state.in_degree(node) == 0 or any(e.data.wcr is not None for e in state.in_edges(node)):
                    candidates.discard(node.data)
        for key, syms in self._symbol_uses(sdfg).items():
            if not isinstance(key, SDFGState):
                candidates -= syms
        self._state_local_cache[sdfg] = candidates
        return candidates

    def _literal_scalars(self, sdfg: SDFG) -> Dict[str, str]:
        """
        The transient scalars of an SDFG that are only ever assigned one literal value, mapped to that value as C++
        code. Every write must come from a tasklet without inputs whose code assigns a numeric or boolean constant
        (possibly cast, e.g., ``float(2.0)``, or negated). Computed once per SDFG.
        """
        cached = self._literal_cache.get(sdfg)
        if cached is not None:
            return cached
        values: Dict[str, Any] = {}
        excluded: Set[str] = set()
        for state in sdfg.states():
            for node in state.data_nodes():
                desc = sdfg.arrays.get(node.data)
                if (not isinstance(desc, data.Scalar) or not desc.transient
                        or desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External,
                                             dtypes.AllocationLifetime.Global)):
                    continue
                for edge in state.in_edges(node):
                    value = _assigned_literal(edge)
                    if value is None or values.setdefault(node.data, value) != value:
                        excluded.add(node.data)
        literals = {}
        for name, value in values.items():
            if name not in excluded:
                literals[name] = ('true' if value else 'false') if isinstance(value, bool) else repr(value)
        self._literal_cache[sdfg] = literals
        return literals

    def generate_function_region(self, region: CodeGeneratorFunctionRegion, dispatch_state: Callable[[SDFGState], str],
                                 symbols: Dict[str, dtypes.typeclass]) -> str:
        """
        Generates the function that a ``CodeGeneratorFunctionRegion`` stands for, and returns the code that calls it.

        The arguments are the state struct, the data the region accesses but does not allocate (pointers are
        ``__restrict__`` unless the data may alias or is a view or reference; scalars are passed by reference if the
        region writes them), and the symbols it uses: by value if it only reads them, by reference if it assigns a
        symbol that is used elsewhere, and as a local variable of the function otherwise. Data the region allocates is
        allocated inside the function.

        :param region: The region.
        :param dispatch_state: The callback that generates the code of a state.
        :param symbols: The symbols defined at the region, with their types.
        :return: The code that calls the function.
        """
        from dace.codegen.targets import cpp  # Avoid import loop

        sdfg = region.sdfg
        placement = region.function_placement
        fname = region.function_name or re.sub(r'\W', '_', f'{region.label}_{region.cfg_id}')
        inlining = region.inlining
        if (placement == dtypes.FunctionPlacement.SeparateUnit
                and inlining in (dtypes.FunctionInlining.Inline, dtypes.FunctionInlining.ForceInline)):
            raise cgx.CodegenError(f'Function region "{region.label}" in a separate translation unit cannot be '
                                   f'inlined into its caller ({inlining.name})')

        inner_blocks = set(region.all_control_flow_blocks())
        inner_states = [b for b in inner_blocks if isinstance(b, SDFGState)]
        inner_edges = set(region.all_interstate_edges())
        for block in inner_blocks:
            if isinstance(block, ReturnBlock) or (isinstance(block, (BreakBlock, ContinueBlock))
                                                  and not _inside_loop_of(block, region)):
                raise cgx.CodegenError(f'Control flow leaves function region "{region.label}" through "{block.label}"')

        ######################################
        # Function body
        outer_unit = self.current_translation_unit
        outer_global_stream = self._global_streams[sdfg]
        unit_global_stream = None
        if placement == dtypes.FunctionPlacement.SeparateUnit:
            self.current_translation_unit = region.translation_unit or fname
            unit_global_stream = CodeIOStream()
            self._global_streams[sdfg] = unit_global_stream

        self._dispatcher.defined_vars.enter_scope(region)
        body = (cflow.allocation_on_entry(region, self) +
                cflow.control_flow_region_to_code(region, dispatch_state, self, symbols) +
                cflow.deallocation_on_exit(region, self))
        self._dispatcher.defined_vars.exit_scope(region)

        self._global_streams[sdfg] = outer_global_stream
        unit = self.current_translation_unit
        self.current_translation_unit = outer_unit

        ######################################
        # Data arguments
        arrays = set(sdfg.arrays.keys())
        accessed: Set[str] = set()
        written: Set[str] = set()
        for state in inner_states:
            for node in state.data_nodes():
                name = node.data.split('.')[0]
                accessed.add(name)
                if state.in_degree(node) > 0:
                    written.add(name)
        for edge in inner_edges:
            accessed |= edge.data.free_symbols & arrays
        for block in itertools.chain([region], inner_blocks):
            accessed |= block.used_symbols(all_symbols=True, with_contents=False) & arrays

        # Data declared in the region is local to the function, data only allocated there is declared outside
        declared_inside: Set[str] = set()
        allocated_inside: Set[str] = set()
        inner_scopes = inner_blocks | {region}
        state_set = set(inner_states)
        for scope, entries in self.to_allocate.items():
            for tsdfg, state, node, declare, allocate, _ in entries:
                if tsdfg is not sdfg:
                    continue
                if scope in inner_scopes or (isinstance(scope, nodes.EntryNode) and state in state_set):
                    if declare:
                        declared_inside.add(node.data)
                    elif allocate:
                        allocated_inside.add(node.data)

        params: List[str] = []
        args: List[str] = []
        local_declarations: List[str] = []
        epilogue: List[str] = []

        def copy_in_out(ctype: str, name: str):
            # A value the region writes and others may read: the function works on a local copy, which the compiler
            # can keep in a register (it could not prove that nothing else accesses a reference)
            params.append(f'{ctype} &__ref_{name}')
            local_declarations.append(f'{ctype} {name} = __ref_{name};\n')
            epilogue.append(f'__ref_{name} = {name};\n')

        state_local = self._state_local_scalars(sdfg)
        literals = self._literal_scalars(sdfg)
        # Persistent data is passed as arguments (which compilers can treat as unaliased, unlike state struct members)
        # unless code in the region may reach it through the state struct by other means
        pass_persistent = region.persistent_arguments and not _reaches_state_struct(sdfg, inner_states)
        persistent_names: Dict[str, str] = {}
        for name in sorted(accessed - declared_inside):
            if name in sdfg.constants_prop:
                continue
            desc = sdfg.arrays[name]
            ptrname = cpp.ptr(name, desc, sdfg, self)
            defined_type, ctype = self._dispatcher.defined_vars.get(ptrname)
            param = ptrname
            if ptrname.startswith('__state->'):
                if not pass_persistent:
                    continue
                param = ptrname[len('__state->'):]
                persistent_names[ptrname] = param
            elif name in state_local:
                # Every state that reads it writes it first, so no value flows into or out of the region
                local_declarations.append(f'{ctype} {ptrname};\n')
                continue
            elif name in literals and name not in written:
                # Only ever assigned one literal: the compiler can fold it, which an argument would prevent across
                # translation units
                local_declarations.append(f'const {ctype} {ptrname} = {literals[name]};\n')
                continue
            if defined_type == disp.DefinedType.Pointer and name not in allocated_inside:
                restrict = (region.restrict_arguments and ctype.rstrip().endswith('*') and not desc.may_alias
                            and not isinstance(desc, (data.View, data.Reference)))
                params.append(f'{ctype} {"__restrict__ " if restrict else ""}{param}')
            elif defined_type == disp.DefinedType.Scalar and name not in written:
                params.append(f'{ctype} {param}')
            elif defined_type == disp.DefinedType.Scalar:
                copy_in_out(ctype, param)
            else:
                params.append(f'{ctype} &{param}')
            args.append(ptrname)
        if persistent_names:
            # The body refers to persistent data through the state struct: refer to the arguments instead
            pattern = re.compile(r'__state->(' + '|'.join(re.escape(p) for p in persistent_names.values()) + r')\b')
            body = pattern.sub(r'\1', body)

        ######################################
        # Symbol arguments
        uses = self._symbol_uses(sdfg)
        used_outside: Set[str] = set()
        for key, syms in uses.items():
            if key not in inner_blocks and key not in inner_edges and key is not region:
                used_outside |= syms
        for desc in sdfg.arrays.values():
            used_outside |= {str(s) for s in desc.free_symbols}

        assigned_inside: Set[str] = set()
        for edge in inner_edges:
            assigned_inside |= set(edge.data.assignments.keys())
        for block in inner_blocks:
            if isinstance(block, LoopRegion) and block.loop_variable:
                assigned_inside.add(block.loop_variable)

        used_inside = set(region.used_symbols(all_symbols=True))
        for name in accessed:
            used_inside |= {str(s) for s in sdfg.arrays[name].free_symbols}
        symbol_types = self._symbol_types[sdfg]
        for sym in sorted((used_inside | assigned_inside) - arrays - set(sdfg.constants_prop.keys())):
            if sym not in symbol_types:
                continue
            ctype = symbol_types[sym].ctype
            if sym not in assigned_inside:
                params.append(symbol_types[sym].as_arg(sym))  # e.g., a function pointer for a callback
                args.append(sym)
            elif sym in used_outside or sym in used_inside:
                # Assigned inside but read before it (``used_inside`` holds the free symbols only) or elsewhere
                copy_in_out(ctype, sym)
                args.append(sym)
            else:
                local_declarations.append(f'{ctype} {sym};\n')

        ######################################
        # Function definition and call
        state_struct = f'{cpp.mangle_dace_state_struct_name(self._toplevel_sdfg)} *__state'
        signature = f'void {fname}({", ".join([state_struct] + params)})'
        definition = signature + ' {\n' + ''.join(local_declarations) + body + ''.join(epilogue) + '}\n'
        specifiers = {
            dtypes.FunctionInlining.Default: '',
            dtypes.FunctionInlining.Inline: 'inline ',
            dtypes.FunctionInlining.NoInline: 'DACE_NOINLINE ',
            dtypes.FunctionInlining.ForceInline: 'DACE_FORCEINLINE ',
        }[inlining]
        if region.attributes:
            specifiers += region.attributes + ' '
        if placement == dtypes.FunctionPlacement.SeparateUnit:
            unit_code = unit_global_stream.getvalue() + 'DACE_HIDDEN ' + specifiers + definition
            # Equal regions (e.g., repeated steps of an algorithm) share one function, unless named explicitly
            duplicate = False
            if not region.function_name:
                first = self._region_functions.setdefault(_function_key(unit_code, fname), fname)
                duplicate = first != fname
                fname = first
                signature = f'void {fname}({", ".join([state_struct] + params)})'
            outer_global_stream.write(f'DACE_HIDDEN {specifiers}{signature};\n', sdfg)
            if not duplicate:
                self.add_to_translation_unit(unit, unit_code)
        else:
            outer_global_stream.write('static ' + specifiers + definition, sdfg)

        return f'{fname}({", ".join(["__state"] + args)});\n'

    def _get_schedule(self, scope: Union[nodes.EntryNode, SDFGState, SDFG]) -> dtypes.ScheduleType:
        TOP_SCHEDULE = dtypes.ScheduleType.Sequential
        if scope is None:
            return TOP_SCHEDULE
        elif isinstance(scope, nodes.EntryNode):
            return scope.schedule
        elif isinstance(scope, (SDFGState, SDFG)):
            sdfg: SDFG = (scope if isinstance(scope, SDFG) else scope.parent)
            if sdfg.parent_nsdfg_node is None:
                return TOP_SCHEDULE

            # Go one SDFG up
            pstate = sdfg.parent
            pscope = pstate.entry_node(sdfg.parent_nsdfg_node)
            if pscope is not None:
                return self._get_schedule(pscope)
            return self._get_schedule(pstate)
        else:
            raise TypeError

    def _can_allocate(self, sdfg: SDFG, state: SDFGState, desc: data.Data, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG]) -> bool:
        # Views allocate no memory: they are bound at their access node, whose subset may use scope parameters
        if isinstance(desc, data.View):
            return True

        schedule = self._get_schedule(scope)
        # if not dtypes.can_allocate(desc.storage, schedule):
        #     return False
        if dtypes.can_allocate(desc.storage, schedule):
            return True

        # Check for device-level memory recursively
        node = scope if isinstance(scope, nodes.EntryNode) else None
        cstate = scope if isinstance(scope, SDFGState) else state
        csdfg = scope if isinstance(scope, SDFG) else sdfg

        if desc.storage in dtypes.GPU_STORAGES:
            return sdscope.is_devicelevel_gpu(csdfg, cstate, node)

        return False

    def determine_allocation_lifetime(self, top_sdfg: SDFG):
        """
        Determines where (at which scope/state/SDFG) each data descriptor will be allocated/deallocated.

        :param top_sdfg: The top-level SDFG to determine for.
        """
        # Gather shared transients, free symbols, and first/last appearance
        shared_transients = {}
        fsyms = {}
        reachability = StateReachability().apply_pass(top_sdfg, {})
        access_instances: Dict[int, Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            shared_transients[sdfg.cfg_id] = sdfg.shared_transients(check_toplevel=False, include_nested_data=True)
            fsyms[sdfg.cfg_id] = self.symbols_and_constants(sdfg)

            #############################################
            # Look for all states in which a scope-allocated array is used in
            instances: Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]] = collections.defaultdict(list)
            array_names = sdfg.arrays.keys(
            )  #set(k for k, v in sdfg.arrays.items() if v.lifetime == dtypes.AllocationLifetime.Scope)
            # Iterate topologically to get state-order
            for state in cfg_analysis.blockorder_topological_sort(sdfg, ignore_nonstate_blocks=True):
                for node in state.data_nodes():
                    if node.data not in array_names:
                        continue
                    instances[node.data].append((state, node))

                # Look in the surrounding edges for usage
                edge_fsyms: Set[str] = set()
                for e in state.parent_graph.all_edges(state):
                    edge_fsyms |= e.data.free_symbols
                for edge_array in edge_fsyms & array_names:
                    instances[edge_array].append((state, nodes.AccessNode(edge_array)))
            #############################################

            access_instances[sdfg.cfg_id] = instances

        # Per-SDFG information for scope-lifetime arrays, computed on first use
        control_flow_symbols: Dict[int, Set[str]] = {}
        root_data_accesses: Dict[int, Dict[str, Dict[SDFGState, List[nodes.AccessNode]]]] = {}

        for sdfg, name, desc in top_sdfg.arrays_recursive(include_nested_data=True):
            if isinstance(desc, data.DistributedDescriptor):
                self._dispatcher.defined_vars.add_global(f'__state->{name}', disp.DefinedType.Scalar,
                                                         desc.state_field_dtype.ctype)
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue
            # NOTE: Assuming here that all Structure members share transient/storage/lifetime properties.
            # TODO: Study what is needed in the DaCe stack to ensure this assumption is correct.
            top_desc = sdfg.arrays[name.split('.')[0]]
            top_transient = top_desc.transient
            top_storage = top_desc.storage
            top_lifetime = top_desc.lifetime
            if not top_transient:
                continue
            if name in sdfg.constants_prop:
                # Constants do not need to be allocated
                continue

            # NOTE: In the code below we infer where a transient should be
            # declared, allocated, and deallocated. The information is stored
            # in the `to_allocate` dictionary. The key of each entry is the
            # scope where one of the above actions must occur, while the value
            # is a tuple containing the following information:
            # 1. The SDFG object that containts the transient.
            # 2. The State id where the action should (approx.) take place.
            # 3. The Access Node id of the transient in the above State.
            # 4. True if declaration should take place, otherwise False.
            # 5. True if allocation should take place, otherwise False.
            # 6. True if deallocation should take place, otherwise False.

            first_state_instance, first_node_instance = access_instances[sdfg.cfg_id].get(name, [(None, None)])[0]
            last_state_instance, last_node_instance = access_instances[sdfg.cfg_id].get(name, [(None, None)])[-1]

            # Cases
            if top_lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
                # Persistent memory is allocated in initialization code and
                # exists in the library state structure

                # If unused, skip
                if first_node_instance is None:
                    continue

                definition = desc.as_arg(name=f'__{sdfg.cfg_id}_{name}') + ';'

                if top_storage != dtypes.StorageType.CPU_ThreadLocal:  # If thread-local, skip struct entry
                    self.statestruct.append(definition)

                self.to_allocate[top_sdfg].append((sdfg, first_state_instance, first_node_instance, True, True, True))
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue
            elif top_lifetime is dtypes.AllocationLifetime.Global:
                # Global memory is allocated in the beginning of the program
                # exists in the library state structure (to be passed along
                # to the right SDFG)

                # If unused, skip
                if first_node_instance is None:
                    continue

                definition = desc.as_arg(name=f'__{sdfg.cfg_id}_{name}') + ';'
                self.statestruct.append(definition)

                self.to_allocate[top_sdfg].append((sdfg, first_state_instance, first_node_instance, True, True, True))
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue

            # The rest of the cases change the starting scope we attempt to
            # allocate from, since the descriptors may only be allocated higher
            # in the hierarchy (e.g., in the case of GPU global memory inside
            # a kernel).
            alloc_scope: Union[nodes.EntryNode, SDFGState, SDFG] = None
            alloc_state: SDFGState = None
            if (name in shared_transients[sdfg.cfg_id] or top_lifetime is dtypes.AllocationLifetime.SDFG):
                # SDFG descriptors are allocated in the beginning of their SDFG
                alloc_scope = sdfg
                if first_state_instance is not None:
                    alloc_state = first_state_instance
                # If unused, skip
                if first_node_instance is None:
                    continue
            elif top_lifetime == dtypes.AllocationLifetime.State:
                # State memory is either allocated in the beginning of the
                # containing state or the SDFG (if used in more than one state)
                curstate: SDFGState = None
                multistate = False
                for state in sdfg.states():
                    if any(n.data == name for n in state.data_nodes()):
                        if curstate is not None:
                            multistate = True
                            break
                        curstate = state
                if multistate:
                    alloc_scope = sdfg
                else:
                    alloc_scope = curstate
                    alloc_state = curstate
            elif top_lifetime == dtypes.AllocationLifetime.Scope:
                # Scope memory (default) is either allocated in the innermost
                # scope (e.g., Map, Consume) it is used in (i.e., greatest
                # common denominator), or in the SDFG if used in multiple states
                curscope: Union[nodes.EntryNode, SDFGState] = None
                curstate: SDFGState = None

                if sdfg.cfg_id not in control_flow_symbols:
                    # Symbols used by inter-state edges and loop / conditional block conditions etc., and the access
                    # nodes of each data container (by state), are shared by all arrays of the SDFG.
                    cf_syms: Set[str] = set()
                    for isedge in sdfg.all_interstate_edges():
                        cf_syms |= self.free_symbols(isedge.data)
                    for cfg in sdfg.all_control_flow_regions():
                        cf_syms |= cfg.used_symbols(all_symbols=True, with_contents=False)
                    control_flow_symbols[sdfg.cfg_id] = cf_syms
                    accesses: Dict[str, Dict[SDFGState, List[nodes.AccessNode]]] = collections.defaultdict(dict)
                    for state in sdfg.states():
                        for node in state.nodes():
                            if isinstance(node, nodes.AccessNode):
                                accesses[node.root_data].setdefault(state, []).append(node)
                    root_data_accesses[sdfg.cfg_id] = accesses

                # Does the array appear in inter-state edges or loop / conditional block conditions etc.?
                multistate = name in control_flow_symbols[sdfg.cfg_id]

                for state, state_accesses in root_data_accesses[sdfg.cfg_id].get(name, {}).items():
                    if multistate:
                        break
                    sdict = state.scope_dict()
                    for node in state_accesses:
                        # If already found in another state, set scope to SDFG
                        if curstate is not None and curstate != state:
                            multistate = True
                            break
                        curstate = state

                        # Current scope (or state object if top-level)
                        scope = sdict[node] or state
                        if curscope is None:
                            curscope = scope
                            continue
                        # States always win
                        if isinstance(scope, SDFGState):
                            curscope = scope
                            continue
                        # Lower/Higher/Disjoint scopes: find common denominator
                        if isinstance(curscope, SDFGState):
                            if scope in curscope.nodes():
                                continue
                        # Scopes that share no scope meet at the top level of the state
                        curscope = sdscope.common_parent_scope(sdict, scope, curscope) or state

                    if multistate:
                        break

                if multistate:
                    alloc_scope = sdfg
                else:
                    alloc_scope = curscope
                    alloc_state = curstate
            else:
                raise TypeError('Unrecognized allocation lifetime "%s"' % desc.lifetime)

            if alloc_scope is None:  # No allocation necessary
                continue

            # If descriptor cannot be allocated in this scope, traverse up the
            # scope tree until it is possible
            cursdfg = sdfg
            curstate = alloc_state
            curscope = alloc_scope
            while not self._can_allocate(cursdfg, curstate, desc, curscope):
                if curscope is None:
                    break
                if isinstance(curscope, nodes.EntryNode):
                    # Go one scope up
                    curscope = curstate.entry_node(curscope)
                    if curscope is None:
                        curscope = curstate
                elif isinstance(curscope, (SDFGState, SDFG)):
                    cursdfg: SDFG = (curscope if isinstance(curscope, SDFG) else curscope.parent)
                    # Go one SDFG up
                    if cursdfg.parent_nsdfg_node is None:
                        curscope = None
                        curstate = None
                        cursdfg = None
                    else:
                        curstate = cursdfg.parent
                        curscope = curstate.entry_node(cursdfg.parent_nsdfg_node)
                        cursdfg = cursdfg.parent_sdfg
                else:
                    raise TypeError

            if curscope is None:
                curscope = top_sdfg

            # Check if Array/View is dependent on non-free SDFG symbols
            # NOTE: Tuple is (SDFG, State, Node, declare, allocate, deallocate)
            fsymbols = fsyms[sdfg.cfg_id]
            if (not isinstance(curscope, nodes.EntryNode)
                    and utils.is_nonfree_sym_dependent(first_node_instance, desc, first_state_instance, fsymbols)):
                # Allocate in first State, deallocate in last State
                if first_state_instance != last_state_instance:
                    # If any state is not reachable from first state, find common denominators in the form of
                    # dominator and postdominator.
                    instances: List[Tuple[SDFGState, nodes.AccessNode]] = access_instances[sdfg.cfg_id][name]

                    # A view gets "allocated" everywhere it appears
                    if isinstance(desc, data.View):
                        for s, n in instances:
                            self.to_allocate[s].append((sdfg, s, n, False, True, False))
                            self.to_allocate[s].append((sdfg, s, n, False, False, True))
                        self.where_allocated[(sdfg, name)] = cursdfg
                        continue

                    if any(inst not in reachability[sdfg.cfg_id][first_state_instance] for inst in instances):
                        first_state_instance, last_state_instance = _get_dominator_and_postdominator(sdfg, instances)
                        # Declare in SDFG scope
                        # NOTE: Even if we declare the data at a common dominator, we keep the first and last node
                        # instances. This is especially needed for Views which require both the SDFGState and the
                        # AccessNode.
                        self.to_allocate[curscope].append((sdfg, None, nodes.AccessNode(name), True, False, False))
                    else:
                        self.to_allocate[curscope].append(
                            (sdfg, first_state_instance, first_node_instance, True, False, False))

                    curscope = allocation_block(first_state_instance, desc, {state for state, _ in instances})
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, False, True, False))
                    curscope = last_state_instance
                    # A control flow region has no state to dispatch the deallocation through
                    dealloc_state = curscope if isinstance(curscope, SDFGState) else instances[-1][0]
                    self.to_allocate[curscope].append((sdfg, dealloc_state, last_node_instance, False, False, True))
                else:
                    curscope = first_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, True, True, True))
            else:
                self.to_allocate[curscope].append((sdfg, first_state_instance, first_node_instance, True, True, True))
            if isinstance(curscope, SDFG):
                self.where_allocated[(sdfg, name)] = curscope
            else:
                self.where_allocated[(sdfg, name)] = cursdfg

        self._allocate_in_function_regions(top_sdfg, access_instances)

    def _allocate_in_function_regions(
            self, top_sdfg: SDFG, access_instances: Dict[int, Dict[str, List[Tuple[SDFGState,
                                                                                   nodes.AccessNode]]]]) -> None:
        """
        Moves the allocation of scalars and register arrays whose accesses all lie in one code generator function
        region into that region, so that they are local variables of its function instead of arguments passed by
        reference (which would keep the compiler from promoting them to registers). Heap data stays where it is: a
        pointer argument costs nothing, while allocating in the function may allocate more often.

        :param top_sdfg: The top-level SDFG.
        :param access_instances: The states (and access nodes) that access each data container, by SDFG.
        """
        for sdfg in top_sdfg.all_sdfgs_recursive():
            regions = [r for r in sdfg.all_control_flow_regions() if isinstance(r, CodeGeneratorFunctionRegion)]
            if not regions:
                continue
            region_blocks = {r: set(r.all_control_flow_blocks()) for r in regions}
            uses = self._symbol_uses(sdfg)

            # The current allocation entries of each container of this SDFG
            entries: Dict[str, List[Tuple[Any, Tuple]]] = collections.defaultdict(list)
            for scope, scope_entries in self.to_allocate.items():
                for entry in scope_entries:
                    if entry[0] is sdfg:
                        entries[entry[2].data].append((scope, entry))

            for name, desc in sdfg.arrays.items():
                if (not desc.transient or isinstance(desc, data.View)
                        or desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External,
                                             dtypes.AllocationLifetime.Global)
                        or not (isinstance(desc, data.Scalar) or desc.storage == dtypes.StorageType.Register)):
                    continue
                instances = access_instances[sdfg.cfg_id].get(name)
                if not instances or name not in entries:
                    continue
                states = {state for state, _ in instances}
                # The innermost region that contains every access
                containing = [r for r in regions if states <= region_blocks[r]]
                if not containing:
                    continue
                region = min(containing, key=lambda r: len(region_blocks[r]))
                inside = region_blocks[region]

                def in_region(scope) -> bool:
                    if scope is region or scope in inside:
                        return True
                    return isinstance(scope, nodes.EntryNode) and any(s in inside for s, _ in instances)

                if all(in_region(scope) for scope, _ in entries[name]):
                    continue
                # Expressions outside the region (e.g., a condition) must not read it
                if any(name in syms for key, syms in uses.items() if isinstance(key, ControlFlowBlock)
                       and not isinstance(key, SDFGState) and key not in inside and key is not region):
                    continue

                for scope, entry in entries[name]:
                    self.to_allocate[scope].remove(entry)
                first_state, first_node = instances[0]
                self.to_allocate[region].append((sdfg, first_state, first_node, True, True, True))

    def allocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode,
                                                                                        ControlFlowBlock, SDFG],
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        if len(self.to_allocate[scope]) == 0:
            return
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_allocation_begin(sdfg, scope, callsite_stream)
        """ Dispatches allocation of all arrays in the given scope. """
        for tsdfg, state, node, declare, allocate, _ in self.to_allocate[scope]:
            if state is not None:
                state_id = state.block_id
            else:
                state_id = -1

            desc = node.desc(tsdfg)

            self._dispatcher.dispatch_allocate(tsdfg, cfg if state is None else state.parent_graph, state, state_id,
                                               node, desc, function_stream, callsite_stream, declare, allocate)
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_allocation_end(sdfg, scope, callsite_stream)

    def deallocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode,
                                                                                          ControlFlowBlock, SDFG],
                                   function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        if len(self.to_allocate[scope]) == 0:
            return
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_deallocation_begin(sdfg, scope, callsite_stream)
        """ Dispatches deallocation of all arrays in the given scope. """
        for tsdfg, state, node, _, _, deallocate in self.to_allocate[scope]:
            if not deallocate:
                continue
            if state is not None:
                state_id = state.block_id
            else:
                state_id = -1

            desc = node.desc(tsdfg)

            self._dispatcher.dispatch_deallocate(tsdfg, state.parent_graph, state, state_id, node, desc,
                                                 function_stream, callsite_stream)
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_deallocation_end(sdfg, scope, callsite_stream)

    def generate_code(self,
                      sdfg: SDFG,
                      schedule: Optional[dtypes.ScheduleType],
                      cfg_id: str = "") -> Tuple[str, str, Set[TargetCodeGenerator], Set[str]]:
        """
        Generate frame code for a given SDFG, calling registered targets'
        code generation callbacks for them to generate their own code.

        :param sdfg: The SDFG to generate code for.
        :param schedule: The schedule the SDFG is currently located, or
                         None if the SDFG is top-level.
        :param cfg_id: An optional string id given to the SDFG label
        :return: A tuple of the generated global frame code, local frame
                 code, and a set of targets that have been used in the
                 generation of this SDFG.
        """
        if len(cfg_id) == 0 and sdfg.cfg_id != 0:
            cfg_id = '_%d' % sdfg.cfg_id

        global_stream = CodeIOStream()
        callsite_stream = CodeIOStream()

        is_top_level = sdfg.parent is None

        # Analyze allocation lifetime of SDFG and all nested SDFGs
        if is_top_level:
            self.determine_allocation_lifetime(sdfg)

        # Generate code
        ###########################

        # Keep track of allocated variables
        allocated = set()

        # Add symbol mappings to allocated variables
        if sdfg.parent_nsdfg_node is not None:
            allocated |= sdfg.parent_nsdfg_node.symbol_mapping.keys()

        # Invoke all instrumentation providers
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_sdfg_begin(sdfg, callsite_stream, global_stream, self)

        # Allocate outer-level transients
        self.allocate_arrays_in_scope(sdfg, sdfg, sdfg, global_stream, callsite_stream)

        # The arguments of the top-level SDFG were computed on construction and are those that the generated function
        # signature and the targets use, so they are reused instead of traversing the whole SDFG again
        outside_symbols = self.arglist if is_top_level else set()

        # Define constants as top-level-allocated
        for cname, (ctype, _) in sdfg.constants_prop.items():
            if isinstance(ctype, data.Array):
                self.dispatcher.defined_vars.add(cname, disp.DefinedType.Pointer, ctype.dtype.ctype)
            else:
                self.dispatcher.defined_vars.add(cname, disp.DefinedType.Scalar, ctype.dtype.ctype)

        # Allocate inter-state variables
        global_symbols = copy.deepcopy(sdfg.symbols)
        global_symbols.update({aname: arr.dtype for aname, arr in sdfg.arrays.items()})
        interstate_symbols = {}
        for cfr in sdfg.all_control_flow_regions():
            if isinstance(cfr, LoopRegion) and cfr.loop_variable is not None and cfr.init_statement is not None:
                if not cfr.loop_variable in interstate_symbols:
                    if cfr.loop_variable in global_symbols:
                        interstate_symbols[cfr.loop_variable] = global_symbols[cfr.loop_variable]
                    else:
                        l_end = loop_analysis.get_loop_end(cfr)
                        l_start = loop_analysis.get_init_assignment(cfr)
                        l_step = loop_analysis.get_loop_stride(cfr)
                        sym_type = dtypes.result_type_of(infer_expr_type(l_start, global_symbols),
                                                         infer_expr_type(l_step, global_symbols),
                                                         infer_expr_type(l_end, global_symbols))
                        interstate_symbols[cfr.loop_variable] = sym_type
                if not cfr.loop_variable in global_symbols:
                    global_symbols[cfr.loop_variable] = interstate_symbols[cfr.loop_variable]

            for e in cfr.dfs_edges(cfr.start_block):
                symbols = e.data.new_symbols(sdfg, global_symbols)
                # Inferred symbols only take precedence if global symbol not defined or None
                symbols = {
                    k: v if (k not in global_symbols or global_symbols[k] is None) else global_symbols[k]
                    for k, v in symbols.items()
                }
                interstate_symbols.update(symbols)
                global_symbols.update(symbols)

        self._symbol_types[sdfg] = global_symbols

        try:
            edge_codegen = self.dispatcher.get_scope_dispatcher(schedule)
        except KeyError:
            edge_codegen = self.dispatcher.get_generic_node_dispatcher()

        for isvarName, isvarType in interstate_symbols.items():
            if isvarType is None:
                raise TypeError(f'Type inference failed for symbol {isvarName}')

            # NOTE: NestedSDFGs frequently contain tautologies in their symbol mapping, e.g., `'i': i`. Do not
            # redefine the symbols in such cases.
            # Additionally, do not redefine a symbol with its type if it was already defined
            # as part of the function's arguments
            if not is_top_level and isvarName in sdfg.parent_nsdfg_node.symbol_mapping:
                continue
            if isvarName not in outside_symbols:
                edge_codegen.emit_interstate_variable_declaration(isvarName, isvarType, callsite_stream, sdfg)
            # If the variable is passed as an input argument to the SDFG, do not need to declare it

        callsite_stream.write('\n', sdfg)

        #######################################################################
        # Generate actual program body

        states_generated = self.generate_states(sdfg, global_stream, callsite_stream)

        #######################################################################

        # Sanity check
        if len(states_generated) != len(sdfg.states()):
            raise RuntimeError(
                "Not all states were generated in SDFG {}!"
                "\n  Generated: {}\n  Missing: {}".format(sdfg.label, [s.label for s in states_generated],
                                                          [s.label for s in (set(sdfg.states()) - states_generated)]))

        # Deallocate transients
        self.deallocate_arrays_in_scope(sdfg, sdfg, sdfg, global_stream, callsite_stream)

        # Now that we have all the information about dependencies, generate
        # header and footer
        if is_top_level:
            header_stream = CodeIOStream()
            header_global_stream = CodeIOStream()
            footer_stream = CodeIOStream()
            footer_global_stream = CodeIOStream()

            # Get all environments used in the generated code, including
            # dependent environments
            from dace.codegen.targets.cpp import mangle_dace_state_struct_name
            self.environments = dace.library.get_environments_and_dependencies(self._dispatcher.used_environments)

            self.generate_header(sdfg, header_global_stream, header_stream)

            # Open program function
            params = sdfg.signature(arglist=self.arglist)
            if params:
                params = ', ' + params
            function_signature = f'void __program_{sdfg.name}_internal({mangle_dace_state_struct_name(sdfg)}*__state{params})\n{{'

            self.generate_footer(sdfg, footer_global_stream, footer_stream)
            self.generate_external_memory_management(sdfg, footer_stream)

            header_global_stream.write(global_stream.getvalue())
            header_global_stream.write(footer_global_stream.getvalue())
            generated_header = header_global_stream.getvalue()

            all_code = CodeIOStream()
            all_code.write(function_signature)
            all_code.write(header_stream.getvalue())
            all_code.write(callsite_stream.getvalue())
            all_code.write(footer_stream.getvalue())
            generated_code = all_code.getvalue()
        else:
            generated_header = global_stream.getvalue()
            generated_code = callsite_stream.getvalue()

        # Clean up generated code
        # NOTE: The lines are collected in a list and joined once, since repeatedly appending to (and truncating) one
        # string copies the code generated so far for every line.
        goto_ctr = collections.Counter(re.findall(r'goto (.*?);', generated_code))
        empty_statement = re.compile(r'^\s*;\s*')
        label_line = re.compile(r'^\s*([a-zA-Z_][a-zA-Z_0-9]*):\s*[;]?\s*////.*$')
        clean_lines = []
        last_line = ''
        for line in generated_code.split('\n'):
            # Empty line
            if not line.strip():
                continue
            # Empty line with semicolon
            if empty_statement.match(line):
                continue
            # Label that might be unused
            label = label_line.findall(line)
            if len(label) > 0:
                if label[0] not in goto_ctr:
                    last_line = ''
                    continue
                if f'goto {label[0]};' in last_line and goto_ctr[label[0]] == 1:  # goto followed by label
                    # ``last_line`` is non-empty only if it is the last line kept
                    clean_lines.pop()
                    last_line = ''
                    continue
            clean_lines.append(line)
            last_line = line
        clean_code = ''.join(line + '\n' for line in clean_lines)

        # Return the generated global and local code strings
        return (generated_header, clean_code, self._dispatcher.used_targets, self._dispatcher.used_environments)


def allocation_block(state: SDFGState, desc: data.Data, access_states: Set[SDFGState]) -> ControlFlowBlock:
    """
    The block whose entry allocates ``desc``, given the state that dominates its accesses. Allocating there would read
    a symbol ``desc`` is sized by before it is assigned when the assignment comes later on the only path to the
    accesses: on the edge leaving a block, or on an edge inside a control flow region the path passes through. The
    allocation then moves past each such block, to the first block after the last assignment; a control flow region
    allocates on entry.

    :param state: The state that dominates the accesses of ``desc``.
    :param desc: The descriptor to allocate.
    :param access_states: The states that access ``desc`` or read it on an adjacent edge.
    """
    sizes = {str(sym) for sym in desc.free_symbols}

    def accesses(block: ControlFlowBlock) -> bool:
        if isinstance(block, SDFGState):
            return block in access_states
        return any(inner in access_states for inner in block.all_states())

    def assigns_inside(block: ControlFlowBlock) -> bool:
        return (isinstance(block, AbstractControlFlowRegion)
                and any(edge.data.assignments.keys() & sizes for edge in block.all_interstate_edges()))

    block = state
    while not accesses(block):
        out_edges = block.parent_graph.out_edges(block)
        if len(out_edges) != 1:
            break
        successor = out_edges[0].dst
        if not (out_edges[0].data.assignments.keys() & sizes or assigns_inside(block) or
                (assigns_inside(successor) and not accesses(successor))):
            break
        block = successor
    return block


def _get_dominator_and_postdominator(sdfg: SDFG, accesses: List[Tuple[SDFGState, nodes.AccessNode]]):
    """
    Gets the closest common dominator and post-dominator for a list of states.
    Used for determining allocation of data used in branched states.
    """
    alldoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = collections.defaultdict(lambda: set())
    allpostdoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = collections.defaultdict(lambda: set())
    idom: Dict[ControlFlowRegion, Dict[ControlFlowBlock, ControlFlowBlock]] = {}
    ipostdom: Dict[ControlFlowRegion, Dict[ControlFlowBlock, ControlFlowBlock]] = {}
    utils.get_control_flow_block_dominators(sdfg, idom, alldoms, ipostdom, allpostdoms)

    states = [a for a, _ in accesses]
    data_name = accesses[0][1].data

    # All dominators and postdominators include the states themselves
    for state in states:
        alldoms[state].add(state)
        allpostdoms[state].add(state)

    start_state = states[0]
    while any(start_state not in alldoms[n] for n in states):
        if idom[start_state] is start_state:
            raise NotImplementedError(f'Could not find an appropriate dominator for allocation of "{data_name}"')
        start_state = idom[start_state]

    end_state = states[-1]
    while any(end_state not in allpostdoms[n] for n in states):
        if ipostdom[end_state] is end_state:
            raise NotImplementedError(f'Could not find an appropriate post-dominator for deallocation of "{data_name}"')
        end_state = ipostdom[end_state]

    # TODO(later): If any of the symbols were not yet defined, or have changed afterwards, fail
    # raise NotImplementedError

    return start_state, end_state
