# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import copy
import pathlib
import re
from typing import Any, DefaultDict, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import dace
from dace import config, data, dtypes, mpr_lowering
from dace.cli import progress
from dace.codegen import control_flow as cflow
from dace.codegen import dispatcher as disp
from dace.codegen.prettycode import CodeIOStream
from dace.transformation.passes.analysis.scopes import (AccessInstances, AllocationScopes, CodegenAnalysisPipeline,
                                                        SymbolScopes)
from dace.codegen.common import codeblock_to_cpp, sym2cpp
from dace.codegen.target import TargetCodeGenerator
from dace.ordered import OrderedSet
from dace.sdfg.type_inference import infer_expr_type
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import scope as sdscope
from dace.sdfg import utils
from dace.sdfg.state import (ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, UnstructuredControlFlow)
from dace.transformation.passes.analysis import StateReachability, loop_analysis


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
        # OrderedSet for the same reason as used_targets: codegen.py iterates this to order
        # the preprocess() calls, and a TargetCodeGenerator hashes by id().
        self.targets: Set[TargetCodeGenerator] = OrderedSet()
        self.to_allocate: DefaultDict[Union[SDFG, SDFGState, nodes.EntryNode],
                                      List[Tuple[SDFG, Optional[SDFGState], Optional[nodes.AccessNode], bool, bool,
                                                 bool]]] = collections.defaultdict(list)
        self.where_allocated: Dict[Tuple[SDFG, str], SDFG] = {}
        # (cfg_id, symbol) -> ctype, for each loop counter whose hoisted declaration was SKIPPED because
        # ``codegen_params.decl_placement`` is ``late`` and the counter is loop-local. The loop emitter
        # declares it in the for-init clause instead. Keyed by cfg_id: two SDFGs may each own a counter
        # of the same name, and only one of them may qualify. Empty under the default ``eager``.
        self.loop_local_counters: Dict[Tuple[int, str], str] = {}
        # id(obj) -> (obj, result). The object itself is kept alive here so its address
        # cannot be recycled by a later, unrelated object for as long as the entry lives
        # (a WeakValueDictionary would not do this: it lets obj die and the id go stale).
        self.fsyms: Dict[int, Tuple[Any, FrozenSet[str]]] = {}
        # Filled by determine_allocation_lifetime; targets read it through symbol_scopes.defined_at,
        # which falls back to symbols_defined_at for anything built after the pass ran.
        self.symbol_scopes: Dict = {}
        # cfg_id -> whether that SDFG's control flow is fully structured (line-graph regions only).
        # Consulted by state_needs_brace to gate the experimental readable state-scope elision.
        self._structured_cfg: Dict[int, bool] = {}
        self._symbols_and_constants: Dict[int, Set[str]] = {}
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

    def free_symbols(self, obj: Any) -> FrozenSet[str]:
        k = id(obj)
        cached = self.fsyms.get(k)
        if cached is not None and cached[0] is obj:
            return cached[1]
        if hasattr(obj, 'used_symbols'):
            result = obj.used_symbols(all_symbols=False)
        else:
            result = obj.free_symbols
        # Frozen so a caller's `fsyms |= ...` rebinds its own local name instead of
        # mutating the shared cache entry (frozenset has no __ior__, so `|=` falls
        # back to `fsyms = fsyms | ...`, which creates a new object).
        result = frozenset(result)
        self.fsyms[k] = (obj, result)
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
                # sym2cpp, not str: a numpy bool prints as Python 'True', and complex as '(1+2j)'.
                # .flat keeps the numpy scalar, so sym2cpp still sees the element's own width.
                const_str += ", ".join(sym2cpp(cstval.flat[i]) for i in range(cstval.size))
                const_str += "};\n"
                callsite_stream.write(const_str, sdfg)
            else:
                callsite_stream.write("constexpr %s %s = %s;\n" % (csttype.dtype.ctype, cstname, sym2cpp(cstval)), sdfg)

    def generate_fileheader(self,
                            sdfg: SDFG,
                            global_stream: CodeIOStream,
                            backend: str = 'frame',
                            include_hash: bool = True):
        """ Generate a header in every output file that includes custom types
            and constants.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param backend: Whose backend this header belongs to.
            :param include_hash: Whether to include ``include/hash.h``. Only meaningful for the
                                 ``frame`` backend, and only the frame code actually uses the
                                 ``__HASH_<name>`` macro it defines (to name an instrumentation
                                 report). The include is written with a path relative to the frame's
                                 own directory (``src/<target>/``), so a file emitted one level
                                 deeper -- a split nested-SDFG translation unit under
                                 ``src/cpu/nsdfg/`` -- must pass False and would otherwise fail to
                                 resolve the include.

        Every TU that needs the state type re-emits an identical ``<sdfg>_state_t`` definition --
        the frame, each ``.cu``, each split nest. It is never forward-declared: Streams and
        persistent storage are state fields, so any TU may dereference ``__state``.
        """
        from dace.codegen.targets.cpp import mangle_dace_state_struct_name  # Avoid circular import
        standalone = mpr_lowering.standalone()
        # Hash file include
        if backend == 'frame' and include_hash and not standalone:
            global_stream.write('#include "../../include/hash.h"\n', sdfg)

        #########################################################
        # Target- and environment-based includes. Skipped entirely for MPR: its preamble carries the
        # system headers (:data:`~dace.mpr_lowering.BASE_HEADERS` plus whatever the emitted text
        # turns out to call), and the DaCe headers in these lists are the ones MPR exists to do
        # without. Dropping a header an expansion really needed does not hide the problem -- that
        # expansion also emits the ``dace::`` symbol the header declares, which
        # ``dace.codegen.mpr.verify`` reports against the SDFG construct that asked for it.
        if not standalone:
            for target in self._dispatcher.used_targets:
                headers = target.get_includes()
                if backend in headers:
                    global_stream.write("\n".join("#include \"" + h + "\"" for h in headers[backend]), sdfg)

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
        # OrderedSet, not set: typeclass.__hash__ folds in hash(self.type), which for a struct is
        # the default id()-based hash of ctypes.Structure -- that id() moves with ASLR run to run,
        # so a plain set() emitted these definitions in a different order every process even under
        # a fixed PYTHONHASHSEED. Insertion order here is first-occurrence order in the SDFG, which
        # is deterministic.
        datatypes = OrderedSet()
        # Types of this SDFG
        for _, arrname, arr in sdfg.arrays_recursive():
            if arr is not None:
                datatypes.add(arr.dtype)

        emitted = OrderedSet()

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
        # Write state struct. MPR has no state: its entry point takes the arguments directly, and
        # everything a state field would hold (persistent buffers, instrumentation reports,
        # environment handles) is either demoted to SDFG lifetime or refused outright by
        # :func:`~dace.codegen.mpr.render`. Emitting an empty struct nobody dereferences would
        # only invite one back.
        if not standalone:
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
        # Write frame code - header. The runtime header is the whole point of MPR's absence: the
        # standalone preamble (system headers plus the inline definitions the unit actually calls)
        # is prepended afterwards by :func:`~dace.codegen.mpr.render`, which is the only place that
        # can see the finished text and therefore which helpers it uses.
        if mpr_lowering.standalone():
            global_stream.write('/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n', sdfg)
        else:
            global_stream.write('/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n' + '#include <dace/dace.h>\n', sdfg)

        # Write header required by environments
        for env in self.environments:
            self.statestruct.extend(env.state_fields)

        # Instrumentation preamble
        if len(self._dispatcher.instrumentation) > 2:
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
        if mpr_lowering.standalone():
            self.generate_standalone_footer(sdfg, callsite_stream)
            return
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
                and len(self._dispatcher.instrumentation) > 2):
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
        # getattr: a user-registered code generator need not define target_name. Both GPU generators
        # define the drain in their .cu, so both get the per-call declaration.
        if any(
                getattr(target, 'target_name', None) in ('cuda', 'experimental_cuda')
                for target in self._dispatcher.used_targets):
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
                and len(self._dispatcher.instrumentation) > 2):
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

    def generate_standalone_footer(self, sdfg: SDFG, callsite_stream: CodeIOStream):
        """Close MPR's entry function, and nothing else.

        The ordinary footer emits four things MPR cannot have: the ``__program_<name>`` wrapper and
        the ``__dace_init_<name>`` / ``__dace_exit_<name>`` pair (all three take a state pointer),
        and the instrumentation report save (a ``dace::perf::Report`` field on that state). MPR's
        entry function IS the program, so only its closing brace is left.

        Init/exit code is REFUSED rather than dropped. Both come from SDFG properties a user set on
        purpose (``sdfg.init_code`` / ``exit_code``, or an environment's initializer); silently
        skipping them would render a program that quietly does less than the SDFG it came from,
        which is exactly the failure the numeric gate cannot catch when the skipped code only sets
        something up.
        """
        refused = []
        if self._initcode.getvalue().strip() or self._exitcode.getvalue().strip():
            refused.append('generated init/exit code')
        for sd in sdfg.all_sdfgs_recursive():
            if any(block.as_string.strip() for block in sd.init_code.values()):
                refused.append(f'{sd.name}.init_code')
            if any(block.as_string.strip() for block in sd.exit_code.values()):
                refused.append(f'{sd.name}.exit_code')
        # An environment is refused only when it has something to RUN or LINK. Most carry nothing
        # but a header list (the ``standard`` library's ``CPU`` is only ``<cstring>``, ``<numeric>``
        # and friends), and those cost the rendering nothing -- MPR supplies the system headers
        # itself. One with initialization code, a state field or a link dependency is a different
        # thing: it needs a handshake or a library that a single self-contained unit does not have.
        for env in self.environments:
            needs = [
                kind for kind, value in (('initialization code', getattr(env, 'init_code', '')),
                                         ('finalization code', getattr(env, 'finalize_code', '')),
                                         ('state fields', getattr(env, 'state_fields', ())),
                                         ('linked libraries', getattr(env, 'cmake_libraries', ())),
                                         ('CMake packages', getattr(env, 'cmake_packages', ()))) if value
            ]
            if needs:
                refused.append(f'the {env.__name__} environment ({", ".join(needs)})')
        if len(self._dispatcher.instrumentation) > 2:
            refused.append('instrumentation')
        if refused:
            raise NotImplementedError('MPR cannot render this SDFG standalone: it needs ' + '; '.join(refused) +
                                      ', which runs outside the entry function and has no place to run in a '
                                      'single self-contained translation unit')
        callsite_stream.write('}', sdfg)

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

        # External lifetime means the CALLER owns the buffer and hands it in through
        # ``__dace_set_external_memory_<storage>`` before the program runs. MPR has no such
        # handshake -- its entry function takes the arglist and nothing else -- so an SDFG that
        # needs one cannot be rendered, and saying so beats emitting a kernel that reads a pointer
        # that was never set.
        if mpr_lowering.standalone():
            if ext_arrays:
                names = sorted(name for arrays in ext_arrays.values() for _, name, _ in arrays)
                raise NotImplementedError('MPR cannot render this SDFG standalone: ' + ', '.join(names) +
                                          ' have External lifetime, which requires the caller to supply the '
                                          'buffer through an init handshake MPR does not emit')
            return

        # Only generate functions as necessary
        if not ext_arrays:
            return

        initparams = sdfg.init_signature(free_symbols=self.free_symbols(sdfg))
        initparams_comma = (', ' + initparams) if initparams else ''

        for storage, arrays in ext_arrays.items():
            size = 0
            for subsdfg, aname, arr in arrays:
                size += arr.total_size * arr.dtype.bytes

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
                offset += arr.total_size * arr.dtype.bytes

            # Footer
            callsite_stream.write('}', sdfg)

    def _readable_cpu_active(self) -> bool:
        """The readable experimental CPU code generator is selected (``compiler.cpu.implementation``)."""
        return config.Config.get('compiler', 'cpu', 'implementation') == 'experimental_readable'

    def _structured_control_flow(self, sdfg: SDFG) -> bool:
        """Whether ``sdfg``'s control flow can only emit gotos that never cross a state-body
        declaration: every region is a strict line graph -- each block has at most one out-edge and
        that edge is UNCONDITIONAL -- with no ``UnstructuredControlFlow`` region. Branching is then
        carried by ``ConditionalBlock`` (its branch bodies each ``{ }``-scoped) and loops by
        ``LoopRegion``, so the emitted state machine only falls through between siblings.

        This is STRICTER than ``control_flow.py``'s ``contains_irreducible`` on purpose: a block with a
        single CONDITIONAL out-edge has ``out_degree == 1`` yet ``control_flow.py`` emits it via the
        ``exit_on_else`` path as ``if (cond) { goto __state_dst; } else { goto __state_exit_<cfg>; }``.
        That ``goto __state_exit`` jumps forward over every following sibling state, so if any crossed
        state had its C scope elided and declared something, the jump would cross an initialization
        (ill-formed C++). Rejecting conditional out-edges removes that hazard. An unstructured region
        (raw multi-edge goto branching) is likewise rejected. Cached per ``cfg_id``. When False, the
        experimental state-scope elision is disabled and every state keeps its scope (matching legacy).
        """
        key = sdfg.cfg_id
        cached = self._structured_cfg.get(key)
        if cached is not None:
            return cached
        result = True
        for region in sdfg.all_control_flow_regions():
            if isinstance(region, UnstructuredControlFlow):
                result = False
                break
            # Only real ControlFlowRegions carry a block graph (a ConditionalBlock holds branch
            # regions, each itself visited and checked). A block is safe only with <=1 out-edge AND,
            # if it has one, an unconditional edge (a conditional edge emits a crossing goto -- above).
            if isinstance(region, ControlFlowRegion):
                for node in region.nodes():
                    out_edges = region.out_edges(node)
                    if len(out_edges) > 1 or (out_edges and not out_edges[0].data.is_unconditional()):
                        result = False
                        break
                if not result:
                    break
        self._structured_cfg[key] = result
        return result

    def state_needs_brace(self, state: SDFGState) -> bool:
        """Whether a non-empty state's body must be wrapped in its own ``{ ... }`` C scope.

        Always True for the legacy generator, so its output is byte-identical. The experimental readable
        generator drops the scope only when the state provably declares NOTHING at its own (state-body)
        scope, so no inter-state ``goto`` can cross an initialization.

        ``to_allocate`` is necessary but NOT a complete inventory of state-scope declarations -- the
        shared tasklet path also emits, directly at state-body scope (``cpu.py`` ``outer_stream_begin``,
        not via ``to_allocate``): inter-tasklet ``code->code`` register temporaries (``T tmp;`` -- for a
        non-trivially-constructible type a goto cannot cross it) and node-level instrumentation timers.
        Rather than enumerate every such source, this is a default-deny positive whitelist: elide only
        when every top-level node is a map scope (``MapEntry``/``MapExit``, whose loops brace their own
        bodies and whose scope transients allocate inside those loops) or an ``AccessNode`` (no decl).
        Any other top-level node -- a state-level tasklet, nested SDFG, library node, reduction, etc. --
        or any node-level instrumentation keeps the scope. Combined with ``_structured_control_flow``
        (no crossing goto) and ``to_allocate`` empty (no tracked transient), the elided state is
        guaranteed declaration-free.
        """
        if not self._readable_cpu_active():
            return True
        if state.instrument != dtypes.InstrumentationType.No_Instrumentation:
            return True
        if not self._structured_control_flow(state.sdfg):
            return True
        if self.to_allocate.get(state):
            return True
        scope = state.scope_dict()
        for node in state.nodes():
            if scope[node] is not None:
                continue  # nested inside a map -> that scope braces it (and its declarations)
            # An instrumented node declares its timers at state scope, so the brace must bound them.
            # Read each property directly and against ITS OWN enum: a Property is stored under a
            # mangled ``_name``, so a ``vars(node).get('instrument')`` lookup silently returns the
            # default and never fires; and an AccessNode's ``instrument`` is a DataInstrumentationType,
            # which never compares equal to an InstrumentationType member of the same name.
            if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
                if node.map.instrument != dtypes.InstrumentationType.No_Instrumentation:
                    return True
            elif isinstance(node, nodes.AccessNode):
                if node.instrument != dtypes.DataInstrumentationType.No_Instrumentation:
                    return True
            else:
                return True  # a top-level tasklet / nested SDFG / library node may declare at state scope
        return False

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
        def dispatch_state(state: SDFGState) -> str:
            stream = CodeIOStream()
            self._dispatcher.dispatch_state(state, global_stream, stream)
            opbar.next()
            states_generated.add(state)  # For sanity check
            return stream.getvalue()

        callsite_stream.write(cflow.control_flow_region_to_code(sdfg, dispatch_state, self, sdfg.symbols), sdfg)

        opbar.done()

        return states_generated

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
        # Every read-only analysis codegen needs, resolved once through one pipeline.
        analysis_results = CodegenAnalysisPipeline().apply_pass(top_sdfg, {})
        reachability = analysis_results[StateReachability.__name__]
        alloc_scopes = analysis_results[AllocationScopes.__name__]
        self.symbol_scopes = analysis_results[SymbolScopes.__name__]
        instances = analysis_results[AccessInstances.__name__]
        access_instances = instances['access_instances']
        code_instances = instances['code_instances']
        shared_transients = instances['shared_transients']

        # Symbols-and-constants stays here: it is memoized on this code generator, not on the SDFG.
        fsyms = {sdfg.cfg_id: self.symbols_and_constants(sdfg) for sdfg in top_sdfg.all_sdfgs_recursive()}

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
                states_with_data = alloc_scopes['data_states'][sdfg.cfg_id].get(name, [])
                curstate: SDFGState = states_with_data[0] if states_with_data else None
                multistate = len(states_with_data) > 1
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
                multistate = False

                # Does the array appear in inter-state edges or loop / conditional block conditions etc.?
                multistate = name in alloc_scopes['meta_symbols'][sdfg.cfg_id]

                # Code nodes reading the container directly from their code (no
                # AccessNode) count as uses for the scope decision as well.
                code_users = code_instances[sdfg.cfg_id].get(name, [])
                # A state with neither an access node for `name` nor a code user of it contributes
                # nothing below, so skipping it avoids its scope_dict and node walk.
                relevant = alloc_scopes['root_data_states'][sdfg.cfg_id].get(name,
                                                                             frozenset()) | {s
                                                                                             for s, _ in code_users}
                for state in sdfg.states():
                    if multistate:
                        break
                    if state not in relevant:
                        continue
                    sdict = state.scope_dict()
                    state_code_users = {n for s, n in code_users if s is state}
                    for node in state.nodes():
                        if node not in state_code_users:
                            if not isinstance(node, nodes.AccessNode):
                                continue
                            if node.root_data != name:
                                continue

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
                        curscope = sdscope.common_parent_scope(sdict, scope, curscope)

                    if multistate:
                        break

                if multistate:
                    alloc_scope = sdfg
                elif (isinstance(curscope, SDFGState) and curstate is not None and desc.storage
                      in (dtypes.StorageType.CPU_Heap, dtypes.StorageType.GPU_Global, dtypes.StorageType.Default)
                      and scope_allocation_repeats_per_iteration(curstate)
                      and first_node_instance is not None and not utils.is_nonfree_sym_dependent(
                          first_node_instance, desc, first_state_instance, fsyms[sdfg.cfg_id])):
                    # Placing it at the one state that uses it re-allocates the buffer on every iteration of
                    # the enclosing loop; the SDFG entry dominates that state, so one buffer serves them all.
                    # Only the PLACEMENT moves -- ``desc.lifetime`` stays Scope.
                    alloc_scope = sdfg
                    alloc_state = curstate
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

                    curscope = first_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, False, True, False))
                    curscope = last_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, last_state_instance, last_node_instance, False, False, True))
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

    def order_views_after_their_sources(self, entries: list[tuple[Any, ...]]) -> list[int]:
        """Order allocations so a view is bound only once the container it points into exists.

        ``to_allocate`` is filled in descriptor order, which says nothing about who views whom. A
        view emitted first takes the address of a pointer that is still null, or still holds the
        previous iteration's freed buffer -- the read through it is then wild.
        """
        allocated_at: dict[str, int] = {}
        for index, (_, _, node, _, allocate, _) in enumerate(entries):
            # A scope entry also lands here, and it names no data.
            if allocate and isinstance(node, nodes.AccessNode) and node.data not in allocated_at:
                allocated_at[node.data] = index

        def source_of(index: int) -> int | None:
            tsdfg, state, node, _, allocate, _ = entries[index]
            if not allocate or state is None or not isinstance(node, nodes.AccessNode):
                return None
            if not isinstance(node.desc(tsdfg), data.View):
                return None
            # A view bound through a scope resolves to the scope node, which names no data.
            viewed = utils.get_view_node(state, node)
            if not isinstance(viewed, nodes.AccessNode) or viewed.data == node.data:
                return None
            source = allocated_at.get(viewed.data)
            return None if source == index else source

        # Depth in the view chain, so a stable sort leaves every unrelated allocation where it was.
        depth: dict[int, int] = {}
        for start in range(len(entries)):
            chain: list[int] = []
            index: int | None = start
            while index is not None and index not in depth and index not in chain:
                chain.append(index)
                index = source_of(index)
            base = 0 if index is None or index in chain else depth[index]
            for offset, member in enumerate(reversed(chain)):
                depth[member] = base + offset + 1
        return sorted(range(len(entries)), key=lambda index: depth[index])

    def allocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG],
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        if len(self.to_allocate[scope]) == 0:
            return
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_allocation_begin(sdfg, scope, callsite_stream)
        """ Dispatches allocation of all arrays in the given scope. """
        entries = self.to_allocate[scope]
        for index in self.order_views_after_their_sources(entries):
            tsdfg, state, node, declare, allocate, _ = entries[index]
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

    def deallocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode, SDFGState,
                                                                                          SDFG],
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

        outside_symbols = sdfg.arglist() if is_top_level else set()

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
            if mpr_lowering.standalone():
                # MPR's entry point IS the program: no state pointer, no ``__program_`` wrapper to
                # forward through, and C linkage so a ctypes / dlopen caller finds it under the
                # SDFG's own name. The argument list is unchanged, so the signature stays the one
                # ``sdfg.arglist()`` describes -- which is what the test harness builds its ctypes
                # argument types from.
                # ``extern "C"`` is a C++ construct and the C dialect's ABI is already C's.
                linkage = '' if mpr_lowering.standalone_c() else 'extern "C" '
                function_signature = f'{linkage}void {sdfg.name}({params})\n{{'
            else:
                if params:
                    params = ', ' + params
                function_signature = f'void __program_{sdfg.name}_internal({mangle_dace_state_struct_name(sdfg)}*__state{params})\n{{'

            self.generate_footer(sdfg, footer_global_stream, footer_stream)
            self.generate_external_memory_management(sdfg, footer_stream)

            header_global_stream.write(global_stream.getvalue())
            header_global_stream.write(footer_global_stream.getvalue())
            generated_header = header_global_stream.getvalue()
            if self._readable_cpu_active():
                from dace.codegen.targets.experimental_cpu import deduplicate_includes  # Avoid circular import
                generated_header = deduplicate_includes(generated_header)

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
        gotos = re.findall(r'goto (.*?);', generated_code)
        goto_ctr = collections.Counter(gotos)
        clean_code = ''
        last_line = ''
        for line in generated_code.split('\n'):
            # Empty line
            if not line.strip():
                continue
            # Empty line with semicolon
            if re.match(r'^\s*;\s*', line):
                continue
            # Label that might be unused
            label = re.findall(r'^\s*([a-zA-Z_][a-zA-Z_0-9]*):\s*[;]?\s*////.*$', line)
            if len(label) > 0:
                if label[0] not in gotos:
                    last_line = ''
                    continue
                if f'goto {label[0]};' in last_line and goto_ctr[label[0]] == 1:  # goto followed by label
                    clean_code = clean_code[:-len(last_line) - 1]
                    last_line = ''
                    continue
            clean_code += line + '\n'
            last_line = line

        # Return the generated global and local code strings
        return (generated_header, clean_code, self._dispatcher.used_targets, self._dispatcher.used_environments)


def scope_allocation_repeats_per_iteration(state: SDFGState) -> bool:
    """Whether a scope allocation placed at ``state`` re-runs on every iteration of an enclosing loop.

    Ascends exactly as the dominator walk does: a block that is its region's entry is dominated by
    whatever dominates the region, so the region collapses to a single node in its parent and the walk
    continues. Reaching a :class:`LoopRegion` on that path means the allocation sits in a loop body.
    """
    block: ControlFlowBlock = state
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        block, region = region, region.parent_graph
    return False


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


def pad_control_flow_region_boundaries(top_sdfg: SDFG):
    """Add an empty state before and after each loop / conditional block.

    A transient whose shape depends on a symbol assigned *inside* such a block is
    allocated at the closest common dominator state of its accesses; without a
    state on the block's boundary that dominator can precede the symbol's
    definition, emitting ``new T[sym]`` with an undefined ``sym``.  A landing
    state on either side gives the allocator a valid placement where the symbol
    is already defined.  Runs before codegen freezes the SDFG.
    """
    for cfg in list(top_sdfg.all_control_flow_regions()):
        for block in list(cfg.nodes()):
            if isinstance(block, (ConditionalBlock, LoopRegion)):
                cfg.add_state_before(block, is_start_block=block is cfg.start_block)
                cfg.add_state_after(block)
    top_sdfg.reset_cfg_list()
