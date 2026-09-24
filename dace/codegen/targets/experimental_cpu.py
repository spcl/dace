# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Experimental "readable" CPU code generator (``compiler.cpu.implementation = experimental_readable``).

Changes only how tasklets and array accesses are emitted: accesses go through a generated
``<array>_idx(...)`` index function, tasklets whose connectors ``InlineTaskletConnectors`` inlined access
arrays directly, and write-once data marked by ``MarkConstInit`` becomes ``const``/``constexpr``. The GPU
generator emits device tasklets through the shared CPU instance, so this also applies inside kernels.
"""
import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy
from pygments.lexers import CppLexer
from pygments.token import Token

from dace import data as dt
from dace import dtypes, cpf_lowering, symbolic
from dace.codegen import cppunparse
from dace.codegen.codeobject import CODE_ANNOTATION
from dace.codegen.common import emits_tree_reductions, sym2cpp
from dace.codegen.prettycode import CodeIOStream
from dace.config import Config
from dace.codegen.dispatcher import DefinedType
from dace.codegen.targets import cpp
from dace.codegen.targets.cpu import (CPUCodeGen, LoopCounterIndex, aligned_new_value, decl_placement, hoist_loop_decls,
                                      loop_counter_index, loop_local_counter_loop, loop_region_index_ctype,
                                      map_schedule_is_sequential, scalar_init_style, use_aligned_operator_new)
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.ordered import OrderedSet
from dace.properties import CodeBlock
from dace.sdfg import SDFG, nodes, type_inference
from dace.sdfg.state import SDFGState
from dace.sdfg.utils import dynamic_map_inputs
from dace.transformation.passes.canonicalize.annotate_loop_kinds import PARALLEL

#: C++ integer type for computed flat indices, per ``codegen_params.index_ctype`` (exact-width types).
INDEX_CTYPES = {'int64': 'int64_t', 'int32': 'int32_t'}
# ``<array>_idx`` / symbolic ``<array>_size`` qualifier: host+device callable, inlined, constexpr.
INDEX_FUNCTION_QUALIFIER = 'static DACE_HDFI constexpr'
# Constant ``<array>_size`` qualifier: ``consteval`` folds the extent at compile time (C++20).
SIZE_CONSTEVAL_QUALIFIER = 'static DACE_HDFI consteval'
# Standalone output has no ``types.h`` defining ``DACE_HDFI``, so its host expansion is written out.
STANDALONE_INDEX_FUNCTION_QUALIFIER = 'static constexpr inline'
STANDALONE_SIZE_CONSTEVAL_QUALIFIER = 'static consteval inline'
# C has no constexpr functions; gcc and clang emit the same code for this and a macro at -O1 and up.
STANDALONE_C_INDEX_FUNCTION_QUALIFIER = 'static inline'
# Identifier tokens of code without an AST here; over-matching only costs a refusal.
IDENTIFIER_TOKENS = re.compile(r'[A-Za-z_]\w*')
# Declarator punctuation _declared_identifiers looks through (``double *p``, ``T &r``, ``vector<T> v``).
DECLARATOR_TOKENS = frozenset({'*', '&', '>'})
INCLUDE_LINE = re.compile(r'^\s*#\s*include\s')
PREPROCESSOR_IF = re.compile(r'^\s*#\s*if')
PREPROCESSOR_ENDIF = re.compile(r'^\s*#\s*endif')
# Storages of a scalar held by value in a plain brace.
VALUE_SCALAR_STORAGES = (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)


@dataclass(slots=True)
class InstrumentReferences:
    """Instrumentation code of ``sdfg`` that may name a loop counter outside its loop: condition texts and
    the symbols symbol-instrumented states dump."""
    sdfg: SDFG
    condition_texts: List[str]
    dumped_symbols: OrderedSet[str]


@dataclass(slots=True)
class DeclarationRun:
    """Counter-gate indices for one run of interstate declarations, which ``framecode`` emits back to back
    into one SDFG's callsite stream before any state is generated."""
    sdfg: SDFG
    stream: CodeIOStream
    counters: LoopCounterIndex
    instruments: Optional[InstrumentReferences] = None


def experimental_loop_local_counter_ctype(name: str, dtype: dtypes.typeclass, run: DeclarationRun) -> Optional[str]:
    """C++ type to declare a LoopRegion counter inside its ``for``-init clause regardless of
    ``decl_placement``, or None to keep it hoisted. Gates match
    :func:`dace.codegen.targets.cpu.loop_local_counter_ctype`, plus instrumentation references.
    """
    loop = loop_local_counter_loop(name, run.counters)
    if loop is None:
        return None
    loop_sdfg = loop.sdfg
    if run.instruments is None or run.instruments.sdfg is not loop_sdfg:
        run.instruments = instrument_references(loop_sdfg)
    if loop_variable_in_instrument_conditions(name, run.instruments):
        return None
    return loop_region_index_ctype() or dtype.ctype


def instrument_references(sdfg: SDFG) -> InstrumentReferences:
    """Instrumentation emitted around state boundaries, where a ``for``-init counter is not in scope."""
    condition_texts: List[str] = []
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            cond = node.instrument_condition
            if isinstance(cond, CodeBlock) and cond.as_string:
                condition_texts.append(cond.as_string)
    dumped_symbols: OrderedSet[str] = OrderedSet()
    for state in sdfg.states():
        if state.symbol_instrument != dtypes.DataInstrumentationType.No_Instrumentation:
            dumped_symbols.update(state.defined_symbols())
        sym_cond = state.symbol_instrument_condition
        if isinstance(sym_cond, CodeBlock) and sym_cond.as_string:
            condition_texts.append(sym_cond.as_string)
    return InstrumentReferences(sdfg, condition_texts, dumped_symbols)


def loop_variable_in_instrument_conditions(name: str, refs: InstrumentReferences) -> bool:
    """Whether instrumentation code emitted outside its loop may reference ``name``."""
    if name in refs.dumped_symbols:
        return True
    name_re = re.compile(r'\b' + re.escape(name) + r'\b')
    return any(name_re.search(text) for text in refs.condition_texts)


def code_blocks_of(value) -> Tuple[CodeBlock, ...]:
    """The ``CodeBlock`` values of one property value: bare, or inside a list / dict. Walking properties
    finds code of node classes this file does not know, so the deferral gate refuses on them."""
    if isinstance(value, CodeBlock):
        return (value, )
    if isinstance(value, dict):
        return tuple(v for v in value.values() if isinstance(v, CodeBlock))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(v for v in value if isinstance(v, CodeBlock))
    return ()


def identifiers_in(blocks) -> Set[str]:
    """Identifier tokens of a sequence of ``CodeBlock``s."""
    names: Set[str] = set()
    for block in blocks:
        text = block.as_string
        if text:
            names |= set(IDENTIFIER_TOKENS.findall(text))
    return names


def index_function_qualifier() -> str:
    """Qualifier of the ``<array>_idx`` / ``<array>_size`` helpers, per
    ``compiler.cpu.codegen_params.index_fn_qualifier``."""
    if Config.get('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier') == 'always_inline':
        return 'static __attribute__((always_inline)) inline constexpr'
    if cpf_lowering.standalone():
        return STANDALONE_INDEX_FUNCTION_QUALIFIER
    return INDEX_FUNCTION_QUALIFIER


def index_ctype() -> str:
    """Integer type the index helpers compute in, per ``compiler.cpu.codegen_params.index_ctype``.
    ``int32`` overflows past 2**31 elements, so it is opt-in only."""
    return INDEX_CTYPES[Config.get('compiler', 'cpu', 'codegen_params', 'index_ctype')]


def size_qualifier(is_constant: bool) -> str:
    """``consteval`` for a constant ``<array>_size`` extent under C++20+, else ``constexpr``."""
    standalone = cpf_lowering.standalone()
    if is_constant and int(str(Config.get('compiler', 'cpp_standard')).strip()) >= 20:
        return STANDALONE_SIZE_CONSTEVAL_QUALIFIER if standalone else SIZE_CONSTEVAL_QUALIFIER
    return STANDALONE_INDEX_FUNCTION_QUALIFIER if standalone else INDEX_FUNCTION_QUALIFIER


def c_heap_alloc_stmt(alloc_name: str, ctype: str, count: str, nodedesc: Optional[dt.Data]) -> str:
    """The C allocation of a heap transient, paired with ``free``. ``aligned_alloc`` needs a multiple of
    the alignment, so the byte count is rounded up.

    :param alloc_name: the assignment target: a pointer name or a full declarator.
    :param ctype: the element type.
    :param count: the printed element count.
    :param nodedesc: the descriptor, read for its alignment.
    :returns: the allocation statement.
    """
    # Explicit ``size_t`` cast: an implicit signed-to-size_t conversion is a -Wsign-conversion diagnostic.
    if nodedesc is None or not use_aligned_operator_new(nodedesc):
        return '%s = malloc(sizeof(%s) * (size_t)(%s));\n' % (alloc_name, ctype, count)
    alignment = aligned_new_value(nodedesc)
    bytes_needed = '((sizeof(%s) * (size_t)(%s) + %d) / %d) * %d' % (ctype, count, alignment - 1, alignment, alignment)
    return '%s = aligned_alloc(%d, %s);\n' % (alloc_name, alignment, bytes_needed)


def format_index_helper(qualifier: str, ctype: str, fnname: str, parameters: List[str], body: str) -> str:
    """The ``<array>_idx`` / ``<array>_size`` helper as a function, scoped and typed unlike a macro. C
    gets ``static inline`` since it has no spelling for the C++ qualifier.

    :param qualifier: the C++ qualifier, ignored in C.
    :param ctype: the integer type the helper computes in.
    :param fnname: the helper's name.
    :param parameters: the parameter names, in order.
    :param body: the printed expression over ``parameters``.
    :returns: the definition to emit once per translation unit.
    """
    declared = ', '.join('%s %s' % (ctype, name) for name in parameters)
    if not cpf_lowering.standalone_c():
        return '%s %s %s(%s) { return %s; }' % (qualifier, ctype, fnname, declared, body)
    return '%s %s %s(%s) { return %s; }' % (STANDALONE_C_INDEX_FUNCTION_QUALIFIER, ctype, fnname, declared
                                            or 'void', body)


def format_index_access(ptrname: str, fnname: str, indices: List[str], extra: List[str]) -> str:
    """C++ ``ptr[fn(idx.., extra..)]`` access through a registered ``<array>_idx`` index function."""
    call_args = [sym2cpp(symbolic.pystr_to_symbolic(ix)) for ix in indices] + list(extra)
    return '%s[%s(%s)]' % (ptrname, fnname, ', '.join(call_args))


def flat_offset(indices: List, desc: dt.Data):
    """Flat element offset ``index . strides + offset . strides``, as the ``<array>_idx`` helper computes."""
    strides = [symbolic.pystr_to_symbolic(str(s)) for s in desc.strides]
    offset = [symbolic.pystr_to_symbolic(str(o)) for o in desc.offset]
    flat = sum(symbolic.pystr_to_symbolic(indices[i]) * strides[i] for i in range(len(strides)))
    return flat + sum(offset[i] * strides[i] for i in range(len(strides)))


def parenthesized_ptr(expr: str) -> str:
    """A base-pointer expression safe to substitute textually for a connector: ``a + 1`` subscripted by
    the body would parse as ``a + (1[_i])``, so a non-identifier is parenthesized."""
    return expr if expr.isidentifier() else f'({expr})'


def loop_access_form() -> str:
    """``compiler.cpu.codegen_params.loop_access_form``: ``indexed`` (default) or ``ptr_increment``, which
    applies only where :meth:`ExperimentalCPUCodeGen.build_walk_plan` proves a pointer walk equivalent."""
    return Config.get('compiler', 'cpu', 'codegen_params', 'loop_access_form')


def index_expr_nodes(slicenode: ast.AST) -> List[ast.AST]:
    """The per-dimension index expressions of a subscript's slice."""
    if isinstance(slicenode, ast.Tuple):
        return list(slicenode.elts)
    return [slicenode]


def subscript_index_strings(slicenode: ast.AST) -> List[str]:
    """The per-dimension index expressions of a subscript's slice as source strings."""
    return [ast.unparse(e) for e in index_expr_nodes(slicenode)]


def deduplicate_includes(code: str) -> str:
    """Drop repeated ``#include`` lines of assembled global code, keeping the first of each. Includes
    under ``#if`` are kept and not counted; lines compare without their ``////__DACE`` annotation.
    """
    seen: Set[str] = set()
    depth = 0
    kept: List[str] = []
    for line in code.split('\n'):
        if PREPROCESSOR_IF.match(line):
            depth += 1
        elif PREPROCESSOR_ENDIF.match(line):
            depth = max(depth - 1, 0)
        elif depth == 0 and INCLUDE_LINE.match(line):
            key = CODE_ANNOTATION.sub('', line).strip()
            if key in seen:
                continue
            seen.add(key)
        kept.append(line)
    return '\n'.join(kept)


# Not registered with the target registry: ``dace.codegen.codegen.generate_code`` selects it for
# ``compiler.cpu.implementation = experimental_readable``, and registering would instantiate it twice.
class ExperimentalCPUCodeGen(CPUCodeGen):
    """ Human-readable CPU/GPU-kernel code generator (see module docstring). """

    experimental_codegen = True

    def __init__(self, frame, sdfg):
        super().__init__(frame, sdfg)
        # Helper name -> C++ definition, for ``<array>_idx`` and ``<array>_size`` helpers.
        self._index_functions: Dict[str, str] = {}
        self._size_functions: Dict[str, str] = {}
        # Output-file key -> helper names already flushed into that file (see _flush_generated_functions).
        self._emitted_functions: Dict[Union[int, str], Set[str]] = {}
        # (base name, signature) -> helper name, so same-named arrays of different shape get distinct helpers.
        self._index_sig_to_name: Dict[tuple, str] = {}
        self._size_sig_to_name: Dict[tuple, str] = {}
        # id(tasklet) -> identifiers in its body / identifiers its C++ body declares.
        self._body_identifiers: Dict[int, Set[str]] = {}
        self._body_declarations: Dict[int, Set[str]] = {}
        # id(C++ tasklet) -> {connector: inlined access}; listed connectors skip their copy-in/out.
        self._cpp_inline: Dict[int, Dict[str, str]] = {}
        # ptr_increment: open MapEntry chain, id(map) -> walk plan ({} if not walkable), and the maps whose
        # pointer declarations / increments were already emitted.
        self._map_scope_stack: List[nodes.MapEntry] = []
        self._walk_plans: Dict[int, dict] = {}
        self._walk_emitted_decls: Set[int] = set()
        self._walk_emitted_incs: Set[int] = set()
        # decl_placement = late: ptrname -> deferred scalar declaration, emitted at its first-use tasklet.
        self._late_pending: Dict[str, dict] = {}
        # Deferral-gate caches keyed by id() of frozen codegen-time objects; None means "not analysable".
        self._eager_alloc_scopes: Optional[Dict[Tuple[int, str], list]] = None
        self._name_owners: Dict[int, Optional[Dict[str, Set[int]]]] = {}
        self._node_references: Dict[int, Optional[Set[str]]] = {}
        self._nested_free_names: Dict[int, Optional[Set[str]]] = {}
        # const_init bindings registered while lowering a tasklet, consumed by emit_tasklet_body_block.
        self.const_pending: List[dict] = []
        # Origin GUIDs of library-node descriptions already written into this unit (see emit_provenance).
        self._emitted_provenance: Set[str] = set()
        self.declaration_run: Optional[DeclarationRun] = None

    def emit_interstate_variable_declaration(self, name, dtype, callsite_stream, sdfg):
        """LoopRegion counters are declared in their ``for``-init clause; other interstate symbols stay
        hoisted."""
        run = self.declaration_run
        if run is None or run.sdfg is not sdfg or run.stream is not callsite_stream:
            run = DeclarationRun(sdfg, callsite_stream, loop_counter_index(sdfg))
            self.declaration_run = run
        local_ctype = experimental_loop_local_counter_ctype(name, dtype, run)
        if local_ctype is not None:
            self._frame.loop_local_counters[(sdfg.cfg_id, name)] = local_ctype
            self._frame.dispatcher.defined_vars.add(name, DefinedType.Scalar, local_ctype)
            return
        super().emit_interstate_variable_declaration(name, dtype, callsite_stream, sdfg)

    def get_generated_codeobjects(self):
        # Split-nest / external units assemble global code from per-tasklet writes, so includes repeat.
        objects = super().get_generated_codeobjects()
        for obj in objects:
            obj.code = deduplicate_includes(obj.code)
        return objects

    def map_scope_needs_brace(self, sdfg, state_dfg, node: nodes.MapEntry) -> bool:
        """Whether the map's ``{ }`` scope bounds a declaration; without one the braces only nest."""
        if dynamic_map_inputs(state_dfg, node):
            return True
        if any(
                hoist_loop_decls(node, self._map_loop_will_have_openmp_pragma(sdfg, state_dfg, node, i))
                for i in range(len(node.map.range))):
            return True
        if self.walk_plan_for(sdfg, state_dfg, node):
            return True
        if node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            return True
        if node.map.instrument != dtypes.InstrumentationType.No_Instrumentation:
            return True
        # A complex-type ``#pragma omp declare reduction`` must not leak into sibling maps.
        if (node.map.schedule == dtypes.ScheduleType.CPU_Multicore and emits_tree_reductions(self.experimental_codegen)
                and any(declare is not None
                        for _op, _ct, _dname, declare in self._collect_omp_reductions(sdfg, state_dfg, node))):
            return True
        return False

    def emit_provenance(self, node, cfg, state_id, callsite_stream) -> None:
        """Write the ``// <original library node>`` line CPF recorded for an expansion, once per
        description per unit. A no-op outside standalone rendering."""
        record = cpf_lowering.describe(node.guid)
        if record is None:
            return
        origin, description = record
        if origin in self._emitted_provenance:
            return
        self._emitted_provenance.add(origin)
        for line in str(description).splitlines():
            if line.strip():
                callsite_stream.write('// %s' % line, cfg, state_id, node)

    def _generate_MapEntry(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        self.emit_provenance(node, cfg, state_id, callsite_stream)
        # A map born in codegen (copy lowering) has no label; a Map is data-parallel by definition.
        hint = cpf_lowering.hint_comment(node.specialization_hint or PARALLEL)
        if hint:
            callsite_stream.write(hint.rstrip('\n'), cfg, state_id, node)
        # Plan before the base emitter, whose map_scope_needs_brace reads it.
        self.walk_plan_for(sdfg, cfg.state(state_id), node)
        self._map_scope_stack.append(node)
        super()._generate_MapEntry(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def _generate_MapExit(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        super()._generate_MapExit(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        # Drop the plan too: ids are reused once a map is freed.
        if self._map_scope_stack:
            entry = self._map_scope_stack.pop()
            self._walk_plans.pop(id(entry.map), None)
            self._walk_emitted_decls.discard(id(entry.map))
            self._walk_emitted_incs.discard(id(entry.map))

    def generate_scope_preamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        super().generate_scope_preamble(sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream)
        # ``outer_stream`` sits before the loop headers: declare the walking pointers there.
        plan = self._current_walk_plan()
        if not plan:
            return
        key = id(self._map_scope_stack[-1].map)
        if key in self._walk_emitted_decls:
            return
        self._walk_emitted_decls.add(key)
        for decl in plan['decls']:
            outer_stream.write(decl + '\n', sdfg)

    def generate_scope_postamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        super().generate_scope_postamble(sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream)
        # ``inner_stream`` ends the loop body: advance each walking pointer there.
        plan = self._current_walk_plan()
        if not plan:
            return
        key = id(self._map_scope_stack[-1].map)
        if key in self._walk_emitted_incs:
            return
        self._walk_emitted_incs.add(key)
        for inc in plan['incs']:
            inner_stream.write(inc + '\n', sdfg)

    def _current_walk_plan(self) -> Optional[dict]:
        if not self._map_scope_stack:
            return None
        plan = self._walk_plans.get(id(self._map_scope_stack[-1].map))
        return plan or None

    def current_walk_accesses(self) -> Optional[Dict[tuple, str]]:
        """``{(array, index strings): pointer}`` of the map being emitted, or None."""
        plan = self._current_walk_plan()
        return plan['accesses'] if plan else None

    def walk_plan_for(self, sdfg, state_dfg, node: nodes.MapEntry) -> dict:
        """Memoized :meth:`build_walk_plan` of map ``node``."""
        cached = self._walk_plans.get(id(node.map))
        if cached is not None:
            return cached
        plan = self.build_walk_plan(sdfg, state_dfg, node)
        self._walk_plans[id(node.map)] = plan
        return plan

    def build_walk_plan(self, sdfg, state_dfg, node: nodes.MapEntry) -> dict:
        """``{'accesses': {(name, idx_tuple): pointer}, 'decls': [str], 'incs': [str]}`` to walk this map's
        accesses with incrementing base pointers, or ``{}`` unless the walk is provably equivalent.
        """
        if loop_access_form() != 'ptr_increment':
            return {}
        # Sequential only: an OpenMP loop keeps the canonical indexed form its pragma needs.
        if not map_schedule_is_sequential(node):
            return {}
        if node.map.unroll:
            return {}
        if len(node.map.range) != 1 or len(node.map.params) != 1:
            return {}
        if dynamic_map_inputs(state_dfg, node):
            return {}
        # Exactly one Python tasklet in the scope.
        scope_nodes = list(state_dfg.scope_subgraph(node, include_entry=False, include_exit=False).nodes())
        if len(scope_nodes) != 1 or not isinstance(scope_nodes[0], nodes.Tasklet):
            return {}
        tasklet = scope_nodes[0]
        if tasklet.language != dtypes.Language.Python:
            return {}
        # A WCR write needs the atomic resolve path, never ``*p =``.
        exit_node = state_dfg.exit_node(node)
        for edge in list(state_dfg.all_edges(tasklet)) + list(state_dfg.all_edges(exit_node)):
            if edge.data is not None and edge.data.wcr is not None:
                return {}

        var = node.map.params[0]
        begin, _end, skip = node.map.range[0]
        loop_sym = symbolic.pystr_to_symbolic(var)
        skip_sym = symbolic.pystr_to_symbolic(str(skip))
        begin_sym = symbolic.pystr_to_symbolic(str(begin))

        stmts = tasklet.code.code
        if isinstance(stmts, str):
            try:
                body = ast.parse(stmts).body
            except SyntaxError:
                return {}
        else:
            body = list(stmts)

        records: List[tuple] = []  # (name, idx_tuple, desc)
        if not all(self._scan_walkable(stmt, sdfg, records) for stmt in body):
            return {}
        if not records:
            return {}

        accesses: Dict[tuple, str] = {}
        decls: List[str] = []
        incs: List[str] = []
        used_names: Set[str] = set()
        for name, idx_tuple, desc in records:
            key = (name, idx_tuple)
            if key in accesses:
                continue
            flat = symbolic.pystr_to_symbolic(flat_offset(idx_tuple, desc))
            per_iter = symbolic.pystr_to_symbolic(flat.subs(loop_sym, loop_sym + skip_sym) - flat).simplify()
            if loop_sym in per_iter.free_symbols:
                return {}
            ptrname = self.ptr(name, desc, sdfg)
            try:
                defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
            except KeyError:
                return {}  # base pointer not in scope yet (e.g. a scope transient)
            if defined_type != DefinedType.Pointer:
                return {}
            pointer = self._unique_walk_name(name, used_names)
            accesses[key] = pointer
            base_off = sym2cpp(flat.subs(loop_sym, begin_sym))
            start = ptrname if base_off == '0' else '%s + (%s)' % (ptrname, base_off)
            decls.append('%s* %s = %s;' % (desc.dtype.ctype, pointer, start))
            step = sym2cpp(per_iter)
            if step != '0':
                incs.append('%s += %s;' % (pointer, step))
        return {'accesses': accesses, 'decls': decls, 'incs': incs}

    def _scan_walkable(self, astnode, sdfg, records: List[tuple]) -> bool:
        """Record every plain-array subscript of ``astnode``; False on an access a pointer walk cannot
        express (a bare array, an indirect index, a rank mismatch)."""
        if isinstance(astnode, ast.Subscript):
            base = rname(astnode)
            desc = sdfg.arrays.get(base)
            if isinstance(desc, dt.Array) and not isinstance(desc, dt.View):
                for idx in index_expr_nodes(astnode.slice):
                    if any(isinstance(sub, (ast.Subscript, ast.Call)) for sub in ast.walk(idx)):
                        return False
                idx_tuple = tuple(subscript_index_strings(astnode.slice))
                if len(idx_tuple) != len(desc.shape):
                    return False
                records.append((base, idx_tuple, desc))
                return True
        elif isinstance(astnode, ast.Name):
            desc = sdfg.arrays.get(astnode.id)
            return not (isinstance(desc, dt.Array) and not isinstance(desc, dt.View))
        return all(self._scan_walkable(child, sdfg, records) for child in ast.iter_child_nodes(astnode))

    def _unique_walk_name(self, data_name: str, used: Set[str]) -> str:
        base = '__walk_' + re.sub(r'\W', '_', data_name)
        name = base
        k = 1
        while name in used:
            name = '%s_%d' % (base, k)
            k += 1
        used.add(name)
        return name

    def make_keyword_remover(self, sdfg, memlets, defined_symbols):
        return ReadableKeywordRemover(sdfg, memlets, sdfg.constants, self, defined_symbols)

    def _connector_needs_copy(self, node, conn) -> bool:
        # NestedSDFGs and other code nodes keep their connectors as function arguments.
        if not isinstance(node, nodes.Tasklet):
            return True
        if node.language == dtypes.Language.CPP:
            return conn not in self._cpp_inline.get(id(node), {})
        # Other non-Python bodies (MLIR, OpenCL, ...) are emitted verbatim.
        if node.language != dtypes.Language.Python:
            return True
        # InlineTaskletConnectors rewrote inlined connectors out of the body.
        return conn in self._used_identifiers(node)

    def _used_identifiers(self, node) -> Set[str]:
        key = id(node)
        cached = self._body_identifiers.get(key)
        if cached is not None:
            return cached
        code = node.code.as_string if node.code else ''
        if node.language == dtypes.Language.Python:
            ids = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
        else:
            ids = set(IDENTIFIER_TOKENS.findall(code))
        self._body_identifiers[key] = ids
        return ids

    def _generate_Tasklet(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen=None):
        # Decide C++ connector inlining before the base generator asks _connector_needs_copy.
        if isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.CPP:
            state_dfg = cfg.nodes()[state_id]
            self._cpp_inline[id(node)] = self._compute_cpp_inline(sdfg, state_dfg, node)
        super()._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen)
        self._flush_generated_functions(function_stream, cfg, state_id, node)

    # No tasklet banner or separators: each tasklet line carries a trailing ``// <label>``.

    def tasklet_body_comment(self, node) -> str:
        return ''

    def tasklet_body_open_marker(self, node) -> str:
        return ''

    def tasklet_body_close_marker(self, node) -> str:
        return ''

    def emit_tasklet_body_block(self, callsite_stream, cfg, state_id, node, inner_body, postamble, has_locals) -> None:
        self.emit_provenance(node, cfg, state_id, callsite_stream)
        # A connector-free single-statement tasklet collapses onto one brace-free line; anything with
        # locals keeps its own ``{ }`` so a declaration cannot leak into the map body.
        if not has_locals and self._single_statement_body(node):
            # Strip the baked-in ``////__DACE`` tag so ``// <label>`` lands before the stream's fresh one.
            line = re.sub(r'[ \t]*////__(DACE:|CODEGEN;).*', '', inner_body).strip()
            if line:
                # Only here is the line known to be brace-free, the one place a fused declaration is visible.
                line = self.explicit_store_conversion(cfg, state_id, node, line)
                line = self.fuse_pending_decl(node, line)
                self.emit_pending_late_decls(cfg, state_id, node, callsite_stream)
                callsite_stream.write('%s  // %s\n' % (line, node.label), cfg, state_id, node)
                return
        self.emit_pending_late_decls(cfg, state_id, node, callsite_stream)
        callsite_stream.write('{  // %s' % node.label, cfg, state_id, node)
        callsite_stream.write(inner_body, cfg, state_id, node)
        callsite_stream.write(postamble)
        callsite_stream.write('}', cfg, state_id, node)

    @staticmethod
    def tasklet_read_types(cfg, state_id: int, node) -> Dict[str, dtypes.typeclass]:
        """Types of every name a tasklet body may read: its connectors and its data containers, since an
        inlined body names the data (``expr_times_a[0, 0]``)."""
        reads = {name: dtype for name, dtype in node.in_connectors.items() if isinstance(dtype, dtypes.typeclass)}
        state = cfg.state(state_id)
        arrays = state.sdfg.arrays
        for edge in state.in_edges(node):
            data = edge.data.data
            if data in arrays:
                reads.setdefault(data, arrays[data].dtype)
        return reads

    def explicit_store_conversion(self, cfg, state_id: int, node, line: str) -> str:
        """Cast a fused store's right-hand side when its type differs from the out connector's, which an
        implicit conversion would report as ``-Wfloat-conversion``. Standalone renders only.

        :param cfg: the control-flow graph being emitted.
        :param state_id: index of the state holding the tasklet.
        :param node: the tasklet, whose body is one assignment.
        :param line: the emitted statement.
        :returns: the statement, cast when the types differ.
        """
        if not cpf_lowering.standalone() or node.language != dtypes.Language.Python:
            return line
        if len(node.out_connectors) != 1:
            return line
        dst = next(iter(node.out_connectors.values()))
        if not isinstance(dst, dtypes.typeclass) or isinstance(dst, (dtypes.pointer, dtypes.vector)):
            return line
        value = self.assigned_expression(node)
        if value is None:
            return line
        try:
            src = type_inference.infer_expr_type(value, self.tasklet_read_types(cfg, state_id, node))
        except Exception:  # inference is best-effort: an unknown call is not a reason to fail codegen
            return line
        if src is None or src == dst or isinstance(src, (dtypes.pointer, dtypes.vector)):
            return line
        split = re.search(r'(?<![=!<>+\-*/%&|^])=(?!=)', line)
        if split is None:
            return line
        head, tail = line[:split.start()], line[split.end():].strip()
        if not tail.endswith(';'):
            return line
        value = tail[:-1].strip()
        if cpf_lowering.standalone_c():
            return f'{head}= ({dst.ctype})({value});'
        return f'{head}= static_cast<{dst.ctype}>({value});'

    @staticmethod
    def assigned_expression(node):
        """The Python source of the single assignment's right-hand side, or ``None``."""
        try:
            body = ast.parse(node.code.as_string).body
        except (SyntaxError, TypeError, ValueError):
            return None
        if len(body) != 1 or not isinstance(body[0], (ast.Assign, ast.AnnAssign)):
            return None
        return astutils.unparse(body[0].value) if body[0].value is not None else None

    def _single_statement_body(self, node) -> bool:
        if not isinstance(node, nodes.Tasklet) or node.language != dtypes.Language.Python:
            return False
        stmts = node.code.code
        if isinstance(stmts, str):
            stmts = ast.parse(stmts).body
        return len(stmts) == 1 and isinstance(stmts[0], (ast.Assign, ast.AugAssign))

    def rewrite_cpp_tasklet_body(self, node, sdfg, state_dfg) -> str:
        """The C++ body of a native tasklet with every inlinable connector replaced by its access. Only
        pygments ``Token.Name`` tokens are rewritten, so strings and comments stay untouched."""
        body = node.code.as_string
        inline = self._cpp_inline.get(id(node)) or {}
        if not inline:
            return body
        out: List[str] = []
        for tok_type, value in CppLexer().get_tokens(body):
            if tok_type in Token.Name and value in inline:
                out.append(inline[value])
            else:
                out.append(value)
        return ''.join(out)

    def _compute_cpp_inline(self, sdfg, state_dfg, node) -> Dict[str, str]:
        """``{connector: access}`` for every connector of a native tasklet that can be inlined: a scalar as
        an ``<array>_idx(...)`` access, a pointer / whole subset as a base-pointer expression."""
        # A preprocessor line (``ExpandReduceOpenMP``'s ``#pragma omp ... reduction(op:_out[0])``) lexes as
        # one token, so a connector in it cannot be rewritten; keep the classic copies for the tasklet.
        if re.search(r'(?m)^[ \t]*#', node.code.as_string):
            return {}
        in_map: Dict[str, str] = {}
        out_map: Dict[str, str] = {}
        in_edges: Dict[str, object] = {}
        out_edges: Dict[str, object] = {}
        for edge in state_dfg.in_edges(node):
            access = self._cpp_connector_access(sdfg, state_dfg, node, edge, is_output=False)
            if access is not None:
                in_map[edge.dst_conn] = access
                in_edges[edge.dst_conn] = edge
        for edge in state_dfg.out_edges(node):
            access = self._cpp_connector_access(sdfg, state_dfg, node, edge, is_output=True)
            if access is not None:
                out_map[edge.src_conn] = access
                out_edges[edge.src_conn] = edge

        # An inout connector is inlined only if both sides resolve to the same access.
        inout = set(node.in_connectors) & set(node.out_connectors)
        inline: Dict[str, str] = {}
        for name in dict.fromkeys((*in_map, *out_map)):
            if name in inout:
                if name in in_map and name in out_map and in_map[name] == out_map[name]:
                    inline[name] = in_map[name]
            else:
                inline[name] = in_map.get(name, out_map.get(name))

        self._drop_captured_inlines(node, inline)

        # Skipping the copy dispatch also skips registering its target (e.g. the CUDA setup a host-side
        # cudaMemcpy needs), so register it here.
        for name in inline:
            if name in in_edges:
                self._register_inlined_copy_target(sdfg, state_dfg, node, in_edges[name], is_output=False)
            if name in out_edges:
                self._register_inlined_copy_target(sdfg, state_dfg, node, out_edges[name], is_output=True)
        return inline

    def _drop_captured_inlines(self, node, inline: Dict[str, str]) -> None:
        # Drop every connector whose access text names an identifier the body declares (``tblis_tensor A``
        # against an array ``A``): spliced in, it would bind to the local. Reads do not shadow. Dropping a
        # connector puts its name back into the body, hence the fixpoint.
        if not inline:
            return
        declared = self._declared_identifiers(node)
        while True:
            occupied = declared - set(inline)
            captured = [c for c, access in inline.items() if not occupied.isdisjoint(IDENTIFIER_TOKENS.findall(access))]
            if not captured:
                return
            for conn in captured:
                del inline[conn]

    def _declared_identifiers(self, node) -> Set[str]:
        # Names a C++ body declares, over-approximated from tokens: an identifier after a keyword or
        # another identifier, or continuing its comma list, looking through DECLARATOR_TOKENS. ``a * b``
        # reads as a declaration too, which only costs a refusal.
        key = id(node)
        cached = self._body_declarations.get(key)
        if cached is not None:
            return cached
        declared: Set[str] = set()
        previous = None
        in_declarator_list = False
        for token_type, value in CppLexer().get_tokens(node.code.as_string):
            if token_type in Token.Text or token_type in Token.Comment or value in DECLARATOR_TOKENS:
                continue
            if token_type in Token.Name:
                if previous is not None and (previous[0] in Token.Keyword or previous[0] in Token.Name or
                                             (in_declarator_list and previous[1] == ',')):
                    declared.add(value)
                    in_declarator_list = True
                else:
                    in_declarator_list = False
            elif value != ',':
                in_declarator_list = False
            previous = (token_type, value)
        self._body_declarations[key] = declared
        return declared

    def _register_inlined_copy_target(self, sdfg, state_dfg, node, edge, is_output: bool) -> None:
        if is_output:
            src_node, dst_node = node, state_dfg.memlet_path(edge)[-1].dst
        else:
            src_node, dst_node = state_dfg.memlet_path(edge)[0].src, node
        target = self._dispatcher.get_copy_dispatcher(src_node, dst_node, edge, sdfg, state_dfg)
        if target is not None:
            self._dispatcher.used_targets.add(target)

    def _cpp_connector_access(self, sdfg, state_dfg, node, edge, is_output: bool) -> Optional[str]:
        # C++ access for one native-tasklet connector, or None to keep the classic copy.
        conn = edge.src_conn if is_output else edge.dst_conn
        memlet = edge.data
        if not conn:
            return None
        if memlet.data is None or memlet.data not in sdfg.arrays:
            return None
        # Only data in memory, never a tasklet-to-tasklet connector.
        path = state_dfg.memlet_path(edge)
        far = path[-1].dst if is_output else path[0].src
        if not isinstance(far, nodes.AccessNode):
            return None
        desc = sdfg.arrays[memlet.data]
        # Plain arrays (views included) and scalars with flat addressing only.
        if isinstance(desc, (dt.Stream, dt.Reference, dt.ContainerArray, dt.Structure)):
            return None
        if not isinstance(desc, (dt.Array, dt.Scalar)):
            return None
        if memlet.wcr is not None:
            return None
        subset = memlet.subset
        if subset is None:
            return None
        conntype = node.out_connectors[conn] if is_output else node.in_connectors[conn]

        # Pointer / whole-subset connector (a library-call argument): the base pointer at the subset start.
        if isinstance(conntype, dtypes.pointer) or subset.num_elements() != 1:
            ptrname = self.ptr(memlet.data, desc, sdfg)
            try:
                defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
            except KeyError:
                defined_type = None
            if defined_type is not None:
                return parenthesized_ptr(cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self))
            offset = cpp.cpp_offset_expr(desc, subset)
            return ptrname if offset == '0' else parenthesized_ptr('%s + %s' % (ptrname, offset))

        # Single element: an ``<array>_idx`` access, as in ReadableKeywordRemover._bare_access.
        indices = [str(rb) for (rb, _re, _rs) in subset.ranges]
        info = self.array_index_access(sdfg, desc, memlet.data)
        if info is None:
            return self.ptr(memlet.data, desc, sdfg)
        ptrname, fnname, ndim, extra = info
        if len(indices) != ndim:
            return None
        return format_index_access(ptrname, fnname, indices, extra)

    def allocate_array(self,
                       sdfg,
                       cfg,
                       dfg,
                       state_id,
                       node,
                       nodedesc,
                       function_stream,
                       declaration_stream,
                       allocation_stream,
                       allocate_nested_data: bool = True):
        # A write-once scope scalar / length-1 stack array is declared by its fused ``const`` write
        # (ReadableKeywordRemover.visit_Assign); only register it here.
        if self._is_const_scalar(nodedesc, node.data, sdfg):
            self._dispatcher.defined_vars.add(self.ptr(node.data, nodedesc, sdfg), DefinedType.Scalar,
                                              nodedesc.dtype.ctype)
            return
        if self._is_const_len1_array(nodedesc, node.data, sdfg):
            self._dispatcher.defined_vars.add(self.ptr(node.data, nodedesc, sdfg), DefinedType.Pointer,
                                              dtypes.pointer(nodedesc.dtype).ctype)
            return
        if self.defer_scalar_declaration(sdfg, dfg, node, nodedesc):
            return
        super().allocate_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                               allocation_stream, allocate_nested_data)
        # heap_alloc_stmt may have registered an ``<array>_size`` helper.
        self._flush_generated_functions(function_stream, cfg, state_id, node)

    def fused_heap_declarator(self, sdfg, name: str, nodedesc: dt.Data, arrsize, declared: bool, declaration_stream,
                              allocation_stream) -> Optional[str]:
        """Declarator fusing ``T *p;`` and ``p = new T[...];`` into ``T* __restrict__ p = new T[...];``, or
        None to keep the split pair. Sound only when both land in one scope: the pointer was not already
        declared in an enclosing scope (``declared``), and the declaration and allocation share a stream
        (Persistent and External lifetimes allocate into ``__dace_init``, where a local would shadow the
        state-struct member).
        """
        if declared or declaration_stream is not allocation_stream:
            return None
        return self.array_pointer_declarator(name, nodedesc)

    def array_pointer_declarator(self, name: str, nodedesc: dt.Data) -> str:
        """``T* __restrict__ name`` for a heap array pointer; no ``__restrict__`` for a ``may_alias``
        descriptor or when ``compiler.cpu.codegen_params.heap_ptr_restrict`` is ``none``."""
        emit_restrict = (not nodedesc.may_alias
                         and Config.get('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict') == 'restrict')
        restrict = '__restrict__ ' if emit_restrict else ''
        return '%s* %s%s' % (nodedesc.dtype.ctype, restrict, name)

    def heap_alloc_stmt(self,
                        alloc_name: str,
                        ctype: str,
                        arrsize: str,
                        alignment: int = 0,
                        sdfg: Optional['SDFG'] = None,
                        nodedesc: Optional[dt.Data] = None,
                        data_name: Optional[str] = None) -> str:
        # The base aligned ``operator new[]``, with the count through an ``<array>_size`` helper when one
        # is worthwhile (_register_size_function).
        count = arrsize
        if sdfg is not None and nodedesc is not None and data_name is not None:
            registered = self._register_size_function(data_name, nodedesc)
            if registered is not None:
                fnname, call_args = registered
                count = '%s(%s)' % (fnname, ', '.join(call_args))
        if cpf_lowering.standalone_c():
            return c_heap_alloc_stmt(alloc_name, ctype, count, nodedesc)
        placement = ''
        if nodedesc is not None and use_aligned_operator_new(nodedesc):
            placement = ' (std::align_val_t(%d))' % aligned_new_value(nodedesc)
        return '%s = new%s %s[%s];\n' % (alloc_name, placement, ctype, count)

    def heap_free_stmt(self, alloc_name: str, is_array: bool, nodedesc: Optional[dt.Data] = None) -> str:
        """The matching free: ``free`` for both allocation shapes in C."""
        if cpf_lowering.standalone_c():
            return 'free(%s);\n' % alloc_name
        return super().heap_free_stmt(alloc_name, is_array, nodedesc)

    def _flush_generated_functions(self, function_stream, cfg, state_id, node) -> None:
        # Emit each registered helper once per output file, since many streams feed one unit. Host code
        # keys on ``_current_tu_key`` (the frame .cpp, or a split nest's own unit); device code keys on the
        # delegating GPU codegen, which owns the single .cu.
        file_key = self._current_tu_key if self.calling_codegen is self else id(self.calling_codegen)
        emitted = self._emitted_functions.setdefault(file_key, set())
        for registry in (self._index_functions, self._size_functions):
            for name, defn in registry.items():
                if name in emitted:
                    continue
                function_stream.write(defn + '\n', cfg, state_id, node)
                emitted.add(name)

    def defer_scalar_declaration(self, sdfg, dfg, node, desc) -> bool:
        """Register a mutable scope scalar for a late declaration at its first-use tasklet and return True,
        or return False to keep the eager one. See :meth:`late_declarable_scalar` for the gate."""
        placement = self.late_declarable_scalar(sdfg, dfg, node, desc)
        if placement is None:
            return False
        scope_tasklets, setzero = placement
        ptrname = self.ptr(node.data, desc, sdfg)
        self._dispatcher.defined_vars.add(ptrname, DefinedType.Scalar, desc.dtype.ctype)
        state: SDFGState = dfg
        # Only a writer can carry a fused ``T x = expr;``; a ``setzero`` scalar's ``T x = 0;`` never fuses.
        writers = {id(edge.src) for an in state.data_nodes() if an.data == node.data for edge in state.in_edges(an)}
        self._late_pending[ptrname] = {
            'ctype': desc.dtype.ctype,
            'ptrname': ptrname,
            'setzero': setzero,
            'tasklets': {id(t)
                         for t in scope_tasklets},
            'writers': writers,
            'fusable': scalar_init_style() == 'fused' and not setzero,
        }
        return True

    def late_declarable_scalar(self, sdfg, dfg, node, desc) -> Optional[Tuple[Set[nodes.Tasklet], bool]]:
        """``(neighbor_tasklets, setzero)`` if the declaration of scalar ``desc`` (allocated at ``node`` in
        state ``dfg``) may move to its first use, else ``None``.

        Sound only for a plain transient value scalar whose every access is a direct-child tasklet of one
        scope. Two refusals cover all remaining uses: the allocation planner's scope for the eager
        declaration must be that scope (:meth:`eager_allocation_scope`), and nothing but those accesses
        and tasklets may mention the name (:meth:`name_owners`)."""
        # ``fused`` also needs the eager declaration skipped, and first use is the nearest correct place.
        if decl_placement() != 'late' and scalar_init_style() != 'fused':
            return None
        if not isinstance(desc, dt.Scalar) or isinstance(desc, (dt.View, dt.Reference, dt.Stream)):
            return None
        if not desc.transient or desc.const_init or node.data in sdfg.constants_prop:
            return None
        if desc.lifetime != dtypes.AllocationLifetime.Scope:
            return None
        if desc.storage not in VALUE_SCALAR_STORAGES:
            return None
        if not isinstance(dfg, SDFGState):
            return None
        state: SDFGState = dfg
        name = node.data
        access_nodes = [n for n in state.data_nodes() if n.data == name]
        if not access_nodes:
            return None
        scope_dict = state.scope_dict()
        # One scope (state top level or one map body) is the brace the eager declaration occupies too.
        scopes = {scope_dict[n] for n in access_nodes}
        if len(scopes) != 1:
            return None
        scope = scopes.pop()
        neighbor_tasklets: Set[nodes.Tasklet] = set()
        has_write = False
        for an in access_nodes:
            if state.in_degree(an) > 0:
                has_write = True
            elif state.out_degree(an) > 0:
                # Read before any write in this state: the value is carried from a previous execution,
                # which a declaration re-entered per execution would lose.
                return None
            for edge in state.all_edges(an):
                other = edge.dst if edge.src is an else edge.src
                if not isinstance(other, nodes.Tasklet) or scope_dict[other] is not scope:
                    return None
                neighbor_tasklets.add(other)
        if not has_write or not neighbor_tasklets:
            return None
        if self.eager_allocation_scope(sdfg, name) is not (state if scope is None else scope):
            return None
        owners = self.name_owners(sdfg)
        if owners is None:
            return None
        permitted = {id(n) for n in access_nodes} | {id(t) for t in neighbor_tasklets}
        if not owners.get(name, frozenset()) <= permitted:
            return None
        return neighbor_tasklets, node.setzero

    def eager_allocation_scope(self, sdfg: SDFG, name: str) -> Optional[Union[nodes.EntryNode, SDFGState, SDFG]]:
        """The scope the frame's allocation planner picked for ``name``'s eager declaration, or ``None`` for
        no entry or several. The planner already saw every use, so an unanticipated use shows up as a
        scope mismatch, i.e. a refusal.
        """
        table = self._eager_alloc_scopes
        if table is None:
            table = {}
            for alloc_scope, entries in self._frame.to_allocate.items():
                for tsdfg, _, alloc_node, _, _, _ in entries:
                    table.setdefault((id(tsdfg), alloc_node.data), []).append(alloc_scope)
            self._eager_alloc_scopes = table
        found = table.get((id(sdfg), name))
        if found is None or len(found) != 1:
            return None
        return found[0]

    def name_mentions(self, sdfg: SDFG) -> Optional[List[Tuple[Set[str], Tuple[int, ...]]]]:
        """Every ``(names, owner ids)`` mention in ``sdfg``, or ``None`` if a node cannot be analysed. A
        memlet's names are charged to both endpoints; interstate edges and loop / branch conditions to
        ``id(sdfg)``, an owner no candidate permits. Names come from a crude tokenizer, since
        ``used_symbols`` cannot see a container named in a C++ body."""
        mentions: List[Tuple[Set[str], Tuple[int, ...]]] = []
        for state in sdfg.states():
            for graph_node in state.nodes():
                references = self.node_name_references(graph_node, sdfg)
                if references is None:
                    return None
                mentions.append((references, (id(graph_node), )))
            for edge in state.edges():
                names = set(edge.data.free_symbols)
                if edge.data.data:
                    names.add(edge.data.data)
                mentions.append((names, (id(edge.src), id(edge.dst))))
        own = (id(sdfg), )
        for edge in sdfg.all_interstate_edges():
            mentions.append((edge.data.read_symbols() | set(edge.data.assignments.keys()), own))
        for block in sdfg.all_control_flow_blocks():
            mentions.append((self.code_property_names(block), own))
        for region in sdfg.all_control_flow_regions():
            # ``get_meta_codeblocks``: loop control and branch conditions (ConditionalBlock keeps them
            # outside its properties).
            mentions.append((identifiers_in(region.get_meta_codeblocks()), own))
        return mentions

    def name_owners(self, sdfg: SDFG) -> Optional[Dict[str, Set[int]]]:
        """Every name in ``sdfg`` mapped to the ids of the nodes that may mention it, or ``None`` if some
        node cannot be analysed. Built once per SDFG."""
        key = id(sdfg)
        if key in self._name_owners:
            return self._name_owners[key]
        mentions = self.name_mentions(sdfg)
        owners: Optional[Dict[str, Set[int]]] = None
        if mentions is not None:
            owners = {}
            for names, owner_ids in mentions:
                for used in names:
                    owners.setdefault(used, set()).update(owner_ids)
        self._name_owners[key] = owners
        return owners

    def node_name_references(self, graph_node: nodes.Node, sdfg: SDFG) -> Optional[Set[str]]:
        """Every name the C++ lowered from ``graph_node`` may reference, or ``None`` when that cannot be
        decided. Code properties are read generically, so unknown node classes are still covered."""
        key = id(graph_node)
        if key in self._node_references:
            return self._node_references[key]
        if isinstance(graph_node, nodes.AccessNode):
            references: Optional[Set[str]] = {graph_node.data, graph_node.root_data}
        elif isinstance(graph_node, nodes.NestedSDFG):
            # Emitted inline, so names it does not define resolve to the enclosing scope.
            nested = self.nested_free_names(graph_node.sdfg) if graph_node.sdfg is not None else None
            references = None if nested is None else set(
                graph_node.free_symbols) | self.code_property_names(graph_node) | nested
        elif isinstance(graph_node, nodes.Tasklet):
            references = set(
                graph_node.free_symbols) | self.code_property_names(graph_node) | self._used_identifiers(graph_node)
        elif isinstance(graph_node, nodes.CodeNode):
            # A library node lowers through its own ``generate_code``, which this file cannot read.
            references = None
        else:
            # Scope nodes carry only property symbols (map ranges etc.), reported by ``free_symbols``.
            references = set(graph_node.free_symbols)
        self._node_references[key] = references
        return references

    def nested_free_names(self, nested: SDFG) -> Optional[Set[str]]:
        """Names used inside ``nested`` that it does not define itself, or ``None`` if a node cannot be
        analysed."""
        key = id(nested)
        if key in self._nested_free_names:
            return self._nested_free_names[key]
        mentions = self.name_mentions(nested)
        result: Optional[Set[str]] = None
        if mentions is not None:
            names: Set[str] = set()
            for mentioned, _ in mentions:
                names |= mentioned
            result = names - (nested.arrays.keys() | nested.symbols.keys() | nested.constants_prop.keys())
        self._nested_free_names[key] = result
        return result

    def code_property_names(self, holder) -> Set[str]:
        """Identifier tokens of every ``CodeBlock`` property of a node or control-flow block."""
        names: Set[str] = set()
        for _, value in holder.properties():
            names |= identifiers_in(code_blocks_of(value))
        return names

    def register_const_binding(self, decl: str, plain: str, fused: str) -> None:
        """Register the ``const T x = expr;`` binding of a write-once transient: ``plain`` is the bare write,
        ``fused`` the write carrying the binding, ``decl`` the standalone declaration.
        :meth:`emit_tasklet_body_block` picks: fused when the tasklet is brace-free, else ``decl`` ahead of
        the block."""
        self.const_pending.append({'decl': decl, 'plain': plain, 'fused': fused})

    def fuse_pending_decl(self, tasklet, line: str) -> str:
        """Fold a pending declaration into ``line``, the brace-free statement of ``tasklet`` (``x = expr;``
        becomes ``T x = expr;``). Unchanged unless the line is the exact registered const write, or
        ``scalar_init_style`` is ``fused`` and the text itself spells the scalar's first write."""
        for info in self.const_pending:
            if line != info['plain']:
                continue
            self.const_pending.remove(info)
            return info['fused']
        for ptrname, info in list(self._late_pending.items()):
            if not info['fusable'] or id(tasklet) not in info['writers']:
                continue
            if not re.match(r'%s\s*=[^=]' % re.escape(ptrname), line):
                continue
            del self._late_pending[ptrname]
            return '%s %s' % (info['ctype'], line)
        return line

    def emit_pending_late_decls(self, cfg, state_id, tasklet, callsite_stream) -> None:
        """Emit before ``tasklet`` the deferred ``T x;`` of every scalar it uses first, and any unfused
        const binding's declaration."""
        # An unfused const binding belongs to a braced tasklet; declare it (mutable) at the enclosing scope.
        for info in self.const_pending:
            callsite_stream.write(info['decl'], cfg, state_id, tasklet)
        self.const_pending.clear()
        if not self._late_pending:
            return
        for ptrname, info in list(self._late_pending.items()):
            if id(tasklet) not in info['tasklets']:
                continue
            zero = ' = 0' if info['setzero'] else ''
            callsite_stream.write('%s %s%s;' % (info['ctype'], info['ptrname'], zero), cfg, state_id, tasklet)
            del self._late_pending[ptrname]

    def const_binding_scope_is_local(self, sdfg, name: str) -> bool:
        """Whether the allocation planner declares ``name`` in the block its write is emitted in. A
        ``Scope`` transient touched in several states (as ``SplitStateByGpuClass`` makes) is declared at
        function scope, where a fused binding would die before its readers."""
        scope = self.eager_allocation_scope(sdfg, name)
        return scope is not None and not isinstance(scope, SDFG)

    def _is_const_scalar(self, desc, name: Optional[str] = None, sdfg=None) -> bool:
        # A write-once scope-lifetime CPU value scalar, emitted as a fused ``const T x = expr;``.
        if name is not None and sdfg is not None and not self.const_binding_scope_is_local(sdfg, name):
            return False
        return (isinstance(desc, dt.Scalar) and desc.const_init and desc.lifetime == dtypes.AllocationLifetime.Scope
                and desc.storage in VALUE_SCALAR_STORAGES)

    def _is_const_len1_array(self, desc, name: Optional[str] = None, sdfg=None) -> bool:
        # A write-once single-element Register array, emitted as ``const T x[1] = {expr};``.
        if name is not None and sdfg is not None and not self.const_binding_scope_is_local(sdfg, name):
            return False
        return (isinstance(desc, dt.Array) and not isinstance(desc, dt.View) and desc.const_init
                and desc.lifetime == dtypes.AllocationLifetime.Scope and desc.storage == dtypes.StorageType.Register
                and len(desc.shape) >= 1 and all(d == 1 for d in desc.shape))

    def array_index_access(self, sdfg, desc, data_name: str):
        """Register the ``<array>_idx`` helper of ``data_name`` and return ``(ptrname, fnname, ndim,
        extra_syms)``, or None for a by-value scalar."""
        ptrname = self.ptr(data_name, desc, sdfg)
        if self._is_value_scalar(ptrname, desc):
            return None
        ndim = len(desc.shape)
        fnname, extra_syms = self._register_index_function(data_name, desc)
        return (ptrname, fnname, ndim, extra_syms)

    def _is_value_scalar(self, ptrname: str, desc) -> bool:
        # Plain value (``x``) or pointer (``x[...]``), by the emitted DefinedType; an undeclared name falls
        # back to storage (a GPU-global Scalar is a device pointer).
        for registry in (self._dispatcher.defined_vars, self._dispatcher.declared_arrays):
            if registry.has(ptrname):
                defined_type, _ = registry.get(ptrname)
                return defined_type == DefinedType.Scalar
        return isinstance(desc, dt.Scalar) and desc.storage != dtypes.StorageType.GPU_Global

    def _register_index_function(self, data_name: str, desc):
        # Register the ``<name>_idx`` helper once per signature; return (name, extra symbol names).
        ndim = len(desc.shape)
        dim_syms = [symbolic.symbol('__d%d' % i) for i in range(ndim)]
        flatexpr = flat_offset(dim_syms, desc)
        extra = sorted((flatexpr.free_symbols - set(dim_syms)), key=lambda s: str(s))
        extra_names = [str(s) for s in extra]

        base = re.sub(r'\W', '_', data_name)
        sig = (ndim, tuple(str(s) for s in desc.strides), tuple(str(o) for o in desc.offset))
        key = (base, sig)
        if key in self._index_sig_to_name:
            return self._index_sig_to_name[key], extra_names

        fnname = base + '_idx'
        if fnname in self._index_functions:  # same name, different signature
            fnname = '%s_%dd_%d_idx' % (base, ndim, len(self._index_sig_to_name))
        self._index_sig_to_name[key] = fnname

        ctype = index_ctype()
        parameters = [str(d) for d in dim_syms] + list(extra_names)
        self._index_functions[fnname] = format_index_helper(index_function_qualifier(), ctype, fnname, parameters,
                                                            sym2cpp(flatexpr))
        return fnname, extra_names

    def _register_size_function(self, data_name: str, desc) -> Optional[Tuple[str, List[str]]]:
        # Register the ``<array>_size`` helper of ``desc.total_size``; return (name, call args), or None
        # for a bare symbol (no readability win) or a data-dependent size, whose container would not be in
        # scope in a free-standing helper (spmv's ``A_indptr[i + 1] - A_indptr[i]``).
        total = symbolic.pystr_to_symbolic(str(desc.total_size))
        if total.is_Symbol:
            return None
        if symbolic.arrays(total):
            return None
        free = sorted(total.free_symbols, key=lambda s: str(s))
        call_args = [str(s) for s in free]
        is_constant = len(free) == 0

        base = re.sub(r'\W', '_', data_name)
        sig = (str(total), tuple(call_args))
        key = (base, sig)
        if key in self._size_sig_to_name:
            return self._size_sig_to_name[key], call_args

        fnname = base + '_size'
        if fnname in self._size_functions:  # same name, different signature
            fnname = '%s_%d_size' % (base, len(self._size_sig_to_name))
        self._size_sig_to_name[key] = fnname

        self._size_functions[fnname] = format_index_helper(size_qualifier(is_constant), index_ctype(), fnname,
                                                           list(call_args), sym2cpp(total))
        return fnname, call_args


class NestedMultiDimSubscriptLowerer(ast.NodeTransformer):
    """Rewrites a nested multi-dim subscript (``idx[i, j]``) inside an index expression to its
    ``<array>_idx`` text. Left as a ``Tuple`` slice it would lower to ``std::make_tuple``; a rank-1 nested
    subscript already round-trips and is left alone."""

    def __init__(self, remover: 'ReadableKeywordRemover'):
        self.remover = remover

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.slice, ast.Tuple) and self.remover._is_bare_data(rname(node)):
            access = self.remover._bare_access(node)
            if access is not None:
                return ast.copy_location(ast.Name(id=access), node)
        return self.generic_visit(node)


class ReadableKeywordRemover(cpp.DaCeKeywordRemover):
    """The classic keyword remover that also lowers direct array accesses (``A[i, j]`` for ``A`` in
    ``sdfg.arrays``, left by InlineTaskletConnectors) to ``A[A_idx(i, j, ...)]``."""

    def __init__(self, sdfg, memlets, constants, codegen, defined_symbols=None):
        super().__init__(sdfg, memlets, constants, codegen, defined_symbols)
        #: Operand text -> dtype for the statements rendered here, so the C++ printer can type a bare
        #: literal against a ``dace::float16`` operand. Inlined connectors are keyed by their access text.
        self.operand_dtypes: Dict[str, dtypes.typeclass] = {
            conn: entry[3]
            for conn, entry in memlets.items() if conn is not None
        }

    def _is_bare_data(self, name: str) -> bool:
        return name not in self.memlets and name not in self.constants and name in self.sdfg.arrays

    def _scalar_constant_name(self, name: str) -> Optional[str]:
        # A 0-d SDFG constant (a MarkConstInit-promoted scalar) is a bare ``constexpr T name``; the classic
        # ``_subscript_expr`` would fail on its empty stride list.
        if name in self.constants and numpy.ndim(self.constants[name]) == 0:
            return name
        return None

    def _bare_access(self, node: ast.AST) -> Optional[str]:
        # C++ access for a direct (inlined) access to an SDFG array.
        name = rname(node)
        # A walked access is ``(*__walk_X)``; checked first so a fully walked array registers no helper.
        if isinstance(node, ast.Subscript):
            walk = self.codegen.current_walk_accesses()
            if walk is not None:
                pointer = walk.get((name, tuple(self._index_list(node.slice))))
                if pointer is not None:
                    return '(*%s)' % pointer
        desc = self.sdfg.arrays[name]
        info = self.codegen.array_index_access(self.sdfg, desc, name)
        if info is None:
            return self.codegen.ptr(name, desc, self.sdfg)
        ptrname, fnname, ndim, extra_syms = info
        if not isinstance(node, ast.Subscript):
            return None
        indices = self._index_list(node.slice)
        if len(indices) != ndim:
            return None
        return format_index_access(ptrname, fnname, indices, extra_syms)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        target_node = node.targets[-1]
        target = rname(target_node)
        if not self._is_bare_data(target):
            return super().visit_Assign(node)
        # A bare-data target is never WCR. One statement, so the unparser does not declare the LHS ``auto``.
        value = self.visit(astutils.copy_tree(node.value))
        lhs = self._bare_access(target_node)
        if lhs is None:
            return self.generic_visit(node)
        rhs = cppunparse.cppunparse(value,
                                    expr_semicolon=False,
                                    defined_symbols={
                                        **self.defined_symbols,
                                        **self.operand_dtypes
                                    },
                                    data_names=self.operand_dtypes)
        desc = self.sdfg.arrays[target]
        plain = '%s = %s;' % (lhs, rhs)
        if self.codegen._is_const_scalar(desc, target, self.sdfg):
            # allocate_array skipped ``T x;``; this write carries it (see register_const_binding).
            ctype = desc.dtype.ctype
            self.codegen.register_const_binding('%s %s;' % (ctype, lhs), plain, 'const %s %s = %s;' % (ctype, lhs, rhs))
        elif self.codegen._is_const_len1_array(desc, target, self.sdfg):
            # ``const T x[1] = {(T)(expr)};``: the cast keeps legacy's silent narrowing, which a braced
            # initializer would report as -Wnarrowing.
            name = self.codegen.ptr(target, desc, self.sdfg)
            ctype = desc.dtype.ctype
            self.codegen.register_const_binding('%s %s[1];' % (ctype, name), plain,
                                                'const %s %s[1] = {(%s)(%s)};' % (ctype, name, ctype, rhs))
        return self._replace_assignment(ast.Name(id=plain), node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        target = rname(node)
        bare_const = self._scalar_constant_name(target)
        if bare_const is not None:
            return ast.copy_location(ast.Name(id=bare_const), node)
        if not self._is_bare_data(target):
            return super().visit_Subscript(node)
        access = self._bare_access(node)
        if access is None:
            return self.generic_visit(node)
        self.operand_dtypes[access] = self.sdfg.arrays[target].dtype
        return ast.copy_location(ast.Name(id=access), node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        name = rname(node)
        if self._is_bare_data(name):
            desc = self.sdfg.arrays[name]
            ptrname = self.codegen.ptr(name, desc, self.sdfg)
            if self.codegen._is_value_scalar(ptrname, desc):
                self.operand_dtypes[ptrname] = desc.dtype
                return ast.copy_location(ast.Name(id=ptrname), node)
        return super().visit_Name(node)

    def _index_list(self, slicenode: ast.AST) -> List[str]:
        lowerer = NestedMultiDimSubscriptLowerer(self)
        return [ast.unparse(lowerer.visit(astutils.copy_tree(e))) for e in index_expr_nodes(slicenode)]
