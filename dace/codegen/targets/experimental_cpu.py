# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Experimental "readable" CPU code generator (``compiler.cpu.implementation = experimental_readable``).

Subclasses :class:`~dace.codegen.targets.cpu.CPUCodeGen` and changes only how tasklets and array
accesses are emitted: array accesses go through a generated ``<array>_idx(...)`` index function
(the offset arithmetic appears once per array); tasklets whose connectors were inlined by
``InlineTaskletConnectors`` access arrays directly (no copy-in/out temporaries); and write-once data
marked by ``MarkConstInit`` is emitted as ``const``/``constexpr``. Both passes run before codegen in
``dace.codegen.codegen.generate_code``. The GPU generator emits device tasklets through the shared
CPU instance, so these changes also apply inside ``__global__`` kernels.
"""
import ast
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy
from pygments.lexers import CppLexer
from pygments.token import Token

from dace import data as dt
from dace import dtypes, symbolic
from dace.codegen import cppunparse
from dace.codegen.common import sym2cpp
from dace.config import Config
from dace.codegen.dispatcher import DefinedType
from dace.codegen.targets import cpp
from dace.codegen.targets.cpu import CPUCodeGen, hoist_loop_decls, map_schedule_is_sequential
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.sdfg import nodes
from dace.sdfg.utils import dynamic_map_inputs

#: C++ integer type for computed flat indices, per ``codegen_params.index_ctype``. ``int`` (not
#: ``int32_t``) and ``long long`` (not ``int64_t``) are spelled out deliberately: ``int`` is exactly
#: what the generated loops declare (``for (auto i = 0; ...)``), so an int32 helper takes its indices
#: with no conversion at all, and ``long long`` preserves the historical signature byte-for-byte.
INDEX_CTYPES = {'int64': 'long long', 'int32': 'int'}
# Qualifier for the generated ``<array>_idx`` and symbolic ``<array>_size`` helpers: host+device
# callable, inlined, usable in constant expressions.
INDEX_FUNCTION_QUALIFIER = 'static DACE_HDFI constexpr'
# Qualifier for a CONSTANT ``<array>_size`` helper: ``consteval`` forces the fixed extent to fold at
# compile time (a C++20 keyword, so size_qualifier falls back to ``constexpr`` before C++20).
SIZE_CONSTEVAL_QUALIFIER = 'static DACE_HDFI consteval'


def index_function_qualifier() -> str:
    """Qualifier for the generated ``<array>_idx`` / ``<array>_size`` helpers, per
    ``compiler.cpu.codegen_params.index_fn_qualifier``. ``inline_constexpr`` is the default
    (``static DACE_HDFI constexpr`` -- the backend inlines these tiny functions at -O2); ``always_inline``
    additionally forces inlining, which only matters where a helper is otherwise left out of line and
    an un-inlined access would block vectorization."""
    if Config.get('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier') == 'always_inline':
        return 'static __attribute__((always_inline)) inline constexpr'
    return INDEX_FUNCTION_QUALIFIER


def index_ctype() -> str:
    """The C++ integer type the ``<array>_idx`` / ``<array>_size`` helpers compute in, per
    ``compiler.cpu.codegen_params.index_ctype``.

    ``int32`` is UNSAFE past 2**31 ELEMENTS and is never selected automatically -- the bound is on the
    element count, not the byte size, so an int8 array overflows at only 2 GiB (a float64 one at 16
    GiB). The generator cannot prove an SDFG stays under that for symbolic shapes, so this stays an
    opt-in knob whose default (``int64``) reproduces the historical signature exactly."""
    return INDEX_CTYPES[Config.get('compiler', 'cpu', 'codegen_params', 'index_ctype')]


def size_qualifier(is_constant: bool) -> str:
    """Qualifier for an ``<array>_size`` helper: ``consteval`` for a constant extent under C++20+
    (folds it at compile time), else ``constexpr`` (the same qualifier as the index functions)."""
    if not is_constant:
        return INDEX_FUNCTION_QUALIFIER
    standard = int(str(Config.get('compiler', 'cpp_standard')).strip())
    return SIZE_CONSTEVAL_QUALIFIER if standard >= 20 else INDEX_FUNCTION_QUALIFIER


def format_index_access(ptrname: str, fnname: str, indices: List[str], extra: List[str]) -> str:
    """C++ ``ptr[fn(idx.., extra..)]`` access through a registered ``<array>_idx`` index function."""
    call_args = [sym2cpp(symbolic.pystr_to_symbolic(ix)) for ix in indices] + list(extra)
    return '%s[%s(%s)]' % (ptrname, fnname, ', '.join(call_args))


def loop_access_form() -> str:
    """How arrays indexed at a sequential loop counter are accessed, per
    ``compiler.cpu.codegen_params.loop_access_form``: ``indexed`` (the default, recompute the flat
    index each iteration) or ``ptr_increment`` (walk a base pointer). ``indexed`` reproduces today's
    output byte-for-byte; ``ptr_increment`` only ever applies where a walk is provably equivalent (see
    :meth:`ExperimentalCPUCodeGen.build_walk_plan`), otherwise it falls back to ``indexed``."""
    return Config.get('compiler', 'cpu', 'codegen_params', 'loop_access_form')


def index_expr_nodes(slicenode: ast.AST) -> List[ast.AST]:
    """The per-dimension index AST expressions of a subscript's slice (a tuple's elements, else the
    single expression)."""
    node = slicenode
    if isinstance(node, ast.Index):  # py<3.9 compatibility
        node = node.value
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def subscript_index_strings(slicenode: ast.AST) -> List[str]:
    """The per-dimension index expressions of a subscript's slice as source strings. Shared by the
    walk-plan builder and the readable rewriter so both key an access on the identical strings."""
    return [ast.unparse(e) for e in index_expr_nodes(slicenode)]


# NOTE: not registered with the target registry -- it is selected explicitly by
# ``dace.codegen.codegen.generate_code`` when ``compiler.cpu.implementation`` is
# ``experimental_readable``. Registering it would cause double instantiation.
class ExperimentalCPUCodeGen(CPUCodeGen):
    """ Human-readable CPU/GPU-kernel code generator (see module docstring). """

    def __init__(self, frame, sdfg):
        super().__init__(frame, sdfg)
        # Helper name -> full C++ definition (deduplicated), for the ``<array>_idx`` index
        # functions and the ``<array>_size`` allocation-extent helpers respectively.
        self._index_functions: Dict[str, str] = {}
        self._size_functions: Dict[str, str] = {}
        # id(function_stream) -> helper names already flushed to THAT stream. Dedup is per stream
        # (= per output file), so an array used on both the host (.cpp) and a device kernel (.cu)
        # gets its definition in each file. Index and size helpers share the set (names never
        # collide: ``A_idx`` vs ``A_size``).
        self._emitted_functions: Dict[Union[int, str], Set[str]] = {}
        # (base-name, signature) -> emitted function name, so one shape reuses a function while
        # two same-named arrays of different shape (e.g. a connector-derived ``_out`` that is 1-D
        # in one inlined SDFG and 2-D in another) get distinct helpers, not a colliding name with
        # wrong arity / strides. A plain name collision is disambiguated against _index_functions /
        # _size_functions (the emitted-name sets), so no separate name->sig maps are kept.
        self._index_sig_to_name: Dict[tuple, str] = {}
        self._size_sig_to_name: Dict[tuple, str] = {}
        # Per-tasklet cache of identifiers appearing in the (rewritten) body.
        self._body_identifiers: Dict[int, Set[str]] = {}
        # Per-native-tasklet cache: id(node) -> {connector_name: cpp_access}. An entry means the
        # connector is accessed directly (scalar index / base pointer) instead of via a copy-in/out.
        self._cpp_inline: Dict[int, Dict[str, str]] = {}
        # ptr_increment (loop_access_form) support. ``_map_scope_stack`` is the chain of currently-open
        # MapEntry nodes (pushed before the base MapEntry emitter runs, popped after the matching
        # MapExit), so the scope-preamble / -postamble hooks and the readable rewriter can find the map
        # whose body they are inside. ``_walk_plans`` memoizes id(map) -> walk plan ({} when a pointer
        # walk is not provably equivalent). The two ``_walk_emitted_*`` sets guard the once-per-map
        # emission of the pointer declarations and increments.
        self._map_scope_stack: List[nodes.MapEntry] = []
        self._walk_plans: Dict[int, dict] = {}
        self._walk_emitted_decls: Set[int] = set()
        self._walk_emitted_incs: Set[int] = set()

    # -- map scope ------------------------------------------------------------

    def map_scope_needs_brace(self, sdfg, state_dfg, node: nodes.MapEntry) -> bool:
        """Drop the map's encapsulating C scope when it would bound nothing.

        The base generator always emits ``{ ... }`` around a map. With connector-free tasklets
        there is usually nothing declared into it -- the loops already scope their bodies and the
        map's scope-lifetime transients are allocated inside the innermost loop -- so the braces
        only add nesting. Keep the scope for every construct that does declare into it.
        """
        if dynamic_map_inputs(state_dfg, node):  # emit memlet_definition declarations
            return True
        if hoist_loop_decls(node):  # declares the induction variables ahead of the loop headers
            return True
        if self.walk_plan_for(sdfg, state_dfg, node):  # declares the walking base pointers ahead of the loop
            return True
        if node.map.schedule == dtypes.ScheduleType.CPU_Persistent:  # declares the thread id
            return True
        if node.map.instrument != dtypes.InstrumentationType.No_Instrumentation:  # declares timers
            return True
        # Complex-type OpenMP tree reductions prepend a ``#pragma omp declare reduction`` bound by
        # this scope; keep the brace so the directive does not leak to the enclosing scope and clash
        # across sibling maps. (Real-type reductions need only a ``reduction(op:var)`` clause on the
        # pragma, which is self-contained, so they do not force the brace.)
        if (node.map.schedule == dtypes.ScheduleType.CPU_Multicore
                and Config.get_bool('compiler', 'emit_tree_reductions')
                and any(declare is not None
                        for _op, _ct, _dname, declare in self._collect_omp_reductions(sdfg, state_dfg, node))):
            return True
        return False

    # -- ptr_increment: walking base pointers for a sequential map -------------

    def _generate_MapEntry(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        # Compute (and memoize) the walk plan BEFORE the base emitter runs, so ``map_scope_needs_brace``
        # -- which the base calls -- sees it, and push this map so the scope hooks below can find it.
        self.walk_plan_for(sdfg, cfg.state(state_id), node)
        self._map_scope_stack.append(node)
        super()._generate_MapEntry(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def _generate_MapExit(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        super()._generate_MapExit(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        # Pop the matching entry pushed in _generate_MapEntry and drop its plan (ids are reused once a
        # map is freed, so do not let a stale plan linger).
        if self._map_scope_stack:
            entry = self._map_scope_stack.pop()
            self._walk_plans.pop(id(entry.map), None)
            self._walk_emitted_decls.discard(id(entry.map))
            self._walk_emitted_incs.discard(id(entry.map))

    def generate_scope_preamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        super().generate_scope_preamble(sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream)
        # ``outer_stream`` here is the code position just after the map's encapsulating brace and before
        # its loop headers: exactly where the walking base pointers must be declared.
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
        # ``inner_stream`` here is the end of the loop body (before the closing brace): where each
        # walking pointer is advanced by its per-iteration stride.
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
        """The (non-empty) walk plan of the map currently being emitted, or None."""
        if not self._map_scope_stack:
            return None
        plan = self._walk_plans.get(id(self._map_scope_stack[-1].map))
        return plan or None

    def current_walk_accesses(self) -> Optional[Dict[tuple, str]]:
        """``{(array_name, index_string_tuple): pointer_name}`` for the map currently being emitted, or
        None. The readable rewriter consults this to replace a walked access with ``(*pointer)``."""
        plan = self._current_walk_plan()
        return plan['accesses'] if plan else None

    def walk_plan_for(self, sdfg, state_dfg, node: nodes.MapEntry) -> dict:
        """Memoized :meth:`build_walk_plan` for map ``node`` (keyed on the map object's id)."""
        cached = self._walk_plans.get(id(node.map))
        if cached is not None:
            return cached
        plan = self.build_walk_plan(sdfg, state_dfg, node)
        self._walk_plans[id(node.map)] = plan
        return plan

    def build_walk_plan(self, sdfg, state_dfg, node: nodes.MapEntry) -> dict:
        """A plan to walk this map's array accesses with incrementing base pointers, or ``{}`` to keep
        the indexed form. A plan is built ONLY where the walk is provably equivalent to the indexed
        access -- otherwise (any shape that cannot be walked with a single simple pointer) it returns
        ``{}`` and the construct stays indexed. See the ``loop_access_form`` schema entry.

        The plan is ``{'accesses': {(name, idx_tuple): pointer}, 'decls': [str], 'incs': [str]}``.
        """
        if loop_access_form() != 'ptr_increment':
            return {}
        # SEQUENTIAL only (reuse the hoisted-declaration gate): an OpenMP map keeps the canonical
        # indexed loop its parallel-for pragma requires.
        if not map_schedule_is_sequential(node):
            return {}
        if node.map.unroll:
            return {}
        # A single 1-D loop: the per-iteration pointer stride is defined against one counter.
        if len(node.map.range) != 1 or len(node.map.params) != 1:
            return {}
        if dynamic_map_inputs(state_dfg, node):
            return {}
        # Exactly one Python tasklet in the scope (no nested maps, extra tasklets, or scope transients).
        scope_nodes = list(state_dfg.scope_subgraph(node, include_entry=False, include_exit=False).nodes())
        if len(scope_nodes) != 1 or not isinstance(scope_nodes[0], nodes.Tasklet):
            return {}
        tasklet = scope_nodes[0]
        if tasklet.language != dtypes.Language.Python:
            return {}
        # A WCR write must go through the atomic resolve path, never a plain ``*p =``.
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

        # Collect every array subscript; bail (whole construct falls back) on any array reference that
        # is not a plain affine subscript in the loop counter -- a bare (unsubscripted) array, a
        # gather / call inside an index, or a subscript whose arity does not match the descriptor.
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
                continue  # an identical access shares its cursor
            flat = self._flat_index(idx_tuple, desc)
            per_iter = symbolic.pystr_to_symbolic(flat.subs(loop_sym, loop_sym + skip_sym) - flat).simplify()
            if loop_sym in per_iter.free_symbols:
                return {}  # non-affine in the loop counter -> not a simple walk
            ptrname = self.ptr(name, desc, sdfg)
            try:
                defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
            except KeyError:
                return {}  # base pointer not yet in scope (e.g. a scope transient) -> keep indexed
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

    def _flat_index(self, idx_tuple, desc):
        """The flat element offset (index . strides + offset . strides) of a subscript, as a symbolic
        expression -- exactly what the ``<array>_idx`` helper computes, but as a Python expression so
        the base offset and per-iteration stride fall out by substitution."""
        strides = [symbolic.pystr_to_symbolic(str(s)) for s in desc.strides]
        offset = [symbolic.pystr_to_symbolic(str(o)) for o in desc.offset]
        flat = sum(symbolic.pystr_to_symbolic(idx_tuple[i]) * strides[i] for i in range(len(strides)))
        flat += sum(offset[i] * strides[i] for i in range(len(strides)))
        return symbolic.pystr_to_symbolic(flat)

    def _scan_walkable(self, astnode, sdfg, records: List[tuple]) -> bool:
        """Walk ``astnode`` recording every plain-array subscript into ``records``; return False the
        moment an array is referenced in a way a simple pointer walk cannot express."""
        if isinstance(astnode, ast.Subscript):
            base = rname(astnode)
            desc = sdfg.arrays.get(base)
            if isinstance(desc, dt.Array) and not isinstance(desc, dt.View):
                # A plain-array subscript: its index expressions must be affine (no nested subscript /
                # call = no gather / indirection), and match the descriptor's rank.
                for idx in index_expr_nodes(astnode.slice):
                    if any(isinstance(sub, (ast.Subscript, ast.Call)) for sub in ast.walk(idx)):
                        return False
                idx_tuple = tuple(subscript_index_strings(astnode.slice))
                if len(idx_tuple) != len(desc.shape):
                    return False
                records.append((base, idx_tuple, desc))
                return True  # indices are plain symbols; nothing further to walk inside
            # Non-array subscript (a connector / constant): scan its children normally.
        elif isinstance(astnode, ast.Name):
            desc = sdfg.arrays.get(astnode.id)
            if isinstance(desc, dt.Array) and not isinstance(desc, dt.View):
                return False  # a bare (unsubscripted) array reference cannot be a single walked element
            return True
        return all(self._scan_walkable(child, sdfg, records) for child in ast.iter_child_nodes(astnode))

    def _unique_walk_name(self, data_name: str, used: Set[str]) -> str:
        """A collision-free C++ name for a walking base pointer over ``data_name``."""
        base = '__walk_' + re.sub(r'\W', '_', data_name)
        name = base
        k = 1
        while name in used:
            name = '%s_%d' % (base, k)
            k += 1
        used.add(name)
        return name

    # -- tasklet lowering hooks ------------------------------------------------

    def make_keyword_remover(self, sdfg, memlets):
        return ReadableKeywordRemover(sdfg, memlets, sdfg.constants, self)

    def _connector_needs_copy(self, node, conn) -> bool:
        # Only tasklets are rewritten. NestedSDFGs and other code nodes always
        # keep their connectors (function arguments).
        if not isinstance(node, nodes.Tasklet):
            return True
        # CPP (C++/library) tasklets: a connector needs a copy only if it was NOT
        # inlined into the body as a direct access / base pointer. The inline map
        # is computed in _generate_Tasklet before the base generator runs.
        if node.language == dtypes.Language.CPP:
            return conn not in self._cpp_inline.get(id(node), {})
        # Other non-Python tasklets (MLIR, OpenCL, ...) are emitted verbatim and
        # keep their classic connector copy-in/out -- only CPP bodies are rewritten.
        if node.language != dtypes.Language.Python:
            return True
        # Python tasklets: a connector needs a copy-in/out only if the (rewritten)
        # body still refers to it. InlineTaskletConnectors rewrites inlined
        # connectors out of the body, so their names no longer appear.
        return conn in self._used_identifiers(node)

    def _used_identifiers(self, node) -> Set[str]:
        key = id(node)
        cached = self._body_identifiers.get(key)
        if cached is not None:
            return cached
        code = node.code.as_string if node.code else ''
        if node.language == dtypes.Language.Python:
            # ``as_string`` unparses the tasklet's already-parsed AST, so it is always valid Python.
            ids = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
        else:
            ids = set(re.findall(r'[A-Za-z_]\w*', code))
        self._body_identifiers[key] = ids
        return ids

    def _generate_Tasklet(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen=None):
        # For CPP (C++/library) tasklets -- the only non-Python bodies
        # rewrite_cpp_tasklet_body rewrites -- compute which connectors can be
        # inlined (direct access / base pointer) BEFORE lowering, so the
        # copy-in/out logic in the base generator -- which calls
        # _connector_needs_copy -- skips those connectors.
        if isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.CPP:
            state_dfg = cfg.nodes()[state_id]
            self._cpp_inline[id(node)] = self._compute_cpp_inline(sdfg, state_dfg, node)
        super()._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen)
        # Flush any index / size helpers registered while lowering this tasklet body.
        self._flush_generated_functions(function_stream, cfg, state_id, node)

    # -- readable tasklet body: single line, no separator noise ----------------

    def tasklet_body_comment(self, node) -> str:
        # No ``// Tasklet code (label)`` banner: each emitted tasklet line already
        # carries a trailing ``// <label>`` (see emit_tasklet_body_block).
        return ''

    def tasklet_body_open_marker(self, node) -> str:
        # No visual separators; the body reads directly as C++.
        return ''

    def tasklet_body_close_marker(self, node) -> str:
        return ''

    def emit_tasklet_body_block(self, callsite_stream, cfg, state_id, node, inner_body, postamble, has_locals) -> None:
        # A connector-free, single-statement tasklet collapses onto one brace-free
        # line: ``C[C_idx(i, j)] = A[A_idx(i, j)] + B[B_idx(i, j)];  // <label>``.
        # Anything with copy-in/out temporaries, code->code locals, or a multi-
        # statement / native body keeps its own ``{ ... }`` scope (a bare local
        # declaration would otherwise leak into the enclosing map body).
        if not has_locals and self._single_statement_body(node):
            # unparse_tasklet already baked a ``////__DACE:...`` provenance tag into
            # the statement line; strip it so the ``// <label>`` comment lands
            # *before* the tag (clean_code removes from the first tag to end-of-line,
            # and the stream re-appends a fresh tag after the whole line). The
            # trailing newline is required, else the ``//`` comment would swallow
            # whatever token the caller writes next (e.g. the map's closing brace).
            line = re.sub(r'[ \t]*////__(DACE:|CODEGEN;).*', '', inner_body).strip()
            if line:
                callsite_stream.write('%s  // %s\n' % (line, node.label), cfg, state_id, node)
                return
        callsite_stream.write('{  // %s' % node.label, cfg, state_id, node)
        callsite_stream.write(inner_body, cfg, state_id, node)
        callsite_stream.write(postamble)
        callsite_stream.write('}', cfg, state_id, node)

    def _single_statement_body(self, node) -> bool:
        """True if ``node`` is a Python tasklet whose body is exactly one assignment (so, once its
        connectors are inlined, it is a single array-store statement with no local to keep in scope)."""
        if not isinstance(node, nodes.Tasklet) or node.language != dtypes.Language.Python:
            return False
        stmts = node.code.code  # a Python CodeBlock always holds a parsed statement list
        if isinstance(stmts, str):
            stmts = ast.parse(stmts).body
        return len(stmts) == 1 and isinstance(stmts[0], (ast.Assign, ast.AugAssign))

    # -- native (C++/library) tasklet connector inlining -----------------------

    def rewrite_cpp_tasklet_body(self, node, sdfg, state_dfg) -> str:
        """
        Returns the C++ body of a native tasklet with every inlinable connector
        replaced by a direct array / base-pointer access. The body is tokenized
        with the pygments C++ lexer so that only identifier tokens are rewritten;
        connector names appearing inside string / char literals or comments are
        left untouched (they are not ``Token.Name`` tokens).
        """
        body = node.code.as_string
        # _generate_Tasklet always fills _cpp_inline for a CPP tasklet before the base generator
        # reaches this hook, so read it (empty map -> nothing to inline, emit the body verbatim).
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
        """
        For a native (C++/library) tasklet, returns ``{connector_name: cpp_access}``
        for every connector that can be safely inlined -- a scalar connector as a
        direct ``<array>_idx(...)`` access, a pointer / whole-subset connector as a
        base-pointer expression. Connectors that must keep the classic connector
        copy (WCR, stream/view/reference data, or a conflicting inout name) are
        omitted from the map.
        """
        # A body that is (or contains) a preprocessor-directive line cannot have its connectors
        # inlined. ``ExpandReduceOpenMP`` is exactly this shape: a
        # ``#pragma omp parallel for reduction(op:_out[0])`` loop whose body is
        # ``_out[0] = op(_out[0], _in[..])``. Two things break: (a) the pygments C++ lexer folds a
        # ``#pragma`` line into a single preprocessor token, so a connector named inside the clause is
        # NOT a rewritable ``Token.Name`` -- it survives verbatim and references an undeclared name;
        # and (b) a pointer-base inline (``&x`` for a scalar sink) textually substituted into the
        # existing ``_out[0]`` subscript mis-parses as ``&x[0]`` (== ``&(x[0])``, subscripting the
        # scalar). Keep the classic connector copy-in/out for the whole tasklet (as legacy does).
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

        # A connector name shared by an in- and an out-edge (inout) can be inlined
        # only if both sides resolve to the SAME access string; otherwise a single
        # identifier in the body cannot stand for two different things -> keep the
        # classic connector for both sides. (Mirrors InlineTaskletConnectors.)
        inout = set(node.in_connectors) & set(node.out_connectors)
        inline: Dict[str, str] = {}
        for name in set(in_map) | set(out_map):
            if name in inout:
                if name in in_map and name in out_map and in_map[name] == out_map[name]:
                    inline[name] = in_map[name]
                # else: keep the connector for both sides
            else:
                inline[name] = in_map.get(name, out_map.get(name))

        # Inlining an edge skips its copy dispatch (see _connector_needs_copy),
        # which is also what registers the edge's copy target as "used" -- and
        # thereby generates that target's file-level setup (e.g. the CUDA
        # context/stream code objects a host-side cudaMemcpy/cudaMemset needs).
        # Preserve that side effect so the codegen invariant `used_targets ==
        # statically-discovered targets` still holds and the setup is emitted.
        for name in inline:
            if name in in_edges:
                self._register_inlined_copy_target(sdfg, state_dfg, node, in_edges[name], is_output=False)
            if name in out_edges:
                self._register_inlined_copy_target(sdfg, state_dfg, node, out_edges[name], is_output=True)
        return inline

    def _register_inlined_copy_target(self, sdfg, state_dfg, node, edge, is_output: bool) -> None:
        """
        Marks the copy target of an inlined connector's ``edge`` as used, mirroring
        the ``used_targets`` side effect of the copy dispatch that inlining skips.
        """
        if is_output:
            src_node, dst_node = node, state_dfg.memlet_path(edge)[-1].dst
        else:
            src_node, dst_node = state_dfg.memlet_path(edge)[0].src, node
        target = self._dispatcher.get_copy_dispatcher(src_node, dst_node, edge, sdfg, state_dfg)
        if target is not None:
            self._dispatcher.used_targets.add(target)

    def _cpp_connector_access(self, sdfg, state_dfg, node, edge, is_output: bool) -> Optional[str]:
        """
        C++ access string for one native-tasklet connector, or None if it must
        keep the classic connector copy. Only ``memlet`` attributes are read
        (never mutated), honouring the deep-copy discipline.
        """
        conn = edge.src_conn if is_output else edge.dst_conn
        memlet = edge.data
        if not conn:
            return None
        if memlet.data is None or memlet.data not in sdfg.arrays:
            return None
        # Only inline accesses to a real AccessNode (data in memory), never a
        # tasklet<->tasklet (code->code) register connector.
        path = state_dfg.memlet_path(edge)
        far = path[-1].dst if is_output else path[0].src
        if not isinstance(far, nodes.AccessNode):
            return None
        desc = sdfg.arrays[memlet.data]
        # Only plain arrays / scalars in memory. Never streams, references (a
        # reference-set edge targets a Reference descriptor), nor container arrays
        # / structures whose element addressing is not the plain flat index
        # (mirrors InlineTaskletConnectors and MarkConstInit). An ArrayView IS a
        # plain-flat pointer into its source, so it routes through the same
        # <array>_idx path (``V[V_idx(...)]``, built from the view's own strides).
        if isinstance(desc, (dt.Stream, dt.Reference, dt.ContainerArray, dt.Structure)):
            return None
        if not isinstance(desc, (dt.Array, dt.Scalar)):
            return None
        # WCR outputs must go through the atomic write_and_resolve path.
        if memlet.wcr is not None:
            return None
        subset = memlet.subset
        if subset is None:
            return None
        conntype = node.out_connectors[conn] if is_output else node.in_connectors[conn]

        # Pointer / whole-subset connector (a library-call argument): the access is
        # the BASE POINTER to the subset start, not a single element.
        if isinstance(conntype, dtypes.pointer) or subset.num_elements() != 1:
            ptrname = self.ptr(memlet.data, desc, sdfg)
            try:
                defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
            except KeyError:
                defined_type = None
            if defined_type is not None:
                return cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self)
            # Fallback: base pointer + offset to the subset start.
            offset = cpp.cpp_offset_expr(desc, subset)
            return ptrname if offset == '0' else '%s + %s' % (ptrname, offset)

        # Scalar connector (single element): a direct indexed access through the
        # generated <array>_idx function -- mirrors ReadableKeywordRemover._bare_access.
        indices = [str(rb) for (rb, _re, _rs) in subset.ranges]
        info = self.array_index_access(sdfg, desc, memlet.data)
        if info is None:
            # A by-value scalar -> plain C++ name.
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
        # (constexpr_static data was promoted to an SDFG constant by MarkConstInit; framecode's
        # allocation planner already skips names in sdfg.constants_prop, so nothing to do here.)
        # A single-write ('const_runtime') scope-local value scalar is emitted as
        # `const T x = expr;` fused at its (single) write site (see
        # ReadableKeywordRemover.visit_Assign), so skip the mutable `T x;`
        # declaration -- but still register it so its reads resolve to a Scalar.
        if self._is_const_scalar(nodedesc):
            self._dispatcher.defined_vars.add(self.ptr(node.data, nodedesc, sdfg), DefinedType.Scalar,
                                              nodedesc.dtype.ctype)
            return
        # Same fusion for a single-element stack array -> `const T x[1] = {expr};`.
        if self._is_const_len1_array(nodedesc):
            self._dispatcher.defined_vars.add(self.ptr(node.data, nodedesc, sdfg), DefinedType.Pointer,
                                              dtypes.pointer(nodedesc.dtype).ctype)
            return
        super().allocate_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                               allocation_stream, allocate_nested_data)
        # heap_alloc_stmt (invoked inside super) may have registered an
        # <array>_size() helper for the allocation count; flush it to the function
        # stream, mirroring the _idx flush in _generate_Tasklet.
        self._flush_generated_functions(function_stream, cfg, state_id, node)

    def fused_heap_declarator(self, sdfg, name: str, nodedesc: dt.Data, arrsize, declared: bool, declaration_stream,
                              allocation_stream) -> Optional[str]:
        """Declarator fusing ``T *p;`` + ``p = new T[...];`` into one ``T* __restrict__ p = new
        T[...];`` definition, or None to keep the classic split pair.

        DaCe deliberately routes the declaration and the allocation to two streams so a transient's
        DECLARATION can be hoisted to an outer scope while its ALLOCATION stays in an inner one.
        Fusing is a purely textual merge of two writes, so it is sound only when both land in the
        same scope with nothing in between. All three of these must hold:

        * ``not declared`` -- otherwise ``declare_array`` already emitted ``T *p = nullptr;`` in an
          enclosing scope (a transient whose size depends on a non-free symbol) and registered it in
          ``declared_arrays``; a fused definition would declare a SECOND, inner ``p`` that shadows it,
          so every access outside this scope would see the still-null outer pointer.
        * ``declaration_stream is allocation_stream`` -- the dispatcher (``dispatch_allocate``) hands
          out two different streams for the Persistent and External lifetimes, where the pointer
          really lives in the state struct: the declaration goes to a throwaway stream and the
          allocation to ``__dace_init``. Fusing there would emit a local definition into
          ``__dace_init`` that shadows the state-struct member, so the program would read an
          unallocated member. For every other lifetime the dispatcher passes ONE stream for both
          (``declaration_stream = callsite_stream``) and the base writes the declaration immediately
          before the allocation, so merging them changes nothing but the text.
        * ``arrsize`` is a RUNTIME extent -- a compile-time constant one keeps the split form because
          GCC rejects the fused spelling. ``heap_alloc_stmt`` emits the element type carrying
          ``DACE_ALIGN(64)`` (== ``__attribute__((aligned(64)))``), which is load-bearing: it makes
          ``new`` call the over-aligned ``operator new[](size_t, align_val_t)``. With a constant
          bound the new-type-id in a DECLARATION names the fixed array type ``double[1]``, whose
          elements would each need 64-byte alignment at 8 bytes of size -- "error: alignment of array
          elements is greater than element size". The very same ``new`` expression is accepted as a
          bare assignment (the split form), and with a runtime bound no fixed array type is formed,
          so both of those stay legal. No fused spelling avoids this (``::new``, a cast, an aligned
          type alias and brace-init were all tried), and dropping ``DACE_ALIGN`` would silently
          de-align the allocation, so a constant-extent heap array stays split.

        Registration is untouched: the caller still runs ``define_var(...)`` after this, so
        ``defined_vars`` (and ``declared_arrays``, which only ``declare_array`` populates) resolve
        later accesses exactly as before.
        """
        if declared or declaration_stream is not allocation_stream:
            return None
        # The same test the base uses to route a variable-length Register array to the heap.
        if not symbolic.issymbolic(arrsize, sdfg.constants):
            return None
        return self.array_pointer_declarator(name, nodedesc)

    def array_pointer_declarator(self, name: str, nodedesc: dt.Data) -> str:
        """``T* __restrict__ name`` for a heap-allocated array pointer.

        ``__restrict__`` is dropped for a ``may_alias`` descriptor -- the same condition
        ``Array.as_arg`` and ``CPUCodeGen.generate_nsdfg_arguments`` already use to decide the
        qualifier on kernel/nested-SDFG arguments. ``may_alias`` marks an array that is deliberately
        reachable through a second pointer in the same function, so promising no-alias for it would
        be a miscompile rather than a readability win. It is also dropped when
        ``compiler.cpu.codegen_params.heap_ptr_restrict`` is ``none`` (the qualifier is a bimodal
        knob; ``restrict``, the default, is the faster bet but not universally so).
        """
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
        # Same aligned ``new[]`` as the base generator (paired with the base ``delete[]``), but
        # route the element count through a generated ``<array>_size(...)`` helper when worthwhile
        # (see _register_size_function) so the allocation extent reads as a named function; fall back
        # to the classic ``sym2cpp(total_size)`` string (``arrsize``) otherwise.
        count = arrsize
        if sdfg is not None and nodedesc is not None and data_name is not None:
            registered = self._register_size_function(data_name, nodedesc)
            if registered is not None:
                fnname, call_args = registered
                count = '%s(%s)' % (fnname, ', '.join(call_args))
        return '%s = new %s DACE_ALIGN(64)[%s];\n' % (alloc_name, ctype, count)

    def _flush_generated_functions(self, function_stream, cfg, state_id, node) -> None:
        # Emit each registered index / size helper once per OUTPUT FILE. A non-inline nested-SDFG
        # function has its own function stream but shares the host .cpp translation unit with every
        # other host stream, so keying the emitted-set on the individual stream re-emits an identical
        # ``<name>_idx`` definition per stream -> a C++ redefinition in that one TU. Key on the
        # output-file owner instead: during host codegen ``calling_codegen is self`` (one key for the
        # whole .cpp, so each helper is emitted exactly once and ``o.code`` is duplicate-free on its own
        # -- a consumer that writes the file itself no longer needs the ``deduplicate_functions``
        # post-pass). A device (.cu) file is generated through a delegating GPU codegen that sets
        # ``calling_codegen`` to itself, so its streams keep the per-stream emitted-set and their own
        # copy, exactly as before.
        #
        # ``_current_tu_key`` (not a bare ``id(self)``) names the host file being generated RIGHT NOW.
        # It equals ``id(self)`` -- the frame .cpp -- for the whole host build unless
        # ``split_nsdfg_translation_units`` is on, in which case ``_generate_NestedSDFG`` re-points it
        # at the nest's own buffer while that nest is generated. Without that, a helper already emitted
        # into the frame TU (say ``A_idx``) would be seen as emitted and SKIPPED in a split nest's TU
        # that also indexes ``A``, leaving that TU referencing an undefined ``A_idx``. Keying on
        # ``id(function_stream)`` instead is NOT the fix: many host streams feed the one frame TU, and
        # that is exactly the duplicate-definition bug the file-owner key exists to prevent.
        file_key = self._current_tu_key if self.calling_codegen is self else id(function_stream)
        emitted = self._emitted_functions.setdefault(file_key, set())
        for registry in (self._index_functions, self._size_functions):
            for name, defn in registry.items():
                if name in emitted:
                    continue
                function_stream.write(defn + '\n', cfg, state_id, node)
                emitted.add(name)

    # -- readable array indexing ----------------------------------------------

    def _is_const_scalar(self, desc) -> bool:
        """A single-write (``const_runtime``) scope-local scalar emitted as a fused
        ``const T x = expr;`` binding. Restricted to scope-lifetime CPU value scalars so the binding
        is declared in exactly the scope its reads live in; a device/persistent scalar stays classic."""
        return (isinstance(desc, dt.Scalar) and desc.const_init and desc.lifetime == dtypes.AllocationLifetime.Scope and
                desc.storage in (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap))

    def _is_const_len1_array(self, desc) -> bool:
        """A single-write (``const_runtime``) single-element STACK (Register) array emitted as a fused
        ``const T x[1] = {expr};`` binding. A heap or device single-element array stays classic."""
        return (isinstance(desc, dt.Array) and not isinstance(desc, dt.View) and desc.const_init
                and desc.lifetime == dtypes.AllocationLifetime.Scope and desc.storage == dtypes.StorageType.Register
                and len(desc.shape) >= 1 and all(d == 1 for d in desc.shape))

    def array_index_access(self, sdfg, desc, data_name: str):
        """Registers (once) the ``<array>_idx`` index function for ``data_name`` and returns
        ``(ptrname, fnname, ndim, extra_syms)``, or None if the access is a plain by-value scalar."""
        ptrname = self.ptr(data_name, desc, sdfg)
        if self._is_value_scalar(ptrname, desc):
            return None  # scalar is a plain C++ value; use the bare name
        ndim = len(desc.shape)
        fnname, extra_syms = self._register_index_function(data_name, desc)
        return (ptrname, fnname, ndim, extra_syms)

    def _is_value_scalar(self, ptrname: str, desc) -> bool:
        """Whether ``ptrname`` is a plain C++ value (``x``) rather than a pointer (``x[...]``), keyed
        on the emitted ``DefinedType`` (a heap/device/persistent Scalar is a pointer, a register one a
        value). Consults both registries; only a genuinely undeclared name falls back to storage."""
        for registry in (self._dispatcher.defined_vars, self._dispatcher.declared_arrays):
            if registry.has(ptrname):
                defined_type, _ = registry.get(ptrname)
                return defined_type == DefinedType.Scalar
        # Genuinely undeclared: a GPU-global Scalar is a device pointer accessed as
        # x[...]; any other Scalar is allocated by value (see CPUCodeGen.allocate_array).
        return isinstance(desc, dt.Scalar) and desc.storage != dtypes.StorageType.GPU_Global

    def _register_index_function(self, data_name: str, desc):
        """Registers (once per distinct descriptor signature) the ``<name>_idx`` index function for
        ``data_name`` and returns ``(function_name, extra_symbol_names)``; a same-named array of
        different shape/strides is given a disambiguated name so arity and offset math still match."""
        ndim = len(desc.shape)
        dim_syms = [symbolic.symbol('__d%d' % i) for i in range(ndim)]
        strides = [symbolic.pystr_to_symbolic(str(s)) for s in desc.strides]
        offset = [symbolic.pystr_to_symbolic(str(o)) for o in desc.offset]
        flat = sum(dim_syms[i] * strides[i] for i in range(ndim))
        const = sum(offset[i] * strides[i] for i in range(ndim))
        flatexpr = flat + const
        extra = sorted((flatexpr.free_symbols - set(dim_syms)), key=lambda s: str(s))
        extra_names = [str(s) for s in extra]

        base = re.sub(r'\W', '_', data_name)
        sig = (ndim, tuple(str(s) for s in desc.strides), tuple(str(o) for o in desc.offset))
        key = (base, sig)
        if key in self._index_sig_to_name:
            return self._index_sig_to_name[key], extra_names

        fnname = base + '_idx'
        # A same-name/different-signature collision (the exact-match case returned above): the base
        # name is already an emitted definition, so disambiguate.
        if fnname in self._index_functions:
            fnname = '%s_%dd_%d_idx' % (base, ndim, len(self._index_sig_to_name))
        self._index_sig_to_name[key] = fnname

        ctype = index_ctype()
        params = ['%s %s' % (ctype, str(d)) for d in dim_syms]
        params += ['%s %s' % (ctype, s) for s in extra_names]
        body = sym2cpp(flatexpr)
        self._index_functions[fnname] = '%s %s %s(%s) { return %s; }' % (index_function_qualifier(), ctype, fnname,
                                                                         ', '.join(params), body)
        return fnname, extra_names

    # -- readable array size --------------------------------------------------

    def _register_size_function(self, data_name: str, desc) -> Optional[Tuple[str, List[str]]]:
        """Registers (once per distinct name + size expression) the ``<array>_size`` helper for
        ``desc.total_size`` and returns ``(function_name, call_args)``, or ``None`` when not
        worthwhile.

        Readability threshold: only a constant or compound symbolic ``total_size`` gets a helper -- a
        bare single symbol (``N``) is skipped (``A_size(N){return N;}`` is no win). A constant folds to
        a nullary ``consteval`` helper; a compound symbolic size gets a ``constexpr`` helper over its
        sorted free symbols (``A_size(N, M)``).
        """
        total = symbolic.pystr_to_symbolic(str(desc.total_size))
        # A bare single symbol carries no readability benefit; keep the plain name.
        if total.is_Symbol:
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
        # A same-name/different-signature collision (exact-match returned above): the base name is
        # already an emitted definition, so disambiguate.
        if fnname in self._size_functions:
            fnname = '%s_%d_size' % (base, len(self._size_sig_to_name))
        self._size_sig_to_name[key] = fnname

        qualifier = size_qualifier(is_constant)
        ctype = index_ctype()
        params = ['%s %s' % (ctype, s) for s in call_args]
        body = sym2cpp(total)
        self._size_functions[fnname] = '%s %s %s(%s) { return %s; }' % (qualifier, ctype, fnname, ', '.join(params),
                                                                        body)
        return fnname, call_args


class ReadableKeywordRemover(cpp.DaCeKeywordRemover):
    """
    Extends the classic keyword remover: in addition to connector accesses, it
    lowers direct array accesses (``A[i, j]`` for ``A`` in ``sdfg.arrays``, as
    produced by InlineTaskletConnectors) to ``A[A_idx(i, j, ...)]``.
    """

    def _is_bare_data(self, name: str) -> bool:
        return name not in self.memlets and name not in self.constants and name in self.sdfg.arrays

    def _scalar_constant_name(self, name: str) -> Optional[str]:
        """``name`` if it is a 0-dimensional (scalar) SDFG constant, else None. MarkConstInit promotes a
        write-once scalar transient to such a constant, which framecode emits as a bare ``constexpr T
        name = v;`` (not ``T name[1]``). A subscript ``name[0]`` on it must lower to the bare ``name`` --
        routing it through the classic ``_subscript_expr`` trips on the scalar's empty (``()``) stride
        list (``Missing dimensions in expression (expected one, got 0)``)."""
        if name in self.constants and numpy.ndim(self.constants[name]) == 0:
            return name
        return None

    def _bare_access(self, node: ast.AST) -> Optional[str]:
        """ C++ access string for a direct (inlined) access to an SDFG array. """
        name = rname(node)
        # ptr_increment: an access this map's walk plan covers is emitted as a pointer dereference
        # ``(*__walk_X)`` rather than a recomputed ``X[X_idx(..)]``. Checked before array_index_access
        # so a fully-walked array never registers an (unused) ``<array>_idx`` helper.
        if isinstance(node, ast.Subscript):
            walk = self.codegen.current_walk_accesses()
            if walk is not None:
                pointer = walk.get((name, tuple(self._index_list(node.slice))))
                if pointer is not None:
                    return '(*%s)' % pointer
        desc = self.sdfg.arrays[name]
        info = self.codegen.array_index_access(self.sdfg, desc, name)
        if info is None:
            # Scalar -> plain C++ value.
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
        # Connector / constant / non-array targets keep the classic lowering.
        if not self._is_bare_data(target):
            return super().visit_Assign(node)
        # A bare-data write target is never WCR (the pass never inlines WCR
        # outputs). Emit the whole assignment as one statement so the C++
        # unparser does not treat the left-hand side as a new 'auto' variable.
        value = self.visit(astutils.copy_tree(node.value))
        lhs = self._bare_access(target_node)
        if lhs is None:
            return self.generic_visit(node)
        rhs = cppunparse.cppunparse(value, expr_semicolon=False)
        desc = self.sdfg.arrays[target]
        if self.codegen._is_const_scalar(desc):
            # Single-write scope-local scalar: fuse the (skipped) declaration and the
            # write into one `const T x = expr;` binding. Safe because this write is a
            # connector-free single assignment -> emitted brace-free at the enclosing
            # scope, and MarkConstInit proved it is the only write and precedes reads.
            newnode = ast.Name(id='const %s %s = %s;' % (desc.dtype.ctype, lhs, rhs))
        elif self.codegen._is_const_len1_array(desc):
            # Single-write single-element stack array -> `const T x[1] = {(T)(expr)};`; reads keep their
            # `x[x_idx(0)]` form (== x[0]). The explicit `(T)` cast matches legacy's implicit narrowing on
            # a plain `x[0] = expr;` assignment (e.g. a float sink of a double-returning ``sqrt``); without
            # it the braced list-initializer would raise -Wnarrowing where legacy is silent.
            name = self.codegen.ptr(target, desc, self.sdfg)
            ctype = desc.dtype.ctype
            newnode = ast.Name(id='const %s %s[1] = {(%s)(%s)};' % (ctype, name, ctype, rhs))
        else:
            newnode = ast.Name(id='%s = %s;' % (lhs, rhs))
        return self._replace_assignment(newnode, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        target = rname(node)
        # A subscript on a 0-d scalar SDFG constant (emitted as a bare ``constexpr T c = v;``) lowers to
        # the bare name; the classic ``_subscript_expr`` would raise on its empty stride list.
        bare_const = self._scalar_constant_name(target)
        if bare_const is not None:
            return ast.copy_location(ast.Name(id=bare_const), node)
        # Connectors and SDFG constants keep the classic lowering.
        if not self._is_bare_data(target):
            return super().visit_Subscript(node)
        access = self._bare_access(node)
        if access is None:
            return self.generic_visit(node)
        return ast.copy_location(ast.Name(id=access), node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # A bare data name that is a by-value scalar -> emit the plain name.
        name = rname(node)
        if self._is_bare_data(name):
            desc = self.sdfg.arrays[name]
            ptrname = self.codegen.ptr(name, desc, self.sdfg)
            if self.codegen._is_value_scalar(ptrname, desc):
                return ast.copy_location(ast.Name(id=ptrname), node)
        return super().visit_Name(node)

    def _index_list(self, slicenode: ast.AST) -> List[str]:
        return subscript_index_strings(slicenode)
