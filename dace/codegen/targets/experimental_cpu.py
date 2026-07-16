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
from typing import Dict, List, Optional, Set, Tuple

import numpy
import sympy
from pygments.lexers import CppLexer
from pygments.token import Token

from dace import data as dt
from dace import dtypes, symbolic
from dace.codegen import cppunparse
from dace.codegen.common import sym2cpp
from dace.config import Config
from dace.codegen.dispatcher import DefinedType
from dace.codegen.targets import cpp
from dace.codegen.targets.cpu import CPUCodeGen
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.sdfg import nodes
from dace.sdfg.utils import dynamic_map_inputs

# The C++ integer type used for computed flat indices.
INDEX_CTYPE = 'long long'
# Qualifier for the generated ``<array>_idx`` and symbolic ``<array>_size`` helpers: host+device
# callable, inlined, usable in constant expressions.
INDEX_FUNCTION_QUALIFIER = 'static DACE_HDFI constexpr'
# Qualifier for a CONSTANT ``<array>_size`` helper: ``consteval`` forces the fixed extent to fold at
# compile time (a C++20 keyword, so size_qualifier falls back to ``constexpr`` before C++20).
SIZE_CONSTEVAL_QUALIFIER = 'static DACE_HDFI consteval'


def size_qualifier(is_constant: bool) -> str:
    """Qualifier for an ``<array>_size`` helper: ``consteval`` for a constant extent under C++20+
    (folds it at compile time), else ``constexpr`` (the same qualifier as the index functions)."""
    if not is_constant:
        return INDEX_FUNCTION_QUALIFIER
    standard = int(str(Config.get('compiler', 'cpp_standard')).strip())
    return SIZE_CONSTEVAL_QUALIFIER if standard >= 20 else INDEX_FUNCTION_QUALIFIER


def constexpr_body(expr) -> str:
    """C++ body for a ``constexpr``/``consteval`` ``_idx``/``_size`` helper. RelaxIntegerPowers rewrites
    integer powers ``x**k`` in shapes/strides/offsets to ``dace::math::ipow(x, k)``, which ``sym2cpp``
    emits as a non-``constexpr`` ``ipow`` call (``-Winvalid-constexpr``, and not a real constant
    expression). Rewriting each ``ipow`` back to a SymPy ``Pow`` lets the printer lower it to the
    repeated-multiply form ``((N * N))`` -- value-identical, and a genuine constant expression. Only the
    readable ``_idx``/``_size`` helper bodies take this path."""
    return sym2cpp(expr.rewrite(sympy.Pow))


def format_index_access(ptrname: str, fnname: str, indices: List[str], extra: List[str]) -> str:
    """C++ ``ptr[fn(idx.., extra..)]`` access through a registered ``<array>_idx`` index function."""
    call_args = [sym2cpp(symbolic.pystr_to_symbolic(ix)) for ix in indices] + list(extra)
    return '%s[%s(%s)]' % (ptrname, fnname, ', '.join(call_args))


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
        self._emitted_functions: Dict[int, Set[str]] = {}
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
        file_key = id(self) if self.calling_codegen is self else id(function_stream)
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
        return (isinstance(desc, dt.Scalar) and vars(desc).get('const_init')
                and desc.lifetime == dtypes.AllocationLifetime.Scope and desc.storage
                in (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap))

    def _is_const_len1_array(self, desc) -> bool:
        """A single-write (``const_runtime``) single-element STACK (Register) array emitted as a fused
        ``const T x[1] = {expr};`` binding. A heap or device single-element array stays classic."""
        return (isinstance(desc, dt.Array) and not isinstance(desc, dt.View) and vars(desc).get('const_init')
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

        params = ['%s %s' % (INDEX_CTYPE, str(d)) for d in dim_syms]
        params += ['%s %s' % (INDEX_CTYPE, s) for s in extra_names]
        body = constexpr_body(flatexpr)
        self._index_functions[fnname] = '%s %s %s(%s) { return %s; }' % (INDEX_FUNCTION_QUALIFIER, INDEX_CTYPE, fnname,
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
        params = ['%s %s' % (INDEX_CTYPE, s) for s in call_args]
        body = constexpr_body(total)
        self._size_functions[fnname] = '%s %s %s(%s) { return %s; }' % (qualifier, INDEX_CTYPE, fnname,
                                                                        ', '.join(params), body)
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
        node = slicenode
        if isinstance(node, ast.Index):  # py<3.9 compatibility
            node = node.value
        if isinstance(node, ast.Tuple):
            return [ast.unparse(e) for e in node.elts]
        return [ast.unparse(node)]
