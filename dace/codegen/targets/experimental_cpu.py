# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Experimental "readable" CPU code generator.

Selected by ``compiler.cpu.implementation = experimental``. It subclasses the
classic :class:`~dace.codegen.targets.cpu.CPUCodeGen` and changes only how
tasklets and array accesses are emitted, producing human-readable C++:

* Every array access goes through a generated ``static DACE_HDFI constexpr``
  ``<array>_idx(...)`` index function, so the linear-offset arithmetic appears
  once (per array) instead of being inlined at every access site.
* Tasklets whose connectors were inlined by
  :class:`~dace.transformation.passes.inline_tasklet_connectors.InlineTaskletConnectors`
  access their arrays directly, with no copy-in / copy-out temporaries.
* Write-once data marked by
  :class:`~dace.transformation.passes.mark_const_init.MarkConstInit` is emitted
  as ``const`` / ``constexpr``.

The two SDFG passes run once, before code generation, in
``dace.codegen.codegen.generate_code`` (gated on the same flag). Because the GPU
(CUDA) code generator emits its device tasklets through the shared CPU
generator instance, these changes also apply inside ``__global__`` kernels.
"""
import ast
import re
from typing import Dict, List, Optional, Set, Tuple

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
# Qualifier for the generated index functions: host+device callable, inlined,
# and usable in constant expressions.
INDEX_FUNCTION_QUALIFIER = 'static DACE_HDFI constexpr'
# Qualifier for a symbolic ``<array>_size`` helper: same as the index functions
# (a runtime-evaluated ``constexpr`` over the size's free symbols).
SIZE_FUNCTION_QUALIFIER = INDEX_FUNCTION_QUALIFIER
# Qualifier for a constant ``<array>_size`` helper: ``consteval`` forces the fixed
# allocation extent to fold at compile time (C++20; host .cpp only -- a constant
# heap size is never registered while emitting a device kernel).
SIZE_CONSTEVAL_QUALIFIER = 'static DACE_HDFI consteval'
# Alignment (bytes) for CPU heap arrays allocated via std::aligned_alloc (matches
# the classic generator's DACE_ALIGN(64)).
HEAP_ALIGNMENT = 64


# NOTE: not registered with the target registry -- it is selected explicitly by
# ``dace.codegen.codegen.generate_code`` when ``compiler.cpu.implementation`` is
# ``experimental``. Registering it would cause double instantiation.
class ExperimentalCPUCodeGen(CPUCodeGen):
    """ Human-readable CPU/GPU-kernel code generator (see module docstring). """

    def __init__(self, frame, sdfg):
        super().__init__(frame, sdfg)
        # Index-function name -> full C++ definition (deduplicated).
        self._index_functions: Dict[str, str] = {}
        # Size-helper name -> full C++ definition (deduplicated). An <array>_size()
        # helper returns the array's total_size; see _register_size_function.
        self._size_functions: Dict[str, str] = {}
        # id(function_stream) -> index / size helper names already flushed to THAT
        # stream. Dedup is per stream (= per generated output file), so an array
        # accessed on both the host (.cpp) and inside a device kernel (.cu) gets
        # its `A_idx` definition in each file rather than only the first one. Index
        # and size helpers share this set (their names never collide: `A_idx` vs
        # `A_size`).
        self._emitted_functions: Dict[int, Set[str]] = {}
        # (base-name, signature) -> emitted function name, and its inverse, so
        # the SAME array shape reuses one function while different shapes that
        # happen to share a (connector-derived) name -- e.g. ``_out`` as a 1-D
        # array in one nested SDFG and a 2-D array in another -- get distinct
        # functions instead of a name collision (wrong arity / wrong strides).
        self._index_sig_to_name: Dict[tuple, str] = {}
        self._index_name_to_sig: Dict[str, tuple] = {}
        # Same dedup for size helpers, keyed on (base-name, size-expression) so two
        # same-named arrays with different total_size expressions get distinct
        # helpers instead of an arity / return collision.
        self._size_sig_to_name: Dict[tuple, str] = {}
        self._size_name_to_sig: Dict[str, tuple] = {}
        # Per-tasklet cache of identifiers appearing in the (rewritten) body.
        self._body_identifiers: Dict[int, Set[str]] = {}
        # Per-native-tasklet cache: id(node) -> {connector_name: cpp_access}. An
        # entry means the connector is accessed directly (scalar index / base
        # pointer) instead of through a copy-in / copy-out temporary.
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
        # An OpenMP ``declare reduction`` directive is emitted ahead of the loop pragma and is
        # bound by this scope; leaking it out would redefine it across sibling maps.
        if (node.map.schedule == dtypes.ScheduleType.CPU_Multicore
                and Config.get_bool('compiler', 'emit_tree_reductions')
                and any(declare is not None
                        for _op, _target, _dname, declare in self._collect_omp_reductions(sdfg, state_dfg, node))):
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
            try:
                tree = ast.parse(code)
                ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            except SyntaxError:
                # Fall back to "everything is used" (classic lowering) on parse failure.
                ids = set(node.in_connectors.keys()) | set(node.out_connectors.keys())
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
        """True if ``node`` is a Python tasklet whose body is exactly one assignment
        (so, once its connectors are inlined, it is a single array-store statement
        with no locally declared variable to keep in scope)."""
        if not isinstance(node, nodes.Tasklet) or node.language != dtypes.Language.Python:
            return False
        stmts = node.code.code
        if isinstance(stmts, str):
            try:
                stmts = ast.parse(stmts).body
            except SyntaxError:
                return False
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
        body = type(node).__properties__["code"].to_string(node.code)
        inline = self._cpp_inline.get(id(node))
        if inline is None:
            inline = self._compute_cpp_inline(sdfg, state_dfg, node)
            self._cpp_inline[id(node)] = inline
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
        call_args = [sym2cpp(symbolic.pystr_to_symbolic(ix)) for ix in indices] + list(extra)
        return '%s[%s(%s)]' % (ptrname, fnname, ', '.join(call_args))

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
        # constexpr_static const-init data is promoted to an SDFG constant by
        # MarkConstInit and emitted once by generate_constants; skip its runtime
        # allocation to avoid a redefinition.
        if nodedesc.const_init and nodedesc.const_init_kind == 'constexpr_static':
            return
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
        # dace::aligned_alloc<T>(count, alignment) (dace/alloc.h) instead of aligned
        # new[] (which would be a new[]/delete[] alignment mismatch); the wrapper
        # hides the (T*) cast, sizeof, and the alignment round-up. Use the
        # descriptor's alignment when set, else HEAP_ALIGNMENT. The count is routed
        # through a generated <array>_size(...) helper when worthwhile (see
        # _register_size_function), else the classic sym2cpp'd string.
        align = alignment or HEAP_ALIGNMENT
        count = self._array_size_access(sdfg, nodedesc, data_name, arrsize)
        return '%s = dace::aligned_alloc<%s>(%s, %d);\n' % (alloc_name, ctype, count, align)

    def heap_free_stmt(self, alloc_name: str, is_array: bool) -> str:
        # Memory from dace::aligned_alloc is released with dace::free.
        return 'dace::free(%s);\n' % alloc_name

    def _flush_generated_functions(self, function_stream, cfg, state_id, node) -> None:
        # Emit each registered index / size helper (a single-line ``static`` free
        # function) once per stream. A non-inline nested-SDFG function has its own
        # function stream that lands in the SAME output file as the outer stream, so
        # the same helper can still be written more than once per file; those
        # cross-stream duplicates are collapsed by ``deduplicate_functions`` in the
        # compiler post-pass (a line-level dedup), so no include-guard macros are
        # needed here. A different file (the .cu device code) has its own stream and
        # therefore its own copy, exactly as required. Index and size helpers share
        # the per-stream emitted set (their names never collide).
        emitted = self._emitted_functions.setdefault(id(function_stream), set())
        for registry in (self._index_functions, self._size_functions):
            for name, defn in registry.items():
                if name in emitted:
                    continue
                function_stream.write(defn + '\n', cfg, state_id, node)
                emitted.add(name)

    # -- readable array indexing ----------------------------------------------

    def _is_const_scalar(self, desc) -> bool:
        """A single-write (``const_runtime``) scope-local scalar emitted as a
        ``const T x = expr;`` binding: the mutable ``T x;`` declaration is skipped and
        the single write becomes the const binding. Restricted to scope-lifetime CPU
        value scalars (declared as ``T x;`` by the base generator) so the binding is
        declared in exactly the scope its reads live in; a device / persistent scalar
        keeps the classic allocation."""
        return (isinstance(desc, dt.Scalar) and desc.const_init and desc.const_init_kind == 'const_runtime'
                and desc.lifetime == dtypes.AllocationLifetime.Scope and desc.storage
                in (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap))

    def _is_const_len1_array(self, desc) -> bool:
        """A single-write (``const_runtime``) single-element STACK array emitted as
        ``const T x[1] = {expr};`` fused at its write site (the length-1 → scalar pass
        turns most of these into scalars first; this covers any that survive as arrays,
        e.g. outside that pass). Restricted to a Register array -- a stack ``T x[1]``,
        detectable by its resolved storage; a heap (``T* x = alloc``) or device
        single-element array keeps the classic allocation."""
        return (isinstance(desc, dt.Array) and not isinstance(desc, dt.View) and desc.const_init
                and desc.const_init_kind == 'const_runtime' and desc.lifetime == dtypes.AllocationLifetime.Scope
                and desc.storage == dtypes.StorageType.Register and len(desc.shape) >= 1
                and all(d == 1 for d in desc.shape))

    def array_index_access(self, sdfg, desc, data_name: str):
        """
        Registers (once) the ``<array>_idx`` index function for ``data_name`` and
        returns ``(ptrname, fnname, ndim, extra_syms)``, or None if this access is
        a plain C++ value (a scalar represented by value, not a pointer). The
        caller fills in the per-dimension index expressions.
        """
        ptrname = self.ptr(data_name, desc, sdfg)
        if self._is_value_scalar(ptrname, desc):
            return None  # scalar is a plain C++ value; use the bare name
        ndim = len(desc.shape)
        fnname, extra_syms = self._register_index_function(data_name, desc)
        return (ptrname, fnname, ndim, extra_syms)

    def _is_value_scalar(self, ptrname: str, desc) -> bool:
        """
        Whether ``ptrname`` is a plain C++ value (accessed as ``x``) rather than a
        pointer (accessed as ``x[...]``). Keyed on the emitted ``DefinedType``,
        because a ``Scalar`` that is heap-, device-, or persistently allocated is a
        pointer, while a register scalar is a value. Consults both the defined and
        declared registries so a pointer-registered scalar is never mistaken for a
        value; only when genuinely undeclared is the descriptor storage used.
        """
        for registry in (self._dispatcher.defined_vars, self._dispatcher.declared_arrays):
            try:
                defined_type, _ = registry.get(ptrname)
                return defined_type == DefinedType.Scalar
            except KeyError:
                continue
        # Genuinely undeclared: a GPU-global Scalar is a device pointer accessed as
        # x[...]; any other Scalar is allocated by value (see CPUCodeGen.allocate_array).
        return isinstance(desc, dt.Scalar) and desc.storage != dtypes.StorageType.GPU_Global

    def _register_index_function(self, data_name: str, desc):
        """
        Registers (once per distinct descriptor signature) the index function for
        ``data_name`` and returns ``(function_name, extra_symbol_names)``. The
        function is named ``<name>_idx``; a second array that shares the name but
        has a different shape/strides gets a disambiguated name so its call arity
        and offset math match its declaration.
        """
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
        if fnname in self._index_name_to_sig and self._index_name_to_sig[fnname] != sig:
            fnname = '%s_%dd_%d_idx' % (base, ndim, len(self._index_sig_to_name))
        self._index_sig_to_name[key] = fnname
        self._index_name_to_sig[fnname] = sig

        params = ['%s %s' % (INDEX_CTYPE, str(d)) for d in dim_syms]
        params += ['%s %s' % (INDEX_CTYPE, s) for s in extra_names]
        body = sym2cpp(flatexpr)
        self._index_functions[fnname] = '%s %s %s(%s) { return %s; }' % (INDEX_FUNCTION_QUALIFIER, INDEX_CTYPE, fnname,
                                                                         ', '.join(params), body)
        return fnname, extra_names

    # -- readable array size --------------------------------------------------

    def _array_size_access(self, sdfg, desc, data_name: Optional[str], fallback: str) -> str:
        """
        C++ expression for an array's allocation element count: ``<array>_size(...)``
        when a size helper is worthwhile, else ``fallback`` (the classic
        ``sym2cpp(total_size)`` string).

        :param sdfg: Owning SDFG (only the descriptor is consulted).
        :param desc: The array's data descriptor.
        :param data_name: The array name used to build the helper name.
        :param fallback: Pre-computed ``sym2cpp(total_size)`` used when no helper is
                         emitted (or when called without descriptor context).
        :return: The count expression to place in the allocation statement.
        """
        if sdfg is None or desc is None or data_name is None:
            return fallback
        registered = self._register_size_function(data_name, desc)
        if registered is None:
            return fallback
        fnname, call_args = registered
        return '%s(%s)' % (fnname, ', '.join(call_args))

    def _register_size_function(self, data_name: str, desc) -> Optional[Tuple[str, List[str]]]:
        """
        Registers (once per distinct name + size expression) the ``<array>_size``
        helper for ``desc.total_size`` and returns ``(function_name, call_args)``, or
        ``None`` when a helper is not worthwhile.

        Threshold (readability): a helper is emitted only when ``total_size`` is a
        constant or a compound symbolic expression. A bare single symbol (``N``) is
        skipped -- wrapping it as ``A_size(N) { return N; }`` is no readability win.
        A constant folds (via SymPy) to a single literal and gets a nullary
        ``consteval`` helper (``A_size()``) that names the fixed allocation extent as
        a compile-time constant; a compound symbolic size (``N*M``, ``N**2``) gets a
        ``constexpr`` helper over its sorted free symbols (``A_size(N, M)``).

        :param data_name: The array name (its non-word chars become underscores).
        :param desc: The array's data descriptor.
        :return: ``(function_name, call_args)`` for the call site, or ``None``.
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
        if fnname in self._size_name_to_sig and self._size_name_to_sig[fnname] != sig:
            fnname = '%s_%d_size' % (base, len(self._size_sig_to_name))
        self._size_sig_to_name[key] = fnname
        self._size_name_to_sig[fnname] = sig

        qualifier = SIZE_CONSTEVAL_QUALIFIER if is_constant else SIZE_FUNCTION_QUALIFIER
        params = ['%s %s' % (INDEX_CTYPE, s) for s in call_args]
        body = sym2cpp(total)
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
        call_args = [sym2cpp(symbolic.pystr_to_symbolic(ix)) for ix in indices] + list(extra_syms)
        return '%s[%s(%s)]' % (ptrname, fnname, ', '.join(call_args))

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
            # Single-write single-element stack array -> `const T x[1] = {expr};`; reads
            # keep their `x[x_idx(0)]` form (== x[0]).
            name = self.codegen.ptr(target, desc, self.sdfg)
            newnode = ast.Name(id='const %s %s[1] = {%s};' % (desc.dtype.ctype, name, rhs))
        else:
            newnode = ast.Name(id='%s = %s;' % (lhs, rhs))
        return self._replace_assignment(newnode, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        target = rname(node)
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
