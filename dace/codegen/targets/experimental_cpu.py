# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Experimental "readable" CPU code generator (``compiler.cpu.implementation = experimental_readable``).

Subclasses :class:`~dace.codegen.targets.cpu.CPUCodeGen` and changes only how tasklets and array
accesses are emitted: array accesses go through a generated ``<array>_idx(...)`` index function
(the offset arithmetic appears once per array); tasklets whose connectors were inlined by
``InlineTaskletConnectors`` access arrays directly (no copy-in/out temporaries); and transients that
only store literals are SDFG constants (``PromoteConstantTransients``), emitted as ``constexpr``.
Both passes run before codegen in ``dace.codegen.codegen.generate_code``. The GPU generator emits
device tasklets through the shared CPU instance, so these changes also apply inside ``__global__``
kernels.
"""

import ast
import re
from typing import Optional

import numpy
from pygments.lexers import CppLexer
from pygments.token import Token

from dace import data as dt
from dace import dtypes, symbolic
from dace.codegen import cppunparse
from dace.codegen.codeobject import CODE_ANNOTATION
from dace.codegen.common import sym2cpp
from dace.codegen.dispatcher import DefinedType
from dace.codegen.targets import cpp
from dace.codegen.targets.cpu import (
    CPUCodeGen,
)
from dace.config import Config
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.properties import CodeBlock
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import dynamic_map_inputs
from dace.utils import find_new_name

# Qualifier for the generated ``<array>_idx`` and symbolic ``<array>_size`` helpers: host+device
# callable, inlined, usable in constant expressions.
INDEX_FUNCTION_QUALIFIER = "static DACE_HDFI constexpr"
# Qualifier for a CONSTANT ``<array>_size`` helper: ``consteval`` forces the fixed extent to fold at
# compile time (a C++20 keyword, so size_qualifier falls back to ``constexpr`` before C++20).
SIZE_CONSTEVAL_QUALIFIER = "static DACE_HDFI consteval"
# Identifier tokens of a code string. Used where a name has to be found in code that has no AST here
# (a C++ tasklet body, a library node's code property): over-matching costs a refusal, missing a
# token costs a miscompile, so the tokenizer is deliberately the crude one.
IDENTIFIER_TOKENS = re.compile(r"[A-Za-z_]\w*")
# Punctuation a C++ declarator may carry between its type and its name (``double *p``, ``T &r``,
# ``std::vector<double> v``). _declared_identifiers looks THROUGH these when deciding whether the
# token before an identifier introduced it, so the same tokens used as operators (``a * b``,
# ``a > b``) read as declarations too -- the safe direction, see that method.
DECLARATOR_TOKENS = frozenset({"*", "&", ">"})
# Preprocessor lines :func:`deduplicate_includes` reasons about.
INCLUDE_LINE = re.compile(r"^\s*#\s*include\s")
PREPROCESSOR_IF = re.compile(r"^\s*#\s*if")
PREPROCESSOR_ENDIF = re.compile(r"^\s*#\s*endif")


def scalar_decl() -> str:
    """Where a mutable scope scalar is declared, per ``codegen_params.scalar_decl``: ``eager`` (the default, ``T x;``
    at the top of its scope), ``late`` (``T x;`` right before its first use) or ``late_fused`` (``T x = expr;``, the
    declaration is the first write)."""
    return Config.get("compiler", "cpu", "codegen_params", "scalar_decl")


def counter_init_assigns_only(loop: LoopRegion) -> bool:
    """Whether ``loop``'s init statement is exactly ``<loop_variable> = <expr>``, the one shape a
    declared-in-place counter can be spelled as (``T i = <expr>``). Any other init (a tuple target, a
    compound statement, an augmented assignment, or one that initialises a DIFFERENT name) has no such
    spelling, so the counter keeps its hoisted declaration."""
    code = loop.init_statement.code
    if not isinstance(code, list) or len(code) != 1:
        return False
    stmt = code[0]
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == loop.loop_variable
    )


def counter_used_outside_loop(name: str, loop: LoopRegion, sdfg: SDFG) -> bool:
    """Whether ``name`` is read or written anywhere outside ``loop``. Declaring the counter in the
    loop's own ``for``-init clause scopes it to the loop, so its value stops being observable after the
    loop closes -- exactly what a use outside would need. DaCe permits such a use (a LoopRegion leaks
    its counter's final value to subsequent blocks), so this must be checked, not assumed.

    Every block of the SDFG is enumerated individually, hence a REGION is asked only for the symbols it
    uses on itself (``with_contents=False`` -- its condition / init / update); its contents arrive as
    their own blocks. A state must be asked WITH contents: ``SDFGState.used_symbols(with_contents=False)``
    returns the empty set, which would silently hide every real use.
    """
    inside = {id(loop)} | {id(block) for block in loop.all_control_flow_blocks()}
    for block in sdfg.all_control_flow_blocks():
        if id(block) in inside:
            continue
        with_contents = not isinstance(block, AbstractControlFlowRegion)
        if name in block.used_symbols(all_symbols=True, with_contents=with_contents):
            return True
    inside_edges = {id(edge) for edge in loop.all_interstate_edges()}
    for edge in sdfg.all_interstate_edges():
        if id(edge) in inside_edges:
            continue
        if name in edge.data.free_symbols or name in edge.data.assignments:
            return True
    # A descriptor whose shape/strides mention the counter is materialised outside the loop.
    for desc in sdfg.arrays.values():
        if name in {str(s) for s in desc.free_symbols}:
            return True
    return False


def loop_local_counter_ctype(name: str, dtype: dtypes.typeclass, sdfg: SDFG) -> str | None:
    """C++ type to declare a LoopRegion counter INSIDE its ``for``-init clause in the readable
    generator, or ``None`` to keep the hoisted declaration. The counter must be owned by exactly one
    non-inverted LoopRegion whose init is a plain assignment, and must not be read outside that loop.
    """
    owners = [
        cfr
        for cfr in sdfg.all_control_flow_regions()
        if isinstance(cfr, LoopRegion) and cfr.loop_variable == name and cfr.init_statement is not None
    ]
    if len(owners) != 1:
        return None
    loop = owners[0]
    if loop.inverted or not counter_init_assigns_only(loop) or counter_used_outside_loop(name, loop, sdfg):
        return None
    # Instrumentation conditions (e.g. data-instrument ``i == 0``) are emitted around access-node
    # uses that may be outside the loop body, so the counter must stay in function scope.
    if loop_variable_in_instrument_conditions(loop, name):
        return None
    return dtype.ctype


def loop_variable_in_instrument_conditions(loop: LoopRegion, name: str) -> bool:
    """Return True if ``name`` may be referenced by instrumentation code emitted outside ``loop``.

    Data-instrumentation conditions and state-level symbol dumps are emitted around state
    boundaries; the loop variable is not in scope before the ``for``-init clause, so any such
    reference forces a hoisted declaration.
    """
    sdfg = loop.sdfg
    name_re = re.compile(r"\b" + re.escape(name) + r"\b")

    # Access-node instrumentation conditions anywhere in the SDFG may reference the loop variable.
    for node, _state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.AccessNode):
            continue
        cond = node.instrument_condition
        if isinstance(cond, CodeBlock) and cond.as_string and name_re.search(cond.as_string):
            return True

    # State-level symbol instrumentation dumps every symbol in state.defined_symbols(); if the loop
    # variable is among them the dump is emitted at the top of the state, which can be before the
    # ``for``-init clause that declares it.
    for state in sdfg.states():
        sym_instr = state.symbol_instrument
        if sym_instr != dtypes.DataInstrumentationType.No_Instrumentation and name in state.defined_symbols():
            return True
        sym_cond = state.symbol_instrument_condition
        if isinstance(sym_cond, CodeBlock) and sym_cond.as_string and name_re.search(sym_cond.as_string):
            return True
    return False


def code_blocks_of(value) -> tuple[CodeBlock, ...]:
    """The ``CodeBlock`` values reachable from one property value: bare, or inside a list / dict.

    Properties are how a DaCe node stores everything it lowers to code, so walking them finds the code
    of a node class this file has never heard of -- which is the point: an unanticipated code-bearing
    property must make the deferral gate refuse, not silently see nothing."""
    if isinstance(value, CodeBlock):
        return (value,)
    if isinstance(value, dict):
        return tuple(v for v in value.values() if isinstance(v, CodeBlock))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(v for v in value if isinstance(v, CodeBlock))
    return ()


def identifiers_in(blocks) -> set[str]:
    """Identifier tokens of a sequence of ``CodeBlock``s."""
    names: set[str] = set()
    for block in blocks:
        text = block.as_string
        if text:
            names |= set(IDENTIFIER_TOKENS.findall(text))
    return names


def size_qualifier(is_constant: bool) -> str:
    """Qualifier for an ``<array>_size`` helper: ``consteval`` for a constant extent under C++20+
    (folds it at compile time), else ``constexpr`` (the same qualifier as the index functions)."""
    if not is_constant:
        return INDEX_FUNCTION_QUALIFIER
    standard = int(str(Config.get("compiler", "cpp_standard")).strip())
    return SIZE_CONSTEVAL_QUALIFIER if standard >= 20 else INDEX_FUNCTION_QUALIFIER


def index_ctype(desc: dt.Data, flat: symbolic.SymbolicType) -> str:
    """The narrowest integer that holds every flat index of ``desc``, ``flat`` over ``__d<i>`` dimension symbols.

    Narrower than ``int`` gains nothing, as C++ promotes to it anyway; a size the compiler cannot see stays 64-bit.
    """
    corner = {f"__d{i}": symbolic.pystr_to_symbolic(str(s)) - 1 for i, s in enumerate(desc.shape)}
    largest = abs(
        symbolic.pystr_to_symbolic(str(flat)).subs({symbolic.pystr_to_symbolic(k): v for k, v in corner.items()})
    )
    return "int32_t" if largest.is_Integer and largest < 2**31 else "int64_t"


def format_index_access(ptrname: str, fnname: str, indices: list[str], extra: list[str]) -> str:
    """C++ ``ptr[fn(idx.., extra..)]`` access through a registered ``<array>_idx`` index function."""
    call_args = [sym2cpp(symbolic.pystr_to_symbolic(ix)) for ix in indices] + list(extra)
    return "%s[%s(%s)]" % (ptrname, fnname, ", ".join(call_args))


def parenthesized_ptr(expr: str) -> str:
    """A base-pointer expression safe to substitute for a connector name in a tasklet body.

    The expression replaces the connector TEXTUALLY, so the body decides how it is used -- and a
    library body such as ``Scan``'s writes ``<conn>[_i]``. An offset pointer ``a + 1`` then reads as
    ``a + 1[_i]``, which C++ parses as ``a + (1[_i])`` and rejects with ``invalid types 'int[int]'
    for array subscript``. Wrapping binds the base first; the same parentheses are equally correct
    where the body dereferences it or passes it on. A bare name needs none."""
    return expr if expr.isidentifier() else f"({expr})"


def index_expr_nodes(slicenode: ast.AST) -> list[ast.AST]:
    """The per-dimension index AST expressions of a subscript's slice (a tuple's elements, else the
    single expression)."""
    node = slicenode
    if isinstance(node, ast.Index):  # py<3.9 compatibility
        node = node.value
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def deduplicate_includes(code: str) -> str:
    """Drop repeated ``#include`` lines from assembled global code, keeping the FIRST of each.

    A translation unit's global code is written by streams that cannot see one another -- the file
    header (target/environment headers, then every SDFG's ``global_code``) and one write per tasklet
    ``code_global`` (``cpp.unparse_tasklet``) -- so a header named by two writers arrives twice. This
    is the one point that sees all of them.

    Only repeats OUTSIDE preprocessor conditionals go: an include under ``#if`` is picked by the
    branch it sits in, so it is neither dropped nor counted as the first occurrence. Lines compare
    without the per-node ``////__DACE`` annotation that makes identical includes textually differ.
    Non-include lines are untouched and first-occurrence order is kept -- config-dependent headers
    need it.
    """
    seen: set[str] = set()
    depth = 0
    kept: list[str] = []
    for line in code.split("\n"):
        if PREPROCESSOR_IF.match(line):
            depth += 1
        elif PREPROCESSOR_ENDIF.match(line):
            depth = max(depth - 1, 0)
        elif depth == 0 and INCLUDE_LINE.match(line):
            key = CODE_ANNOTATION.sub("", line).strip()
            if key in seen:
                continue
            seen.add(key)
        kept.append(line)
    return "\n".join(kept)


# NOTE: not registered with the target registry -- it is selected explicitly by
# ``dace.codegen.codegen.generate_code`` when ``compiler.cpu.implementation`` is
# ``experimental_readable``. Registering it would cause double instantiation.
class ExperimentalCPUCodeGen(CPUCodeGen):
    """Human-readable CPU/GPU-kernel code generator (see module docstring)."""

    def __init__(self, frame, sdfg):
        super().__init__(frame, sdfg)
        # Helper name -> full C++ definition (deduplicated), for the ``<array>_idx`` index
        # functions and the ``<array>_size`` allocation-extent helpers respectively.
        self._index_functions: dict[str, str] = {}
        self._size_functions: dict[str, str] = {}
        # id(function_stream) -> helper names already flushed to THAT stream. Dedup is per stream
        # (= per output file), so an array used on both the host (.cpp) and a device kernel (.cu)
        # gets its definition in each file. Index and size helpers share the set (names never
        # collide: ``A_idx`` vs ``A_size``).
        self._emitted_functions: dict[int, set[str]] = {}
        # (base-name, signature) -> emitted function name, so one shape reuses a function while
        # two same-named arrays of different shape (e.g. a connector-derived ``_out`` that is 1-D
        # in one inlined SDFG and 2-D in another) get distinct helpers, not a colliding name with
        # wrong arity / strides. A plain name collision is disambiguated against _index_functions /
        # _size_functions (the emitted-name sets), so no separate name->sig maps are kept.
        self._index_sig_to_name: dict[tuple, str] = {}
        self._size_sig_to_name: dict[tuple, str] = {}
        # Container, symbol and constant names of the whole program, built on the first helper registration.
        self._program_sdfg = sdfg
        self._program_names: set[str] | None = None
        # Per-tasklet cache of identifiers appearing in the (rewritten) body.
        self._body_identifiers: dict[int, set[str]] = {}
        # Per-native-tasklet cache of identifiers the C++ body DECLARES (see _declared_identifiers).
        self._body_declarations: dict[int, set[str]] = {}
        # Per-native-tasklet cache: id(node) -> {connector_name: cpp_access}. An entry means the
        # connector is accessed directly (scalar index / base pointer) instead of via a copy-in/out.
        self._cpp_inline: dict[int, dict[str, str]] = {}
        # scalar_decl = late: a mutable scope-lifetime value scalar whose ``T x;`` declaration
        # was deferred by allocate_array is held here until the first tasklet that uses it is emitted
        # (see late_declarable_scalar / emit_pending_late_decls). ptrname -> declaration info. Empty in
        # the default ``eager`` mode, so the emit hook is a no-op and the output stays byte-identical.
        self._late_pending: dict[str, dict] = {}
        # Caches for the deferral gate (late_declarable_scalar), all keyed by id() of frozen codegen-
        # time objects. ``_eager_alloc_scopes`` is built once from the frame's allocation plan;
        # ``_node_references`` / ``_nested_free_names`` memoize the per-node and per-nested-SDFG name
        # sweeps, whose value is None for "not analysable" (which the gate reads as a refusal).
        self._eager_alloc_scopes: dict[tuple[int, str], list] | None = None
        self._name_owners: dict[int, dict[str, set[int]] | None] = {}
        self._node_references: dict[int, set[str] | None] = {}
        self._nested_free_names: dict[int, set[str] | None] = {}

    def emit_interstate_variable_declaration(self, name, dtype, callsite_stream, sdfg):
        """LoopRegion counters are declared inside their own ``for``-init clause in the readable
        generator (``for (T i = ...)``); only non-loop interstate symbols keep the hoisted declaration.
        """
        local_ctype = loop_local_counter_ctype(name, dtype, sdfg)
        if local_ctype is not None:
            self._frame.loop_local_counters[(sdfg.cfg_id, name)] = local_ctype
            self._frame.dispatcher.defined_vars.add(name, DefinedType.Scalar, local_ctype)
            return
        super().emit_interstate_variable_declaration(name, dtype, callsite_stream, sdfg)

    def get_generated_codeobjects(self):
        # A split-nest / external translation unit assembles its global code the way the frame does --
        # a re-emitted file header plus one write per tasklet ``code_global`` -- so a header named by
        # two tasklets of the same nest arrives twice. The frame's own file is deduplicated where it
        # is assembled (framecode.py, generate_code).
        objects = super().get_generated_codeobjects()
        for obj in objects:
            obj.code = deduplicate_includes(obj.code)
        return objects

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

    def _used_identifiers(self, node) -> set[str]:
        key = id(node)
        cached = self._body_identifiers.get(key)
        if cached is not None:
            return cached
        code = node.code.as_string if node.code else ""
        if node.language == dtypes.Language.Python:
            # ``as_string`` unparses the tasklet's already-parsed AST, so it is always valid Python.
            ids = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
        else:
            ids = set(IDENTIFIER_TOKENS.findall(code))
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

    def _generate_AITasklet(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen=None):
        # The classic layout keeps the markers AI diagnostics attribute errors by; its body still indexes
        # through ``<array>_idx`` helpers, so they are flushed the same way.
        super()._generate_AITasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen)
        self._flush_generated_functions(function_stream, cfg, state_id, node)

    # -- readable tasklet body: single line, no separator noise ----------------

    def tasklet_body_comment(self, node) -> str:
        # No ``// Tasklet code (label)`` banner: each emitted tasklet line already
        # carries a trailing ``// <label>`` (see emit_tasklet_body_block).
        return ""

    def tasklet_body_open_marker(self, node) -> str:
        # No visual separators; the body reads directly as C++.
        return ""

    def tasklet_body_close_marker(self, node) -> str:
        return ""

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
            line = re.sub(r"[ \t]*////__(DACE:|CODEGEN;).*", "", inner_body).strip()
            if line:
                # This is the ONLY point that knows both that the tasklet is emitted brace-free at the
                # enclosing scope AND what its statement reads, which is exactly what a fused
                # declaration needs: `T x = expr;` is only in scope for later readers if this line is
                # not wrapped in a brace of its own.
                line = self.fuse_pending_decl(node, line)
                self.emit_pending_late_decls(cfg, state_id, node, callsite_stream)
                callsite_stream.write("%s  // %s\n" % (line, node.label), cfg, state_id, node)
                return
        # Braced (or multi-statement) body: a declaration folded into it would be scoped to that brace,
        # so any pending declaration is emitted as its own line ahead of the block instead.
        self.emit_pending_late_decls(cfg, state_id, node, callsite_stream)
        callsite_stream.write("{  // %s" % node.label, cfg, state_id, node)
        callsite_stream.write(inner_body, cfg, state_id, node)
        callsite_stream.write(postamble)
        callsite_stream.write("}", cfg, state_id, node)

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
        out: list[str] = []
        for tok_type, value in CppLexer().get_tokens(body):
            if tok_type in Token.Name and value in inline:
                out.append(inline[value])
            else:
                out.append(value)
        return "".join(out)

    def _compute_cpp_inline(self, sdfg, state_dfg, node) -> dict[str, str]:
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
        if re.search(r"(?m)^[ \t]*#", node.code.as_string):
            return {}
        in_map: dict[str, str] = {}
        out_map: dict[str, str] = {}
        in_edges: dict[str, object] = {}
        out_edges: dict[str, object] = {}
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
        inline: dict[str, str] = {}
        for name in dict.fromkeys((*in_map, *out_map)):
            if name in inout:
                if name in in_map and name in out_map and in_map[name] == out_map[name]:
                    inline[name] = in_map[name]
                # else: keep the connector for both sides
            else:
                inline[name] = in_map.get(name, out_map.get(name))

        self._drop_captured_inlines(node, inline)

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

    def _drop_captured_inlines(self, node, inline: dict[str, str]) -> None:
        """
        Removes from ``inline`` every connector whose access text would be CAPTURED by the tasklet
        body, i.e. whose access text names an identifier the body DECLARES as a local.

        The access text is spliced into a raw C++ body that this generator does not parse, so it lands
        inside whatever scope the body declares. If the body declares that identifier (``tblis_tensor
        A, B, C;`` in the TBLIS ``TensorDot`` expansion, with the contracted array also named ``A``),
        the inlined name binds to the body's local instead of the array -- a wrong program, not a
        compile error, whenever the local happens to have a compatible type.

        Only a *declaration* shadows. A body that merely READS a name the access text also names is
        reading the very same variable -- the enclosing map parameter in ``arr[arr_idx(i)]`` spliced
        into ``_out = (i < 2) ? 0.0 : _inp;`` is the loop's own ``i`` -- so refusing on any occurrence
        would refuse almost every scalar connector under a map.

        Connectors that stay inlined vanish from the body, so they cannot capture anything; dropping
        one puts its name back into the body, hence the fixpoint.
        """
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

    def _declared_identifiers(self, node) -> set[str]:
        """
        Names the C++ tasklet body declares as its own, over-approximated from the pygments token
        stream: an identifier is taken as declared when the previous significant token is a keyword
        (``int``, ``auto``, ``const``, ...) or another identifier (a user type -- ``tblis_tensor A``),
        or when it continues that declarator's comma list (``, B, C``). ``DECLARATOR_TOKENS`` are
        skipped over rather than ended on, so ``double *p`` and ``std::vector<double> v`` still read
        as declarations of ``p`` and ``v``.

        C++ cannot be disambiguated without a parser, so ``a * b`` and ``a > b`` also read as
        declarations. Erring towards "declared" only costs a refusal (the classic connector
        copy-in/out, always correct); missing a declaration costs a miscompile.
        """
        key = id(node)
        cached = self._body_declarations.get(key)
        if cached is not None:
            return cached
        declared: set[str] = set()
        previous = None  # last significant token, with the declarator punctuation skipped
        in_declarator_list = False
        for token_type, value in CppLexer().get_tokens(node.code.as_string):
            if token_type in Token.Text or token_type in Token.Comment or value in DECLARATOR_TOKENS:
                continue
            if token_type in Token.Name:
                if previous is not None and (
                    previous[0] in Token.Keyword
                    or previous[0] in Token.Name
                    or (in_declarator_list and previous[1] == ",")
                ):
                    declared.add(value)
                    in_declarator_list = True
                else:
                    in_declarator_list = False
            elif value != ",":
                in_declarator_list = False
            previous = (token_type, value)
        self._body_declarations[key] = declared
        return declared

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

    def _cpp_connector_access(self, sdfg, state_dfg, node, edge, is_output: bool) -> str | None:
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
        # (mirrors InlineTaskletConnectors and PromoteConstantTransients). An ArrayView IS a
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
                return parenthesized_ptr(cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self))
            # Fallback: base pointer + offset to the subset start.
            offset = cpp.cpp_offset_expr(desc, subset)
            return ptrname if offset == "0" else parenthesized_ptr("%s + %s" % (ptrname, offset))

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

    def allocate_array(
        self,
        sdfg,
        cfg,
        dfg,
        state_id,
        node,
        nodedesc,
        function_stream,
        declaration_stream,
        allocation_stream,
        allocate_nested_data: bool = True,
    ):
        # scalar_decl = late: a MUTABLE scope-lifetime value scalar whose declaration is provably
        # safe to move keeps its `T x;` out of the scope
        # preamble; it is registered now (so reads resolve) and the declaration is emitted just before
        # its first-use tasklet (emit_pending_late_decls). Eager mode / any non-deferrable scalar falls
        # through to the base declaration below unchanged.
        if self.defer_scalar_declaration(sdfg, dfg, node, nodedesc):
            return
        super().allocate_array(
            sdfg,
            cfg,
            dfg,
            state_id,
            node,
            nodedesc,
            function_stream,
            declaration_stream,
            allocation_stream,
            allocate_nested_data,
        )
        # heap_alloc_stmt (invoked inside super) may have registered an
        # <array>_size() helper for the allocation count; flush it to the function
        # stream, mirroring the _idx flush in _generate_Tasklet.
        self._flush_generated_functions(function_stream, cfg, state_id, node)

    def fused_heap_declarator(
        self, sdfg, name: str, nodedesc: dt.Data, arrsize, declared: bool, declaration_stream, allocation_stream
    ) -> str | None:
        """Declarator fusing ``T *p;`` + ``p = new T[...];`` into one ``T* __restrict__ p = new
        T[...];`` definition, or None to keep the classic split pair.

        DaCe deliberately routes the declaration and the allocation to two streams so a transient's
        DECLARATION can be hoisted to an outer scope while its ALLOCATION stays in an inner one.
        Fusing is a purely textual merge of two writes, so it is sound only when both land in the
        same scope with nothing in between. Both of these must hold:

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

        Registration is untouched: the caller still runs ``define_var(...)`` after this, so
        ``defined_vars`` (and ``declared_arrays``, which only ``declare_array`` populates) resolve
        later accesses exactly as before.
        """
        if declared or declaration_stream is not allocation_stream:
            return None
        return self.array_pointer_declarator(name, nodedesc)

    def array_pointer_declarator(self, name: str, nodedesc: dt.Data) -> str:
        """``T* __restrict__ name`` for a heap-allocated array pointer.

        ``__restrict__`` is dropped for a ``may_alias`` descriptor -- the same condition
        ``Array.as_arg`` and ``CPUCodeGen.generate_nsdfg_arguments`` already use to decide the
        qualifier on kernel/nested-SDFG arguments. ``may_alias`` marks an array that is deliberately
        reachable through a second pointer in the same function, so promising no-alias for it would
        be a miscompile rather than a readability win.
        """
        restrict = "" if nodedesc.may_alias else "__restrict__ "
        return "%s* %s%s" % (nodedesc.dtype.ctype, restrict, name)

    def heap_alloc_stmt(
        self,
        alloc_name: str,
        ctype: str,
        arrsize: str,
        alignment: int = 0,
        sdfg: Optional["SDFG"] = None,
        nodedesc: dt.Data | None = None,
        data_name: str | None = None,
    ) -> str:
        # The base aligned ``operator new[]``, with the count through an ``<array>_size`` helper when one
        # is worthwhile (_register_size_function).
        count = arrsize
        if sdfg is not None and nodedesc is not None and data_name is not None:
            registered = self._register_size_function(data_name, nodedesc)
            if registered is not None:
                fnname, call_args = registered
                count = "%s(%s)" % (fnname, ", ".join(call_args))
        return super().heap_alloc_stmt(alloc_name, ctype, count, alignment, sdfg, nodedesc, data_name)

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
        # A device (.cu) file has the same shape: the delegating GPU codegen sets ``calling_codegen`` to itself and
        # feeds MANY streams (one per kernel/nested-SDFG scope, plus the file-scope global stream) into the single
        # ``<name>_cuda.cu`` translation unit it owns. Keying on ``id(function_stream)`` there re-emits an identical
        # ``<name>_idx`` per stream -> a C++ redefinition. ``ExperimentalCUDACodeGen`` builds exactly one CodeObject
        # (.cu), so key device emission on ``id(self.calling_codegen)`` -- one key for the whole .cu, each helper
        # emitted once.
        file_key = id(self) if self.calling_codegen is self else id(self.calling_codegen)
        emitted = self._emitted_functions.setdefault(file_key, set())
        for registry in (self._index_functions, self._size_functions):
            for name, defn in registry.items():
                if name in emitted:
                    continue
                function_stream.write(defn + "\n", cfg, state_id, node)
                emitted.add(name)

    # -- readable array indexing ----------------------------------------------

    # -- late scalar declarations (scalar_decl = late / late_fused) ------

    def defer_scalar_declaration(self, sdfg, dfg, node, desc) -> bool:
        """Register a mutable scope scalar as a deferred (late) declaration and return True, or return
        False to leave it to the base eager declaration.

        Deferral moves ``T x;`` out of the scope preamble and re-emits it just before the first tasklet
        that uses ``x``, giving a shorter live range. It is applied ONLY where provably sound: an eager
        ``T x;`` and a deferred one are visible to exactly the same uses only when every use lives in the
        one scope the deferred declaration lands in, as a direct-child tasklet of it. See
        ``late_declarable_scalar`` for the full gate. Returns False in the default ``eager`` mode."""
        placement = self.late_declarable_scalar(sdfg, dfg, node, desc)
        if placement is None:
            return False
        scope_tasklets, setzero = placement
        ptrname = self.ptr(node.data, desc, sdfg)
        # Register the read/write type now so accesses between here and the deferred declaration resolve
        # exactly as the eager declaration would (there are none in emission order -- the scope preamble
        # precedes every tasklet -- but the registration must exist before the first use is emitted).
        self._dispatcher.defined_vars.add(ptrname, DefinedType.Scalar, desc.dtype.ctype)
        state: SDFGState = dfg
        # Tasklets that WRITE the scalar: only one of those can carry a fused ``T x = expr;`` binding.
        # A ``setzero`` scalar is never fused -- its ``T x = 0;`` init is the declaration already.
        writers = {id(edge.src) for an in state.data_nodes() if an.data == node.data for edge in state.in_edges(an)}
        self._late_pending[ptrname] = {
            "ctype": desc.dtype.ctype,
            "ptrname": ptrname,
            "setzero": setzero,
            "tasklets": {id(t) for t in scope_tasklets},
            "writers": writers,
            "fusable": scalar_decl() == "late_fused" and not setzero,
        }
        return True

    def late_declarable_scalar(self, sdfg, dfg, node, desc) -> tuple[set[nodes.Tasklet], bool] | None:
        """Whether ``desc`` (the scalar being allocated at ``node`` in state ``dfg``) may have its
        declaration deferred to first use, per ``scalar_decl``. Returns ``(neighbor_tasklets,
        setzero)`` when deferrable, else ``None`` (keep the eager declaration).

        Sound only for a plain transient VALUE scalar (never a view/reference/stream, a const write-once
        scalar, a len-1 array, or a heap/device/GPU-global scalar) whose EVERY access in the state is a
        direct-child tasklet of ONE scope. That single-scope, tasklet-only shape guarantees the
        first-use tasklet is emitted into the same brace the eager declaration would occupy, so the
        deferred ``T x;`` precedes -- and is visible to -- every use. Anything else falls back to eager.

        The two obligations at the end are stated as REFUSALS over all uses rather than as searches for
        a known use shape, because a search only ever finds what it was written to look for:

        1. the scope the frame's allocation planner picked for the eager declaration must be exactly the
           scope this deferral lands in (``eager_allocation_scope``). Any use anywhere in the SDFG that
           pushes that scope outwards -- another state, an interstate edge, a loop or branch condition, a
           code node naming the container as a free symbol -- makes the scopes differ and refuses here,
           without this method enumerating those shapes at all;
        2. nothing anywhere in the SDFG but those access nodes and their neighbour tasklets may mention
           the name at all (``name_owners``), since the deferred declaration sits at the FIRST of those
           tasklets rather than at the top of the brace, so even a use inside the same brace can precede
           it. That check starts from every mention of the name and proves the mentions are the expected
           ones, so a use shape nobody anticipated lands outside the permitted set and refuses."""
        # ``late`` re-emits the skipped ``T x;`` at first use; ``late_fused`` folds it into the first write, and
        # falls back to ``late`` for a candidate it cannot fuse (a braced first-use tasklet).
        if scalar_decl() == "eager":
            return None
        # A plain mutable value scalar only. GPU-global scalars are device pointers, and heap/persistent
        # ones do not live in a plain brace.
        if not isinstance(desc, dt.Scalar) or isinstance(desc, (dt.View, dt.Reference, dt.Stream)):
            return None
        if not desc.transient or node.data in sdfg.constants_prop:
            return None
        if desc.lifetime != dtypes.AllocationLifetime.Scope:
            return None
        if desc.storage not in (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap):
            return None
        if not isinstance(dfg, SDFGState):
            return None
        state: SDFGState = dfg
        name = node.data
        access_nodes = [n for n in state.data_nodes() if n.data == name]
        if not access_nodes:
            return None
        scope_dict = state.scope_dict()
        # Every access must live in ONE scope -- state top level (``None``) or one map's body. That is
        # the brace the eager declaration occupies too: a map's scope transients are declared inside its
        # innermost loop body, exactly where its tasklets are emitted, so a declaration deferred within
        # that scope lands in the same brace and is re-entered per iteration exactly as the eager one is.
        # A scalar spread over two scopes has no single brace that dominates every use.
        scopes = {scope_dict[n] for n in access_nodes}
        if len(scopes) != 1:
            return None
        scope = scopes.pop()
        neighbor_tasklets: set[nodes.Tasklet] = set()
        has_write = False
        for an in access_nodes:
            if state.in_degree(an) > 0:
                has_write = True
            elif state.out_degree(an) > 0:
                # Read from a source this state never wrote: the value comes from a PREVIOUS execution
                # of the state. An eager declaration at scope top persists across executions and carries
                # it; a deferred one inside the state's brace is re-declared each time the state runs
                # (the brace sits inside the loop), so the carried value would silently become
                # uninitialized. Not a deferral target.
                return None
            for edge in state.all_edges(an):
                other = edge.dst if edge.src is an else edge.src
                # Every reader/writer must be a plain tasklet in the SAME scope; a copy / map / nested
                # SDFG / library-node neighbour is not the single-brace direct-child shape we require.
                if not isinstance(other, nodes.Tasklet) or scope_dict[other] is not scope:
                    return None
                neighbor_tasklets.add(other)
        # A mutable transient must be written before it is read; without a write there is no first-use
        # point that dominates the reads, so it is not a safe deferral target.
        if not has_write or not neighbor_tasklets:
            return None
        # Obligation 1: the eager declaration's scope must BE the scope this deferral lands in. Any use
        # outside it -- in any shape, anywhere in the SDFG -- has already moved the planner's answer.
        if self.eager_allocation_scope(sdfg, name) is not (state if scope is None else scope):
            return None
        # Obligation 2: NOTHING in the SDFG other than those accesses and their neighbour tasklets may
        # mention the name. The deferred declaration sits at the first of those tasklets rather than at
        # the top of the brace, so even a use inside the same scope can precede it.
        owners = self.name_owners(sdfg)
        if owners is None:
            return None
        permitted = {id(n) for n in access_nodes} | {id(t) for t in neighbor_tasklets}
        if not owners.get(name, frozenset()) <= permitted:
            return None
        return neighbor_tasklets, node.setzero

    def eager_allocation_scope(self, sdfg: SDFG, name: str) -> nodes.EntryNode | SDFGState | SDFG | None:
        """The scope the frame's allocation planner picked for ``name``'s EAGER declaration in ``sdfg``
        (a ``MapEntry``, an ``SDFGState`` or an ``SDFG``), or ``None`` when it recorded no entry -- or
        more than one, a multi-site shape this generator does not model.

        This is where the deferral gate inverts. ``determine_allocation_lifetime`` has already walked
        every use of every container -- access nodes, code nodes naming it as a free symbol, interstate
        edges, loop and conditional-block conditions -- to pick the innermost scope that dominates them
        all. Re-deriving that here as a list of "can I see a use?" scans means every use shape this file
        failed to anticipate reads as "no use found", which is a silent miss. Asking the planner instead
        makes the deferral's obligation exactly what it claims to be -- an eager ``T x;`` and a deferred
        one are interchangeable -- and turns an unanticipated use into a scope mismatch, i.e. a refusal.
        """
        table = self._eager_alloc_scopes
        if table is None:
            table = {}
            # (SDFG, container) -> the to_allocate keys that mention it. Built once: the plan is fixed
            # for the whole code generation run.
            for alloc_scope, entries in self._frame.to_allocate.items():
                for tsdfg, _, alloc_node, _, _, _ in entries:
                    table.setdefault((id(tsdfg), alloc_node.data), []).append(alloc_scope)
            self._eager_alloc_scopes = table
        found = table.get((id(sdfg), name))
        if found is None or len(found) != 1:
            return None
        return found[0]

    def name_owners(self, sdfg: SDFG) -> dict[str, set[int]] | None:
        """Every name mentioned anywhere in ``sdfg`` mapped to the ``id``s of the dataflow nodes that
        may mention it, or ``None`` if some node could not be analysed (which refuses every candidate in
        that SDFG). Built once per SDFG: the graph is frozen for the whole code generation run.

        This is the "every use of this name" side of the deferral gate. A memlet's names are charged to
        BOTH endpoints, so an edge is covered exactly when its two nodes are; a name on an interstate
        edge or in a loop / branch condition is charged to the SDFG itself, an owner no candidate can
        ever permit, so it always refuses. Names are read out with a crude tokenizer rather than with
        ``used_symbols``, which filters to registered SYMBOLS and so cannot see a data container named
        as a free name in a C++ tasklet body -- the shape that motivated this rewrite.
        """
        key = id(sdfg)
        if key in self._name_owners:
            return self._name_owners[key]
        owners: dict[str, set[int]] = {}

        def charge(names, owner_id: int) -> None:
            for used in names:
                owners.setdefault(used, set()).add(owner_id)

        result: dict[str, set[int]] | None = owners
        for state in sdfg.states():
            for graph_node in state.nodes():
                references = self.node_name_references(graph_node, sdfg)
                if references is None:
                    result = None
                    break
                charge(references, id(graph_node))
            if result is None:
                break
            for edge in state.edges():
                names = set(edge.data.free_symbols)
                if edge.data.data:
                    names.add(edge.data.data)
                charge(names, id(edge.src))
                charge(names, id(edge.dst))
        if result is not None:
            for edge in sdfg.all_interstate_edges():
                charge(edge.data.read_symbols() | set(edge.data.assignments.keys()), key)
            for block in sdfg.all_control_flow_blocks():
                charge(self.code_property_names(block), key)
            for region in sdfg.all_control_flow_regions():
                # ``get_meta_codeblocks`` is the region class's own answer for loop control / branch
                # conditions, which a ConditionalBlock keeps in ``_branches``, not in a property.
                charge(identifiers_in(region.get_meta_codeblocks()), key)
        self._name_owners[key] = result
        return result

    def node_name_references(self, graph_node: nodes.Node, sdfg: SDFG) -> set[str] | None:
        """Every name the C++ lowered from ``graph_node`` may reference, or ``None`` when that cannot be
        decided. Property-driven rather than class-driven, so a node holding its body in a code property
        this file never heard of is still covered; a node whose body is NOT in a property it can read
        returns ``None`` and refuses the candidate rather than under-reporting its references."""
        key = id(graph_node)
        if key in self._node_references:
            return self._node_references[key]
        if isinstance(graph_node, nodes.AccessNode):
            references: set[str] | None = {graph_node.data, graph_node.root_data}
        elif isinstance(graph_node, nodes.NestedSDFG):
            # A nested SDFG is emitted as an inline block, so a name it does not define itself resolves
            # to the ENCLOSING scope's C++ -- exactly like a free name in a tasklet body. ``None`` (an
            # unloaded external nest) propagates as a refusal.
            nested = self.nested_free_names(graph_node.sdfg) if graph_node.sdfg is not None else None
            references = (
                None if nested is None else set(graph_node.free_symbols) | self.code_property_names(graph_node) | nested
            )
        elif isinstance(graph_node, nodes.Tasklet):
            # RTLTasklet included: ``_used_identifiers`` tokenizes ``node.code`` for any tasklet language.
            references = (
                set(graph_node.free_symbols) | self.code_property_names(graph_node) | self._used_identifiers(graph_node)
            )
        elif isinstance(graph_node, nodes.CodeNode):
            # A library node that survived to codegen lowers through its own ``generate_code`` -- code
            # this file cannot read from a property -- so its references are unknown and it refuses.
            references = None
        else:
            # Scope/other nodes (Map/Consume entry & exit, ...) carry no body: their only names are the
            # symbols in their properties (map ranges etc.), which ``free_symbols`` reports in full.
            references = set(graph_node.free_symbols)
        self._node_references[key] = references
        return references

    def nested_free_names(self, nested: SDFG) -> set[str] | None:
        """Names used anywhere inside ``nested`` that it does not define itself, so they resolve to the
        enclosing scope. ``None`` if any node in it could not be analysed."""
        key = id(nested)
        if key in self._nested_free_names:
            return self._nested_free_names[key]
        names: set[str] = set()
        result: set[str] | None = names
        for inner_state in nested.states():
            for graph_node in inner_state.nodes():
                references = self.node_name_references(graph_node, nested)
                if references is None:
                    result = None
                    break
                names |= references
            if result is None:
                break
            for edge in inner_state.edges():
                names |= edge.data.free_symbols
                if edge.data.data:
                    names.add(edge.data.data)
        if result is not None:
            for edge in nested.all_interstate_edges():
                names |= edge.data.read_symbols() | set(edge.data.assignments.keys())
            for block in nested.all_control_flow_blocks():
                names |= self.code_property_names(block)
            for region in nested.all_control_flow_regions():
                names |= identifiers_in(region.get_meta_codeblocks())
            # A name the nest defines itself is declared inside its own block and shadows the outer one.
            result = names - (nested.arrays.keys() | nested.symbols.keys() | nested.constants_prop.keys())
        self._nested_free_names[key] = result
        return result

    def code_property_names(self, holder) -> set[str]:
        """Identifier tokens of every ``CodeBlock`` property of ``holder`` (a node or a control-flow
        block): the loop init / condition / update of a ``LoopRegion``, a library node's code, a
        tasklet's ``code_init`` / ``code_exit``, and whatever a future class stores the same way."""
        names: set[str] = set()
        for _, value in holder.properties():
            names |= identifiers_in(code_blocks_of(value))
        return names

    def fuse_pending_decl(self, tasklet, line: str) -> str:
        """Fold a pending declaration into ``line``, the brace-free single statement of ``tasklet``,
        turning ``x = expr;`` into ``T x = expr;`` -- the declaration IS the first write. Returns the
        line unchanged (leaving the declaration pending, to be emitted on its own) unless
        ``scalar_decl`` is ``late_fused`` and this line is genuinely that scalar's first write.

        The write must be spelled by this line: the scalar is deferred on the strength of its DATAFLOW
        (see ``late_declarable_scalar``), but only the emitted text proves the statement really is
        ``x = ...`` and not, say, a read of ``x`` feeding another store."""
        for ptrname, info in list(self._late_pending.items()):
            if not info["fusable"] or id(tasklet) not in info["writers"]:
                continue
            if not re.match(r"%s\s*=[^=]" % re.escape(ptrname), line):
                continue
            del self._late_pending[ptrname]
            return "%s %s" % (info["ctype"], line)
        return line

    def emit_pending_late_decls(self, cfg, state_id, tasklet, callsite_stream) -> None:
        """Emit the deferred ``T x;`` declaration of every scalar this ``tasklet`` is the first to use,
        immediately before the tasklet body. A no-op (and byte-identical) unless a declaration was
        deferred (``scalar_decl`` other than ``eager``)."""
        if not self._late_pending:
            return
        for ptrname, info in list(self._late_pending.items()):
            if id(tasklet) not in info["tasklets"]:
                continue
            zero = " = 0" if info["setzero"] else ""
            callsite_stream.write("%s %s%s;" % (info["ctype"], info["ptrname"], zero), cfg, state_id, tasklet)
            del self._late_pending[ptrname]

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

    def unshadowed_helper_name(self, fnname: str, registry: dict[str, str]) -> str:
        """``fnname``, or a fresh name when a container, symbol or constant of the program is called ``fnname``:
        helpers are file-scope functions, so a parameter or local of the same name shadows them at every call."""
        if self._program_names is None:
            self._program_names = {
                name
                for sd in self._program_sdfg.all_sdfgs_recursive()
                for names in (sd.arrays, sd.symbols, sd.constants)
                for name in names
            }
        if fnname not in self._program_names:
            return fnname
        return find_new_name(fnname, self._program_names | registry.keys())

    def _register_index_function(self, data_name: str, desc):
        """Registers (once per distinct descriptor signature) the ``<name>_idx`` index function for
        ``data_name`` and returns ``(function_name, extra_symbol_names)``; a same-named array of
        different shape/strides is given a disambiguated name so arity and offset math still match."""
        ndim = len(desc.shape)
        dim_syms = [symbolic.symbol("__d%d" % i) for i in range(ndim)]
        strides = [symbolic.pystr_to_symbolic(str(s)) for s in desc.strides]
        offset = [symbolic.pystr_to_symbolic(str(o)) for o in desc.offset]
        flat = sum(dim_syms[i] * strides[i] for i in range(ndim))
        const = sum(offset[i] * strides[i] for i in range(ndim))
        flatexpr = flat + const
        extra = sorted((flatexpr.free_symbols - set(dim_syms)), key=lambda s: str(s))
        extra_names = [str(s) for s in extra]

        base = re.sub(r"\W", "_", data_name)
        ctype = index_ctype(desc, flatexpr)
        sig = (ndim, tuple(str(s) for s in desc.strides), tuple(str(o) for o in desc.offset), ctype)
        key = (base, sig)
        if key in self._index_sig_to_name:
            return self._index_sig_to_name[key], extra_names

        fnname = base + "_idx"
        # A same-name/different-signature collision (the exact-match case returned above): the base
        # name is already an emitted definition, so disambiguate.
        if fnname in self._index_functions:
            fnname = "%s_%dd_%d_idx" % (base, ndim, len(self._index_sig_to_name))
        fnname = self.unshadowed_helper_name(fnname, self._index_functions)
        self._index_sig_to_name[key] = fnname

        params = ["%s %s" % (ctype, str(d)) for d in dim_syms]
        params += ["%s %s" % (ctype, s) for s in extra_names]
        body = sym2cpp(flatexpr)
        self._index_functions[fnname] = "%s %s %s(%s) { return %s; }" % (
            INDEX_FUNCTION_QUALIFIER,
            ctype,
            fnname,
            ", ".join(params),
            body,
        )
        return fnname, extra_names

    # -- readable array size --------------------------------------------------

    def _register_size_function(self, data_name: str, desc) -> tuple[str, list[str]] | None:
        """Registers (once per distinct name + size expression) the ``<array>_size`` helper for
        ``desc.total_size`` and returns ``(function_name, call_args)``, or ``None`` when not
        worthwhile.

        Readability threshold: only a constant or compound symbolic ``total_size`` gets a helper -- a
        bare single symbol (``N``) is skipped (``A_size(N){return N;}`` is no win). A constant folds to
        a nullary ``consteval`` helper; a compound symbolic size gets a ``constexpr`` helper over its
        sorted free symbols (``A_size(N, M)``).

        A DATA-DEPENDENT size is never hoisted. The helper is a free-standing static function, so its
        body may reference nothing but its parameters -- and those are the expression's free SYMBOLS.
        A subscripted data access (spmv's CSR row length ``A_indptr[i + 1] - A_indptr[i]``) carries its
        container as a ``Subscript`` head rather than a free symbol, so the container is never in the
        parameter list while the body still names it: ``__tmp0_size(int64_t i) { return -A_indptr[i] +
        A_indptr[i + 1]; }`` -> "``A_indptr`` was not declared in this scope". Fall back to the inline
        ``sym2cpp(total_size)`` extent, which is emitted at the allocation site where the container's
        pointer IS in scope.
        """
        total = symbolic.pystr_to_symbolic(str(desc.total_size))
        # A bare single symbol carries no readability benefit; keep the plain name.
        if total.is_Symbol:
            return None
        # Data-dependent extent: not expressible as a free-standing helper (see above).
        if symbolic.arrays(total):
            return None
        free = sorted(total.free_symbols, key=lambda s: str(s))
        call_args = [str(s) for s in free]
        is_constant = len(free) == 0

        base = re.sub(r"\W", "_", data_name)
        sig = (str(total), tuple(call_args))
        key = (base, sig)
        if key in self._size_sig_to_name:
            return self._size_sig_to_name[key], call_args

        fnname = base + "_size"
        # A same-name/different-signature collision (exact-match returned above): the base name is
        # already an emitted definition, so disambiguate.
        if fnname in self._size_functions:
            fnname = "%s_%d_size" % (base, len(self._size_sig_to_name))
        fnname = self.unshadowed_helper_name(fnname, self._size_functions)
        self._size_sig_to_name[key] = fnname

        qualifier = size_qualifier(is_constant)
        ctype = "int64_t"
        params = ["%s %s" % (ctype, s) for s in call_args]
        body = sym2cpp(total)
        self._size_functions[fnname] = "%s %s %s(%s) { return %s; }" % (
            qualifier,
            ctype,
            fnname,
            ", ".join(params),
            body,
        )
        return fnname, call_args


class NestedMultiDimSubscriptLowerer(ast.NodeTransformer):
    """Rewrites a nested multi-dim array subscript (``idx[i, j]``) found inside an index
    expression to its ``<array>_idx`` C++ text, via the same ``_bare_access`` path a top-level
    subscript uses. A rank>=2 nested subscript's ``Tuple`` slice reparses as a Python
    ``ast.Tuple`` downstream (``format_index_access`` -> ``sym2cpp`` -> ``pystr_to_symbolic``)
    and corrupts to ``std::make_tuple`` on a raw pointer; a rank-1 nested subscript (``idx[i]``)
    already round-trips as valid C++ and is left untouched."""

    def __init__(self, remover: "ReadableKeywordRemover"):
        self.remover = remover

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.slice, ast.Tuple) and self.remover._is_bare_data(rname(node)):
            access = self.remover._bare_access(node)
            if access is not None:
                return ast.copy_location(ast.Name(id=access), node)
        return self.generic_visit(node)


class ReadableKeywordRemover(cpp.DaCeKeywordRemover):
    """
    Extends the classic keyword remover: in addition to connector accesses, it
    lowers direct array accesses (``A[i, j]`` for ``A`` in ``sdfg.arrays``, as
    produced by InlineTaskletConnectors) to ``A[A_idx(i, j, ...)]``.
    """

    def __init__(self, sdfg, memlets, constants, codegen):
        super().__init__(sdfg, memlets, constants, codegen)
        #: Operand text -> its dtype, for the statements this class renders itself (below) instead of
        #: leaving to ``unparse_tasklet``'s unparser, which is handed the same thing as
        #: ``defined_symbols``. A surviving connector keeps its name and its declared dtype; an
        #: inlined one is gone from the body, so its access text stands in for it. The C++ printer
        #: needs this to type a bare numeric literal against a ``dace::float16`` operand
        #: (``cppunparse.CPPUnparser.dispatch_operand``).
        self.operand_dtypes: dict[str, dtypes.typeclass] = {
            conn: entry[3] for conn, entry in memlets.items() if conn is not None
        }

    def _is_bare_data(self, name: str) -> bool:
        return name not in self.memlets and name not in self.constants and name in self.sdfg.arrays

    def _scalar_constant_name(self, name: str) -> str | None:
        """``name`` if it is a 0-dimensional (scalar) SDFG constant, else None. PromoteConstantTransients promotes a
        literal-only scalar transient to such a constant, which framecode emits as a bare ``constexpr T
        name = v;`` (not ``T name[1]``). A subscript ``name[0]`` on it must lower to the bare ``name`` --
        routing it through the classic ``_subscript_expr`` trips on the scalar's empty (``()``) stride
        list (``Missing dimensions in expression (expected one, got 0)``)."""
        if name in self.constants and numpy.ndim(self.constants[name]) == 0:
            return name
        return None

    def _bare_access(self, node: ast.AST) -> str | None:
        """C++ access string for a direct (inlined) access to an SDFG array."""
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
        rhs = cppunparse.cppunparse(value, expr_semicolon=False, defined_symbols=self.operand_dtypes)
        plain = "%s = %s;" % (lhs, rhs)
        return self._replace_assignment(ast.Name(id=plain), node)

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
        self.operand_dtypes[access] = self.sdfg.arrays[target].dtype
        return ast.copy_location(ast.Name(id=access), node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # A bare data name that is a by-value scalar -> emit the plain name.
        name = rname(node)
        if self._is_bare_data(name):
            desc = self.sdfg.arrays[name]
            ptrname = self.codegen.ptr(name, desc, self.sdfg)
            if self.codegen._is_value_scalar(ptrname, desc):
                self.operand_dtypes[ptrname] = desc.dtype
                return ast.copy_location(ast.Name(id=ptrname), node)
        return super().visit_Name(node)

    def _index_list(self, slicenode: ast.AST) -> list[str]:
        lowerer = NestedMultiDimSubscriptLowerer(self)
        return [ast.unparse(lowerer.visit(astutils.copy_tree(e))) for e in index_expr_nodes(slicenode)]
