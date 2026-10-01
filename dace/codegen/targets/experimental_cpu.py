# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Readable CPU code generator, selected by ``compiler.cpu.implementation = experimental_readable``.

It differs from :class:`~dace.codegen.targets.cpu.CPUCodeGen` in how tasklets, array accesses and scopes are emitted.
Array offsets appear once per array as a generated ``<array>_idx(...)`` function. Tasklets whose connectors
``InlineTaskletConnectors`` inlined access arrays directly, without copy-in/out temporaries. A write-once transient
is bound ``const`` at its write. Scopes that bound no declaration are dropped. GPU kernels emit their tasklets through
the shared CPU instance, so they follow the same rules.
"""
import ast
import re

import numpy

from dace import data as dt
from dace import dtypes, symbolic
from dace.codegen import cppunparse
from dace.codegen.common import sym2cpp
from dace.codegen.control_flow import falls_through
from dace.codegen.dispatcher import DefinedType
from dace.codegen.targets import cpp
from dace.codegen.targets.const_bindings import find_const_bindings
from dace.codegen.targets.cpu import CPUCodeGen
from dace.config import Config
from dace.frontend.python import astutils
from dace.frontend.python.astutils import rname
from dace.sdfg import SDFG, infer_types, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.utils import dynamic_map_inputs
from dace.transformation.passes.inline_tasklet_connectors import (InlineTaskletConnectors, inlinable_connector,
                                                                  resolve_inout)
from dace.transformation.passes.promote_constant_transients import PromoteConstantTransients

INDEX_CTYPE = 'long long'
# Host and device callable, inlined, usable in constant expressions
HELPER_QUALIFIER = 'static DACE_HDFI constexpr'
# A constant extent folds at compile time (``consteval`` needs C++20)
CONSTANT_SIZE_QUALIFIER = 'static DACE_HDFI consteval'

# Comments and literals are matched so that identifiers inside them are left alone
CPP_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*', re.DOTALL)
PROVENANCE_TAG = re.compile(r'[ \t]*////__(DACE:|CODEGEN;).*')


def prepare_sdfg(sdfg: SDFG) -> None:
    """Lowers an SDFG with expanded library nodes for readable emission: flattens nested SDFGs, so that inlining
    applies uniformly, promotes literal-only transients to constants and inlines tasklet connectors."""
    sdutil.inline_sdfgs(sdfg)
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    PromoteConstantTransients().apply_pass(sdfg, {})
    InlineTaskletConnectors().apply_pass(sdfg, {})
    sdfg.validate()


class ReadableKeywordRemover(cpp.DaCeKeywordRemover):
    """ Lowers direct array accesses (``A[i, j]``, as inlined connectors produce) to ``A[A_idx(i, j)]``. """

    def is_bare_data(self, name: str) -> bool:
        return name not in self.memlets and name not in self.constants and name in self.sdfg.arrays

    def scalar_constant(self, name: str) -> bool:
        """A 0-dimensional SDFG constant, emitted as a bare ``constexpr T name = v;``."""
        return name in self.constants and numpy.ndim(self.constants[name]) == 0

    def bare_access(self, node: ast.AST) -> str | None:
        name = rname(node)
        indices = []
        if isinstance(node, ast.Subscript):
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            indices = [ast.unparse(e) for e in elements]
        return self.codegen.index_access(self.sdfg, self.sdfg.arrays[name], name, indices)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        target_node = node.targets[-1]
        target = rname(target_node)
        if not self.is_bare_data(target):
            return super().visit_Assign(node)
        lhs = self.bare_access(target_node)
        if lhs is None:
            return self.generic_visit(node)
        rhs = cppunparse.cppunparse(self.visit(astutils.copy_tree(node.value)), expr_semicolon=False)
        desc = self.sdfg.arrays[target]
        if target not in self.codegen.const_bindings.get(self.sdfg.cfg_id, ()):
            statement = f'{lhs} = {rhs};'
        elif isinstance(desc, dt.Scalar):
            statement = f'const {desc.dtype.ctype} {lhs} = {rhs};'
        else:
            # The cast keeps the implicit narrowing of an assignment, which a braced initializer would reject
            ctype = desc.dtype.ctype
            statement = f'const {ctype} {self.codegen.ptr(target, desc, self.sdfg)}[1] = {{({ctype})({rhs})}};'
        # A single Name keeps the C++ unparser from declaring the left-hand side as a new ``auto``
        return self._replace_assignment(ast.Name(id=statement), node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        target = rname(node)
        if self.scalar_constant(target):
            return ast.copy_location(ast.Name(id=target), node)
        if not self.is_bare_data(target):
            return super().visit_Subscript(node)
        access = self.bare_access(node)
        return self.generic_visit(node) if access is None else ast.copy_location(ast.Name(id=access), node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        name = rname(node)
        if self.is_bare_data(name):
            desc = self.sdfg.arrays[name]
            ptrname = self.codegen.ptr(name, desc, self.sdfg)
            if self.codegen.is_value_scalar(ptrname, desc):
                return ast.copy_location(ast.Name(id=ptrname), node)
        return super().visit_Name(node)


class ExperimentalCPUCodeGen(CPUCodeGen):
    """ Readable CPU and GPU-kernel code generator. """

    # Not registered with the target registry: ``generate_code`` selects it explicitly. Each tasklet line
    # carries its label as a trailing comment instead.
    tasklet_banners = False
    keyword_remover = ReadableKeywordRemover

    def __init__(self, frame, sdfg):
        super().__init__(frame, sdfg)
        # Definitions of the ``<array>_idx`` and ``<array>_size`` helpers by name, and the name given to each
        # (kind, array, signature), so one shape reuses a helper and two same-named arrays of different shape
        # (a connector-derived name in two inlined SDFGs) get distinct ones
        self.helpers: dict[str, str] = {}
        self.helper_names: dict[tuple, str] = {}
        # Helpers already written, per output file
        self.emitted_helpers: dict[int, set[str]] = {}
        self.body_identifiers: dict[int, set[str]] = {}
        # Inlined connectors of each native tasklet and their C++ accesses
        self.cpp_inline: dict[int, dict[str, str]] = {}
        # Names of write-once data per ``cfg_id``
        self.const_bindings: dict[int, set[str]] = {}

    def preprocess(self, sdfg: SDFG) -> None:
        self.const_bindings = find_const_bindings(sdfg, self.connector_needs_copy)

    def map_scope_needs_brace(self, sdfg: SDFG, state_dfg, node: nodes.MapEntry) -> bool:
        # Schedules this generator does not open itself (device) keep their braces
        if node.map.schedule not in (dtypes.ScheduleType.Sequential, dtypes.ScheduleType.CPU_Multicore):
            return True
        instrumented = node.map.instrument != dtypes.InstrumentationType.No_Instrumentation
        return instrumented or bool(dynamic_map_inputs(state_dfg, node))

    def state_needs_brace(self, state) -> bool:
        """A state is unbraced only if nothing in it declares at state level. A top-level node other than a map
        scope or a plain access node may declare (code-to-code temporaries, timers, nested-SDFG locals), as may a
        tracked transient. A jump in the parent region must not cross such a declaration."""
        if state.instrument != dtypes.InstrumentationType.No_Instrumentation or self._frame.to_allocate.get(state):
            return True
        if not falls_through(state.parent_graph):
            return True
        scope = state.scope_dict()
        for node in state.nodes():
            if scope[node] is not None:
                continue
            if isinstance(node, nodes.AccessNode):
                if node.instrument != dtypes.DataInstrumentationType.No_Instrumentation:
                    return True
            elif not isinstance(node, (nodes.MapEntry, nodes.MapExit)):
                return True
        return False

    def connector_needs_copy(self, node: nodes.CodeNode, conn: str) -> bool:
        """Whether a connector still needs a copy-in/out temporary: for a Python tasklet if its body still names
        it, for a native one unless it was inlined. Other code nodes and languages keep their connectors."""
        if not isinstance(node, nodes.Tasklet):
            return True
        if node.language == dtypes.Language.CPP:
            return conn not in self.cpp_inline.get(id(node), {})
        if node.language != dtypes.Language.Python:
            return True
        if id(node) not in self.body_identifiers:
            tree = ast.parse(node.code.as_string)
            self.body_identifiers[id(node)] = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return conn in self.body_identifiers[id(node)]

    def _generate_Tasklet(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen=None):
        # The base generator skips the copies of inlined connectors, so they must be known before it runs
        if isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.CPP:
            self.cpp_inline[id(node)] = self.find_cpp_inline(sdfg, cfg.nodes()[state_id], node)
        super()._generate_Tasklet(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream, codegen)
        self.flush_helpers(function_stream, cfg, state_id, node)

    def emit_tasklet_body_block(self, callsite_stream, cfg, state_id, node, inner_body, postamble, has_locals) -> None:
        """A tasklet without locals and with a single assignment collapses to one line, ``stmt;  // label``.
        Anything else keeps a scope, so that a local declaration cannot leak into the enclosing body."""
        if not has_locals and self.is_single_assignment(node):
            # The line's provenance tag is removed so that the label comment precedes it; the stream appends a
            # new tag. The newline keeps the comment from swallowing the next token (a closing brace).
            line = PROVENANCE_TAG.sub('', inner_body).strip()
            if line:
                callsite_stream.write(f'{line}  // {node.label}\n', cfg, state_id, node)
                return
        callsite_stream.write(f'{{  // {node.label}', cfg, state_id, node)
        callsite_stream.write(inner_body, cfg, state_id, node)
        callsite_stream.write(postamble)
        callsite_stream.write('}', cfg, state_id, node)

    @staticmethod
    def is_single_assignment(node: nodes.Node) -> bool:
        if not isinstance(node, nodes.Tasklet) or node.language != dtypes.Language.Python:
            return False
        stmts = node.code.code
        return len(stmts) == 1 and isinstance(stmts[0], (ast.Assign, ast.AugAssign))

    def rewrite_cpp_tasklet_body(self, node, sdfg, state_dfg) -> str:
        body = node.code.as_string
        inline = self.cpp_inline[id(node)]
        if not inline:
            return body
        return CPP_TOKEN.sub(lambda match: inline.get(match.group(0), match.group(0)), body)

    def find_cpp_inline(self, sdfg: SDFG, state, node: nodes.Tasklet) -> dict[str, str]:
        """The C++ access (element or base pointer) of every connector of a native tasklet that can be inlined."""
        # A preprocessor line (an OpenMP reduction) cannot have its connectors substituted: they sit in a pragma
        # the body shares, and a base pointer would turn ``_out[0]`` into ``&x[0]``
        if re.search(r'(?m)^[ \t]*#', node.code.as_string):
            return {}
        accesses: dict[str, dict[str, str]] = {'in': {}, 'out': {}}
        edges = {}
        for edge in state.in_edges(node):
            access = self.cpp_connector_access(sdfg, state, node, edge, False)
            if access is not None:
                accesses['in'][edge.dst_conn] = access
                edges[edge.dst_conn] = (edge, False)
        for edge in state.out_edges(node):
            access = self.cpp_connector_access(sdfg, state, node, edge, True)
            if access is not None:
                accesses['out'][edge.src_conn] = access
                edges[edge.src_conn] = (edge, True)
        inline = resolve_inout(accesses['in'], accesses['out'], node)
        # Skipping a copy also skips registering its target, whose file-level setup (a CUDA context for a
        # host-side memcpy) must still be generated
        for conn in inline:
            edge, is_output = edges[conn]
            if is_output:
                src, dst = node, state.memlet_path(edge)[-1].dst
            else:
                src, dst = state.memlet_path(edge)[0].src, node
            target = self._dispatcher.get_copy_dispatcher(src, dst, edge, sdfg, state)
            if target is not None:
                self._dispatcher.used_targets.add(target)
        return inline

    def cpp_connector_access(self, sdfg: SDFG, state, node: nodes.Tasklet, edge, is_output: bool) -> str | None:
        desc = inlinable_connector(sdfg, state, edge, is_output)
        if desc is None:
            return None
        memlet = edge.data
        conn_type = node.out_connectors[edge.src_conn] if is_output else node.in_connectors[edge.dst_conn]
        if not isinstance(conn_type, dtypes.pointer) and memlet.subset.num_elements() == 1:
            indices = [str(begin) for begin, _, _ in memlet.subset.ranges]
            return self.index_access(sdfg, desc, memlet.data, indices)
        # A library-call argument: the base pointer of the subset
        ptrname = self.ptr(memlet.data, desc, sdfg)
        if self._dispatcher.defined_vars.has(ptrname):
            defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
            return cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self)
        offset = cpp.cpp_offset_expr(desc, memlet.subset)
        return ptrname if offset == '0' else f'{ptrname} + {offset}'

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
        # The declaration of a write-once value is fused into its write, but reads must still resolve to it
        if node.data in self.const_bindings.get(sdfg.cfg_id, ()):
            ptr = self.ptr(node.data, nodedesc, sdfg)
            if isinstance(nodedesc, dt.Scalar):
                self._dispatcher.defined_vars.add(ptr, DefinedType.Scalar, nodedesc.dtype.ctype)
            else:
                self._dispatcher.defined_vars.add(ptr, DefinedType.Pointer, dtypes.pointer(nodedesc.dtype).ctype)
            return
        super().allocate_array(sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, declaration_stream,
                               allocation_stream, allocate_nested_data)
        self.flush_helpers(function_stream, cfg, state_id, node)

    def heap_array_count(self, sdfg: SDFG, data_name: str, desc: dt.Data, count: str) -> str:
        total = symbolic.pystr_to_symbolic(str(desc.total_size))
        # A bare symbol is no more readable behind a helper
        if total.is_Symbol:
            return count
        args = sorted(str(s) for s in total.free_symbols)
        qualifier = CONSTANT_SIZE_QUALIFIER if not args and self.cpp_standard() >= 20 else HELPER_QUALIFIER
        name = self.register_helper('size',
                                    data_name, (str(total), tuple(args)),
                                    params=args,
                                    body=sym2cpp(total),
                                    qualifier=qualifier)
        return f'{name}({", ".join(args)})'

    @staticmethod
    def cpp_standard() -> int:
        return int(str(Config.get('compiler', 'cpp_standard')).strip())

    def is_value_scalar(self, ptrname: str, desc: dt.Data) -> bool:
        """Whether a name is emitted as a plain value (``x``) rather than a pointer (``x[...]``)."""
        for registry in (self._dispatcher.defined_vars, self._dispatcher.declared_arrays):
            if registry.has(ptrname):
                return registry.get(ptrname)[0] == DefinedType.Scalar
        # Undeclared: a scalar is a value, except in GPU global memory where it is a device pointer
        return isinstance(desc, dt.Scalar) and desc.storage != dtypes.StorageType.GPU_Global

    def index_access(self, sdfg: SDFG, desc: dt.Data, data_name: str, indices: list[str]) -> str | None:
        """C++ for an access to an element: the plain name for a value, ``ptr[<array>_idx(...)]`` for an array,
        None if the number of indices does not match the array."""
        ptrname = self.ptr(data_name, desc, sdfg)
        if self.is_value_scalar(ptrname, desc):
            return ptrname
        if len(indices) != len(desc.shape):
            return None
        name, extra = self.register_index_function(data_name, desc)
        args = [sym2cpp(symbolic.pystr_to_symbolic(index)) for index in indices] + extra
        return f'{ptrname}[{name}({", ".join(args)})]'

    def register_index_function(self, data_name: str, desc: dt.Data) -> tuple[str, list[str]]:
        """Registers the offset function of an array and returns its name and the symbols it takes after the
        indices."""
        dims = [symbolic.symbol(f'__d{i}') for i in range(len(desc.shape))]
        strides = [symbolic.pystr_to_symbolic(str(s)) for s in desc.strides]
        offsets = [symbolic.pystr_to_symbolic(str(o)) for o in desc.offset]
        flat = sum((dim + offset) * stride for dim, offset, stride in zip(dims, offsets, strides, strict=True))
        extra = sorted(str(s) for s in flat.free_symbols - set(dims))
        signature = (len(dims), tuple(map(str, strides)), tuple(map(str, offsets)))
        name = self.register_helper('idx',
                                    data_name,
                                    signature,
                                    params=[str(d) for d in dims] + extra,
                                    body=sym2cpp(flat),
                                    qualifier=HELPER_QUALIFIER)
        return name, extra

    def register_helper(self, kind: str, data_name: str, signature: tuple, *, params: list[str], body: str,
                        qualifier: str) -> str:
        """Registers ``<array>_<kind>`` once per array and signature, and returns its name."""
        base = re.sub(r'\W', '_', data_name)
        key = (kind, base, signature)
        if key not in self.helper_names:
            name = f'{base}_{kind}'
            if name in self.helpers:
                name = f'{name}_{len(self.helper_names)}'
            self.helper_names[key] = name
            arguments = ', '.join(f'{INDEX_CTYPE} {p}' for p in params)
            self.helpers[name] = f'{qualifier} {INDEX_CTYPE} {name}({arguments}) {{ return {body}; }}'
        return self.helper_names[key]

    def flush_helpers(self, function_stream, cfg, state_id, node) -> None:
        """Writes the registered helpers once per output file. A nested SDFG function has its own stream but
        shares the host file with the others, so streams are keyed by the file's owner. A device file is generated
        through a delegating GPU generator and keeps its own copy."""
        file_key = id(self) if self.calling_codegen is self else id(function_stream)
        emitted = self.emitted_helpers.setdefault(file_key, set())
        for name, definition in self.helpers.items():
            if name not in emitted:
                function_stream.write(definition + '\n', cfg, state_id, node)
                emitted.add(name)
