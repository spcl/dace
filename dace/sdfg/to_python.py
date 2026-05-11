"""Emit Python source that reconstructs an SDFG via the imperative DaCe API.

Hard rule: imperative API only. No ``from_json`` / ``set_properties_from_json``
shortcuts. If a node, descriptor, or property has no imperative path, this
module raises ``NotImplementedError`` naming the offending object.
"""

from __future__ import annotations

import collections
import inspect
import re
from typing import Any, Dict, List, Optional, Tuple

import sympy

from dace import data as dt
from dace import dtypes, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowRegion,
    FunctionCallRegion,
    LoopRegion,
    NamedRegion,
    ReturnBlock,
    SDFGState,
)


def sdfg_to_python(sdfg: SDFG) -> str:
    """Return Python source code that reconstructs ``sdfg`` via the imperative API.

    The returned source defines ``build_sdfg() -> SDFG`` and re-creates the
    input by calling the public ``add_*`` API. Executing the source produces
    a structurally and semantically equivalent SDFG.
    """
    return PythonEmitter(sdfg).emit()


class PythonEmitter:
    """Walks an SDFG and produces Python source that rebuilds it."""

    HEADER_IMPORTS = (
        "import dace",
        "from dace import dtypes, symbolic",
        "from dace.memlet import Memlet",
        "from dace.properties import CodeBlock",
        "from dace.sdfg import nodes as dace_nodes",
        "from dace.sdfg.sdfg import SDFG, InterstateEdge",
        "from dace.sdfg.state import (",
        "    BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion,",
        "    FunctionCallRegion, LoopRegion, NamedRegion, ReturnBlock,",
        ")",
    )

    # Minimum number of occurrences required before a helper is emitted.
    # Below this, the pattern is left as a literal API call to keep the
    # learning cost down for small SDFGs.
    HELPER_THRESHOLD = 5

    def __init__(self, sdfg: SDFG):
        self.root = sdfg
        self._nested_factories: List[List[str]] = []
        self._state_populators: List[List[str]] = []
        self._var_for: Dict[int, str] = {}
        self._name_counter: Dict[str, int] = {}
        self._taken_names: set = set()
        self._populator_counter: int = 0
        self._extra_imports: Dict[str, None] = {}
        # Scan the SDFG up front and decide which compact helpers earn
        # their keep. See _scan_helper_eligibility.
        self._helpers_used: set = self._scan_helper_eligibility()

    def emit(self) -> str:
        body = self._emit_factory("build_sdfg", self.root)
        lines: List[str] = []
        lines.append(self._module_docstring())
        lines.append("")
        lines.extend(self.HEADER_IMPORTS)
        for extra in self._extra_imports:
            lines.append(extra)
        lines.append("")
        lines.append("")
        # Compact helper kit — emitted only for patterns that recur often
        # enough in this SDFG to earn their slot in the file. The set is
        # decided up-front by ``_scan_helper_eligibility``.
        if self._helpers_used:
            lines.append("# === Compact helpers ===")
            lines.append(
                "# Stamped shortcuts for the most repetitive patterns in this SDFG."
            )
            for helper in _HELPER_ORDER:
                if helper in self._helpers_used:
                    lines.extend(_HELPER_DEFS[helper])
                    lines.append("")
            lines.append("")
        # Per-state populator helpers come first so the larger SDFG
        # factories below can reference them by name.
        for populator in self._state_populators:
            lines.extend(populator)
            lines.append("")
            lines.append("")
        for nested in self._nested_factories:
            lines.extend(nested)
            lines.append("")
            lines.append("")
        lines.extend(body)
        lines.append("")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    build_sdfg().validate()")
        lines.append("")
        return "\n".join(lines)

    def _render_memlet(self, memlet) -> str:
        """Memlet emission with optional helper-kit substitution.

        When the ``scalar_memlet`` helper is enabled (used >= threshold), a
        ``Memlet(data='X', subset='0')`` collapses to ``_S('X')``.
        """
        if "scalar_memlet" in self._helpers_used and _is_scalar_memlet(memlet):
            return f"_S({_pyrepr(memlet.data)})"
        return _emit_memlet(memlet)

    def _scan_helper_eligibility(self) -> set:
        """Walk the SDFG once and count how often each compact-helper pattern
        appears. Helpers whose count clears ``HELPER_THRESHOLD`` are switched
        on for the emit pass; the rest stay as literal API calls.
        """
        counts: Dict[str, int] = collections.Counter()
        for sub_sdfg in self.root.all_sdfgs_recursive():
            for state in sub_sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, nd.Tasklet):
                        kind = _classify_tasklet(node)
                        if kind is not None:
                            counts[kind[0] + "_tasklet"] += 1
                for edge in state.edges():
                    if _is_scalar_memlet(edge.data):
                        counts["scalar_memlet"] += 1
        return {name for name, n in counts.items() if n >= self.HELPER_THRESHOLD}

    def _module_docstring(self) -> str:
        sdfg = self.root
        n_arrays = len(sdfg.arrays)
        n_symbols = len(sdfg.symbols)
        n_states = sum(1 for _ in sdfg.all_states())
        return (
            f'"""Reconstructed SDFG ``{sdfg.name}`` (auto-generated).\n\n'
            f"This file was emitted by ``dace.sdfg.to_python.sdfg_to_python``.\n"
            f"Calling :func:`build_sdfg` constructs an SDFG equivalent to the\n"
            f"original via DaCe's imperative public API. No JSON deserialisation\n"
            f"is involved.\n\n"
            f"Stats: {n_symbols} symbols, {n_arrays} data descriptors, "
            f"{n_states} states.\n"
            f'"""'
        )

    def _emit_factory(self, fn_name: str, sdfg: SDFG) -> List[str]:
        lines: List[str] = [f"def {fn_name}() -> SDFG:"]
        body = _IndentedBuffer(indent=1)
        sdfg_var = self._fresh("sdfg")
        self._emit_sdfg(sdfg, sdfg_var, body)
        body.line(f"return {sdfg_var}")
        lines.extend(body.lines)
        return lines

    # ------------------------------------------------------------------
    # SDFG-level emission
    # ------------------------------------------------------------------

    def _emit_sdfg(self, sdfg: SDFG, var: str, buf: "_IndentedBuffer"):
        buf.line(f"# Top-level SDFG: {sdfg.name!r}")
        buf.line(f"{var} = SDFG({_pyrepr(sdfg.name)})")

        if getattr(sdfg, "using_explicit_control_flow", False):
            buf.line(f"{var}.using_explicit_control_flow = True")

        if sdfg.symbols:
            buf.line("")
            buf.line(
                f"# --- Symbols ({len(sdfg.symbols)}) ---"
            )
            self._emit_symbol_block(sdfg, var, buf)

        if sdfg.constants_prop:
            buf.line("")
            buf.line(
                f"# --- Constants ({len(sdfg.constants_prop)}) ---"
            )
            self._emit_constant_block(sdfg, var, buf)

        if sdfg.arrays:
            buf.line("")
            buf.line(
                f"# --- Data descriptors ({len(sdfg.arrays)}) ---"
            )
            self._emit_descriptor_block(sdfg, var, buf)

        # Init/global/exit code blocks — only emit a section when at least
        # one block has non-empty source.
        code_pairs = [
            (lang, code, "append_global_code")
            for lang, code in sdfg.global_code.items() if code.as_string
        ] + [
            (lang, code, "append_init_code")
            for lang, code in sdfg.init_code.items() if code.as_string
        ] + [
            (lang, code, "append_exit_code")
            for lang, code in sdfg.exit_code.items() if code.as_string
        ]
        if code_pairs:
            buf.line("")
            buf.line("# --- Init/Global/Exit code blocks ---")
            for lang, code, method in code_pairs:
                buf.line(
                    f"{var}.{method}({_pyrepr(code.as_string)}, "
                    f"location={_emit_language(lang)})"
                )

        # Hoist the SDFG's CFG body (states, regions, interstate edges)
        # into a populator function so this factory stays a thin orchestrator.
        populator = self._emit_cfg_populator(sdfg, kind="sdfg")
        if populator is not None:
            buf.line("")
            buf.line("# --- Control flow body ---")
            buf.line(f"{populator}({var}, {var})")

    # ------------------------------------------------------------------
    # Bulk emission for symbols / constants / descriptors
    #
    # The SDFG-level pre-state setup (symbols, constants, data descriptors)
    # is emitted as data-driven loops over Python literal lists rather than
    # one ``add_X`` line per item. This keeps a 200-symbol SDFG readable —
    # the table of names is data, the loop body is the "how to add" code.
    # ------------------------------------------------------------------

    def _emit_symbol_block(self, sdfg: SDFG, var: str,
                           buf: "_IndentedBuffer"):
        items = list(sdfg.symbols.items())
        if not items:
            return
        if len(items) == 1:
            name, dtype = items[0]
            buf.line(
                f"{var}.add_symbol({_pyrepr(name)}, {_emit_dtype(dtype)})"
            )
            return
        # Group by dtype: an SDFG with 250 bool symbols collapses into a
        # single name list + one-line loop body, not 250 ``add_symbol`` lines.
        by_dtype: Dict[str, List[str]] = collections.OrderedDict()
        for name, dtype in items:
            by_dtype.setdefault(_emit_dtype(dtype), []).append(name)

        for dtype_str, names in by_dtype.items():
            if len(names) == 1:
                buf.line(f"{var}.add_symbol({_pyrepr(names[0])}, {dtype_str})")
                continue
            buf.line("for _name in [")
            for name in names:
                buf.line(f"    {_pyrepr(name)},")
            buf.line("]:")
            buf.line(f"    {var}.add_symbol(_name, {dtype_str})")

    def _emit_constant_block(self, sdfg: SDFG, var: str,
                             buf: "_IndentedBuffer"):
        items = list(sdfg.constants_prop.items())
        if not items:
            return
        if len(items) == 1:
            name, (cdtype, cval) = items[0]
            dtype_arg = _emit_dtype(cdtype) if cdtype is not None else "None"
            buf.line(
                f"{var}.add_constant({_pyrepr(name)}, "
                f"{_emit_constant_value(cval)}, {dtype_arg})"
            )
            return
        buf.line("for _name, _value, _dtype in [")
        for name, (cdtype, cval) in items:
            dtype_arg = _emit_dtype(cdtype) if cdtype is not None else "None"
            buf.line(
                f"    ({_pyrepr(name)}, {_emit_constant_value(cval)}, "
                f"{dtype_arg}),"
            )
        buf.line("]:")
        buf.line(f"    {var}.add_constant(_name, _value, _dtype)")

    def _emit_descriptor_block(self, sdfg: SDFG, var: str,
                               buf: "_IndentedBuffer"):
        scalars: List = []
        arrays: List = []
        streams: List = []
        views: List = []
        refs: List = []
        for name, desc in sdfg.arrays.items():
            if isinstance(desc, dt.Scalar):
                scalars.append((name, desc))
            elif isinstance(desc, dt.ArrayView):
                views.append((name, desc))
            elif isinstance(desc, dt.ArrayReference):
                refs.append((name, desc))
            elif isinstance(desc, dt.Stream):
                streams.append((name, desc))
            elif type(desc) is dt.Array:
                arrays.append((name, desc))
            else:
                raise NotImplementedError(
                    f"Descriptor emission not implemented for {name!r} "
                    f"({type(desc).__module__}.{type(desc).__name__}); extend "
                    f"to_python._emit_descriptor_block with imperative API"
                )

        self._emit_array_loop(arrays, var, "add_array", buf, _array_kwargs)
        self._emit_descriptor_loop(scalars, var, "add_scalar", buf,
                                   _scalar_kwargs, takes_shape=False)
        self._emit_array_loop(views, var, "add_view", buf, _arrayview_kwargs)
        self._emit_array_loop(refs, var, "add_reference", buf,
                              _arrayview_kwargs)
        self._emit_descriptor_loop(streams, var, "add_stream", buf,
                                   _stream_kwargs, takes_shape=False)

    def _emit_array_loop(self, items: List, var: str, method: str,
                         buf: "_IndentedBuffer", kwargs_fn):
        """Emit grouped loops for ``var.add_X(name, shape, dtype, **kwargs)``.

        Arrays sharing the same (shape, dtype, kwargs) signature collapse into
        one name list + a one-line loop body. CloudSC has hundreds of
        ``('klev', 'klon')`` float64 arrays — that whole block becomes a single
        for-over-names list instead of hundreds of repeated ``add_array`` lines.
        """
        if not items:
            return
        if len(items) == 1:
            name, desc = items[0]
            kw = kwargs_fn(desc)
            buf.line(_call(f"{var}.{method}",
                           [_pyrepr(name), _emit_shape(desc.shape),
                            _emit_dtype(desc.dtype)], kw))
            return

        groups = self._group_descriptors(items, kwargs_fn)
        # Group by (shape, dtype, kwargs) so arrays sharing a signature
        # collapse into one name list.
        buf.line(
            f"# {method}: {sum(len(n) for n in groups.values())} arrays, "
            f"{len(groups)} signature group(s)"
        )
        for (shape_str, dtype_str, kw_tuple), names in groups.items():
            kw = dict(kw_tuple)
            if len(names) == 1:
                buf.line(_call(f"{var}.{method}",
                               [_pyrepr(names[0]), shape_str, dtype_str], kw))
                continue
            buf.line("for _name in [")
            for name in names:
                buf.line(f"    {_pyrepr(name)},")
            buf.line("]:")
            kw_args = "".join(f", {k}={v}" for k, v in kw.items())
            buf.line(f"    {var}.{method}(_name, {shape_str}, {dtype_str}{kw_args})")

    def _emit_descriptor_loop(self, items: List, var: str, method: str,
                              buf: "_IndentedBuffer", kwargs_fn,
                              takes_shape: bool):
        """Emit grouped loops for ``var.add_X(name, dtype, **kwargs)`` (no shape)."""
        if not items:
            return
        if len(items) == 1:
            name, desc = items[0]
            kw = kwargs_fn(desc)
            buf.line(_call(f"{var}.{method}",
                           [_pyrepr(name), _emit_dtype(desc.dtype)], kw))
            return

        groups: Dict[Tuple[str, Tuple], List[str]] = collections.OrderedDict()
        for name, desc in items:
            kw = kwargs_fn(desc)
            key = (_emit_dtype(desc.dtype), tuple(sorted(kw.items())))
            groups.setdefault(key, []).append(name)

        buf.line(
            f"# {method}: {sum(len(n) for n in groups.values())} entries, "
            f"{len(groups)} dtype group(s)"
        )
        for (dtype_str, kw_tuple), names in groups.items():
            kw = dict(kw_tuple)
            if len(names) == 1:
                buf.line(_call(f"{var}.{method}",
                               [_pyrepr(names[0]), dtype_str], kw))
                continue
            buf.line("for _name in [")
            for name in names:
                buf.line(f"    {_pyrepr(name)},")
            buf.line("]:")
            kw_args = "".join(f", {k}={v}" for k, v in kw.items())
            buf.line(f"    {var}.{method}(_name, {dtype_str}{kw_args})")

    @staticmethod
    def _group_descriptors(items: List, kwargs_fn):
        """Group descriptors by their (shape, dtype, kwargs) signature."""
        groups: Dict[Tuple[str, str, Tuple], List[str]] = collections.OrderedDict()
        for name, desc in items:
            kw = kwargs_fn(desc)
            key = (_emit_shape(desc.shape), _emit_dtype(desc.dtype),
                   tuple(sorted(kw.items())))
            groups.setdefault(key, []).append(name)
        return groups

    # ------------------------------------------------------------------
    # Control flow region body
    # ------------------------------------------------------------------

    def _emit_cfg_populator(self, cfg: AbstractControlFlowRegion,
                            kind: str = "cfg") -> Optional[str]:
        """Emit a top-level ``_populate_<kind>_<N>(cfg, sdfg)`` function for
        this CFG region's body, and return the function name. Returns ``None``
        for empty regions so the caller can skip emitting a noop call.

        The CFG-region populator is the parallel of the per-state populator:
        instead of inlining nested LoopRegion/ConditionalBlock bodies into
        ``build_sdfg()``, each region's body lives in its own helper. The
        result is that a 200-state SDFG factory degenerates to:
        ``add_state(...)``, ``_populate_state_N(...)``, plus a flat list of
        inter-state edges — readable at a glance.
        """
        if cfg.number_of_nodes() == 0:
            return None

        idx = self._populator_counter
        self._populator_counter += 1
        fn_name = f"_populate_{kind}_{idx}"
        # Reserve our slot before recursing so any nested CFG populator
        # we encounter while emitting our body claims a later index.
        self._state_populators.append([])

        outer_var_for = self._var_for
        outer_counter = self._name_counter
        outer_taken = self._taken_names
        self._var_for = {}
        self._name_counter = {}
        self._taken_names = set()
        self._taken_names.update({"cfg", "sdfg"})  # protect parameter names

        body = _IndentedBuffer(indent=1)
        try:
            start_block = cfg.start_block
        except ValueError:
            start_block = None

        for block in cfg.nodes():
            self._emit_cfg_block(block, "cfg", body,
                                 is_start=block is start_block)

        self._emit_interstate_edges(cfg, "cfg", body)

        self._var_for = outer_var_for
        self._name_counter = outer_counter
        self._taken_names = outer_taken

        lines = [f"def {fn_name}(cfg, sdfg):"]
        descr = _describe_region(cfg)
        lines.append(f'    """Populate the body of {descr}."""')
        if not body.lines:
            lines.append("    pass")
        else:
            lines.extend(body.lines)
        self._state_populators[idx] = lines
        return fn_name

    def _emit_interstate_edges(self, cfg: AbstractControlFlowRegion,
                               cfg_var: str, buf: "_IndentedBuffer"):
        """Batch trivial ``InterstateEdge()`` edges into a single for-loop;
        emit conditioned/assigning edges imperatively (one line each)."""
        trivial: List = []
        rich: List = []
        for edge in cfg.edges():
            src_var = self._var_for[id(edge.src)]
            dst_var = self._var_for[id(edge.dst)]
            data = edge.data
            cond_str = data.condition.as_string if data.condition is not None else ""
            is_trivial = (
                cond_str.strip() in {"", "1", "True", "true"}
                and not data.assignments
            )
            if is_trivial:
                trivial.append((src_var, dst_var))
            else:
                rich.append((src_var, dst_var, _emit_interstate_edge(data)))

        if len(trivial) == 1:
            src_var, dst_var = trivial[0]
            buf.line(f"{cfg_var}.add_edge({src_var}, {dst_var}, InterstateEdge())")
        elif trivial:
            buf.line("for _src, _dst in [")
            for src_var, dst_var in trivial:
                buf.line(f"    ({src_var}, {dst_var}),")
            buf.line("]:")
            buf.line(f"    {cfg_var}.add_edge(_src, _dst, InterstateEdge())")

        for src_var, dst_var, edge_str in rich:
            buf.line(f"{cfg_var}.add_edge({src_var}, {dst_var}, {edge_str})")

    def _emit_cfg_block(self, block, parent_var: str, buf: "_IndentedBuffer",
                        is_start: bool):
        if isinstance(block, SDFGState):
            var = self._fresh(f"s_{_sanitize(block.label)}")
            extra = ", is_start_block=True" if is_start else ""
            buf.line(
                f"{var} = {parent_var}.add_state({_pyrepr(block.label)}{extra})"
            )
            self._var_for[id(block)] = var
            populator = self._emit_state_populator(block)
            if populator is not None:
                buf.line(f"{populator}({var})")
            return

        if isinstance(block, LoopRegion):
            var = self._fresh(f"loop_{_sanitize(block.label)}")
            kwargs: Dict[str, str] = {}
            if block.loop_condition is not None and block.loop_condition.as_string:
                kwargs["condition_expr"] = _pyrepr(block.loop_condition.as_string)
            if block.loop_variable:
                kwargs["loop_var"] = _pyrepr(block.loop_variable)
            if block.init_statement is not None and block.init_statement.as_string:
                kwargs["initialize_expr"] = _pyrepr(block.init_statement.as_string)
            if block.update_statement is not None and block.update_statement.as_string:
                kwargs["update_expr"] = _pyrepr(block.update_statement.as_string)
            if block.inverted:
                kwargs["inverted"] = "True"
            if not block.update_before_condition:
                kwargs["update_before_condition"] = "False"
            if block.unroll:
                kwargs["unroll"] = "True"
            if block.unroll_factor:
                kwargs["unroll_factor"] = repr(int(block.unroll_factor))
            # Multi-line LoopRegion call for readability — its parameter
            # list is long enough that a single line would exceed any
            # reasonable wrap.
            buf.line(f"{var} = LoopRegion(")
            buf.line(f"    label={_pyrepr(block.label)},")
            for k, v in kwargs.items():
                buf.line(f"    {k}={v},")
            buf.line(f")")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            populator = self._emit_cfg_populator(block, kind="loop")
            if populator is not None:
                buf.line(f"{populator}({var}, sdfg)")
            return

        if isinstance(block, ConditionalBlock):
            var = self._fresh(f"cond_{_sanitize(block.label)}")
            buf.line(f"{var} = ConditionalBlock({_pyrepr(block.label)})")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            for branch_idx, (cond, region) in enumerate(block.branches):
                region_var = self._fresh(f"branch_{_sanitize(region.label)}")
                buf.line(
                    f"{region_var} = ControlFlowRegion("
                    f"{_pyrepr(region.label)}, sdfg=sdfg)"
                )
                self._var_for[id(region)] = region_var
                populator = self._emit_cfg_populator(region, kind="branch")
                if populator is not None:
                    buf.line(f"{populator}({region_var}, sdfg)")
                cond_arg = (
                    f"CodeBlock({_pyrepr(cond.as_string)})" if cond is not None else "None"
                )
                buf.line(f"{var}.add_branch({cond_arg}, {region_var})")
            return

        if isinstance(block, FunctionCallRegion):
            var = self._fresh(f"call_{_sanitize(block.label)}")
            args_dict = "{" + ", ".join(
                f"{_pyrepr(k)}: {_pyrepr(v)}" for k, v in block.arguments.items()
            ) + "}"
            buf.line(
                f"{var} = FunctionCallRegion({_pyrepr(block.label)}, "
                f"arguments={args_dict})"
            )
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            populator = self._emit_cfg_populator(block, kind="call")
            if populator is not None:
                buf.line(f"{populator}({var}, sdfg)")
            return

        if isinstance(block, NamedRegion):
            var = self._fresh(f"named_{_sanitize(block.label)}")
            buf.line(f"{var} = NamedRegion({_pyrepr(block.label)})")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            populator = self._emit_cfg_populator(block, kind="named")
            if populator is not None:
                buf.line(f"{populator}({var}, sdfg)")
            return

        if isinstance(block, ControlFlowRegion):
            var = self._fresh(f"cfg_{_sanitize(block.label)}")
            buf.line(f"{var} = ControlFlowRegion({_pyrepr(block.label)})")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            populator = self._emit_cfg_populator(block, kind="cfr")
            if populator is not None:
                buf.line(f"{populator}({var}, sdfg)")
            return

        if isinstance(block, BreakBlock):
            var = self._fresh(f"brk_{_sanitize(block.label)}")
            buf.line(f"{var} = BreakBlock({_pyrepr(block.label)})")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            return

        if isinstance(block, ContinueBlock):
            var = self._fresh(f"cnt_{_sanitize(block.label)}")
            buf.line(f"{var} = ContinueBlock({_pyrepr(block.label)})")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            return

        if isinstance(block, ReturnBlock):
            var = self._fresh(f"ret_{_sanitize(block.label)}")
            buf.line(f"{var} = ReturnBlock({_pyrepr(block.label)})")
            self._add_node(parent_var, var, is_start, buf)
            self._var_for[id(block)] = var
            return

        raise NotImplementedError(
            f"CFG block emission not implemented for "
            f"{type(block).__module__}.{type(block).__name__} "
            f"(label={getattr(block, 'label', '?')!r})"
        )

    @staticmethod
    def _add_node(parent_var: str, var: str, is_start: bool,
                  buf: "_IndentedBuffer"):
        extra = ", is_start_block=True" if is_start else ""
        buf.line(f"{parent_var}.add_node({var}{extra})")

    # ------------------------------------------------------------------
    # State body
    # ------------------------------------------------------------------

    def _emit_state_populator(self, state: SDFGState) -> Optional[str]:
        """Emit a top-level ``_populate_state_<N>(state)`` function for this
        state's contents. Returns the function name, or ``None`` if the
        state has no nodes (so the caller can skip emitting a noop call).

        Each state body lives in its own function so the parent SDFG
        factory stays a thin orchestration of ``add_state(...)`` calls
        and a populator dispatch — readable at a glance even for SDFGs
        with hundreds of states.
        """
        if state.number_of_nodes() == 0:
            return None

        idx = self._populator_counter
        self._populator_counter += 1
        fn_name = f"_populate_state_{idx}"
        # Reserve slot before emitting so any nested SDFG inside the
        # state can register its own populators with later indices.
        self._state_populators.append([])

        # Push a fresh local scope; node vars within the populator are
        # local Python variables, and they shouldn't collide with
        # anything in the parent factory's scope.
        outer_var_for = self._var_for
        outer_counter = self._name_counter
        outer_taken = self._taken_names
        self._var_for = {}
        self._name_counter = {}
        self._taken_names = set()

        body = _IndentedBuffer(indent=1)
        skip: set = set()
        for node in state.nodes():
            if id(node) in skip:
                continue
            self._emit_state_node(node, state, "state", body, skip)
        for edge in state.edges():
            src_var = self._var_for[id(edge.src)]
            dst_var = self._var_for[id(edge.dst)]
            src_conn = _pyrepr(edge.src_conn) if edge.src_conn is not None else "None"
            dst_conn = _pyrepr(edge.dst_conn) if edge.dst_conn is not None else "None"
            body.line(
                f"state.add_edge({src_var}, {src_conn}, {dst_var}, "
                f"{dst_conn}, {self._render_memlet(edge.data)})"
            )

        self._var_for = outer_var_for
        self._name_counter = outer_counter
        self._taken_names = outer_taken

        lines = [f"def {fn_name}(state):"]
        lines.append(
            f'    """Populate state {state.label!r} with its data-flow body."""'
        )
        if not body.lines:
            lines.append("    pass")
        else:
            lines.extend(body.lines)
        self._state_populators[idx] = lines
        return fn_name

    def _emit_state_node(self, node, state: SDFGState, state_var: str,
                         buf: "_IndentedBuffer", skip: set):
        if isinstance(node, nd.AccessNode):
            var = self._fresh(f"acc_{_sanitize(node.data)}")
            buf.line(f"{var} = {state_var}.add_access({_pyrepr(node.data)})")
            self._var_for[id(node)] = var
            return

        if isinstance(node, nd.MapEntry):
            entry_var = self._fresh(f"me_{_sanitize(node.label)}")
            exit_node = state.exit_node(node)
            exit_var = self._fresh(f"mx_{_sanitize(node.label)}")
            map_obj = node.map
            ndrange_parts = []
            for p, r in zip(map_obj.params, map_obj.range.ranges):
                ndrange_parts.append(f"{_pyrepr(p)}: {_pyrepr(_subset_range_str([r]))}")
            ndrange = "{" + ", ".join(ndrange_parts) + "}"
            kwargs: Dict[str, str] = {}
            if map_obj.schedule is not dtypes.ScheduleType.Default:
                kwargs["schedule"] = _emit_schedule(map_obj.schedule)
            if map_obj.unroll:
                kwargs["unroll"] = "True"
            buf.line(
                f"{entry_var}, {exit_var} = "
                + _call(f"{state_var}.add_map", [_pyrepr(map_obj.label), ndrange], kwargs)
            )
            self._var_for[id(node)] = entry_var
            self._var_for[id(exit_node)] = exit_var
            skip.add(id(exit_node))
            _emit_extra_connectors(entry_var, node, buf)
            _emit_extra_connectors(exit_var, exit_node, buf)
            return

        if isinstance(node, nd.MapExit):
            # Should have been emitted alongside its entry; if we hit a bare
            # exit, something is malformed.
            raise NotImplementedError(
                f"MapExit {node} encountered without paired MapEntry; "
                f"malformed SDFG"
            )

        if isinstance(node, nd.ConsumeEntry):
            entry_var = self._fresh(f"ce_{_sanitize(node.label)}")
            exit_node = state.exit_node(node)
            exit_var = self._fresh(f"cx_{_sanitize(node.label)}")
            consume = node.consume
            elements = (
                consume.pe_index,
                _emit_symbolic_inline(consume.num_pes),
            )
            args = [_pyrepr(consume.label),
                    f"({_pyrepr(elements[0])}, {elements[1]})"]
            kwargs: Dict[str, str] = {}
            if consume.condition is not None and consume.condition.as_string:
                kwargs["condition"] = _pyrepr(consume.condition.as_string)
            if consume.schedule is not dtypes.ScheduleType.Default:
                kwargs["schedule"] = _emit_schedule(consume.schedule)
            if int(consume.chunksize) != 1:
                kwargs["chunksize"] = repr(int(consume.chunksize))
            buf.line(
                f"{entry_var}, {exit_var} = "
                + _call(f"{state_var}.add_consume", args, kwargs)
            )
            self._var_for[id(node)] = entry_var
            self._var_for[id(exit_node)] = exit_var
            skip.add(id(exit_node))
            _emit_extra_connectors(entry_var, node, buf)
            _emit_extra_connectors(exit_var, exit_node, buf)
            return

        if isinstance(node, nd.ConsumeExit):
            raise NotImplementedError(
                "ConsumeExit without paired ConsumeEntry; malformed SDFG"
            )

        if isinstance(node, nd.NestedSDFG):
            factory_name = self._emit_nested_factory(node.sdfg)
            var = self._fresh(f"nsdfg_{_sanitize(node.label)}")
            args = [
                f"{factory_name}()",
                _emit_connector_dict(node.in_connectors),
                _emit_connector_dict(node.out_connectors),
            ]
            kwargs: Dict[str, str] = {
                "name": _pyrepr(node.label),
            }
            if node.symbol_mapping:
                mapping = "{" + ", ".join(
                    f"{_pyrepr(k)}: {_pyrepr(symbolic.symstr(v) if isinstance(v, sympy.Basic) else str(v))}"
                    for k, v in node.symbol_mapping.items()
                ) + "}"
                kwargs["symbol_mapping"] = mapping
            buf.line(
                f"{var} = " + _call(f"{state_var}.add_nested_sdfg", args, kwargs)
            )
            self._var_for[id(node)] = var
            return

        if isinstance(node, nd.Tasklet):
            # Try the compact helper kit first — saves ~60% of the per-line
            # noise on common stamped tasklets (binary ops, const assignments).
            kind = _classify_tasklet(node)
            if kind is not None:
                helper_name = kind[0] + "_tasklet"
                if helper_name in self._helpers_used:
                    var = self._fresh(f"t_{_sanitize(node.label)}")
                    if kind[0] == "const":
                        buf.line(
                            f"{var} = _const_tasklet({state_var}, "
                            f"{_pyrepr(node.label)}, {_pyrepr(kind[1])})"
                        )
                    else:  # binop
                        buf.line(
                            f"{var} = _binop_tasklet({state_var}, "
                            f"{_pyrepr(node.label)}, {_pyrepr(kind[1])})"
                        )
                    self._var_for[id(node)] = var
                    return

            var = self._fresh(f"t_{_sanitize(node.label)}")
            args = [
                _pyrepr(node.label),
                _emit_connector_dict(node.in_connectors),
                _emit_connector_dict(node.out_connectors),
                _pyrepr(node.code.as_string),
            ]
            kwargs: Dict[str, str] = {}
            if node.code.language is not dtypes.Language.Python:
                kwargs["language"] = _emit_language(node.code.language)
            if node.code_global and node.code_global.as_string:
                kwargs["code_global"] = _pyrepr(node.code_global.as_string)
            if node.code_init and node.code_init.as_string:
                kwargs["code_init"] = _pyrepr(node.code_init.as_string)
            if node.code_exit and node.code_exit.as_string:
                kwargs["code_exit"] = _pyrepr(node.code_exit.as_string)
            if node.side_effects is not None:
                kwargs["side_effects"] = repr(bool(node.side_effects))
            buf.line(f"{var} = " + _call(f"{state_var}.add_tasklet", args, kwargs))
            self._var_for[id(node)] = var
            return

        if isinstance(node, nd.LibraryNode):
            self._emit_library_node(node, state, state_var, buf)
            return

        raise NotImplementedError(
            f"State-node emission not implemented for "
            f"{type(node).__module__}.{type(node).__name__} "
            f"(label={getattr(node, 'label', '?')!r}); extend "
            f"to_python._emit_state_node"
        )

    # ------------------------------------------------------------------
    # NestedSDFG support
    # ------------------------------------------------------------------

    def _emit_nested_factory(self, nested: SDFG) -> str:
        # Reserve our slot BEFORE recursing — otherwise any nested-SDFG
        # encountered inside ``nested`` will read the same ``len(...)`` and
        # overwrite our function name.
        idx = len(self._nested_factories)
        self._nested_factories.append([])
        fn_name = f"_build_nested_{idx}"
        outer_var_for = self._var_for
        outer_counter = self._name_counter
        outer_taken = self._taken_names
        self._var_for = {}
        self._name_counter = {}
        self._taken_names = set()
        nested_lines = self._emit_factory(fn_name, nested)
        self._var_for = outer_var_for
        self._name_counter = outer_counter
        self._taken_names = outer_taken
        self._nested_factories[idx] = nested_lines
        return fn_name

    # ------------------------------------------------------------------
    # LibraryNode
    # ------------------------------------------------------------------

    def _emit_library_node(self, node, state: SDFGState, state_var: str,
                           buf: "_IndentedBuffer"):
        cls = type(node)
        module = cls.__module__
        qualname = cls.__qualname__
        var = self._fresh(f"ln_{_sanitize(getattr(node, 'label', cls.__name__))}")
        self._extra_imports[f"from {module} import {cls.__name__}"] = None

        init_fn = _unwrap_make_properties_init(cls)
        try:
            sig = inspect.signature(init_fn)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError(
                f"LibraryNode {module}.{qualname} has no introspectable "
                f"__init__: {exc}; expose a typed constructor or extend "
                f"to_python._emit_library_node"
            ) from exc

        ctor_params = [
            p for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]

        ctor_args: List[str] = []
        ctor_kwargs: Dict[str, str] = {}
        consumed_props: set = set()

        prop_defaults = {
            name: prop.default
            for name, prop in cls.__properties__.items()
        } if hasattr(cls, "__properties__") else {}

        for p in ctor_params:
            value = _read_node_attr(node, p.name)
            if value is _MISSING:
                if p.default is inspect.Parameter.empty:
                    raise NotImplementedError(
                        f"LibraryNode {module}.{qualname}.__init__ requires "
                        f"parameter {p.name!r} but the node has no matching "
                        f"attribute or property; extend to_python or expose "
                        f"the value as a Property/attr"
                    )
                continue
            rendered = _emit_value(value)
            if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                ctor_args.append(rendered)
            elif p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and not ctor_kwargs:
                ctor_args.append(rendered)
            else:
                ctor_kwargs[p.name] = rendered
            consumed_props.add(p.name)

        buf.line(f"{var} = " + _call(cls.__name__, ctor_args, ctor_kwargs))
        buf.line(f"{state_var}.add_node({var})")
        self._var_for[id(node)] = var

        for cname, cdtype in node.in_connectors.items():
            buf.line(
                f"{var}.add_in_connector({_pyrepr(cname)}, "
                f"dtype={_emit_dtype_optional(cdtype)})"
            )
        for cname, cdtype in node.out_connectors.items():
            buf.line(
                f"{var}.add_out_connector({_pyrepr(cname)}, "
                f"dtype={_emit_dtype_optional(cdtype)})"
            )

        # Properties not covered by __init__ — set imperatively via attribute
        # assignment if they differ from their declared default.
        for prop_name, default in prop_defaults.items():
            if prop_name in consumed_props or prop_name in {
                "in_connectors", "out_connectors", "guid", "debuginfo"
            }:
                continue
            try:
                current = getattr(node, prop_name)
            except AttributeError:
                continue
            if _values_equal(current, default):
                continue
            if not _attr_is_settable(node, prop_name):
                raise NotImplementedError(
                    f"LibraryNode {module}.{qualname} property "
                    f"{prop_name!r} is not exposed via a setter and is not a "
                    f"constructor parameter; extend to_python or the node "
                    f"class with imperative API"
                )
            buf.line(f"{var}.{prop_name} = {_emit_value(current)}")

    def _fresh(self, base: str) -> str:
        base = _sanitize(base) or "x"
        # Track allocated identifiers globally so two different ``base`` values
        # whose suffixed forms would collide (e.g. ``foo_1`` from ``foo`` plus
        # ``foo_1`` from another caller) don't reuse the same var name.
        n = self._name_counter.get(base, 0)
        while True:
            candidate = f"{base}_{n}" if n else base
            if candidate not in self._taken_names:
                self._taken_names.add(candidate)
                self._name_counter[base] = n + 1
                return candidate
            n += 1


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------


_MISSING = object()


class _IndentedBuffer:
    def __init__(self, indent: int = 0):
        self.indent = indent
        self.lines: List[str] = []

    def line(self, text: str = ""):
        if text:
            self.lines.append("    " * self.indent + text)
        else:
            self.lines.append("")


_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(name: str) -> str:
    s = _IDENT_RE.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _pyrepr(value: Any) -> str:
    return repr(value)


def _call(fn: str, args: List[str], kwargs: Dict[str, str]) -> str:
    parts = list(args) + [f"{k}={v}" for k, v in kwargs.items()]
    return f"{fn}({', '.join(parts)})"


def _emit_kwargs_dict(kwargs: Dict[str, str]) -> str:
    """Render a kwargs dict as a Python dict literal (keys are strings)."""
    if not kwargs:
        return "{}"
    items = ", ".join(f"{_pyrepr(k)}: {v}" for k, v in kwargs.items())
    return "{" + items + "}"


def _describe_region(cfg) -> str:
    """Human-readable label for a CFG region, used in populator docstrings."""
    if isinstance(cfg, SDFG):
        return f"SDFG {cfg.name!r}"
    if isinstance(cfg, LoopRegion):
        cond = cfg.loop_condition.as_string if cfg.loop_condition else ""
        if cond:
            return f"LoopRegion {cfg.label!r} (while {cond})"
        return f"LoopRegion {cfg.label!r}"
    if isinstance(cfg, ConditionalBlock):
        return f"ConditionalBlock {cfg.label!r}"
    if isinstance(cfg, FunctionCallRegion):
        return f"FunctionCallRegion {cfg.label!r}"
    if isinstance(cfg, NamedRegion):
        return f"NamedRegion {cfg.label!r}"
    if isinstance(cfg, ControlFlowRegion):
        return f"ControlFlowRegion {cfg.label!r}"
    return f"{type(cfg).__name__} {getattr(cfg, 'label', '?')!r}"


def _emit_dtype(dtype) -> str:
    if dtype is None:
        return "None"
    if isinstance(dtype, dtypes.typeclass):
        name = dtype.to_string()
        if hasattr(dtypes, name):
            return f"dtypes.{name}"
    raise NotImplementedError(
        f"Cannot emit dtype {dtype!r} ({type(dtype).__name__}) imperatively"
    )


def _emit_dtype_optional(dtype) -> str:
    if dtype is None:
        return "None"
    return _emit_dtype(dtype)


def _emit_storage(storage) -> str:
    if isinstance(storage, dtypes.StorageType):
        return f"dtypes.StorageType.{storage.name}"
    raise NotImplementedError(f"Cannot emit StorageType {storage!r}")


def _emit_lifetime(lifetime) -> str:
    if isinstance(lifetime, dtypes.AllocationLifetime):
        return f"dtypes.AllocationLifetime.{lifetime.name}"
    raise NotImplementedError(f"Cannot emit AllocationLifetime {lifetime!r}")


def _emit_schedule(sched) -> str:
    if isinstance(sched, dtypes.ScheduleType):
        return f"dtypes.ScheduleType.{sched.name}"
    raise NotImplementedError(f"Cannot emit ScheduleType {sched!r}")


def _emit_language(lang) -> str:
    if isinstance(lang, dtypes.Language):
        return f"dtypes.Language.{lang.name}"
    raise NotImplementedError(f"Cannot emit Language {lang!r}")


def _emit_constant_value(value: Any) -> str:
    if isinstance(value, (int, float, bool, str, type(None))):
        return repr(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_emit_constant_value(v) for v in value)
        if isinstance(value, tuple):
            return f"({inner}{',' if len(value) == 1 else ''})"
        return f"[{inner}]"
    raise NotImplementedError(
        f"Cannot emit constant value {value!r} ({type(value).__name__})"
    )


def _emit_symbolic(expr) -> str:
    """Render a sympy / symbolic expression as a Python source *string literal*."""
    if expr is None:
        return "None"
    if isinstance(expr, bool):
        return repr(expr)
    if isinstance(expr, (int, float)):
        return repr(expr)
    if isinstance(expr, str):
        return repr(expr)
    if isinstance(expr, sympy.Basic):
        if expr.is_Integer:
            return repr(int(expr))
        return repr(symbolic.symstr(expr, cpp_mode=False))
    return repr(str(expr))


def _emit_symbolic_inline(expr) -> str:
    """Render a symbolic expression as an inline Python expression (not a string)."""
    if expr is None:
        return "None"
    if isinstance(expr, bool):
        return repr(expr)
    if isinstance(expr, (int, float)):
        return repr(expr)
    if isinstance(expr, sympy.Basic):
        if expr.is_Integer:
            return repr(int(expr))
        return f"symbolic.pystr_to_symbolic({_pyrepr(symbolic.symstr(expr))})"
    return _pyrepr(str(expr))


def _emit_shape(shape) -> str:
    parts = [_emit_symbolic(s) for s in shape]
    if not parts:
        return "()"
    if len(parts) == 1:
        return f"({parts[0]},)"
    return f"({', '.join(parts)})"


def _emit_shape_inline(shape) -> str:
    """Render a shape/stride/offset tuple with each entry as an INLINE symbolic
    expression (wrapped in ``symbolic.pystr_to_symbolic(...)`` when needed),
    not a string literal. Required for ``Array``'s ``strides``/``offset``
    args, which aren't auto-parsed by the constructor.
    """
    parts = [_emit_symbolic_inline(s) for s in shape]
    if not parts:
        return "()"
    if len(parts) == 1:
        return f"({parts[0]},)"
    return f"({', '.join(parts)})"


def _shape_matches(strides, shape) -> bool:
    if len(strides) != len(shape):
        return False
    expected = []
    acc = sympy.Integer(1)
    for dim in reversed(shape):
        expected.append(acc)
        d = dim if isinstance(dim, sympy.Basic) else symbolic.pystr_to_symbolic(str(dim))
        acc = acc * d
    expected = list(reversed(expected))
    for s, e in zip(strides, expected):
        try:
            if symbolic.simplify(s - e) != 0:
                return False
        except Exception:
            if str(s) != str(e):
                return False
    return True


def _offset_is_zero(offset, shape) -> bool:
    if len(offset) != len(shape):
        return False
    for o in offset:
        try:
            if int(o) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _total_size_matches(total_size, shape) -> bool:
    acc = sympy.Integer(1)
    for dim in shape:
        d = dim if isinstance(dim, sympy.Basic) else symbolic.pystr_to_symbolic(str(dim))
        acc = acc * d
    try:
        return symbolic.simplify(total_size - acc) == 0
    except Exception:
        return str(total_size) == str(acc)


def _array_kwargs(desc: dt.Array) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    if desc.transient:
        kwargs["transient"] = "True"
    if desc.storage is not dtypes.StorageType.Default:
        kwargs["storage"] = _emit_storage(desc.storage)
    if desc.location:
        kwargs["location"] = _pyrepr(dict(desc.location))
    if not _shape_matches(desc.strides, desc.shape):
        kwargs["strides"] = _emit_shape_inline(desc.strides)
    if not _offset_is_zero(desc.offset, desc.shape):
        kwargs["offset"] = _emit_shape_inline(desc.offset)
    if desc.lifetime is not dtypes.AllocationLifetime.Scope:
        kwargs["lifetime"] = _emit_lifetime(desc.lifetime)
    if desc.allow_conflicts:
        kwargs["allow_conflicts"] = "True"
    if not _total_size_matches(desc.total_size, desc.shape):
        kwargs["total_size"] = _emit_symbolic(desc.total_size)
    if desc.alignment:
        kwargs["alignment"] = repr(int(desc.alignment))
    if desc.may_alias:
        kwargs["may_alias"] = "True"
    return kwargs


def _arrayview_kwargs(desc) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    if desc.storage is not dtypes.StorageType.Default:
        kwargs["storage"] = _emit_storage(desc.storage)
    if not _shape_matches(desc.strides, desc.shape):
        kwargs["strides"] = _emit_shape_inline(desc.strides)
    if not _offset_is_zero(desc.offset, desc.shape):
        kwargs["offset"] = _emit_shape_inline(desc.offset)
    if desc.allow_conflicts:
        kwargs["allow_conflicts"] = "True"
    if not _total_size_matches(desc.total_size, desc.shape):
        kwargs["total_size"] = _emit_symbolic(desc.total_size)
    if desc.alignment:
        kwargs["alignment"] = repr(int(desc.alignment))
    if desc.may_alias:
        kwargs["may_alias"] = "True"
    return kwargs


def _scalar_kwargs(desc: dt.Scalar) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    if desc.storage is not dtypes.StorageType.Default:
        kwargs["storage"] = _emit_storage(desc.storage)
    if desc.transient:
        kwargs["transient"] = "True"
    if desc.lifetime is not dtypes.AllocationLifetime.Scope:
        kwargs["lifetime"] = _emit_lifetime(desc.lifetime)
    return kwargs


def _stream_kwargs(desc: dt.Stream) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    bs = desc.buffer_size
    try:
        if int(bs) != 1:
            kwargs["buffer_size"] = _emit_symbolic(bs)
    except (TypeError, ValueError):
        kwargs["buffer_size"] = _emit_symbolic(bs)
    shape = tuple(desc.shape)
    if shape != (1,):
        kwargs["shape"] = _emit_shape_inline(shape)
    if desc.storage is not dtypes.StorageType.Default:
        kwargs["storage"] = _emit_storage(desc.storage)
    if desc.transient:
        kwargs["transient"] = "True"
    if not _offset_is_zero(desc.offset, desc.shape):
        kwargs["offset"] = _emit_shape_inline(desc.offset)
    if desc.lifetime is not dtypes.AllocationLifetime.Scope:
        kwargs["lifetime"] = _emit_lifetime(desc.lifetime)
    return kwargs


def _emit_extra_connectors(var: str, node, buf: "_IndentedBuffer"):
    """Emit add_in_connector / add_out_connector calls for a node.

    Used for nodes whose constructor doesn't take connector dicts (e.g.
    MapEntry, MapExit, ConsumeEntry, ConsumeExit) — they're created bare and
    grow connectors as memlets are wired.
    """
    for cname, cdtype in node.in_connectors.items():
        dt_arg = "None" if _is_void_dtype(cdtype) else _emit_dtype(cdtype)
        buf.line(f"{var}.add_in_connector({_pyrepr(cname)}, dtype={dt_arg})")
    for cname, cdtype in node.out_connectors.items():
        dt_arg = "None" if _is_void_dtype(cdtype) else _emit_dtype(cdtype)
        buf.line(f"{var}.add_out_connector({_pyrepr(cname)}, dtype={dt_arg})")


def _is_void_dtype(dtype) -> bool:
    if dtype is None:
        return True
    if isinstance(dtype, dtypes.typeclass) and dtype.type is None:
        return True
    return False


def _emit_connector_dict(conns) -> str:
    """Render in/out connectors. If all dtypes are absent, emit a set of names."""
    items = list(conns.items())
    if not items:
        return "{}"
    if all(_is_void_dtype(v) for _, v in items):
        names = ", ".join(_pyrepr(k) for k, _ in items)
        return "{" + names + "}"
    parts = [
        f"{_pyrepr(k)}: {'None' if _is_void_dtype(v) else _emit_dtype(v)}"
        for k, v in items
    ]
    return "{" + ", ".join(parts) + "}"


def _emit_interstate_edge(edge: InterstateEdge) -> str:
    kwargs: Dict[str, str] = {}
    cond_str = edge.condition.as_string if edge.condition is not None else ""
    trivial = cond_str.strip() in {"", "1", "True", "true"}
    if not trivial:
        kwargs["condition"] = f"CodeBlock({_pyrepr(cond_str)})"
    if edge.assignments:
        items = ", ".join(
            f"{_pyrepr(k)}: {_pyrepr(str(v))}" for k, v in edge.assignments.items()
        )
        kwargs["assignments"] = "{" + items + "}"
    if not kwargs:
        return "InterstateEdge()"
    return _call("InterstateEdge", [], kwargs)


def _subset_range_str(rng_list) -> str:
    """Render a list of (start, end, step[, tile]) tuples as DaCe subset syntax."""
    parts = []
    for entry in rng_list:
        if len(entry) >= 4:
            start, end, step, tile = entry[0], entry[1], entry[2], entry[3]
        else:
            start, end, step = entry[0], entry[1], entry[2]
            tile = 1
        s_part = symbolic.symstr(start)
        e_part = symbolic.symstr(end)
        st_part = symbolic.symstr(step)
        try:
            step_one = int(step) == 1
        except Exception:
            step_one = False
        try:
            tile_one = int(tile) == 1
        except Exception:
            tile_one = False
        if step_one and tile_one:
            parts.append(f"{s_part}:{e_part} + 1")
        elif tile_one:
            parts.append(f"{s_part}:{e_part} + 1:{st_part}")
        else:
            parts.append(f"{s_part}:{e_part} + 1:{st_part}:{symbolic.symstr(tile)}")
    return ", ".join(parts)


def _emit_subset(subset) -> str:
    if subset is None:
        return "None"
    return _pyrepr(str(subset))


# ----------------------------------------------------------------------
# Compact helper kit: pattern recognition + helper definitions
#
# These detect the most common stamped patterns in DaCe codegen output
# (binary-op tasklets, constant-assign tasklets, scalar memlets) so the
# emitter can collapse them to short helper calls. Definitions of the
# helpers themselves are emitted at the top of the generated file when
# their pattern is used often enough.
# ----------------------------------------------------------------------

_BINOP_TASKLET_RE = re.compile(r'^__out = \(__in1 ([+\-*/%]) __in2\)$')
_CONST_TASKLET_RE = re.compile(r'^__out = (.+)$')


def _is_plain_tasklet(node) -> bool:
    """True if a Tasklet has no extra-code / language overrides — i.e. it's
    safe to compress with the helper kit."""
    if node.code.language is not dtypes.Language.Python:
        return False
    if node.code_global and node.code_global.as_string:
        return False
    if node.code_init and node.code_init.as_string:
        return False
    if node.code_exit and node.code_exit.as_string:
        return False
    if node.side_effects is not None:
        return False
    if node.state_fields:
        return False
    return True


def _classify_tasklet(node):
    """Return a ``(kind, payload)`` tuple if ``node`` matches a stamped pattern
    the helper kit can express, else ``None``.

    * ``('const', expr)`` — no inputs, ``__out`` only, body ``__out = <expr>``
      with no ``__in*`` references on the right.
    * ``('binop', op)`` — ``__in1``/``__in2`` inputs, ``__out`` only, body
      ``__out = (__in1 OP __in2)``.
    """
    if not _is_plain_tasklet(node):
        return None
    if not all(_is_void_dtype(v) for v in node.in_connectors.values()):
        return None
    if not all(_is_void_dtype(v) for v in node.out_connectors.values()):
        return None

    in_set = set(node.in_connectors.keys())
    out_set = set(node.out_connectors.keys())
    code = node.code.as_string.strip()

    if not in_set and out_set == {"__out"}:
        m = _CONST_TASKLET_RE.match(code)
        if m and "__in" not in m.group(1):
            return ("const", m.group(1))

    if in_set == {"__in1", "__in2"} and out_set == {"__out"}:
        m = _BINOP_TASKLET_RE.match(code)
        if m:
            return ("binop", m.group(1))

    return None


def _is_scalar_memlet(memlet) -> bool:
    """True for ``Memlet(data='X', subset='0')`` with no other adornments —
    the most common stamped memlet shape in real SDFGs."""
    if memlet is None or memlet.is_empty():
        return False
    if memlet.data is None or memlet.subset is None:
        return False
    if memlet.other_subset is not None:
        return False
    if memlet.dynamic or memlet.wcr is not None:
        return False
    if memlet.wcr_nonatomic or memlet.allow_oob:
        return False
    return str(memlet.subset) == "0"


# Emitted at the top of the generated file when the corresponding pattern
# clears the helper-threshold. Order matches a logical reading order
# (memlet shorthand first, then tasklet shortcuts).
_HELPER_ORDER = ("scalar_memlet", "const_tasklet", "binop_tasklet")

_HELPER_DEFS: Dict[str, List[str]] = {
    "scalar_memlet": [
        "def _S(data):",
        "    \"\"\"Scalar memlet: ``Memlet(data=data, subset='0')``.\"\"\"",
        "    return Memlet(data=data, subset='0')",
    ],
    "const_tasklet": [
        "def _const_tasklet(state, name, expr):",
        "    \"\"\"Tasklet with no inputs and ``__out = <expr>`` as its body.\"\"\"",
        "    return state.add_tasklet(name, {}, {'__out'}, f'__out = {expr}')",
    ],
    "binop_tasklet": [
        "def _binop_tasklet(state, name, op):",
        "    \"\"\"Tasklet with two inputs and ``__out = (__in1 OP __in2)`` as its body.\"\"\"",
        "    return state.add_tasklet(",
        "        name, {'__in1', '__in2'}, {'__out'},",
        "        f'__out = (__in1 {op} __in2)',",
        "    )",
    ],
}


def _emit_memlet(memlet) -> str:
    if memlet is None or memlet.is_empty():
        return "Memlet()"
    kwargs: Dict[str, str] = {}
    if memlet.data is not None:
        kwargs["data"] = _pyrepr(memlet.data)
    if memlet.subset is not None:
        kwargs["subset"] = _emit_subset(memlet.subset)
    if memlet.other_subset is not None:
        kwargs["other_subset"] = _emit_subset(memlet.other_subset)
    if memlet.dynamic:
        kwargs["dynamic"] = "True"
    if memlet.wcr is not None:
        kwargs["wcr"] = _pyrepr(str(memlet.wcr))
    if memlet.wcr_nonatomic:
        kwargs["wcr_nonatomic"] = "True"
    if memlet.allow_oob:
        kwargs["allow_oob"] = "True"
    return _call("Memlet", [], kwargs)


def _emit_value(value) -> str:
    """Render an arbitrary attribute value as Python source.

    Used by the LibraryNode emitter to render constructor args and
    attribute assignments. Falls back to ``NotImplementedError`` for shapes
    we haven't taught it about, rather than guessing.
    """
    if value is None:
        return "None"
    if isinstance(value, dtypes.DebugInfo):
        # DebugInfo carries source-line metadata; pass None and let the
        # constructor synthesise a fresh one. Round-trip equivalence is
        # structural — line numbers don't affect semantics.
        return "None"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float, str)):
        return repr(value)
    if isinstance(value, dtypes.typeclass):
        return _emit_dtype(value)
    if isinstance(value, dtypes.StorageType):
        return _emit_storage(value)
    if isinstance(value, dtypes.AllocationLifetime):
        return _emit_lifetime(value)
    if isinstance(value, dtypes.ScheduleType):
        return _emit_schedule(value)
    if isinstance(value, dtypes.Language):
        return _emit_language(value)
    if isinstance(value, sympy.Basic):
        if value.is_Integer:
            return repr(int(value))
        return f"symbolic.pystr_to_symbolic({_pyrepr(symbolic.symstr(value))})"
    if isinstance(value, CodeBlock):
        return f"CodeBlock({_pyrepr(value.as_string)})"
    if isinstance(value, (list, tuple)):
        rendered = [_emit_value(v) for v in value]
        if isinstance(value, tuple):
            inner = ", ".join(rendered)
            return f"({inner}{',' if len(rendered) == 1 else ''})"
        return "[" + ", ".join(rendered) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_emit_value(k)}: {_emit_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    raise NotImplementedError(
        f"Cannot emit value {value!r} of type {type(value).__name__} as "
        f"imperative Python; teach to_python._emit_value how to render it"
    )


def _read_node_attr(node, name: str):
    """Read an attribute from a LibraryNode for ctor reconstruction.

    Tries ``getattr`` first, then falls back to scanning the property table
    for differently-cased names. Returns ``_MISSING`` if not found.
    """
    if hasattr(node, name):
        return getattr(node, name)
    # Some constructors use 'name' for what's stored as 'label'.
    if name == "name" and hasattr(node, "label"):
        return node.label
    return _MISSING


def _attr_is_settable(node, name: str) -> bool:
    cls = type(node)
    for klass in cls.__mro__:
        if name in klass.__dict__:
            attr = klass.__dict__[name]
            if isinstance(attr, property) and attr.fset is not None:
                return True
            # DaCe Property descriptors implement __set__.
            if hasattr(attr, "__set__"):
                return True
    # Fall back: instance attribute that we can clobber.
    return name in getattr(node, "__dict__", {})


def _unwrap_make_properties_init(cls):
    """Return the original ``__init__`` of ``cls``, peeling away the
    ``@make_properties`` wrapper inserted by ``dace.properties.make_properties``.

    The wrapper is ``initialize_properties(obj, *args, **kwargs)`` with a
    closure containing the real init function as one of its cells. If ``cls``
    isn't decorated, returns ``cls.__init__`` unchanged.
    """
    init = cls.__init__
    qualname = getattr(init, "__qualname__", "")
    if "initialize_properties" not in qualname:
        return init
    closure = getattr(init, "__closure__", None) or ()
    for cell in closure:
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if callable(v) and "initialize_properties" not in getattr(v, "__qualname__", ""):
            # Recurse — base classes may have been wrapped too.
            tmp_qualname = getattr(v, "__qualname__", "")
            if tmp_qualname.endswith(".__init__"):
                return v
    return init


def _values_equal(a, b) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        if isinstance(a, sympy.Basic) or isinstance(b, sympy.Basic):
            return symbolic.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        pass
    try:
        return a == b
    except Exception:
        return False


# ----------------------------------------------------------------------
# CLI entry point — ``python -m dace.sdfg.to_python <sdfg> [-o out.py]``
# ----------------------------------------------------------------------


def _main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m dace.sdfg.to_python",
        description=(
            "Emit a Python source file that reconstructs the given SDFG via "
            "DaCe's imperative public API. The emitted file defines "
            "``build_sdfg()`` and validates the rebuilt SDFG when run."
        ),
    )
    parser.add_argument(
        "sdfg_path",
        help="Path to a .sdfg / .sdfgz file (DaCe's serialized SDFG format)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Python file path (default: write to stdout)",
    )
    args = parser.parse_args()

    sdfg = SDFG.from_file(args.sdfg_path)
    source = sdfg_to_python(sdfg)
    if args.output:
        with open(args.output, "w") as f:
            f.write(source)
        print(
            f"Wrote {len(source.splitlines())} lines to {args.output}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(source)


if __name__ == "__main__":
    _main()
