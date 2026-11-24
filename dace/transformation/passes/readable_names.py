# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ReadableNames``: short identifiers for an SDFG that is rendered as source for a reader.

Each SDFG of the tree renders as one C function, so names only have to be unique per SDFG. Generated names (those
starting with ``_``) are renamed; names the program spelled are kept, and so are the top-level arguments (the ABI):

* loop iterators and map parameters become ``i, j, k, ...`` by nesting depth (``__dace_tile`` becomes ``tile``,
  ``__c`` becomes ``c``);
* a nested SDFG's symbol or container bound to a plain name of its parent takes that name;
* other generated containers and symbols drop their underscores and pass prefixes (``__gather0_o`` -> ``gather0``);
* a map bound that several maps of a nested SDFG repeat is assigned once to ``lo`` / ``hi`` at its start;
* every loop counter and map parameter is unique in the rendered file (``i``, then ``i2``), because the loop comments
  and the unit header name loops by them;
* each nested SDFG gets a short function name (:attr:`NestedSDFG.unique_name`), shared by equal bodies.
"""

import ast
import re
from collections.abc import Iterable

import sympy

from dace import SDFG, SDFGState, dtypes, symbolic
from dace.ordered import OrderedSet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict, replace_properties_dict
from dace.sdfg.sdfg import interstate_bound_names
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, StateSubgraphView
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.cpu_specialization.strip_mine_interchange import TILE_PARAM
from dace.transformation.passes.unique_names import rename_connector, rename_strings, sdfg_names
from dace.transformation.transformation import explicit_cf_compatible

#: Iteration variable names, outermost first.
LETTERS = ("i", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w")
#: Names for DaCe-reserved symbols that read better than their stripped form.
PREFERRED = {TILE_PARAM: "tile", symbolic.NUM_THREADS_SYMBOL: "nthreads"}
#: Prefixes passes put in front of a container name they derive, stripped after the underscores.
PASS_PREFIXES = ("map_fusion_split_", "map_fusion_")
#: A derived name longer than this becomes ``tmp``.
MAX_STEM = 24
#: Function names that say nothing about the body.
GENERIC_FUNCTIONS = ("sdfg", "nested_sdfg", "nested", "")
#: Return containers keep their name: the entry and the frontend find them by it.
RETURN_PREFIX = "__return"
#: Keywords of C and C++ and the library names a rendered unit may call.
RESERVED = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if inline int long "
    "register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while "
    "bool true false alignas alignof and asm catch class constexpr delete explicit export friend mutable namespace "
    "new noexcept not operator or private protected public template this throw try typeid typename using virtual "
    "xor nullptr NULL abs fabs min max fmin fmax sqrt cbrt exp exp2 log log2 log10 pow sin cos tan asin acos atan "
    "atan2 sinh cosh tanh floor ceil round trunc fmod isnan isinf INFINITY NAN assert printf malloc calloc free "
    "memcpy memset abort exit main std dace omp I".split()
)
RESERVED_PREFIXES = ("cpf_", "omp_", "dace_", "__")


def is_generated(name: str) -> bool:
    return name.startswith("_") and not name.startswith(RETURN_PREFIX)


def is_free(name: str, taken: OrderedSet[str]) -> bool:
    return name not in taken and name not in RESERVED and not name.startswith(RESERVED_PREFIXES)


def fresh(stem: str, taken: OrderedSet[str]) -> str:
    """``stem``, or ``stem<N>`` (``stem_<N>`` after a digit), whichever is first free; it joins ``taken``. A stem
    with a reserved prefix (``cpf_tmp``) has no free numbered form, so it becomes ``tmp``."""
    if stem.startswith(RESERVED_PREFIXES):
        stem = "tmp"
    name, n = stem, 1
    while not is_free(name, taken):
        name = f"{stem}_{n}" if stem[-1].isdigit() else f"{stem}{n}"
        n += 1
    taken.add(name)
    return name


def container_stem(name: str) -> str:
    """``name`` without leading underscores, pass prefixes and a one-letter tail (``__gather0_o`` -> ``gather0``)."""
    stem = name.lstrip("_")
    for prefix in PASS_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :].lstrip("_")
    stem = re.sub(r"_+", "_", stem).strip("_")
    stem = re.sub(r"(\d)_[a-z]$", r"\1", stem)
    if len(stem) > MAX_STEM or not stem or not stem[0].isalpha():
        return "tmp"
    return stem


def function_stem(name: str) -> str:
    """A function name from a nested SDFG's name: ``nested_BinOp_75`` -> ``binop``, ``nested_sdfg`` -> ``body``."""
    stem = re.sub(r"(_\d+)+$", "", name.strip("_")).lower()
    if stem in GENERIC_FUNCTIONS:
        return "body"
    stem = stem.removeprefix("nested_")
    return container_stem(stem) if stem else "body"


def iteration_name(depth: int, avoid: OrderedSet[str], old: str) -> str:
    """The name ``old`` stands for (``__dace_tile`` -> ``tile``, ``__c`` -> ``c``, else ``c1``), or else the letter
    for nesting ``depth``, the next free one, or a numbered letter."""
    stripped = old.lstrip("_")
    preferred = PREFERRED.get(old) or (stripped if len(stripped) == 1 and stripped.isalpha() else None)
    if preferred is not None:
        return next(name for name in (preferred, *(f"{preferred}{n}" for n in range(1, 100))) if is_free(name, avoid))
    order = LETTERS[depth:] + LETTERS[:depth]
    for suffix in ("", *map(str, range(1, 100))):
        for letter in order:
            if is_free(letter + suffix, avoid):
                return letter + suffix
    raise RuntimeError("no free iteration variable name")


def code_identifiers(node: nodes.CodeNode) -> OrderedSet[str]:
    """The identifiers a tasklet's code spells."""
    if not isinstance(node, nodes.Tasklet):
        return OrderedSet()
    text = node.code.as_string
    if node.language is dtypes.Language.Python:
        return OrderedSet(n.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name))
    return OrderedSet(re.findall(r"[A-Za-z_]\w*", text))


def function_namespace(sdfg: SDFG) -> OrderedSet[str]:
    """Every name the function rendered from ``sdfg`` may hold, iteration variables included."""
    names = OrderedSet(sdfg_names(sdfg))
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                names.update(node.map.params)
            if isinstance(node, (nodes.CodeNode, nodes.EntryNode)):
                names.update(node.in_connectors)
                names.update(node.out_connectors)
            if isinstance(node, nodes.CodeNode):
                names.update(code_identifiers(node))
    return names


def rename_cpp_code(graph_nodes: Iterable[nodes.Node], repl: dict[str, str]) -> None:
    """Rename identifiers in C++ tasklet code by token: the body, and the code the tasklet places at file scope and
    around the program. ``replace_dict`` would shadow them with ``auto old = new;``, which at file scope reads a
    name that does not exist there (an external call's prototype naming a loop iterator: ``auto _loop_it_1 = i;``)."""
    if not repl:
        return
    pattern = re.compile(rf"\b({'|'.join(map(re.escape, repl))})\b")
    for node in graph_nodes:
        if isinstance(node, nodes.Tasklet) and node.language is dtypes.Language.CPP:
            own = {*node.in_connectors, *node.out_connectors}
            text = pattern.sub(lambda m: m.group(1) if m.group(1) in own else repl[m.group(1)], node.code.as_string)
            if text != node.code.as_string:
                node.code = CodeBlock(text, dtypes.Language.CPP)
            node.code_global = renamed_block(node.code_global, pattern, repl)
            node.code_init = renamed_block(node.code_init, pattern, repl)
            node.code_exit = renamed_block(node.code_exit, pattern, repl)


def renamed_block(block: CodeBlock, pattern: re.Pattern[str], repl: dict[str, str]) -> CodeBlock:
    """``block`` with every name of ``repl`` renamed by token; ``block`` itself when none occurs."""
    text = block.as_string
    renamed = pattern.sub(lambda m: repl[m.group(1)], text) if text else text
    return block if renamed == text else CodeBlock(renamed, block.language)


def rewrite_text(text: str | None, repl: dict[str, str]) -> str | None:
    if not text or not repl:
        return text
    pattern = re.compile(rf"\b({'|'.join(map(re.escape, repl))})\b")
    return pattern.sub(lambda m: repl[m.group(1)], text)


def read_after(sdfg: SDFG, loop: LoopRegion, name: str) -> bool:
    """Whether ``sdfg`` reads ``name`` outside ``loop``: the iterator's final value is live after it (a nest that
    exports it), so a rename inside the loop alone would leave that read naming nothing."""
    inside = OrderedSet(loop.all_control_flow_blocks())
    inside_edges = OrderedSet(loop.all_interstate_edges())
    blocks = (b for b in sdfg.all_control_flow_blocks() if b not in inside and isinstance(b, SDFGState))
    edges = (e for e in sdfg.all_interstate_edges() if e not in inside_edges)
    return any(name in b.free_symbols for b in blocks) or any(name in e.data.free_symbols for e in edges)


def bare_name(value: object) -> str | None:
    text = str(value)
    return text if text.isidentifier() else None


class FunctionRenamer:
    """Renames the names of one SDFG, which renders as one function."""

    def __init__(self, sdfg: SDFG, parent: tuple[SDFGState, nodes.NestedSDFG] | None, outer_symbols: OrderedSet[str]):
        self.sdfg = sdfg
        self.parent = parent
        self.taken = function_namespace(sdfg)
        #: Symbols of the enclosing SDFGs: validation refuses a scope that binds one of them anywhere below.
        self.outer_symbols = outer_symbols
        #: Container and symbol renames, valid in the whole function.
        self.names: dict[str, str] = {}
        #: Each renamed scope with the iteration renames visible in it.
        self.scopes: list[tuple[nodes.MapEntry | LoopRegion, dict[str, str]]] = []
        #: Every iteration rename; an old name renamed two ways maps to ``None``.
        self.iterations: dict[str, str | None] = {}

    def run(self) -> None:
        self.rename_data_and_symbols()
        self.walk(self.sdfg, [], {}, dict(self.sdfg.symbols))
        self.hoist_bounds()
        self.rewrite_hints()

    def imported(self) -> dict[str, str]:
        """Generated names of this nested SDFG that hold a plain-named symbol or container of the parent."""
        if self.parent is None:
            return {}
        state, node = self.parent
        found: dict[str, str] = {}
        for inner, outer in node.symbol_mapping.items():
            name = bare_name(outer)
            if is_generated(inner) and name is not None and not is_generated(name):
                found[inner] = name
        for conn in OrderedSet([*node.in_connectors, *node.out_connectors]):
            edges = [*state.in_edges_by_connector(node, conn), *state.out_edges_by_connector(node, conn)]
            outer = OrderedSet(e.data.data for e in edges if not e.data.is_empty())
            if is_generated(conn) and len(outer) == 1 and not is_generated(outer[0]):
                found[conn] = outer[0]
        return found

    def rename_data_and_symbols(self) -> None:
        sdfg, nested = self.sdfg, self.parent is not None
        loops = {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)}
        candidates = [name for name, desc in sdfg.arrays.items() if is_generated(name) and (desc.transient or nested)]
        bound = interstate_bound_names(sdfg)
        symbols = OrderedSet(sdfg.symbols) | OrderedSet(sorted(bound))
        candidates += [
            s
            for s in symbols
            if is_generated(s) and s not in loops and (nested or (s in bound and s != symbolic.NUM_THREADS_SYMBOL))
        ]
        imported = self.imported()
        repl: dict[str, str] = {}
        for name in candidates:
            wanted = imported.get(name)
            if wanted is not None and is_free(wanted, self.taken):
                self.taken.add(wanted)
                repl[name] = wanted
            else:
                repl[name] = fresh(PREFERRED.get(name) or container_stem(name), self.taken)
        if not repl:
            return
        with symbolic.serialization_symbol_dtypes({repl[s]: t for s, t in sdfg.symbols.items() if s in repl}, True):
            rename_cpp_code((n for state in sdfg.states() for n in state.nodes()), repl)
            sdfg.replace_dict(repl)
        rename_strings(sdfg, repl)
        if self.parent is not None:
            state, node = self.parent
            for old, new in repl.items():
                rename_connector(state, node, old, new)
                if old in node.symbol_mapping:
                    node.symbol_mapping[new] = node.symbol_mapping.pop(old)
        self.names.update(repl)

    def record(self, scope: nodes.MapEntry | LoopRegion, visible: dict[str, str], own: dict[str, str]) -> None:
        self.scopes.append((scope, {**visible, **own}))
        # A plain name renamed for shadowing an enclosing counter means that counter elsewhere: scoped only.
        for old, new in ((old, new) for old, new in own.items() if is_generated(old)):
            self.iterations[old] = new if self.iterations.get(old, new) == new else None

    def walk(
        self, region: ControlFlowRegion, enclosing: list[str], visible: dict[str, str], table: dict[str, object]
    ) -> None:
        for block in region.nodes():
            if isinstance(block, LoopRegion):
                own: dict[str, str] = {}
                inner_table = {**table, **block.new_symbols(table)}
                name = block.loop_variable
                if name and is_generated(name) and not read_after(self.sdfg, block, name):
                    own[name] = iteration_name(
                        len(enclosing), self.taken | self.outer_symbols | OrderedSet(enclosing), name
                    )
                    self.rename_loop(block, name, own[name], inner_table.get(name))
                    inner_table[own[name]] = inner_table.pop(name, None)
                self.record(block, visible, own)
                names = [*enclosing, block.loop_variable] if block.loop_variable else enclosing
                self.walk(block, names, {**visible, **own}, inner_table)
            elif isinstance(block, ConditionalBlock):
                for branch in block.sub_regions():
                    self.walk(branch, enclosing, visible, table)
            elif isinstance(block, ControlFlowRegion):
                self.walk(block, enclosing, visible, table)
            elif isinstance(block, SDFGState):
                self.rename_maps(block, enclosing, visible, table)

    def rename_loop(self, loop: LoopRegion, old: str, new: str, dtype: object) -> None:
        repl = {old: new}
        sdfg = self.sdfg
        with symbolic.serialization_symbol_dtypes({new: dtype} if dtype is not None else {}, True):
            rename_cpp_code((n for state in loop.states() for n in state.nodes()), repl)
            loop.replace_meta_accesses(repl)
            loop.replace_dict(repl)
            for name in OrderedSet(n.data for state in loop.states() for n in state.data_nodes()):
                desc = sdfg.arrays[name]
                if old in {str(s) for s in desc.free_symbols}:
                    replace_properties_dict(desc, repl)
        for state in loop.states():
            rename_strings(StateSubgraphView(state, state.nodes()), repl)

    def rename_maps(
        self, state: SDFGState, enclosing: list[str], visible: dict[str, str], table: dict[str, object]
    ) -> None:
        scope = state.scope_dict()
        entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
        depth: dict[nodes.Node, list[str]] = {}
        seen: dict[nodes.Node, dict[str, str]] = {}

        def chain(entry: nodes.MapEntry) -> list[nodes.MapEntry]:
            found, parent = [], scope[entry]
            while parent is not None:
                found.insert(0, parent)
                parent = scope[parent]
            return found

        for entry in sorted(entries, key=lambda e: len(chain(e))):
            outer = chain(entry)
            names = [*enclosing, *(p for parent in outer for p in parent.map.params)]
            outer_visible = {**visible, **(seen[outer[-1]] if outer else {})}
            entry_table = dict(table)
            for parent in outer:
                entry_table.update(parent.new_symbols(self.sdfg, state, entry_table))
            declared = entry.new_symbols(self.sdfg, state, entry_table)
            own: dict[str, str] = {}
            for param in entry.map.params:
                # A parameter named like an enclosing loop's counter shadows it in the rendering.
                if is_generated(param) or param in names:
                    avoid = self.taken | self.outer_symbols | OrderedSet(names) | OrderedSet(own.values())
                    own[param] = iteration_name(len(names), avoid, param)
                names.append(own.get(param, param))
            depth[entry] = names
            seen[entry] = {**outer_visible, **own}
            self.record(entry, outer_visible, own)
            if not own:
                continue
            subgraph = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            with symbolic.serialization_symbol_dtypes({own[p]: declared[p] for p in own if p in declared}, True):
                rename_cpp_code(subgraph.nodes(), own)
                replace_dict(subgraph, own)
            rename_strings(subgraph, own)

    def hoist_bounds(self) -> None:
        """Assign a map bound that several maps repeat to ``lo`` / ``hi`` once, at the start of the function."""
        sdfg = self.sdfg
        params = OrderedSet(
            p for state in sdfg.states() for n in state.nodes() if isinstance(n, nodes.MapEntry) for p in n.map.params
        )
        invariant = OrderedSet(sdfg.symbols) - OrderedSet(sorted(interstate_bound_names(sdfg))) - params
        found: dict[tuple[str, sympy.Basic], list[tuple[nodes.MapEntry, int]]] = {}
        types: dict[tuple[str, sympy.Basic], object] = {}
        for state in sdfg.states():
            for entry in (n for n in state.nodes() if isinstance(n, nodes.MapEntry)):
                declared: dict[str, object] | None = None
                for dim, bounds in enumerate(entry.map.range):
                    for kind, expr in (("lo", bounds[0]), ("hi", bounds[1] + 1)):
                        if not hoistable(expr, invariant):
                            continue
                        if declared is None:
                            declared = entry.new_symbols(sdfg, state, dict(sdfg.symbols))
                        found.setdefault((kind, expr), []).append((entry, dim))
                        types.setdefault((kind, expr), declared[entry.map.params[dim]])
        repeated = [key for key, uses in found.items() if len(uses) > 1]
        if not repeated:
            return
        assignments: dict[str, str] = {}
        for kind, expr in repeated:
            name = fresh(kind, self.taken)
            sdfg.add_symbol(name, types[(kind, expr)])
            assignments[name] = symbolic.symstr(expr, cpp_mode=False)
            sym = symbolic.symbol(name, types[(kind, expr)])
            for entry, dim in found[(kind, expr)]:
                begin, end, step = entry.map.range[dim]
                if kind == "lo":
                    entry.map.range[dim] = (sym, end, step)
                else:
                    entry.map.range[dim] = (begin, sym - 1, step)
        sdfg.add_state_before(sdfg.start_block, "bounds", is_start_block=True, assignments=assignments)

    def rewrite_hints(self) -> None:
        unambiguous = {old: new for old, new in self.iterations.items() if new is not None}
        common = {**unambiguous, **self.names}
        scoped = {id(scope): repl for scope, repl in self.scopes}
        for block in self.sdfg.all_control_flow_blocks():
            block.specialization_hint = rewrite_text(block.specialization_hint, {**common, **scoped.get(id(block), {})})
        for state in self.sdfg.states():
            for node in state.nodes():
                repl = {**common, **scoped.get(id(node), {})}
                node.specialization_hint = rewrite_text(node.specialization_hint, repl)


def hoistable(expr: object, invariant: OrderedSet[str]) -> bool:
    """Whether ``expr`` is a compound expression over names fixed for the whole function that cannot trap."""
    if not isinstance(expr, sympy.Basic) or expr.is_Atom:
        return False
    if sympy.count_ops(expr) < 2:
        return False
    return all(str(s) in invariant for s in expr.free_symbols) and cannot_trap(expr)


def cannot_trap(expr: sympy.Basic) -> bool:
    """Whether evaluating ``expr`` is safe on every path: the hoisted bound runs at function entry, also where a
    branch or an abort guard kept the map from running (``if M > 0`` around ``0:N // M``), so a division by a symbol
    is refused."""
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, (symbolic.int_floor, symbolic.int_ceil, symbolic.CMod, sympy.Mod)):
            if not (node.args[1].is_Number and node.args[1] != 0):
                return False
        elif isinstance(node, sympy.Pow):
            if not (node.exp.is_Integer and node.exp >= 0):
                return False
        elif not (node.is_Atom or isinstance(node, (sympy.Add, sympy.Mul, sympy.Max, sympy.Min))):
            return False
    return True


def rename_tree(sdfg: SDFG, parent: tuple[SDFGState, nodes.NestedSDFG] | None, outer_symbols: OrderedSet[str]) -> None:
    FunctionRenamer(sdfg, parent, outer_symbols).run()
    inner_symbols = outer_symbols | OrderedSet(sdfg.symbols)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                rename_tree(node.sdfg, (state, node), inner_symbols)


def name_functions(sdfg: SDFG) -> None:
    """Give every nested SDFG a function name; nested SDFGs with equal bodies share one."""
    taken = OrderedSet([sdfg.name])
    for nested in sdfg.all_sdfgs_recursive():
        taken |= function_namespace(nested)
    by_body: dict[str, str] = {}
    for nested in sdfg.all_sdfgs_recursive():
        node = nested.parent_nsdfg_node
        if node is not None:
            body = node.sdfg.hash_sdfg()
            if body not in by_body:
                by_body[body] = fresh(function_stem(node.sdfg.name), taken)
            node.unique_name = by_body[body]


def iteration_scopes(region: ControlFlowRegion) -> list[LoopRegion | tuple[SDFGState, nodes.MapEntry]]:
    """The loops and maps of one function, outermost first, in the order the rendering emits them."""
    found: list[LoopRegion | tuple[SDFGState, nodes.MapEntry]] = []
    for block in region.nodes():
        if isinstance(block, LoopRegion):
            if block.loop_variable:
                found.append(block)
            found += iteration_scopes(block)
        elif isinstance(block, ConditionalBlock):
            for branch in block.sub_regions():
                found += iteration_scopes(branch)
        elif isinstance(block, ControlFlowRegion):
            found += iteration_scopes(block)
        elif isinstance(block, SDFGState):
            found += [(block, node) for node in block.nodes() if isinstance(node, nodes.MapEntry)]
    return found


def numbered(name: str, taken: OrderedSet[str]) -> str:
    """``name2``, ``name3``, ... (``name_2`` after a digit), the first free one; it joins ``taken``."""
    n = 2
    while not is_free(f"{name}_{n}" if name[-1].isdigit() else f"{name}{n}", taken):
        n += 1
    found = f"{name}_{n}" if name[-1].isdigit() else f"{name}{n}"
    taken.add(found)
    return found


def rename_scope(sdfg: SDFG, scope: LoopRegion | tuple[SDFGState, nodes.MapEntry], repl: dict[str, str]) -> None:
    """Rename the iterators ``repl`` names in one loop or map, its body and the hints inside it."""
    if isinstance(scope, LoopRegion):
        (old,) = repl
        dtype = scope.new_symbols(dict(sdfg.symbols)).get(old)
        FunctionRenamer(sdfg, None, OrderedSet()).rename_loop(scope, old, repl[old], dtype)
        holders: list[object] = [*scope.all_control_flow_blocks(), scope]
        holders += [n for state in scope.states() for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    else:
        state, entry = scope
        declared = entry.new_symbols(sdfg, state, dict(sdfg.symbols))
        subgraph = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        with symbolic.serialization_symbol_dtypes({new: declared[old] for old, new in repl.items()}, True):
            rename_cpp_code(subgraph.nodes(), repl)
            replace_dict(subgraph, repl)
        rename_strings(subgraph, repl)
        holders = [n for n in subgraph.nodes() if isinstance(n, nodes.MapEntry)]
    for holder in holders:
        holder.specialization_hint = rewrite_text(holder.specialization_hint, repl)


def unique_iterators(sdfg: SDFG) -> None:
    """Make every loop counter and map parameter unique in the rendered file: the first keeps its name, a later one
    gets a number (``i``, ``i2``). Loop comments and the unit header name loops by their counters (seidel_2d had two
    loops ``i`` in two functions). Equal nested bodies are renamed alike, so they still share one function."""
    taken = OrderedSet(name for nested in sdfg.all_sdfgs_recursive() for name in function_namespace(nested))
    used: OrderedSet[str] = OrderedSet()
    bodies = {id(nested): nested.hash_sdfg() for nested in sdfg.all_sdfgs_recursive()}
    done: dict[str, list[dict[str, str]]] = {}
    for nested in sdfg.all_sdfgs_recursive():
        scopes = iteration_scopes(nested)
        plan = done.get(bodies[id(nested)]) if nested is not sdfg else None
        if plan is None:
            plan = []
            for scope in scopes:
                names = [scope.loop_variable] if isinstance(scope, LoopRegion) else list(scope[1].map.params)
                # A counter read after its loop keeps its name: the reads outside would name nothing
                kept = isinstance(scope, LoopRegion) and read_after(nested, scope, scope.loop_variable)
                plan.append({name: numbered(name, taken) for name in names if name in used and not kept})
                used.update(plan[-1].get(name, name) for name in names)
            done[bodies[id(nested)]] = plan
        for scope, repl in zip(scopes, plan):
            if repl:
                rename_scope(nested, scope, repl)


@explicit_cf_compatible
class ReadableNames(ppl.Pass):
    """Rename generated names to short readable ones, function by function. Requires the code generator's
    ``compiler.unique_functions`` set to ``unique_name`` for the function names to take effect."""

    CATEGORY: str = "Cleanup"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> None:
        rename_tree(sdfg, None, OrderedSet())
        unique_iterators(sdfg)
        name_functions(sdfg)
        sdfg.reset_cfg_list()
