#!/usr/bin/env python3
"""
hlfir_to_sdfg.py — Walk the HLFIR AST to directly construct a DaCe SDFG.

Pipeline:
    flang-20 -fc1 -emit-hlfir code.f90 -o code.hlfir
    sdfg = generate_sdfg("code.hlfir")   # → dace.SDFG, validated

Architecture:
    SDFGBuilder parses the HLFIR via the C++ bridge (hlfir_bridge.so), runs
    the default pass pipeline (currently: shape propagation), then walks the
    recursive ASTNode tree to emit DaCe constructs:

        ASTNode kind="assign"      → Tasklet in current state
        ASTNode kind="loop"        → LoopRegion (nested for-loop)
        ASTNode kind="conditional" → branching states (TODO)
        ASTNode kind="call"        → nested SDFG / library node (TODO)

    Each Fortran DO loop becomes a dace.sdfg.state.LoopRegion, preserving the
    exact Fortran bounds (e.g. DO jk=1,nlev → for jk=1; jk<nlev+1; jk++).
    Nested DOs become nested LoopRegions.  The innermost loop body contains a
    single SDFGState with one Tasklet per assignment.

Passes:
    The bridge exposes every registered pass by mlir-opt name.  The default
    pipeline is DEFAULT_PIPELINE below.  Override per-call:
        generate_sdfg("code.hlfir", pipeline="builtin.module()")  # no passes
        generate_sdfg("code.hlfir", pipeline="builtin.module("
                     "hlfir-propagate-shapes,my-other-pass)")

Variable classification (from the C++ bridge):
    symbol    — scalar in array shapes / loop bounds  → dace.symbol
    scalar    — other scalars                         → dace.data.Scalar
    array     — rank > 0                              → dace.data.Array
    loop_iter — DO induction variable                 → LoopRegion loop_var

State-change rules (what triggers a new SDFGState):
    - LoopRegion or conditional → always a new state/region
    - Write to a symbol         → interstate edge with assignment
    - Write to a scalar         → Tasklet in CURRENT state (no state change)

NOTE on nanobind bindings:
    Every read of a std::vector-typed attribute (e.g. ast_node.children,
    var.shape_symbols, assign.accesses) returns a FRESH Python list copy.
    Hot paths cache such attributes into locals.
"""

import re
import dace
from dace import SDFG, Memlet, InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes as nd
from build_bridge import hb

# The default pipeline run before AST/variable extraction.  Shape
# propagation fills in assumed-shape (:,:) dummies with real Fortran
# names wherever callers supplied them.
DEFAULT_PIPELINE = ("hlfir-inline-all,"
                    "hlfir-flatten-structs,"
                    "hlfir-propagate-shapes")


class SDFGBuilder:
    """Walks the HLFIR ASTNode tree and emits a DaCe SDFG."""

    DTYPE = {
        'float64': dace.float64,
        'float32': dace.float32,
        'int32': dace.int32,
        'int64': dace.int64,
    }

    def __init__(self, hlfir_path: str, pipeline: str = DEFAULT_PIPELINE):
        """Parse HLFIR, run the pass pipeline, and classify variables.

        After construction, the following dicts are available:
            self.arrays   — {name: VarInfo} for rank>0 variables
            self.symbols  — {name: VarInfo} for scalars used in shapes/bounds
            self.scalars  — {name: VarInfo} for other scalars (c1, c2, …)
        """
        self.module = hb.HLFIRModule()
        if not self.module.parse_file(hlfir_path):
            raise RuntimeError(f"Cannot parse {hlfir_path}")

        # Run bridge passes (shape propagation by default) BEFORE extracting
        # variables, so assumed-shape dummies pick up real names.
        if pipeline:
            self.module.run_passes(pipeline)

        self.variables = self.module.get_variables()
        self.ast = self.module.get_ast()
        self.arrays = {v.fortran_name: v for v in self.variables if v.role == "array"}
        self.symbols = {v.fortran_name: v for v in self.variables if v.role == "symbol"}
        self.scalars = {v.fortran_name: v for v in self.variables if v.role == "scalar"}

    def build(self) -> SDFG:
        """Construct and return a validated SDFG."""
        self._id_counter = 0
        sdfg = SDFG(self._name())
        self._add_descriptors(sdfg)
        ctx = _Ctx(sdfg, self)
        self._emit(ctx, self.ast, sdfg)
        ctx.flush(self)
        return sdfg

    def nid(self) -> int:
        """Globally unique integer.  Shared across _Ctx instances so that
        loop variable names (jk_0, jc_1, jk_2, …) never collide."""
        i = self._id_counter
        self._id_counter += 1
        return i

    # ------------------------------------------------------------------
    # SDFG name / type mapping
    # ------------------------------------------------------------------

    def _name(self) -> str:
        """Derive the SDFG name from the Flang mangled name."""
        for v in self.arrays.values():
            mn = v.mangled_name
            if '_QF' in mn and 'E' in mn:
                return mn.split('_QF')[1].split('E')[0]
        return "sdfg"

    def _dt(self, s: str) -> dace.typeclass:
        return self.DTYPE.get(s, dace.float64)

    # ------------------------------------------------------------------
    # Phase 2: Register data descriptors on the SDFG
    # ------------------------------------------------------------------

    def _add_descriptors(self, sdfg: SDFG):
        """Add symbols, arrays, and scalars to the SDFG."""
        # Named Fortran symbols (nproma, nlev, …).
        for v in self.symbols.values():
            sdfg.add_symbol(v.fortran_name, self._dt(v.dtype))

        # Synthetic symbols for dims that stayed unresolved after passes.
        known = {v.fortran_name for v in self.variables}
        for v in self.arrays.values():
            syms = v.shape_symbols
            for s in syms:
                if s not in known and s not in sdfg.symbols:
                    sdfg.add_symbol(s, dace.int64)

        # Arrays.
        for v in self.arrays.values():
            sdfg.add_array(
                v.fortran_name,
                shape=[dace.symbol(s) for s in v.shape_symbols],
                dtype=self._dt(v.dtype),
                transient=(v.intent == ''),
            )

        # Scalars as data nodes (not symbols).
        for v in self.scalars.values():
            sdfg.add_scalar(
                v.fortran_name,
                dtype=self._dt(v.dtype),
                transient=True,
            )

    # ------------------------------------------------------------------
    # Phase 3: Recursive AST walk → SDFG state machine
    # ------------------------------------------------------------------

    def _emit(self, ctx: '_Ctx', nodes: list, region):
        for n in nodes:
            if n.kind == "assign": self._emit_assign(ctx, n, region)
            elif n.kind == "loop": self._emit_loop(ctx, n, region)
            elif n.kind == "conditional": self._emit_cond(ctx, n, region)
            # "call" → TODO: nested SDFG or library node

    def _emit_assign(self, ctx: '_Ctx', n, region):
        """Scalar or symbol assignment."""
        if n.target in self.symbols:
            ctx.flush(self)
            ctx.ensure(region)
            dst = region.add_state(f"post_{n.target}_{self.nid()}")
            region.add_edge(ctx.cur, dst, InterstateEdge(assignments={n.target: n.expr}))
            ctx.cur = dst
        elif not n.target_is_array:
            ctx.pending.append((n.target, n.expr))

    def _emit_loop(self, ctx: '_Ctx', n, region, iter_map=None):
        """Fortran DO loop → LoopRegion with exact Fortran bounds."""
        ctx.flush(self)
        if iter_map is None:
            iter_map = {}

        uid = f"{n.loop_iter}_{self.nid()}"
        iter_map = {**iter_map, n.loop_iter: uid}

        bound = n.loop_bound
        lower = n.loop_lower if n.loop_lower >= 0 else 1

        loop = LoopRegion(
            label=f"loop_{uid}",
            condition_expr=f"{uid} < {bound} + 1",
            loop_var=uid,
            initialize_expr=f"{uid} = {lower}",
            update_expr=f"{uid} = {uid} + 1",
        )
        region.add_node(loop)
        if ctx.cur is not None:
            region.add_edge(ctx.cur, loop, InterstateEdge())
        ctx.cur = loop

        # Cache .children once — nanobind copies on every access.
        children = n.children
        child_loops = [c for c in children if c.kind == "loop"]
        child_assigns = [c for c in children if c.kind == "assign"]

        if child_loops:
            inner_ctx = _Ctx(ctx.sdfg, self)
            for c in children:
                if c.kind == "loop":
                    self._emit_loop(inner_ctx, c, loop, iter_map)
                elif c.kind == "assign":
                    self._emit_assign(inner_ctx, c, loop)
            inner_ctx.flush(self)
        elif child_assigns:
            body = loop.add_state('body')
            for idx, a in enumerate(child_assigns):
                self._emit_tasklet(body, a, idx, iter_map)

    def _emit_tasklet(self, state, assign_node, idx: int, iter_map: dict):
        """One Tasklet per array assignment."""
        accesses = assign_node.accesses

        tokens = set(re.findall(r'[a-zA-Z_]\w*', assign_node.expr))
        r_arr = tokens & set(self.arrays)
        r_scl = tokens & set(self.scalars)
        target = assign_node.target

        in_c = {f"_in_{x}" for x in r_arr | r_scl}
        out_c = {f"_out_{target}"}

        # Rewrite: longest names first, to avoid partial matches.
        code = assign_node.expr
        for nm in sorted(r_arr | r_scl, key=len, reverse=True):
            code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)
        code = f"_out_{target} = {code}"

        t = state.add_tasklet(f"t_{idx}", in_c, out_c, code)

        for nm in sorted(r_arr):
            r = self._acc(state, nm)
            ivars = self._get_index_vars(accesses, nm, is_read=True)
            ix = self._build_memlet_index(nm, ivars, iter_map)
            state.add_edge(r, None, t, f"_in_{nm}", Memlet(f"{nm}[{ix}]"))

        for sc in sorted(r_scl):
            r = self._acc(state, sc)
            state.add_edge(r, None, t, f"_in_{sc}", Memlet(data=sc, subset="0"))

        w = self._acc(state, target)
        ivars = self._get_index_vars(accesses, target, is_read=False)
        ix = self._build_memlet_index(target, ivars, iter_map)
        state.add_edge(t, f"_out_{target}", w, None, Memlet(f"{target}[{ix}]"))

    def _acc(self, state, name: str):
        """Single access node for `name` in `state`, reused across reads/writes.

        Without this, every tasklet in the same state would fabricate its own
        disconnected access node, so a later read could not see the value
        produced by an earlier write in the same state.
        """
        cache = getattr(state, '_hlfir_access', None)
        if cache is None:
            cache = {}
            state._hlfir_access = cache
        node = cache.get(name)
        if node is None:
            node = state.add_access(name)
            cache[name] = node
        return node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_index_vars(self, accesses: list, array_name: str, is_read: bool) -> list:
        for ac in accesses:
            if ac.array_name == array_name:
                if is_read and ac.is_read:
                    return ac.index_vars
                if not is_read and ac.is_write:
                    return ac.index_vars
        for ac in accesses:
            if ac.array_name == array_name:
                return ac.index_vars
        return []

    def _build_memlet_index(self, array_name: str, index_vars: list, iter_map: dict) -> str:
        """Build a memlet subset, offsetting Fortran→DaCe indices."""
        arr = self.arrays.get(array_name)
        lbs = arr.lower_bounds if arr else []

        parts = []
        for dim, v in enumerate(index_vars):
            uid = iter_map.get(v, v)
            lb = lbs[dim] if dim < len(lbs) else "1"

            if lb == "0":
                parts.append(uid)
            elif lb == "1":
                parts.append(f"{uid} - 1")
            else:
                try:
                    lb_int = int(lb)
                    if lb_int > 0:
                        parts.append(f"{uid} - {lb_int}")
                    else:
                        parts.append(f"{uid} + {-lb_int}")
                except ValueError:
                    parts.append(f"{uid} - {lb}")

        return ", ".join(parts)

    def _emit_cond(self, ctx: '_Ctx', n, region):
        """TODO: branching states."""
        ctx.flush(self)

    def emit_scalar_assign(self, state, target: str, value: str):
        t = state.add_tasklet(f"set_{target}", set(), {'_out'}, f"_out = {value}")
        a = self._acc(state, target)
        state.add_edge(t, '_out', a, None, Memlet(data=target, subset='0'))


# ======================================================================
# Emission context
# ======================================================================


class _Ctx:
    """Tracks the current state and pending scalar assignments."""

    def __init__(self, sdfg: SDFG, builder: SDFGBuilder):
        self.sdfg = sdfg
        self.builder = builder
        self.cur = None
        self.pending = []

    def ensure(self, region=None):
        if not self.cur:
            r = region or self.sdfg
            self.cur = r.add_state(f"s_{self.builder.nid()}")

    def flush(self, builder: SDFGBuilder, region=None):
        if not self.pending:
            return
        r = region or self.sdfg
        self.ensure(r)
        for target, value in self.pending:
            builder.emit_scalar_assign(self.cur, target, value)
        self.pending.clear()

    def new_state(self, builder: SDFGBuilder, region=None, label=None):
        self.flush(builder, region)
        r = region or self.sdfg
        s = r.add_state(label or f"s_{self.builder.nid()}")
        if self.cur:
            r.add_edge(self.cur, s, InterstateEdge())
        self.cur = s
        return s


# ======================================================================
# Public convenience
# ======================================================================


def generate_sdfg(path: str, pipeline: str = DEFAULT_PIPELINE) -> SDFG:
    """One-liner: parse HLFIR file → run passes → validated DaCe SDFG."""
    return SDFGBuilder(path, pipeline=pipeline).build()


# ======================================================================
# CLI: python hlfir_to_sdfg.py <input.hlfir> [output.sdfg]
# ======================================================================

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "multi_stmt.hlfir"
    sdfg = generate_sdfg(path)
    sdfg.validate()
    print(f"SDFG: {sdfg.name}")

    def show_region(region, indent=0):
        p = "  " * indent
        for node in region.nodes():
            if isinstance(node, LoopRegion):
                li = node.init_statement.as_string if node.init_statement else "?"
                lc = node.loop_condition.as_string if node.loop_condition else "?"
                lu = node.update_statement.as_string if node.update_statement else "?"
                print(f"{p}FOR {li}; {lc}; {lu}")
                show_region(node, indent + 1)
            elif isinstance(node, dace.SDFGState):
                print(f"{p}State '{node.label}':")
                for n in node.nodes():
                    if isinstance(n, nd.Tasklet):
                        print(f"{p}  Tasklet: {n.code.as_string.strip()}")
                    elif isinstance(n, nd.AccessNode):
                        print(f"{p}  Data: {n.data}")

    show_region(sdfg)
    out = sys.argv[2] if len(sys.argv) > 2 else f"{sdfg.name}.sdfg"
    sdfg.save(out)
    print(f"\nSaved: {out}")
