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
DEFAULT_PIPELINE = (
    "hlfir-inline-all,"
    "hlfir-flatten-structs,"
    "hlfir-propagate-shapes,"
    "hlfir-default-intent,"
    # Lift cf.br / cf.cond_br loops (Flang's DO WHILE shape)
    # into scf.while so extract_ast can walk them.
    "lift-cf-to-scf,"
    # Constant propagation + fold + CSE after all the HLFIR
    # rewrites have exposed as many constants as they will
    # (inline dissolves call boundaries, flatten-structs
    # exposes scalar member loads, lift-cf-to-scf makes IV
    # bounds visible).
    "sccp,canonicalize,cse")


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
        # Literal-integer dimensions (e.g. the "3" in ``edge_idx(nc, 3)``)
        # stay as Python ints and do not need a symbol registration.
        known = {v.fortran_name for v in self.variables}
        for v in self.arrays.values():
            for s in v.shape_symbols:
                if s.lstrip('-').isdigit():
                    continue
                if s not in known and s not in sdfg.symbols:
                    sdfg.add_symbol(s, dace.int64)

        def _dim(s: str):
            if s.lstrip('-').isdigit():
                return int(s)
            return dace.symbol(s)

        # Arrays.
        for v in self.arrays.values():
            sdfg.add_array(
                v.fortran_name,
                shape=[_dim(s) for s in v.shape_symbols],
                dtype=self._dt(v.dtype),
                transient=(v.intent == ''),
            )

        # Scalars as Scalar descriptors — Fortran arrays of known size map
        # to SDFG Arrays regardless of extent (``dimension(1)`` still gets
        # add_array), but plain Fortran variables stay scalars.
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
            elif n.kind == "while": self._emit_while(ctx, n, region)
            elif n.kind == "conditional": self._emit_cond(ctx, n, region)
            # "call" → TODO: nested SDFG or library node

    def _emit_while(self, ctx: '_Ctx', n, region):
        """Fortran ``DO WHILE`` — lifted by ``lift-cf-to-scf`` into scf.while
        and extracted as kind="while".  Emit a DaCe LoopRegion whose
        condition is the before-region's comparison; no init / update
        expression since the IV bookkeeping lives inside the body as
        regular tasklets.

        The induction variable is promoted from SDFG scalar data to a
        symbol so the LoopRegion condition can evaluate symbolically, and
        an interstate-edge assignment carries the IV into the loop on each
        iteration.
        """
        ctx.flush(self)
        cond = n.condition if n.condition else "True"
        loop = LoopRegion(label=f"while_{self.nid()}", condition_expr=cond)
        region.add_node(loop)
        if ctx.cur is not None:
            region.add_edge(ctx.cur, loop, InterstateEdge())
        ctx.cur = loop

        # A start state is required for DaCe codegen to treat the loop body
        # as non-empty.  We also use it as the insertion point for any
        # trailing scalar flushes.
        body_start = loop.add_state(f"while_body_{self.nid()}", is_start_block=True)
        inner_ctx = _Ctx(ctx.sdfg, self)
        inner_ctx.cur = body_start
        self._emit(inner_ctx, list(n.children), loop)
        inner_ctx.flush(self, loop)

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
            # Indirect accesses in the body turn into fresh SDFG symbols; the
            # value is assigned on an interstate edge so a new state is forced
            # before the compute tasklet runs.
            indirect_syms = self._collect_indirect(child_assigns)
            if indirect_syms:
                pre = loop.add_state(f"pre_{self.nid()}")
                body = loop.add_state('body')
                assigns = {sym: self._indirect_to_dace(expr, iter_map) for expr, sym in indirect_syms.items()}
                for sym in indirect_syms.values():
                    if sym not in ctx.sdfg.symbols:
                        ctx.sdfg.add_symbol(sym, dace.int64)
                loop.add_edge(pre, body, InterstateEdge(assignments=assigns))
            else:
                body = loop.add_state('body')
            for idx, a in enumerate(child_assigns):
                self._emit_tasklet(body, a, idx, iter_map, indirect_syms)

    def _emit_tasklet(self, state, assign_node, idx: int, iter_map: dict, indirect_syms: dict = None):
        """One Tasklet per array assignment.

        Expressions like ``e_bln(jc,1)*z_kin(...) + e_bln(jc,2)*z_kin(...)``
        access the same array at several positions.  Each *occurrence* in
        the RHS becomes its own tasklet input connector so every access
        carries the correct memlet; otherwise the generated code would
        collapse all three terms onto a single connector and silently
        compute a wrong result.
        """
        indirect_syms = indirect_syms or {}
        accesses = assign_node.accesses

        tokens = set(re.findall(r'[a-zA-Z_]\w*', assign_node.expr))
        r_arr = tokens & set(self.arrays)
        r_scl = tokens & set(self.scalars)
        target = assign_node.target

        # Index arrays (e.g. edge_idx) show up in the RHS token scan but we
        # move their values onto the interstate edge as symbols.
        indirect_arrays = {self._indirect_host(expr) for expr in indirect_syms}
        r_arr -= indirect_arrays

        # One AccessInfo per occurrence, in the order buildExpr produced.
        reads_by_name = {}
        for ac in accesses:
            if ac.is_read and ac.array_name in r_arr:
                reads_by_name.setdefault(ac.array_name, []).append(ac)

        # Rewrite the RHS, replacing the Nth occurrence of each array name
        # with `_in_<name>_<N>`.  Longest-first guards against partial
        # matches between related names.
        occ = {nm: 0 for nm in r_arr}
        sorted_tokens = sorted(r_arr | r_scl, key=len, reverse=True)

        def rewrite(code: str) -> str:
            for nm in sorted_tokens:
                if nm in r_scl:
                    code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)
                    continue

                def sub(_m, _nm=nm):
                    n = occ[_nm]
                    occ[_nm] += 1
                    return f"_in_{_nm}_{n}"

                code = re.sub(rf'\b{re.escape(nm)}\b', sub, code)
            return code

        in_c = {f"_in_{sc}" for sc in r_scl}
        for nm, acs in reads_by_name.items():
            for i in range(len(acs)):
                in_c.add(f"_in_{nm}_{i}")
        out_c = {f"_out_{target}"}

        code = f"_out_{target} = {rewrite(assign_node.expr)}"
        t = state.add_tasklet(f"t_{idx}", in_c, out_c, code)

        for nm in sorted(reads_by_name):
            r = self._acc(state, nm)
            for i, ac in enumerate(reads_by_name[nm]):
                ix = self._build_memlet_index(nm, ac, iter_map, indirect_syms)
                state.add_edge(r, None, t, f"_in_{nm}_{i}", Memlet(f"{nm}[{ix}]"))

        for sc in sorted(r_scl):
            r = self._acc(state, sc)
            state.add_edge(r, None, t, f"_in_{sc}", Memlet(data=sc, subset="0"))

        w = self._acc(state, target)
        ac = self._get_access(accesses, target, is_read=False)
        ix = self._build_memlet_index(target, ac, iter_map, indirect_syms)
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

    def _get_access(self, accesses: list, array_name: str, is_read: bool):
        """Return the matching AccessInfo (exact read/write match preferred)."""
        for ac in accesses:
            if ac.array_name == array_name:
                if is_read and ac.is_read:
                    return ac
                if not is_read and ac.is_write:
                    return ac
        for ac in accesses:
            if ac.array_name == array_name:
                return ac
        return None

    _INDIRECT_RE = re.compile(r'^(\w+)\[([^\]]*)\]$')

    def _indirect_host(self, expr: str) -> str:
        m = self._INDIRECT_RE.match(expr)
        return m.group(1) if m else ""

    def _collect_indirect(self, assigns: list) -> dict:
        """Walk every access in ``assigns`` and mint a fresh SDFG symbol for
        each distinct indirect index expression.  Returns a map from the
        Fortran-style expression (``edge_idx[jc,1]``) to the symbol name."""
        out = {}
        for a in assigns:
            for ac in a.accesses:
                for expr in getattr(ac, 'index_exprs', None) or []:
                    if '[' in expr and expr not in out:
                        out[expr] = f"_idx_{self.nid()}"
        return out

    def _indirect_to_dace(self, expr: str, iter_map: dict) -> str:
        """Convert ``arr[i,j]`` (Fortran 1-based) into DaCe's 0-based
        subscript form using the array's lower bounds and the current loop
        iter_map."""
        m = self._INDIRECT_RE.match(expr)
        if not m:
            return expr
        arr, inner = m.group(1), m.group(2)
        info = self.arrays.get(arr)
        lbs = info.lower_bounds if info else []
        parts = []
        for dim, raw in enumerate(p.strip() for p in inner.split(',')):
            lb = lbs[dim] if dim < len(lbs) else "1"
            parts.append(self._offset_index_token(raw, lb, iter_map))
        return f"{arr}[{', '.join(parts)}]"

    def _offset_index_token(self, tok: str, lb: str, iter_map: dict) -> str:
        """Apply lower-bound offset to a single index token (``jc`` or ``3``)."""
        try:
            lb_int = int(lb)
        except (TypeError, ValueError):
            lb_int = 1

        if tok.lstrip('-').isdigit():
            return str(int(tok) - lb_int)
        uid = iter_map.get(tok, tok)
        if lb_int == 0:
            return uid
        return f"{uid} - {lb_int}" if lb_int >= 0 else f"{uid} + {-lb_int}"

    def _build_memlet_index(self, array_name: str, access, iter_map: dict, indirect_syms: dict = None) -> str:
        """Build a memlet subset, offsetting Fortran→DaCe indices and
        resolving indirect index expressions against their minted symbols."""
        indirect_syms = indirect_syms or {}
        arr = self.arrays.get(array_name)
        lbs = arr.lower_bounds if arr else []
        if access is None:
            return ""
        exprs = list(access.index_exprs) if access.index_exprs else []
        ivars = list(access.index_vars)

        parts = []
        for dim, v in enumerate(ivars):
            lb = lbs[dim] if dim < len(lbs) else "1"
            expr = exprs[dim] if dim < len(exprs) else v

            # Indirect: use the minted symbol (holds the Fortran 1-based index).
            if '[' in expr and expr in indirect_syms:
                parts.append(self._offset_index_token(indirect_syms[expr], lb, iter_map))
                continue

            # Constant literal: subtract the lower bound directly.
            if expr.lstrip('-').isdigit():
                parts.append(self._offset_index_token(expr, lb, iter_map))
                continue

            uid = iter_map.get(v, v)

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
        """Tasklet for ``target = value`` on a scalar target.

        Inputs are derived from the identifier tokens that appear in
        ``value`` — every one that names an SDFG scalar gets its own
        input connector so the tasklet can read ``i`` for ``i = i + 1``
        and similar self-updates."""
        value = str(value)
        tokens = set(re.findall(r'[a-zA-Z_]\w*', value))
        # ``nm != target`` was wrong — ``i = i + 1`` genuinely needs a read
        # edge on the target itself.
        reads = [nm for nm in sorted(tokens, key=len, reverse=True) if nm in self.scalars]

        code = value
        for nm in reads:
            code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)

        in_c = {f"_in_{nm}" for nm in reads}
        out_c = {'_out'}
        t = state.add_tasklet(f"set_{target}", in_c, out_c, f"_out = {code}")

        for nm in reads:
            r = self._acc(state, nm)
            state.add_edge(r, None, t, f"_in_{nm}", Memlet(data=nm, subset='0'))

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
        # ``not self.cur`` misfires the same way ``region or self.sdfg`` did:
        # SDFGState / LoopRegion define __len__ that returns 0 when empty,
        # so a freshly-created state is treated as falsy even though we
        # want to keep emitting into it.  Use explicit None checks.
        if self.cur is None:
            r = self.sdfg if region is None else region
            self.cur = r.add_state(f"s_{self.builder.nid()}")

    def flush(self, builder: SDFGBuilder, region=None):
        if not self.pending:
            return
        r = self.sdfg if region is None else region
        self.ensure(r)
        for target, value in self.pending:
            builder.emit_scalar_assign(self.cur, target, value)
        self.pending.clear()

    def new_state(self, builder: SDFGBuilder, region=None, label=None):
        self.flush(builder, region)
        r = self.sdfg if region is None else region
        s = r.add_state(label or f"s_{self.builder.nid()}")
        if self.cur is not None:
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
