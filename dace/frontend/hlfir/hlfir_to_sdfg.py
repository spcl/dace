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
import sys as _sys
from pathlib import Path as _Path
import dace
from dace import SDFG, Memlet, InterstateEdge
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.sdfg import nodes as nd
from build_bridge import hb

# intrinsics/ lives alongside this file but uses absolute imports
# (``dace.frontend.hlfir.intrinsics``).  hlfir_to_sdfg is usually imported
# via the ``sys.path.insert(<hlfir_dir>, …)`` pattern in build_bridge.py,
# which does not expose the full ``dace.frontend.hlfir`` package.  Add the
# DaCe source root so the package import resolves either way.
_HLFIR_DIR = _Path(__file__).resolve().parent
_DACE_ROOT = _HLFIR_DIR.parents[2]
if str(_DACE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_DACE_ROOT))

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


def _assign_reads_array(assign_node, arrays: dict) -> bool:
    """True iff any ``accesses`` entry on ``assign_node`` is a read against an
    array descriptor.  Used to promote a nominally-scalar assign (``s = d(i)
    + 1``) onto the per-occurrence-connector tasklet path so the array read
    gets a real memlet instead of a bare identifier in the code string."""
    for ac in assign_node.accesses:
        if ac.is_read and ac.array_name in arrays:
            return True
    return False


class SDFGBuilder:
    """Walks the HLFIR ASTNode tree and emits a DaCe SDFG."""

    DTYPE = {
        'float64': dace.float64,
        'float32': dace.float32,
        'int32': dace.int32,
        'int64': dace.int64,
        'bool': dace.bool_,
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

        # Scalars: local variables (``intent=''``) keep the ``dace.data.Scalar``
        # descriptor, while dummy-arg scalars (``intent(in|out|inout)``) land
        # as length-1 ``dace.data.Array``.  DaCe doesn't add non-transient
        # scalars to the SDFG signature at all — pass-by-reference only works
        # through Array descriptors — so Fortran scalar parameters have to be
        # modelled as size-1 arrays on the Python binding surface.
        for v in self.scalars.values():
            if v.intent == '':
                sdfg.add_scalar(
                    v.fortran_name,
                    dtype=self._dt(v.dtype),
                    transient=True,
                )
            else:
                sdfg.add_array(
                    v.fortran_name,
                    shape=(1, ),
                    dtype=self._dt(v.dtype),
                    transient=False,
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
            elif n.kind == "reduce": self._emit_reduce(ctx, n, region)
            elif n.kind == "copy": self._emit_copy(ctx, n, region)
            elif n.kind == "memset": self._emit_memset(ctx, n, region)
            elif n.kind == "libcall": self._emit_libcall(ctx, n, region)
            elif n.kind == "break": self._emit_break(ctx, n, region)
            elif n.kind == "return": self._emit_return(ctx, n, region)
            # "call" → TODO: nested SDFG or library node

    def _emit_copy(self, ctx: '_Ctx', n, region):
        """Whole-array ``b = a`` → ``CopyLibraryNode`` with ``_in`` / ``_out``
        memlets covering the full source / destination arrays."""
        from dace.libraries.standard.nodes import CopyLibraryNode
        ctx.flush(self)
        ctx.ensure(region)
        state = ctx.cur

        src_name = n.reduce_src  # buildCopyNode stored the source here
        tgt_name = n.target
        src_desc = ctx.sdfg.arrays[src_name]
        tgt_desc = ctx.sdfg.arrays[tgt_name]

        cp = CopyLibraryNode(f"copy_{tgt_name}_{self.nid()}")
        state.add_node(cp)

        src_access = self._acc(state, src_name)
        tgt_access = self._acc(state, tgt_name)
        state.add_edge(src_access, None, cp, "_in", Memlet.from_array(src_name, src_desc))
        state.add_edge(cp, "_out", tgt_access, None, Memlet.from_array(tgt_name, tgt_desc))

    def _emit_break(self, ctx: '_Ctx', n, region):
        """Fortran ``EXIT`` → ``BreakBlock`` added to the current region.
        The block is a leaf and implicitly transfers control to the
        nearest enclosing loop's exit edge at codegen time.  When the
        break is the region's first block (a branch body whose only
        statement is ``exit``), it becomes the region's start block."""
        from dace.sdfg.state import BreakBlock
        ctx.flush(self, region)
        is_start = ctx.cur is None
        blk = BreakBlock(f"break_{self.nid()}")
        region.add_node(blk, is_start_block=is_start)
        if ctx.cur is not None:
            region.add_edge(ctx.cur, blk, InterstateEdge())
        ctx.cur = blk

    def _emit_return(self, ctx: '_Ctx', n, region):
        """Fortran ``RETURN`` → ``ReturnBlock``.  Added to the current
        region so RETURNs nested inside a loop or conditional get
        placed correctly; codegen still emits a plain ``return`` that
        bails out of the whole subroutine."""
        from dace.sdfg.state import ReturnBlock
        ctx.flush(self, region)
        is_start = ctx.cur is None
        blk = ReturnBlock(f"return_{self.nid()}")
        region.add_node(blk, is_start_block=is_start)
        if ctx.cur is not None:
            region.add_edge(ctx.cur, blk, InterstateEdge())
        ctx.cur = blk

    def _emit_memset(self, ctx: '_Ctx', n, region):
        """Scalar-zero → array fill → ``MemsetLibraryNode`` with a single
        ``_out`` memlet covering the destination."""
        from dace.libraries.standard.nodes import MemsetLibraryNode
        ctx.flush(self)
        ctx.ensure(region)
        state = ctx.cur

        tgt_name = n.target
        tgt_desc = ctx.sdfg.arrays[tgt_name]

        ms = MemsetLibraryNode(f"memset_{tgt_name}_{self.nid()}")
        ms.add_out_connector("_out")
        state.add_node(ms)

        tgt_access = self._acc(state, tgt_name)
        state.add_edge(ms, "_out", tgt_access, None, Memlet.from_array(tgt_name, tgt_desc))

    # Per-node-class connector conventions (set via ``inputs=`` / ``outputs=``
    # in each library node's ``__init__``).  Kept next to ``_emit_libcall``
    # rather than on ``LibNodeIntrinsic`` because the names are a property
    # of the DaCe node, not of the Fortran intrinsic.
    _LIBCALL_CONNECTORS = {
        "MatMul": (("_a", "_b"), "_c"),
        "Dot": (("_x", "_y"), "_result"),
        "Transpose": (("_inp", ), "_out"),
    }

    def _emit_libcall(self, ctx: '_Ctx', n, region):
        """``target = matmul(a, b)`` / ``transpose(a)`` / ``dot_product(x, y)``
        lowered to the matching DaCe library node.  ``MatMul`` specializes
        internally (GEMM / GEMV / Dot) based on operand ranks, so we only
        need one emission path for all three linalg intrinsics."""
        import importlib
        from dace.frontend.hlfir.intrinsics import libnode_spec

        ctx.flush(self)
        ctx.ensure(region)
        state = ctx.cur

        spec = libnode_spec(n.callee)
        if spec is None:
            raise RuntimeError(f"unregistered libnode intrinsic {n.callee!r}")
        mod = importlib.import_module(f"dace.libraries.{spec.module}.nodes")
        cls = getattr(mod, spec.node_cls)
        in_conns, out_conn = self._LIBCALL_CONNECTORS[spec.node_cls]

        # ``Transpose`` needs an explicit ``dtype`` so its expansion can
        # produce the right element type; every other library node picks
        # types up from the attached memlets.
        tgt_desc = ctx.sdfg.arrays[n.target]
        if spec.node_cls == "Transpose":
            node = cls(f"{spec.name}_{n.target}_{self.nid()}", dtype=tgt_desc.dtype)
        else:
            node = cls(f"{spec.name}_{n.target}_{self.nid()}")
        state.add_node(node)

        for conn, src in zip(in_conns, n.call_args):
            src_desc = ctx.sdfg.arrays[src]
            state.add_edge(self._acc(state, src), None, node, conn, Memlet.from_array(src, src_desc))

        state.add_edge(node, out_conn, self._acc(state, n.target), None, Memlet.from_array(n.target, tgt_desc))

    def _emit_reduce(self, ctx: '_Ctx', n, region):
        """``target = sum(src)`` (and product / minval / maxval) lowered as
        a DaCe ``standard.Reduce`` library node via
        ``state.add_reduce(wcr, axes, identity)``.

        ``axes=None`` reduces all dimensions (whole-array scalar result);
        a non-empty ``reduce_axes`` list reduces along those dims only.
        """
        import math as _math
        ctx.flush(self)
        ctx.ensure(region)
        state = ctx.cur

        src_name = n.reduce_src
        src_desc = ctx.sdfg.arrays.get(src_name)
        if src_desc is None:
            raise RuntimeError(f"reduction source {src_name!r} not registered as SDFG data")
        axes = list(n.reduce_axes) if n.reduce_axes else None

        # DaCe's Reduce expects a value (or None) for ``identity``.
        # ``math.inf`` / ``-math.inf`` come through as strings and must
        # evaluate against an environment that sees ``math``.
        identity_val = None
        if n.reduce_identity:
            identity_val = eval(n.reduce_identity, {'math': _math})

        red = state.add_reduce(n.reduce_wcr, axes, identity_val)

        src_access = self._acc(state, src_name)
        tgt_access = self._acc(state, n.target)
        state.add_edge(src_access, None, red, None, Memlet.from_array(src_name, src_desc))
        tgt_desc = ctx.sdfg.arrays[n.target]
        state.add_edge(red, None, tgt_access, None, Memlet.from_array(n.target, tgt_desc))

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
        # ``?`` is the bridge's placeholder for an unextractable condition
        # (EXIT shapes where ``buildWhileNode`` can't fold the keep-going
        # predicate).  Default to ``True`` so ast.parse succeeds and leave
        # the faithful structure visible in the SDFG for inspection.
        cond = n.condition if n.condition and n.condition != "?" else "True"
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

    def _auto_declare_synth(self, name: str, ctx: '_Ctx'):
        """Lazy-declare a synthetic scalar minted by the bridge's faithful
        scf.while walker.  ``__sc_N`` names materialise ``scf.if -> T``
        results; ``__al_N`` names come from bare ``fir.alloca`` ops that
        lift-cf-to-scf uses as scratch counters.  Both need an SDFG
        descriptor + an entry in ``self.scalars`` so ``_emit_assign``'s
        existing dispatch (scalar pending, or symbol state-change) can
        fire normally.  Treated as transient ints — they only live for
        the loop's lifetime and are read only by downstream generated
        conditions."""
        if name in self.scalars or name in self.symbols:
            return
        if not (name.startswith("__sc_") or name.startswith("__al_")):
            return
        # Fake a VarInfo-like record so _add_descriptors-consistent paths
        # work.  A ``SimpleNamespace`` is enough — the scalar dispatch
        # only reads ``.intent`` and ``.dtype``.
        from types import SimpleNamespace
        v = SimpleNamespace(fortran_name=name,
                            intent='',
                            dtype='int32',
                            rank=0,
                            is_dynamic=False,
                            role='scalar',
                            shape_symbols=[],
                            lower_bounds=[])
        self.scalars[name] = v
        if name not in ctx.sdfg.arrays:
            ctx.sdfg.add_scalar(name, dtype=dace.int32, transient=True)

    def _emit_assign(self, ctx: '_Ctx', n, region):
        """Scalar or symbol assignment.

        Routes by target kind:
          * ``symbols``    → interstate-edge assignment that forces a new state.
          * ``array``      → tasklet via ``_emit_tasklet`` with per-occurrence
                             array-read connectors.
          * ``scalar`` whose RHS reads an array element (``s = d(2,1) + 1.0``)
                             → same tasklet path; the subscripted read needs a
                             real memlet so the codegen sees a connector, not
                             a bare array-pointer identifier.
          * plain ``scalar`` (``i = i + 1``, ``c = 0.5``) → queued on
                             ``ctx.pending`` for a flat ``emit_scalar_assign``
                             tasklet at flush time.
        """
        # Synthetic scalars (``__sc_N`` / ``__al_N``) from the faithful
        # scf.while walker don't come in as ``hlfir.declare`` ops, so
        # ``_add_descriptors`` never saw them.  Register on first assign.
        self._auto_declare_synth(n.target, ctx)
        if n.target in self.symbols:
            ctx.flush(self)
            ctx.ensure(region)
            dst = region.add_state(f"post_{n.target}_{self.nid()}")
            region.add_edge(ctx.cur, dst, InterstateEdge(assignments={n.target: n.expr}))
            ctx.cur = dst
            return
        if n.target_is_array or _assign_reads_array(n, self.arrays):
            ctx.flush(self, region)
            ctx.ensure(region)
            self._emit_tasklet(ctx.cur, n, self.nid(), ctx.iter_map)
            return
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
        # Anything beyond nested DO loops and plain assignments (IF/ELSE,
        # WHILE, reductions, library-node calls, …) forces the generic
        # state-machine walk — the flat ``body`` tasklet path can't host
        # interstate edges.
        has_structured = any(c.kind not in ("loop", "assign") for c in children)

        if has_structured:
            inner_ctx = _Ctx(ctx.sdfg, self)
            inner_ctx.iter_map = iter_map
            body_start = loop.add_state(f"body_{self.nid()}", is_start_block=True)
            inner_ctx.cur = body_start
            self._emit(inner_ctx, list(children), loop)
            inner_ctx.flush(self, loop)
        elif child_loops:
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

        # ----------------------------------------------------------------
        # Pick the write-side access node for the tasklet's output edge.
        # ----------------------------------------------------------------
        #
        # An SDFGState is a DAG of AccessNodes and Tasklets.  For each
        # data name we keep ONE "live sink" in ``state._hlfir_access[name]``
        # — the access node that subsequent reads from that name should
        # pull from (because it holds the latest write).  Two rules govern
        # whether a new write reuses that sink or allocates a fresh one:
        #
        # 1. A write that is paired with a read of the SAME name in the
        #    SAME tasklet must target a NEW access node, not the one the
        #    read came from.  Otherwise the tasklet would have both an
        #    incoming and outgoing edge on the same node — a cycle — and
        #    DaCe's state validator would reject it.  Fortran patterns
        #    that trigger this: ``i = i + 1`` (scalar self-update),
        #    ``d(1) = d(1) * 2.0`` (array element self-update), and
        #    ``temp = min(d(1), temp)`` (scalar target reads its own value
        #    plus an array element).
        #
        # 2. A write whose cached sink has ALREADY been read by a later
        #    tasklet in this state must also get a new access node.
        #    Sharing it would give the DAG scheduler freedom to reorder
        #    the new write before the earlier read — changing the
        #    observable value of the read.  Concretely: if ``A`` has been
        #    read after its last write, a further write creates a new
        #    version of ``A``; the old version keeps the earlier read
        #    wired to the right producer, the new version becomes the
        #    live sink for any later reads.
        #
        # In every other case — pure write-only update over the latest
        # sink, without intervening reads — we reuse the cached access
        # node.  Multiple write edges into one access node are legal in
        # a DAG (they just represent two writers on the same version),
        # and sharing keeps the data-flow graph connected so later reads
        # can chain off of it.
        #
        # The cache is refreshed on every fresh-allocation path so
        # subsequent calls to ``self._acc(name)`` / reads through the
        # cache always hit the most recent sink.
        cache = getattr(state, '_hlfir_access', None)
        is_self_update = (target in r_scl) or (target in reads_by_name)
        cached_has_readers = False
        if cache is not None and target in cache:
            # out_degree > 0 ⇒ some later tasklet already consumed this
            # node, so writing to it again would race with that read.
            cached_has_readers = state.out_degree(cache[target]) > 0
        if is_self_update or cached_has_readers:
            w = state.add_access(target)
            if cache is not None:
                cache[target] = w
        else:
            w = self._acc(state, target)

        if target in self.scalars:
            # Scalar target: no buildable index, subset is always element 0.
            state.add_edge(t, f"_out_{target}", w, None, Memlet(data=target, subset="0"))
        else:
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
        """``if (cond) then ... else ... end if`` → ``ConditionalBlock`` with
        a ``ControlFlowRegion`` per branch.

        The block itself is a single node in ``region``; subsequent statements
        land in a fresh successor state wired from the block.  Each branch's
        body lives in its own ``ControlFlowRegion``, populated through a
        nested ``_emit`` dispatch so conditionals, loops, library-node calls
        etc. all compose naturally.

        A missing ``else`` is encoded as ``add_branch(None, empty_region)`` —
        DaCe's codegen treats a ``None``-conditioned branch as the default
        (``else``) arm.
        """
        ctx.flush(self, region)
        ctx.ensure(region)
        pre = ctx.cur

        cond = n.condition if n.condition and n.condition != "?" else "True"
        # Substitute Fortran iterator names (``i``) with their unique DaCe
        # loop-var names (``i_0``) picked by the enclosing ``_emit_loop``.
        for fname, uname in ctx.iter_map.items():
            cond = re.sub(rf'\b{re.escape(fname)}\b', uname, cond)
        # Scalars with intent land as size-1 Arrays on the SDFG signature,
        # so referring to a bare name in a branch condition would pick up
        # the array pointer.  Subscript each one to read element 0.
        for nm, v in self.scalars.items():
            if v.intent:
                cond = re.sub(rf'\b{re.escape(nm)}\b', f"{nm}[0]", cond)

        uid = self.nid()
        cond_block = ConditionalBlock(f"if_{uid}")
        region.add_node(cond_block, ensure_unique_name=True)
        if pre is not None:
            region.add_edge(pre, cond_block, InterstateEdge())

        def _populate_branch(label: str, children: list) -> ControlFlowRegion:
            region = ControlFlowRegion(label, sdfg=ctx.sdfg)
            inner = _Ctx(ctx.sdfg, self)
            inner.iter_map = ctx.iter_map
            self._emit(inner, children, region)
            inner.flush(self, region)
            # An empty branch (e.g. the EXIT arm of a Flang-lowered DO+EXIT)
            # still needs a start block, otherwise the validator complains.
            # Add a trivial no-op state marked as start.
            if len(region.nodes()) == 0:
                region.add_state(f"{label}_noop", is_start_block=True)
            return region

        then_region = _populate_branch(f"if_{uid}_then", list(n.children))
        cond_block.add_branch(cond, then_region)

        else_children = list(n.else_children)
        if else_children:
            else_region = _populate_branch(f"if_{uid}_else", else_children)
            cond_block.add_branch(None, else_region)

        # The ConditionalBlock is itself the "current" control-flow node;
        # subsequent statements get a fresh state edge-connected to it.
        ctx.cur = cond_block

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

        # Self-update (``i = i + 1``): the read and write need DIFFERENT
        # access nodes so the state remains a DAG — ``Access(i_read) →
        # Tasklet → Access(i_write)`` instead of a cycle on one node.
        # Plain ``i = 0`` still reuses the cached node.
        if target in reads:
            a = state.add_access(target)
            # Refresh the cache so subsequent tasklets in the same state
            # see this "latest" version when they read ``target``.
            cache = getattr(state, '_hlfir_access', None)
            if cache is not None:
                cache[target] = a
        else:
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
        # Active DO-loop iterator renames (Fortran name → unique DaCe name).
        # Populated by ``_emit_loop`` for the duration of each loop body so
        # downstream emitters (``_emit_cond`` / ``_emit_tasklet``) can
        # substitute iterators referenced in conditions or RHS expressions.
        self.iter_map = {}

    def ensure(self, region=None):
        # ``not self.cur`` misfires the same way ``region or self.sdfg`` did:
        # SDFGState / LoopRegion define __len__ that returns 0 when empty,
        # so a freshly-created state is treated as falsy even though we
        # want to keep emitting into it.  Use explicit None checks.
        from dace.sdfg.state import SDFGState
        if self.cur is None:
            r = self.sdfg if region is None else region
            # First state added to an otherwise-empty control-flow region
            # must be marked as the starting block, otherwise DaCe's
            # validator raises "Ambiguous or undefined starting block".
            is_start = (len(r.nodes()) == 0)
            self.cur = r.add_state(f"s_{self.builder.nid()}", is_start_block=is_start)
            return
        # After a ConditionalBlock (or any non-SDFGState control-flow block
        # like a LoopRegion), the next emitter needs a fresh successor state
        # wired from that block so tasklets / memlets have somewhere to land.
        if not isinstance(self.cur, SDFGState):
            r = self.sdfg if region is None else region
            succ = r.add_state(f"s_{self.builder.nid()}")
            r.add_edge(self.cur, succ, InterstateEdge())
            self.cur = succ

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
