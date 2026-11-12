# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Dropout library node with PyTorch ``aten.native_dropout`` training semantics.

The ``pure`` expansion draws, per element, a counter-based Philox4x32-10 random
number (counter = flat element index + ``offset``, key = ``seed``), so the result
is deterministic and identical on CPU and GPU regardless of how the work is
scheduled. For ``keep = u >= p`` the output is ``x / (1 - p)`` when kept and ``0``
otherwise; the optional mask holds ``keep``.
"""

from dace import SDFG, Memlet, SDFGState, dtypes, library, nodes, properties, symbolic
from dace.transformation.transformation import ExpandTransformation


def _edges_by_connector(state, node):
    ins = {e.dst_conn: e for e in state.in_edges(node) if e.dst_conn}
    outs = {e.src_conn: e for e in state.out_edges(node) if e.src_conn}
    return ins, outs


@library.node
class Dropout(nodes.LibraryNode):
    """
    Element-wise dropout. Input connector ``_x``; output connectors ``_y`` and
    optional ``_mask`` (bool/uint8, same shape as ``_x``).
    """

    implementations = {}
    default_implementation = "pure"

    p = properties.SymbolicProperty(default=0.5, category="Semantics", desc="Drop probability")
    seed = properties.SymbolicProperty(default=0, category="Semantics", desc="RNG seed (Philox key)")
    offset = properties.SymbolicProperty(
        default=0, category="Semantics", desc="RNG counter offset; advances the stream between calls"
    )

    def __init__(self, name, p=None, seed=None, offset=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_x"}, outputs={"_y"}, **kwargs)
        if p is not None:
            self.p = p
        if seed is not None:
            self.seed = seed
        if offset is not None:
            self.offset = offset

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins:
            raise ValueError("Dropout requires an '_x' input connector")
        if "_y" not in outs:
            raise ValueError("Dropout requires a '_y' output connector")
        x = sdfg.arrays[ins["_x"].data.data]
        y = sdfg.arrays[outs["_y"].data.data]
        mask = sdfg.arrays[outs["_mask"].data.data] if "_mask" in outs else None
        if len(x.shape) != len(y.shape):
            raise ValueError("Dropout input and output must have the same rank")
        if mask is not None and len(mask.shape) != len(x.shape):
            raise ValueError("Dropout mask must have the same rank as the input")
        return x, y, mask


_PHILOX_BODY = """
unsigned int __key0 = (unsigned int)(((unsigned long long)({seed})) & 0xffffffffULL);
unsigned int __key1 = (unsigned int)(((unsigned long long)({seed})) >> 32);
unsigned int __c0 = (unsigned int)(__ctr & 0xffffffffULL);
unsigned int __c1 = (unsigned int)(__ctr >> 32);
unsigned int __c2 = 0u, __c3 = 0u;
for (int __r = 0; __r < 10; ++__r) {{
    unsigned long long __p0 = (unsigned long long)0xD2511F53u * (unsigned long long)__c0;
    unsigned long long __p1 = (unsigned long long)0xCD9E8D57u * (unsigned long long)__c2;
    unsigned int __hi0 = (unsigned int)(__p0 >> 32), __lo0 = (unsigned int)__p0;
    unsigned int __hi1 = (unsigned int)(__p1 >> 32), __lo1 = (unsigned int)__p1;
    unsigned int __n0 = __hi1 ^ __c1 ^ __key0;
    unsigned int __n1 = __lo1;
    unsigned int __n2 = __hi0 ^ __c3 ^ __key1;
    unsigned int __n3 = __lo0;
    __c0 = __n0; __c1 = __n1; __c2 = __n2; __c3 = __n3;
    __key0 += 0x9E3779B9u; __key1 += 0xBB67AE85u;
}}
double __u = (double)(__c0 >> 8) * (1.0 / 16777216.0);
"""


@library.register_expansion(Dropout, "pure")
class ExpandDropoutPure(ExpandTransformation):
    """
    Native SDFG expansion: one map over the elements with a tasklet that draws a
    counter-based Philox4x32-10 uniform and applies inverted dropout.
    """

    environments = []

    @staticmethod
    def expansion(node: "Dropout", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, y, mask = node.validate(parent_sdfg, parent_state)
        has_mask = mask is not None

        sdfg = SDFG("dropout_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_y", y.shape, y.dtype, strides=y.strides, storage=y.storage)
        if has_mask:
            sdfg.add_array("_mask", mask.shape, mask.dtype, strides=mask.strides, storage=mask.storage)
        state = sdfg.add_state("main")

        params = [f"__i{i}" for i in range(len(x.shape))]
        index = ", ".join(params)
        ranges = {p: f"0:{symbolic.symstr(s)}" for p, s in zip(params, x.shape)}

        # Flat (row-major logical) element index used as the Philox counter.
        flat = params[0]
        for i in range(1, len(x.shape)):
            flat = f"({flat}) * ({symbolic.symstr(x.shape[i])}) + {params[i]}"

        seed = symbolic.symstr(node.seed)
        pstr = symbolic.symstr(node.p)
        offset = symbolic.symstr(node.offset)

        philox = _PHILOX_BODY.format(seed=seed)
        code = (
            f"unsigned long long __ctr = (unsigned long long)({flat}) + (unsigned long long)({offset});\n"
            + philox
            + f"""
bool __keep = (__u >= (double)({pstr}));
__y = __keep ? (__x / (dtype)(1 - ({pstr}))) : (dtype)0;
"""
        )
        outputs = {"__y": Memlet(f"_y[{index}]")}
        if has_mask:
            outputs["__mask"] = Memlet(f"_mask[{index}]")
            code += "__mask = __keep;\n"

        # ``dtype`` is a convenience alias for the element type inside the tasklet.
        code = code.replace("(dtype)", f"({x.dtype.ctype})")

        state.add_mapped_tasklet(
            "dropout",
            ranges,
            {"__x": Memlet(f"_x[{index}]")},
            code,
            outputs,
            external_edges=True,
            language=dtypes.Language.CPP,
        )

        # The Philox parameters are interpolated as text into the C++ tasklet, which
        # codegen does not scan for symbols. Register any free symbols they contain
        # (e.g. a symbolic seed) both as SDFG symbols and on an interstate edge so
        # they become arguments of the generated function.
        used = {}
        for expr in (node.p, node.seed, node.offset):
            for s in symbolic.pystr_to_symbolic(expr).free_symbols:
                used[str(s)] = getattr(s, "dtype", dtypes.int64)
        # Make sure the parent SDFG also knows these symbols so they become arguments.
        for name, stype in used.items():
            if name not in parent_sdfg.symbols:
                parent_sdfg.add_symbol(name, stype or dtypes.int64)
        used = {k: v for k, v in used.items() if k not in sdfg.symbols}
        if used:
            for name, stype in used.items():
                sdfg.add_symbol(name, stype or dtypes.int64)
            assignments = {f"__dropout_use_{i}": s for i, s in enumerate(sorted(used))}
            sdfg.add_state_before(state, "register_symbols", is_start_block=True, assignments=assignments)
        return sdfg
