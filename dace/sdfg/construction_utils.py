# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Graph construction utilities."""

from ordered_set import OrderedSet
from sympy import Function

import dace


def generate_assignment_as_tasklet_in_state(state: dace.SDFGState, lhs: str, rhs: str):
    import sympy

    rhs = rhs.strip()
    rhs_sym_expr = dace.symbolic.SymExpr(rhs)
    lhs = lhs.strip()
    lhs_sym_expr = dace.symbolic.SymExpr(lhs)

    in_connectors = dict()
    out_connectors = dict()

    # Get functions for indirect accesses and free symbols
    i = 0
    for free_sym in rhs_sym_expr.free_symbols:
        if str(free_sym) in state.sdfg.arrays:
            in_connectors[free_sym] = f"_in_{free_sym}_{i}"
            i += 1

    for sub in rhs_sym_expr.atoms(dace.symbolic.Subscript):
        # An array access ``arr[idx]`` is a ``Subscript``: ``args[0]`` is the
        # array head and ``args[1:]`` the indices. ``str(f.func)`` would yield
        # ``"Subscript"`` rather than the array name, so detect via the head.
        if str(sub.args[0]) in state.sdfg.arrays:
            # Array access can come from different arrays:
            if sub not in in_connectors:
                in_connectors[sub] = f"_in_{sub.args[0]}_{i}"
                i += 1

    assert len(lhs_sym_expr.free_symbols) == 1
    free_sym = next(iter(lhs_sym_expr.free_symbols))
    assert str(free_sym) in state.sdfg.arrays
    out_access_expr = free_sym
    out_connectors[free_sym] = f"_out_{free_sym}_{i}"
    i += 1

    assert len(set(lhs_sym_expr.atoms(Function))) == 0

    if in_connectors == {} and out_connectors == {}:
        raise Exception("Generated tasklets result in no or out connectors")

    # Process interstate edge, extract brackets for access patterns
    # Collect array accesses
    in_access_exprs: dict[sympy.Basic, list[list[sympy.Basic]]] = {
        sub.args[0]: [] for sub in rhs_sym_expr.atoms(dace.symbolic.Subscript) if str(sub.args[0]) in state.sdfg.arrays
    }
    for sub in rhs_sym_expr.atoms(dace.symbolic.Subscript):
        if str(sub.args[0]) not in state.sdfg.arrays:
            continue
        in_access_exprs[sub.args[0]].append(list(sub.args[1:]))

    # Get scalar and arrays that are free symbols currently
    name_mapping: dict[sympy.Basic, str] = {
        sub: in_connectors[sub]
        for sub in rhs_sym_expr.atoms(dace.symbolic.Subscript)
        if str(sub.args[0]) in state.sdfg.arrays
    }
    name_mapping.update({s: in_connectors[s] for s in rhs_sym_expr.free_symbols if str(s) in state.sdfg.arrays})

    printer = dace.symbolic.DaceSympyPrinter(arrays=state.sdfg.arrays)
    # Replace each array-access ``Subscript`` (and bare array symbol) with its
    # connector symbol. Use ``xreplace`` for exact-node replacement: ``subs``
    # does not reliably rewrite a whole ``Subscript`` node (its custom ``_subs``
    # recurses into the indices instead of swapping the node out).
    #
    # The two sides are rewritten with DIFFERENT maps. A read-modify-write names the same
    # array on both (``c = c >> 1``), and under one shared map the LHS entry overwrites the
    # RHS one -- the body then reads the OUT connector, which nothing feeds, while the in
    # connector and its edge sit unused. That name survives as a free symbol of the tasklet
    # and reaches ``arglist`` as a symbol the SDFG has no entry for.
    rhs_repl = {k: sympy.Symbol(v) if isinstance(v, str) else v for k, v in name_mapping.items()}
    lhs_repl = {out_access_expr: sympy.Symbol(out_connectors[out_access_expr])}
    rhs = printer.doprint(rhs_sym_expr.xreplace(rhs_repl))
    lhs = printer.doprint(lhs_sym_expr.xreplace(lhs_repl))

    # Ass tasklets
    t = state.add_tasklet(
        name=f"assign_{lhs}",
        inputs=OrderedSet(in_connectors.values()),
        outputs=OrderedSet(out_connectors.values()),
        code=f"{lhs} = {rhs}",
    )

    # Add connectors and accesses, do not duplicate array access nodes
    # As we might have array accesses to different subsets
    in_access_dict = dict()
    out_access_dict = dict()
    accesses = dict()
    for k, v in in_connectors.items():
        if isinstance(k, dace.symbolic.Subscript):
            head = str(k.args[0])
            if head not in accesses:
                accesses[head] = state.add_access(head)
            in_access_dict[k] = accesses[head]
        else:
            if str(k) not in accesses:
                accesses[str(k)] = state.add_access(str(k))
            in_access_dict[k] = accesses[str(k)]
    assert len(out_connectors.items()) == 1
    for k, v in out_connectors.items():
        if isinstance(k, sympy.Function):
            out_access_dict[k] = state.add_access(str(k.func))
        else:
            out_access_dict[k] = state.add_access(str(k))

    # Add in and out connections
    for k, v in in_connectors.items():
        if isinstance(k, dace.symbolic.Subscript):
            access_node = in_access_dict[k]
            complete_access_str = printer.doprint(k)
            state.add_edge(access_node, None, t, in_connectors[k], dace.memlet.Memlet(expr=complete_access_str))
        else:
            access_node = in_access_dict[k]
            data_name = str(k)
            state.add_edge(access_node, None, t, in_connectors[k], dace.memlet.Memlet(expr=f"{data_name}[0]"))
    assert len(out_access_dict.items()) == 1
    for k, v in out_access_dict.items():
        out_data_name = v.data
        access_str = "0"
        state.add_edge(t, out_connectors[k], v, None, dace.memlet.Memlet(expr=f"{out_data_name}[{access_str}]"))
