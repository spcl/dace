# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime checks for assumptions used by symbolic proofs."""

from collections.abc import Collection

from dace import SDFG, symbolic
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.sdfg.symbol_repo import program_assumptions


def generate_assumption_check(sdfg: SDFG, readable: Collection[str]) -> tuple[str, str]:
    """Emit a checker and its call, including assumptions of nested SDFGs.

    :param sdfg: The top-level SDFG after code generation has recorded its proofs.
    :param readable: Names available as program arguments.
    :returns: The helper definition and a call using the current arguments.
    :raises ValueError: If an assumption cannot be checked from those arguments.
    """
    assumptions = program_assumptions(sdfg)
    readable = set(readable)
    parameters = sdfg.symbol_repo.params
    for name, predicates in sorted(parameters.predicates.items()):
        if name in readable:
            assumptions.extend(
                symbolic.predicate_relation(predicate, symbolic.symbol(name))
                for predicate in sorted(predicates, key=lambda predicate: predicate.name)
            )
    constants = {name: value for name, value in sdfg.constants.items() if name not in sdfg.arrays}
    for relation in parameters.relations:
        relation = symbolic.Relation(
            relation.kind,
            symbolic.replace_symbols(relation.lhs, constants),
            symbolic.replace_symbols(relation.rhs, constants),
        )
        # Relations over unused parameters cannot affect an executed expression.
        if not symbolic.relation_names(relation) or symbolic.relation_names(relation) & readable:
            assumptions.append(relation)
    assumptions = list(dict.fromkeys(assumptions))
    if not assumptions:
        return "", ""
    names = sorted(set().union(*(symbolic.relation_names(relation) for relation in assumptions)))
    unreadable = set(names) - set(readable)
    if unreadable:
        raise ValueError(f"Runtime assumptions name {sorted(unreadable)}, which {sdfg.name} does not take")
    name = f"__dace_check_assumptions_{sdfg.name}"
    parameters = ", ".join([*names, "**__dace_unused"])
    stream = PythonCodeIOStream()
    prefix = "*, " if names else ""
    stream.write(f"\ndef {name}({prefix}{parameters}):")
    with stream.indented():
        # Evaluate integer expressions with Python integers, including direct calls
        # with NumPy scalar arguments: fixed-width addition could wrap otherwise.
        for symbol in names:
            stream.write(
                f"if isinstance({symbol}, numpy.generic) or "
                f"(isinstance({symbol}, numpy.ndarray) and {symbol}.ndim == 0):"
            )
            with stream.indented():
                stream.write(f"{symbol} = {symbol}.item()")
        for relation in assumptions:
            # A guard must evaluate the relation, without assuming it is already true.
            with symbolic.printing_facts(symbolic.Facts.none):
                lhs, rhs = (symbolic.symstr(side, cpp_mode=False) for side in (relation.lhs, relation.rhs))
            operators = {
                symbolic.RelationKind.LT: "<",
                symbolic.RelationKind.LE: "<=",
                symbolic.RelationKind.EQ: "==",
                symbolic.RelationKind.NE: "!=",
                symbolic.RelationKind.MUCH_LT: "<",
            }
            if relation.kind is symbolic.RelationKind.DIVIDES:
                condition = f"(({rhs}) == 0 if ({lhs}) == 0 else ({rhs}) % ({lhs}) == 0)"
                description = f"{relation.lhs} divides {relation.rhs}"
            elif relation.kind in operators:
                operator = operators[relation.kind]
                condition = f"({lhs}) {operator} ({rhs})"
                description = f"{relation.lhs} {operator} {relation.rhs}"
            else:
                raise ValueError(f"Unsupported runtime assumption: {relation}")
            stream.write(f"if not ({condition}):")
            with stream.indented():
                message = f"{sdfg.name!r} requires {description}, which does not hold for "
                values = ", ".join(f"{symbol}={{{symbol}!r}}" for symbol in sorted(symbolic.relation_names(relation)))
                stream.write(f"raise ValueError({message!r} + f{values!r})")
    arguments = ", ".join(f"{symbol}={symbol}" for symbol in names)
    return stream.getvalue(), f"{name}({arguments})"
