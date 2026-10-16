# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The user's assumptions of a performance analysis (``N>5``, ``x==y``), as relations the prover assumes."""

from collections.abc import Iterable

from dace.symbolic import Relation, RelationKind, pystr_to_symbolic


def assumption_relations(assumptions: Iterable[str]) -> list[Relation]:
    """The relations stated by ``lhs==rhs``, ``lhs>rhs`` and ``lhs<rhs`` assumption strings.

    :raise ValueError: If an assumption is none of these.
    """
    relations = []
    for assumption in assumptions:
        for operator in ("==", ">", "<"):
            if operator in assumption:
                lhs, rhs = (pystr_to_symbolic(side.strip()) for side in assumption.split(operator))
                break
        else:
            raise ValueError(f'Cannot parse assumption "{assumption}": expected "==", ">" or "<"')
        if operator == "==":
            relations.append(Relation(RelationKind.EQ, lhs, rhs))
        elif operator == ">":
            relations.append(Relation(RelationKind.LT, rhs, lhs))
        else:
            relations.append(Relation(RelationKind.LT, lhs, rhs))
    return relations
