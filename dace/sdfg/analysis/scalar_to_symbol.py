# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Scalar to symbol promotion functionality. """

import ast
from dace import dtypes, nodes, sdfg as sd, data as dt, properties as props
import re
from typing import Dict, Set


def find_promotable_scalars(sdfg: sd.SDFG) -> Set[str]:
    """
    Finds scalars that can be promoted to symbols in the given SDFG.
    Conditions for matching a scalar for symbol-promotion are as follows:
        * Size of data must be 1, it must not be a stream and must be transient.
        * Only inputs to candidate scalars must be either arrays or tasklets.
        * All tasklets that lead to it must have one statement, one output, 
          and may have zero or more **array** inputs and not be in a scope.
        * Scalar must not be accessed with a write-conflict resolution.
        * Scalar must not be written to more than once in a state.

    These conditions must apply on all occurences of the scalar in order for
    it to be promotable.

    :param sdfg: The SDFG to query.
    :return: A set of promotable scalar names.
    """
    # Keep set of active candidates
    candidates: Set[str] = set()

    # General array checks
    for aname, desc in sdfg.arrays.items():
        if not desc.transient or isinstance(desc, dt.Stream):
            continue
        if desc.total_size != 1:
            continue
        candidates.add(aname)

    # Check all occurrences of candidates in SDFG and filter out
    for state in sdfg.nodes():
        candidates_in_state: Set[str] = set()

        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            candidate = node.data
            if candidate not in candidates:
                continue

            # If candidate is read-only, continue normally
            if state.in_degree(node) == 0:
                continue

            # Candidate may only be accessed in a top-level scope
            if state.entry_node(node) is not None:
                candidates.remove(candidate)
                continue

            # Candidate may only be written to once within a state
            if candidate in candidates_in_state:
                if state.in_degree(node) == 1:
                    candidates.remove(candidate)
                    continue
            candidates_in_state.add(candidate)

            # If input is not a single array nor tasklet, skip
            if state.in_degree(node) > 1:
                candidates.remove(candidate)
                continue
            edge = state.in_edges(node)[0]

            # Edge must not be WCR
            if edge.data.wcr is not None:
                candidates.remove(candidate)
                continue

            # Check inputs
            if isinstance(edge.src, nodes.AccessNode):
                # If input is array, ensure it is not a stream
                if isinstance(sdfg.arrays[edge.src.data], dt.Stream):
                    candidates.remove(candidate)
            elif isinstance(edge.src, nodes.Tasklet):
                # If input tasklet has more than one output, skip
                if state.out_degree(edge.src) > 1:
                    candidates.remove(candidate)
                    continue
                # If inputs to tasklets are not arrays, skip
                for tinput in state.in_edges(edge.src):
                    if not isinstance(tinput, nodes.AccessNode):
                        candidates.remove(candidate)
                        break
                    if isinstance(sdfg.arrays[tinput.data], dt.Stream):
                        candidates.remove(candidate)
                        break
                else:
                    # Check that tasklets have only one statement
                    cb: props.CodeBlock = edge.src.code
                    if cb.language is dtypes.Language.Python:
                        if (len(cb.code) > 1
                                or not isinstance(cb.code[0], ast.Assign)):
                            candidates.remove(candidate)
                            break
                    elif cb.language is dtypes.Language.CPP:
                        # Try to match a single C assignment
                        cstr = cb.as_string.strip()
                        if re.match(r'^[a-zA-Z_][a-zA-Z_0-9]* = .*;$',
                                    cstr) is None:
                            candidates.remove(candidate)
                            break
                    else:  # Other languages are currently unsupported
                        candidates.remove(candidate)
                        break
            else:  # If input is not an acceptable node type, skip
                candidates.remove(candidate)

    return candidates


def promote_scalars_to_symbols(sdfg: sd.SDFG):
    """
    Promotes all matching transient scalars to SDFG symbols, changing all
    tasklets to inter-state assignments. This enables the transformed symbols
    to be used within states as part of memlets, and allows further
    transformations (such as loop detection) to use the information for
    optimization.

    :param sdfg: The SDFG to run the pass on.
    :note: Operates in-place.
    """
    to_promote = find_promotable_scalars(sdfg)
