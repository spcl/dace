# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Differential-testing harness comparing the next-generation frontend against
the current staged schedule-tree frontend. Both emit
:class:`~dace.sdfg.analysis.schedule_tree.treenodes.ScheduleTreeRoot`, so the
comparison is structural: a normalized *signature* of the semantic surface
(arguments, returns, node-kind counts) rather than exact tree isomorphism,
which the two frontends do not (and need not) share.

This module is a test helper, not a test file (no ``test_`` functions).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: Node families counted by the signature. Loop-family covers all sequential
#: loop scopes; compute-family covers tasklets and library calls.
_LOOP_FAMILY = (tn.ForScope, tn.WhileScope, tn.LoopScope)
_COMPUTE_FAMILY = (tn.TaskletNode, tn.LibraryCall)


@dataclass
class TreeSignature:
    """The comparable semantic surface of a frontend-produced schedule tree."""
    arg_names: List[str]
    argument_types: Dict[str, Tuple[str, Tuple]]  #: name -> (dtype string, shape)
    #: Number of values returned by the program (0 if it returns nothing).
    #: Normalized from ReturnNodes: the frontends materialize returns into
    #: differently named containers (__return vs. source names).
    return_arity: int
    map_count: int = 0
    loop_count: int = 0
    conditional_count: int = 0
    compute_count: int = 0
    copy_count: int = 0
    callback_count: int = 0
    call_scope_count: int = 0
    containers: List[str] = field(default_factory=list)


def signature(root: tn.ScheduleTreeRoot) -> TreeSignature:
    """Compute the normalized structural signature of a schedule tree."""
    result = TreeSignature(
        arg_names=list(root.arg_names),
        argument_types={
            name: (str(root.containers[name].dtype), tuple(root.containers[name].shape))
            for name in root.arg_names if name in root.containers
        },
        return_arity=max((len(node.values) for node in root.preorder_traversal() if isinstance(node, tn.ReturnNode)),
                         default=0),
        containers=sorted(root.containers),
    )
    for node in root.preorder_traversal():
        if isinstance(node, tn.MapScope):
            result.map_count += 1
        elif isinstance(node, _LOOP_FAMILY):
            result.loop_count += 1
        elif isinstance(node, tn.IfScope):
            result.conditional_count += 1
        elif isinstance(node, _COMPUTE_FAMILY):
            result.compute_count += 1
        elif isinstance(node, tn.CopyNode):
            result.copy_count += 1
        elif isinstance(node, tn.PythonCallbackNode):
            result.callback_count += 1
        elif isinstance(node, (tn.FunctionCallScope, tn.SDFGCallNode)):
            result.call_scope_count += 1
    return result


def build_both(program) -> Tuple[tn.ScheduleTreeRoot, tn.ScheduleTreeRoot]:
    """
    Build the schedule tree of a fully annotated ``@dace.program`` with both
    frontends.

    :return: A 2-tuple of (staged-frontend tree, nextgen tree).
    """
    from dace.frontend.python import nextgen
    old_root = program.to_schedule_tree()
    new_root = nextgen.parse_program(program)
    return old_root, new_root


def has_compute(sig: TreeSignature) -> bool:
    """Whether a signature contains any lowered dataflow at all."""
    return (sig.map_count + sig.loop_count + sig.compute_count + sig.copy_count) > 0


#: Node kinds tree-to-SDFG conversion cannot lower yet.
_UNSUPPORTED_EXECUTION_KINDS = (tn.PythonCallbackNode, tn.SDFGCallNode, tn.ElifScope, tn.ViewNode, tn.RefSetNode,
                                tn.BreakNode, tn.ContinueNode)


def execution_gap(root: tn.ScheduleTreeRoot) -> str:
    """
    The first reason a tree cannot be converted to an executable SDFG yet, or
    an empty string if conversion should succeed (used to gate the execution
    comparison level of the differential harness).
    """
    for node in root.preorder_traversal():
        if isinstance(node, _UNSUPPORTED_EXECUTION_KINDS):
            return type(node).__name__
        if isinstance(node, tn.ReturnNode) and not isinstance(node.parent, tn.ScheduleTreeRoot):
            return 'early ReturnNode'
    return ''
