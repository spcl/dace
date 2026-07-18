# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Structural verification of frontend-produced schedule trees.

Runs after lowering and checks the frontend-legal output contract:

- every node type belongs to the closed frontend-legal set (in particular, no
  ``StatementNode`` survives),
- every memlet references a registered container,
- parent links are consistent,
- callback nodes reference registered containers and carry a reason.
"""
from typing import List

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.common import TreeVerificationError
from dace.frontend.python.nextgen.lowering.emitter import FRONTEND_LEGAL_NODES


def verify_tree(root: tn.ScheduleTreeRoot) -> None:
    """
    Verify the frontend output contract of a schedule tree.

    :raises TreeVerificationError: On any violation. Violations indicate
        frontend bugs, not user errors.
    """
    problems: List[str] = []

    def _known_container(name: str) -> bool:
        # The repository is a NestedDict: dotted structure-member paths
        # (``tracers.data``) resolve through the registered base Structure.
        return name in root.containers

    def _check_memlet(memlet, where: str) -> None:
        if memlet is None:
            return
        if not _known_container(memlet.data):
            problems.append(f'{where}: memlet references unknown container "{memlet.data}"')

    def _walk(scope: tn.ScheduleTreeScope) -> None:
        for child in scope.children:
            if child.parent is not scope:
                problems.append(f'{type(child).__name__}: inconsistent parent link')
            if type(child) not in FRONTEND_LEGAL_NODES:
                problems.append(f'Illegal node type in frontend output: {type(child).__name__}')
            where = type(child).__name__
            if isinstance(child, tn.TaskletNode):
                for memlet in list(child.in_memlets.values()) + list(child.out_memlets.values()):
                    _check_memlet(memlet, where)
            elif isinstance(child, tn.CopyNode):
                _check_memlet(child.memlet, where)
                if not _known_container(child.target):
                    problems.append(f'{where}: copy target "{child.target}" is not a registered container')
            elif isinstance(child, (tn.ViewNode, tn.RefSetNode)):
                _check_memlet(child.memlet, where)
                if not _known_container(child.target):
                    problems.append(f'{where}: target "{child.target}" is not a registered container')
            elif isinstance(child, tn.DynScopeCopyNode):
                # The target of a dynamic map-range copy is a symbol (fed
                # into a map's range), not a data container.
                _check_memlet(child.memlet, where)
                if child.target not in root.symbols:
                    problems.append(f'{where}: target "{child.target}" is not a registered symbol')
            elif isinstance(child, tn.PythonCallbackNode):
                if not child.reason:
                    problems.append(f'{where}: callback node without a reason')
                for name in list(child.input_names) + list(child.output_names):
                    if name not in root.containers and name not in root.symbols:
                        problems.append(f'{where}: callback references unknown name "{name}"')
            elif isinstance(child, tn.ReturnNode):
                for name in child.values:
                    if name not in root.containers:
                        problems.append(f'{where}: return references unknown container "{name}"')
            if isinstance(child, tn.ScheduleTreeScope):
                _walk(child)

    _walk(root)
    if problems:
        raise TreeVerificationError('Schedule tree verification failed (frontend bug):\n' + '\n'.join(problems))
