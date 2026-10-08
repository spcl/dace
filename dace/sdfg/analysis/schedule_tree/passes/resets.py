# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Skipping resets of containers that nothing wrote since they were last reset."""
import ast
from typing import Dict, List, Optional, Set

from dace import data, dtypes, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import ancestors, names_written

# Storage and lifetimes of containers allocated at most once per call, so that their values stay between resets
_ONCE_STORAGE = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_Pinned)
_ONCE_LIFETIMES = (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.Global,
                   dtypes.AllocationLifetime.SDFG, dtypes.AllocationLifetime.External)


def _perfect_nest(loop: tn.ForScope) -> List[tn.ForScope]:
    chain = [loop]
    while len(chain[-1].children) == 1 and type(chain[-1].children[0]) is tn.ForScope:
        chain.append(chain[-1].children[0])
    return chain


def _constant_header(loop: tn.ForScope) -> bool:
    """Whether a loop runs over the same iterations every time: its header reads no names but its variable."""
    variable = loop.loop.loop_variable
    return bool(variable) and all(code.get_free_symbols() <= {variable} for code in loop.loop.get_meta_codeblocks())


def _constant_store(node: tn.ScheduleTreeNode, variables: Set[str]) -> bool:
    """Whether a statement stores constants into single elements indexed only by ``variables``."""
    if not isinstance(node, tn.TaskletNode) or node.in_memlets:
        return False
    tasklet = node.node
    if tasklet.language != dtypes.Language.Python or getattr(tasklet, 'side_effects', False) or tasklet.free_symbols:
        return False
    for statement in tasklet.code.code:
        for n in ast.walk(statement):
            if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in ('float', 'int', 'bool')):
                return False
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in ('float', 'int', 'bool'):
                return False  # Reads something other than literals
    for memlet in node.out_memlets.values():
        if memlet.wcr is not None or not set(map(str, memlet.free_symbols)) <= variables:
            return False
        if any(symbolic.simplify(end - start) != 0 for start, end, _ in memlet.subset.ndrange()):
            return False
    return True


def _writers(root: tn.ScheduleTreeRoot) -> Dict[str, Optional[List[tn.TaskletNode]]]:
    """The tasklets that write each container, or ``None`` for containers that other nodes write or alias."""
    containers = root.containers
    writers: Dict[str, Optional[List[tn.TaskletNode]]] = {}
    for node in root.preorder_traversal():
        if isinstance(node, (tn.ViewNode, tn.RefSetNode)):
            for name in (node.source, node.target) if isinstance(node, tn.ViewNode) else names_written(node):
                writers[name] = None
            continue
        written = names_written(node) & containers.keys()
        if isinstance(node, tn.TaskletNode):
            for name in written:
                if writers.get(name, []) is not None:
                    writers.setdefault(name, []).append(node)
        else:
            for name in written:
                writers[name] = None
    return writers


def _flag_tasklet(flag: str, value: bool) -> tn.TaskletNode:
    return tn.TaskletNode(node=nodes.Tasklet('dirty' if value else 'clean', {}, {'__out'}, f'__out = {value}'),
                          in_memlets={},
                          out_memlets={'__out': Memlet(f'{flag}[0]')})


def skip_redundant_resets(stree: tn.ScheduleTreeScope) -> int:
    """
    Skip resets of containers that nothing wrote since they were last reset: a loop nest that stores constants into
    the same elements of some containers every time it runs (e.g., zeroing arrays of corrections before a loop that
    rarely writes them) runs only if one of those containers was written elsewhere since.

    A flag records that: it is set at the beginning of the program and after every other write to the containers,
    and cleared after the nest, which then runs only if the flag is set. If the flag is clear, the nest ran before
    and nothing wrote the containers since, so they still hold the values the nest would store.

    The nest must be a perfect nest of ``for`` loops with headers that read only their own variables, inside a loop
    (otherwise it runs once anyway), whose body consists of tasklets without inputs that store literals into single
    elements indexed by the loop variables. The containers must be transients allocated at most once per call (not on
    the stack or per scope), written elsewhere only by tasklets under ``if`` scopes (writes that run more often would
    set the flag all the time), and neither viewed nor written by other nodes. Run it before ``reuse_transients``,
    which would otherwise let other transients share these containers: after this pass, the containers are read before
    they are written in calls that skip the nest, so that pass leaves them alone, and ``move_small_transients_to_stack``
    allocates such containers once per call (only with ``zero_read_before_written``).

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of nests made to skip.
    """
    root = stree.get_root()
    containers = root.containers
    skipped = 0
    # The outermost loops of perfect nests
    candidates = [
        n for n in stree.preorder_traversal()
        if type(n) is tn.ForScope and not (type(n.parent) is tn.ForScope and len(n.parent.children) == 1)
    ]
    for loop in candidates:
        nest = _perfect_nest(loop)
        if not all(_constant_header(l) for l in nest):
            continue
        if not any(isinstance(a, tn.LoopScope) for a in ancestors(loop)):
            continue
        if any(isinstance(a, tn.MapScope) for a in ancestors(loop)):
            continue
        variables = {l.loop.loop_variable for l in nest}
        body = nest[-1].children
        if not body or not all(_constant_store(n, variables) for n in body):
            continue
        reset_containers = {m.data for n in body for m in n.out_memlets.values()}
        if not all(containers[c].transient and type(containers[c]) is data.Array
                   and containers[c].storage in _ONCE_STORAGE and containers[c].lifetime in _ONCE_LIFETIMES
                   for c in reset_containers):
            continue
        writers = _writers(root)
        others: List[tn.TaskletNode] = []
        ok = True
        own = {id(n) for n in body}
        for name in reset_containers:
            if writers.get(name) is None:
                ok = False
                break
            others += [w for w in writers[name] if id(w) not in own]
        if not ok or not others:
            continue
        if any(not any(type(a) is tn.IfScope
                       for a in ancestors(w)) or any(isinstance(a, tn.MapScope) for a in ancestors(w)) for w in others):
            continue
        flag = data.find_new_name('__dirty', containers)
        containers[flag] = data.Scalar(dtypes.bool_, transient=True)
        # Set at the beginning of the program
        children = list(root.children)
        root.children = []
        root.add_children([_flag_tasklet(flag, True)] + children)
        # Set after every other write (once per writing statement)
        seen = set()
        for writer in others:
            if id(writer) in seen:
                continue
            seen.add(id(writer))
            parent = writer.parent
            position = next(k for k, c in enumerate(parent.children) if c is writer)
            siblings = list(parent.children)
            siblings.insert(position + 1, _flag_tasklet(flag, True))
            parent.children = []
            parent.add_children(siblings)
        # The nest runs only if the flag is set, and clears it
        parent = loop.parent
        position = next(k for k, c in enumerate(parent.children) if c is loop)
        siblings = list(parent.children)
        siblings[position] = tn.IfScope(condition=CodeBlock(flag), children=[loop, _flag_tasklet(flag, False)])
        parent.children = []
        parent.add_children(siblings)
        skipped += 1
    return skipped
