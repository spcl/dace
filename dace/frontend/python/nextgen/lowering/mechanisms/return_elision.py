# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Elision of the copy that materializes a return value.

``rules.returns`` materializes every returned value into a ``__return``
container with an explicit copy, because by the time the ``return`` statement
is reached the container holding the value has long been written. When the
returned value IS a whole program-local array (``b = np.zeros(...); ...;
return b`` -- the shape of most programs that return an array), that copy is
pure overhead: it allocates a second full-size buffer and fills it on every
call.

The copy cannot be removed downstream. ``RedundantArray`` declines a transient
that has access nodes in more than one state, which is exactly what a returned
array is: allocated in one state and filled in another.

So remove it here, where the whole tree is in hand: rename the source container
to the return container and drop the copy. This is an optimization, and it
declines whenever the rename cannot be shown safe -- the copy then simply
stays.
"""
import re
from typing import List, Optional, Set, Tuple

from dace import data
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: Node types whose every reference to a container this pass can rewrite. A
#: type outside this set is not necessarily unsafe -- it has simply not been
#: taught to the rewriter, so a container it mentions is left alone.
_REWRITABLE = (tn.TaskletNode, tn.LibraryCall, tn.CopyNode, tn.DynScopeCopyNode, tn.ViewNode, tn.RefSetNode,
               tn.ReturnNode)

#: Attributes holding memlets, whichever way a node stores them.
_MEMLET_FIELDS = ('memlet', 'in_memlets', 'out_memlets', 'memlets')

#: Attributes naming containers directly.
_NAME_FIELDS = ('target', 'source', 'values', 'return_targets', 'data_arguments', 'receiver', 'extra_targets',
                'input_names', 'output_names', 'targets', 'container', 'name')

#: Attributes holding free-form expressions, where a container can be named
#: without this pass being able to tell where.
_EXPRESSION_FIELDS = ('condition', 'init', 'update', 'value', 'expression')

#: The subset of :data:`_NAME_FIELDS` the rewriter knows how to write back.
_REWRITABLE_NAME_FIELDS = ('target', 'source', 'values', 'return_targets', 'extra_targets', 'data_arguments')


def elide_return_copies(root: tn.ScheduleTreeRoot) -> None:
    """
    Rename returned program-local arrays to their ``__return`` container,
    dropping the copies that would otherwise materialize them.

    :param root: The lowered tree, modified in place.
    """
    returns = [node for node in root.preorder_traversal() if isinstance(node, tn.ReturnNode) and node.values]
    if len(returns) != 1:
        # With early returns, several containers reach the same ``__return``;
        # renaming one of them would leave the others writing a container that
        # no longer exists.
        return
    return_node = returns[0]

    elided: Set[str] = set()
    for return_name in list(return_node.values):
        candidate = _elidable_source(root, return_name, return_node, elided)
        if candidate is None:
            continue
        copy_node, source = candidate
        root.children.remove(copy_node)
        _rename_container(root, source, return_name)
        elided.add(source)


def _elidable_source(root: tn.ScheduleTreeRoot, return_name: str, return_node: tn.ReturnNode,
                     elided: Set[str]) -> Optional[Tuple[tn.CopyNode, str]]:
    """
    The copy materializing ``return_name`` and the container it copies from, if
    that container can take the return container's name instead.

    :param root: The lowered tree.
    :param return_name: The ``__return`` container the copy writes.
    :param return_node: The tree's single value-returning node.
    :param elided: Containers already renamed for an earlier return value, so
                   ``return b, b`` does not rename ``b`` twice.
    :return: The (copy node, source container) pair, or None to keep the copy.
    """
    copies = [
        node for node in root.preorder_traversal() if isinstance(node, tn.CopyNode) and node.target == return_name
    ]
    if len(copies) != 1 or copies[0] not in root.children:
        return None
    copy_node = copies[0]
    source = copy_node.memlet.data
    if source is None or source == return_name or source in elided:
        return None
    if source not in root.containers or return_name not in root.containers:
        return None

    source_descriptor = root.containers[source]
    return_descriptor = root.containers[return_name]
    if not source_descriptor.transient or source in root.arg_names:
        return None
    if not _interchangeable(source_descriptor, return_descriptor):
        return None
    if not _copies_whole(copy_node.memlet, source_descriptor, return_descriptor):
        return None

    # Every use of the source becomes a use of the return container, so each
    # one has to be rewritable -- and none may come after the copy, where a
    # write would land in the buffer the caller is about to read.
    seen_copy = False
    for node in root.preorder_traversal():
        if node is copy_node:
            seen_copy = True
            continue
        if node is root or not _references(node, source):
            continue
        if seen_copy or not isinstance(node, _REWRITABLE):
            return None

    # And nothing else may already be using the return container's name.
    for node in root.preorder_traversal():
        if node is root or node is copy_node or node is return_node:
            continue
        if _references(node, return_name):
            return None
    return copy_node, source


def _interchangeable(source: data.Data, target: data.Data) -> bool:
    """Whether one container can carry the other's contents under its name."""
    if type(source) is not data.Array or type(target) is not data.Array:
        # A view aliases another container and a stream has its own lifetime:
        # neither is a buffer the caller can simply be handed.
        return False
    return (source.dtype == target.dtype and list(source.shape) == list(target.shape)
            and list(source.strides) == list(target.strides) and source.storage == target.storage
            and source.start_offset == target.start_offset and source.alignment == target.alignment)


def _copies_whole(memlet: Memlet, source: data.Data, target: data.Data) -> bool:
    """Whether a copy moves all of ``source`` into all of ``target``."""
    if memlet.subset is None or memlet.subset.num_elements() != source.total_size:
        return False
    if memlet.other_subset is not None and memlet.other_subset.num_elements() != target.total_size:
        return False
    return True


def _references(node: tn.ScheduleTreeNode, name: str) -> bool:
    """
    Whether a node mentions a container, counting only what the node holds
    itself: a scope's memlet sets aggregate its children, which the traversal
    visits in their own right.
    """
    if any(memlet.data == name for memlet in _own_memlets(node)):
        return True
    if name in _own_names(node):
        return True
    return _mentions_in_expression(node, name)


def _own_memlets(node: tn.ScheduleTreeNode) -> List[Memlet]:
    """The memlets a node holds directly, in no particular order."""
    memlets: List[object] = []
    for attribute in _MEMLET_FIELDS:
        value = getattr(node, attribute, None)
        if isinstance(value, Memlet):
            memlets.append(value)
        elif isinstance(value, dict):
            memlets.extend(value.values())
        elif isinstance(value, (list, set, tuple)):
            memlets.extend(value)
    return [memlet for memlet in memlets if isinstance(memlet, Memlet)]


def _own_names(node: tn.ScheduleTreeNode) -> Set[str]:
    """The container names a node holds directly as strings."""
    names: Set[str] = set()
    for attribute in _NAME_FIELDS:
        value = getattr(node, attribute, None)
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, (list, set, tuple)):
            names.update(item for item in value if isinstance(item, str))
    if isinstance(node, tn.SDFGCallNode):
        # A nested SDFG call binds containers through argument expressions.
        names.update(str(expression) for expression in node.call.arguments.values())
    return names


def _mentions_in_expression(node: tn.ScheduleTreeNode, name: str) -> bool:
    """
    Whether a container appears inside one of a node's free-form expressions
    (a loop condition, an interstate assignment). Such an occurrence is a real
    reference this pass cannot rewrite, so finding one declines the elision.
    """
    pattern = re.compile(rf'\b{re.escape(name)}\b')
    for attribute in _EXPRESSION_FIELDS:
        value = getattr(node, attribute, None)
        if value is None:
            continue
        text = value if isinstance(value, str) else getattr(value, 'as_string', None)
        if isinstance(text, str) and pattern.search(text):
            return True
    return False


def _rename_container(root: tn.ScheduleTreeRoot, old: str, new: str) -> None:
    """
    Rename a container throughout the tree, keeping ``new``'s descriptor: it is
    the non-transient one, which is what makes the container a program output.
    """
    for node in root.preorder_traversal():
        if node is root:
            continue
        for memlet in _own_memlets(node):
            if memlet.data == old:
                memlet.data = new
        _rename_names(node, old, new)
    del root.containers[old]


def _rename_names(node: tn.ScheduleTreeNode, old: str, new: str) -> None:
    """Rewrite a node's direct string references to a container."""
    for attribute in _REWRITABLE_NAME_FIELDS:
        value = getattr(node, attribute, None)
        if isinstance(value, str):
            if value == old:
                setattr(node, attribute, new)
        elif isinstance(value, list):
            setattr(node, attribute, [new if item == old else item for item in value])
        elif isinstance(value, set):
            setattr(node, attribute, {new if item == old else item for item in value})
