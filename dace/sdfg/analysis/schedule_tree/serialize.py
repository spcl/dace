# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""JSON (de)serialization of schedule trees (:mod:`dace.sdfg.analysis.schedule_tree.treenodes`).

The format mirrors the one used for ``.sdfg`` files as closely as possible so that external readers (e.g., the Lean
importer in ``formal/``) can reuse the same parsers for the embedded DaCe objects.  Every node becomes a dictionary

.. code-block:: json

    {"type": "<ClassName>", "<field>": ..., "children": [...]}

where scopes carry a ``"children"`` list.  Dataclass fields are serialized by name.  Embedded DaCe objects
(:class:`~dace.memlet.Memlet`, data descriptors, :class:`~dace.sdfg.nodes.Map`, :class:`~dace.sdfg.nodes.Consume`,
:class:`~dace.sdfg.nodes.Tasklet`, :class:`~dace.sdfg.nodes.LibraryNode`, :class:`~dace.properties.CodeBlock`,
loop metadata, symbolic expressions and dtypes) are encoded through the existing :mod:`dace.serialize` machinery with the
same encodings as ``.sdfg`` files.

Parent pointers are not stored; they are restored on :func:`from_json` by the scope constructors.  A small example of a
root with a single tasklet writing to ``A[0]``::

    {
      "type": "ScheduleTreeRoot",
      "name": "example",
      "containers": {"A": {"type": "Array", "attributes": {"dtype": "float64", "shape": ["10"], ...}}},
      "symbols": {},
      "constants": {},
      "callback_mapping": {},
      "arg_names": ["A"],
      "global_code": {},
      "init_code": {},
      "exit_code": {},
      "children": [
        {
          "type": "TaskletNode",
          "node": {"type": "Tasklet", "label": "t", "attributes": {...}},
          "in_memlets": {"kind": "dict", "items": []},
          "out_memlets": {"kind": "dict", "items": [["out", {"type": "Memlet", "attributes": {...}}]]}
        }
      ]
    }
"""

import json
from typing import Any

import numpy as np

import dace
from dace import data, dtypes, serialize, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.memlet_utils import MemletSet
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import LoopRegion

__all__ = ["to_json", "from_json", "save", "load"]


# --------------------------------------------------------------------------- #
# Helpers for the embedded DaCe objects
# --------------------------------------------------------------------------- #
def _datadesc_to_json(desc: data.Data) -> dict:
    """Serialize a data descriptor (``Array``/``Scalar``/``Stream``/``View``/``Reference``/...)."""
    return serialize.to_json(desc)


def _datadesc_from_json(obj: dict, context: dict) -> data.Data:
    return serialize.from_json(obj, context=context)


def _code_node_to_json(node: nodes.CodeNode) -> dict:
    """Serialize a :class:`~dace.sdfg.nodes.Tasklet` or :class:`~dace.sdfg.nodes.LibraryNode`.

    The generic ``Node.to_json`` requires the enclosing graph (for node ids and scope links), which a schedule-tree node
    does not have, so the dictionary is built directly from the node's properties, matching the ``.sdfg`` node encoding.
    """
    result = {
        "type": getattr(node, "__jsontype__", type(node).__name__),
        "label": node.label,
        "attributes": serialize.all_properties_to_json(node),
    }
    if isinstance(node, nodes.LibraryNode):
        result["classpath"] = nodes.full_class_path(node)
    return result


def _code_node_from_json(obj: dict, context: dict) -> nodes.CodeNode:
    return serialize.from_json(obj, context=context)


def _properties_obj_to_json(obj: Any) -> dict:
    """Serialize a ``@make_properties`` object (``Map``/``Consume``) without an enclosing graph."""
    return {"type": type(obj).__name__, "attributes": serialize.all_properties_to_json(obj)}


def _properties_obj_from_json(cls: type, obj: dict, context: dict):
    instance = cls.__new__(cls)
    serialize.set_properties_from_json(instance, obj, context=context)
    return instance


def _memlet_mapping_to_json(mapping: "dict[str, Memlet] | MemletSet") -> dict:
    """Serialize ``TaskletNode``/``LibraryCall`` memlet containers, preserving whether it is a dict or a set.

    Dict connector order is preserved; set entries are sorted by their serialized form for determinism.
    """
    if isinstance(mapping, dict):
        return {"kind": "dict", "items": [[connector, m.to_json()] for connector, m in mapping.items()]}
    items = sorted((m.to_json() for m in mapping), key=lambda j: json.dumps(j, sort_keys=True))
    return {"kind": "set", "items": items}


def _memlet_mapping_from_json(obj: dict, context: dict) -> "dict[str, Memlet] | MemletSet":
    if obj["kind"] == "dict":
        return {connector: Memlet.from_json(m, context=context) for connector, m in obj["items"]}
    return MemletSet(Memlet.from_json(m, context=context) for m in obj["items"])


def _loop_to_json(loop: LoopRegion) -> dict:
    """Serialize the loop metadata needed to reconstruct a bare :class:`~dace.sdfg.state.LoopRegion`.

    The loop body lives in the scope's children, so only the loop's own fields are stored.
    """
    return {
        "label": loop.label,
        "loop_condition": loop.loop_condition.to_json() if loop.loop_condition is not None else None,
        "loop_variable": loop.loop_variable,
        "init_statement": loop.init_statement.to_json() if loop.init_statement is not None else None,
        "update_statement": loop.update_statement.to_json() if loop.update_statement is not None else None,
        "inverted": loop.inverted,
        "update_before_condition": loop.update_before_condition,
        "unroll": loop.unroll,
        "unroll_factor": loop.unroll_factor,
    }


def _loop_from_json(obj: dict) -> LoopRegion:
    return LoopRegion(
        label=obj["label"],
        condition_expr=CodeBlock.from_json(obj["loop_condition"]),
        loop_var=obj["loop_variable"],
        initialize_expr=CodeBlock.from_json(obj["init_statement"]),
        update_expr=CodeBlock.from_json(obj["update_statement"]),
        inverted=obj["inverted"],
        update_before_condition=obj["update_before_condition"],
        unroll=obj["unroll"],
        unroll_factor=obj["unroll_factor"],
    )


def _codeblock_dict_to_json(mapping: "dict[str, CodeBlock]") -> dict:
    return {target: code.to_json() for target, code in mapping.items()}


def _codeblock_dict_from_json(obj: dict) -> "dict[str, CodeBlock]":
    return {target: CodeBlock.from_json(code) for target, code in obj.items()}


# --------------------------------------------------------------------------- #
# Serialization (to JSON)
# --------------------------------------------------------------------------- #
def _node_to_json(node: tn.ScheduleTreeNode) -> dict:
    cls = type(node)
    name = cls.__name__

    if cls is tn.ScheduleTreeRoot:
        result = _root_fields_to_json(node)
    elif cls is tn.NamedRegionScope:
        result = {"type": name, "label": node.label}
    elif cls in (tn.GBlock, tn.ElseScope):
        result = {"type": name}
    elif cls in (tn.LoopScope, tn.ForScope, tn.WhileScope, tn.DoWhileScope):
        result = {"type": name, "loop": _loop_to_json(node.loop)}
    elif cls in (tn.IfScope, tn.StateIfScope, tn.ElifScope):
        result = {"type": name, "condition": node.condition.to_json()}
    elif cls is tn.MapScope:
        result = {"type": name, "node": _properties_obj_to_json(node.node.map)}
    elif cls is tn.ConsumeScope:
        result = {"type": name, "node": _properties_obj_to_json(node.node.consume)}
    elif cls is tn.StateLabel:
        result = {"type": name, "state": node.name}
    elif cls is tn.GotoNode:
        result = {"type": name, "target": node.target}
    elif cls is tn.AssignNode:
        result = {"type": name, "name": node.name, "value": node.value.to_json(), "edge": node.edge.to_json()}
    elif cls in (tn.BreakNode, tn.ContinueNode):
        result = {"type": name}
    elif cls in (tn.TaskletNode, tn.LibraryCall):
        result = {
            "type": name,
            "node": _code_node_to_json(node.node),
            "in_memlets": _memlet_mapping_to_json(node.in_memlets),
            "out_memlets": _memlet_mapping_to_json(node.out_memlets),
        }
    elif cls in (tn.CopyNode, tn.DynScopeCopyNode):
        result = {"type": name, "target": node.target, "memlet": node.memlet.to_json()}
    elif cls is tn.ViewNode:
        result = {
            "type": name,
            "target": node.target,
            "source": node.source,
            "memlet": node.memlet.to_json(),
            "src_desc": _datadesc_to_json(node.src_desc),
            "view_desc": _datadesc_to_json(node.view_desc),
        }
    elif cls is tn.RefSetNode:
        result = {
            "type": name,
            "target": node.target,
            "memlet": node.memlet.to_json(),
            "src_desc": _refset_src_to_json(node.src_desc),
            "ref_desc": _datadesc_to_json(node.ref_desc),
        }
    elif cls is tn.StateBoundaryNode:
        result = {"type": name, "due_to_control_flow": node.due_to_control_flow}
    else:
        raise TypeError(f"Cannot serialize schedule tree node of unsupported type {name!r}.")

    if isinstance(node, tn.ScheduleTreeScope):
        result["children"] = [_node_to_json(child) for child in node.children]
    return result


def _refset_src_to_json(src_desc: "data.Data | nodes.CodeNode") -> dict:
    if isinstance(src_desc, nodes.CodeNode):
        # The source is a code node elsewhere in the tree; reference it by its guid and relink on load.
        return {"kind": "code_node", "guid": src_desc.guid}
    return {"kind": "data", "desc": _datadesc_to_json(src_desc)}


def _root_fields_to_json(root: tn.ScheduleTreeRoot) -> dict:
    return {
        "type": "ScheduleTreeRoot",
        "name": root.name,
        "containers": {name: _datadesc_to_json(desc) for name, desc in root.containers.items()},
        "symbols": {name: serialize.to_json(stype) for name, stype in root.symbols.items()},
        "constants": {
            name: {"desc": _datadesc_to_json(desc), "value": _constant_value_to_json(value)}
            for name, (desc, value) in root.constants.items()
        },
        "callback_mapping": dict(root.callback_mapping),
        "arg_names": list(root.arg_names),
        "global_code": _codeblock_dict_to_json(root.global_code),
        "init_code": _codeblock_dict_to_json(root.init_code),
        "exit_code": _codeblock_dict_to_json(root.exit_code),
    }


def _constant_value_to_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return serialize.to_json(value)


def to_json(root: tn.ScheduleTreeRoot) -> dict:
    """Serialize a :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ScheduleTreeRoot` to a JSON-compatible dict."""
    if not isinstance(root, tn.ScheduleTreeRoot):
        raise TypeError(f"Expected a ScheduleTreeRoot, got {type(root).__name__}.")
    # Serialize symbolic expressions with the root's declared symbol dtypes, as the SDFG serializer does, so that
    # the emitted strings are deterministic and match the ``.sdfg`` encoding.
    with symbolic.serialization_symbol_dtypes(dict(root.symbols)):
        return _node_to_json(root)


# --------------------------------------------------------------------------- #
# Deserialization (from JSON)
# --------------------------------------------------------------------------- #
def _build_context(obj: dict) -> dict:
    """Build a :mod:`dace.serialize` context with a scratch SDFG that carries the tree's symbols and containers.

    Memlets and nodes reference data containers by name and may contain symbolic expressions; the scratch SDFG lets
    :class:`~dace.properties.DataProperty` and symbolic deserialization resolve them, exactly as when loading a ``.sdfg``.
    """
    version = dace.__version__
    bare = {"sdfg": None, "version": version}

    scratch = SDFG(obj["name"])
    for name, stype in obj.get("symbols", {}).items():
        scratch.add_symbol(name, dtypes.typeclass.from_json(stype, bare), find_new_name=False)
    for name, desc_json in obj.get("containers", {}).items():
        scratch.add_datadesc(name, _datadesc_from_json(desc_json, bare))

    return {"sdfg": scratch, "sdfg_state": None, "version": version}


def _node_from_json(obj: dict, context: dict) -> tn.ScheduleTreeNode:
    name = obj["type"]

    if name == "ScheduleTreeRoot":
        return _root_from_json(obj, context)

    children = [_node_from_json(child, context) for child in obj.get("children", [])]

    if name == "NamedRegionScope":
        return tn.NamedRegionScope(label=obj["label"], children=children)
    if name == "GBlock":
        return tn.GBlock(children=children)
    if name == "ElseScope":
        return tn.ElseScope(children=children)
    if name in ("LoopScope", "ForScope", "WhileScope", "DoWhileScope"):
        return getattr(tn, name)(loop=_loop_from_json(obj["loop"]), children=children)
    if name in ("IfScope", "StateIfScope", "ElifScope"):
        return getattr(tn, name)(condition=CodeBlock.from_json(obj["condition"]), children=children)
    if name == "MapScope":
        mapobj = _properties_obj_from_json(nodes.Map, obj["node"], context)
        return tn.MapScope(node=nodes.MapEntry(mapobj), children=children)
    if name == "ConsumeScope":
        consume = _properties_obj_from_json(nodes.Consume, obj["node"], context)
        return tn.ConsumeScope(node=nodes.ConsumeEntry(consume), children=children)
    if name == "StateLabel":
        return tn.StateLabel(state=obj["state"])
    if name == "GotoNode":
        return tn.GotoNode(target=obj["target"])
    if name == "AssignNode":
        return tn.AssignNode(
            name=obj["name"],
            value=CodeBlock.from_json(obj["value"]),
            edge=InterstateEdge.from_json(obj["edge"], context=context),
        )
    if name == "BreakNode":
        return tn.BreakNode()
    if name == "ContinueNode":
        return tn.ContinueNode()
    if name in ("TaskletNode", "LibraryCall"):
        node = _code_node_from_json(obj["node"], context)
        return getattr(tn, name)(
            node=node,
            in_memlets=_memlet_mapping_from_json(obj["in_memlets"], context),
            out_memlets=_memlet_mapping_from_json(obj["out_memlets"], context),
        )
    if name in ("CopyNode", "DynScopeCopyNode"):
        return getattr(tn, name)(target=obj["target"], memlet=Memlet.from_json(obj["memlet"], context=context))
    if name == "ViewNode":
        return tn.ViewNode(
            target=obj["target"],
            source=obj["source"],
            memlet=Memlet.from_json(obj["memlet"], context=context),
            src_desc=_datadesc_from_json(obj["src_desc"], context),
            view_desc=_datadesc_from_json(obj["view_desc"], context),
        )
    if name == "RefSetNode":
        return tn.RefSetNode(
            target=obj["target"],
            memlet=Memlet.from_json(obj["memlet"], context=context),
            src_desc=_refset_src_from_json(obj["src_desc"], context),
            ref_desc=_datadesc_from_json(obj["ref_desc"], context),
        )
    if name == "StateBoundaryNode":
        return tn.StateBoundaryNode(due_to_control_flow=obj["due_to_control_flow"])

    raise TypeError(f"Cannot deserialize schedule tree node of unsupported type {name!r}.")


class _CodeNodeRef:
    """Placeholder for a ``RefSetNode.src_desc`` that is a code node, resolved to the real node after the tree loads."""

    __slots__ = ("guid",)

    def __init__(self, guid: str) -> None:
        self.guid = guid


def _refset_src_from_json(obj: dict, context: dict) -> "data.Data | _CodeNodeRef":
    if obj["kind"] == "code_node":
        return _CodeNodeRef(obj["guid"])
    return _datadesc_from_json(obj["desc"], context)


def _resolve_code_node_refs(root: tn.ScheduleTreeRoot) -> None:
    """Relink ``RefSetNode.src_desc`` placeholders to the identical code node objects they reference, by guid."""
    code_nodes = {
        n.node.guid: n.node for n in root.preorder_traversal() if isinstance(n, (tn.TaskletNode, tn.LibraryCall))
    }
    for node in root.preorder_traversal():
        if isinstance(node, tn.RefSetNode) and isinstance(node.src_desc, _CodeNodeRef):
            resolved = code_nodes.get(node.src_desc.guid)
            if resolved is None:
                raise ValueError(
                    f"Reference '{node.target}' is set by a code node (guid {node.src_desc.guid}) that is not in the "
                    "schedule tree."
                )
            node.src_desc = resolved


def _root_from_json(obj: dict, context: dict) -> tn.ScheduleTreeRoot:
    children = [_node_from_json(child, context) for child in obj.get("children", [])]
    scratch: SDFG = context["sdfg"]

    symbols = {name: dtypes.typeclass.from_json(stype, context) for name, stype in obj.get("symbols", {}).items()}
    containers = dict(scratch.arrays)
    constants = {
        name: (_datadesc_from_json(entry["desc"], context), _constant_value_from_json(entry["value"]))
        for name, entry in obj.get("constants", {}).items()
    }

    root = tn.ScheduleTreeRoot(
        name=obj["name"],
        children=children,
        containers=containers,
        symbols=symbols,
        constants=constants,
        callback_mapping=dict(obj.get("callback_mapping", {})),
        arg_names=list(obj.get("arg_names", [])),
        global_code=_codeblock_dict_from_json(obj.get("global_code", {})),
        init_code=_codeblock_dict_from_json(obj.get("init_code", {})),
        exit_code=_codeblock_dict_from_json(obj.get("exit_code", {})),
    )
    return root


def _constant_value_from_json(value: Any) -> Any:
    if isinstance(value, dict):
        return serialize.from_json(value)
    return value


def from_json(obj: dict) -> tn.ScheduleTreeRoot:
    """Deserialize a dict produced by :func:`to_json` back into a ``ScheduleTreeRoot``.

    Parent pointers (not stored) are restored by the scope constructors, and code-node references are relinked.
    """
    if not isinstance(obj, dict) or obj.get("type") != "ScheduleTreeRoot":
        raise TypeError("Expected a serialized ScheduleTreeRoot dictionary.")
    context = _build_context(obj)
    with symbolic.serialization_symbol_dtypes(dict(context["sdfg"].symbols)):
        root = _node_from_json(obj, context)
    _resolve_code_node_refs(root)
    return root


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def save(root: tn.ScheduleTreeRoot, path: str) -> None:
    """Serialize ``root`` to ``path`` as JSON (sorted keys, two-space indent, trailing newline)."""
    with open(path, "w") as f:
        f.write(json.dumps(to_json(root), sort_keys=True, indent=2))
        f.write("\n")


def load(path: str) -> tn.ScheduleTreeRoot:
    """Load a schedule tree previously written by :func:`save`."""
    with open(path) as f:
        return from_json(json.load(f))
