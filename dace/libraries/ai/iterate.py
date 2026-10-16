# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Revising generated tasklets: a new round of a node's session, with feedback::

    ai.refine(sdfg, 'gemm_tile', 'Too slow: 4.2 ms vs 1.1 ms for MKL. Try 8x8 register blocking.')

:class:`~dace.libraries.ai.nodes.ai_tasklet.AITasklet` stores the library node it replaced, so
this also works on an SDFG loaded from disk. If the new round fails, the previous tasklet is kept.
"""

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any

import dace.serialize
from dace import dtypes
from dace.libraries.ai import session as ai_session
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AITasklet
from dace.properties import CodeBlock
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil

#: Selects a generated tasklet: its name, guid or session id, the tasklet, or ``None`` if there is one.
NodeSelector = str | AITasklet | None


@dataclass
class Round:
    """One round of generation."""

    number: int
    feedback: str
    outcome: str
    started: str = ""
    finished: str = ""
    error: str = ""

    def __str__(self) -> str:
        summary = self.feedback.strip().splitlines()[0] if self.feedback.strip() else "(initial generation)"
        return f"round {self.number}  {self.outcome:9}  {summary}"


@dataclass
class SlotInfo:
    """A generated tasklet in an SDFG."""

    name: str
    session: str
    round: int
    pinned: bool
    edited: bool  #: True if the code was changed after its round
    tasklet: AITasklet
    state: SDFGState
    sdfg: SDFG

    def __str__(self) -> str:
        flags = "".join((" [pinned]" if self.pinned else "", " [hand-edited]" if self.edited else ""))
        return f"{self.name}  session={self.session}  round={self.round}{flags}"


def code_fingerprint(tasklet: nodes.Tasklet) -> str:
    """
    Fingerprints the code of a tasklet, to detect changes made after it was generated.

    :param tasklet: The tasklet to fingerprint.
    :return: A short hexadecimal digest.
    """
    material = (
        f"{tasklet.code.as_string}\0{tasklet.code_global.as_string}\0"
        f"{tasklet.code_init.as_string}\0{tasklet.code_exit.as_string}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def stamp(tasklet: AITasklet, session: str, round_number: int, node_json: dict[str, Any]) -> None:
    """
    Records on a tasklet how it was produced.

    :param tasklet: The generated tasklet.
    :param session: The session id.
    :param round_number: Which round of that conversation.
    :param node_json: The library node, as returned by ``node.to_json(state)``.
    """
    tasklet.provenance = json.dumps(
        {
            "session": session,
            "round": round_number,
            "node": node_json,
            "code_sha": code_fingerprint(tasklet),
        },
        indent=1,
    )


def read(tasklet: nodes.Tasklet) -> dict[str, Any] | None:
    """
    Reads the provenance of a tasklet.

    :param tasklet: The tasklet to inspect.
    :return: The decoded provenance, or ``None`` if it has none or it is unreadable.
    """
    if not isinstance(tasklet, AITasklet) or not tasklet.provenance:
        return None
    try:
        return json.loads(tasklet.provenance)
    except ValueError:
        warnings.warn(f'Ignoring unreadable provenance on tasklet "{tasklet.label}"')
        return None


def _iter_slots(sdfg: SDFG):
    """
    Yields every generated tasklet with provenance in an SDFG, including nested SDFGs.

    :param sdfg: The SDFG to search.
    :return: Generator of ``(tasklet, state, containing sdfg)``.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, AITasklet) and read(node) is not None:
                yield node, state, sdfg
            elif isinstance(node, nodes.NestedSDFG):
                yield from _iter_slots(node.sdfg)


def sessions(sdfg: SDFG) -> list[SlotInfo]:
    """
    Lists the generated tasklets in an SDFG.

    :param sdfg: The SDFG to search.
    :return: One entry per generated tasklet, in graph order.
    """
    found = []
    for tasklet, state, owner in _iter_slots(sdfg):
        prov = read(tasklet)
        found.append(
            SlotInfo(
                name=tasklet.label,
                session=prov.get("session", ""),
                round=int(prov.get("round", 0)),
                pinned=bool(prov.get("pinned")),
                edited=prov.get("code_sha", "") != code_fingerprint(tasklet),
                tasklet=tasklet,
                state=state,
                sdfg=owner,
            )
        )
    return found


def _select(sdfg: SDFG, node: NodeSelector) -> SlotInfo:
    """
    Finds the generated tasklet a selector names.

    :param sdfg: The SDFG to search.
    :param node: A name, a guid, an :class:`AITasklet`, or ``None`` when there is exactly one slot.
    :return: The selected slot.
    :raises AIExpansionError: If nothing matches, or the selection is ambiguous.
    """
    slots = sessions(sdfg)
    if not slots:
        raise AIExpansionError(f'SDFG "{sdfg.name}" holds no AI-generated tasklets.')

    if isinstance(node, AITasklet):
        matches = [s for s in slots if s.tasklet is node]
    elif node is None:
        matches = slots
    else:
        matches = [s for s in slots if s.name == node or s.tasklet.guid == node or s.session == node]

    if not matches:
        listing = "\n".join(f"  {s}" for s in slots)
        raise AIExpansionError(f'No AI-generated tasklet matches {node!r} in SDFG "{sdfg.name}". Available:\n{listing}')
    if len(matches) > 1:
        listing = "\n".join(f"  {s}" for s in matches)
        what = "this SDFG holds more than one" if node is None else f"{node!r} is ambiguous"
        raise AIExpansionError(f"Cannot tell which tasklet to use: {what}. Name one of:\n{listing}")
    return matches[0]


def _restore_node(slot: SlotInfo) -> nodes.LibraryNode:
    """
    Puts the library node back in place of its generated tasklet.

    :param slot: The generated tasklet.
    :return: The restored library node, now in the state and carrying the tasklet's edges.
    :raises AIExpansionError: If the recorded node cannot be deserialized.
    """
    prov = read(slot.tasklet)
    node_json = (prov or {}).get("node")
    if not node_json:
        raise AIExpansionError(f'Tasklet "{slot.name}" records no library node, so it cannot be regenerated.')

    context = {"sdfg": slot.sdfg, "sdfg_state": slot.state}
    restored = dace.serialize.from_json(node_json, context=context)
    if not isinstance(restored, nodes.LibraryNode):
        raise AIExpansionError(
            f'The library node recorded on tasklet "{slot.name}" deserialized into {type(restored).__name__}; '
            "its class may no longer be importable."
        )

    slot.state.add_node(restored)
    sdutil.change_edge_dest(slot.state, slot.tasklet, restored)
    sdutil.change_edge_src(slot.state, slot.tasklet, restored)
    slot.state.remove_node(slot.tasklet)
    return restored


def _reinstate(slot: SlotInfo, library_node: nodes.LibraryNode, tasklet: AITasklet) -> None:
    """
    Undoes :func:`_restore_node` after a failed round.

    :param slot: The generated tasklet.
    :param library_node: The library node currently in the graph.
    :param tasklet: The tasklet to put back.
    """
    slot.state.add_node(tasklet)
    sdutil.change_edge_dest(slot.state, library_node, tasklet)
    sdutil.change_edge_src(slot.state, library_node, tasklet)
    slot.state.remove_node(library_node)


def refine(sdfg: SDFG, node: NodeSelector = None, message: str = "", **kwargs) -> AITasklet:
    """
    Revises a generated tasklet in a new round of its session. On failure, the previous tasklet is kept.

    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet (see :data:`NodeSelector`).
    :param message: The feedback. If empty, the model is asked for a different version.
    :param kwargs: Forwarded to the expansion.
    :return: The new tasklet.
    :raises AIExpansionError: If the tasklet cannot be found, is pinned, or the new round fails.
    """
    slot = _select(sdfg, node)
    if slot.pinned:
        raise AIExpansionError(
            f'Tasklet "{slot.name}" is pinned, so it will not be regenerated. Call '
            "dace.libraries.ai.unpin() first if that is what you want."
        )

    if slot.edited:
        # The model revises the edited code
        message = (f"{message}\n\n" if message else "") + (
            "Note: the code was edited by hand after you wrote it. This is what is in the program "
            f"now, and it is what you should revise:\n\n{slot.tasklet.code.as_string}"
        )

    prov = read(slot.tasklet) or {}
    previous = slot.tasklet
    library_node = _restore_node(slot)
    try:
        library_node.expand(slot.state, "ai", feedback=message or " ", session=prov.get("session"), **kwargs)
    except Exception:
        _reinstate(slot, library_node, previous)
        raise

    return _select(sdfg, slot.name).tasklet


def history(sdfg: SDFG, node: NodeSelector = None) -> list[Round]:
    """
    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet.
    :return: The rounds, oldest first. Empty if the session is no longer on disk.
    """
    slot = _select(sdfg, node)
    record = ai_session.load(slot.session)
    if record is None:
        return []
    return [
        Round(
            number=int(r.get("round", 0)),
            feedback=r.get("feedback", ""),
            outcome=r.get("outcome", ""),
            started=r.get("started", ""),
            finished=r.get("finished", ""),
            error=r.get("error", ""),
        )
        for r in record.rounds
    ]


def rollback(sdfg: SDFG, node: NodeSelector = None, *, round: int) -> AITasklet:
    """
    Puts the code of an earlier round back, from the session on disk.

    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet.
    :param round: The round to restore, one-based.
    :return: The tasklet now in the graph.
    :raises AIExpansionError: If that round has no code on record.
    """
    slot = _select(sdfg, node)
    record = ai_session.load(slot.session)
    spec = record.spec_of(round) if record is not None else None
    if spec is None:
        available = ", ".join(str(r.number) for r in history(sdfg, node)) or "none"
        raise AIExpansionError(
            f'Round {round} of session "{slot.session}" has no code on record. Rounds available: {available}.'
        )

    language = dtypes.Language.Python if spec.language.upper() == "PYTHON" else dtypes.Language.CPP
    tasklet = slot.tasklet
    tasklet.code = CodeBlock(spec.code, language)
    # The auxiliary blocks are always C++, whatever language the body is written in
    tasklet.code_global = CodeBlock(spec.code_global, dtypes.Language.CPP)
    tasklet.code_init = CodeBlock(spec.code_init, dtypes.Language.CPP)
    tasklet.code_exit = CodeBlock(spec.code_exit, dtypes.Language.CPP)
    tasklet.state_fields = list(spec.state_fields)
    tasklet.side_effects = spec.side_effects or None
    tasklet.ignored_symbols = set(spec.ignored_symbols)

    prov = read(tasklet) or {}
    stamp(tasklet, session=prov.get("session", slot.session), round_number=round, node_json=prov.get("node", {}))
    return tasklet


def show(sdfg: SDFG, node: NodeSelector = None) -> str:
    """
    Prints the code of a generated tasklet.

    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet.
    :return: The printed text.
    """
    slot = _select(sdfg, node)
    lines = [str(slot), ""]
    if slot.tasklet.code_global.as_string.strip():
        lines += ["--- code_global ---", slot.tasklet.code_global.as_string, ""]
    lines += ["--- code ---", slot.tasklet.code.as_string]
    for label, block in (("code_init", slot.tasklet.code_init), ("code_exit", slot.tasklet.code_exit)):
        if block.as_string.strip():
            lines += ["", f"--- {label} ---", block.as_string]
    if slot.tasklet.state_fields:
        lines += ["", "--- state_fields ---"] + [f"  {f}" for f in slot.tasklet.state_fields]

    rendered = "\n".join(lines)
    print(rendered)
    return rendered


def pin(sdfg: SDFG, node: NodeSelector = None) -> None:
    """
    Protects a generated tasklet from being regenerated, e.g. by :func:`refine` or a repair.

    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet.
    """
    _set_pinned(sdfg, node, True)


def unpin(sdfg: SDFG, node: NodeSelector = None) -> None:
    """
    Allows a pinned tasklet to be regenerated again.

    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet.
    """
    _set_pinned(sdfg, node, False)


def _set_pinned(sdfg: SDFG, node: NodeSelector, value: bool) -> None:
    """
    Sets the pinned flag in a tasklet's provenance.

    :param sdfg: The SDFG holding the tasklet.
    :param node: Which tasklet.
    :param value: The new value.
    """
    slot = _select(sdfg, node)
    prov = read(slot.tasklet) or {}
    prov["pinned"] = value
    slot.tasklet.provenance = json.dumps(prov, indent=1)
