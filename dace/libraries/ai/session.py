# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The durable record of one generated slot, across every round it goes through.

A slot is one library node in one place in one SDFG. Generating code for it is a paid,
non-reproducible call, and the first answer is rarely the last: it may compile and be slow, or run
and turn out to need a case nobody stated. So the material is kept -- the prompts, the answers, the
probe sources and diagnostics -- not as a write-only log but as a conversation that the next round
resumes.

Layout, one directory per slot::

    <session id>/
        session.json          slot identity, serialized node, and the index of rounds
        conversation.json     the running message list, replayed on the next round
        round_1/              system_prompt.md prompt.md answer.json tasklet.json probe.cpp
        round_2/              feedback.md prompt.md answer.json tasklet.json

The session id is derived from the SDFG and node names rather than from a ``guid``, because it has
to be stable *across program runs*: a guid is freshly generated every time a node is constructed,
so a guid-keyed session would start empty on every invocation of the user's script.

Writing is controlled by ``ai.sessions``; sessions are stored in ``<build folder root>/ai-sessions``.
"""

import datetime
import json
import os
import urllib.parse
import warnings
from typing import Any

from dace.config import Config
from dace.libraries.ai import backend, cache
from dace.libraries.ai.exceptions import AIExpansionError

#: Name of the machine-readable index in every session directory.
INDEX_NAME = "session.json"

#: Name of the resumable conversation in every session directory.
CONVERSATION_NAME = "conversation.json"


def enabled() -> bool:
    """
    :return: True if sessions should be written.
    """
    return Config.get_bool("ai", "sessions")


def session_dir(create: bool = False) -> str:
    """
    :param create: If True, the directory is created when it does not exist.
    :return: The directory holding all sessions.
    """
    return cache.storage_dir("sessions", create)


def make_id(node: Any, state: Any) -> str:
    """
    Builds the session id of a slot: its path from the root SDFG, as
    ``<root SDFG>/<state>/[<nested SDFG node>/<state>/...]<node>``.

    Names are percent-encoded, so that different names never map to the same path.

    :param node: The library node.
    :param state: The state containing it.
    :return: A ``/``-separated relative path.
    """
    parts = [state.label, node.label]
    sdfg = state.sdfg
    while sdfg.parent_nsdfg_node is not None:
        parts = [sdfg.parent.label, sdfg.parent_nsdfg_node.label] + parts
        sdfg = sdfg.parent_sdfg
    return "/".join(urllib.parse.quote(part, safe="") for part in [sdfg.name] + parts)


class Session:
    """
    One slot's conversation and its rounds.

    A session with writing disabled behaves identically but never touches the disk, so callers need
    no conditionals. Every mutation rewrites the index, so an interrupted expansion still leaves a
    usable record behind.
    """

    def __init__(
        self, session_id: str, directory: str | None, index: dict[str, Any], conversation: list[dict[str, str]]
    ) -> None:
        """
        :param session_id: The slot identifier.
        :param directory: Where to write, or ``None`` when sessions are disabled.
        :param index: Contents of ``session.json``.
        :param conversation: The message list carried over from previous rounds.
        """
        self.id = session_id
        self.directory = directory
        self.index = index
        self.conversation = conversation
        self.index.setdefault("rounds", [])
        self._round: dict[str, Any] | None = None

    @property
    def path(self) -> str | None:
        """
        :return: The session directory, or ``None`` if nothing is being written.
        """
        return self.directory

    @property
    def rounds(self) -> list[dict[str, Any]]:
        """
        :return: The recorded rounds, oldest first.
        """
        return self.index["rounds"]

    @property
    def number(self) -> int:
        """
        :return: The number of the round currently open, or of the last completed one.
        """
        return self._round["round"] if self._round is not None else len(self.rounds)

    # -- writing ---------------------------------------------------------------------------------

    def _round_dir(self, create: bool = False) -> str | None:
        """
        :param create: If True, the directory is created.
        :return: Directory of the open round, or ``None`` when not writing.
        """
        if self.directory is None or self._round is None:
            return None
        path = os.path.join(self.directory, f"round_{self._round['round']}")
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def _write(self, filename: str, content: str, in_round: bool = True) -> None:
        """
        Writes one file into the session or the open round.

        A session is a diagnostic aid: failing to write one must never fail an expansion that was
        otherwise about to succeed.

        :param filename: Name of the file.
        :param content: What to write.
        :param in_round: Write into the open round's directory rather than the session root.
        """
        directory = self._round_dir(create=True) if in_round else self.directory
        if directory is None:
            return
        try:
            with open(os.path.join(directory, filename), "w", encoding="utf-8") as fp:
                fp.write(content)
        except OSError as e:
            warnings.warn(f"Could not write the AI session file {filename}: {e}")

    def flush(self) -> None:
        """Rewrites the index and the conversation so an interrupted round still leaves a record."""
        if self.directory is None:
            return
        self._write(INDEX_NAME, json.dumps(self.index, indent=1, default=str), in_round=False)
        self._write(CONVERSATION_NAME, json.dumps(self.conversation, indent=1), in_round=False)

    def begin_round(self, feedback: str = "") -> None:
        """
        Opens a new round.

        :param feedback: What prompted this round: a human note, a measurement, or a build failure.
                         Empty for the first generation.
        """
        self._round = {
            "round": len(self.rounds) + 1,
            "feedback": feedback,
            "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "outcome": "incomplete",
            "attempts": [],
        }
        self.rounds.append(self._round)
        if feedback:
            self._write("feedback.md", feedback)
        self.flush()

    def record_system_prompt(self, system: str) -> None:
        """
        Records the system prompt of the open round.

        :param system: The system prompt handed to the provider.
        """
        self.index["system_prompt_chars"] = len(system)
        self._write("system_prompt.md", system)
        self.flush()

    def begin_attempt(self, index: int, prompt: str) -> dict[str, Any]:
        """
        Opens a generation attempt within the round.

        A round can take several attempts, because a probe failure is answered by asking again.

        :param index: One-based attempt number within the round.
        :param prompt: The user message this attempt sends.
        :return: The attempt record, to pass back to the other methods.
        """
        attempt: dict[str, Any] = {"attempt": index, "prompt": prompt}
        if self._round is not None:
            self._round["attempts"].append(attempt)
        self._write(f"attempt_{index}_prompt.md" if index > 1 else "prompt.md", prompt)
        self.flush()
        return attempt

    def record_answer(self, attempt: dict[str, Any], spec: Any, cached: bool = False) -> None:
        """
        Records what the model returned for one attempt.

        Both the provider's raw text and the parsed specification are kept: a response that parses
        into something unexpected is only explicable from the text that arrived.

        :param attempt: The record returned by :meth:`begin_attempt`.
        :param spec: The generated specification.
        :param cached: Whether the answer was reused rather than requested.
        """
        attempt["cached"] = cached
        attempt["answer"] = spec.model_dump()
        suffix = f"_{attempt['attempt']}" if attempt["attempt"] > 1 else ""
        self._write(f"answer{suffix}.json", spec.raw_response or spec.model_dump_json(indent=1))
        self.flush()

    def record_verification(self, attempt: dict[str, Any], result: Any, source: str, device: bool) -> None:
        """
        Records a probe compilation.

        The probe source is written out verbatim: it is the translation unit that was actually
        accepted or rejected, and comparing it against what DaCe later generates is the only way to
        explain a tasklet that passes verification and then fails the real build.

        :param attempt: The record returned by :meth:`begin_attempt`.
        :param result: The :class:`~dace.libraries.ai.verify.ProbeResult`.
        :param source: The probe translation unit.
        :param device: Whether the probe was compiled as GPU device code.
        """
        attempt["verification"] = {
            "ok": result.ok,
            "inconclusive": result.inconclusive,
            "command": result.command,
            "diagnostics": result.stderr,
        }
        suffix = f"_{attempt['attempt']}" if attempt["attempt"] > 1 else ""
        if source:
            self._write(f"probe{suffix}.{'cu' if device else 'cpp'}", source)
        if result.stderr:
            self._write(f"probe{suffix}.log", f"{result.command}\n\n{result.stderr}")
        self.flush()

    def record_outcome(
        self, outcome: str, detail: str = "", spec: Any = None, environments: list[str] | None = None
    ) -> None:
        """
        Closes the open round.

        The accepted specification is stored whole, which is what lets :func:`rollback` put an
        earlier round back without asking the model anything.

        :param outcome: ``'expanded'`` or ``'failed'``.
        :param detail: The error message, when it failed.
        :param spec: The accepted tasklet specification, when it succeeded.
        :param environments: Class paths of the environments attached to the expansion.
        """
        if self._round is None:
            return
        self._round["outcome"] = outcome
        if detail:
            self._round["error"] = detail
        if environments is not None:
            self._round["environments"] = list(environments)
        self._round["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        if spec is not None:
            self._write("tasklet.json", spec.model_dump_json(indent=1))
        self.flush()

    def record_node(self, node_json: dict[str, Any]) -> None:
        """
        Stores the serialized library node this session generates for.

        :param node_json: The output of ``node.to_json(state)``.
        """
        self.index["node"] = node_json
        self.flush()

    # -- reading ---------------------------------------------------------------------------------

    def spec_of(self, round_number: int) -> Any | None:
        """
        Returns the tasklet accepted in a given round.

        :param round_number: One-based round number.
        :return: The specification, or ``None`` if that round has none on record.
        """
        if self.directory is None:
            return None
        path = os.path.join(self.directory, f"round_{round_number}", "tasklet.json")
        try:
            with open(path, encoding="utf-8") as fp:
                return backend.parse_response(fp.read(), f"round {round_number}")
        except (OSError, AIExpansionError):
            return None


def load(session_id: str) -> Session | None:
    """
    Opens an existing session.

    :param session_id: The slot identifier.
    :return: The session, or ``None`` if it is not on disk.
    """
    directory = os.path.join(session_dir(), *session_id.split("/"))
    index_path = os.path.join(directory, INDEX_NAME)
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as fp:
            index = json.load(fp)
    except (OSError, ValueError) as e:
        warnings.warn(f"Ignoring an unreadable AI session index at {index_path}: {e}")
        return None

    conversation: list[dict[str, str]] = []
    try:
        with open(os.path.join(directory, CONVERSATION_NAME), encoding="utf-8") as fp:
            conversation = json.load(fp)
    except (OSError, ValueError):
        pass
    return Session(session_id, directory, index, conversation)


def begin(node: Any, state: Any, sdfg: Any, session_id: str | None = None) -> Session:
    """
    Opens the session for a slot, resuming it when one already exists.

    :param node: The library node being expanded.
    :param state: The state containing it.
    :param sdfg: The SDFG containing the state.
    :param session_id: An explicit identifier, when continuing a known session. Derived from the
                       SDFG and node names otherwise.
    :return: The session. Writing is skipped, transparently, when ``ai.sessions`` is off or the
             directory cannot be created.
    """
    session_id = session_id or make_id(node, state)
    index = {
        "id": session_id,
        "sdfg": getattr(sdfg, "name", None),
        "state": getattr(state, "label", None),
        "node_type": type(node).__name__,
        "node_name": getattr(node, "name", None),
        "provider": Config.get("ai", "provider"),
        "model": Config.get("ai", "model"),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    if not enabled():
        return Session(session_id, None, index, [])

    existing = load(session_id)
    if existing is not None:
        # Keep the identity recorded when the slot was first seen; refresh what may have changed
        existing.index["provider"] = index["provider"]
        existing.index["model"] = index["model"]
        return existing

    try:
        root = session_dir(create=True)
        directory = os.path.join(root, *session_id.split("/"))
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        warnings.warn(f"Could not create an AI session directory; continuing without one: {e}")
        return Session(session_id, None, index, [])

    created = Session(session_id, directory, index, [])
    created.flush()
    return created
