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

Writing is controlled by ``ai.sessions`` and located by ``ai.session_dir``.
"""

import dataclasses
import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dace.config import Config

logger = logging.getLogger(__name__)

#: Name of the machine-readable index in every session directory.
INDEX_NAME = 'session.json'

#: Name of the resumable conversation in every session directory.
CONVERSATION_NAME = 'conversation.json'


def enabled() -> bool:
    """
    :return: True if sessions should be written.
    """
    return Config.get_bool('ai', 'sessions')


def session_dir(create: bool = False) -> str:
    """
    Returns the directory holding all sessions.

    :param create: If True, the directory is created when it does not exist.
    :return: An absolute path. Defaults to ``~/.dace/ai_sessions`` when ``ai.session_dir`` is empty.
    """
    configured = Config.get('ai', 'session_dir')
    path = os.path.expanduser(os.path.expandvars(configured)) if configured else os.path.join(
        os.path.expanduser('~'), '.dace', 'ai_sessions')
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _sanitize(name: str) -> str:
    """
    Makes a name safe to use as a path component.

    :param name: The name to clean.
    :return: The cleaned name, never empty.
    """
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_.')
    return cleaned or 'unnamed'


def make_id(sdfg_name: str, node_name: str) -> str:
    """
    Builds the session id for a slot.

    :param sdfg_name: Name of the SDFG holding the node.
    :param node_name: Name of the library node.
    :return: A path-safe, human-readable identifier.
    """
    return f'{_sanitize(sdfg_name)}.{_sanitize(node_name)}'


class Session:
    """
    One slot's conversation and its rounds.

    A session with writing disabled behaves identically but never touches the disk, so callers need
    no conditionals. Every mutation rewrites the index, so an interrupted expansion still leaves a
    usable record behind.
    """

    def __init__(self, session_id: str, directory: Optional[str], index: Dict[str, Any],
                 conversation: List[Dict[str, str]]) -> None:
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
        self.index.setdefault('rounds', [])
        self._round: Optional[Dict[str, Any]] = None

    @property
    def path(self) -> Optional[str]:
        """
        :return: The session directory, or ``None`` if nothing is being written.
        """
        return self.directory

    @property
    def rounds(self) -> List[Dict[str, Any]]:
        """
        :return: The recorded rounds, oldest first.
        """
        return self.index['rounds']

    @property
    def number(self) -> int:
        """
        :return: The number of the round currently open, or of the last completed one.
        """
        return self._round['round'] if self._round is not None else len(self.rounds)

    # -- writing ---------------------------------------------------------------------------------

    def _round_dir(self, create: bool = False) -> Optional[str]:
        """
        :param create: If True, the directory is created.
        :return: Directory of the open round, or ``None`` when not writing.
        """
        if self.directory is None or self._round is None:
            return None
        path = os.path.join(self.directory, f'round_{self._round["round"]}')
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
            with open(os.path.join(directory, filename), 'w') as fp:
                fp.write(content)
        except OSError:
            logger.warning('Could not write the AI session file %s', filename, exc_info=True)

    def flush(self) -> None:
        """ Rewrites the index and the conversation so an interrupted round still leaves a record. """
        if self.directory is None:
            return
        self._write(INDEX_NAME, json.dumps(self.index, indent=1, default=str), in_round=False)
        self._write(CONVERSATION_NAME, json.dumps(self.conversation, indent=1), in_round=False)

    def begin_round(self, feedback: str = '') -> None:
        """
        Opens a new round.

        :param feedback: What prompted this round: a human note, a measurement, or a build failure.
                         Empty for the first generation.
        """
        self._round = {
            'round': len(self.rounds) + 1,
            'feedback': feedback,
            'started': datetime.datetime.now().isoformat(timespec='seconds'),
            'outcome': 'incomplete',
            'attempts': [],
        }
        self.rounds.append(self._round)
        if feedback:
            self._write('feedback.md', feedback)
        self.flush()

    def record_system_prompt(self, system: str) -> None:
        """
        Records the system prompt of the open round.

        :param system: The system prompt handed to the provider.
        """
        self.index['system_prompt_chars'] = len(system)
        self._write('system_prompt.md', system)
        self.flush()

    def begin_attempt(self, index: int, prompt: str) -> Dict[str, Any]:
        """
        Opens a generation attempt within the round.

        A round can take several attempts, because a probe failure is answered by asking again.

        :param index: One-based attempt number within the round.
        :param prompt: The user message this attempt sends.
        :return: The attempt record, to pass back to the other methods.
        """
        attempt: Dict[str, Any] = {'attempt': index, 'prompt': prompt}
        if self._round is not None:
            self._round['attempts'].append(attempt)
        self._write(f'attempt_{index}_prompt.md' if index > 1 else 'prompt.md', prompt)
        self.flush()
        return attempt

    def record_answer(self, attempt: Dict[str, Any], spec: Any, cached: bool = False) -> None:
        """
        Records what the model returned for one attempt.

        Both the provider's raw text and the parsed specification are kept: a response that parses
        into something unexpected is only explicable from the text that arrived.

        :param attempt: The record returned by :meth:`begin_attempt`.
        :param spec: The generated specification.
        :param cached: Whether the answer was reused rather than requested.
        """
        attempt['cached'] = cached
        attempt['answer'] = dataclasses.asdict(spec)
        suffix = f'_{attempt["attempt"]}' if attempt['attempt'] > 1 else ''
        raw = getattr(spec, 'raw_response', '')
        self._write(f'answer{suffix}.json', raw or json.dumps(attempt['answer'], indent=1))
        self.flush()

    def record_verification(self, attempt: Dict[str, Any], result: Any, source: str, device: bool) -> None:
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
        attempt['verification'] = {
            'ok': result.ok,
            'inconclusive': result.inconclusive,
            'command': result.command,
            'diagnostics': result.stderr,
        }
        suffix = f'_{attempt["attempt"]}' if attempt['attempt'] > 1 else ''
        if source:
            self._write(f'probe{suffix}.{"cu" if device else "cpp"}', source)
        if result.stderr:
            self._write(f'probe{suffix}.log', f'{result.command}\n\n{result.stderr}')
        self.flush()

    def record_outcome(self,
                       outcome: str,
                       detail: str = '',
                       spec: Any = None,
                       environments: Optional[List[str]] = None) -> None:
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
        self._round['outcome'] = outcome
        if detail:
            self._round['error'] = detail
        if environments is not None:
            self._round['environments'] = list(environments)
        self._round['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        if spec is not None:
            self._write('tasklet.json', json.dumps(dataclasses.asdict(spec), indent=1))
        self.flush()

    def record_node(self, node_json: Dict[str, Any]) -> None:
        """
        Stores the serialized library node this session generates for.

        :param node_json: The output of ``node.to_json(state)``.
        """
        self.index['node'] = node_json
        self.flush()

    # -- reading ---------------------------------------------------------------------------------

    def spec_of(self, round_number: int) -> Optional[Any]:
        """
        Returns the tasklet accepted in a given round.

        :param round_number: One-based round number.
        :return: The specification, or ``None`` if that round has none on record.
        """
        from dace.libraries.ai import backend

        if self.directory is None:
            return None
        path = os.path.join(self.directory, f'round_{round_number}', 'tasklet.json')
        try:
            with open(path, 'r') as fp:
                return backend.spec_from_dict(json.load(fp))
        except (OSError, ValueError, KeyError):
            return None


def load(session_id: str) -> Optional[Session]:
    """
    Opens an existing session.

    :param session_id: The slot identifier.
    :return: The session, or ``None`` if it is not on disk.
    """
    directory = os.path.join(session_dir(), session_id)
    index_path = os.path.join(directory, INDEX_NAME)
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, 'r') as fp:
            index = json.load(fp)
    except (OSError, ValueError):
        logger.warning('Ignoring an unreadable AI session index at %s', index_path, exc_info=True)
        return None

    conversation: List[Dict[str, str]] = []
    try:
        with open(os.path.join(directory, CONVERSATION_NAME), 'r') as fp:
            conversation = json.load(fp)
    except (OSError, ValueError):
        pass
    return Session(session_id, directory, index, conversation)


def begin(node: Any, state: Any, sdfg: Any, session_id: Optional[str] = None) -> Session:
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
    session_id = session_id or make_id(getattr(sdfg, 'name', 'sdfg'), getattr(node, 'name', 'node'))
    index = {
        'id': session_id,
        'sdfg': getattr(sdfg, 'name', None),
        'state': getattr(state, 'label', None),
        'node_type': type(node).__name__,
        'node_name': getattr(node, 'name', None),
        'provider': Config.get('ai', 'provider'),
        'model': Config.get('ai', 'model'),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }

    if not enabled():
        return Session(session_id, None, index, [])

    existing = load(session_id)
    if existing is not None:
        # Keep the identity recorded when the slot was first seen; refresh what may have changed
        existing.index['provider'] = index['provider']
        existing.index['model'] = index['model']
        return existing

    try:
        root = session_dir(create=True)
        directory = os.path.join(root, session_id)
        os.makedirs(directory, exist_ok=True)
    except OSError:
        logger.warning('Could not create an AI session directory; continuing without one', exc_info=True)
        return Session(session_id, None, index, [])

    created = Session(session_id, directory, index, [])
    created.flush()
    return created
