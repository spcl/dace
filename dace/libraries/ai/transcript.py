# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
On-disk record of everything one AI expansion said, was told, and verified.

An expansion is a paid, non-reproducible network call whose result is baked into the SDFG, so the
material that produced it is worth keeping: when generated code turns out to be wrong -- or, worse,
when it passes the probe compile and then fails the real build -- the prompt, the model's answer
and the exact translation unit the probe accepted are what explain why.

Each expansion gets its own directory holding the prompts, one file per attempt, and the probe
source and diagnostics for each verification round, plus a machine-readable ``transcript.json``
tying them together. The directory is written incrementally, so a crashed or interrupted expansion
still leaves behind everything it had done up to that point.

Writing is controlled by ``ai.transcripts`` and located by ``ai.transcript_dir``.
"""

import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dace.config import Config

logger = logging.getLogger(__name__)

#: Name of the machine-readable index written into every transcript directory.
INDEX_NAME = 'transcript.json'


def enabled() -> bool:
    """
    :return: True if transcripts should be written.
    """
    return Config.get_bool('ai', 'transcripts')


def transcript_dir(create: bool = False) -> str:
    """
    Returns the directory holding expansion transcripts.

    :param create: If True, the directory is created when it does not exist.
    :return: An absolute path. Defaults to ``~/.dace/ai_transcripts`` when ``ai.transcript_dir``
             is empty.
    """
    configured = Config.get('ai', 'transcript_dir')
    path = os.path.expanduser(os.path.expandvars(configured)) if configured else os.path.join(
        os.path.expanduser('~'), '.dace', 'ai_transcripts')
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


class Transcript:
    """
    The record of a single library node expansion.

    Every ``record_*`` method both appends to :attr:`index` and, where the material is worth
    reading on its own, writes a separate file next to it. A transcript with writing disabled is a
    working object that simply never touches the disk, so callers need no conditionals.
    """

    def __init__(self, directory: Optional[str], index: Dict[str, Any]) -> None:
        """
        :param directory: Where to write, or ``None`` when transcripts are disabled.
        :param index: The initial contents of ``transcript.json``.
        """
        self.directory = directory
        self.index = index
        self.attempts: List[Dict[str, Any]] = []
        self.index['attempts'] = self.attempts

    @property
    def path(self) -> Optional[str]:
        """
        :return: The transcript directory, or ``None`` if nothing is being written.
        """
        return self.directory

    def _write(self, filename: str, content: str) -> None:
        """
        Writes one file into the transcript directory.

        A transcript is a diagnostic aid: failing to write one must never fail the expansion that
        was otherwise about to succeed.

        :param filename: Name of the file within the directory.
        :param content: What to write.
        """
        if self.directory is None:
            return
        try:
            with open(os.path.join(self.directory, filename), 'w') as fp:
                fp.write(content)
        except OSError:
            logger.warning('Could not write the AI expansion transcript file %s', filename, exc_info=True)

    def _flush(self) -> None:
        """ Rewrites ``transcript.json`` so an interrupted expansion still leaves a usable record. """
        if self.directory is None:
            return
        self._write(INDEX_NAME, json.dumps(self.index, indent=1, default=str))

    def record_system_prompt(self, system: str) -> None:
        """
        Records the system prompt.

        :param system: The system prompt handed to the provider.
        """
        self.index['system_prompt'] = system
        self._write('system_prompt.md', system)
        self._flush()

    def begin_attempt(self, index: int, prompt: str) -> Dict[str, Any]:
        """
        Opens a new generation attempt.

        :param index: One-based attempt number.
        :param prompt: The user message this attempt sends, which is the original prompt on the
                       first attempt and a repair request afterwards.
        :return: The attempt record, for the caller to pass back to the other methods.
        """
        attempt: Dict[str, Any] = {'attempt': index, 'prompt': prompt}
        self.attempts.append(attempt)
        self._write(f'attempt_{index}_prompt.md', prompt)
        self._flush()
        return attempt

    def record_answer(self, attempt: Dict[str, Any], spec: Any, cached: bool = False) -> None:
        """
        Records what the model returned for one attempt.

        Both the provider's raw text and the parsed specification are kept: a response that parses
        into something unexpected is only explicable from the text that arrived.

        :param attempt: The record returned by :meth:`begin_attempt`.
        :param spec: The generated specification.
        :param cached: Whether the answer was reused rather than requested, which is worth knowing
                       before concluding that a model produced something twice.
        """
        import dataclasses

        attempt['cached'] = cached
        attempt['answer'] = dataclasses.asdict(spec)
        raw = getattr(spec, 'raw_response', '')
        self._write(f'attempt_{attempt["attempt"]}_answer.json', raw or json.dumps(attempt['answer'], indent=1))
        self._flush()

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
        if source:
            self._write(f'attempt_{attempt["attempt"]}_probe.{"cu" if device else "cpp"}', source)
        if result.stderr:
            self._write(f'attempt_{attempt["attempt"]}_probe.log', f'{result.command}\n\n{result.stderr}')
        self._flush()

    def record_outcome(self, outcome: str, detail: str = '', environments: Optional[List[str]] = None) -> None:
        """
        Records how the expansion ended.

        :param outcome: ``'expanded'`` or ``'failed'``.
        :param detail: The error message, when it failed.
        :param environments: Class paths of the environments attached to the expansion.
        """
        self.index['outcome'] = outcome
        if detail:
            self.index['error'] = detail
        if environments is not None:
            self.index['environments'] = list(environments)
        self.index['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        self._flush()


def begin(node: Any, state: Any, sdfg: Any) -> Transcript:
    """
    Starts a transcript for one expansion.

    :param node: The library node being expanded.
    :param state: The state containing it.
    :param sdfg: The SDFG containing the state.
    :return: The transcript. Writing is skipped, transparently, when ``ai.transcripts`` is off or
             the directory cannot be created.
    """
    started = datetime.datetime.now()
    index = {
        'sdfg': getattr(sdfg, 'name', None),
        'state': getattr(state, 'label', None),
        'node_type': type(node).__name__,
        'node': getattr(node, 'name', None),
        'provider': Config.get('ai', 'provider'),
        'model': Config.get('ai', 'model'),
        'started': started.isoformat(timespec='seconds'),
        'outcome': 'incomplete',
    }

    if not enabled():
        return Transcript(None, index)

    stamp = started.strftime('%Y%m%d-%H%M%S')
    name = f'{_sanitize(index["sdfg"])}.{_sanitize(index["node_type"])}.{_sanitize(index["node"])}.{stamp}'
    try:
        root = transcript_dir(create=True)
        directory = os.path.join(root, name)
        # Two expansions of identically named nodes can start within the same second
        suffix = 0
        while os.path.exists(directory):
            suffix += 1
            directory = os.path.join(root, f'{name}.{suffix}')
        os.makedirs(directory)
    except OSError:
        logger.warning('Could not create an AI expansion transcript directory; continuing without one', exc_info=True)
        return Transcript(None, index)

    transcript = Transcript(directory, index)
    transcript._flush()
    return transcript
