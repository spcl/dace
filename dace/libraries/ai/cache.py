# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reuse of model answers across runs.

Expanding the same library node in the same context twice produces the same prompt, and asking the
same model the same question again costs money and time for an answer that is, at best, the one
already in hand. Answers are therefore stored on disk keyed by everything that determines them --
the provider, the model, the effort setting, the system prompt and the whole conversation -- and a
later run with an identical key skips the request entirely. A repair round asks a different
question, so it gets its own entry; changing the SDFG, the node, or the model changes the prompt or
the key and misses the cache, which is what makes stale reuse impossible rather than merely
unlikely.

This is separate from the generated tasklet living in the SDFG: that is what makes one expanded
SDFG reproducible, while this is what keeps a *re-expansion* from paying twice.

Controlled by ``ai.cache`` and located by ``ai.cache_dir``.
"""

import dataclasses
import datetime
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

from dace.config import Config
from dace.libraries.ai import backend

logger = logging.getLogger(__name__)


def enabled() -> bool:
    """
    :return: True if answers should be reused and stored.
    """
    return Config.get_bool('ai', 'cache')


def cache_dir(create: bool = False) -> str:
    """
    Returns the directory holding cached answers.

    :param create: If True, the directory is created when it does not exist.
    :return: An absolute path. Defaults to ``~/.dace/ai_cache`` when ``ai.cache_dir`` is empty.
    """
    configured = Config.get('ai', 'cache_dir')
    path = os.path.expanduser(os.path.expandvars(configured)) if configured else os.path.join(
        os.path.expanduser('~'), '.dace', 'ai_cache')
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def key(system: str, messages: List[Dict[str, str]]) -> str:
    """
    Computes the cache key of one request.

    Everything that changes the answer is hashed, and nothing that does not: the node's name and
    the time of day are absent, so two identical questions asked from different SDFGs share an
    entry, while a different model or a different effort setting does not.

    :param system: The system prompt.
    :param messages: The conversation being sent.
    :return: A hexadecimal key.
    """
    material = json.dumps(
        {
            'provider': Config.get('ai', 'provider'),
            'model': Config.get('ai', 'model'),
            'effort': Config.get('ai', 'effort'),
            'system': system,
            'messages': messages,
        },
        sort_keys=True)
    return hashlib.sha256(material.encode('utf-8')).hexdigest()


def _path(cache_key: str) -> str:
    """
    :param cache_key: The key returned by :func:`key`.
    :return: The file an entry with that key lives in.
    """
    return os.path.join(cache_dir(), f'{cache_key}.json')


def lookup(cache_key: str) -> Optional[backend.TaskletSpec]:
    """
    Returns a previously stored answer, if there is one.

    A cache that cannot be read is not an error: the request is simply made again.

    :param cache_key: The key returned by :func:`key`.
    :return: The stored specification, or ``None``.
    """
    if not enabled():
        return None
    try:
        with open(_path(cache_key), 'r') as fp:
            entry = json.load(fp)
        return backend.spec_from_dict(entry['answer'], raw=entry.get('raw', ''))
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning('Ignoring an unreadable AI answer cache entry at %s', _path(cache_key), exc_info=True)
        return None


def store(cache_key: str, spec: backend.TaskletSpec, system: str, messages: List[Dict[str, str]]) -> None:
    """
    Stores an answer for later reuse.

    The prompt is written alongside it. Nothing reads it back -- the key already covers it -- but a
    cache whose entries cannot be read by a person is one nobody can audit or prune by hand.

    :param cache_key: The key returned by :func:`key`.
    :param spec: The specification the model returned.
    :param system: The system prompt that produced it.
    :param messages: The conversation that produced it.
    """
    if not enabled():
        return
    entry = {
        'provider': Config.get('ai', 'provider'),
        'model': Config.get('ai', 'model'),
        'effort': Config.get('ai', 'effort'),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'system': system,
        'messages': messages,
        'raw': spec.raw_response,
        'answer': dataclasses.asdict(spec),
    }
    try:
        cache_dir(create=True)
        with open(_path(cache_key), 'w') as fp:
            json.dump(entry, fp, indent=1)
    except OSError:
        logger.warning('Could not write the AI answer cache entry at %s', _path(cache_key), exc_info=True)
