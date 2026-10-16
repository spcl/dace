# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
On-disk storage of the AI library under the DaCe build folder root, and the answer cache.

Answers are keyed by the request settings, the system prompt and the conversation, so a repeated
request is answered from disk. Controlled by ``ai.cache``.
"""

import datetime
import hashlib
import json
import os
import warnings

from dace.config import Config
from dace.libraries.ai import backend
from dace.sdfg import sdfg as sdfg_module


def enabled() -> bool:
    """
    :return: True if answers should be reused and stored.
    """
    return Config.get_bool("ai", "cache")


def storage_dir(kind: str, create: bool = False) -> str:
    """
    Returns a directory of the AI library under the DaCe build folder root.

    A hyphen cannot occur in an SDFG name, so these directories cannot clash with a program's build folder.

    :param kind: ``cache``, ``check``, ``environments``, ``sessions`` or ``prompts``.
    :param create: If True, the directory is created when it does not exist.
    :return: ``<build folder root>/ai-<kind>``.
    """
    path = os.path.join(sdfg_module.build_folder_root(), f"ai-{kind}")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def cache_dir(create: bool = False) -> str:
    """
    :param create: If True, the directory is created when it does not exist.
    :return: The directory holding cached answers.
    """
    return storage_dir("cache", create)


def key(system: str, messages: list[dict[str, str]], salt: str = "") -> str:
    """
    Computes the cache key of one request.

    :param system: The system prompt.
    :param messages: The conversation being sent.
    :param salt: Added to the key to ask an already asked question again, e.g. for a new round
                 without feedback.
    :return: A hexadecimal key.
    """
    material = json.dumps(
        {**backend.request_settings(), "system": system, "messages": messages, "salt": salt},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _path(cache_key: str) -> str:
    """
    :param cache_key: The key returned by :func:`key`.
    :return: The file an entry with that key lives in.
    """
    return os.path.join(cache_dir(), f"{cache_key}.json")


def lookup(cache_key: str) -> backend.TaskletSpec | None:
    """
    :param cache_key: The key returned by :func:`key`.
    :return: The stored answer, or ``None`` if there is none or it cannot be read.
    """
    if not enabled():
        return None
    try:
        with open(_path(cache_key), encoding="utf-8") as fp:
            entry = json.load(fp)
        return backend.parse_response(entry["raw"], "a cached answer")
    except FileNotFoundError:
        return None
    except Exception as e:
        warnings.warn(f"Ignoring an unreadable AI answer cache entry at {_path(cache_key)}: {e}")
        return None


def store(cache_key: str, spec: backend.TaskletSpec, system: str, messages: list[dict[str, str]]) -> None:
    """
    Stores an answer, with the request that produced it.

    :param cache_key: The key returned by :func:`key`.
    :param spec: The specification the model returned.
    :param system: The system prompt that produced it.
    :param messages: The conversation that produced it.
    """
    if not enabled():
        return
    entry = {
        **backend.request_settings(),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": system,
        "messages": messages,
        "raw": spec.raw_response or spec.model_dump_json(indent=1),
    }
    try:
        cache_dir(create=True)
        with open(_path(cache_key), "w", encoding="utf-8") as fp:
            json.dump(entry, fp, indent=1)
    except OSError as e:
        warnings.warn(f"Could not write the AI answer cache entry at {_path(cache_key)}: {e}")
