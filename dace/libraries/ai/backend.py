# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Provider-independent interface between the AI library node expansion and a language model.

The expansion never imports a vendor SDK directly. It asks :func:`get_provider` for an
:class:`LLMProvider`, hands it a system prompt and a conversation, and receives a
:class:`TaskletSpec` back. Each provider imports its own SDK lazily, so ``dace`` remains importable
-- and every other library node keeps working -- on a machine where no provider SDK is installed.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from dace.config import Config
from dace.libraries.ai.exceptions import AIExpansionError

#: Environment variable names used when ``ai.api_key_envvar`` is left at its schema default.
DEFAULT_KEY_ENVVARS = {'anthropic': 'ANTHROPIC_API_KEY', 'responses': 'OPENAI_API_KEY'}

#: Providers accepted by ``ai.provider``. ``manual`` needs neither a key nor an SDK.
SUPPORTED_PROVIDERS = ('anthropic', 'responses', 'manual')


@dataclass
class EnvironmentSpec:
    """
    A DaCe library environment requested by the model.

    The fields mirror those required by :func:`dace.library.environment`; see
    :mod:`dace.libraries.ai.environments` for how one is turned into a real environment class.
    """

    name: str
    headers: List[str] = field(default_factory=list)
    cmake_packages: List[str] = field(default_factory=list)
    cmake_libraries: List[str] = field(default_factory=list)
    cmake_includes: List[str] = field(default_factory=list)
    cmake_compile_flags: List[str] = field(default_factory=list)
    cmake_link_flags: List[str] = field(default_factory=list)
    cmake_variables: Dict[str, str] = field(default_factory=dict)
    cmake_minimum_version: Optional[str] = None
    state_fields: List[str] = field(default_factory=list)
    init_code: str = ''
    finalize_code: str = ''


@dataclass
class TaskletSpec:
    """ A tasklet as described by the model, before it is turned into a :class:`~dace.sdfg.nodes.Tasklet`. """

    code: str
    language: str = 'CPP'
    code_global: str = ''
    code_init: str = ''
    code_exit: str = ''
    state_fields: List[str] = field(default_factory=list)
    side_effects: bool = False
    ignored_symbols: List[str] = field(default_factory=list)
    use_environments: List[str] = field(default_factory=list)
    environments: List[EnvironmentSpec] = field(default_factory=list)
    notes: str = ''

    #: The provider's answer exactly as it arrived, kept for the transcript. A response that parses
    #: into something unexpected can only be explained from the text that produced it.
    raw_response: str = ''


#: JSON schema describing :class:`TaskletSpec`, used for structured model output.
RESPONSE_SCHEMA: Dict[str, Any] = {
    'type':
    'object',
    'additionalProperties':
    False,
    'required': [
        'notes', 'language', 'code', 'code_global', 'code_init', 'code_exit', 'state_fields', 'side_effects',
        'ignored_symbols', 'use_environments', 'environments'
    ],
    'properties': {
        'notes': {
            'type': 'string',
            'description': 'Why the code is written this way, and any assumption it relies on.'
        },
        'language': {
            'type':
            'string',
            'enum': ['CPP', 'Python'],
            'description': ('Language of the tasklet body. Use CPP: the whole contract above -- code_global, '
                            'include placement, state_fields, the ban on "return" -- describes a C++ tasklet, and '
                            'anything but a bare Python assignment expression needs it. Choose Python only for a '
                            'body that is a single assignment of a Python expression and needs none of those.')
        },
        'code': {
            'type': 'string',
            'description': 'The tasklet body. Never empty.'
        },
        'code_global': {
            'type': 'string',
            'description': 'File-scope code: includes, helper functions, type definitions. May be empty.'
        },
        'code_init': {
            'type': 'string',
            'description': 'Host code run once at program initialization. __state is in scope. May be empty.'
        },
        'code_exit': {
            'type': 'string',
            'description': 'Host code run once at program finalization. __state is in scope. May be empty.'
        },
        'state_fields': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'Raw C++ member declarations added to the program state struct, e.g. "fftw_plan p0;".'
        },
        'side_effects': {
            'type': 'boolean',
            'description': 'True if the code performs I/O or mutates global state and must not be eliminated.'
        },
        'ignored_symbols': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'Identifiers in the code that coincide with SDFG symbol names but are local to the code.'
        },
        'use_environments': {
            'type':
            'array',
            'items': {
                'type': 'string'
            },
            'description': ('Environments that already exist on this machine, by the exact class path listed in the '
                            'context, e.g. "dace.libraries.blas.environments.openblas.OpenBLAS". Prefer this over '
                            'describing the same library again in "environments".')
        },
        'environments': {
            'type':
            'array',
            'description': ('External libraries the code needs and that are NOT already listed in the context. Empty '
                            'unless a new dependency is truly required.'),
            # Every property is listed in "required": structured output requires an exhaustive
            # list, and a field that is not needed is returned empty rather than omitted.
            'items': {
                'type':
                'object',
                'additionalProperties':
                False,
                'required': [
                    'name', 'headers', 'cmake_packages', 'cmake_libraries', 'cmake_includes', 'cmake_compile_flags',
                    'cmake_link_flags', 'cmake_minimum_version', 'state_fields', 'init_code', 'finalize_code'
                ],
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'A short identifier for the environment, e.g. "FFTW".'
                    },
                    'headers': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'cmake_packages': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'cmake_libraries': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'cmake_includes': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'cmake_compile_flags': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'cmake_link_flags': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'cmake_minimum_version': {
                        'type': ['string', 'null']
                    },
                    'state_fields': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    },
                    'init_code': {
                        'type': 'string'
                    },
                    'finalize_code': {
                        'type': 'string'
                    },
                },
            },
        },
    },
}


class LLMProvider(Protocol):
    """ Minimal interface a language model backend must implement. """

    def generate(self, system: str, messages: List[Dict[str, str]]) -> TaskletSpec:
        """
        Requests a tasklet from the model.

        :param system: The system prompt.
        :param messages: The conversation so far, as ``{'role': ..., 'content': ...}`` dicts.
        :return: The tasklet described by the model.
        """
        ...


def spec_from_dict(payload: Dict[str, Any], raw: str = '') -> TaskletSpec:
    """
    Converts a decoded JSON response into a :class:`TaskletSpec`.

    :param payload: The decoded response object.
    :param raw: The provider's answer as text, kept verbatim for the transcript.
    :return: The corresponding tasklet specification.
    :raises AIExpansionError: If the response does not contain a tasklet body.
    """
    code = (payload.get('code') or '').strip()
    if not code:
        raise AIExpansionError('The model returned an empty tasklet body. An empty body would also '
                               'suppress the generated global code and state fields.')

    environments = []
    for env in payload.get('environments') or []:
        environments.append(
            EnvironmentSpec(name=env['name'],
                            headers=list(env.get('headers') or []),
                            cmake_packages=list(env.get('cmake_packages') or []),
                            cmake_libraries=list(env.get('cmake_libraries') or []),
                            cmake_includes=list(env.get('cmake_includes') or []),
                            cmake_compile_flags=list(env.get('cmake_compile_flags') or []),
                            cmake_link_flags=list(env.get('cmake_link_flags') or []),
                            cmake_variables=dict(env.get('cmake_variables') or {}),
                            cmake_minimum_version=env.get('cmake_minimum_version') or None,
                            state_fields=list(env.get('state_fields') or []),
                            init_code=env.get('init_code') or '',
                            finalize_code=env.get('finalize_code') or ''))

    return TaskletSpec(code=code,
                       language=payload.get('language') or 'CPP',
                       code_global=payload.get('code_global') or '',
                       code_init=payload.get('code_init') or '',
                       code_exit=payload.get('code_exit') or '',
                       state_fields=list(payload.get('state_fields') or []),
                       side_effects=bool(payload.get('side_effects')),
                       ignored_symbols=list(payload.get('ignored_symbols') or []),
                       use_environments=list(payload.get('use_environments') or []),
                       environments=environments,
                       notes=payload.get('notes') or '',
                       raw_response=raw)


def api_key(provider: str) -> Optional[str]:
    """
    Reads the API key for a provider from the configured environment variable.

    Only the *name* of the variable is configurable -- a key is never stored in the DaCe
    configuration file.

    :param provider: The configured provider name.
    :return: The key, or ``None`` if the variable is unset or empty (in which case the provider may
             still authenticate through its own credential chain).
    """
    envvar = Config.get('ai', 'api_key_envvar') or DEFAULT_KEY_ENVVARS.get(provider, '')
    if not envvar:
        return None
    return os.environ.get(envvar) or None


def missing_sdk_error(provider: str, package: str, extra: str, exc: BaseException) -> AIExpansionError:
    """
    Builds the error raised when a provider's SDK is not installed.

    :param provider: The configured provider name.
    :param package: The name of the missing Python package.
    :param extra: The DaCe optional-dependency extra that installs it.
    :param exc: The original :class:`ImportError`.
    :return: An error with actionable installation instructions.
    """
    error = AIExpansionError(f"The 'ai' library node expansion needs the {package} Python package, which is not "
                             f'installed.\n'
                             f"    pip install 'dace[{extra}]'   (or: pip install {package})\n"
                             f'Configured provider: {provider} (change it with DACE_ai_provider).')
    error.__cause__ = exc
    return error


def missing_credentials_error(provider: str, detail: Optional[str] = None) -> AIExpansionError:
    """
    Builds the error raised when a provider has no usable credentials.

    :param provider: The configured provider name.
    :param detail: What went wrong, if the provider's SDK reported something specific.
    :return: An error naming the environment variable that is actually consulted.
    """
    envvar = Config.get('ai', 'api_key_envvar') or DEFAULT_KEY_ENVVARS.get(provider, '(unset)')
    message = (f"The '{provider}' provider has no usable credentials, so the 'ai' library node implementation "
               f'cannot generate code.\n'
               f'    export {envvar}=...   (the variable name comes from DACE_ai_api_key_envvar)')
    if detail:
        message += f'\nUnderlying error: {detail}'
    return AIExpansionError(message)


def get_provider() -> LLMProvider:
    """
    Instantiates the provider selected by the ``ai.provider`` configuration entry.

    :return: A ready-to-use provider.
    :raises AIExpansionError: If the provider is unknown or its SDK is unavailable.
    """
    provider = (Config.get('ai', 'provider') or 'anthropic').strip().lower()
    if provider == 'anthropic':
        from dace.libraries.ai.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if provider == 'responses':
        from dace.libraries.ai.providers.responses_provider import ResponsesProvider
        return ResponsesProvider()
    if provider == 'manual':
        from dace.libraries.ai.providers.manual_provider import ManualProvider
        return ManualProvider()
    raise AIExpansionError(f"Unknown AI provider '{provider}'. Supported providers are "
                           f"{', '.join(SUPPORTED_PROVIDERS)} (set DACE_ai_provider).")
