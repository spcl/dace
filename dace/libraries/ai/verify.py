# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Compilation probe for AI-generated tasklets.

Compiling the whole SDFG in order to check a generated tasklet is not possible: code generation
runs after expansion, and would recurse straight back into it. Instead a small standalone
translation unit is synthesized -- the connectors and symbols become function parameters, the
program state struct is rebuilt from the requested state fields -- and handed to the same compiler
DaCe would use. It is compiled but never linked or run, so the probe catches syntax and type
errors, not wrong results.

A probe that fails because an environment header cannot be found is treated as *inconclusive*
rather than as an error: header search paths generally come from CMake package discovery, which
the probe does not perform.
"""

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from dace.config import Config
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.context import ExpansionContext

logger = logging.getLogger(__name__)

#: Maximum time (in seconds) to wait for the probe compilation.
_COMPILE_TIMEOUT = 180

#: Host compiler flags that make no sense for, or actively break, a syntax-only probe.
_DROPPED_HOST_FLAGS = ('-Werror', '-fPIC', '-shared')

#: Stands in for the probe's real path in reported diagnostics, which would otherwise carry the
#: name of a temporary directory that changes on every run.
PROBE_DISPLAY_NAME = 'generated_tasklet'


@dataclass
class ProbeResult:
    """ Outcome of a probe compilation. """

    ok: bool
    inconclusive: bool = False
    command: str = ''
    stderr: str = ''
    #: The translation unit that was compiled, kept so that a probe which accepted code the real
    #: build later rejects can be compared against what DaCe actually generates.
    source: str = ''


def _headers_of(env: Any) -> List[str]:
    """
    Extracts the header list of a DaCe environment.

    ``headers`` may be a list, a per-target dictionary, or a callable returning either.

    :param env: The environment class.
    :return: The headers that apply to host or frame code.
    """
    headers = env.headers
    if callable(headers):
        try:
            headers = headers()
        except Exception:
            return []
    if isinstance(headers, dict):
        collected: List[str] = []
        for value in headers.values():
            collected.extend(value)
        return collected
    return list(headers or [])


def _env_flags(environments: Sequence[Any]) -> List[str]:
    """
    Collects include directories and compile flags contributed by environments.

    CMake variable references (``${...}``) are dropped, since the probe does not run CMake.

    :param environments: The environment classes attached to the expansion.
    :return: Compiler flags for the probe.
    """
    flags: List[str] = []
    for env in environments:
        for include in getattr(env, 'cmake_includes', None) or []:
            if isinstance(include, str) and '${' not in include:
                flags.append(f'-I{include}')
        for flag in getattr(env, 'cmake_compile_flags', None) or []:
            if isinstance(flag, str) and '${' not in flag:
                flags.append(flag)
        # Headers stored next to the environment definition are resolved relative to its file
        env_path = getattr(env, '_dace_file_path', None)
        if env_path:
            flags.append(f'-I{os.path.dirname(env_path)}')
    return flags


def probe_ctype(conn: Any) -> Optional[str]:
    """
    Returns the C++ type to declare a connector with in the probe.

    A connector whose own type could not be inferred falls back to the element type of the
    container behind it. It must fall back to *something* concrete: declaring the probe function as
    a template instead would leave its body uninstantiated, and an uninstantiated template body is
    barely checked at all -- which would turn the probe into a source of false passes.

    :param conn: The connector information.
    :return: The type, or ``None`` if nothing about the connector is known.
    """
    if conn.ctype != 'auto':
        return conn.ctype
    if not conn.data_ctype:
        return None
    return conn.data_ctype if conn.num_elements == '1' else f'{conn.data_ctype}*'


def build_probe_source(spec: TaskletSpec, ctx: ExpansionContext, environments: Sequence[Any]) -> Optional[str]:
    """
    Synthesizes a standalone translation unit around a generated tasklet.

    :param spec: The generated tasklet.
    :param ctx: The context the tasklet was generated for.
    :param environments: The environment classes attached to the expansion.
    :return: The source of the probe translation unit, or ``None`` if a connector's type is
             unknown, in which case no faithful probe can be built.
    """
    device = ctx.capabilities is not None and ctx.capabilities.device_level

    lines: List[str] = ['// Standalone probe generated by the DaCe AI library node expansion.']
    for env in environments:
        for header in _headers_of(env):
            lines.append(f'#include "{header}"' if header.startswith('.') else f'#include <{header}>')

    state_fields = list(spec.state_fields)
    for env in environments:
        state_fields.extend(getattr(env, 'state_fields', None) or [])
    lines.append('struct __dace_probe_state {')
    lines.extend(f'    {declaration}' for declaration in state_fields)
    lines.append('    int __dace_probe_placeholder;')
    lines.append('};')

    if spec.code_global.strip():
        lines.append(spec.code_global)

    params = []
    for conn in ctx.connectors:
        ctype = probe_ctype(conn)
        if ctype is None:
            logger.info(
                'Skipping AI expansion verification: the type of connector "%s" is not known, so the probe '
                'would not check the generated code faithfully.', conn.name)
            return None
        params.append(f'{ctype} {conn.name}')
    params += [f'{ctype} {name}' for name, ctype in sorted(ctx.symbols.items())]
    qualifier = '__device__ ' if device else ''
    lines.append(f'{qualifier}static void __dace_probe_tasklet({", ".join(params) or "void"}) {{')
    if not device:
        lines.append('    __dace_probe_state __dace_probe_state_value;')
        lines.append('    __dace_probe_state* __state = &__dace_probe_state_value;')
        lines.append('    (void)__state;')
        if ctx.capabilities is not None and ctx.capabilities.current_stream_available:
            lines.append('    cudaStream_t __dace_current_stream = nullptr;')
            lines.append('    (void)__dace_current_stream;')
    lines.append('    ///////////////////')
    lines.append(spec.code)
    lines.append('    ///////////////////')
    lines.append('}')

    if spec.code_init.strip() or spec.code_exit.strip():
        for name, body in (('__dace_probe_init', spec.code_init), ('__dace_probe_exit', spec.code_exit)):
            if not body.strip():
                continue
            lines.append(f'static void {name}(__dace_probe_state* __state) {{')
            lines.append('    (void)__state;')
            lines.append(body)
            lines.append('}')

    return '\n'.join(lines) + '\n'


def _standard_flag(flags: Sequence[str]) -> List[str]:
    """
    Returns the C++ standard flag to add, if the existing flags do not already set one.

    The host flags DaCe passes to CMake often already carry a ``-std=``; appending a second one
    puts two conflicting standards on the probe's command line, which then contradicts the standard
    reported to the model.

    :param flags: Flags already on the command line.
    :return: A single-element list with the flag to add, or an empty list.
    """
    if any(f.startswith('-std=') for f in flags):
        return []
    standard = Config.get('compiler', 'cpp_standard')
    return [f'-std=c++{standard}'] if standard else []


def _host_command(source: str) -> Optional[List[str]]:
    """
    Builds the command line for a host-code probe.

    :param source: Path of the probe source file.
    :return: The command line, or ``None`` if no usable compiler was found.
    """
    from dace.codegen import compiler_family  # Avoid a cyclic import through the code generator

    compiler = compiler_family.host_compiler()
    if not compiler or shutil.which(compiler) is None:
        return None
    try:
        family = compiler_family.detect(compiler)
    except Exception:
        family = None
    if family == 'msvc':
        # The probe's flag vocabulary is GCC/Clang-flavored
        return None

    try:
        flags = shlex.split(compiler_family.cpu_args())
    except Exception:
        flags = shlex.split(Config.get('compiler', 'cpu', 'args'))
    flags = [f for f in flags if f not in _DROPPED_HOST_FLAGS]
    flags += _standard_flag(flags)
    return [compiler, '-fsyntax-only', '-x', 'c++', *flags, source]


def _device_command(source: str, ctx: ExpansionContext) -> Optional[List[str]]:
    """
    Builds the command line for a GPU device-code probe.

    :param source: Path of the probe source file.
    :param ctx: The context the tasklet was generated for.
    :return: The command line, or ``None`` if no usable compiler was found.
    """
    backend = ctx.capabilities.gpu_backend if ctx.capabilities is not None else None
    architectures = ctx.target.gpu_architectures if ctx.target is not None else None
    first_arch = architectures.split(',')[0].strip() if architectures else None

    if backend == 'hip':
        compiler = shutil.which('hipcc')
        if compiler is None:
            return None
        flags = ['-fsyntax-only', '-x', 'hip']
        if first_arch:
            flags.append(f'--offload-arch={first_arch}')
        return [compiler, *flags, source]

    compiler = shutil.which('nvcc')
    if compiler is None:
        return None
    flags = ['-x', 'cu', '-c', '-o', os.devnull]
    if first_arch:
        flags.append(f'-arch={first_arch}')
    flags += _standard_flag(flags)
    return [compiler, *flags, source]


def _is_inconclusive(stderr: str) -> bool:
    """
    Decides whether a failed probe should be reported to the model.

    :param stderr: The compiler's diagnostic output.
    :return: True if the failure is an artifact of the probe rather than of the generated code.
    """
    lowered = stderr.lower()
    return 'no such file or directory' in lowered and ('#include' in lowered or 'fatal error' in lowered)


def probe_compile(spec: TaskletSpec, ctx: ExpansionContext, environments: Sequence[Any]) -> ProbeResult:
    """
    Compiles a generated tasklet in isolation.

    :param spec: The generated tasklet.
    :param ctx: The context the tasklet was generated for.
    :param environments: The environment classes attached to the expansion.
    :return: The probe outcome. An outcome with ``inconclusive`` set should be treated as a pass.
    """
    if spec.language.upper() != 'CPP':
        return ProbeResult(ok=True, inconclusive=True)

    device = ctx.capabilities is not None and ctx.capabilities.device_level
    suffix = '.cu' if device else '.cpp'
    source = build_probe_source(spec, ctx, environments)
    if source is None:
        return ProbeResult(ok=True, inconclusive=True)

    directory = tempfile.mkdtemp(prefix='dace_ai_probe_')
    source_path = os.path.join(directory, f'probe{suffix}')
    try:
        with open(source_path, 'w') as fp:
            fp.write(source)

        command = _device_command(source_path, ctx) if device else _host_command(source_path)
        if command is None:
            logger.info('Skipping AI expansion verification: no suitable compiler was found.')
            return ProbeResult(ok=True, inconclusive=True, source=source)
        command = command[:1] + _env_flags(environments) + command[1:]

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            logger.info('Skipping AI expansion verification: the probe could not be run (%s).', e)
            return ProbeResult(ok=True, inconclusive=True, source=source)

        # The probe lives in a randomly named temporary directory. Replacing that path with a
        # stable placeholder keeps the diagnostics -- and therefore any repair prompt built from
        # them -- identical across runs for identical code, which is what lets a saved answer be
        # reused and the repair loop converge.
        def stable(text: str) -> str:
            return text.replace(source_path, PROBE_DISPLAY_NAME).replace(directory, '')

        if result.returncode == 0:
            return ProbeResult(ok=True, command=stable(' '.join(command)), source=source)
        if _is_inconclusive(result.stderr):
            logger.info('AI expansion verification was inconclusive (a header could not be located by the probe).')
            return ProbeResult(ok=False,
                               inconclusive=True,
                               command=stable(' '.join(command)),
                               stderr=stable(result.stderr),
                               source=source)
        return ProbeResult(ok=False,
                           command=stable(' '.join(command)),
                           stderr=stable(result.stderr or result.stdout),
                           source=source)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
