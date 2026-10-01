# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Turning a failed build into feedback for the tasklets that caused it.

The probe in :mod:`dace.libraries.ai.verify` compiles each generated tasklet **in isolation**, so
by construction it cannot see what only breaks when several of them land in one translation unit:
two tasklets whose ``code_global`` defines the same helper, colliding ``state_fields``, an
environment header the probe could only report inconclusive. Those diagnostics are exactly the
feedback the model needs, and the build is where they appear.

The hard part is saying *which* tasklet a diagnostic belongs to when the SDFG holds several.
DaCe already answers that: :meth:`dace.codegen.prettycode.CodeIOStream.write` appends
``////__DACE:<cfg>:<state>:<node>`` to every generated line, and both the tasklet body
(``targets/cpu.py``) and its ``code_global`` (``targets/cpp.py``) are emitted through annotated
writes. Attribution is therefore exact, and a diagnostic that maps to no generated tasklet is
reported rather than pinned on whichever one happened to be nearby.

The line index is built by regenerating the annotated code rather than by reading the
``map_cpp.json`` that :mod:`dace.sourcemap` writes: that file records only the *first contiguous
run* of lines per node, so for a tasklet with a ``code_global`` it describes the global block and
omits the body entirely -- which is where compile errors most often are.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dace.libraries.ai.exceptions import AIExpansionError
from dace.sdfg import SDFG

logger = logging.getLogger(__name__)

#: The annotation :class:`~dace.codegen.prettycode.CodeIOStream` leaves on every generated line.
_ANNOTATION = re.compile(r'////__DACE:(\d+):(\d+):([\d,]+)')

#: ``file:line:col: severity: message`` (GCC, Clang, nvcc) and ``file(line[,col]): severity ...``
#: (MSVC). Only the leading position matters; the rest of the line is kept as the message.
_POSIX_DIAGNOSTIC = re.compile(r'^\s*(?P<file>[^\s:][^:]*):(?P<line>\d+)(?::(?P<col>\d+))?:\s*'
                               r'(?P<severity>error|warning|note|fatal error):\s*(?P<message>.*)$')
_MSVC_DIAGNOSTIC = re.compile(r'^\s*(?P<file>[^\s(][^(]*)\((?P<line>\d+)(?:,(?P<col>\d+))?\)\s*:\s*'
                              r'(?P<severity>error|warning|note)\s*(?P<message>.*)$')

#: A location in the generated code: ``(cfg_id, state_id, node_id)``.
Location = Tuple[int, int, int]


@dataclass
class Diagnostic:
    """ One line of compiler output that names a position. """

    file: str
    line: int
    severity: str
    message: str
    column: Optional[int] = None

    def __str__(self) -> str:
        position = f'{os.path.basename(self.file)}:{self.line}'
        if self.column is not None:
            position += f':{self.column}'
        return f'{position}: {self.severity}: {self.message}'


@dataclass
class Attribution:
    """ The result of matching a build's diagnostics against the generated tasklets. """

    #: Diagnostics grouped by the tasklet they belong to, keyed by its label.
    by_slot: Dict[str, List[Diagnostic]] = field(default_factory=dict)
    #: Diagnostics that belong to no generated tasklet, and must not be pinned on one.
    unattributed: List[Diagnostic] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.by_slot)


def parse_diagnostics(text: str) -> List[Diagnostic]:
    """
    Extracts the positioned lines from compiler output.

    ``note:`` lines are kept as well as errors. They are what ties the two halves of a collision
    together -- a redefinition names the offending line and, separately, the previous definition --
    so dropping them would lose the second tasklet involved.

    :param text: The compiler's output.
    :return: One entry per diagnostic that names a file and a line, in order.
    """
    found: List[Diagnostic] = []
    for raw in (text or '').splitlines():
        match = _POSIX_DIAGNOSTIC.match(raw) or _MSVC_DIAGNOSTIC.match(raw)
        if match is None:
            continue
        column = match.group('col')
        found.append(
            Diagnostic(file=match.group('file').strip(),
                       line=int(match.group('line')),
                       severity=match.group('severity').strip(),
                       message=match.group('message').strip(),
                       column=int(column) if column else None))
    return found


def build_line_index(sdfg: SDFG) -> Dict[Tuple[str, int], Set[Location]]:
    """
    Maps every generated line to the SDFG elements that produced it.

    :param sdfg: The SDFG that was built.
    :return: ``(source file basename, line number) -> {(cfg_id, state_id, node_id)}``.
    :raises AIExpansionError: If the code cannot be regenerated.
    """
    try:
        # Non-destructive: SDFG.generate_code deep-copies before generating, and node ids are
        # positional, so they match the SDFG in hand.
        objects = sdfg.generate_code()
    except Exception as e:
        raise AIExpansionError(f'Could not regenerate the code of SDFG "{sdfg.name}" in order to attribute the '
                               f'build diagnostics to a tasklet: {e}') from e

    index: Dict[Tuple[str, int], Set[Location]] = {}
    for obj in objects:
        # The file on disk is `clean_code`, which strips the annotations as trailing comments
        # without removing any newline, so line numbers agree with the annotated text.
        name = f'{obj.name}.{obj.language}'
        for number, line in enumerate(obj.code.split('\n'), 1):
            for cfg_id, state_id, node_ids in _ANNOTATION.findall(line):
                for node_id in node_ids.split(','):
                    if not node_id:
                        continue
                    index.setdefault((name, number), set()).add((int(cfg_id), int(state_id), int(node_id)))
    return index


def _resolve(sdfg: SDFG, location: Location):
    """
    Resolves a generated-code location to the node that produced it.

    :param sdfg: The SDFG that was built.
    :param location: A ``(cfg_id, state_id, node_id)`` triple.
    :return: The node, or ``None`` if the location does not name one.
    """
    cfg_id, state_id, node_id = location
    try:
        cfg = sdfg.cfg_list[cfg_id]
        state = cfg.node(state_id)
        return state.node(node_id)
    except (IndexError, KeyError, AttributeError, TypeError):
        return None


def attribute(sdfg: SDFG, diagnostics: List[Diagnostic]) -> Attribution:
    """
    Decides which generated tasklet each diagnostic belongs to.

    :param sdfg: The SDFG that was built.
    :param diagnostics: The parsed compiler output.
    :return: The grouping, with anything unattributable kept separate.
    """
    from dace.libraries.ai import iterate

    index = build_line_index(sdfg)
    slots = {id(s.tasklet): s for s in iterate.sessions(sdfg)}

    result = Attribution()
    for diagnostic in diagnostics:
        basename = os.path.basename(diagnostic.file)
        locations = index.get((basename, diagnostic.line), set())
        implicated = {
            slots[id(node)].name
            for node in (_resolve(sdfg, loc) for loc in locations) if node is not None and id(node) in slots
        }
        if not implicated:
            result.unattributed.append(diagnostic)
            continue
        for name in implicated:
            result.by_slot.setdefault(name, []).append(diagnostic)
    return result


def _message_for(name: str, diagnostics: List[Diagnostic], others: List[str]) -> str:
    """
    Renders the feedback sent to one tasklet.

    :param name: The tasklet's label.
    :param diagnostics: The diagnostics attributed to it.
    :param others: Labels of the other tasklets implicated in the same build.
    :return: The feedback text.
    """
    lines = [
        'The program did not compile. Your tasklet was generated and verified on its own, but it '
        'is built together with the rest of the program, and these diagnostics point at code you '
        'wrote:',
        '',
    ]
    lines += [f'    {d}' for d in diagnostics]
    if others:
        listing = ', '.join(f'"{o}"' for o in others)
        lines += [
            '',
            f'The same build also implicated {listing}, which {"is" if len(others) == 1 else "are"} '
            'generated the same way and emitted into the same translation unit. If the problem is a '
            'name collision, rename what you define or give it internal linkage rather than '
            'assuming the other one will change.',
        ]
    lines += ['', 'Fix it and return the complete JSON object.']
    return '\n'.join(lines)


def repair(sdfg: SDFG, error: Any, **kwargs) -> List[str]:
    """
    Refines the tasklets a failed build blames, once.

    :param sdfg: The SDFG that failed to build.
    :param error: The exception, or the compiler output as text.
    :param kwargs: Forwarded to :func:`dace.libraries.ai.iterate.refine`.
    :return: The labels of the tasklets that were refined.
    :raises AIExpansionError: If no diagnostic can be attributed to a generated tasklet.
    """
    from dace.libraries.ai import iterate

    diagnostics = parse_diagnostics(str(error))
    if not diagnostics:
        # DaCe streams the compiler's output live and omits it from the exception when debugprint
        # is on (dace/codegen/compiler.py), so there may be nothing here to attribute. Say that,
        # rather than reporting it as "no generated code is at fault", which is a different claim.
        raise AIExpansionError('The build failure carries no compiler diagnostics to attribute, so there is nothing '
                               'to feed back. With DACE_debugprint set, DaCe prints the compiler output live and '
                               'leaves it out of the exception. Use dace.libraries.ai.build(), which captures it, or '
                               'pass the compiler output to repair() as text.')

    found = attribute(sdfg, diagnostics)
    errors = {name: [d for d in ds if 'error' in d.severity] for name, ds in found.by_slot.items()}
    blamed = sorted(name for name, ds in errors.items() if ds)
    if not blamed:
        # Refining a tasklet because it was the only candidate would be worse than doing nothing
        summary = '\n'.join(f'    {d}' for d in found.unattributed[:10]) or '    (no positioned diagnostics)'
        raise AIExpansionError('The build failed, but none of the errors is in code generated by the "ai" '
                               f'implementation, so there is nothing to ask a model to fix:\n{summary}')

    refined = []
    for name in blamed:
        slot = next((s for s in iterate.sessions(sdfg) if s.name == name), None)
        if slot is None or slot.pinned:
            logger.info('Not refining tasklet "%s": it is pinned or no longer present.', name)
            continue
        others = [o for o in blamed if o != name]
        iterate.refine(sdfg, slot.tasklet, _message_for(name, found.by_slot[name], others), **kwargs)
        refined.append(name)
    return refined


def build(sdfg: SDFG, rounds: int = 1, **kwargs):
    """
    Compiles an SDFG, asking the model to fix generated code that the build rejects.

    This is a function rather than a context manager because it has to be able to *retry*: a
    context manager cannot re-run its own body, and repairing without rebuilding would only tell
    the user their program still does not compile.

    Each round is a full rebuild and one model call per implicated tasklet, so it is bounded and
    never automatic -- an ordinary ``sdfg.compile()`` still fails the way it always has.

    :param sdfg: The SDFG to build.
    :param rounds: How many times to repair and rebuild before giving up.
    :param kwargs: Forwarded to :func:`dace.libraries.ai.iterate.refine`.
    :return: The compiled SDFG.
    :raises CompilationError: If the build still fails, or fails for a reason no generated tasklet
                              is responsible for.
    """
    from dace.codegen import exceptions as cgx
    from dace.config import set_temporary
    from dace.libraries.ai.expansion import _detail, _status

    for attempt in range(max(0, rounds) + 1):
        try:
            # DaCe prints the compiler's output live and drops it from the exception when
            # debugprint is on. The diagnostics are the entire point here, so take them in the
            # exception and re-emit them below rather than lose them to the terminal.
            with set_temporary('debugprint', value=False):
                return sdfg.compile()
        except (cgx.CompilationError, cgx.CompilerConfigurationError) as e:
            _detail('build diagnostics', str(e))
            if attempt == rounds:
                raise
            try:
                refined = repair(sdfg, e, **kwargs)
            except AIExpansionError as repair_error:
                # The build failure is not something a model can be asked to fix. Report the
                # original failure, with the reason this could not help attached.
                raise e from repair_error
            _status(f'build failed; refined {", ".join(refined)} and rebuilding '
                    f'(round {attempt + 1}/{rounds})')
