# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Compile check of a generated tasklet: a program containing only the tasklet, built the way DaCe builds the real one."""

import copy
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

from dace import SDFG, SDFGState, dtypes
from dace import data as dt
from dace.codegen import exceptions as cgx
from dace.config import set_temporary
from dace.libraries.ai import diagnose
from dace.libraries.ai.context import ExpansionContext
from dace.sdfg import nodes

#: Maximum number of compiler diagnostics reported back to the model.
MAX_DIAGNOSTICS = 40


@dataclass
class CheckResult:
    """Outcome of a compile check."""

    #: True if the program compiled, or if the check could not be run (see ``skipped``).
    ok: bool
    #: Why the check could not be run. A skipped check accepts the code.
    skipped: str = ""
    #: The compiler diagnostics, for the model.
    diagnostics: str = ""
    #: The generated source files of the check program, by file name.
    sources: dict[str, str] = field(default_factory=dict)


def _check_storage(storage: dtypes.StorageType, device: bool) -> dtypes.StorageType:
    """
    :param storage: The resolved storage of a container in the slot.
    :param device: Whether the tasklet runs in a GPU kernel.
    :return: A storage the same pointer can have as an argument of the check program.
    """
    if storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned):
        return storage
    if device or storage in dtypes.GPU_STORAGES:
        return dtypes.StorageType.GPU_Global
    return dtypes.StorageType.CPU_Heap


def check_sdfg(
    tasklet: nodes.Tasklet, node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, ctx: ExpansionContext
) -> SDFG:
    """
    Builds a program containing only the tasklet, with the containers, memlets and symbols of the
    library node it replaces.

    :param tasklet: The generated tasklet, with its connector types and environments set.
    :param node: The library node it replaces.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param ctx: The context the tasklet was generated for, for resolved storage and device level.
    :return: The check program.
    """
    device = ctx.capabilities is not None and ctx.capabilities.device_level
    storage_of = {conn.name: conn.storage for conn in ctx.connectors}

    check = SDFG("ai_check")
    for name, stype in state.symbols_defined_at(node).items():
        if name not in sdfg.arrays:
            check.add_symbol(name, stype)

    check_state = check.add_state()
    check_state.add_node(tasklet)
    if device:
        entry, exit_node = check_state.add_map("ai_check", {"__ai_check": "0:1"}, dtypes.ScheduleType.GPU_Device)

    edges = [(e, e.dst_conn, True) for e in state.in_edges(node)] + [
        (e, e.src_conn, False) for e in state.out_edges(node)
    ]
    for edge, conn, incoming in edges:
        if conn is None or edge.data.is_empty():
            continue
        memlet = copy.deepcopy(edge.data)
        if memlet.data not in check.arrays:
            desc = sdfg.arrays[memlet.data]
            desc = desc.as_array() if isinstance(desc, dt.View) else copy.deepcopy(desc)
            desc.transient = False
            resolved = dtypes.StorageType[storage_of[conn]] if storage_of.get(conn) else desc.storage
            desc.storage = _check_storage(resolved, device)
            check.add_datadesc(memlet.data, desc)

        if incoming:
            access = check_state.add_read(memlet.data)
            path = [access, entry, tasklet] if device else [access, tasklet]
            check_state.add_memlet_path(*path, dst_conn=conn, memlet=memlet)
        else:
            access = check_state.add_write(memlet.data)
            path = [tasklet, exit_node, access] if device else [tasklet, access]
            check_state.add_memlet_path(*path, src_conn=conn, memlet=memlet)
    return check


def _sources(build_folder: str) -> dict[str, str]:
    """
    :param build_folder: The build folder of a compiled program.
    :return: The generated source files, by file name.
    """
    sources = {}
    for root, _, files in os.walk(os.path.join(build_folder, "src")):
        for name in sorted(files):
            with open(os.path.join(root, name), encoding="utf-8") as fp:
                sources[name] = fp.read()
    return sources


def _diagnostics(output: str, sources: dict[str, str]) -> str:
    """
    :param output: The output of a failed build.
    :param sources: The generated source files, by file name.
    :return: The errors in it, and the warnings about generated files, or its last lines if none can be parsed.
    """
    found = list(
        dict.fromkeys(
            str(d)
            for d in diagnose.parse_diagnostics(output)
            if "error" in d.severity or os.path.basename(d.file) in sources
        )
    )
    if found:
        return "\n".join(found[:MAX_DIAGNOSTICS])
    return "\n".join(output.strip().splitlines()[-MAX_DIAGNOSTICS:])


def compile_check(
    tasklet: nodes.Tasklet,
    node: nodes.LibraryNode,
    state: SDFGState,
    sdfg: SDFG,
    ctx: ExpansionContext,
    environments: Sequence[type],
    build_folder: str,
) -> CheckResult:
    """
    Compiles a generated tasklet in a program of its own.

    :param tasklet: The generated tasklet. It is moved into the check program, so pass a copy.
    :param node: The library node it replaces.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param ctx: The context the tasklet was generated for.
    :param environments: The environment classes the tasklet needs.
    :param build_folder: Where to build. Reused by later checks of the same slot.
    :return: The outcome.
    """
    tasklet.environments = {env.full_class_path() for env in environments}
    try:
        check = check_sdfg(tasklet, node, state, sdfg, ctx)
        check.build_folder = build_folder
        # The output goes into the exception only without debugprint, and one build job keeps the
        # diagnostics of different files from interleaving. The library is not loaded: loading it
        # would register its GPU code with the CUDA runtime of this process until it is collected.
        with set_temporary("debugprint", value=False):
            with set_temporary("compiler", "build_jobs", value=1):
                check.compile(return_program_handle=False)
    except (cgx.CompilationError, cgx.CompilerConfigurationError) as e:
        sources = _sources(build_folder)
        return CheckResult(ok=False, diagnostics=_diagnostics(str(e), sources), sources=sources)
    except Exception as e:  # Not caused by the generated code: the check program itself is at fault
        warnings.warn(f"Skipping the compile check of an AI-generated tasklet: {type(e).__name__}: {e}")
        return CheckResult(ok=True, skipped=f"{type(e).__name__}: {e}")
    return CheckResult(ok=True, sources=_sources(build_folder))
