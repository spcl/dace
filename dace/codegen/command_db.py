# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Record a program's build once, replay it for later programs of the same shape.

CMake authors the compile and link commands, and ``ninja -t compdb`` reports all of them -- unlike
``compile_commands.json``, which holds compiles only. Templating the program name and folders out of
that report leaves a recipe any later SDFG configured the same way can run directly, with no CMake
and no Ninja.

Advisory, like the other build caches: an entry that does not describe the program being built is
rejected and the caller falls back to a full CMake build.
"""
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

#: Stand-ins for the three paths that differ between two programs of the same shape. Substituted
#: longest-first, since the build folder lies inside the program folder.
PLACEHOLDERS = ('$BUILD', '$PROG', '$NAME')

#: Fields of a compile-database entry that carry paths.
FIELDS = ('directory', 'command', 'file', 'output')


def entry_path(cache_root: str, key: str) -> str:
    return os.path.join(cache_root, 'commands', key + '.json')


def load(cache_root: str, key: str) -> Optional[List[Dict[str, str]]]:
    """Return the recorded build for ``key``, or ``None`` if there is none to replay."""
    try:
        with open(entry_path(cache_root, key)) as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return None


def publish(cache_root: str, key: str, entries: Sequence[Dict[str, str]]) -> None:
    """Record ``entries`` under ``key``, never overwriting an entry another build got there first."""
    path = entry_path(cache_root, key)
    if not entries or os.path.exists(path):
        return
    staging = f'{path}.{os.getpid()}'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(staging, 'w') as fp:
            json.dump(list(entries), fp)
        os.rename(staging, path)  # atomic, and loses harmlessly to a concurrent publisher
    except OSError:
        try:
            os.remove(staging)
        except OSError:
            pass


def drop(cache_root: str, key: str) -> None:
    """Forget ``key``, so one bad recording cannot go on poisoning every later build of its shape."""
    try:
        os.remove(entry_path(cache_root, key))
    except OSError:
        pass


def capture(build_folder: str) -> List[Dict[str, str]]:
    """Read back the commands Ninja just ran, or ``[]`` if they cannot be read."""
    try:
        report = subprocess.run(['ninja', '-t', 'compdb'], cwd=build_folder, capture_output=True, text=True,
                                check=True).stdout
        entries = json.loads(report)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    # Rules with no command are phony; the build.ninja rule reruns CMake, which is the point of this.
    return [e for e in entries if e.get('command') and e.get('output') != 'build.ninja']


def rewrite(entries: Sequence[Dict[str, str]], pairs: Sequence[Sequence[str]]) -> List[Dict[str, str]]:
    """Apply ``(from, to)`` string substitutions to every path-bearing field of every entry."""
    out = []
    for entry in entries:
        rewritten = dict(entry)
        for field in FIELDS:
            if field in rewritten:
                for old, new in pairs:
                    rewritten[field] = rewritten[field].replace(old, new)
        out.append(rewritten)
    return out


def template(entries: Sequence[Dict[str, str]], build_folder: str, program_folder: str,
             program_name: str) -> List[Dict[str, str]]:
    """Replace this program's identity with placeholders, leaving a recipe for its whole shape."""
    return rewrite(entries, [(build_folder, '$BUILD'), (program_folder, '$PROG'), (program_name, '$NAME')])


def accepts(entries: Sequence[Dict[str, str]], build_folder: str, program_folder: str, program_name: str,
            files: Sequence[str]) -> Optional[List[Dict[str, str]]]:
    """Substitute this program into ``entries``, or ``None`` if the recording is not about it.

    This is what makes a replay safe to trust. Every path in a recorded command comes from one of the
    three placeholders, so a substitution that lands wrong yields paths that do not exist -- but a
    recipe recorded for a *different* set of translation units would still name real files. Requiring
    the generated sources it compiles to be exactly the ones this program has closes that gap.

    Rejecting here, rather than inside :func:`replay`, is what lets the caller tell a recording that
    was never started from one that failed halfway and left a build folder to clean up.
    """
    concrete = rewrite(entries, [('$BUILD', build_folder), ('$PROG', program_folder), ('$NAME', program_name)])
    src_folder = os.path.join(program_folder, 'src')
    compiled = {os.path.relpath(e['file'], src_folder) for e in concrete if e.get('file', '').startswith(src_folder)}
    return concrete if compiled == {os.path.normpath(f) for f in files} else None


def replay(entries: Sequence[Dict[str, str]], build_folder: str, jobs: int) -> bool:
    """Run an accepted recipe. ``False`` means it failed partway and the build folder needs clearing."""

    def run(entry: Dict[str, str]) -> bool:
        directory = entry.get('directory') or build_folder
        os.makedirs(os.path.join(directory, os.path.dirname(entry['output'])), exist_ok=True)
        return subprocess.run(entry['command'], shell=True, cwd=directory, capture_output=True).returncode == 0

    # An entry reading a file no other entry produces is a source compile, so those all run at once.
    # The rest consume what the build makes -- objects, and under CUDA separable compilation a device
    # link the shared-library link then reads. Ninja reports one input per entry, too little to order
    # those by, so they run serially in the order it declared them.
    produced = {e['output'] for e in entries}
    compiles = [e for e in entries if e.get('file') not in produced]
    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            # list(), not all(): short-circuiting would leave the remaining futures unconsumed, and
            # an exception raised by one of those is dropped rather than reaching the handler below.
            if not all(list(pool.map(run, compiles))):
                return False
        if not all(run(e) for e in entries if e.get('file') in produced):
            return False
    except OSError:
        return False
    write_compile_commands(compiles, build_folder)
    return True


def write_compile_commands(compiles: Sequence[Dict[str, str]], build_folder: str) -> None:
    """Write ``compile_commands.json`` for a replayed build.

    CMake emits it whenever it configures; a replay never runs CMake, so without this a build folder
    reached by a cache hit has no compile database -- clangd stops working on generated code, and the
    only record of the exact flags a translation unit was compiled with disappears on precisely the
    builds we do most often. The recipe already holds every compile, so this is one file write.
    """
    database = [{key: entry[key] for key in ('directory', 'command', 'file')} for entry in compiles]
    with open(os.path.join(build_folder, 'compile_commands.json'), 'w') as fh:
        json.dump(database, fh, indent=2)


def clear(build_folder: str) -> None:
    """Empty a build folder a failed replay left half-populated."""
    shutil.rmtree(build_folder, ignore_errors=True)
    os.makedirs(build_folder, exist_ok=True)
