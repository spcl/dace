# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" dace.graphlib is a drop-in stand-in for networkx (`from dace import graphlib as nx`), so a call
    site can reference a symbol networkx has but graphlib never wrapped and nothing complains until
    that line is actually executed. That is not hypothetical: dace/codegen/targets/cuda.py called
    `nx.node_connected_component` through the graphlib alias for real, a live AttributeError on a
    path main handled fine via real networkx. This scan pins the whole class in one pass. """
import ast
import pathlib

import dace.graphlib as gl

_DACE_ROOT = pathlib.Path(gl.__file__).parent.parent


def _graphlib_aliases(tree: ast.Module) -> tuple[set, set]:
    """Local names bound to dace.graphlib, and symbols pulled straight out of it."""
    module_aliases, direct = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_aliases.update(a.asname for a in node.names if a.name == 'dace.graphlib' and a.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'dace':
                module_aliases.update(a.asname or a.name for a in node.names if a.name == 'graphlib')
            elif node.module == 'dace.graphlib':
                direct.update(a.name for a in node.names)
    return module_aliases, direct


def test_every_graphlib_symbol_reached_from_dace_is_exported():
    accesses, missing = 0, []
    for path in sorted(_DACE_ROOT.rglob('*.py')):
        if 'graphlib' in path.parts:
            continue  # graphlib itself IS the wrapper, so it may name anything
        if 'external' in path.parts:
            continue  # vendored third-party submodules, some of which are not even Python 3
        tree = ast.parse(path.read_text())
        module_aliases, direct = _graphlib_aliases(tree)
        for name in direct:
            accesses += 1
            if not hasattr(gl, name):
                missing.append(f'{path.relative_to(_DACE_ROOT.parent)}: from dace.graphlib import {name}')
        if not module_aliases:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in module_aliases:
                    accesses += 1
                    if not hasattr(gl, node.attr):
                        missing.append(f'{path.relative_to(_DACE_ROOT.parent)}:{node.lineno}: '
                                       f'{node.value.id}.{node.attr}')
    assert not missing, ('dace/ reaches graphlib symbols that graphlib does not provide -- add a wrapper (with '
                         'an implementation on every backend) rather than reaching past it:\n  ' + '\n  '.join(missing))
    # guard against the scan silently matching nothing (e.g. after an import-style change)
    assert accesses > 100, f'only {accesses} graphlib symbol accesses found -- the scan has stopped matching'
