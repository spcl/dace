# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The leading-underscore guard must catch every binding site the underscore convention covers,
leave read-only uses and dunders alone, and -- in its default diff mode -- never punish legacy
`_private` code for a commit that never touched it.
"""
import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECK_NAMES_PATH = ROOT / 'tools' / 'check_names.py'


def load_check_names() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('check_names_under_test', CHECK_NAMES_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass(slots=True) resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


check_names = load_check_names()


def codes_for(source: str, allow_bare_underscore: bool = False) -> set:
    tree = check_names.ast.parse(source)
    visitor = check_names.NameVisitor('test.py', allow_bare_underscore)
    visitor.visit(tree)
    return {violation.code for violation in visitor.violations}


BINDING_SITES = [
    pytest.param('_x = 1', 'NAME002', id='plain-assignment'),
    pytest.param('a, _b = (1, 2)', 'NAME002', id='tuple-unpacking'),
    pytest.param('_n += 1', 'NAME002', id='augmented-assignment'),
    pytest.param('_n: int = 1', 'NAME002', id='annotated-assignment'),
    pytest.param('if (_w := 1) > 0:\n    pass', 'NAME002', id='walrus'),
    pytest.param('for _item in range(3):\n    pass', 'NAME002', id='for-target'),
    pytest.param("with open('f') as _handle:\n    pass", 'NAME002', id='with-target'),
    pytest.param('try:\n    pass\nexcept ValueError as _err:\n    pass', 'NAME002', id='except-target'),
    pytest.param('[_v for _v in range(3)]', 'NAME002', id='comprehension-variable'),
    pytest.param('def _func():\n    pass', 'NAME002', id='function-name'),
    pytest.param('class _Cls:\n    pass', 'NAME002', id='class-name'),
    pytest.param('def f(_a):\n    pass', 'NAME002', id='positional-param'),
    pytest.param('def f(_a, /):\n    pass', 'NAME002', id='positional-only-param'),
    pytest.param('def f(*, _a):\n    pass', 'NAME002', id='keyword-only-param'),
    pytest.param('def f(*_args):\n    pass', 'NAME002', id='star-args-param'),
    pytest.param('def f(**_kwargs):\n    pass', 'NAME002', id='double-star-kwargs-param'),
    pytest.param('f = lambda _x: _x', 'NAME002', id='lambda-param'),
    pytest.param('def f():\n    global _g\n    _g = 1', 'NAME002', id='global-name'),
    pytest.param(
        'def outer():\n    def inner():\n        nonlocal _n\n        _n = 1\n    return inner',
        'NAME002',
        id='nonlocal-name',
    ),
]


@pytest.mark.parametrize('source, code', BINDING_SITES)
def test_a_leading_underscore_binding_is_flagged_at_its_site(source: str, code: str) -> None:
    assert code in codes_for(source), source


ALLOWED_BINDINGS = [
    pytest.param('x = 1', id='plain-assignment'),
    pytest.param('def f(a, b):\n    return a + b', id='normal-params'),
    pytest.param('class Normal:\n    pass', id='normal-class'),
    pytest.param("__version__ = '1.0'", id='dunder-assignment'),
    pytest.param('def __init__(self):\n    pass', id='dunder-method-name'),
    pytest.param("__all__ = ['x']", id='dunder-all'),
    pytest.param('obj._private = 1', id='attribute-assignment-not-a-binding'),
    pytest.param('print(_read_only)', id='reading-a-name-is-not-a-binding'),
]


@pytest.mark.parametrize('source', ALLOWED_BINDINGS)
def test_a_normal_or_dunder_binding_is_never_flagged(source: str) -> None:
    assert codes_for(source) == set(), source


def test_a_bare_underscore_is_flagged_by_default() -> None:
    assert 'NAME003' in codes_for('_ = compute()')


def test_a_bare_underscore_is_allowed_with_the_opt_out_flag() -> None:
    assert codes_for('_ = compute()', allow_bare_underscore=True) == set()


IMPORT_CASES = [
    pytest.param('import os as os', {'NAME004'}, id='import-as-itself-is-redundant'),
    pytest.param('from collections import OrderedDict as OrderedDict', {'NAME004'},
                 id='from-import-as-itself-is-redundant'),
    pytest.param('import sys as _sys', {'NAME005'}, id='import-alias-starts-with-underscore'),
    pytest.param('from math import _floor', {'NAME005'}, id='from-import-of-underscore-name'),
    pytest.param('from math import floor as _f', {'NAME005'}, id='from-import-alias-starts-with-underscore'),
    pytest.param('import _pkg', {'NAME005'}, id='bare-import-of-underscore-name'),
    pytest.param('from __future__ import annotations', set(), id='future-import-is-always-allowed'),
    pytest.param('import os', set(), id='plain-import-is-fine'),
    pytest.param('from math import floor as fl', set(), id='ordinary-alias-is-fine'),
]


@pytest.mark.parametrize('source, expected_codes', IMPORT_CASES)
def test_import_alias_patterns_are_judged_by_the_import_specific_codes(source: str, expected_codes: set) -> None:
    assert codes_for(source) == expected_codes, source


MODULE_NAME_CASES = [
    pytest.param('_private.py', True, id='leading-underscore-module'),
    pytest.param('__helper.py', True, id='leading-double-underscore-non-dunder-module'),
    pytest.param('__init__.py', False, id='dunder-init-is-allowed'),
    pytest.param('__main__.py', False, id='dunder-main-is-allowed'),
    pytest.param('normal.py', False, id='ordinary-module-name'),
]


@pytest.mark.parametrize('name, forbidden', MODULE_NAME_CASES)
def test_a_module_file_name_starting_with_underscore_is_flagged_unless_dunder(name: str, forbidden: bool) -> None:
    violation = check_names.check_module_name(pathlib.Path(name))
    assert (violation is not None) == forbidden, name


def run_git(args: list, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def git_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    run_git(['init', '-q'], tmp_path)
    run_git(['config', 'user.email', 'test@test.com'], tmp_path)
    run_git(['config', 'user.name', 'test'], tmp_path)
    (tmp_path / 'mod.py').write_text('def ok_func():\n    _old = 1\n    return _old\n')
    run_git(['add', 'mod.py'], tmp_path)
    run_git(['commit', '-q', '-m', 'init'], tmp_path)
    return tmp_path


def test_diff_mode_flags_only_the_newly_added_leading_underscore_line(git_repo: pathlib.Path) -> None:
    """A commit that never touches `_old` must not be blocked by legacy code around it."""
    (git_repo / 'mod.py').write_text(
        'def ok_func():\n    _old = 1\n    return _old\n\n\ndef added_func():\n    _new = 2\n    return _new\n')
    run_git(['add', 'mod.py'], git_repo)
    result = subprocess.run([sys.executable, str(CHECK_NAMES_PATH), 'mod.py'],
                            cwd=git_repo,
                            capture_output=True,
                            text=True)
    assert result.returncode == 1
    assert '_new' in result.stdout
    assert '_old' not in result.stdout


def test_diff_mode_flags_the_module_name_only_for_a_newly_added_file(git_repo: pathlib.Path) -> None:
    (git_repo / '_brand_new.py').write_text('x = 1\n')
    run_git(['add', '_brand_new.py'], git_repo)
    result = subprocess.run([sys.executable, str(CHECK_NAMES_PATH), '_brand_new.py'],
                            cwd=git_repo,
                            capture_output=True,
                            text=True)
    assert result.returncode == 1
    assert 'NAME001' in result.stdout


def test_diff_mode_passes_on_a_no_op_commit(git_repo: pathlib.Path) -> None:
    result = subprocess.run([sys.executable, str(CHECK_NAMES_PATH), 'mod.py'],
                            cwd=git_repo,
                            capture_output=True,
                            text=True)
    assert result.returncode == 0
    assert result.stdout == ''


def test_all_mode_catches_the_preexisting_leading_underscore_name(git_repo: pathlib.Path) -> None:
    result = subprocess.run([sys.executable, str(CHECK_NAMES_PATH), '--all', 'mod.py'],
                            cwd=git_repo,
                            capture_output=True,
                            text=True)
    assert result.returncode == 1
    assert '_old' in result.stdout
