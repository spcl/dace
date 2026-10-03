"""Forbid leading-underscore names and redundant/underscore import aliases.

Flags, by AST inspection:

1. A module file name starting with ``_`` (dunders such as ``__init__.py`` are
   allowed).
2. Any NAME BOUND in the file that starts with ``_``: assignment targets
   (plain, tuple/list unpacking, augmented, annotated, walrus), for/with/except
   targets, comprehension variables, function and class names, every parameter
   kind (positional, positional-only, keyword-only, ``*args``, ``**kwargs``),
   lambda parameters, ``global``/``nonlocal`` names, and import bindings.
   Dunders (``__x__``) are allowed. A bare ``_`` is a violation unless
   ``--allow-bare-underscore`` is passed. Attribute access and other
   read-only uses are never flagged.
3. Import patterns: a redundant self-alias (``import x as x`` or
   ``from m import x as x``), any import alias starting with ``_``, and
   ``from m import _name`` with no alias. ``from __future__ import ...`` is
   always allowed.

Two modes. ``--diff [REF]`` (the default, used by pre-commit) reports only
violations on lines added relative to the git index (``git diff --cached``) or
to REF when given; the module-name rule then applies only to newly added
files, so pre-existing ``_private`` code never blocks a commit. ``--all``
audits whole files regardless of git history.

Kept byte-identical between optarena's tools/check_names.py and dace's
tools/check_names.py; edit one, then copy it verbatim over the other.

Output: ``path:line:col: NAMEnnn message``, one per violation, sorted by
position; exit 1 when any violation is reported. Non-``.py`` arguments are
skipped, so it accepts file lists the way pre-commit passes them.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    col: int
    code: str
    message: str


def is_dunder(name: str) -> bool:
    """True for a real dunder like ``__init__`` or ``__all__``, not bare ``__``/``___``."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__") and name[2:-2].strip("_") != ""


class NameVisitor(ast.NodeVisitor):
    """Walk one module's AST and collect every leading-underscore binding."""

    def __init__(self, path: str, allow_bare_underscore: bool) -> None:
        self.path = path
        self.allow_bare_underscore = allow_bare_underscore
        self.violations: list[Violation] = []

    def flag_binding(self, name: str, line: int, col: int) -> None:
        if name == "_":
            if not self.allow_bare_underscore:
                self.violations.append(Violation(self.path, line, col, "NAME003", "bare '_' binding not allowed"))
            return
        if name.startswith("_") and not is_dunder(name):
            self.violations.append(Violation(self.path, line, col, "NAME002", f"name '{name}' must not start with '_'"))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.flag_binding(node.id, node.lineno, node.col_offset + 1)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.flag_binding(node.arg, node.lineno, node.col_offset + 1)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.flag_binding(node.name, node.lineno, node.col_offset + 1)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.flag_binding(node.name, node.lineno, node.col_offset + 1)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.flag_binding(node.name, node.lineno, node.col_offset + 1)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.flag_binding(node.name, node.lineno, node.col_offset + 1)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self.flag_binding(name, node.lineno, node.col_offset + 1)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self.flag_binding(name, node.lineno, node.col_offset + 1)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.flag_import_alias(alias, node, dotted=True)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "__future__":
            return
        for alias in node.names:
            self.flag_import_alias(alias, node, dotted=False)

    def flag_import_alias(self, alias: ast.alias, stmt: ast.stmt, dotted: bool) -> None:
        line = getattr(alias, "lineno", stmt.lineno)
        col = getattr(alias, "col_offset", stmt.col_offset) + 1
        bound_name = alias.asname if alias.asname else (alias.name.split(".")[0] if dotted else alias.name)
        if alias.asname and alias.asname == alias.name:
            self.violations.append(Violation(self.path, line, col, "NAME004", f"redundant alias 'as {alias.asname}'"))
        if bound_name.startswith("_"):
            self.violations.append(Violation(self.path, line, col, "NAME005", f"import '{bound_name}' starts with '_'"))


def check_module_name(path: Path) -> Violation | None:
    stem = path.stem
    if stem.startswith("_") and not is_dunder(stem):
        return Violation(str(path), 1, 1, "NAME001", f"module file name '{path.name}' must not start with '_'")
    return None


def check_file(path: Path, allow_bare_underscore: bool) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = NameVisitor(str(path), allow_bare_underscore)
    visitor.visit(tree)
    return visitor.violations


def parse_diff(diff_output: str) -> dict[str, tuple[bool, set[int]]]:
    """Map each touched path to (is newly added, set of added line numbers in the new file)."""
    ranges: dict[str, tuple[bool, set[int]]] = {}
    current_path: str | None = None
    is_new_file = False
    added_lines: set[int] = set()
    new_lineno = 0
    for line in diff_output.splitlines():
        if line.startswith("diff --git "):
            if current_path is not None:
                ranges[current_path] = (is_new_file, added_lines)
            current_path, is_new_file, added_lines, new_lineno = None, False, set(), 0
        elif line.startswith("--- "):
            if line.strip() == "--- /dev/null":
                is_new_file = True
        elif line.startswith("+++ "):
            current_path = line[4:].strip().removeprefix("b/")
        elif line.startswith("@@"):
            match = HUNK_HEADER.match(line)
            if match:
                new_lineno = int(match.group(1))
        elif current_path is not None:
            if line.startswith("+"):
                added_lines.add(new_lineno)
                new_lineno += 1
            elif not line.startswith("-"):
                new_lineno += 1
    if current_path is not None:
        ranges[current_path] = (is_new_file, added_lines)
    return ranges


def collect_diff_ranges(file_args: list[str], ref: str | None) -> dict[str, tuple[bool, set[int]]]:
    diff_target = [ref] if ref else ["--cached"]
    command = ["git", "diff", *diff_target, "-U0", "--", *file_args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"warning: git diff failed: {result.stderr.strip()}", file=sys.stderr)
        return {}
    return parse_diff(result.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--diff",
        nargs="?",
        const="",
        default=None,
        metavar="REF",
        help="diff mode (default): only lines added relative to REF, or the git index when omitted",
    )
    parser.add_argument("--all", action="store_true", help="audit whole files instead of only added lines")
    parser.add_argument("--allow-bare-underscore", action="store_true", help="do not flag a bare '_' binding")
    parser.add_argument("files", nargs="*", help="files to check (non-.py arguments are skipped)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all and args.diff is not None:
        parser.error("--all and --diff are mutually exclusive")

    file_args = [name for name in args.files if name.endswith(".py")]
    if not file_args:
        return 0

    diff_ranges: dict[str, tuple[bool, set[int]]] = {}
    if not args.all:
        diff_ranges = collect_diff_ranges(file_args, args.diff or None)

    violations: list[Violation] = []
    exit_code = 0
    for file_arg in file_args:
        path = Path(file_arg)
        if not path.is_file():
            continue
        is_new_file, added_lines = diff_ranges.get(file_arg, (False, set()))
        module_violation = check_module_name(path)
        if module_violation is not None and (args.all or is_new_file):
            violations.append(module_violation)
        try:
            file_violations = check_file(path, args.allow_bare_underscore)
        except SyntaxError as exc:
            print(f"{file_arg}: error: cannot parse: {exc.msg}", file=sys.stderr)
            exit_code = 1
            continue
        if args.all:
            violations.extend(file_violations)
        else:
            violations.extend(violation for violation in file_violations if violation.line in added_lines)

    for violation in sorted(violations, key=lambda item: (item.path, item.line, item.col)):
        print(f"{violation.path}:{violation.line}:{violation.col}: {violation.code} {violation.message}")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
