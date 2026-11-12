#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Audit of the Lean development: unreachable declarations, duplicate statements, size metrics.

Runs ``scripts/audit.lean`` once per import group (the library and each executable, whose ``main``
functions cannot be imported together), builds the dependency graph of the user-written declarations and
reports:

* declarations unreachable from the roots: the executables, the theorems checked in
  ``tests/Headlines.lean`` and the definitions the other ``tests/*.lean`` files mention;
* theorems with identical statements;
* theorems whose statement mentions no constant of the library (candidates for ``Core/ListLemmas.lean``
  or for a Mathlib lemma);
* lines, theorems and definitions per module.

Run from ``formal/`` after ``lake build``.  Exits with status 1 if ``--strict`` is given and anything is
unreachable or duplicated.
"""
import argparse
import collections
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Set

FORMAL = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GROUPS = ['Dace', 'Dace,Tools.Interp', 'Dace,Tools.Equiv']
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_'?!]*(?:\.[A-Za-z_][A-Za-z0-9_'?!]*)*")


class Decl:
    """A constant of the library as dumped by ``audit.lean``."""

    def __init__(self, fields: List[str]):
        fields = fields + [''] * (9 - len(fields))
        self.name, self.kind, self.module = fields[0], fields[1], fields[2]
        self.first, self.last = int(fields[3]), int(fields[4])
        self.type_hash, self.own_in_type, self.simp = fields[5], int(fields[6]), fields[7] == 'true'
        self.deps = fields[8].split() if fields[8] else []

    @property
    def lines(self) -> int:
        return self.last - self.first + 1 if self.first else 0


def dump(groups: List[str]) -> Dict[str, Decl]:
    """Runs the Lean dump for every import group and merges the records."""
    template = (FORMAL / 'scripts' / 'audit.lean').read_text()
    decls: Dict[str, Decl] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, group in enumerate(groups):
            imports = '\n'.join(f'import {m}' for m in group.split(','))
            src = pathlib.Path(tmp) / f'audit{i}.lean'
            out = pathlib.Path(tmp) / f'audit{i}.tsv'
            src.write_text(template.replace('import Dace', imports, 1))
            env = dict(os.environ, DACE_AUDIT_OUT=str(out))
            subprocess.run(['lake', 'env', 'lean', str(src)], cwd=FORMAL, env=env, check=True)
            for line in out.read_text().splitlines():
                d = Decl(line.split('\t'))
                decls.setdefault(d.name, d)
    return decls


def owners(decls: Dict[str, Decl]) -> Dict[str, str]:
    """Maps every constant to the user-written declaration it belongs to.

    Constants sharing a source range (constructors, recursors, equation lemmas, instances of ``deriving``)
    belong to the one with the shortest name; constants without a range belong to their closest prefix.
    """
    by_range = collections.defaultdict(list)
    for d in decls.values():
        if d.first:
            by_range[(d.module, d.first, d.last)].append(d.name)
    owner: Dict[str, str] = {}
    for names in by_range.values():
        head = min(names, key=len)
        for n in names:
            owner[n] = head
    for n in decls:
        q = n
        while n not in owner and '.' in q:
            q = q.rsplit('.', 1)[0]
            if q in owner:
                owner[n] = owner[q]
    return owner


def test_roots(decls: Dict[str, Decl]) -> Set[str]:
    """Theorems checked in ``tests/Headlines.lean`` and definitions the other tests mention.

    The other tests only run code, so an identifier in them never keeps a theorem alive.
    """
    roots: Set[str] = set()
    short = collections.defaultdict(set)
    for n, d in decls.items():
        if d.kind != 'thm':
            short[n.rsplit('.', 1)[-1]].add(n)
    for path in sorted((FORMAL / 'tests').glob('*.lean')):
        text = path.read_text()
        if path.name == 'Headlines.lean':
            for n in re.findall(r'^#check @(\S+)', text, re.M):
                if n not in decls:
                    sys.exit(f'tests/Headlines.lean: unknown declaration {n}')
                roots.add(n)
        else:
            for tok in IDENT.findall(text):
                roots |= short.get(tok.rsplit('.', 1)[-1], set())
    return roots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('groups', nargs='*', default=DEFAULT_GROUPS, help='comma-separated import groups')
    ap.add_argument('--strict', action='store_true', help='fail on unreachable or duplicate declarations')
    ap.add_argument('--list', action='store_true', help='print every unreachable declaration')
    ap.add_argument('--out', type=pathlib.Path, help='write the unreachable declarations to this file')
    args = ap.parse_args()

    decls = dump(args.groups)
    owner = owners(decls)
    user = {n: decls[n] for n in set(owner.values())}
    uses: Dict[str, Set[str]] = collections.defaultdict(set)
    for n, d in decls.items():
        o = owner.get(n)
        if o is None:
            continue
        uses[o] |= {owner[x] for x in d.deps if x in owner and owner[x] != o}

    roots = {n for n in user if n == 'main' or n.endswith('.main')} | test_roots(user)
    roots |= {n for n, d in user.items() if d.module.startswith('Tools.')}
    seen, stack = set(roots), list(roots)
    while stack:
        for x in uses[stack.pop()]:
            if x not in seen:
                seen.add(x)
                stack.append(x)
    dead = sorted((d.module, d.first, n) for n, d in user.items()
                  if n not in seen and d.kind in ('thm', 'def') and d.module != 'Dace.AxiomCheck')

    by_hash = collections.defaultdict(list)
    for n, d in user.items():
        if d.kind == 'thm':
            by_hash[d.type_hash].append(n)
    dups = sorted(sorted(v) for v in by_hash.values() if len(v) > 1)
    generic = sorted(n for n, d in user.items() if d.kind == 'thm' and d.own_in_type == 0)

    stats = collections.defaultdict(lambda: [0, 0, 0, 0])
    for d in user.values():
        if d.kind == 'thm':
            stats[d.module][0] += 1
            stats[d.module][1] += d.lines
        elif d.kind == 'def':
            stats[d.module][2] += 1
            stats[d.module][3] += d.lines
    src_lines = {}
    for m in stats:
        p = FORMAL / (m.replace('.', '/') + '.lean')
        src_lines[m] = len(p.read_text().splitlines()) if p.exists() else 0

    print(f'{"module":34s} {"lines":>6s} {"thms":>5s} {"thmL":>6s} {"defs":>5s} {"defL":>6s}')
    for m in sorted(stats, key=lambda m: -src_lines[m]):
        t, tl, f, fl = stats[m]
        print(f'{m:34s} {src_lines[m]:6d} {t:5d} {tl:6d} {f:5d} {fl:6d}')
    tot = [sum(v[i] for v in stats.values()) for i in range(4)]
    print(f'{"total":34s} {sum(src_lines.values()):6d} {tot[0]:5d} {tot[1]:6d} {tot[2]:5d} {tot[3]:6d}')

    dead_lines = sum(user[n].lines for _, _, n in dead)
    print(f'\nunreachable declarations: {len(dead)} ({dead_lines} lines)')
    per_mod = collections.Counter()
    for m, _, n in dead:
        per_mod[m] += user[n].lines
    for m, k in per_mod.most_common():
        print(f'  {m:34s} {k:5d}')
    listing = [f'{m}:{first}\t{n}\t{user[n].kind}\t{user[n].lines}' for m, first, n in dead]
    if args.list:
        print('\n'.join(listing))
    if args.out:
        args.out.write_text('\n'.join(listing) + '\n')
    print(f'\nduplicate statements: {len(dups)}')
    for group in dups:
        print('  ' + ' = '.join(group))
    print(f'\nstatements without library constants: {len(generic)} ({sum(user[n].lines for n in generic)} lines)')
    for n in generic:
        print(f'  {user[n].module}: {n}')
    return 1 if args.strict and (dead or dups) else 0


if __name__ == '__main__':
    sys.exit(main())
