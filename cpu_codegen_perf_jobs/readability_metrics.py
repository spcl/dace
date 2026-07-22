#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Code-readability metrics over a generated DaCe C++/CUDA source (.cpp / .cu).

``readability_score(path)`` returns one dict per file. The metrics, with honest labels
(what each one is actually good for, measured on the legacy-vs-experimental lanes):

  nloc             lizard's non-comment/non-blank line count.
                   DISCRIMINATES strongly and monotonically -- the headline number.
  max_nesting      maximum brace-nesting depth, counted here (comments and string
                   literals stripped first). DISCRIMINATES best and is SIZE-INDEPENDENT:
                   it measures the shape of the code, not how much of it there is.
  tokens           sum of lizard's per-function token counts.
                   SIZE-SENSITIVE -- it FLIPS SIGN on small kernels, because the
                   experimental lane pays tokens back up front for its constexpr `_idx`
                   index-helper prologue. Never quote it on its own.
  tokens_per_stmt  tokens / number of statements (``;`` count on the stripped source).
                   The normalized form of ``tokens``; report this one alongside.
  max_ccn          max per-function cyclomatic complexity.
                   CONTROL, not a win: both lanes emit the same loop structure, so this
                   is expected to be UNCHANGED and is reported to prove that readability
                   did not come from dropping semantics. (Average CCN is deliberately NOT
                   reported: it drops purely because the experimental lane adds trivial
                   constexpr `_idx` helpers, which is an artifact, not a simplification.)

Optionally, with ``--multimetric`` / ``multimetric=True``, Halstead volume and the
maintainability index are shelled out of the ``multimetric`` package. Both are
SIZE-SENSITIVE in the same way ``tokens`` is, so they are reported only when asked for.

Everything degrades gracefully: without ``lizard`` the lizard-derived metrics come back
as None (``nloc`` falls back to a local non-blank/non-comment line count, flagged in
``nloc_source``), the locally computed metrics still work, and the note field carries the
exact pip command to fix it. Same for ``multimetric``.

    pip install lizard          # for nloc / tokens / max_ccn
    pip install multimetric     # optional, for halstead_volume / maintainability_index

CLI::

    python readability_metrics.py <file>...                  # print a table
    python readability_metrics.py <file>... --multimetric    # + halstead / MI
    python readability_metrics.py <file>... --csv metrics.csv # feed plot_codegen_perf.py
"""
import argparse
import csv
import json
import os
import subprocess
import sys

try:
    import lizard
    LIZARD_IMPORT_ERROR = None
except ImportError as exc:  # degrade gracefully -- the CLI/plots still run without it
    lizard = None
    LIZARD_IMPORT_ERROR = str(exc)

LIZARD_HINT = 'pip install lizard'
MULTIMETRIC_HINT = 'pip install multimetric'

#: Source extensions lizard handles with no configuration (both C++ and CUDA).
SOURCE_EXTENSIONS = ('.cpp', '.cu', '.cxx', '.cc', '.h', '.hpp', '.cuh')

#: The two codegen lanes, as they appear in the TSV / in generated-source paths.
CODEGENS = ('legacy', 'experimental')

#: Path components that carry no kernel identity (dacecache / build-tree scaffolding).
PATH_NOISE = {'src', 'cpu', 'cuda', 'gpu', 'build', 'dacecache', '.dacecache', 'sample', 'include'}

#: Honest one-line label per metric: what it measures and whether it discriminates.
METRIC_LABEL = {
    'nloc': 'lines of code (lizard nloc) -- DISCRIMINATES, monotone [headline]',
    'max_nesting': 'max brace nesting depth -- DISCRIMINATES best, size-independent [headline]',
    'tokens': 'raw token count -- SIZE-SENSITIVE, flips sign on small kernels',
    'tokens_per_stmt': 'tokens per statement -- normalized form of tokens',
    'max_ccn': 'max cyclomatic complexity -- CONTROL, expected unchanged',
    'halstead_volume': 'Halstead volume (multimetric) -- SIZE-SENSITIVE, optional',
    'maintainability_index': 'maintainability index (multimetric) -- SIZE-SENSITIVE, optional',
}

#: Metrics where a LOWER value is the better/more readable one (all but MI).
LOWER_IS_BETTER = ('nloc', 'max_nesting', 'tokens', 'tokens_per_stmt', 'halstead_volume')

#: The metrics that are controls rather than claims (reported to prove semantics held).
CONTROL_METRICS = ('max_ccn', )

#: Metrics that must not be quoted without a normalized companion.
SIZE_SENSITIVE_METRICS = ('tokens', 'halstead_volume', 'maintainability_index')

#: Metric order for the CLI table / CSV.
METRICS = ('nloc', 'max_nesting', 'tokens', 'tokens_per_stmt', 'max_ccn')
OPTIONAL_METRICS = ('halstead_volume', 'maintainability_index')


def strip_comments(src):
    """Return `src` with // and /* */ comments and all string/char literals blanked out.

    Literals collapse to a single space so a ``//`` or ``;`` inside a string cannot be
    mistaken for a comment or a statement. Newlines are preserved, so line structure (and
    hence a line count over the result) stays meaningful. Raw string literals (``R"(...)"``)
    are not special-cased -- DaCe's generated code does not emit them."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ''
        if c == '/' and nxt == '/':
            while i < n and src[i] != '\n':  # keep the newline for the next iteration
                i += 1
        elif c == '/' and nxt == '*':
            i += 2
            while i < n and not (src[i] == '*' and i + 1 < n and src[i + 1] == '/'):
                if src[i] == '\n':
                    out.append('\n')
                i += 1
            i += 2
            out.append(' ')
        elif c in ('"', "'"):
            quote = c
            i += 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                if src[i] == '\n':  # unterminated literal: bail out rather than eat the file
                    break
                i += 1
            out.append(' ')
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def max_brace_depth(stripped):
    """Maximum brace-nesting depth of a comment/literal-stripped source.

    Counted from file scope, so the outermost function body already sits at depth 1; the
    number is a relative measure and is only ever compared between lanes of the same
    kernel. Unbalanced closers clamp at 0 rather than going negative."""
    depth = max_depth = 0
    for ch in stripped:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == '}':
            depth = max(0, depth - 1)
    return max_depth


def fallback_nloc(stripped):
    """Non-blank line count of the stripped source -- the no-lizard stand-in for nloc."""
    return sum(1 for line in stripped.splitlines() if line.strip())


def multimetric_scores(path):
    """(dict, note): Halstead volume / maintainability index via ``python -m multimetric``.

    Returns ({}, note) whenever the package is missing or does not produce the expected
    JSON -- both are optional extras, so nothing here is allowed to raise."""
    cmd = [sys.executable, '-m', 'multimetric', path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f'multimetric unavailable ({exc}); {MULTIMETRIC_HINT}'
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()
        hint = tail[-1] if tail else f'exit {proc.returncode}'
        return {}, f'multimetric failed ({hint}); {MULTIMETRIC_HINT}'
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return {}, 'multimetric produced non-JSON output'
    files = data.get('files', {}) if isinstance(data, dict) else {}
    entry = files.get(path) or files.get(os.path.abspath(path))
    if entry is None and len(files) == 1:
        entry = next(iter(files.values()))
    if not isinstance(entry, dict):
        return {}, 'multimetric JSON carried no entry for this file'
    return {
        'halstead_volume': entry.get('halstead_volume'),
        'maintainability_index': entry.get('maintainability_index'),
    }, None


def readability_score(path, multimetric=False):
    """Readability metrics for one generated .cpp/.cu. Never raises: on any failure the
    metrics come back None and ``error`` / ``notes`` explain why.

    Returns a dict with: path, ok, error, notes (list of str), nloc, nloc_source
    ('lizard' | 'fallback'), max_nesting, tokens, statements, tokens_per_stmt, max_ccn,
    functions, and -- only with multimetric=True -- halstead_volume /
    maintainability_index. See the module docstring for what each metric is worth."""
    result = {
        'path': path,
        'ok': False,
        'error': None,
        'notes': [],
        'nloc': None,
        'nloc_source': None,
        'max_nesting': None,
        'tokens': None,
        'statements': None,
        'tokens_per_stmt': None,
        'max_ccn': None,
        'functions': None,
    }
    if not os.path.isfile(path):
        result['error'] = 'file not found'
        return result
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            src = fh.read()
    except OSError as exc:
        result['error'] = f'unreadable ({exc})'
        return result

    stripped = strip_comments(src)
    result['max_nesting'] = max_brace_depth(stripped)
    statements = stripped.count(';')
    result['statements'] = statements

    if lizard is not None:
        try:
            analysis = lizard.analyze_file(path)
        except Exception as exc:  # lizard raising on a file must not sink the sweep
            result['notes'].append(f'lizard.analyze_file failed ({exc})')
        else:
            functions = list(analysis.function_list)
            result['nloc'] = analysis.nloc
            result['nloc_source'] = 'lizard'
            result['functions'] = len(functions)
            result['tokens'] = sum(f.token_count for f in functions) or None
            ccns = [f.cyclomatic_complexity for f in functions]
            result['max_ccn'] = max(ccns) if ccns else None
    else:
        result['notes'].append(f'lizard not importable ({LIZARD_IMPORT_ERROR}); '
                               f'nloc is a local fallback, tokens/max_ccn unavailable -- {LIZARD_HINT}')

    if result['nloc'] is None:
        result['nloc'] = fallback_nloc(stripped)
        result['nloc_source'] = 'fallback'

    if result['tokens'] and statements:
        result['tokens_per_stmt'] = result['tokens'] / statements

    if multimetric:
        scores, note = multimetric_scores(path)
        result.update(scores)
        if note:
            result['notes'].append(note)

    result['ok'] = True
    return result


def infer_kernel_codegen(path, root=None):
    """(kernel, codegen | None) from a generated-source path, for any of the layouts the
    runners produce: ``<root>/<codegen>/<kernel>.cpp``, ``<root>/<kernel>/<codegen>.cpp``,
    ``<root>/<kernel>_<codegen>.cpp`` and dacecache-style ``<kernel>_<codegen>_<preset>/
    src/cpu/<same>.cpp``. Unknown layouts fall back to the file stem as the kernel."""
    rel = os.path.relpath(path, root) if root else os.path.basename(path)
    parts = [p for p in rel.replace('\\', '/').split('/') if p not in ('', '.', '..')]
    stem = os.path.splitext(parts[-1])[0] if parts else ''
    lowered = [p.lower() for p in parts]

    codegen = None
    for token in CODEGENS:
        if token in lowered or token in stem.lower():
            codegen = token
            break

    stem_clean = stem
    if codegen:
        low = stem_clean.lower()
        for sep in ('_', '-', '.'):
            for pattern in (sep + codegen, codegen + sep):
                idx = low.find(pattern)
                while idx != -1:
                    stem_clean = stem_clean[:idx] + stem_clean[idx + len(pattern):]
                    low = stem_clean.lower()
                    idx = low.find(pattern)
        idx = low.find(codegen)
        if idx != -1:
            stem_clean = stem_clean[:idx] + stem_clean[idx + len(codegen):]
    stem_clean = stem_clean.strip('_-. ')

    if stem_clean:
        return stem_clean, codegen
    parents = [p for p in parts[:-1] if p.lower() not in PATH_NOISE and p.lower() not in CODEGENS]
    if parents:
        return parents[-1], codegen
    return stem or rel, codegen


def find_sources(srcdir):
    """[(path, kernel, codegen)] for every generated source under `srcdir` (recursive)."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(srcdir):
        for fn in sorted(filenames):
            if not fn.lower().endswith(SOURCE_EXTENSIONS):
                continue
            path = os.path.join(dirpath, fn)
            kernel, codegen = infer_kernel_codegen(path, srcdir)
            found.append((path, kernel, codegen))
    return found


def format_metric(val):
    if val is None:
        return '-'
    if isinstance(val, float):
        return f'{val:.2f}'
    return str(val)


def display_name(path):
    """<parent>/<file>, so the two lanes of one kernel stay apart in the table (they share
    a basename in every per-variant layout)."""
    parent = os.path.basename(os.path.dirname(path))
    return os.path.join(parent, os.path.basename(path)) if parent else os.path.basename(path)


def print_table(results, metrics):
    """Small fixed-width table, one row per file."""
    name_w = max([len('file')] + [len(display_name(r['path'])) for r in results])
    cols = [('file', name_w)] + [(m, max(len(m), 8)) for m in metrics]
    header = '  '.join(name.ljust(w) for name, w in cols)
    print(header)
    print('-' * len(header))
    for r in results:
        cells = [display_name(r['path']).ljust(name_w)]
        for m, w in cols[1:]:
            cells.append(format_metric(r.get(m)).rjust(w))
        print('  '.join(cells))
    print()
    print('metric labels:')
    for m in metrics:
        print(f'  {m:<22} {METRIC_LABEL.get(m, "")}')
    notes = {n for r in results for n in r['notes']}
    errors = [(r['path'], r['error']) for r in results if r['error']]
    fallbacks = sum(1 for r in results if r.get('nloc_source') == 'fallback')
    if fallbacks:
        print(f'\nnote: {fallbacks} file(s) used the local nloc fallback (lizard absent)')
    for n in sorted(notes):
        print(f'note: {n}')
    for path, err in errors:
        print(f'error: {path}: {err}')


def write_csv(results, path, metrics, srcdir=None):
    """Precomputed-metrics CSV for plot_codegen_perf.py --metrics-csv."""
    fields = ['kernel', 'codegen', 'path'] + list(metrics)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            kernel, codegen = infer_kernel_codegen(r['path'], srcdir)
            row = {'kernel': kernel, 'codegen': codegen or '', 'path': r['path']}
            row.update({m: ('' if r.get(m) is None else r.get(m)) for m in metrics})
            writer.writerow(row)
    print(f'wrote {path} ({len(results)} row(s))')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='generated .cpp / .cu sources (or directories to walk)')
    ap.add_argument('--multimetric',
                    action='store_true',
                    help='also report Halstead volume / maintainability index (needs: %s)' % MULTIMETRIC_HINT)
    ap.add_argument('--csv', help='write the metrics to this CSV (feeds plot_codegen_perf.py --metrics-csv)')
    ap.add_argument('--srcdir', help='root to make --csv kernel/codegen inference relative to (default: per-file dir)')
    args = ap.parse_args()

    paths = []
    for entry in args.files:
        if os.path.isdir(entry):
            paths.extend(p for p, _k, _c in find_sources(entry))
        else:
            paths.append(entry)
    if not paths:
        print('no source files given (nothing under the directories provided)')
        return

    if lizard is None:
        print(f'note: lizard is not installed -- nloc falls back to a local line count and '
              f'tokens/max_ccn are unavailable. Install with: {LIZARD_HINT}\n')

    metrics = list(METRICS) + (list(OPTIONAL_METRICS) if args.multimetric else [])
    results = [readability_score(p, multimetric=args.multimetric) for p in paths]
    print_table(results, metrics)
    if args.csv:
        write_csv(results, args.csv, metrics, args.srcdir)


if __name__ == '__main__':
    main()
