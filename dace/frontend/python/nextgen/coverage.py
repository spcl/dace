# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Gap-report aggregation for the next-generation frontend.

The parallel schedule-tree lowering check appends one JSON line per parsed
program to the file configured through the ``frontend.stree_report``
configuration entry (environment variable ``DACE_frontend_stree_report``).
This module turns the collected lines into a prioritized gap worklist::

    DACE_frontend_stree_report=/tmp/stree.jsonl pytest tests/...
    python -m dace.frontend.python.nextgen.coverage report /tmp/stree.jsonl

Programs parse multiple times (argument specializations, cache misses), so
aggregation keys on the program name and keeps the record with the most
nextgen callback nodes. Discrepant programs (more nextgen callback nodes
than classic frontend callbacks — see
``DaceProgram._check_callback_discrepancy``) print first, followed by the
most frequent gap categories and unknown-call qualified names.
"""
import argparse
import json
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

#: A category prefix at the start of a reason or of a '; '-joined segment.
_CATEGORY_PREFIX = re.compile(r'(?:^|; )\[([^\]]+)\]')


def reason_categories(reason: str) -> List[str]:
    """All ``[category]`` prefixes of a (possibly merged) callback reason."""
    return _CATEGORY_PREFIX.findall(reason)


def load_report(path: str) -> List[dict]:
    """Parse a gap-report file into records, skipping unparseable lines."""
    records = []
    with open(path, 'r') as report:
        for line in report:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Torn line from an interrupted writer
    return records


def aggregate_by_program(records: List[dict]) -> Dict[str, dict]:
    """One record per program: the specialization with the most nextgen
    callback nodes (ties keep the first seen)."""
    programs: Dict[str, dict] = {}
    for record in records:
        name = record.get('program', '<unknown>')
        current = programs.get(name)
        if current is None or record.get('nextgen_nodes', 0) > current.get('nextgen_nodes', 0):
            programs[name] = record
    return programs


def print_report(records: List[dict], top: int = 20, output=None) -> None:
    """Print the aggregated gap worklist for the given report records."""
    output = output or sys.stdout
    programs = aggregate_by_program(records)
    discrepant = {name: record for name, record in programs.items() if record.get('discrepancy')}

    print(
        f'=== Schedule-tree gap report: {len(records)} record(s), {len(programs)} program(s), '
        f'{len(discrepant)} discrepant ===',
        file=output)

    if discrepant:
        print('\nDiscrepant programs (nextgen callback nodes > classic callbacks):', file=output)
        by_excess = sorted(discrepant.items(),
                           key=lambda item: item[1].get('nextgen_nodes', 0) - item[1].get('classic_callbacks', 0),
                           reverse=True)
        for name, record in by_excess:
            categories = sorted(record.get('category_counts', {}))
            test = record.get('test')
            suffix = f'  (test: {test})' if test else ''
            print(
                f'  {name}: nextgen={record.get("nextgen_nodes", 0)} '
                f'classic={record.get("classic_callbacks", 0)}  [{", ".join(categories)}]{suffix}',
                file=output)

    category_counts: Counter = Counter()
    unknown_calls: Counter = Counter()
    for record in programs.values():
        for category, count in record.get('category_counts', {}).items():
            if category.startswith('unknown-call:'):
                unknown_calls[category[len('unknown-call:'):]] += count
                category = 'unknown-call'
            category_counts[category] += count

    if category_counts:
        print(f'\nTop gap categories (per program, max over specializations):', file=output)
        for category, count in category_counts.most_common(top):
            print(f'  {count:5d}  {category}', file=output)

    if unknown_calls:
        print(f'\nTop unknown-call qualified names (the missing-replacements worklist):', file=output)
        for qualname, count in unknown_calls.most_common(top):
            print(f'  {count:5d}  {qualname}', file=output)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m dace.frontend.python.nextgen.coverage',
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    report_parser = subparsers.add_parser('report', help='Aggregate a gap-report JSONL file into a gap worklist')
    report_parser.add_argument('path', help='Path to the JSONL file written via frontend.stree_report')
    report_parser.add_argument('--top', type=int, default=20, help='How many categories/qualnames to list (default 20)')
    arguments = parser.parse_args(argv)

    print_report(load_report(arguments.path), top=arguments.top)
    return 0


if __name__ == '__main__':
    sys.exit(main())
