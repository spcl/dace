# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Static (compile-time) perf analysis of the DaCe CPU pipelines. No timing anywhere.

Two metrics over two kernel groups, see ``plan.md``:

* METRIC 1 -- parallel loops/regions per arm: gcc autopar and Polly on the serialized
  simplify-only code, versus the regions DaCe emits itself for simplify / canonicalize /
  auto_optimize.
* METRIC 2 -- vectorization capability with the cost model neutralized: canonicalize + the new
  CPU code generator versus simplify + the legacy one, each built with gcc and with clang.

Run ``python run_analysis.py`` (optionally ``--groups A``) to (re)generate ``report.md``. Each
kernel is analyzed in its own serialized worker process under a systemd scope; per-kernel JSON
lands in the build root, so a re-run resumes.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(OUT_DIR))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: group id -> (title, corpus suites of ``tests/corpus/corpus_suite.py``).
GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    'A': ('npbench + polybench', ('poly', 'np')),
    'B': ('tsvc + tsvc_2_5', ('tsvc', 'tsvc25')),
}

#: Build root for generated program folders, objects and per-kernel results. Repo-relative paths
#: and $HOME/.cache only.
BUILD_ROOT = os.environ.get('PERF_ANALYSIS_BUILD_ROOT',
                            os.path.join(os.path.expanduser('~'), '.cache', 'dace-build-perfanalysis'))
RESULTS_DIR = os.path.join(BUILD_ROOT, 'results')
WORK_DIR = os.path.join(BUILD_ROOT, 'work')

#: Bumped when a counting rule changes, so stale result files are re-measured instead of aggregated.
SCHEMA = 4

#: Env every python invocation carries (MPI/UCX off, deterministic hashing, isolated build folder).
ENV_PINS = {
    'PYTHONHASHSEED': '0',
    'MPI4PY_RC_INITIALIZE': '0',
    'OMPI_MCA_pml': 'ob1',
    'OMPI_MCA_btl': 'self,vader',
    'UCX_VFS_ENABLE': 'n',
    'DACE_cache': 'unique',
    'DACE_default_build_folder': BUILD_ROOT,
    'PYTHONPATH': REPO_ROOT,
    # The perf corpus test module is imported for its pipelines; point its result dir away from the
    # checked-in ``perf_results/`` so importing it can never touch a six-arm measurement.
    'CANON_PERF_DIR': os.path.join(BUILD_ROOT, 'unused_canon_perf_dir'),
}

CXX_GCC = os.environ.get('PERF_ANALYSIS_GCC', 'g++')
CXX_CLANG = os.environ.get('PERF_ANALYSIS_CLANG', 'clang++')

#: The timing harness' shared base flag string, plus ``-fopenmp`` (the emitted code has omp pragmas)
#: and the DaCe runtime headers. Every arm below adds only its own flags on top.
BASE_FLAGS = ('-std=c++20 -fPIC -Wall -Wextra -O3 -march=native -fno-math-errno -fno-trapping-math '
              '-fno-signed-zeros -ffp-contract=fast -fopenmp')
RUNTIME_INCLUDE = os.path.join(REPO_ROOT, 'dace', 'runtime', 'include')

GCC_AUTOPAR_FLAGS = os.environ.get('PERF_ANALYSIS_GCC_AUTOPAR',
                                   '-ftree-parallelize-loops=8 -floop-parallelize-all -fgraphite-identity')
LLVM_AUTOPAR_FLAGS = os.environ.get(
    'PERF_ANALYSIS_LLVM_AUTOPAR', '-mllvm -polly -mllvm -polly-parallel -mllvm -polly-omp-backend=LLVM '
    '-mllvm -polly-process-unprofitable -mllvm -polly-parallel-force')

#: Cost model neutralization. gcc has a direct switch; clang has none, so the global stand-in for the
#: per-loop ``#pragma clang loop vectorize(enable)`` is a forced vectorization factor, which sets
#: ``UserVF`` and bypasses the cost model's profitability decision.
GCC_VEC_FLAGS = '-fvect-cost-model=unlimited'
CLANG_VEC_FLAGS = os.environ.get('PERF_ANALYSIS_CLANG_VEC', '-mllvm -force-vector-width=4')
#: Also build the clang configs with the stock cost model, to report how much the forcing added and
#: to prove that forcing a width never LOSES a loop clang would otherwise vectorize.
CLANG_STOCK_CHECK = os.environ.get('PERF_ANALYSIS_CLANG_STOCK', '1') not in ('0', 'false', 'False')

COMPILE_TIMEOUT = int(os.environ.get('PERF_ANALYSIS_COMPILE_TIMEOUT', '600'))
KERNEL_TIMEOUT = int(os.environ.get('PERF_ANALYSIS_KERNEL_TIMEOUT', '1800'))
MEMORY_MAX = os.environ.get('PERF_ANALYSIS_MEMORY_MAX', '5G')

#: Arms of metric 1, in report column order: (result key, label).
METRIC1_ARMS = (
    ('gcc_autopar', '(a) gcc autopar'),
    ('llvm_autopar', '(b) llvm autopar (Polly)'),
    ('simplify_new', '(c) dace-simplify'),
    ('canon', '(d) dace-canon'),
    ('autoopt', '(e) dace-autoopt'),
)
#: Configs of metric 2: (config id, variant, label).
METRIC2_CONFIGS = (
    ('canon_new', 'canon', '(i) canon + new codegen'),
    ('simplify_old', 'simplify_old', '(ii) simplify + old codegen'),
)

# ---------------------------------------------------------------------------
# Worker: one kernel, all variants, all counts
# ---------------------------------------------------------------------------


def pipelines() -> tuple[tuple[str, Callable, str], ...]:
    """``(variant, SDFG pipeline, cpu codegen implementation)`` triples, imported from the timing
    harness so the arms are constructed exactly once in the tree. Importing is deliberate: the
    module's pytest entry points own ``perf_results/`` and are never invoked from here."""
    import dace
    from dace.transformation.auto.auto_optimize import auto_optimize
    import tests.passes.canonicalize.canonicalize_perf_corpus_test as harness

    def autoopt_pure(sdfg):
        # ``find_fast_library_fn`` puts ``pure`` at the head of the implementation priority list, so
        # auto_optimize's own ``set_fast_implementations`` stops picking OpenBLAS. Both metrics count
        # LOOPS: a vendor call has none, and lowering one side to it would delete what is counted.
        return auto_optimize(sdfg, dace.DeviceType.CPU, find_fast_library_fn=lambda device: ['pure'])

    return (
        ('serialize', harness.serialize, 'experimental_readable'),
        ('simplify_new', harness.untransformed, 'experimental_readable'),
        ('simplify_old', harness.untransformed, 'legacy'),
        ('canon', harness._canon, 'experimental_readable'),
        ('autoopt', autoopt_pure, 'legacy'),
    )


def kernel_context(suite: str, name: str) -> dict:
    """Minimal ``corpus_suite`` context: the kernel handle only.

    ``corpus_suite.make`` also allocates datasets and runs a numpy or scalar-python oracle, none of
    which a compile-time count needs -- and the tsvc oracles are interpreter loops.
    """
    from tests.corpus import corpus_suite as suite_mod
    from tests.corpus.npbench import npbench
    from tests.corpus.polybench import polybench
    from tests.corpus.tsvc import tsvc

    if suite == 'poly':
        return {'suite': suite, 'name': name, 'k': polybench.collect(name)[0]}
    if suite == 'np':
        return {'suite': suite, 'name': name, 'c': npbench.collect(name)[0]}
    if suite == 'tsvc':
        return {'suite': suite, 'name': name, 'k': tsvc.collect(name=name)[0]}
    return {'suite': suite, 'name': name, 'k': suite_mod.t25_program(name)}


def force_pure_library_nodes(sdfg) -> None:
    """Point every remaining library node at its ``pure`` expansion; see ``autoopt_pure``.

    A node WITHOUT a pure expansion (Cholesky, for one) is left alone rather than forced: forcing it
    raises at codegen, and since the same rule applies to every variant, the configs still compare
    the same lowering.
    """
    from dace.sdfg import nodes
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.LibraryNode) and 'pure' in node.implementations:
            node.implementation = 'pure'


def strip_all_parallelism(sdfg) -> None:
    """Expand every library node and force every map sequential, to a FIXED POINT.

    One pass is not enough and the difference is the whole llvm/gcc autopar arm: an expansion can
    introduce a fresh nested library node, and re-pointing a node at its ``pure`` expansion after
    ``serialize`` ran puts a DEFAULT-scheduled (i.e. CPU_Multicore) map back into the graph, which
    codegen then emits as ``#pragma omp parallel for``. That leaves DaCe parallelism inside the
    input the external auto-parallelizers are supposed to be measured on -- measured on polybench
    covariance, which came out with 2 residual pragmas. ``metric1['serialize_residual']`` records
    the leftovers so the report can show the arm really was handed sequential code.
    """
    import dace
    from dace.sdfg import nodes
    for _ in range(4):
        sdfg.expand_library_nodes()
        for node, _parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                node.map.schedule = dace.ScheduleType.Sequential
        if not any(isinstance(n, nodes.LibraryNode) for n, _parent in sdfg.all_nodes_recursive()):
            return


def strip_omp_pragmas(src: str) -> int:
    """Delete every ``#pragma omp`` line from ``src``; return how many were deleted.

    The last step of handing the external auto-parallelizers genuinely sequential code, and it is
    needed even after ``strip_all_parallelism``: codegen expands COPY library nodes internally,
    AFTER any pipeline can reach them, and their default-scheduled maps come out as
    ``#pragma omp parallel for`` (measured on polybench covariance: 2 of them). Deleting the
    directive line leaves the loop and its body untouched, i.e. the sequential form of the same
    program, which is exactly what arms (a) and (b) are defined on.
    """
    with open(src, errors='replace') as handle:
        lines = handle.readlines()
    kept = [line for line in lines if not line.lstrip().startswith('#pragma omp')]
    if len(kept) != len(lines):
        with open(src, 'w') as handle:
            handle.writelines(kept)
    return len(lines) - len(kept)


def stage_library_headers(folder: str) -> None:
    """Put the library environment headers where the generated code expects them.

    Emitted code includes them as ``../include/<h>`` relative to ``src/cpu``; DaCe's CMake build
    stages them there, and this analysis compiles the sources directly, so it must do the same.
    A library node that has no ``pure`` expansion keeps its environment header and would otherwise
    fail to compile in EVERY arm at once.
    """
    staged = os.path.join(folder, 'src', 'include')
    os.makedirs(staged, exist_ok=True)
    libraries = os.path.join(REPO_ROOT, 'dace', 'libraries')
    for library in sorted(os.listdir(libraries)):
        include = os.path.join(libraries, library, 'include')
        if not os.path.isdir(include):
            continue
        for header in sorted(os.listdir(include)):
            if header.endswith('.h'):
                shutil.copyfile(os.path.join(include, header), os.path.join(staged, header))


def emit_variant(ctx: dict, variant: str, transform: Callable, codegen: str) -> str:
    """Run one pipeline and write its program folder; return the generated ``.cpp`` path."""
    from dace.codegen.compiler import generate_program_folder
    from tests.corpus import corpus_suite as suite_mod

    os.environ['DACE_compiler_cpu_implementation'] = codegen  # noqa: SIM112 -- DaCe env keys are mixed case
    sdfg = suite_mod.build(ctx, transform, variant)
    force_pure_library_nodes(sdfg)
    if variant == 'serialize':
        strip_all_parallelism(sdfg)
    folder = os.path.join(WORK_DIR, f"{ctx['suite']}__{ctx['name']}", variant)
    shutil.rmtree(folder, ignore_errors=True)
    generate_program_folder(sdfg, sdfg.generate_code(), folder)
    stage_library_headers(folder)
    sources = []
    for root, _, files in os.walk(os.path.join(folder, 'src')):
        sources += [os.path.join(root, f) for f in files if f.endswith('.cpp')]
    if len(sources) != 1:
        raise RuntimeError(f'expected one generated cpu source, got {sorted(sources)}')
    return sources[0]


def run_compile(cxx: str, src: str, extra: str, obj: str | None, info: Sequence[str]) -> tuple[int, str]:
    """Compile ``src`` to an object (or discard it); return ``(returncode, diagnostics)``.

    ``info`` are ``-fopt-info-*`` requests written to files, whose contents are appended to the
    returned text -- gcc mixes opt-info into stderr alongside ``-Wall`` output otherwise.
    """
    folder = os.path.dirname(os.path.dirname(os.path.dirname(src)))
    out = obj or os.path.join(os.path.dirname(src), 'discard.o')
    info_files = [os.path.join(os.path.dirname(src), f'{kind}.txt') for kind in info]
    cmd = [cxx, *shlex.split(BASE_FLAGS), f'-I{RUNTIME_INCLUDE}', f'-I{os.path.join(folder, "include")}']
    cmd += shlex.split(extra)
    cmd += [f'-fopt-info-{kind}={path}' for kind, path in zip(info, info_files, strict=True)]
    cmd += ['-c', src, '-o', out]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return 124, 'compile timeout'
    text = proc.stdout + proc.stderr
    for path in info_files:
        if os.path.exists(path):
            with open(path, errors='replace') as handle:
                text += handle.read()
    return proc.returncode, text


def count_undefined(obj: str, symbol: str) -> int:
    """Call sites of ``symbol`` in ``obj``, counted from its relocations.

    One versioned ``if (cond) parallel else sequential`` construct is one runtime fork call, which is
    the counting rule the report states.
    """
    proc = subprocess.run(['objdump', '-dr', obj], capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
    return sum(1 for line in proc.stdout.splitlines() if symbol in line)


LOCATION_RE = re.compile(r'^(?P<path>[^\s:]+):(?P<line>\d+):(?P<col>\d+):')


def distinct_locations(text: str, marker: str, src: str) -> int:
    """Distinct ``file:line:col`` sites reporting ``marker``, restricted to USER kernel code.

    Both halves are load-bearing. gcc prints one line per vectorized COPY of a loop (a 64-byte main
    body and a 32-byte epilogue are two lines for one source loop), so raw line counts would inflate
    gcc against clang by roughly 2x. The ``src`` filter drops every diagnostic that belongs to
    ``dace/runtime/include`` or a system header, leaving only the generated kernel.
    """
    seen: dict[str, None] = {}
    target = os.path.abspath(src)
    for line in text.splitlines():
        if marker not in line:
            continue
        hit = LOCATION_RE.match(line.strip())
        if hit is None or os.path.abspath(hit.group('path')) != target:
            continue
        seen.setdefault(f"{hit.group('line')}:{hit.group('col')}", None)
    return len(seen)


def analyze_kernel(suite: str, name: str) -> dict:
    """Every count for one kernel. Failures are recorded per stage, never swallowed."""
    result: dict = {
        'schema': SCHEMA,
        'suite': suite,
        'name': name,
        'variants': {},
        'metric1': {},
        'metric2': {},
        'errors': {}
    }
    ctx = kernel_context(suite, name)
    sources: dict[str, str] = {}
    for variant, transform, codegen in pipelines():
        started = time.time()
        try:
            sources[variant] = emit_variant(ctx, variant, transform, codegen)
            result['variants'][variant] = round(time.time() - started, 2)
        except Exception as exc:  # a pipeline that cannot emit code excludes the kernel from its arms
            result['errors'][variant] = f'{type(exc).__name__}: {exc}'[:400]

    for variant in ('simplify_new', 'canon', 'autoopt'):
        if variant not in sources:
            continue
        with open(sources[variant], errors='replace') as handle:
            result['metric1'][variant] = handle.read().count('#pragma omp parallel')

    if 'serialize' in sources:
        src = sources['serialize']
        result['metric1']['serialize_stripped'] = strip_omp_pragmas(src)
        with open(src, errors='replace') as handle:
            result['metric1']['serialize_residual'] = handle.read().count('#pragma omp parallel')
        obj = os.path.join(os.path.dirname(src), 'gcc_autopar.o')
        code, text = run_compile(CXX_GCC, src, GCC_AUTOPAR_FLAGS, obj, ('loop-optimized', ))
        if code != 0:
            result['errors']['gcc_autopar'] = text.strip()[-400:]
        else:
            result['metric1']['gcc_autopar'] = len(re.findall(r'parallelizing (?:outer|inner) loop', text))
            result['metric1']['gcc_autopar_forks'] = count_undefined(obj, 'GOMP_parallel')
        obj = os.path.join(os.path.dirname(src), 'llvm_autopar.o')
        code, text = run_compile(CXX_CLANG, src, LLVM_AUTOPAR_FLAGS, obj, ())
        if code != 0:
            result['errors']['llvm_autopar'] = text.strip()[-400:]
        else:
            result['metric1']['llvm_autopar'] = count_undefined(obj, '__kmpc_fork_call')

    for config, variant, _ in METRIC2_CONFIGS:
        if variant not in sources:
            continue
        src = sources[variant]
        # ONE opt-info request: gcc warns and drops the second when two name different files, so
        # ``-all`` (optimized + missed + note) is the only way to get both halves from one compile.
        code, text = run_compile(CXX_GCC, src, GCC_VEC_FLAGS, None, ('vec-all', ))
        if code != 0:
            result['errors'][f'{config}_gcc'] = text.strip()[-400:]
        else:
            result['metric2'][f'{config}_gcc'] = distinct_locations(text, 'loop vectorized', src)
            result['metric2'][f'{config}_gcc_missed'] = distinct_locations(text, 'missed:', src)
        code, text = run_compile(CXX_CLANG, src, f'{CLANG_VEC_FLAGS} -Rpass=loop-vectorize', None, ())
        if code != 0:
            result['errors'][f'{config}_clang'] = text.strip()[-400:]
        else:
            result['metric2'][f'{config}_clang'] = distinct_locations(text, 'vectorized loop', src)
        if CLANG_STOCK_CHECK:
            code, text = run_compile(CXX_CLANG, src, '-Rpass=loop-vectorize', None, ())
            if code == 0:
                result['metric2'][f'{config}_clang_stock'] = distinct_locations(text, 'vectorized loop', src)
    return result


# ---------------------------------------------------------------------------
# Parent: serialized sweep + report
# ---------------------------------------------------------------------------


def result_path(suite: str, name: str) -> str:
    return os.path.join(RESULTS_DIR, f'{suite}__{name}.json')


def worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(ENV_PINS)
    return env


def run_worker(suite: str, name: str) -> tuple[bool, str]:
    """One kernel in its own memory-capped scope. Serialized by the caller: builds are heavy."""
    cmd = [
        'systemd-run', '--user', '--scope', '--quiet', '-p', f'MemoryMax={MEMORY_MAX}', '--', sys.executable,
        os.path.abspath(__file__), '--worker', f'{suite}:{name}'
    ]
    try:
        proc = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=KERNEL_TIMEOUT,
                              env=worker_env(),
                              cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        return False, f'worker timeout after {KERNEL_TIMEOUT}s'
    if proc.returncode != 0:
        return False, (proc.stdout + proc.stderr).strip()[-400:]
    return True, ''


def sweep(group_ids: Sequence[str], force: bool) -> dict[str, str]:
    """Run every kernel of every requested group; return ``kernel -> worker failure``."""
    from tests.corpus import corpus_suite as suite_mod
    all_kernels = suite_mod.kernels()
    crashed: dict[str, str] = {}
    for group in group_ids:
        title, suites = GROUPS[group]
        todo = [(s, n) for s, n in all_kernels if s in suites]
        print(f'== group {group} ({title}): {len(todo)} kernels', flush=True)
        print(subprocess.run(['free', '-g'], capture_output=True, text=True).stdout, flush=True)
        for index, (suite, name) in enumerate(todo, 1):
            path = result_path(suite, name)
            if not force and os.path.exists(path):
                with open(path) as handle:
                    if json.load(handle).get('schema') == SCHEMA:
                        continue
            started = time.time()
            ok, err = run_worker(suite, name)
            if not ok:
                crashed[f'{suite}:{name}'] = err
            print(
                f'[{group} {index}/{len(todo)}] {suite}:{name} {"ok" if ok else "FAILED"} '
                f'{time.time() - started:.1f}s',
                flush=True)
    return crashed


def load_results(suites: Sequence[str]) -> list[dict]:
    out = []
    for entry in sorted(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else []:
        if not entry.endswith('.json'):
            continue
        with open(os.path.join(RESULTS_DIR, entry)) as handle:
            record = json.load(handle)
        if record.get('schema') == SCHEMA and record['suite'] in suites:
            out.append(record)
    return out


def column(records: Sequence[dict], metric: str, key: str) -> tuple[int, int, int]:
    """``(kernels counted, total, kernels with at least one)`` for one metric column."""
    values = [r[metric][key] for r in records if key in r[metric]]
    return len(values), sum(values), sum(1 for v in values if v > 0)


def metric1_table(records: Sequence[dict]) -> str:
    rows = ['| arm | kernels | parallel loops/regions | kernels with >=1 | mean |', '|---|---|---|---|---|']
    for key, label in METRIC1_ARMS:
        count, total, nonzero = column(records, 'metric1', key)
        mean = f'{total / count:.2f}' if count else '-'
        rows.append(f'| {label} | {count} | {total} | {nonzero} | {mean} |')
    return '\n'.join(rows)


def metric2_table(records: Sequence[dict]) -> str:
    rows = [
        '| config | compiler | kernels | vectorized loops | kernels with >=1 | mean | gcc missed |',
        '|---|---|---|---|---|---|---|'
    ]
    for config, _, label in METRIC2_CONFIGS:
        for compiler in ('gcc', 'clang'):
            count, total, nonzero = column(records, 'metric2', f'{config}_{compiler}')
            mean = f'{total / count:.2f}' if count else '-'
            missed = column(records, 'metric2', f'{config}_gcc_missed')[1] if compiler == 'gcc' else ''
            rows.append(f'| {label} | {compiler} | {count} | {total} | {nonzero} | {mean} | {missed} |')
    return '\n'.join(rows)


def exclusions(records: Sequence[dict], crashed: dict[str, str]) -> list[str]:
    """One line per kernel that could not be counted somewhere, with the stage that failed."""
    lines = []
    for record in records:
        if record['errors']:
            stages = ', '.join(sorted(record['errors']))
            lines.append(f"`{record['suite']}:{record['name']}` -- {stages}")
    for kernel, err in sorted(crashed.items()):
        lines.append(f'`{kernel}` -- worker failed: {err.splitlines()[-1][:120] if err else "?"}')
    return lines


def evidence(records: Sequence[dict]) -> list[str]:
    """Integrity lines: the counters that must agree, and the input that must be sequential."""
    gained, lost = 0, 0
    for record in records:
        for config, _, _ in METRIC2_CONFIGS:
            forced = record['metric2'].get(f'{config}_clang')
            stock = record['metric2'].get(f'{config}_clang_stock')
            if forced is None or stock is None:
                continue
            gained += max(0, forced - stock)
            lost += max(0, stock - forced)
    disagree = sum(1 for r in records if r['metric1'].get('gcc_autopar', 0) != r['metric1'].get('gcc_autopar_forks', 0))
    residual = sum(1 for r in records if r['metric1'].get('serialize_residual', 0) > 0)
    return [
        f'clang forced-VF vs stock cost model: +{gained} loops gained, -{lost} lost',
        f'gcc opt-info vs GOMP_parallel call sites disagree on {disagree}/{len(records)} kernels',
        f'{residual}/{len(records)} kernels had residual DaCe pragmas in the autopar input (must be 0)'
    ]


def takeaways(per_group: Sequence[tuple[str, Sequence[dict]]]) -> list[str]:
    """At most five data-derived bullets: no prose is written by hand into the report."""
    out: list[str] = []
    for group, records in per_group:
        totals = {key: column(records, 'metric1', key)[1] for key, _ in METRIC1_ARMS}
        best = max(METRIC1_ARMS, key=lambda arm: totals[arm[0]])
        out.append(f'Group {group} metric 1: {best[1]} leads with {totals[best[0]]} parallel regions; external '
                   f'autopar reaches {totals["gcc_autopar"]} (gcc) / {totals["llvm_autopar"]} (llvm) against '
                   f'DaCe\'s {totals["simplify_new"]} (simplify), {totals["canon"]} (canon), '
                   f'{totals["autoopt"]} (autoopt).')
    for group, records in per_group:
        cells = {
            f'{c}_{cc}': column(records, 'metric2', f'{c}_{cc}')[1]
            for c, _, _ in METRIC2_CONFIGS
            for cc in ('gcc', 'clang')
        }
        out.append(f'Group {group} metric 2: canon+new codegen vectorizes {cells["canon_new_gcc"]} loops (gcc) / '
                   f'{cells["canon_new_clang"]} (clang) versus simplify+old codegen '
                   f'{cells["simplify_old_gcc"]} / {cells["simplify_old_clang"]}.')
    zero = sum(1 for _, records in per_group for r in records if r['metric1'].get('canon', -1) == 0)
    out.append(f'{zero} kernels leave canonicalize with no parallel region at all -- the remaining '
               'parallelization gap, and the shortlist worth reading kernel by kernel.')
    return out[:5]


def write_report(group_ids: Sequence[str], crashed: dict[str, str]) -> str:
    parts = [
        '# Static perf analysis: parallel regions and vectorization capability', '',
        'Compile-time counts, no timing. Method, flags and counting rules: `plan.md`. '
        'Regenerate: `python dace/perf_analysis/run_analysis.py`.', ''
    ]
    notes: list[str] = []
    per_group: list[tuple[str, list[dict]]] = []
    for group in group_ids:
        title, suites = GROUPS[group]
        records = load_results(suites)
        per_group.append((group, records))
        parts += [
            f'## Group {group} -- {title} ({len(records)} kernels)', '', '### Metric 1 -- parallel loops/regions', '',
            metric1_table(records), '', '### Metric 2 -- vectorized loops, cost model neutralized', '',
            metric2_table(records), ''
        ]
        notes += [f'**Group {group}**: {line}.' for line in evidence(records)]
        excluded = exclusions(records, {k: v for k, v in crashed.items() if k.split(':')[0] in suites})
        notes += [f'**Group {group} exclusions** ({len(excluded)}):'] + ([f'* {line}'
                                                                          for line in excluded] or ['* none'])
    parts += ['## Takeaways', ''] + [f'* {line}' for line in takeaways(per_group)] + ['', '## Notes', '']
    parts += notes + ['']
    text = '\n'.join(parts)
    with open(os.path.join(OUT_DIR, 'report.md'), 'w') as handle:
        handle.write(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help='internal: analyze one kernel, given as suite:name')
    parser.add_argument('--groups', default=','.join(GROUPS), help='comma-separated group ids')
    parser.add_argument('--force', action='store_true', help='re-measure kernels that already have results')
    parser.add_argument('--report-only', action='store_true', help='aggregate existing results only')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    if args.worker:
        suite, name = args.worker.split(':', 1)
        record = analyze_kernel(suite, name)
        with open(result_path(suite, name), 'w') as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
        return 0

    group_ids = [g for g in args.groups.split(',') if g in GROUPS]
    crashed = {} if args.report_only else sweep(group_ids, args.force)
    print(write_report(group_ids, crashed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
