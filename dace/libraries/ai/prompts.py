# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Prompt construction for AI-generated library node expansions.

:data:`SYSTEM_PROMPT` documents the exact contract a DaCe tasklet must satisfy -- what variables
are in scope, where each of the four code blocks is emitted, and which of them are unavailable in
GPU device code. :func:`build_user_prompt` renders the :class:`~dace.libraries.ai.context.
ExpansionContext` collected for one specific library node.
"""

import textwrap
from typing import List

from dace.config import Config
from dace.libraries.ai.context import ConnectorInfo, ExpansionContext, NestingFrame

SYSTEM_PROMPT = """\
You write the body of a DaCe *tasklet*: a small block of code that replaces a library node inside \
a data-centric dataflow graph (an SDFG). DaCe has already decided where this code runs, what data \
reaches it, and how it is compiled. Your job is to write code that is correct *for that specific \
slot*, not generic C++.

# What a tasklet is

Your `code` is pasted **verbatim** into the generated source, inside a `{ ... }` block, between \
two `///////////////////` delimiter comments. Consequences:

- You may declare locals, loops, and lambdas freely.
- You must **not** write `return` -- you are inside a larger function.
- You must **not** re-declare a connector or a symbol; they are already in scope.
- Never emit an empty body. An empty `code` makes DaCe silently drop your `code_global` and \
`state_fields` as well.

# Connectors: the variables in scope

Each connector of the library node is already declared as a C++ variable whose name is **exactly \
the connector name**. There are two shapes, and the context below tells you which is which:

- **Scalar connector** (shown below as `kind: scalar value`). An input is a value already loaded for you \
(`float _a = A[i * N + j];`). An *output* is an **uninitialized local** (`float _c;`) that you \
must assign; DaCe writes it back to memory after your body runs. Assign it on every path.
- **Pointer connector** (shown below as `kind: pointer`). An input is a pointer aimed at the **first element \
of its own memlet subset** -- every enclosing map index in that subset is already folded into the \
offset. An *output* is likewise a pointer aimed at its destination -- write **through** it; there \
is no write-back for pointer outputs, so anything you do not store is lost.
- **Stream connector**. Use `.pop()` to read and `.push(value)` to write.

## Pointer connectors are strided views, not repacked buffers

This is the one thing that is easy to get wrong in a way the compiler cannot catch, so read it \
twice:

- A pointer connector is a **view into its container**. It is never copied or repacked into a \
dense buffer for you.
- Step between rows using the **container's strides**, not the extent of the subset. A 8x8 tile \
of a 64x64 row-major array has a row stride of 64, not 8.
- `elements moved` counts the elements the view covers. It does **not** mean they are contiguous.
- Each connector is offset by **its own** subset. Two connectors of the same tasklet routinely \
have different offsets, and the map indices may appear in a different order in each.

Every **pointer** connector below states `points at`, whether the view is contiguous, and an \
explicit `address element (i0, ...) as` formula. Use that formula rather than deriving indexing \
yourself. A scalar connector carries none of those lines, because there is nothing to address.

# Symbols

Map parameters (`i`, `j`, ...) and SDFG symbols (`N`, `tile`, ...) are plain C++ variables in \
scope. DaCe finds them by scanning your code as text, so any identifier of yours that happens to \
match a symbol name would be turned into a required kernel argument -- list such incidental \
collisions in `ignored_symbols`.

# Where each code block lands

- `code` -- the tasklet body, as described above.
- `code_global` -- file scope of the generated translation unit, before any function. This is \
where `#include` directives, helper functions, `__device__` helpers, and type definitions go. \
System and library headers are always fine here -- put `#include <immintrin.h>` and the like in \
this block, never in the body. Every tasklet's `code_global` is emitted into the **same** \
translation unit, so a name you define can collide with one another generated tasklet defines. \
Guard against it with a name unlikely to clash, or with an include guard \
(`#ifndef MY_HELPER_H` / `#define` / `#endif`) around the definition. `static` does **not** help \
here: it prevents collisions between separate translation units, and there is only one.
- `code_init` -- the body of the program's `__dace_init_*` function, run once at program \
initialization. `__state` is in scope here.
- `code_exit` -- the body of `__dace_exit_*`, run once at finalization, for releasing whatever \
`code_init` acquired.
- `state_fields` -- raw C++ member declarations spliced into the program's state struct (e.g. \
`"fftw_plan plan_0;"`). Access them as `__state->plan_0`. Use these for handles and plans whose \
lifetime must span calls; create them in `code_init` and destroy them in `code_exit`.

`code_init` and `code_exit` always run on the **host**, even when the tasklet body itself is GPU \
device code.

# Hard rules for this slot

The "Capabilities of this slot" section below tells you which of these apply. It is a statement of \
fact about where your code will be compiled and run -- read it first, and treat it as binding.

- **`GPU device code: yes`** -- your body is compiled into a CUDA/HIP kernel that DaCe has \
**already launched**. Every thread of the enclosing thread block executes it, once per iteration \
of the innermost scope in the outline; the block dimensions are given. `threadIdx`, `blockIdx` and \
`blockDim` are in scope and are yours to use -- DaCe has not consumed them for the map. All \
threads of the block reach this body, so `__syncthreads()` and `__shared__` are safe. Do **not** \
launch a kernel, allocate device memory, or call a host API. `__state` and `state_fields` do not \
exist here, and only headers providing `__device__` functions may be included.
- **`GPU device code: no` with GPU-resident operands** -- your body runs on the **host** while its \
data sits in GPU memory. Do the work either by calling a GPU library, or by launching a kernel; \
`code_global` for a host tasklet is compiled by the host compiler named below, so a `__global__` \
kernel and `<<<...>>>` launch syntax will **not** compile there -- prefer a library call. Bind \
whatever you call to `__dace_current_stream` so your work is ordered with the rest of the program, \
and do not synchronize the stream or the device yourself: DaCe inserts the synchronization.
- **`__dace_current_stream in scope: yes`** -- a `cudaStream_t __dace_current_stream` variable is \
declared for you in the body.
- **`dereferenceable by this code: no`** on a connector -- that pointer's memory is not addressable \
from where your code runs (GPU memory from the host, or host memory from device code). You may \
pass it to a library call, but you must not read or write through it. Connectors that do not carry \
this line are perfectly safe to dereference.
- The body executes **once per iteration of the innermost enclosing scope** shown in the nesting \
outline. Do not re-loop over, and do not re-parallelize, an iteration space an enclosing map \
already covers. In particular, a `CPU_Multicore` parent map means your body is already running in \
parallel across threads -- do not add `#pragma omp parallel` of your own.
- Unless a connector reports a **write-conflict resolution**, every element of an output subset \
must be written exactly once, and the output is an overwrite rather than an accumulation. Do not \
assume an output buffer arrives zeroed.

# Side effects

Set `side_effects: true` if your code performs I/O, calls MPI, mutates global state, or otherwise \
must not be optimized away. DaCe cannot detect this for C++ tasklets, and without the flag your \
tasklet may be removed as dead code.

# Environments

If your code needs an external library, DaCe generates the CMake glue for it. There are two ways \
to ask, and they are not interchangeable:

- **`use_environments`** -- a list of class paths of environments that already exist on this \
machine. They are listed in the context under "DaCe environments already available on this \
machine"; copy a path from that list exactly. This is the preferred route: those environments \
already carry working package names, libraries and flags, so naming one cannot get them wrong. \
Note the headers each brings in are its own -- consult it rather than re-declaring its includes.
- **`environments`** -- a full description of a library that is *not* in that list. Only use this \
when nothing listed provides what you need, and only for a library you are confident is installed \
here; a wrong CMake package name turns into a build failure much later, in a place that no longer \
mentions this tasklet.

Prefer a header-only solution over either. Leave both lists empty when the code needs nothing \
beyond the standard library and the compiler's own intrinsics headers.

# How to answer

Return one JSON object matching the provided schema. Put your design reasoning in `notes`, not in \
the code blocks. Leave a block as an empty string (or an empty list) when it is not needed. \
Comment the generated code where the reasoning is not obvious, and prefer clear, correct code \
over maximally clever code.\
"""


def _render_connector(conn: ConnectorInfo) -> str:
    """
    Renders one connector as a block of ``key: value`` lines.

    :param conn: The connector to render.
    :return: The rendered description.
    """
    lines = [f'- {conn.name} ({conn.direction}put)']

    points_at = None
    if conn.base_offset is not None and conn.data:
        points_at = (f'{conn.data} + {conn.base_offset}' if conn.base_offset != '0' else f'{conn.data}')
        points_at += '   (the first element of the memlet subset)'

    layout = None
    if conn.contiguous is not None:
        extent = ' x '.join(conn.view_shape) if conn.view_shape else '?'
        layout = (f'{extent} elements, contiguous'
                  if conn.contiguous else f'{extent} elements, NOT contiguous -- a strided view into "{conn.data}"')

    fields = [
        ('declared as', f'{conn.ctype} {conn.name}'),
        ('kind', 'pointer' if conn.is_pointer else 'scalar value'),
        ('element type', conn.element_type),
        ('container', f'{conn.container_kind} "{conn.data}"' if conn.data else None),
        ('container shape', ' x '.join(conn.shape) if conn.shape else None),
        ('container strides, in elements', ', '.join(conn.strides) if conn.strides else None),
        ('storage',
         f'{conn.storage} (resolved from {conn.storage_declared})' if conn.storage_declared else conn.storage),
        ('memlet subset', conn.subset),
        ('points at', points_at),
        ('view', layout),
        ('address element (i0, ...) as', conn.index_formula),
        ('elements moved', conn.num_elements),
        ('write-conflict resolution', conn.wcr),
        ('dynamic memlet', 'yes' if conn.dynamic else None),
        ('dereferenceable by this code',
         'no -- not addressable from where this code runs' if not conn.dereferenceable else None),
    ]
    lines.extend(f'    {key}: {value}' for key, value in fields if value)
    return '\n'.join(lines)


def _render_nesting(frames: List[NestingFrame]) -> str:
    """
    Renders the nesting chain as an outermost-first indented outline.

    :param frames: The frames returned by :func:`~dace.libraries.ai.context.collect_nesting`,
                   ordered innermost first.
    :return: The rendered outline.
    """
    lines = []
    for depth, frame in enumerate(reversed(frames)):
        indent = '  ' * depth
        annotations = []
        if frame.schedule:
            annotations.append(frame.schedule)
        if frame.unroll:
            annotations.append('unrolled')
        if frame.block_size:
            annotations.append(f'block size {frame.block_size}')
        suffix = f'  [{", ".join(annotations)}]' if annotations else ''
        detail = f'  {frame.detail}' if frame.detail else ''
        lines.append(f'{indent}{frame.kind} "{frame.label}"{detail}{suffix}')
        if frame.symbol_mapping:
            mapping = ', '.join(f'{k} = {v}' for k, v in frame.symbol_mapping.items())
            lines.append(f'{indent}  symbol mapping: {mapping}')
    lines.append('  ' * len(frames) + '<-- your tasklet replaces the library node here')
    return '\n'.join(lines)


def build_user_prompt(ctx: ExpansionContext) -> str:
    """
    Renders an expansion context into the user half of the prompt.

    :param ctx: The context collected for the library node being expanded.
    :return: The rendered prompt.
    """
    caps = ctx.capabilities
    target = ctx.target
    sections: List[str] = []

    task = ['# Task', '', f'Library node type: {ctx.node_type}', f'Node name: {ctx.node_name}']
    if ctx.class_docstring:
        task += [
            '', f'Documentation of the {ctx.node_type} class, which defines the semantics you must '
            'reproduce:', '',
            textwrap.indent(ctx.class_docstring, '    ')
        ]
    if ctx.description:
        task += [
            '', 'What this node must compute (written by the user of this SDFG):', '',
            textwrap.indent(ctx.description.strip(), '    ')
        ]
    if ctx.node_properties:
        task += ['', 'Node properties:']
        task += [f'  {k} = {v}' for k, v in sorted(ctx.node_properties.items())]
    sections.append('\n'.join(task))

    if caps is not None:
        caps_lines = [
            '# Capabilities of this slot',
            '',
            f'GPU device code: {"yes" if caps.device_level else "no"}',
            f'Effective schedule: {caps.effective_schedule}',
            f'__state available in the body: {"yes" if caps.state_available else "no"}',
            f'__dace_current_stream in scope: {"yes" if caps.current_stream_available else "no"}',
            f'Environments allowed: {caps.environments_allowed}',
        ]
        # Only relevant where the generated code could actually touch the GPU; on a pure host slot
        # it is a fact about the machine, and belongs in the target section instead.
        if caps.gpu_backend and (caps.device_level or caps.current_stream_available):
            caps_lines.append(f'GPU backend: {caps.gpu_backend}')
        if caps.block_size:
            caps_lines.append(f'Enclosing thread-block size: {caps.block_size}')
        not_deref = [name for name, ok in caps.dereferenceable.items() if not ok]
        if not_deref:
            space = 'host memory' if caps.device_level else 'GPU memory'
            caps_lines.append(f'Pointers you must NOT dereference here (they are in {space}): '
                              f'{", ".join(sorted(not_deref))}')
        sections.append('\n'.join(caps_lines))

    sections.append('# Connectors (the variables in scope)\n\n' +
                    ('\n'.join(_render_connector(c) for c in ctx.connectors) or '(none)'))

    sections.append('# Where this node sits (outermost first)\n\n' + _render_nesting(ctx.nesting))

    if ctx.symbols:
        sections.append('# Symbols in scope\n\n' + '\n'.join(f'  {k}: {v}' for k, v in sorted(ctx.symbols.items())))

    if target is not None:
        target_lines = ['# Target machine and compiler', '', f'Platform: {target.platform} ({target.machine})']
        optional = [
            ('CPU', target.cpu_model),
            ('CPU ISA extensions', target.cpu_features),
            ('Host compiler',
             f'{target.host_compiler} ({target.compiler_family})' if target.compiler_family else target.host_compiler),
            ('Host compiler flags', target.host_flags),
            ('Build type', target.build_type),
            ('C++ standard', target.cpp_standard),
            ('GPU backend', target.gpu_backend),
            ('GPU architectures to compile for', target.gpu_architectures),
            ('GPU devices detected', target.gpu_names),
            ('GPU compiler flags', target.gpu_flags),
        ]
        target_lines += [f'{label}: {value}' for label, value in optional if value]
        sections.append('\n'.join(target_lines))

    if ctx.available_environments:
        sections.append('# DaCe environments already available on this machine\n\n'
                        'Name any of these in `use_environments`, copying the path verbatim, instead of describing '
                        'the same library again in `environments`.\n\n' +
                        '\n'.join(f'  {e}' for e in ctx.available_environments))

    extra = Config.get('ai', 'extra_instructions')
    if extra:
        sections.append('# Additional instructions from the user\n\n' + extra)

    sections.append('Write the tasklet now.')
    return '\n\n'.join(sections)


def build_repair_prompt(stderr: str, command: str) -> str:
    """
    Builds the follow-up message sent when generated code fails to compile.

    :param stderr: The compiler's diagnostic output.
    :param command: The command line that was used to compile the probe.
    :return: The rendered prompt.
    """
    return ('The generated code does not compile. This is the output of a standalone probe '
            'compilation of your tasklet (the connectors and symbols were declared as function '
            'parameters, and the state struct was rebuilt from your `state_fields`), so line '
            'numbers will not match your snippet exactly.\n\n'
            f'Command:\n\n    {command}\n\n'
            f'Diagnostics:\n\n{textwrap.indent(stderr.strip(), "    ")}\n\n'
            'Fix the problem and return the complete JSON object again, including any parts that '
            'did not change. Do not remove functionality to make it compile.')


def build_feedback_prompt(feedback: str, context: str, standalone: bool = False) -> str:
    """
    Builds the message that opens a new round on an already-generated tasklet.

    The compile-and-repair loop and this are the same mechanism with different critics: one carries
    a compiler's complaint, the other a person's. What differs is that a compile failure is
    self-evidently a defect, while this feedback is about code that already works -- so the message
    has to say that the previous answer was accepted and is being improved, not rejected, or the
    model is liable to start over and lose what was already right.

    :param feedback: The critique: a performance measurement, a missed requirement, a preference.
    :param context: The rendered slot context, unchanged since the previous round or not.
    :param standalone: True when the previous conversation could not be recovered, so the context
                       has to be restated in full rather than referred back to.
    :return: The rendered prompt.
    """
    if standalone:
        return (f'{context}\n\n# Feedback on an earlier version\n\n'
                'You have written code for this slot before. That version is not shown here -- the '
                'conversation could not be recovered -- but this is the feedback on it:\n\n'
                f'{textwrap.indent(feedback.strip(), "    ")}\n\n'
                'Write the tasklet again, addressing that feedback. Return the complete JSON object.')

    return ('The tasklet you wrote above was accepted and used. This is feedback on it:\n\n'
            f'{textwrap.indent(feedback.strip(), "    ")}\n\n'
            'Revise it accordingly. Keep everything that was already correct -- this is an '
            'improvement to working code, not a fresh start, and losing behavior that the feedback '
            'did not ask you to change is a regression. The slot has not moved; the context you '
            'were given still holds. If the feedback asks for something the contract in the system '
            'prompt forbids, say so in `notes` and do the closest thing that is allowed.\n\n'
            'Return the complete JSON object, including any parts that did not change.')
