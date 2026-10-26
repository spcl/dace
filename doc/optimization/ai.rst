.. _ai:

AI-Generated Implementations
============================

DaCe can ask a language model to write the tasklet that replaces a :ref:`library node <libnodes>`,
e.g. a vectorized microkernel, a sequence of vendor intrinsics, or a call into an external library.
The model is given the node's context in the SDFG: the connectors and the layout of the data behind
them, the enclosing maps and nested SDFGs, whether the code runs on a GPU, and the compiler and
hardware. The answer is compile-checked and becomes an ordinary tasklet in the SDFG, so compiling
the program again does not query the model.

.. note::
    Generated code is compiled and run on your machine. The compile check does not check that it
    is correct. Read the code before trusting a result (``dace.libraries.ai.show`` prints it).

Requesting an implementation
----------------------------

**In a Python program**, with :func:`dace.ai <dace.frontend.python.interface.ai>`:

.. code-block:: python

    N = dace.symbol('N')

    @dace.program
    def saxpy(a: dace.float32, x: dace.float32[N], y: dace.float32[N]):
        return dace.ai('Write _out[i] = _a * _x[i] + _y[i] for every element, using AVX2 FMA '
                       'intrinsics and a scalar remainder loop.',
                       a=a, x=x, y=y)

The description refers to the node's connectors, which are named after the call: a keyword input
``a=A`` becomes ``_a``, positional inputs become ``_in0``, ``_in1``, ..., and the output is
``_out`` (``_out0``, ``_out1``, ... when ``out`` names several containers). Program symbols such as
``N`` are in scope by name. The output is a new transient shaped like the first input, unless
``shape``, ``dtype``, ``storage`` or ``out`` say otherwise. ``name`` names the node (it must be
unique in the program) and ``schedule`` sets its schedule.

**In an SDFG**, with :class:`~dace.libraries.ai.nodes.ai_node.AINode`:

.. code-block:: python

    from dace.libraries.ai import AINode

    node = AINode('gemm_tile', 'Compute one 8x8 output tile of C = A @ B ...',
                  inputs={'_a', '_b'}, outputs={'_c'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[ti:ti+8, 0:N]'))
    ...

**For an existing library node**, through the reserved ``'ai'`` implementation, which every
library node has, including ones defined outside DaCe:

.. code-block:: python

    node.expand(state, 'ai')            # expand now
    node.implementation = 'ai'          # or when the SDFG is compiled

    import dace.libraries.blas as blas
    blas.default_implementation = 'ai'  # every BLAS node

The prompt includes the node's class docstring and properties, so e.g. a
:class:`~dace.libraries.blas.nodes.gemm.Gemm` needs no description. :envvar:`ai.extra_instructions`
is appended to every prompt.

Providers
---------

.. list-table::
    :header-rows: 1
    :widths: 12 30 58

    * - :envvar:`ai.provider`
      - Install
      - Description
    * - ``anthropic`` (default)
      - ``pip install 'dace[agent-anthropic]'``
      - Anthropic Messages API. The key is read from ``ANTHROPIC_API_KEY``, or the SDK's
        credential lookup.
    * - ``openai``
      - ``pip install 'dace[agent-openai]'``
      - OpenAI Responses API. The key is read from ``OPENAI_API_KEY``.
    * - ``manual``
      - --
      - Writes the prompt to a file, to be pasted into a chat interface, and reads the reply from
        the terminal or from a response file.

``pip install 'dace[agent]'`` installs both SDKs; in a DaCe checkout, use ``uv sync --extra agent``.
:envvar:`ai.base_url` sets the API endpoint, e.g. for a gateway or a compatible server.

.. code-block:: bash

    ANTHROPIC_API_KEY=... python my_program.py

    DACE_ai_provider=openai DACE_ai_model=<model> OPENAI_API_KEY=... python my_program.py

    DACE_ai_provider=manual python my_program.py

The ``manual`` provider writes each prompt to ``<build folder root>/ai-prompts`` and reuses a saved
reply while the prompt does not change. To answer the prompts of all nodes at once:

.. code-block:: python

    import dace.libraries.ai as ai

    ai.generate_prompts(sdfg, 'prompts')       # prompts/<node>_<hash>_prompt.md
    # ... save each reply as prompts/<node>_<hash>_response.json ...
    ai.read_prompt_responses(sdfg, 'prompts')  # expands the nodes that have a response

Per node, the same is available as ``node.generate_prompt(state)`` and
``node.read_prompt_response(state, reply)``. Such a reply is compile-checked, and a failure raises
an error.

The prompt
----------

The prompt contains:

* **The node**: its type, class docstring, properties and description.
* **The connectors**: their C++ types, the containers behind them with shape and strides, the
  memlet subset, the offset of the pointer, and how to index it.
* **What the code may do**: whether it runs in a GPU kernel, whether ``__state`` and a GPU stream
  are in scope, and which pointers it may not dereference.
* **The enclosing scopes**: maps, loops, regions and nested SDFGs, with their resolved schedules.
* **The target**: platform, CPU and ISA extensions, GPU architecture, compiler and flags.
* **The DaCe environments** registered on the machine, with their headers and state fields, split
  into those whose installation check passes and those without a check.

When to expand
~~~~~~~~~~~~~~

Schedules, storage and connector types are resolved by DaCe's type inference, which normally runs
during compilation before library nodes are expanded. A node expanded earlier (``node.expand``,
``sdfg.expand_library_nodes``) runs the same inference on a copy of the SDFG, and the resulting
connector types are fixed on the generated tasklet. A later transformation that moves data or the
node (e.g. to the GPU) leaves the tasklet written for the old placement: expand after such
transformations, or regenerate the tasklet with ``refine``.

Compile check and repair
------------------------

DaCe builds a program containing only the generated tasklet, with the same containers, memlets,
symbols and environments, in ``<build folder root>/ai-check``. Compiler errors go back to the model,
up to :envvar:`ai.max_repair_attempts` times. :envvar:`ai.compile_check` turns the check off.

Some errors only appear when several generated tasklets are in one program, e.g. two definitions of
the same helper. :func:`~dace.libraries.ai.diagnose.build` compiles the whole program, attributes
each diagnostic to the tasklet that generated the line, and asks for repairs. ``sdfg.compile()``
itself never repairs.

.. code-block:: python

    from dace.libraries import ai

    compiled = ai.build(sdfg, rounds=2)   # compile, repairing up to twice
    ai.repair(sdfg, error)                # or repair once from a build error

With ``DACE_debugprint`` set, DaCe prints the compiler output instead of including it in the
exception, so ``repair`` cannot use it; ``build`` captures it in either case.

Revising a generated tasklet
----------------------------

A generated tasklet can be revised in a new round of its session, with feedback:

.. code-block:: python

    from dace.libraries import ai

    ai.sessions(sdfg)                   # the generated tasklets: name, session, round, flags
    ai.show(sdfg, 'gemm_tile')          # print the current code
    ai.history(sdfg, 'gemm_tile')       # the rounds so far

    ai.refine(sdfg, 'gemm_tile', 'Too slow: 4.2 ms against 1.1 ms for MKL. Use 8x8 register '
                                 'blocking and unroll the reduction.')

    ai.rollback(sdfg, 'gemm_tile', round=1)   # put the code of an earlier round back
    ai.pin(sdfg, 'gemm_tile')                 # exclude from refine and repair

The tasklet is selected by node name, as an ``AITasklet``, or omitted if there is only one. The
generated tasklet (:class:`~dace.libraries.ai.nodes.ai_tasklet.AITasklet`) stores the library node
it replaced, and the session is on disk, so this also works on an SDFG loaded from a file. If a
round fails, the previous tasklet is kept. Code edited by hand is what the next round revises.

Storage
-------

Everything is stored under the build folder root (:envvar:`default_build_folder`, ``.dacecache``
by default):

* ``ai-cache``: answers, keyed by the provider, endpoint, model, effort, token limit, response
  schema and conversation (:envvar:`ai.cache`).
* ``ai-sessions``: per node, the prompts, answers and compile checks of every round
  (:envvar:`ai.sessions`).
* ``ai-environments``: the DaCe environments requested by the model, as Python modules.
* ``ai-check``: the build folders of the compile checks.
* ``ai-prompts``: the files of the ``manual`` provider.

An expanded SDFG does not depend on any of these, except for generated environments: it compiles
on a machine without an API key.

External libraries
------------------

An answer may reuse a registered DaCe environment or request a new one: headers, CMake packages and
libraries, and initialization and finalization code (e.g. for an FFTW plan or a cuBLAS handle). A
new environment is written as a Python module to ``<build folder root>/ai-environments``, which is
imported again when :mod:`dace.libraries.ai` is loaded. Inside a GPU kernel, only headers with
``__device__`` functions are allowed.

Configuration
-------------

The entries are in the ``ai`` category of the configuration (see :ref:`config`):

* Provider and model: :envvar:`ai.provider`, :envvar:`ai.model`, :envvar:`ai.api_key_envvar`,
  :envvar:`ai.base_url`, :envvar:`ai.effort`, :envvar:`ai.max_tokens`, :envvar:`ai.timeout`
* Prompt: :envvar:`ai.extra_instructions`
* Compile check: :envvar:`ai.compile_check`, :envvar:`ai.max_repair_attempts`
* Storage: :envvar:`ai.cache`, :envvar:`ai.sessions`

Example
-------

`samples/optimization/ai_generated_matmul.py <https://github.com/spcl/dace/blob/main/samples/optimization/ai_generated_matmul.py>`_
is a tiled matrix multiplication with a generated microkernel. It prints the kernel, compares it
with NumPy, and optionally refines it with the measurement as feedback::

    ANTHROPIC_API_KEY=... python samples/optimization/ai_generated_matmul.py 512 --refine
