.. _ai:

AI-Generated Implementations
============================

Some code is not worth writing by hand, and is not available in a library either: a vectorized
microkernel for one specific tile shape, a sequence of vendor intrinsics, a call into a library
DaCe does not wrap. DaCe can ask a language model to write those, as **tasklets**, in place of a
:ref:`library node <libnodes>`.

This is a code generation feature rather than a chat interface. The model is not asked "how would
you multiply matrices"; it is given one slot in one SDFG -- the connectors and the strides of the
data behind them, the maps and nested SDFGs around it, whether the code will be compiled as GPU
device code, and the compiler and CPU it will be built for -- and asked for code that is correct
*there*. What comes back is compiled in isolation before it is accepted, and then becomes an
ordinary tasklet in the SDFG: it is saved with the SDFG, it is transformed like any other node,
and compiling the program again does not query the model again.

.. note::
    Generated code is compiled and run on your machine, and the probe described below only checks
    that it *compiles*, never that it is correct or safe. Read what a model wrote before you trust
    a result -- ``dace.libraries.ai.show`` prints it -- and keep the sessions on disk, which record
    exactly what was asked and answered.

Three ways to ask for one
-------------------------

**In a Python program**, with :func:`dace.ai <dace.frontend.python.interface.ai>`. This is the
common case: the description sits where the code would have been.

.. code-block:: python

    N = dace.symbol('N')

    @dace.program
    def saxpy(a: dace.float32, x: dace.float32[N], y: dace.float32[N]):
        return dace.ai('Write _out[i] = _a * _x[i] + _y[i] for every element, using AVX2 FMA '
                       'intrinsics on aligned loads and a scalar remainder loop.',
                       a=a, x=x, y=y)

Because the description is the specification, it has to be able to name the variables the
generated code will see. Those are the node's connectors, and they come from the call:

* a keyword input ``a=A`` becomes the connector ``_a``,
* a positional input becomes ``_in0``, ``_in1``, ... in the order given,
* the output is ``_out``, or ``_out0``, ``_out1``, ... when ``out`` names several containers.

Symbols of the program (``N`` above) are in scope in the generated code under their own names and
are not passed in. The result is a new transient shaped like the first input unless ``shape``,
``dtype`` and ``storage`` say otherwise, or ``out`` names containers to write into. The remaining
keyword arguments configure the node rather than naming inputs: ``name`` labels it (and is how the
iteration API addresses it later), and ``schedule`` sets its schedule.

**In an SDFG**, with :class:`~dace.libraries.ai.nodes.ai_node.AINode`, which is the same node the
frontend adds:

.. code-block:: python

    from dace.libraries.ai import AINode

    node = AINode('gemm_tile', 'Compute one 8x8 output tile of C = A @ B ...',
                  inputs={'_a', '_b'}, outputs={'_c'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[ti:ti+8, 0:N]'))
    ...

**On a library node that already exists**, through the reserved ``'ai'`` implementation. Every
DaCe library node has it, including nodes defined outside DaCe by a downstream project, and
without anything having been registered:

.. code-block:: python

    node.expand(state, 'ai')          # expand this node, now
    node.implementation = 'ai'        # ... or when the SDFG is compiled

    import dace.libraries.blas as blas
    blas.default_implementation = 'ai'  # every BLAS node in this process

The node's class docstring, its properties and its name are all part of the prompt, so a
:class:`~dace.libraries.blas.nodes.gemm.Gemm` does not need a description to be expanded this way
-- the semantics it must reproduce are already documented. :envvar:`ai.extra_instructions` appends
text to every prompt, which is where project-wide conventions belong.

Choosing a provider
-------------------

The model is reached through a provider, selected with :envvar:`ai.provider`. The SDK is imported
only when an expansion actually runs, so none of them is a dependency of DaCe itself.

.. list-table::
    :header-rows: 1
    :widths: 12 28 60

    * - Provider
      - Install
      - Use
    * - ``anthropic``
      - ``pip install 'dace[ai-anthropic]'``
      - The Anthropic Messages API, and the default. Reads the key from ``ANTHROPIC_API_KEY``,
        falling back to the SDK's own credential chain if that variable is unset.
    * - ``responses``
      - ``pip install 'dace[ai-openai]'``
      - The Responses API of the ``openai`` package. Set :envvar:`ai.api_key_envvar` to
        ``OPENAI_API_KEY`` and :envvar:`ai.model` to a model that package serves.
    * - ``manual``
      - --
      - Neither an SDK nor a key: the prompt is written to a file for you to paste into a chat
        interface, and the reply is read back from the terminal or from a file next to it.

.. code-block:: bash

    # Anthropic (the default provider)
    ANTHROPIC_API_KEY=... python my_program.py

    # OpenAI
    export DACE_ai_provider=responses
    export DACE_ai_api_key_envvar=OPENAI_API_KEY
    export DACE_ai_model=gpt-6-astra
    OPENAI_API_KEY=... python my_program.py

    # No API key: relay the prompt by hand
    DACE_ai_provider=manual python my_program.py

The ``manual`` provider is also the way to *read* a prompt: it writes the full text to
:envvar:`ai.manual_dir` and waits for the model's JSON reply on standard input, or for it to be
saved to the response file it names. A saved reply is reused on later runs, so the same program can
be re-run, and re-compiled, without being asked again.

Which model is used, how much effort it spends, and how long a request may take are set by
:envvar:`ai.model`, :envvar:`ai.effort`, :envvar:`ai.max_tokens` and :envvar:`ai.timeout`.

What the model is told
----------------------

The prompt is built from the node's slot in the SDFG, and is worth understanding, because most
disappointing answers are answers to a question the description left open. It contains:

* **The node**: its type, its class docstring, its properties, and the description if it has one.
* **The connectors**: for each one, the C++ declaration, whether it is a pointer or a scalar
  value, the container behind it with its shape and *strides*, the memlet subset, the address of
  the first element, and an explicit ``address element (i0, ...) as`` formula. A sliced operand is
  a strided view, not a repacked buffer, which is the mistake this section exists to prevent.
* **The capabilities of the slot**: whether the body is compiled into a GPU kernel that DaCe has
  already launched (and with what block size), whether ``__state`` and a GPU stream are in scope,
  and which pointers must not be dereferenced from where the code runs -- host code over GPU
  memory, or the other way around.
* **The nesting**: every map, loop, control flow region and nested SDFG between the node and the
  top-level SDFG, with the resolved schedule of each. A body inside a ``CPU_Multicore`` map is told
  not to open a parallel region of its own.
* **The target**: the platform, the CPU and its ISA extensions, the GPU architecture where one is
  involved, and the compiler with the exact flags DaCe will build with.

The same node in two places therefore gets two different answers, which is the point: a ``Gemm``
inside a GPU kernel, a ``Gemm`` on the host over GPU arrays, and a ``Gemm`` over CPU arrays need
different code, and nothing about the node itself says which is which.

Verification and repair
-----------------------

Generated code is compiled before it is accepted. DaCe synthesizes a small standalone translation
unit -- the connectors and symbols become parameters, the state struct is rebuilt from the state
fields the answer asked for -- and hands it to the compiler it would use for the real build. It is
compiled but never linked or run, so this catches syntax and type errors, not wrong results. A
failure goes back to the model as a repair round; :envvar:`ai.max_repair_attempts` bounds how many
times. Set :envvar:`ai.verify` to false to skip the probe.

The probe compiles each tasklet *alone*, so it cannot see what only breaks when several land in
one translation unit -- two answers defining the same helper in their file-scope code, colliding
state fields, a header the probe could not locate. Those appear at build time, and
:func:`~dace.libraries.ai.diagnose.build` closes that loop: it compiles, attributes each compiler
diagnostic to the tasklet that generated the line, feeds it back, and rebuilds.

.. code-block:: python

    from dace.libraries import ai

    compiled = ai.build(sdfg, rounds=2)   # compile, repairing generated code up to twice
    ai.repair(sdfg, error)                # or feed one build failure back by hand

Attribution is exact rather than heuristic: DaCe annotates every generated line with the node it
came from. A diagnostic that belongs to no generated tasklet is reported as such instead of being
blamed on whichever one was nearby, and an ordinary ``sdfg.compile()`` still fails the way it
always has -- nothing is repaired unless you ask.

Iterating on a generated tasklet
--------------------------------

The first answer is rarely the last: code that compiles and is correct can still be slower than
what it replaced, and running it can surface a requirement nobody stated. What is needed then is
not a fresh generation, which throws away everything the model worked out about this slot, but the
next round of the same conversation.

.. code-block:: python

    from dace.libraries import ai

    ai.sessions(sdfg)                   # every refinable slot: name, session, round, flags
    ai.show(sdfg, 'gemm_tile')          # print what the slot holds right now
    ai.history(sdfg, 'gemm_tile')       # the rounds it has been through, and why

    ai.refine(sdfg, 'gemm_tile', 'Too slow: 4.2 ms against 1.1 ms for MKL. Use 8x8 register '
                                 'blocking and unroll the reduction.')

    ai.rollback(sdfg, 'gemm_tile', round=1)   # put an earlier round back; asks the model nothing
    ai.pin(sdfg, 'gemm_tile')                 # this one is good: leave it alone from now on

The selector (``'gemm_tile'`` above) is the node's name; it can also be an ``AITasklet`` object,
or be left out when the SDFG holds exactly one generated tasklet.

Two things make this work after the fact. The generated tasklet is an
:class:`~dace.libraries.ai.nodes.ai_tasklet.AITasklet`, which carries the serialized library node
it replaced, so the node can be put back and expanded again. And the conversation lives in a
session directory on disk (:envvar:`ai.session_dir`) rather than in the process that started it.
Both survive saving the SDFG and loading it days later, which is when this is most often wanted.
Refinement is atomic: if the new round fails to generate or to verify, the tasklet that was
working a moment ago is put back exactly as it was. If you edited the code by hand, the next round
is told so and revises *your* version rather than silently discarding it.

Caching and reproducibility
---------------------------

Every answer is stored on disk (:envvar:`ai.cache`, :envvar:`ai.cache_dir`), keyed by the provider,
the model, the effort setting and the entire conversation. Re-expanding an unchanged node is then
free, and two identical questions asked from different SDFGs share one entry. Any change to the
node, its context or the model changes the key and asks again. Deleting the cache only costs
requests.

Sessions (:envvar:`ai.sessions`, :envvar:`ai.session_dir`) are the other half: one directory per
slot, holding the prompts, the answers, the probe source and the compiler diagnostics of every
round. An expansion is a paid, non-reproducible call whose result is baked into an SDFG, so this
material is kept by default -- it is what ``history`` and ``rollback`` read, and what explains a
tasklet nobody remembers asking for.

Nothing about a *compiled* program depends on any of this. Once expanded, the SDFG holds the code:
it serializes to ``.sdfg`` files, compiles on a machine with no API key, and can be handed to
someone who does not use this feature at all.

Calling external libraries
--------------------------

An answer may ask for a **DaCe environment**: headers, CMake packages and libraries, and the
initialization and finalization code for a resource whose lifetime spans calls -- an FFTW plan, a
cuBLAS handle. DaCe renders each request into a real Python module under
:envvar:`ai.environment_dir` and imports it, so the environment is a plain, readable file that can
be edited or deleted. The directory is re-imported when :mod:`dace.libraries.ai` is first imported,
which is how an SDFG expanded last week still resolves the environments its tasklets link against.

Inside a GPU kernel, only headers providing ``__device__`` functions are allowed, and the prompt
says so.

Configuration
-------------

All entries live under the ``ai`` category of the configuration, and can be set in ``.dace.conf``, through
:class:`dace.config.Config`, or as ``DACE_ai_*`` environment variables (see :ref:`config`):

* Provider and model: :envvar:`ai.provider`, :envvar:`ai.model`, :envvar:`ai.api_key_envvar`,
  :envvar:`ai.effort`, :envvar:`ai.max_tokens`, :envvar:`ai.timeout`, :envvar:`ai.manual_dir`
* Prompting: :envvar:`ai.extra_instructions`
* Verification: :envvar:`ai.verify`, :envvar:`ai.max_repair_attempts`
* Storage: :envvar:`ai.cache`, :envvar:`ai.cache_dir`, :envvar:`ai.sessions`,
  :envvar:`ai.session_dir`, :envvar:`ai.environment_dir`

A complete example
------------------

`samples/optimization/ai_generated_matmul.py <https://github.com/spcl/dace/blob/main/samples/optimization/ai_generated_matmul.py>`_
is a tiled matrix multiplication whose innermost microkernel is described rather than written. It
prints the generated kernel, measures it against NumPy, hands the measurement back as feedback for
another round, and lists the rounds the kernel has been through::

    ANTHROPIC_API_KEY=... python samples/optimization/ai_generated_matmul.py 512 --refine
