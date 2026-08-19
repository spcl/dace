.. _new-instrumentation:

Writing a Custom Instrumentation Provider
=========================================

Instrumentation providers are the mechanism through which DaCe's code
generator inserts measurement code (timers, hardware counters, NVTX
ranges, ...) around SDFG elements at compile time. The built-in providers
live under :mod:`dace.codegen.instrumentation`; users can add new ones
without modifying DaCe by subclassing
:class:`~dace.codegen.instrumentation.provider.InstrumentationProvider` and
registering the class with :class:`~dace.dtypes.InstrumentationType`.

When does a provider run?
-------------------------

Every SDFG element has an ``instrument`` (or ``instrument_condition``)
property that takes an :class:`~dace.dtypes.InstrumentationType`. During
code generation, the framecode driver looks up the provider associated with
that enum value and calls back into it at well-defined points around the
element being lowered. A single provider may handle many elements - the
:class:`~dace.codegen.instrumentation.timer.TimerProvider`, for example,
emits start/stop calls around states, scopes, and full SDFGs.

Skeleton of a provider
----------------------

A minimal provider extends one or more of the ``on_*`` callbacks defined on
the base class. Every callback receives the relevant SDFG/state/node along
with a *local* and *global* :class:`~dace.codegen.prettycode.CodeIOStream`
into which to emit code. The local stream injects code at the current
position in the generated function; the global stream injects code at the
top of the translation unit (useful for ``#include`` directives, helper
functions, and global state).

.. code-block:: python

    from dace import dtypes, registry
    from dace.codegen.instrumentation.provider import InstrumentationProvider


    @registry.autoregister_params(type=dtypes.InstrumentationType.MyTimer)
    class MyTimerProvider(InstrumentationProvider):
        """Example provider that brackets every state with a custom timer."""

        def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
            global_stream.write('#include "my_timer.h"', sdfg)

        def on_state_begin(self, sdfg, cfg, state, local_stream, global_stream):
            local_stream.write(f'my_timer_start("{state.label}");', sdfg, state.block_id)

        def on_state_end(self, sdfg, cfg, state, local_stream, global_stream):
            local_stream.write(f'my_timer_stop("{state.label}");', sdfg, state.block_id)

The decorator :func:`@registry.autoregister_params <dace.registry.autoregister_params>`
adds ``MyTimer`` to :class:`~dace.dtypes.InstrumentationType` so that any
SDFG element can opt in via ``element.instrument = dtypes.InstrumentationType.MyTimer``
or by setting the ``instrument`` property in the SDFG editor.

Available hooks
---------------

The full set of callbacks is documented on
:class:`~dace.codegen.instrumentation.provider.InstrumentationProvider`.
The most commonly overridden ones are:

* ``on_sdfg_begin`` / ``on_sdfg_end`` - bracket the entire program. Use
  the global stream to emit ``#include``\ s and helper definitions.
* ``on_sdfg_init_begin`` / ``on_sdfg_init_end`` - run inside the
  generated ``__dace_init`` function (state allocation, handle creation).
* ``on_sdfg_exit_begin`` / ``on_sdfg_exit_end`` - run inside
  ``__dace_exit`` (state teardown).
* ``on_state_begin`` / ``on_state_end`` - bracket each state.
* ``on_scope_entry`` / ``on_scope_exit`` - bracket a map or consume scope.
* ``on_node_begin`` / ``on_node_end`` - bracket an individual node.
* ``on_copy_begin`` / ``on_copy_end`` - bracket a memory copy.
* ``on_allocation_begin`` / ``on_allocation_end`` - around an allocation
  performed by the codegen.
* ``on_deallocation_begin`` / ``on_deallocation_end`` - around a
  deallocation.

Each callback's default implementation is a no-op, so a provider only
needs to override the events it cares about.

Producing instrumentation reports
---------------------------------

The :class:`~dace.codegen.instrumentation.report.InstrumentationReport`
class is the standard container for runtime measurements: it deserializes
JSON written by the runtime, and the SDFV viewer can render it as overlays
on the SDFG. If your provider writes its results in the same JSON format
(see :func:`~dace.codegen.instrumentation.report.InstrumentationReport.from_file`
for the schema), the resulting reports integrate transparently into the
existing tooling. Providers that need a custom report layout can return an
``InstrumentationReport`` subclass; the only requirement is that
``durations`` and ``counters`` are accessible per SDFG/state/node so that
overlays render correctly.

Built-in providers
------------------

The built-in providers are concise and instructive examples of the
technique:

* :mod:`dace.codegen.instrumentation.timer` - host-side timers using
  ``std::chrono``. Demonstrates state and scope-level brackets.
* :mod:`dace.codegen.instrumentation.gpu_events` - CUDA/HIP event-based
  timers. Demonstrates how to interact with target-specific runtime APIs.
* :mod:`dace.codegen.instrumentation.papi` - PAPI hardware-counter
  integration. Demonstrates how to thread per-thread state through the
  generated runtime struct via ``state_fields``.
* :mod:`dace.codegen.instrumentation.likwid` - LIKWID region markers,
  another good cross-platform reference.
