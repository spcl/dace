.. _new-backend:

New Backends
============

A *backend* in DaCe is a code generator that lowers some subset of the SDFG IR
to source code in a specific language and emits the build artifacts needed to
compile that source. The built-in backends live under
:mod:`dace.codegen.targets` (CPU, CUDA, MPI, SVE, MLIR,
Snitch, ...). Adding a new backend means plugging into three points of the
codegen pipeline: the user-facing enumerations, the dispatcher, and the
build/link environment.

This page covers the plumbing. For a worked example, see the
`Extending the Code Generator
<https://nbviewer.org/github/spcl/dace/blob/main/tutorials/codegen.ipynb>`_
tutorial, which builds a small custom target end to end. The
``dace/codegen/targets/cpu.py`` and ``dace/codegen/targets/cuda.py`` files are
the canonical references for any non-trivial backend.

.. _enums:

Extensible Enumerations
-----------------------

Most DaCe enumerations - :class:`~dace.dtypes.DeviceType`,
:class:`~dace.dtypes.ScheduleType`, :class:`~dace.dtypes.StorageType`,
:class:`~dace.dtypes.InstrumentationType`,
:class:`~dace.dtypes.DataInstrumentationType`, and
:class:`~dace.dtypes.Language`, among others - are *user-extensible*.
They subclass :class:`dace.attr_enum.ExtensibleAttributeEnum`, which is a
drop-in extension of the standard library's :class:`enum.Enum` that adds
two capabilities a backend frequently needs:

1. **New members can be registered at runtime**, with either a plain
   value or a frozen dataclass acting as a *template* for attributed
   instances.
2. **Members can carry attributes** (via dataclass templates), which is
   what lets ``ScheduleType.GPU_ThreadBlock`` and friends remain symbolic
   while richer entries (e.g., a vendor-specific schedule with tunable
   parameters) carry the parameters on the enum value itself.

Adding plain entries
~~~~~~~~~~~~~~~~~~~~

The simplest case is adding a new constant. Use the
:meth:`~dace.attr_enum.ExtensibleAttributeEnum.register` classmethod from
your backend's setup code (typically the package's ``__init__.py``):

.. code-block:: python

    from dace import dtypes

    # Auto-assigned value (analogous to `auto()`):
    dtypes.DeviceType.register('MyAccelerator')
    dtypes.ScheduleType.register('MyAccelerator_Device')
    dtypes.ScheduleType.register('MyAccelerator_ThreadBlock')
    dtypes.StorageType.register('MyAccelerator_Global')

    # Explicit value (pass it as the second argument):
    dtypes.StorageType.register('MyAccelerator_Shared', 301)

After registration, the new entries are reachable as
``dtypes.ScheduleType.MyAccelerator_Device`` and serialize cleanly in
saved SDFGs - the SDFG (de)serializer round-trips any
``ExtensibleAttributeEnum`` member through the registry.

Adding attributed entries (templates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a backend needs to attach data to an enum member - for example, the
parameters of a vendor-specific schedule - register a *template*. A
template is just a frozen dataclass; calling the template produces an
instance of the enum that carries the supplied field values:

.. code-block:: python

    from dataclasses import dataclass
    from dace import dtypes

    @dataclass(frozen=True)
    class MyAcceleratorKernel:
        block_size: int
        smem_kb: int

    dtypes.ScheduleType.register_template('MyAccelerator_Kernel',
                                          MyAcceleratorKernel)

    # Use the template like any other enum member:
    sched = dtypes.ScheduleType.MyAccelerator_Kernel(block_size=128,
                                                      smem_kb=48)
    sched.block_size  # -> 128

Two design points worth noting (mirrored in
:mod:`tests.utils.attrenum_test`):

* The *template* compares equal to any of its instances, which makes
  attributed members ergonomic in ``match`` / ``case`` blocks
  (``case dtypes.ScheduleType.MyAccelerator_Kernel:``).
* Two instances created with the same arguments are interned to the
  same object (``sched is dtypes.ScheduleType.MyAccelerator_Kernel(block_size=128, smem_kb=48)``),
  which keeps SDFG hashing and identity comparisons predictable.

Templates can also be declared inline at class-definition time when you
own the enum class, by writing ``@dataclass(frozen=True)`` classes
inside the enum body; this is the form used in the test suite. From a
backend extension's perspective, ``register_template`` is the equivalent
runtime form.

Optional: ``undefined_safe_enum``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decorator :func:`dace.registry.undefined_safe_enum` is an *optional*
helper that registers a single extra member named ``Undefined`` on an
``ExtensibleAttributeEnum``. The built-in DaCe enums use it so that
deserialization, partially-specified IR, and forward-compatibility paths
have a well-known sentinel to fall back to instead of raising on an
unknown enum value:

.. code-block:: python

    from dace.attr_enum import ExtensibleAttributeEnum
    from dace.registry import undefined_safe_enum
    from enum import auto

    @undefined_safe_enum
    class MyEnum(ExtensibleAttributeEnum):
        First = auto()
        Second = auto()

    MyEnum.Undefined  # added by the decorator

Using the decorator is purely a convenience - if you do not need an
``Undefined`` sentinel for your own enum, omit it. Custom enums that
extend the built-in ones (via ``register`` / ``register_template``)
inherit the sentinel from their parent class and do not need to apply
the decorator themselves.

Implementing a Target Code Generator
------------------------------------

A backend is a subclass of
:class:`~dace.codegen.targets.target.TargetCodeGenerator` registered through
:func:`@dace.registry.autoregister_params`. The class is instantiated once
per SDFG by the framecode driver and is responsible for:

* declaring which schedules, storage types, and node patterns it can handle;
* emitting the source files to be compiled (returning them as
  ``CodeObject`` instances from :meth:`get_generated_codeobjects`);
* providing the per-node, per-state, per-copy, and per-allocation
  generators that the dispatcher will call back into.

Inside the constructor, the backend registers itself with the
:class:`~dace.codegen.dispatcher.TargetDispatcher` for the events it wants
to handle:

.. code-block:: python

    class MyTargetCodeGen(target.TargetCodeGenerator):
        def __init__(self, frame, sdfg):
            self._frame = frame
            self._dispatcher = frame.dispatcher

            # Maps with these schedules will go through `generate_node`/_state.
            self._dispatcher.register_map_dispatcher(
                [dtypes.ScheduleType.MyAccelerator_Device], self)

            # Arrays in this storage will go through this target's
            # allocation/deallocation hooks.
            self._dispatcher.register_array_dispatcher(
                dtypes.StorageType.MyAccelerator_Global, self)

            # Copies between these storages will be lowered by this target.
            self._dispatcher.register_copy_dispatcher(
                dtypes.StorageType.CPU_Heap,
                dtypes.StorageType.MyAccelerator_Global,
                None,
                self)

The full list of registration entry points is documented on
:class:`~dace.codegen.dispatcher.TargetDispatcher`:

* :meth:`~dace.codegen.dispatcher.TargetDispatcher.register_state_dispatcher`
* :meth:`~dace.codegen.dispatcher.TargetDispatcher.register_node_dispatcher`
* :meth:`~dace.codegen.dispatcher.TargetDispatcher.register_map_dispatcher`
* :meth:`~dace.codegen.dispatcher.TargetDispatcher.register_array_dispatcher`
* :meth:`~dace.codegen.dispatcher.TargetDispatcher.register_copy_dispatcher`

Each ``register_*`` call accepts an optional predicate so that a target can
opt in or out based on richer criteria than just an enum value.

The target then implements the corresponding callbacks
(``generate_state``, ``generate_node``, ``allocate_array``,
``deallocate_array``, ``copy_memory``, ...). Each callback receives the
relevant SDFG, state, node(s), and a pair of :class:`CodeIOStream` objects
into which to emit code at the local (current scope) and global
(translation-unit-level) positions.

Build environments and linked libraries
---------------------------------------

A backend almost always needs to add headers, runtime support files, and
link-time dependencies to the build. The recommended way to declare these is
via :ref:`environments <libraries>`: define a ``@dace.library.environment``
that lists the ``cmake_packages``, ``headers``, ``cmake_libraries``, and
runtime ``state_fields`` your generated code requires, and add it to the
``environments`` list of the relevant code objects.

For more invasive integration (e.g., additional source
files generated alongside the SDFG), the ``CodeObject`` returned by the
target can declare extra ``additional_files`` and target-specific
``additional_compiler_flags``. The CMake driver in
:mod:`dace.codegen.compiler` consumes those fields when building the shared
library.

Once registered, the backend is selected automatically whenever an SDFG
contains nodes/arrays/maps with the schedules, storage types, or node
classes the target claimed. No changes to user-facing APIs are required.
