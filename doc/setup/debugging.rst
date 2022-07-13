Debugging
=========

Compiled SDFGs
--------------
To debug compiled programs, set build type to Debug, clear cache and rerun

``gdb --args python myscript.py args...``

Use the Visual Studio Code plugin with the DaCe debugger. Alternatively, use the Python C++ Debugger extension


DaCe Framework Behavior
-----------------------

``debugprint`` config (``DACE_debugprint=1`` or ``DACE_debugprint=verbose`` for more)

other important configuration entries (for frontend, ``frontend.verbose_errors``; for transformations, ``optimizer.match_exception``;
for serialization issues, ``testing.serialization``)

For debugging code generation, see :ref:`debug_codegen`
