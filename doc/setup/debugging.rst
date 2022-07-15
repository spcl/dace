Debugging
=========

Graph Validation
----------------

_dacegraphs/invalid.sdfg validation failure - it even zooms in on the error in vscode!

Compiled Programs
-----------------

For debugging code generation, see :ref:`debug_codegen`


To debug compiled programs, set build type to Debug, clear cache and rerun

``gdb --args python myscript.py args...``

Use the Visual Studio Code plugin with the DaCe debugger. Alternatively, use the Python C++ Debugger extension

GPU Debugging in DaCe
~~~~~~~~~~~~~~~~~~~~~

``compiler.cuda.syncdebug``: If True, calls device-synchronization after every GPU kernel and checks for errors. 
Good for checking crashes or invalid memory accesses.

``cuda-gdb`` also useful


Verbose Framework Printouts
---------------------------

``debugprint`` config (``DACE_debugprint=1`` or ``DACE_debugprint=verbose`` for more)

other important configuration entries (for frontend, ``frontend.verbose_errors``; for transformations, ``optimizer.match_exception``;
for serialization issues, ``testing.serialization``)

