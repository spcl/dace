"""
This package contains the AST desugaring and analysis tools for the Fortran frontend.

AST Desugaring: In the context of the DaCe Fortran frontend, many high-level Fortran
features cannot be directly represented in the internal Fortran AST, which is used to
construct the data-centric representation (SDFG). The desugaring process replaces these
complex, expressive features with simpler, equivalent constructs that are well-supported
by the frontend. This transformation results in a simplified AST that the internal
representation can process correctly.

Analysis: The package also includes various analysis passes that gather information from the
AST, such as variable scopes, types, and array dimensions. This information is essential
for both the desugaring process and final SDFG generation. These analyses can also be
useful for certain early optimizations.
"""
