"""Back-compat shim — the real builder lives in ``.builder``.

Existing callers do ``from dace.frontend.hlfir.hlfir_to_sdfg import
SDFGBuilder, DEFAULT_PIPELINE, generate_sdfg``; this one-line re-export
keeps that import path working after the split into per-concern
modules under ``builder/``.

``__all__`` pins the public surface so re-exports aren't flagged as
"imported but unused" by linters (F401) — no per-line suppressions
needed.
"""
from dace.frontend.hlfir.builder import SDFGBuilder, DEFAULT_PIPELINE, generate_sdfg
from dace.frontend.hlfir.builder.context import _Ctx

__all__ = ["SDFGBuilder", "DEFAULT_PIPELINE", "generate_sdfg", "_Ctx"]

if __name__ == "__main__":
    # Minimal CLI preserved for quick manual inspection:
    #   python3 -m dace.frontend.hlfir.hlfir_to_sdfg <input.hlfir> [output.sdfg]
    import sys

    import dace
    from dace.sdfg import nodes as nd
    from dace.sdfg.state import LoopRegion

    path = sys.argv[1] if len(sys.argv) > 1 else "multi_stmt.hlfir"
    sdfg = generate_sdfg(path)
    sdfg.validate()
    print(f"SDFG: {sdfg.name}")

    def show_region(region, indent=0):
        p = "  " * indent
        for node in region.nodes():
            if isinstance(node, LoopRegion):
                li = node.init_statement.as_string if node.init_statement else "?"
                lc = node.loop_condition.as_string if node.loop_condition else "?"
                lu = node.update_statement.as_string if node.update_statement else "?"
                print(f"{p}FOR {li}; {lc}; {lu}")
                show_region(node, indent + 1)
            elif isinstance(node, dace.SDFGState):
                print(f"{p}State '{node.label}':")
                for n in node.nodes():
                    if isinstance(n, nd.Tasklet):
                        print(f"{p}  Tasklet: {n.code.as_string.strip()}")
                    elif isinstance(n, nd.AccessNode):
                        print(f"{p}  Data: {n.data}")

    show_region(sdfg)
    out = sys.argv[2] if len(sys.argv) > 2 else f"{sdfg.name}.sdfg"
    sdfg.save(out)
    print(f"\nSaved: {out}")
