# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""idxalg is an OPTIONAL dependency of the symbolic_engine seam.

Two halves of one contract: DaCe must work with idxalg absent, and an explicit
opt-in must fail loudly rather than silently running the other engine. Each case
runs in a subprocess because the backend is chosen once at import, and idxalg is
made genuinely unimportable via a ``sys.meta_path`` finder that raises -- merely
checking ``sys.modules`` would pass even when idxalg is installed.
"""
import os
import subprocess
import sys
import textwrap
from typing import Dict, List

BLOCK_IDXALG = textwrap.dedent("""
    import sys
    from importlib.abc import MetaPathFinder


    class BlockIdxalg(MetaPathFinder):

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "idxalg" or fullname.startswith("idxalg."):
                raise ImportError("idxalg is not installed")
            return None


    sys.meta_path.insert(0, BlockIdxalg())
    """)


def run_isolated(body: str, env_backend: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter with idxalg unimportable and the backend pinned."""
    script = BLOCK_IDXALG + textwrap.dedent(body)
    argv: List[str] = [sys.executable, "-c", script]
    # Inherit the environment so the child resolves the same DaCe; only the backend is pinned.
    env: Dict[str, str] = dict(os.environ)
    env["DACE_SYMBOLIC_BACKEND"] = env_backend
    return subprocess.run(argv, capture_output=True, text=True, env=env)


def test_default_backend_works_without_idxalg_installed():
    """The default (sympy) backend must not need idxalg, nor import it."""
    result = run_isolated(
        """
        import dace
        from dace import symbolic_engine as sp

        assert "idxalg" not in sys.modules, "default backend must not import idxalg"
        n = sp.Symbol("N")
        assert sp.Min(n, 4) is not None
        assert isinstance(n, sp.Basic) and isinstance(n, sp.Expr)

        import dace.symbolic as dsym
        assert dsym.symstr(dsym.pystr_to_symbolic("(N-1)//8"))
        print("OK")
        """, "sympy")
    assert result.returncode == 0, f"DaCe must import without idxalg:\n{result.stderr}"
    assert "OK" in result.stdout


def test_idxalg_opt_in_without_package_fails_loudly():
    """Opting in without the package names the variable and the package, and does NOT fall back."""
    result = run_isolated(
        """
        try:
            from dace import symbolic_engine  # noqa: F401
        except ImportError as ex:
            msg = str(ex)
            assert "DACE_SYMBOLIC_BACKEND" in msg, msg
            assert "idxalg" in msg, msg
            print("RAISED")
        else:
            print("FELL_BACK")
        """, "idxalg")
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout, f"expected a loud failure, not a silent sympy fallback: {result.stdout}"


if __name__ == "__main__":
    test_default_backend_works_without_idxalg_installed()
    test_idxalg_opt_in_without_package_fails_loudly()
