#!/usr/bin/env python3
"""
build_bridge.py  --  Auto-build and import hlfir_bridge.

Usage from other modules:
    from build_bridge import hb          # builds on first import if needed

Usage standalone:
    python build_bridge.py               # force rebuild
    python build_bridge.py --clean       # wipe build dir and rebuild

The build directory lives at <this_file's_dir>/build.  The compiled .so is
symlinked next to this file so that ``import hlfir_bridge`` works without
PYTHONPATH manipulation.
"""

import importlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  --  override via environment variables if needed.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_BUILD_DIR = _HERE / "build"

# Default LLVM version.  Override with LLVM_VERSION env var.
_LLVM_VERSION = os.environ.get("LLVM_VERSION", "21")

# LLVM cmake dir.  Override with env var; CMakeLists.txt also has its own
# defaults, so this is only needed if cmake's auto-detection misses.
# NOTE: we do NOT need MLIR_DIR  --  CMakeLists.txt locates MLIR through
# LLVM's install prefix using find_library, bypassing MLIR's broken cmake.
_LLVM_DIR = os.environ.get("LLVM_DIR", "")


def _find_llvm_prefix(version: str) -> str:
    """Derive the LLVM installation prefix from flang-new-<version>.

    /usr/bin/flang-new-21 is typically a symlink to
    /usr/lib/llvm-21/bin/flang-new-21.  Resolving it gives us the prefix.
    Falls back to /usr/lib/llvm-<version> if the binary isn't found.
    """
    flang = shutil.which(f"flang-new-{version}")
    if flang:
        real = Path(flang).resolve()
        prefix = real.parent.parent
        if (prefix / "lib" / "cmake").is_dir():
            return str(prefix)

    fallback = Path(f"/usr/lib/llvm-{version}")
    if fallback.is_dir():
        return str(fallback)

    return ""


def _find_llvm_cmake_dir(prefix: str, version: str) -> str:
    """Find LLVMConfig.cmake under a known prefix."""
    candidates = [
        f"{prefix}/lib/cmake/llvm",
        f"/usr/lib/llvm-{version}/lib/cmake/llvm",
        f"/usr/lib/llvm-{version}/cmake",
        f"/usr/local/lib/cmake/llvm",
        f"/opt/homebrew/lib/cmake/llvm",
    ]
    for c in candidates:
        p = Path(c)
        if p.is_dir() and (p / "LLVMConfig.cmake").exists():
            return str(p)

    for pkg in [f"llvm-{version}-dev", f"libllvm-{version}-dev"]:
        try:
            out = subprocess.check_output(
                ["dpkg", "-L", pkg],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                if line.endswith("LLVMConfig.cmake"):
                    return str(Path(line).parent)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    return ""


def _detect_dirs():
    """Populate _LLVM_DIR if not set."""
    global _LLVM_DIR
    prefix = _find_llvm_prefix(_LLVM_VERSION)
    if not _LLVM_DIR:
        _LLVM_DIR = _find_llvm_cmake_dir(prefix, _LLVM_VERSION)
    if not _LLVM_DIR:
        raise RuntimeError(f"Cannot find LLVMConfig.cmake for LLVM {_LLVM_VERSION}.  "
                           "Set LLVM_DIR env var.")


# ---------------------------------------------------------------------------
# Build logic
# ---------------------------------------------------------------------------


def _ext_suffix() -> str:
    """Python extension suffix, e.g. '.cpython-312-x86_64-linux-gnu.so'."""
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _so_name() -> str:
    return f"hlfir_bridge{_ext_suffix()}"


def _built_so() -> Path:
    """Path to the .so inside the build directory."""
    return _BUILD_DIR / _so_name()


def _local_so() -> Path:
    """Symlink target next to this file."""
    return _HERE / _so_name()


def needs_build() -> bool:
    """True if the extension is missing or older than any source file."""
    so = _local_so()
    if not so.exists():
        return True
    so_mtime = so.stat().st_mtime
    for pat in ("**/*.cpp", "**/*.h", "CMakeLists.txt"):
        for src in _HERE.glob(pat):
            if "build" in src.parts:
                continue
            if src.stat().st_mtime > so_mtime:
                return True
    return False


def build(clean: bool = False, verbose: bool = True):
    """Run cmake + make.  Raises on failure."""
    _detect_dirs()

    if clean and _BUILD_DIR.exists():
        if verbose:
            print(f"[build_bridge] cleaning {_BUILD_DIR}", file=sys.stderr)
        shutil.rmtree(_BUILD_DIR)

    _BUILD_DIR.mkdir(exist_ok=True)

    python = sys.executable
    nproc = os.cpu_count() or 4

    # --- cmake configure ---
    cmake_args = [
        "cmake",
        str(_HERE),
        f"-DLLVM_VERSION={_LLVM_VERSION}",
        f"-DLLVM_DIR={_LLVM_DIR}",
        f"-DPython_EXECUTABLE={python}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if verbose:
        print(f"[build_bridge] configure: {' '.join(cmake_args)}", file=sys.stderr)
    subprocess.check_call(cmake_args, cwd=_BUILD_DIR)

    # --- cmake build ---
    build_args = ["cmake", "--build", ".", f"-j{nproc}"]
    if verbose:
        print(f"[build_bridge] build: {' '.join(build_args)}", file=sys.stderr)
    subprocess.check_call(build_args, cwd=_BUILD_DIR)

    # --- symlink .so next to this file ---
    target = _built_so()
    link = _local_so()
    if not target.exists():
        candidates = list(_BUILD_DIR.rglob(_so_name()))
        if candidates:
            target = candidates[0]
        else:
            raise RuntimeError(f"Build succeeded but cannot find {_so_name()} "
                               f"under {_BUILD_DIR}")
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    if verbose:
        print(f"[build_bridge] linked {link} -> {target}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Import-or-build
# ---------------------------------------------------------------------------

_BRIDGE_MODULE = "dace.frontend.hlfir.hlfir_bridge"


def ensure_bridge():
    """Import the compiled bridge, building first if necessary.

    The extension is imported as the package submodule
    ``dace.frontend.hlfir.hlfir_bridge`` (``build`` symlinks the freshly
    built ``.so`` next to this file, i.e. into the ``dace.frontend.hlfir``
    package), so no ``sys.path`` manipulation is required.
    """
    try:
        return importlib.import_module(_BRIDGE_MODULE)
    except ImportError:
        pass

    print("[build_bridge] hlfir_bridge not found, building...", file=sys.stderr)
    build()
    importlib.invalidate_caches()
    return importlib.import_module(_BRIDGE_MODULE)


def ensure_fresh():
    """Import hlfir_bridge, rebuilding if any source is newer than the .so."""
    if needs_build():
        print("[build_bridge] sources newer than .so, rebuilding...", file=sys.stderr)
        build()
    return ensure_bridge()


# Module-level singleton: import this from other files.
hb = ensure_fresh()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the hlfir_bridge nanobind extension.")
    parser.add_argument("--clean", action="store_true", help="Wipe build dir before building.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build(clean=args.clean, verbose=not args.quiet)
    print(f"[build_bridge] OK: {_local_so()}")
