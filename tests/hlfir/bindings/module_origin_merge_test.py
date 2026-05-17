"""Unit coverage for the auto-detected-vs-explicit module-global
provenance merge (``effective_module_sources``) and the
``FrozenSignature`` JSON round-trip of the ``module_symbol_origins``
field.

The end-to-end recovery of ``_QM<mod>E<entity>`` provenance from real
HLFIR is exercised by ``test_velocity_full_auto_module_e2e``; this
file pins the cheaper invariants: explicit override precedence, the
auto-only path, and serialisation stability.
"""

from dace.frontend.hlfir.bindings.block_builders import effective_module_sources
from dace.frontend.hlfir.bindings.fortran_interface import OriginalInterface
from dace.frontend.hlfir.bindings.frozen_signature import FrozenArg, FrozenSignature


def _frozen(origins):
    return FrozenSignature(entry="k", mangled="_QPk", args=(), free_symbols=("nproma", ), module_symbol_origins=origins)


def _iface(sources):
    return OriginalInterface(entry="k", args=(), module_symbol_sources=sources)


def test_auto_only_no_handauthored_list():
    """Bridge-auto-detected origin is used when the iface map is empty."""
    fs = _frozen({"nproma": ("mo_parallel_config", "nproma")})
    merged = effective_module_sources(fs, _iface({}))
    assert merged == {"nproma": ("mo_parallel_config", "nproma")}


def test_explicit_override_wins_on_conflict():
    """A hand-authored entry overrides a (mis-)auto-detected origin."""
    fs = _frozen({"nproma": ("wrong_mod", "nproma")})
    merged = effective_module_sources(fs, _iface({"nproma": ("mo_parallel_config", "nproma")}))
    assert merged["nproma"] == ("mo_parallel_config", "nproma")


def test_explicit_supplements_auto():
    """Explicit entries the bridge could not recover are still honoured;
    auto entries the iface omits survive the merge."""
    fs = _frozen({"nproma": ("mo_parallel_config", "nproma")})
    merged = effective_module_sources(fs, _iface({"nrdmax": ("mo_vertical_grid", "nrdmax")}))
    assert merged == {
        "nproma": ("mo_parallel_config", "nproma"),
        "nrdmax": ("mo_vertical_grid", "nrdmax"),
    }


def test_missing_origins_attr_degrades_to_explicit():
    """A frozen signature without the new field (old snapshot) falls
    back to the explicit map without raising."""
    fs = FrozenSignature(entry="k", mangled="_QPk", args=())
    merged = effective_module_sources(fs, _iface({"nproma": ("mo_parallel_config", "nproma")}))
    assert merged == {"nproma": ("mo_parallel_config", "nproma")}


def test_frozen_signature_json_roundtrip(tmp_path):
    """``module_symbol_origins`` survives a JSON to-disk / from-disk
    round-trip as tuples (the binding emitter consumes this map; the
    per-arg origin is intentionally not a separate representation)."""
    fs = FrozenSignature(entry="k",
                         mangled="_QPk",
                         args=(FrozenArg(fortran_name="nrdmax",
                                         sdfg_name="nrdmax",
                                         kind="array",
                                         dtype="int32",
                                         rank=1,
                                         shape=("10", ),
                                         intent="inout"), ),
                         free_symbols=("nproma", ),
                         module_symbol_origins={"nproma": ("mo_parallel_config", "nproma")})
    p = tmp_path / "fs.json"
    fs.to_json(str(p))
    rt = FrozenSignature.from_json(str(p))
    assert rt.module_symbol_origins == {"nproma": ("mo_parallel_config", "nproma")}
    assert rt.args[0].shape == ("10", )
