# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Inspect and validate a torch.export .pt2 model with the DaCe Dynamo backend.

Run from the repository root with PyTorch and DaCe installed::

    PYTHONPATH=. python samples/ml/check_exported_model.py model.pt2
    PYTHONPATH=. python samples/ml/check_exported_model.py model.pt2 --input-shape 1,3,32,32

The default input shapes/dtypes come from the export's example inputs. Floating
inputs are replaced with seeded random values; non-tensor arguments are kept.
The export's inference/training behavior is preserved (exported modules cannot
be switched with .eval()). CPU execution uses DaCe's default implementations.

"Before lowering" means the imported SDFG with library nodes; "after lowering"
means those nodes have been expanded into code-generatable nodes. If expansion
fails, the second file contains the partially expanded graph and report.json
records the failures. Correctness is checked only after successful compilation,
with fullgraph=True and PyTorch's eager fallback disabled.
"""

import argparse
import copy
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from torch.utils import _pytree as pytree

import dace
from dace.frontend.ml.torch.dynamo import DaceBackend, ops
from dace.frontend.ml.torch.dynamo.backend import select_implementations
from dace.frontend.ml.torch.dynamo.context import UnsupportedOpError
from dace.frontend.ml.torch.dynamo.importer import GraphImporter
from dace.frontend.ml.torch.dynamo.interface import _with_capture_config
from dace.frontend.ml.torch.dynamo.runtime import CompiledGraph
from dace.libraries.torch.nodes.aten import node_class
from dace.sdfg.nodes import LibraryNode


def input_shape(text):
    try:
        shape = tuple(int(dim) for dim in text.split(","))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError
        return shape
    except ValueError:
        raise argparse.ArgumentTypeError("expected positive dimensions separated by commas") from None


def clone_inputs(inputs):
    return pytree.tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x), inputs)


def random_inputs(examples, shapes, device):
    leaves, spec = pytree.tree_flatten(examples)
    tensors = [x for x in leaves if isinstance(x, torch.Tensor)]
    if shapes and len(shapes) != len(tensors):
        raise ValueError(f"provide one --input-shape per tensor input (this export has {len(tensors)})")
    result = []
    tensor_index = 0
    for x in leaves:
        if not isinstance(x, torch.Tensor):
            result.append(x)
            continue
        shape = shapes[tensor_index] if shapes else tuple(x.shape)
        tensor_index += 1
        if x.is_floating_point() or x.is_complex():
            sample = torch.randn(shape, dtype=x.dtype, device=device)
        elif x.dtype == torch.bool:
            sample = torch.randint(0, 2, shape, device=device).bool()
        else:
            # Generic integer samples cannot infer application-specific bounds
            # (e.g., embedding vocabulary size); use the exported values.
            if shape != tuple(x.shape):
                raise ValueError("shape overrides for integer inputs are unsupported")
            sample = x.to(device).clone()
        print(f"Input {tensor_index - 1}: shape={tuple(sample.shape)}, dtype={sample.dtype}, device={sample.device}")
        result.append(sample)
    return pytree.tree_unflatten(result, spec)


class InspectBackend(DaceBackend):
    """Keep AOTAutograd's runtime while inspecting each SDFG before compilation."""

    def __init__(self, output_dir, report):
        super().__init__(sdfg_name="pt2_model", onnx_fallback=False)
        self.output_dir = output_dir
        self.report = report
        self.current_graph = None

    def _compile_graph(self, gm, example_inputs, name, symbol_names, input_names):
        calls = [
            n
            for module in gm.modules()
            if isinstance(module, torch.fx.GraphModule)
            for n in module.graph.nodes
            if n.op == "call_function"
        ]
        histogram = Counter(str(n.target) for n in calls)
        print("Operators after DaCe/AOTAutograd decompositions:")
        for target, count in sorted(histogram.items()):
            print(f"  {count:4d}  {target}")
        entry = {"name": name, "operators": dict(histogram), "lowering_complete": False}
        self.report["graphs"].append(entry)
        self.current_graph = entry
        (self.output_dir / f"{name}.fx.py").write_text(gm.code)
        missing = Counter()
        for node in calls:
            if ops.lookup(node.target) is not None:
                continue
            if isinstance(node.target, torch._ops.OpOverload):
                if not node_class(str(node.target)).implementations:
                    missing[str(node.target)] += 1
            else:
                missing[str(node.target)] += 1
        entry["missing_operator_lowerings"] = dict(missing)
        if missing:
            print("Missing operator lowerings/implementations:")
            for target, count in sorted(missing.items()):
                print(f"  {target}: {count} occurrence(s)")
        else:
            print("All decomposed operators have registered lowerings; checking their implementations next.")
        importer = GraphImporter(self.options)
        result = importer.import_graph(gm, example_inputs, name, symbol_names=symbol_names, input_names=input_names)
        self.last_result = result
        return CompiledGraph(
            self._compile_sdfg(result.sdfg), result.inputs, result.outputs, communicators=result.communicators
        )

    def _compile_sdfg(self, sdfg):
        self.last_sdfg = sdfg
        entry = self.current_graph
        before = self.output_dir / f"{sdfg.name}.before_lowering.sdfgz"
        after = self.output_dir / f"{sdfg.name}.after_lowering.sdfgz"
        sdfg.save(str(before))
        entry["before_lowering"] = str(before)
        print(f"Saved before lowering: {before}")
        lowered = copy.deepcopy(sdfg)
        select_implementations(lowered)
        failures = []
        failed_nodes = set()
        # Continue past unsupported nodes so the report covers every occurrence,
        # including library nodes created by successful expansions.
        while True:
            pending = [
                (node, state)
                for node, state in lowered.all_nodes_recursive()
                if isinstance(node, LibraryNode) and node not in failed_nodes
            ]
            if not pending:
                break
            for node, state in pending:
                try:
                    node.expand(state)
                except Exception as exc:
                    failed_nodes.add(node)
                    failures.append(
                        {
                            "operator": getattr(node, "target", type(node).__name__),
                            "node": node.label,
                            "state": state.label,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        lowered.save(str(after))
        entry["after_lowering"] = str(after)
        entry["expansion_failures"] = failures
        entry["lowering_complete"] = not failures
        print(f"Saved after lowering{' (partial)' if failures else ''}: {after}")
        if failures:
            summary = Counter(f["operator"] for f in failures)
            print("Missing or failing library implementations:")
            for operator, count in sorted(summary.items()):
                error = next(f["error"] for f in failures if f["operator"] == operator)
                print(f"  {operator}: {count} occurrence(s): {error}")
            raise UnsupportedOpError("library expansion", "see the failures above and report.json")
        lowered.validate()
        print("All library nodes lowered successfully; compiling the SDFG.")
        compiled = lowered.compile()
        self.compile_count += 1
        return compiled


def check_outputs(actual, expected, rtol, atol):
    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    if actual_spec != expected_spec:
        raise AssertionError(f"output structure differs: {actual_spec} vs {expected_spec}")
    metrics = []
    for i, (out, ref) in enumerate(zip(actual_leaves, expected_leaves)):
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=False)
        if isinstance(ref, torch.Tensor):
            difference = (out.to(torch.complex128 if ref.is_complex() else torch.float64) - ref).abs()
            max_abs = difference.max().item() if difference.numel() else 0.0
            metrics.append({"output": i, "shape": list(ref.shape), "max_absolute_error": max_abs})
            print(f"Output {i}: shape={tuple(ref.shape)}, max absolute error={max_abs:.6g}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="a model saved with torch.export.save")
    parser.add_argument("--output-dir", type=Path, help="default: .dacecache/pt2_validation/<model stem>")
    parser.add_argument(
        "--input-shape",
        action="append",
        type=input_shape,
        default=[],
        help="comma-separated dimensions; repeat for each tensor input",
    )
    parser.add_argument("--device", default="cpu", help="execution device, e.g. cpu or cuda:0")
    parser.add_argument("--dynamic", action="store_true", help="capture symbolic sizes (default: specialize to sample)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true", help="show DaCe debug output and full tracebacks")
    args = parser.parse_args()
    output_dir = (args.output_dir or Path(".dacecache/pt2_validation") / args.model.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model.resolve()),
        "torch_version": torch.__version__,
        "dace_source": dace.__file__,
        "seed": args.seed,
        "device": args.device,
        "dynamic": args.dynamic,
        "rtol": args.rtol,
        "atol": args.atol,
        "graphs": [],
        "correctness": "not checked",
    }
    start = time.monotonic()
    exit_code = 1
    try:
        torch.manual_seed(args.seed)
        exported = torch.export.load(args.model)
        model = exported.module().to(args.device)
        inputs = random_inputs(exported.example_inputs, args.input_shape, args.device)
        report["input_shapes"] = [list(x.shape) for x in pytree.tree_leaves(inputs) if isinstance(x, torch.Tensor)]
        print(f"Running original exported model (constraints: {exported.range_constraints}).")
        with torch.no_grad():
            positional, keyword = clone_inputs(inputs)
            reference = model(*positional, **keyword)
        # Exports can contain unused symbolic shape expressions whose static
        # boolean results Dynamo cannot recapture. Remove dead code on a copy,
        # keeping the original reference untouched.
        capture_model = copy.deepcopy(model)
        capture_model.graph.eliminate_dead_code()
        capture_model.recompile()
        backend = InspectBackend(output_dir, report)
        # Use the frontend's tracing configuration, including convolution guards.
        compiled = _with_capture_config(
            torch.compile(capture_model, backend=backend, dynamic=args.dynamic, fullgraph=True)
        )
        with (
            torch.no_grad(),
            torch._dynamo.config.patch(suppress_errors=False),
            dace.config.set_temporary("debugprint", value=args.verbose),
        ):
            positional, keyword = clone_inputs(inputs)
            actual = compiled(*positional, **keyword)
        if backend.compile_count == 0:
            raise RuntimeError("DaCe did not compile an SDFG; cannot claim a correctness check")
        report["outputs"] = check_outputs(actual, reference, args.rtol, args.atol)
        report["correctness"] = "passed"
        print(f"PASS: DaCe matches the original model (rtol={args.rtol:g}, atol={args.atol:g}).")
        exit_code = 0
    except Exception as exc:
        detail = getattr(exc, "inner_exception", exc)
        report["error"] = f"{type(detail).__name__}: {detail}"
        if isinstance(detail, AssertionError):
            report["correctness"] = "failed"
        if args.verbose:
            traceback.print_exc()
        print("FAIL: " + report["error"])
        print(f"Correctness: {report['correctness']}.")
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {report_path} ({report['elapsed_seconds']:.1f}s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
