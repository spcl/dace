# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Inference with a torchvision model through the DaCe TorchDynamo frontend, two ways.

The first route is ``torch.compile(model, backend='dace', dynamic=True)``: the SDFG is built and compiled on the first
call and reused for every batch size. ``dace.ml.sdfg_of(compiled)`` returns the SDFG the shared backend produced, whose
``build_folder`` holds the generated C++.

The second route is ahead of time: ``dace.ml.to_sdfg(model, dace.float32[N, 3, H, W])`` builds the SDFG straight from
DaCe data descriptors (no example tensors needed), with a symbolic batch size ``N``. The SDFG is saved and compiled
with ``dace.ml.compile(...)``, which returns a callable that runs on torch tensors of any batch size and binds the
model's parameters at call time. Both routes run on CPU with pure expansions; the model is in eval mode, so the whole
network is a single inference graph.
"""

import argparse
import time

import torch
from torchvision.models import squeezenet1_1

import dace
import dace.ml

N = dace.symbol("N")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # The real ImageNet size (224x224) compiles here in well under a minute on CPU; --size lowers it if needed.
    parser.add_argument("--size", type=int, default=224, help="spatial size (height and width) of the input")
    parser.add_argument("--save", type=str, default="squeezenet_inference.sdfgz", help="where to save the SDFG")
    args = parser.parse_args()
    h = w = args.size

    torch.manual_seed(0)
    model = squeezenet1_1(weights=None).eval()
    start = time.time()

    # Route 1: torch.compile with the 'dace' backend. The SDFG is created on the first call.
    compiled = torch.compile(model, backend="dace", dynamic=True)
    with torch.no_grad():
        compiled(torch.randn(4, 3, h, w))
    sdfg = dace.ml.sdfg_of(compiled)  # Also reachable as the shared dace.ml.dace_backend.last_sdfg
    print(f"[torch.compile] SDFG '{sdfg.name}' built; generated C++ is under {sdfg.build_folder}")

    # Route 2: ahead of time from a descriptor with a symbolic batch size N.
    exported = dace.ml.to_sdfg(model, dace.float32[N, 3, h, w])
    exported.save(args.save)
    print(
        f"[to_sdfg] saved SDFG '{exported.name}' to {args.save} (symbols: {sorted(str(s) for s in exported.free_symbols)})"
    )

    runner = dace.ml.compile(model, dace.float32[N, 3, h, w])
    for batch in (2, 8):  # One compiled SDFG serves every batch size
        x = torch.randn(batch, 3, h, w)
        with torch.no_grad():
            reference = model(x)
        output = runner(x)
        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
        print(f"[to_sdfg] batch {batch}: output {tuple(output.shape)} matches eager torch")

    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
