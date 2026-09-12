# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The tile vectorizer's refusal signal.

Lives in ``utils`` so any pass in the pipeline can raise it without importing the
orchestrator that catches it (which imports every pass in turn).
"""


class VectorizeUnsupported(Exception):
    """A kernel the K-dim tile pipeline cannot soundly vectorize.

    Raised by the pre-tiling soundness gates when the kernel carries a shape the tile widener would
    mis-lower -- a loop-carried / nested-reduction body WCR that would race the lanes (the
    ``no_wcr_in_map_body`` / ``no_wcr_inside_nested_sdfgs`` invariants), or a prep pass that could
    not lower the kernel to a valid tileable form. :meth:`VectorizeMultiDim.apply_pass` catches it,
    restores the pre-vectorization SDFG, and returns without tiling, leaving the kernel as its
    correct, un-tiled input dataflow.

    This turns a genuine capability limit into a clean *refusal to vectorize* rather than a hard
    crash (or a silently mis-tiled, wrong-numeric result). The underlying invariant CHECK is kept
    intact -- it is the detector -- per the maintainer direction: keep the soundness guard, but let
    it decline the kernel instead of aborting the whole run. Only the safe (e.g. perfectly-nested,
    lane-disjoint) reductions the widener DOES lower pass the guard and tile; every unsound shape is
    declined here.
    """
