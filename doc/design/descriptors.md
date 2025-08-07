# Shared Descriptor Repository and Modifications to Nested SDFGs

With Control Flow Regions fully integrated into DaCe, some changes to how data descriptors are defined in nested contexts
can be both user-friendly and simpler to optimize.

## Executive Summary

An SDFG and all nested SDFGs will share one descriptor repository (data, symbols) and naming convention. Connectors
of nested SDFGs will be Passthrough Connectors, and Memlets going into and out of nested SDFGs will contain the smallest
subset of data being moved, similarly to Control Flow Regions and dataflow scopes (e.g. Maps), rather than acting as a
view.

## Problem Setting

Nested SDFGs were added at a time that the SDFG IR did not have Views and References. Thus, they served both as a way to introduce control flow inside dataflow scopes (i.e., map/consume) and reshape/offset/reinterpret data containers. This dual use is no longer relevant and it is detrimental to working with DaCe.

Consider the following simple program:

```python
@dace.program
def sample(A: dace.float32[M + 3, N], ind: dace.int32[M, N]):
    for i, j in dace.map[1:M, 2:N]:
        A[3 + i, 4 + ind[i, j]] += 1
```

The generated SDFG as of DaCe 1.0.2 will contain an SDFG for the map and a nested SDFG for the contents of the map.
The following issues can be observed:

1. `ind[i, j]` correctly appears on the outside of the nested SDFG. However, the `__tmp_*` internal data containers are named in a confusing manner, and the internal index is `[0]`, which loses the ability to understand, at O(1) time, what address is directly pointed at.
2. The outgoing memlet has the data-dependent index (2nd dimension) internally, which is as expected. However, the other half of the index is also in the nested SDFG, which creates a very over-approximated range of `[0:i+4]` on the outside. This makes analysis harder, but also transformations such as `LocalStorage` impossible.
3. The behavior is also inconsistent with the map outside, as the index may repeat inside and outside the map via propagated memlets (and still yield an analyzable expression), but that is not the case for the nested SDFG control flow region.
4. The symbol mapping in the nested SDFGs can lead to confusion if they are mapped in certain ways (e.g., `N` (external) -> `M` (internal) and `M` (external) -> `N` (internal)), which creates undue stress on transformation authoring and analysis checks.

## Proposal

Change nested SDFG semantics in the following manner:

* Memlets going into and out of a nested SDFG behave as if they are going to a scope via a passthrough connector. This means no offsetting in the code generation and only a union operation in memlet propagation.
* Nested SDFGs will not have their own descriptor repositories (i.e., arrays, symbols, constants, callbacks), and will instead share theirs with their parent.
    * Squeezing and unsqueezing memlets will no longer be necessary.
    * The symbol_mapping property of NestedSDFG will no longer exist.
* Nested SDFGs will contain a ControlFlowRegion instead of a fully-fledged SDFG object.

The changes will be made to code generation, memlet propagation, the `add_nested_sdfg` method (which will adapt the nested SDFG to its new parent), and all built-in transformations and passes that deal with nested SDFGs.

We will not lose any representational power (as the old behavior can be achieved with `View`s), and gain both readability and analyzability.

The only aspect lost is the ability to save a nested SDFG as its own separate unit. However, External SDFGs (i.e., #1671)
already enable this functionality as necessary.

## Work Plan

The work is designed as two separate pull requests:
1. #1696 - No-View Nested SDFGs makes the memlet behavior change going into/out of nested SDFGs
2. Descriptor repository unification (No PR yet)

Additionally, minor adaptations to the Python frontend (#2121) as part of the [frontend refactoring](doc/design/frontend.md)
will assist with mitigating many `tmp*` names in SDFGs.
