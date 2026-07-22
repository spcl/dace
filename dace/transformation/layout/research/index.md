# Layout research

Research backing the SC26 layout suite (five-primitive layout algebra: Pad, Permute, Block, Shuffle, Zip/Unzip).

## Reports, by importance

1. [report2_layout_optimization_survey.md](report2_layout_optimization_survey.md) — catalog of published and production layout optimizations, each reduced to a before/after address function and mapped onto the five-primitive algebra.
2. [report1_cost_models_and_optimality_plan.md](report1_cost_models_and_optimality_plan.md) — cost-model families (LogP, ECM/roofline, pebble games, sparse traffic, accelerator mapping, learned) unified into a three-roof time bound, plus the action plan for proving layouts optimal.
3. [report3_provenance_and_bug_evidence.md](report3_provenance_and_bug_evidence.md) — verified citations, real-world layout bugs (12 repo issues/commits with measured numbers), and the finding that layout is absent from the performance-bug literature.
4. [coverage_gap_addendum.md](coverage_gap_addendum.md) — six bodies of work missing from the first sweep: Timeloop/MAESTRO/ZigZag/TCM/CMDS, OSKI/SPARSITY, YASK vector folding, GraphIt, WACO, XLA layout assignment/TpuGraphs.

## Bibliography

[references.bib](references.bib) — every external work cited across the four reports, deduplicated, BibTeX.
[references.md](references.md) — same list, numbered, human-readable, annotated by citing report.

## Parent suite

[../README.md](../README.md) — the layout transformation suite: primitives, passes, global assignment.
[../GLOBAL_LAYOUT_DESIGN.md](../GLOBAL_LAYOUT_DESIGN.md) — multi-nest layout assignment design.
