# Layout Optimizations Survey — with before/after NumPy pseudocode

A catalog of published and production data-layout optimizations, each reduced to a BEFORE and AFTER address function with a runnable NumPy sketch, and decomposed into the SC26 five-primitive algebra {Pad, Permute, Block, Shuffle, Zip/Unzip}. Organized by transformation family. Entries marked [RECONSTRUCTED] have address functions inferred from figures/text, not printed verbatim in the source. This version integrates a deep sweep of engineering blogs and production systems (PyTorch, NVIDIA/AMD, Kokkos, GROMACS, databases, data-oriented design, search-tree layouts).

Notation: `addr(i,j,...)` is the linear byte/element offset; `B` = block/cache-line length in elements; row-major (C-order) is the baseline linearization.

---

## Family 1 — Padding (break set-mapping pathologies; no schedule fixes these)

### 1.1 Intra-array & inter-array padding — Rivera & Tseng, PLDI 1998
Motivating pattern: stencil/GEMM where a power-of-two leading dimension makes columns collide in the same cache set.
- BEFORE: `A = zeros((N, N))`, `addr(i,j) = i*N + j`; column stride N is a power of two → conflict misses every access.
- AFTER (pad dimension): `A = zeros((N, N+p))`, `addr(i,j) = i*(N+p) + j`; the pad `p` breaks the set periodicity.
```python
A = np.ascontiguousarray(np.zeros((N, N+P)))[:, :N]   # pad leading dim, view logical
```
Primitive: **Pad**. Automated (compiler heuristic: choose p so leading dim is coprime with #sets).

### 1.2 GCD-based padding formula — Hong et al., PLDI 2016
Same idea, principled pad size from set-mapping (GCD of stride and cache parameters). Primitive: **Pad**. Automated (analytic).

### 1.3 Shared-memory tile padding — Mark Harris, NVIDIA Technical Blog ("An Efficient Matrix Transpose in CUDA C/C++")
Motivating pattern: a 32×32 float tile in shared memory; reading a column is a 32-way bank conflict because a whole column maps to one bank.
- BEFORE: `tile[32][32]`, `bank(r,c) = c % 32` constant down a column → 32-way conflict.
- AFTER: `tile[32][33]` (pad the inner dim by 1); now `bank(r,c) = (r*33 + c) % 32` walks all banks down a column.
Reported: padding to 33 brings the transpose to ~95% of the copy-kernel throughput (bank-conflict-free); the CUDA-Fortran companion reports ~93%. Lei Mao's reproduction: padding and XOR-swizzle give identical latency, ~20% over the conflicted kernel on RTX 3090.
```python
tile = np.zeros((32, 33))          # only [:, :32] holds data; col 32 is the pad
```
Primitive: **Pad** (alternative to Shuffle/XOR-swizzle). Manual (idiom); this is the canonical k02/k-suite bank-conflict entry.

### 1.4 False-sharing / alignment padding — JVM `@Contended`, Linux `cacheline_aligned`
Pad a hot field or per-thread slot to a full cache line so independent writers stop invalidating each other. Primitive: **Pad** (base-address / trailing pad). Manual.

---

## Family 2 — Dimension permutation / transposition

### 2.1 Nonsingular data-transformation matrices — Kandemir et al. MICRO'98 / JPDC'99; O'Boyle & Knijnenburg IJPP'99
A layout is an invertible integer matrix M applied to the index vector: `addr = linearize(M · i)`. Permutation and skew are special cases.
- BEFORE (M = I): `A[i,j]`, `addr = i*N + j`.
- AFTER (M = [[0,1],[1,0]] = transpose): `addr = j*N + i` (row-major↔column-major); skew M = [[1,0],[1,1]] gives `addr = i*N + (i+j mod N)`.
```python
A_T = np.ascontiguousarray(A.T)          # permutation
# skew:
S = np.ascontiguousarray(np.stack([np.roll(A[i], i) for i in range(N)]))
```
Primitive: **Permute** (+ **Shuffle** for skew). Automated (unimodular framework, closed under composition).

### 2.2 PyTorch `channels_last` (NCHW→NHWC as a stride change) — PyTorch/Intel engineering blogs
The most-deployed permutation in ML. The logical shape stays NCHW; only the physical strides change, so ops see the same indices but different memory order.
- BEFORE (contiguous NCHW): shape (N,C,H,W), `addr(n,c,h,w) = ((n*C+c)*H+h)*W+w`; channel reduction strides by H·W.
- AFTER (channels_last): same logical (N,C,H,W) but strides = NHWC, `addr = ((n*H+h)*W+w)*C+c`; the C reduction is contiguous → matches cuDNN/oneDNN tensor-core kernels, eliminating transposes.
Reported: PyTorch measured 22% ResNet50 training speedup and 8–35% across models on V100 with AMP; ~3× MobileNetV2 on mobile; on CPU, oneDNN consumes NHWC directly (saves a conversion). "Channel thrashing" if an unsupported op forces a round-trip to NCHW — the cost of a layout conflict, documented in production.
```python
x = np.ascontiguousarray(x_nchw.transpose(0,2,3,1))   # NHWC bytes
# logical NCHW view with NHWC strides:
x_logical = x.transpose(0,3,1,2)                       # no copy
```
Primitive: **Permute**. Automated (framework propagates the format; conflicts materialize as copies). Directly your k13.

### 2.3 Kokkos `LayoutLeft` vs `LayoutRight` (polymorphic per memory space) — Trott et al. TPDS 2022; Kokkos docs
A `View`'s layout is a compile-time template; the default flips by memory space — HostSpace defaults to LayoutRight (row-major, cache-friendly), CudaSpace to LayoutLeft (column-major, coalescing-friendly), so the SAME source gets opposite layouts on CPU vs GPU. The mapping of the parallel index to the contiguous dimension is co-chosen.
- LayoutRight: `addr(i,j) = i*N1 + j`; LayoutLeft: `addr(i,j) = i + j*N0`.
Reported: docs state uncoalesced/uncached access can cost ~10×; a Kokkos issue reports 34% from forcing LayoutRight on a CUDA shared-memory view (bank-conflict avoidance); portability studies report Kokkos holding 96–100% efficiency by swapping layout where hand-code needed refactoring, at up to ~2.5× slowdown vs fully hand-optimized in some configs.
```python
A_right = np.ascontiguousarray(A)        # row-major (CPU default)
A_left  = np.asfortranarray(A)           # column-major (GPU default)
```
Primitive: **Permute** (compile-time polymorphic). Automated (the layout is chosen by memory space; this is the production instance of "same program, layout flips by device" that your model must predict). Relates to k04, k07.

### 2.4 XLA / TPU layout assignment (minor-to-major permutation, propagated) — OpenXLA docs; TpuGraphs (Phothilimthana et al., NeurIPS'23)
Each HLO op gets a physical layout (a permutation over dims, "minor-to-major"); layouts propagate through the graph; conflicts at edges materialize as inserted copy/transpose ops. TPU MXUs prefer 128×8-tile-aligned layouts, so XLA transposes/reshapes/pads and sometimes propagates preferred layouts backward to minimize total transform cost.
Reported: XLA graph-level layout autotuning delivers 10–20% on production models; layout tuning "offers the most speedup in general"; TpuGraphs is the public dataset of (HLO graph × layout config × runtime). The Ragged-Paged-Attention TPU report documents XLA overriding user layout constraints and silently transposing intermediates — evidence even mature layout-assignment lacks a per-kernel-accurate cost model.
Primitive: **Permute** (+ Pad to tiles). Automated (constraint propagation + learned/autotuned cost). The graph-level analog of Kennedy–Kremer; complements k04/k10 cross-op conflicts.

---

## Family 3 — Blocking / tiling and nonlinear (Morton) layouts

### 3.1 4D blocked & Morton/Z-order — Chatterjee et al., ICS 1999
- BEFORE (row-major): `addr(i,j) = i*N + j`.
- AFTER (4D tiled, tile T): `addr = ((i//T)*(N//T) + j//T)*T*T + (i%T)*T + (j%T)` — tiles contiguous.
- AFTER (Morton): `addr = interleave_bits(i, j)`.
```python
Ab = np.ascontiguousarray(A.reshape(N//T,T,N//T,T).transpose(0,2,1,3))  # 4D blocked
```
Reported: 1.1–2.5× on dense kernels. Primitive: **Block** (+ **Shuffle** for Morton). Manual/automated. Your k02.

### 3.2 YASK vector folding — Yount, HPCC'15 / WOLFHPC'16 (Intel)
Instead of 1D in-line SIMD along the unit-stride dim, pack neighbors into small multi-dim tiles per SIMD register, and order those tiles in memory along a dimension independent of the vectorized dims — increases reuse for multi-dimensional stencils.
- BEFORE (1D fold): `addr(i,j,k)=((i*Nj)+j)*Nk+k`, vector = 8 consecutive k.
- AFTER (2×2×2 fold, [RECONSTRUCTED]): `A_fold[i//2, j//2, k//2, i%2, j%2, k%2]` — Block of (i,j,k) into 2³ tiles + Permute of tile order.
```python
Af = np.ascontiguousarray(A.reshape(Ni//2,2,Nj//2,2,Nk//2,2).transpose(0,2,4,1,3,5))
```
Primitive: **Block + Permute**. Automated (YASK DSL + genetic-algorithm tuner). Stencil counterpart of k12/DLT.

### 3.3 Dimension-lifted transposition (DLT) — Henretty et al., CC 2011
Reshape 1D (N,) → (V, N/V) → transpose to (N/V, V): stencil neighbors land in the same SIMD lane of adjacent rows instead of adjacent lanes of one vector, removing alignment shuffles. Primitive: **Block + Permute**. Your k12.

### 3.4 IREE data-tiling / `linalg.mmt4d`, MLIR `tensor.pack`/`unpack` — IREE blog (Jacob et al.)
GEMM operands are packed into a 4D tiled "mmt4d" layout matched to the target's register tile before the matmul, then unpacked; the pack is a first-class op the compiler inserts and fuses.
- BEFORE: A (M,K) row-major.
- AFTER (mmt4d, tile M0×K0): A4 (M//M0, K//K0, M0, K0), `addr = ((mo*(K//K0)+ko)*M0+mi)*K0+ki`.
```python
A4 = np.ascontiguousarray(A.reshape(M//M0,M0,K//K0,K0).transpose(0,2,1,3))
```
Primitive: **Block** (+ Permute). Automated (compiler pass with a cost model). Same shape as BLIS packing (Family 6) and k04-repack.

### 3.5 GPU texture tiling / VK_IMAGE_TILING_OPTIMAL, Morton-swizzled textures — vendor docs
Textures are stored in an opaque Z-order/tiled layout so 2D neighborhoods (filtering) are memory-local; `LINEAR` tiling is row-major and slower for 2D access. Primitive: **Shuffle/Block** (Morton). Automated (driver-chosen; opaque).

---

## Family 4 — Element shuffles / swizzles / reordering (irregular & GPU shared memory)

### 4.1 XOR swizzles — CUTLASS/CuTe `Swizzle<B,M,S>`; Gonzalez et al. ICS'97; Rau ISCA'91
- BEFORE: `bank(row,col) = col % 32`; a column of a row-major tile is one bank (32-way conflict).
- AFTER (XOR): `col' = col XOR (row & mask)`; consecutive rows read conflict-free, and one physical layout serves both cp.async row-writes and ldmatrix column-reads.
```python
sw = np.empty_like(tile)
for r in range(R):
    sw[r] = tile[r, (np.arange(C) ^ (r & MASK))]
```
Reported: eliminating ldmatrix conflicts ≈ 2× in a from-scratch tensor-core GEMM; identical to +1 padding on RTX 3090. Primitive: **Shuffle**. Manual/library.

### 4.2 Triton Linear Layouts (F₂ matrices) — arXiv 2505.23819
A layout is an invertible matrix over GF(2) mapping (thread,warp,register) bits to tensor-index bits; optimal swizzles found by linear algebra over F₂, maximizing vectorization while minimizing bank conflicts. Primitive: **Shuffle** (F₂-linear). Automated (the strongest automation for the Shuffle primitive; a reviewer from this community will expect it cited next to your Shuffle).

### 4.3 Vertex/graph reordering — Gorder (Wei et al. SIGMOD'16); Rabbit Order (Arai et al. IPDPS'16); RCM (Cuthill–McKee 1969)
A permutation π relabels vertices; new CSR = π applied to indptr/col, and x is gathered under π.
- Gorder maximizes a windowed locality score `F(π)=Σ S(u,v)` over `|π(u)−π(v)|<w`; NP-hard, greedy; high reorder cost amortized only under repeated processing (the layout-state-graph break-even).
Reported: OP2-line GPU work: renumbering cut global read transactions ~19% and raised per-block reuse 2→3.6; Gorder greedy solver; RCM effective on banded/FEM but mismatched to hub-dominated scale-free graphs (degree distribution decides the ordering family). Primitive: **Shuffle** (permutation of the data array + its index array). Your k05, k08, LayoutBench N1/N3.

### 4.4 G-Streamline data reordering vs reference redirection — Zhang et al. ASPLOS'11
For `A[P[i]]`: either **reorder data** (`A' = A[P]`, then access `A'[i]` contiguously) or **redirect references** (change traversal order). Two remaps with different cost/benefit. Primitive: **Shuffle**. Your k06.

### 4.5 Sorted gather-scatter (argsort composition) — QE addusxx_g; SC26 §IV-C
`y[σ(i)] += x[i]` → `y[σπ(i)] += x[π(i)]` with π=argsort(σ) moves the scatter irregularity onto the gather side. Reported: +9–34% on GPUs (CPU minimal — prefetcher-dependent). Primitive: **Shuffle**. Your k06.

### 4.6 Hilbert / Peano space-filling curves — Bader & Zenger 2006; Sulyok et al. JPDC'19
Cell/particle index = curve position; compresses neighbor access distance. Primitive: **Shuffle**. LayoutBench N3, k08 (Morton variant).

---

## Family 5 — AoS / SoA / AoSoA and field grouping (Zip/Unzip)

### 5.1 ASTA / DL in-place AoS→AoSoA — Sung, Liu, Hwu, InPar 2012
- BEFORE (AoS): `addr(i,f) = i*F + f`.
- AFTER (ASTA, tile T): (N/T, F, T), `addr = (i//T)*F*T + f*T + i%T`; in-place marshaling by cycle-following.
Note ASTA fixes coalescing but pure SoA can introduce partition camping — per-level effect. Primitive: **Block + Zip**. Your k03, k09.

### 5.2 Cabana AoSoA — Slattery et al., JOSS 2022 (ECP CoPA)
Particles as tiles `aosoa[tile][field][lane]`, vector length V: `addr(i,f) = (i//V)*F*V + f*V + (i%V)`. Compromise: SoA coalescing + AoS neighbor locality. Primitive: **Block + Zip**. Your k09.

### 5.3 GROMACS SIMD cluster layouts (xyzq packed SoA, 4×N / 2×(N×N)) — Páll & Hess, Comput. Phys. Commun. 184, 2013
The Verlet list is replaced by clusters of j particles sized to the SIMD width; coordinates stored as packed `xyzq` (x,y,z,charge interleaved per cluster) in an SoA-of-clusters so a cluster's coordinates load in whole vectors.
- BEFORE (plain SoA per particle): x[i], y[i], z[i], q[i] separate; a j-cluster of width N gathers 4 strided streams.
- AFTER ([RECONSTRUCTED] xyzq cluster): `xyzq[cluster][4][N]` — for cluster c, `x = xyzq[c,0,:], q = xyzq[c,3,:]`, each a contiguous N-vector; equivalently a Zip of {x,y,z,q} at cluster granularity + Block by cluster.
```python
# N particles -> N/W clusters, 4 fields, W lanes
xyzq = np.ascontiguousarray(np.stack([x,y,z,q]).reshape(4, Nc, W).transpose(1,0,2))
xc = xyzq[c, 0, :]           # contiguous W-vector of x for cluster c
```
Reported: enabled portable SIMD across 14 instruction sets; the cluster-pair algorithm is the basis of GROMACS's CPU+GPU nonbonded performance. Primitive: **Zip + Block**. Manual (hand-designed kernels). A stronger real-world k09 variant (interleave-a-few-fields-at-tile-granularity).

### 5.4 Array regrouping — Zhong et al., PLDI 2004
Merge two arrays co-accessed with 100% affinity into one interleaved array (reduces streams/TLB). `a[i], b[i]` → `ab[i] = (a[i], b[i])`. Primitive: **Zip**. Reference-affinity-driven (the rule behind LayoutBench N2). Your k09 affinity partition.

### 5.5 Cache-conscious structure splitting (hot/cold) — Chilimbi et al., PLDI 1999
Split a struct into hot fields (accessed together, packed) and a cold pointer to the rest. Primitive: **Unzip**. Automated (profile-driven). LayoutBench N2 endpoint.

### 5.6 GraphIt `fuseFields` (AoS/SoA as a schedule directive) — Zhang et al., OOPSLA 2018
Vertex-data vectors tagged AoS or separate-array (SoA) via the `fuseFields` scheduling function; an autotuner searches the schedule space including this layout choice. Primitive: **Zip/Unzip** as a searchable dimension. Automated (autotuner). The clearest precedent for treating Zip/Unzip as a search axis; baseline for LayoutBench N1/N3.

### 5.7 Data-oriented design: Unity DOTS/ECS archetype chunks — Unity docs; Mike Acton CppCon'14
ECS stores entities of one archetype in ~16 KB chunks, each chunk holding per-component SoA arrays = AoSoA at chunk granularity: `chunk[component][entity_in_chunk]`. Mike Acton's canonical examples show AoS→SoA wins on culling/animation/particle loops. Primitive: **Block + Zip/Unzip**. Manual (design methodology). Game-industry k09.

### 5.8 Rust `soa_derive`, Zig `MultiArrayList`, GCC `-fipa-struct-reorg`, DSM (Copeland & Khoshafian 1985)
Language/compiler mechanizations of AoS→SoA (soa_derive generates parallel Vecs; MultiArrayList stores a struct-of-slices; DSM is the 1985 database ancestor). Primitive: **Unzip**. Automated/library.

---

## Family 6 — Domain-specific packed formats (hardware- or library-mandated)

### 6.1 SELL-C-σ — Kreutzer et al., SISM 2014
Sort rows by nnz within σ-windows (Permute), chunk C rows (Block), pad to chunk max (Pad), store column-major within chunk (Permute): `addr(chunk c, lane r, slot s) = chunk_offset[c] + s*C + r`. The α parameter models the x[col] gather locality. Primitive: **Permute+Block+Pad+Permute**. Your k05.

### 6.2 oneDNN `nChw16c` (channel-blocked) — Intel oneDNN docs; Georganas et al. SC'18
Split C = 16·co + ci: `addr(n,co,h,w,ci) = ((((n*C/16+co)*H+h)*W+w)*16+ci)`; 16 channels SIMD-lane-contiguous per pixel. Primitive: **Block**. Your k13.

### 6.3 TensorRT reformat layers (kLINEAR, kCHW4, kCHW16, kCHW32, kHWC8) & cuBLASLt IMMA orders (COL32, COL4_4R2_8C) — NVIDIA docs
Tensor-core int8/fp16 kernels mandate specific interleaved/tiled operand layouts; the framework inserts "reformat" layers to convert. E.g. kCHW32 packs 32 channels innermost; CUBLASLT_ORDER_COL32 stores 32-wide column tiles for IMMA. Primitive: **Block + Permute** (+ value-interleave that partly falls outside the bijective algebra). Automated (framework-inserted). Reinforces the "library-mandated layout" note next to AMX in your Table II; k10/k13 adjacent.

### 6.4 Hopper TMA swizzle modes (32B/64B/128B) — NVIDIA docs/blogs
The TMA descriptor carries a swizzle mode so the DMA engine writes shared memory in a bank-conflict-free interleaving as it copies — layout as a first-class DMA/ISA concept. Primitive: **Shuffle** (hardware). Manual (descriptor config).

### 6.5 AMD MFMA / LDS layouts — ROCm/GPUOpen blogs
Matrix-core (MFMA) instructions require specific register/LDS data layouts; LDS bank-conflict avoidance mirrors the CUDA shared-memory case. Primitive: **Block + Shuffle**. Manual/library. AMD counterpart for MI300A validation.

### 6.6 Interleaved batched small solvers — László, Giles, Appleyard, ACM TOMS 2016; MAGMA batched [FLAG: verify MAGMA specifics]
Batched Thomas/GEMM store system k of all batches contiguous (interleaved) for coalescing: `addr(k,batch) = k*NB + batch`. Primitive: **Permute**. Your k11. (MAGMA interleaved/strided-batch numbers not verified this sweep.)

---

## Family 7 — Joint layout+schedule and layout propagation frameworks

### 7.1 Integrated loop+layout (polyhedral) — Shirako & Sarkar LCPC'22; Shirako et al. IMPACT'19; Kandemir integrated framework
Joint affine layout+schedule; the best layout depends on the intra-tile loop permutation, so phase-ordering (schedule-then-layout) misses optima. Primitive: all. Automated. The theory anchor for "layouts are global"; LayoutBench §1.

### 7.2 ALT — Xu et al., EuroSys 2023
Graph-level layout propagation with operator rewrites and transpose elimination for DNN compilation. Primitive: Permute/Block. Automated (cost-model-driven).

### 7.3 WACO — Won et al., ASPLOS 2023
Co-optimizes sparse **format** and **schedule** with a learned (sparse-CNN) cost model over the sparsity pattern + ANNS search; 1.14–1.43× over MKL/TACO/ASpT across 726 patterns. Primitive: format (layout) + schedule jointly. Automated (learned). The learned-cost-model alternative; LayoutBench N1 protocol precedent.

### 7.4 Accelerator mapping+dataplacement — Timeloop (ISPASS'19), MAESTRO (MICRO'19), ZigZag (TC'21), TCM (arXiv 2602.15172), CMDS (arXiv 2406.14574)
Mapspace = tilings × loop orders × spatial partitionings under an analytical cost model. Timeloop's Fig. 1: 6,582 mappings share minimal DRAM traffic yet vary 11× in energy (volume-optimal ≠ optimal). TCM's "dataplacement" prunes the mapspace 10³⁷→10⁵ while preserving optimality (full optimal search in 17 s). CMDS does cross-layer data-layout optimization with a fine-grained inter-layer cost model. Primitive: Block/Permute at accelerator scale. Automated. Anchors the optimality plan and the Maia path-(i) proposal.

---

## Family 8 — Algorithmic / structure layouts (search trees, databases)

### 8.1 Eytzinger (BFS) array layout for binary search — Khuong & Morin, ACM JEA 2017; Algorithmica blog (Slotin)
Store a sorted array in heap/BFS order so the root and its likely-next comparisons are cache-local.
- BEFORE (sorted): `A_sorted`, binary search hits addresses N/2, N/4… scattered across cache lines.
- AFTER (Eytzinger): `E[1]=root`, children of k at `2k` and `2k+1`; search walks `k = 2k + (x > E[k])`, branch-free with prefetch.
```python
def eytzinger(sorted_arr):
    n = len(sorted_arr); E = np.zeros(n+1, dtype=sorted_arr.dtype); i = 0
    def rec(k):
        nonlocal i
        if k <= n:
            rec(2*k); E[k] = sorted_arr[i]; i += 1; rec(2*k+1)
    rec(1); return E
def search(E, x):
    k = 1
    while k < len(E):
        k = 2*k + (x > E[k])          # branch-free descent
    return k
```
Reported: for large n the branch-free Eytzinger + prefetch is the fastest general layout/search combination on a wide range of CPUs (overturning the SODA'03 conclusion that B-tree/vEB beat it); for small n, sorted+binary search still wins. Primitive: **Shuffle** (a fixed permutation of a sorted array). Manual. NumPy-portable → excellent new benchmark candidate (k14).

### 8.2 Implicit B-tree & van Emde Boas layouts — Khuong & Morin 2017; Bender–Demaine–Farach-Colton
B-tree layout: (B+1)-ary generalization of Eytzinger (block of B keys per node). vEB: recursive √n split for cache-obliviousness. Primitive: **Block + Shuffle**. Manual. B-tree layout is the natural k14 extension.

### 8.3 FAST / CSB+-tree SIMD-blocked search trees — Kim et al. SIGMOD'10; Rao & Ross SIGMOD'00
Tree nodes laid out to match SIMD width and cache-line/page hierarchy (hierarchical blocking: SIMD block ⊂ cache-line block ⊂ page block). Primitive: **Block + Shuffle**. Manual.

### 8.4 Row→column stores (DSM/C-Store/columnar) — Copeland 1985; Stonebraker et al. VLDB'05
Store each attribute contiguously so scans touch only needed columns. `row[i].f` → `col_f[i]`. Primitive: **Unzip** (whole-table). The database instance of SoA; PAX (Ailamaki VLDB'01) is the page-internal Block+Unzip variant.

### 8.5 Dremel record shredding (nested→columnar striping) — Melnik et al. VLDB'10; Parquet/ORC/Arrow
Nested records shredded to per-leaf-path columns + repetition/definition levels that encode structure for reassembly. Arrow's alternative: validity bitmap + offsets buffers. This is a layout transform on nested data, but reassembly needs the extra level streams → partly **outside** the bijective five-primitive algebra (it encodes structure, like compression), worth citing as a scope boundary alongside im2col/CSR.
Primitive: **Unzip** + auxiliary level streams (flag: not purely bijective). Automated.

---

## Synthesis: every entry → five-primitive algebra

| Entry | Pad | Permute | Block | Shuffle | Zip/Unzip | Automated? |
|---|---|---|---|---|---|---|
| Rivera–Tseng, Hong padding | ✓ | | | | | analytic |
| Harris SMEM +1 pad | ✓ | | | (alt) | | manual |
| Kandemir/O'Boyle matrices | | ✓ | | (skew) | | ✓ unimodular |
| PyTorch channels_last | | ✓ | | | | ✓ framework |
| Kokkos LayoutLeft/Right | | ✓ | | | | ✓ by mem-space |
| XLA layout assignment | ✓ | ✓ | | | | ✓ propagation |
| Chatterjee 4D/Morton | | | ✓ | (Morton) | | both |
| YASK vector folding | | ✓ | ✓ | | | ✓ DSL+GA |
| DLT (Henretty) | | ✓ | ✓ | | | ✓ |
| IREE mmt4d / tensor.pack | | ✓ | ✓ | | | ✓ pass |
| CUTLASS XOR / Triton F₂ | | | | ✓ | | manual / ✓ |
| Gorder/Rabbit/RCM | | | | ✓ | | ✓ (NP-hard) |
| G-Streamline / sorted GAS | | | | ✓ | | both |
| Hilbert/Peano/Morton mesh | | | (✓) | ✓ | | both |
| ASTA / Cabana AoSoA | ✓ | | ✓ | | ✓ | both |
| GROMACS xyzq clusters | | | ✓ | | ✓ | manual |
| Zhong regrouping | | | | | ✓ (Zip) | ✓ affinity |
| Chilimbi hot/cold | | | | | ✓ (Unzip) | ✓ profile |
| GraphIt fuseFields | | | | | ✓ | ✓ autotuner |
| Unity ECS chunks | | | ✓ | | ✓ | manual |
| SELL-C-σ | ✓ | ✓ | ✓ | | | manual/auto |
| oneDNN nChw16c | | | ✓ | | | ✓ |
| TensorRT/cuBLASLt orders | | ✓ | ✓ | (interleave) | | ✓ framework |
| TMA / MFMA / LDS swizzle | | | ✓ | ✓ | | manual |
| Interleaved batched solver | | ✓ | | | | manual |
| Shirako–Sarkar / ALT / WACO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ joint |
| Timeloop/TCM/CMDS | | ✓ | ✓ | | | ✓ mapspace |
| Eytzinger/B-tree/vEB | | | ✓ | ✓ | | manual |
| Row→column / Dremel | | | ✓ | | ✓ | ✓ |

## Gap analysis — outside the bijective five-primitive algebra
- **Duplication**: im2col, overlapped tiling (ALT), halo replication — expand data; not bijective.
- **Compression / structure encoding**: CSR/ELL value-vs-index split, Dremel repetition/definition levels, Arrow validity bitmaps, sparse formats — carry auxiliary structure streams.
- **Value repacking**: VNNI/IMMA 4-element interleave and int8 tensor-core orders partly repack values, not just permute indices.
- **Dropping**: hot/cold splitting that *discards* cold fields from the hot working set (Unzip captures the split, not the eviction).
These match the SC26 stated scope exclusions; citing Dremel/Arrow and TensorRT orders sharpens the boundary.

## Best new NumPy benchmark candidates surfaced this sweep
1. **Eytzinger vs sorted binary search (k14)** — pure NumPy, CPU-visible, attributable (Khuong–Morin); the branch-free descent vectorizes over a batch of queries.
2. **GROMACS xyzq cluster gather (k09 variant)** — interleave-4-fields-at-cluster-granularity vs full SoA.
3. **Row→column scan selectivity (k15)** — column store vs row store under varying #columns-touched (DSM/C-Store).
4. **Kokkos-style device-flip (k04/k07 framing)** — same kernel, LayoutRight vs LayoutLeft, showing the crossover.
5. **channels_last propagation with a "thrash" op (k13 variant)** — measure the conflict cost when one op forces a round-trip.
