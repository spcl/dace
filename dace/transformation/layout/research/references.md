# References

Numbered bibliography mirroring `references.bib`, same order. Each entry tags which of the
four source reports cites it (`r1`=report1 cost-models/optimality, `r2`=report2 survey,
`r3`=report3 provenance/bugs, `gap`=coverage_gap_addendum). Entries the sources name without a
full title carry `[bracketed]` descriptive titles and a verification note — not verbatim.

## 0. Target paper

1. "Why Schedule Transformations Are Not Enough: Layout Optimizations for Block-Granular I/O." SC26 (submission, authors omitted, under review). — r1, r2, r3

## 1. Cost models — LogP / message-based family

2. Culler et al. LogP: Towards a Realistic Model of Parallel Computation. PPoPP 1993. — r1
3. Alexandrov, Ionescu, Schauser, Scheiman. LogGP: Incorporating Long Messages into the LogP Model. SPAA 1995. — r1
4. Ino, Fujimoto, Hagihara. LogGPS: A Parallel Computational Model for Synchronization Analysis. PPoPP 2001. — r1
5. Hoefler, Schneider, Lumsdaine. LogGOPSim: Simulating Large-Scale Applications in the LogGOPS Model. HPDC 2010. — r1
6. Kielmann, Bal, Verstoep. Fast Measurement of LogP Parameters for Message Passing Platforms. IPDPS Workshops 2000. — r1
7. Cameron, Sun. Quantifying Locality Effect in Data Access Delay: Memory logP. IPDPS 2003. — r1
8. Altaf, Wood. LogCA: A High-Level Performance Model for Hardware Accelerators. ISCA 2017. — r1

## 2. Cost models — memory/cache-hierarchy analytical models

9. Stengel, Treibig, Hager, Wellein. Quantifying Performance Bottlenecks of Stencil Computations Using the Execution-Cache-Memory Model. ICS 2015. (ECM) — r1
10. Williams, Waterman, Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. CACM 52(4), 2009. — r1
11. Ilic, Pratas, Sousa. Cache-Aware Roofline Model: Upgrading the Loft. IEEE CAL 13(1), 2014. (CARM) — r1
12. Gysi, Grosser, Hoefler. A Fast Analytical Model of Fully Associative Caches. PLDI 2019. (HAYSTACK) — r1
13. Bao, Krishnamoorthy, Pouchet, Sadayappan. Analytical Modeling of Cache Behavior for Affine Programs. POPL 2018. — r1
14. Hong, Kim. An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness. ISCA 2009. (MWP-CWP) — r1
15. Huang, Lee, Kim, Lee. GPUMech: GPU Performance Modeling Technique Based on Interval Analysis. MICRO 2014. — r1
16. Wang, Jahre, Adileh, Eeckhout. MDM: The GPU Memory Divergence Model. MICRO 2020. — r1
17. GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs. ISCA 2022. Authors unconfirmed this sweep; report1 cites venue as ISPASS, corrected here to ISCA'22. — r1
18. Volkov. Understanding Latency Hiding on GPUs. PhD dissertation, UC Berkeley, 2016. — r1

## 3. Cost models — I/O-complexity / pebble-game models

19. Hong, Kung. I/O Complexity: The Red-Blue Pebble Game. STOC 1981. — r1
20. Demaine, Liu. Red-Blue Pebble Game: Complexity of Computing the Trade-Off Between Cache Size and Memory Transfers. SPAA 2018. — r1
21. Aggarwal, Vitter. The Input/Output Complexity of Sorting and Related Problems. CACM 31(9), 1988. — r1
22. Savage. Extending the Hong-Kung Model to Memory Hierarchies. COCOON 1995. (red-blue-white) — r1
23. Olivry, Langou, Pouchet, Sadayappan, Rastello. Automated Derivation of Parametric Data Movement Lower Bounds for Affine Programs. PLDI 2020. (IOLB) — r1
24. Olivry et al. IOOpt: Automatic Derivation of I/O Complexity Bounds for Affine Programs. PLDI 2021. Full author list unconfirmed this sweep. — r1
25. Kwasniewski, Kabić, Besta, VandeVondele, Solcà, Hoefler. Red-Blue Pebbling Revisited: Near Optimal Parallel Matrix-Matrix Multiplication. SC 2019. (COSMA) — r1
26. Kwasniewski et al. Pebbles, Graphs, and a Pinch of Combinatorics: Towards Tight I/O Lower Bounds for Statically Analyzable Programs. SPAA 2021. (SOAP) Full author list unconfirmed this sweep. — r1
27. Böhnlein, Papp, Yzelman. Red-Blue Pebbling with Multiple Processors: Time, Communication and Memory Trade-Offs. SIROCCO 2025. — r1
28. [Disaggregated-memory pebble games]. MEMSYS 2025. Authors not stated in source; descriptive title. — r1

## 4. Cost models — layout-parametric sparse traffic models

29. Kreutzer, Hager, Wellein, Fehske, Bishop. A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiplication on Modern Processors with Wide SIMD Units. SIAM J. Sci. Comput. 36(5), 2014. (SELL-C-σ) — r1, r2 (§6.1), gap
30. Vuduc, Demmel, Yelick, Kamil, Nishtala, Lee. Performance Optimizations and Bounds for Sparse Matrix-Vector Multiply. SC 2002. — r1, gap
31. Im, Yelick, Vuduc. SPARSITY: Optimization Framework for Sparse Matrix Kernels. IJHPCA 18(1), 2004. — gap
32. Vuduc, Demmel, Yelick. OSKI: A Library of Automatically Tuned Sparse Matrix Kernels. SciDAC / J. Phys. Conf. Ser. 16, 2005. — r1, gap
33. Choi, Singh, Vuduc. Model-Driven Autotuning of Sparse Matrix-Vector Multiply on GPUs. PPoPP 2010. — gap

## 5. Cost models — accelerator mapping / dataplacement

34. Parashar et al. Timeloop: A Systematic Approach to DNN Accelerator Evaluation. ISPASS 2019. — r1, r2 (§7.4), gap
35. Kwon, Chatarasi, Pellauer, Parashar, Sarkar, Krishna. Understanding Reuse, Performance, and Hardware Cost of DNN Dataflow: A Data-Centric Approach. MICRO 2019 (also IEEE Micro Top Picks 2020). (MAESTRO) — r1, r2 (§7.4), gap
36. Mei, Houshmand, Jain, Giraldo, Verhelst. ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators. IEEE Trans. Computers 70(8), 2021. — r1, r2 (§7.4), gap
37. The Turbo-Charged Mapper: Fast and Optimal Mapping for Energy-Efficient and Low-Latency Accelerator Design. arXiv:2602.15172, 2026. (TCM) Authors not named in source. — r1, r2 (§7.4), gap
38. Shi, Colleman, VanDeMieroop, Joseph, Meijer, Dehaene, Verhelst. CMDS: Cross-Layer Dataflow Optimization for DNN Accelerators Exploiting Multi-Bank Memories. arXiv:2406.14574, 2024. — r1, r2 (§7.4), gap

## 6. Cost models — learned cost models

39. Won, Mendis, Emer, Amarasinghe. WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program. ASPLOS 2023. — r1, r2 (§7.3), gap
40. Kaufman, Phothilimthana, Zhou, Mendis, Roy, Sabne, Burrows. A Learned Performance Model for Tensor Processing Units. MLSys 2021. — r1, gap
41. Phothilimthana et al. TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs. NeurIPS 2023 D&B; arXiv:2308.13490. — r1, r2 (§2.4), gap
42. Ahrens (Peter Ahrens), Kjolstad, Amarasinghe. Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model. PLDI 2022. — gap
43. Ragan-Kelley, Barnes, Adams, Paris, Durand, Amarasinghe. Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines. PLDI 2013. — r1
44. Chen et al. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. OSDI 2018. — r1

## 7. Hardness results / optimality theory

45. Petrank, Rawitz. The Hardness of Cache Conscious Data Placement. POPL 2002. — r1
46. [Wu et al. GPU data repositioning is NP-hard.] Not independently located this sweep; do not treat as verbatim. — r1
47. Kremer. NP-Completeness of Dynamic Remapping. Workshop on Compilers for Parallel Computers, Delft, 1993. — r1
48. Garey, Johnson, Stockmeyer. Some Simplified NP-Complete Graph Problems. Theoretical Computer Science 1(3), 1976. (minimum linear arrangement NP-hardness) — r1
49. Kennedy, Kremer. Automatic Data Layout for Distributed-Memory Machines. ACM TOPLAS 20(4), 1998. (the 0-1 ILP) — r1

## 8. Inspector-executor / runtime reordering

50. Ding, Kennedy. Improving Cache Performance in Dynamic Applications Through Data and Computation Reorganization at Run Time. PLDI 1999. — r1, r3
51. Mellor-Crummey, Whalley, Kennedy. Improving Memory Hierarchy Performance for Irregular Applications. IJPP 29(3), 2001. — r1, r3

## 9. Layout transformations — padding

52. Rivera, Tseng. Data Transformations for Eliminating Conflict Misses. PLDI 1998. — r2, r3
53. Hong, Sadayappan, et al. Effective Padding of Multi-Dimensional Arrays to Avoid Cache Conflict Misses. PLDI 2016. (PAdvisor) Full author list unconfirmed this sweep. — r2, r3
54. Harris. An Efficient Matrix Transpose in CUDA C/C++. NVIDIA Technical Blog, 2013. — r2
55. Mao (Lei Mao). CUDA shared-memory bank-conflict reproduction (padding vs. XOR swizzle). Blog. — r2
56. JEP 142: Reduce Cache Contention on Specified Fields (`@Contended`). OpenJDK. — r2
57. Linux Kernel Documentation. False Sharing. docs.kernel.org/kernel-hacking/false-sharing.html. — r2, r3

## 10. Layout transformations — permutation / transposition

58. Kandemir, Ramanujam, Choudhary, Banerjee. Improving Locality Using Loop and Data Transformations in an Integrated Framework. MICRO 1998 (journal version JPDC 1999). — r2
59. O'Boyle, Knijnenburg. Nonsingular Data Transformations: Definition, Validity, Applications. IJPP 27(3), 1999. — r2
60. PyTorch. Channels Last Memory Format in PyTorch. Engineering blog, 2021. — r2
61. Trott et al. Kokkos 3: Programming Model Extensions for the Exascale Era. IEEE TPDS 33(4), 2022. — r2
62. OpenXLA. XLA: GPU Architecture — Layout Assignment. Project documentation. — r2, gap
63. Jiang, Chen, Hechtman, Zhang, Mu. Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU. arXiv:2604.15464, 2026. — r2, gap

## 11. Layout transformations — blocking / tiling / Morton

64. Chatterjee, Jain, Lebeck, Mundhra, Thottethodi. Nonlinear Array Layouts for Hierarchical Memory Systems. ICS 1999. — r2
65. Yount. Vector Folding: Improving Stencil Performance via Multi-Dimensional SIMD-Vector Representation. HPCC 2015. — r2, gap
66. Yount, Tobin, Breuer, Duran. YASK — Yet Another Stencil Kernel. WOLFHPC@SC 2016. — r2, gap
67. Henretty, Stock, Pouchet, Franchetti, Ramanujam, Sadayappan. Data Layout Transformation for Stencil Computations on Short-Vector SIMD. CC 2011. (DLT) — r2, r3
68. IREE project blog. Data Tiling in IREE (`linalg.mmt4d`, `tensor.pack`/`unpack`). — r2
69. Vulkan / GPU vendor documentation. VK_IMAGE_TILING_OPTIMAL (opaque Morton/Z-order texture tiling). — r2

## 12. Layout transformations — shuffles / swizzles / reordering

70. NVIDIA CUTLASS documentation. CUTLASS/CuTe `Swizzle<B,M,S>`. — r2
71. Gonzalez et al. [Bank-conflict-free XOR-based memory interleaving.] ICS 1997. Descriptive title, not verified verbatim. — r2
72. Rau. Pseudo-Randomly Interleaved Memory. ISCA 1991. — r2
73. Zhou, Lezcano, Goucher, Rakhmati, Niu, Lebar, Szczerbuk, Bell, Tillet, Raoux, Moudallal. Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using 𝔽₂. arXiv:2505.23819, 2025 (rev. 2026; also ASPLOS 2026). — r2, r3
74. Wei, Yu, Lu, Lin. Speedup Graph Processing by Graph Ordering. SIGMOD 2016. (Gorder) — r2
75. Arai, Shiokawa, Yamamuro, Onizuka, Iwamura. Rabbit Order: Just-in-Time Parallel Reordering for Fast Graph Analysis. IPDPS 2016. — r2
76. Cuthill, McKee. Reducing the Bandwidth of Sparse Symmetric Matrices. Proc. 1969 24th ACM National Conference. (RCM) report3's own sweep flags the original as referenced-but-not-independently-verified; kept here on report2's direct citation. — r2
77. Zhang, Jiang, Guo, Tian, Shen. On-the-Fly Elimination of Dynamic Irregularities for GPU Computing. ASPLOS 2011. (G-Streamline) — r2
78. Bader, Zenger. Cache Oblivious Matrix Multiplication Using an Element Ordering Based on a Peano Curve. Linear Algebra and its Applications, 2006. — r2
79. Sulyok et al. [Space-filling-curve mesh/particle ordering.] JPDC 2019. Descriptive title, not verified verbatim. — r2

## 13. Layout transformations — AoS/SoA/AoSoA, Zip/Unzip

80. Sung, Liu, Hwu. DL: A Data Layout Transformation System for Heterogeneous Computing. InPar 2012. (ASTA) — r2
81. Slattery et al. Cabana: A Performance Portable Library for Particle-Based Simulations. JOSS 7(70), 2022. — r2
82. Páll, Hess. A Flexible Algorithm for Calculating Pair Interactions on SIMD Architectures. Computer Physics Communications 184(12), 2013. (GROMACS) — r2
83. Zhong, Orlovich, Shen, Ding. Array Regrouping and Structure Splitting Using Whole-Program Reference Affinity. PLDI 2004. — r2
84. Chilimbi, Hill, Larus. Cache-Conscious Structure Layout. PLDI 1999. — r2, r3
85. Zhang, Yang, Baghdadi, Kamil, Shun, Amarasinghe. GraphIt: A High-Performance Graph DSL. OOPSLA 2018. — r2, gap
86. Unity Technologies. Unity Data-Oriented Technology Stack (DOTS) / Entities. Documentation. — r2
87. Acton. Data-Oriented Design and C++. CppCon 2014. — r2
88. `soa_derive`. Rust crate. — r2
89. `std.MultiArrayList`. Zig standard library documentation. — r2
90. `-fipa-struct-reorg`. GCC compiler documentation. — r2
91. Copeland, Khoshafian. A Decomposition Storage Model. SIGMOD 1985. (DSM) — r2

## 14. Domain-specific packed formats

92. Georganas, Avancha, Banerjee, Kalamkar, Henry, Pabst, Heinecke. Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures. SC 2018. — r2
93. Intel. oneDNN Memory Format Propagation. Developer documentation. — r2
94. NVIDIA. TensorRT Developer Guide (reformat layers, tensor formats). — r2
95. NVIDIA. cuBLASLt Library (IMMA orders). — r2
96. NVIDIA. Hopper Architecture: TMA Swizzle Modes. Technical blog/whitepaper. — r2
97. AMD. Matrix Cores (MFMA) / LDS Layouts. ROCm/GPUOpen technical blog. — r2
98. László, Giles, Appleyard. Manycore Algorithms for Batch Scalar and Block Tridiagonal Solvers. ACM TOMS 42(4), 2016. — r2
99. Abdelfattah et al. MAGMA batched linear algebra (interleaved/strided-batch layouts). Flagged unverified (MAGMA specifics not checked). — r2, gap

## 15. Joint layout+schedule / layout-propagation frameworks

100. Shirako, Sarkar. [Integrated polyhedral loop and data-layout optimization.] LCPC 2022. Descriptive title, not verified verbatim. — r2
101. Shirako et al. [Integrated loop and data-layout transformation.] IMPACT 2019. Descriptive title, not verified verbatim. — r2
102. Xu et al. ALT: Breaking the Wall Between Data Layout and Loop Optimizations. EuroSys 2023. — r2, r3
103. Ben-Nun, Ates, Calotoiu, Hoefler. Bridging Control-Centric and Data-Centric Optimization. CGO 2023. (DaCe) — r3
104. Kjolstad, Kamil, Chou, Lugato, Amarasinghe. The Tensor Algebra Compiler. OOPSLA 2017. (taco) — r3
105. Chou, Kjolstad, Amarasinghe. Format Abstraction for Sparse Tensor Algebra Compilers. OOPSLA 2018. — r3
106. Liu et al. UniSparse: An Intermediate Language for General Sparse Format Customization. OOPSLA 2024. Full author list unconfirmed this sweep. — r3

## 16. Algorithmic / structure layouts — search trees, databases

107. Khuong, Morin. Array Layouts for Comparison-Based Searching. ACM JEA 22, 2017 (arXiv:1509.05053). (Eytzinger, B-tree, van Emde Boas) — r2
108. Slotin. Eytzinger Binary Search. Algorithmica (blog/book). — r2
109. Bender, Demaine, Farach-Colton. Cache-Oblivious B-Trees. FOCS 2000. — r2
110. Kim, Chhugani, Satish, Sedlar, Nguyen, Kaldewey, Lee, Brandt, Dubey. FAST: Fast Architecture Sensitive Tree Search on Modern CPUs and GPUs. SIGMOD 2010. — r2
111. Rao, Ross. Making B+-Trees Cache Conscious in Main Memory. SIGMOD 2000. (CSB+-tree) — r2
112. Stonebraker et al. C-Store: A Column-Oriented DBMS. VLDB 2005. — r2
113. Ailamaki, DeWitt, Hill, Skounakis. Weaving Relations for Cache Performance. VLDB 2001. (PAX) — r2
114. Melnik, Gubarev, Long, Romer, Shivakumar, Tolton, Vassilakis. Dremel: Interactive Analysis of Web-Scale Datasets. VLDB 2010. — r2

## 17. Foundational layout papers — pre-2000 lineage

115. Lam, Rothberg, Wolf. The Cache Performance and Optimizations of Blocked Algorithms. ASPLOS 1991. — r3
116. Cierniak, Li. Unifying Data and Control Transformations for DSM Machines. PLDI 1995. — r3
117. Anderson, Amarasinghe, Lam. Data and Computation Transformations for Multiprocessors. PPoPP 1995. — r3
118. Coleman, McKinley. Tile Size Selection Using Cache Organization and Data Layout. PLDI 1995. — r3
119. Kandemir, Choudhary, Ramanujam, Banerjee. Improving Cache Locality by a Combination of Loop and Data Transformations. IEEE TC 48(2), 1999. (a distinct paper from #58, same core author team) — r2 (§7.1, as "Kandemir integrated framework"), r3
120. Chatterjee, Lebeck, Patnala, Thottethodi. Recursive Array Layouts and Fast Parallel Matrix Multiplication. SPAA 1999 / IEEE TPDS 13(11), 2002. — r3
121. Che, Sheaffer, Skadron. Dymaxion: Optimizing Memory Access Patterns for Heterogeneous Systems. SC 2011. — r3
122. Reinders, Jeffers, et al. Vectorization with SDLT (chapter). In Intel Xeon Phi Processor High Performance Programming, 2nd ed., Morgan Kaufmann/Elsevier, 2016. The citable artifact for "Intel SDLT" — report3 found no independent SDLT paper. — r3

## 18. Recent layout papers, 2023-2026

123. Zhang, Ding, Sun, Hu, Shpeisman, Pekhimenko. Hexcute: Automating Layout Synthesis in GPU Programs. arXiv:2504.16214, 2025. Published forms carry slightly different titles (see .bib note). — r3
124. Carlisle, Shah, Stern, VanKoughnett. Categorical Foundations for CuTe Layouts. arXiv:2601.05972, 2026. — r3
125. Swatman et al. Evolutionary Algorithms to Find Cache-Friendly Generalized Morton Layouts. arXiv:2309.07002, 2023. — r3
126. Singhal, Koparkar, Zullo, Pelenitsyn, Vollmer, Rainey, Newton, Kulkarni. Optimizing Layout of Recursive Datatypes with Marmoset. ECOOP 2024 (arXiv:2405.17590). — r3
127. Singhal et al. [SoCal: whole-program ADT layout, SoA for tree-shaped data.] arXiv:2605.01140, 2026. Descriptive title, not verified verbatim. — r3
128. Radtke, Weinzierl. [Annotation-guided AoS→SoA transformation with GPU offload.] PPAM 2024; Concurrency and Computation: Practice and Experience, 2025; arXiv:2512.05516. Descriptive title, not verified verbatim; reports 2.6x on GH200. — r3
129. Idreos, Zoumpatianos, Hentschel, Kester, Guo. The Data Calculator: Data Structure Design and Cost Synthesis from First Principles and Learned Cost Models. SIGMOD 2018. Flagged unverified in coverage_gap_addendum. — gap

## 19. Performance-bug empirical studies

130. Jin, Song, Shi, Scherpelz, Lu. Understanding and Detecting Real-World Performance Bugs. PLDI 2012. — r3
131. Zaman, Adams, Hassan. A Qualitative Study on Performance Bugs. MSR 2012. — r3
132. Nistor, Jiang, Tan. Discovering, Reporting, and Fixing Performance Bugs. MSR 2013. — r3
133. Han, Yu. An Empirical Study on Performance Bugs for Highly Configurable Software Systems. ESEM 2016. — r3
134. Song, Lu. Statistical Debugging for Real-World Performance Problems. OOPSLA 2014. Verification status unconfirmed (⚠️) in report3. — r3
135. Selakovic, Pradel. Performance Issues and Optimizations in JavaScript: An Empirical Study. ICSE 2016. — r3
136. Liu, Xu, Cheung. Characterizing and Detecting Performance Bugs for Smartphone Applications. ICSE 2014. — r3
137. Mazuera-Rozo, Bautista-Mora, Linares-Vásquez, Rueda, Bavota. Types and Survivability of Performance Bugs in Mobile Apps. Empirical Software Engineering 25(3), 2020. — r3
138. Zhao et al. Large-Scale Empirical Study of Real-Life Performance Issues. IEEE TSE, 2022. Full author list unconfirmed this sweep. — r3
139. Azad, Iqbal, Hassan, Roy. An Empirical Study of HPC Performance Bugs. MSR 2023. — r3
140. Makkouk, Kim, Chen. An Empirical Study of Performance Bugs in Deep Learning Frameworks. ICSME 2022. — r3
141. Cao et al. Understanding Performance Problems in Deep Learning Systems. ESEC/FSE 2022. — r3
142. Liao et al. Android Performance Issues in Real-World Apps vs. Literature. TOSEM 2025. — r3
143. Bi et al. Understanding Performance Problems in CUDA Programs. PACMSE/FSE 2026. Taxonomy not accessible this sweep. — r3
144. Muse, Nafi, Khomh, Antoniol. Data-Access Performance Anti-Patterns in Data-Intensive Systems. Empirical Software Engineering 29(144), 2024 (arXiv:2208.08918 is a registered-report v1). — r3
145. Yi, Ding, Shi, Gligoric. Understanding and Finding JIT Compiler Performance Bugs. PACMPL OOPSLA1, 2026. (jittery; 0 layout hits) — r3
146. Rathnasuriya et al. Real-World Bugs in Tile Programs. 2026. A correctness taxonomy, not a performance one; 35/301 bugs are indexing/stride/layout-transformation related. — r3
147. Zhou, Ren, Gao, Jiang. An Empirical Study of Optimization Bugs in GCC and LLVM. Journal of Systems and Software 174, 2021. Report3 cites as JSS'20; publication year confirmed 2021 this sweep. — r3
148. Theodoridis, Grosser, Su. Understanding and Exploiting Optimal Function Inlining. ASPLOS 2022 (Best Paper). — r3
149. Zhang, Yi. Detection of Optimizations Missed by the Compiler. ESEC/FSE 2023. (MOD) — r3
150. Garg et al. PerfBench: Can Agents Resolve Real-World Performance Bugs? arXiv:2509.24091, 2025. — r3
151. Yi, Gay, Leitner. Do AI Models Dream of Faster Code? arXiv:2510.15494, 2025. — r3

## 20. Real-world layout bugs — repos, issues, commits

152. No channels-last RoIAlign CPU kernel; Conv2d reorder thrash. GitHub issue pytorch/vision#6619. RoIAlign 82.6s→2.3s (~36x); fixed. — r3
153. Per-layer schedule tuning injects a layout transform between every pair of layers. GitHub issue apache/tvm#1585. ResNet50 v1 +29%, SSD-ResNet50 +35%; fixed via graph tuner. — r3
154. `pcpu_chunk`: hot-read `base_addr` shares a line with hot-write `free_bytes`/`chunk_md`. Linux commit 3a6358c0dbe6. +24% at 160-way parallelism; merged v6.5-rc1. — r3
155. `page_counter`: hot-write `usage` adjacent to hot-read `parent`. Linux commit 802f1d522d5f. +8.9%/+9.9%/+4.5% across benchmarks; merged. — r3
156. SPSC `readIndex_`/`writeIndex_` co-location causes line ping-pong. GitHub PR facebook/folly#378. +10% ops/ms, L1 store misses -95%; merged. — r3
157. `__rmatmul__` transposes internally, giving an inefficient access pattern for transposed data. GitHub issue scipy/scipy#13211. 846ms vs. 70.5μs (~1e5x); open. — r3
158. `np.dot` on strided views misses the BLAS fast path. GitHub issue numpy/numpy#19650. 1.59s→8.06ms (~197x); open. — r3
159. Channels-last propagates through autograd, forcing slow backward paths. GitHub issue pytorch/pytorch#37142. CPU 5.07→21.5s; GPU 0.665→3.31s; open. — r3
160. Incomplete channels-last op coverage treats NHWC input as non-contiguous NCHW. GitHub issue pytorch/pytorch#50036. NCHW 0.97s vs. NHWC 2.35s; open. — r3
161. SD 2.1 VAE decoder degenerates to a transpose/conv/reshape chain. GitHub issue microsoft/onnxruntime#18128. No numbers given; closed, not planned. — r3
162. Epilogue shared-memory pad [0,16] instead of [0,8] drops 128-bit access to `st.shared.u32`. GitHub discussion NVIDIA/cutlass#281. No numbers in thread; partly resolved. — r3
163. Struct field reordering (least- to most-aligned) to eliminate padding holes. GitHub PR rust-lang/rust#37429. Merged then disabled by #38523; reverted. — r3

## 21. Benchmark suites

164. McCalpin. Memory Bandwidth and Machine Balance in Current High Performance Computers. IEEE TCCA Newsletter, 1995. (STREAM) — r3
165. UoB-HPC. BabelStream. GitHub repository. — r3
166. Pouchet. PolyBench/C: The Polyhedral Benchmark Suite. Software. — r3
167. Lai, Lin, Gokhale, Peng, Patel, Lee. RISC-V Vectorization Coverage for HPC: A TSVC-Based Analysis. SC'25 Workshops. — r3
168. Che, Boyer, Meng, Tarjan, Sheaffer, Lee, Skadron. Rodinia: A Benchmark Suite for Heterogeneous Computing. IISWC 2009. — r3
169. Davis, Hu. The University of Florida Sparse Matrix Collection. ACM TOMS 38(1), 2011. (now SuiteSparse) — r3
170. Kobzol. hardware-effects-gpu: bank-conflicts. GitHub repository. Offset=1→0 conflicts/95.74% eff.; offset=32→~310000 conflicts/3.14% eff. — r3
171. PerfCurator: Curating a Large-Scale Dataset of Performance Bug-Related Commits from Public Repositories. arXiv:2406.11731, 2024. Authors unconfirmed this sweep. — r3
