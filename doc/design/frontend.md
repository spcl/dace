# Design Document: Modular Frontend Architecture

## Executive Summary

This document outlines a design for refactoring DaCe's frontend architecture to be more portable, verifiable, and modular. The proposed architecture introduces Schedule Tree as a common intermediate representation (IR) shared across language frontends, structured as a multi-pass pipeline using DaCe's existing Pass Pipeline infrastructure.

### Overview
This document proposes a new modular architecture for DaCe frontends using Schedule Tree as an intermediate representation with a structured multi-pass pipeline.

### Current Problems
- **Code Duplication**: Each frontend reimplements similar AST-to-SDFG conversion logic
- **Maintainability**: Bug fixes must be replicated across all frontends
- **Limited Optimization**: No shared high-level optimization infrastructure
- **Verification Difficulty**: Direct AST-to-SDFG conversion is hard to verify

### Proposed Solution

Four-Pass Pipeline Architecture:

1. **Pass 1**: Language-specific AST preprocessing (existing logic)
2. **Pass 2**: Convert AST to Schedule Tree (language-specific)
3. **Pass 3**: High-level optimizations on Schedule Tree (shared)
4. **Pass 4**: Convert Schedule Tree to SDFG (shared, implements #1466)

### Key Components
- **Schedule Tree IR**: Common intermediate representation for all frontends
- **Pass Pipeline Integration**: Uses existing `dace.transformation.pass_pipeline.py`
- **Shared Backend**: Single Schedule Tree → SDFG converter for all languages


## 1. Current Architecture Analysis

### 1.1 Current Frontend Architecture

DaCe currently has several language frontends:
- **Python Frontend**: Located in `dace/frontend/python/`
- **Fortran Frontend**: Located in `dace/frontend/fortran/`
- **TensorFlow/Octave frontends**: Additional specialized frontends

Each frontend implements direct AST-to-SDFG conversion with language-specific preprocessing.

### 1.2 Python Frontend Analysis

The Python frontend preprocessing pipeline in `preprocessing.py` includes:
1. **StructTransformer**: Handles struct initialization
2. **ModuleResolver**: Resolves imported modules
3. **MPIResolver**: Handles MPI-specific constructs
4. **ModuloConverter**: Converts modulo operations
5. **GlobalResolver**: Resolves global variables outside the function context, as well as performs some type inference
6. **Multiple passes loop**:
   - **LoopUnroller**: Unrolls loops where possible
   - **ExpressionInliner**: Inlines dace.inline() expressions
   - **ContextManagerInliner**: Handles context managers (`with` scopes)
   - **ConditionalCodeResolver**: Resolves compile-time conditionals
   - **DeadCodeEliminator**: Removes unreachable code

### 1.3 Fortran Frontend Analysis

The Fortran frontend in `fortran_parser.py`:
- **AST_translator**: Direct translation from Fortran AST to SDFG
- Methods like `call2sdfg()`, `binop2sdfg()`, `forstmt2sdfg()` directly generate SDFG constructs

### 1.4 Current Limitations

1. **Code Duplication**: Similar patterns (loops, conditionals, function calls) implemented separately in each frontend
2. **Maintainability**: Bug fixes and optimizations must be replicated across frontends
3. **Verification Difficulty**: Direct AST-to-SDFG makes intermediate verification challenging
4. **Limited Optimization**: No shared high-level optimization infrastructure
5. **Inconsistent Pipeline**: Python uses ad-hoc preprocessing; Fortran has no structured pipeline

## 2. Proposed Architecture

### 2.1 Architecture Overview

```
Language AST →
    [Pass 1: Preprocessing] →
    [Pass 2: AST→ScheduleTree] →
    [Pass 3: ScheduleTree Optimizations] →
    [Pass 4: ScheduleTree→SDFG] →
    SDFG
```

### 2.2 Multi-Pass Pipeline Design

#### Pass 1: Language-Specific AST Preprocessing
- **Purpose**: Language-specific analysis, syntactic sugar removal, global embedding
- **Input**: Language-specific AST
- **Output**: Normalized language AST
- **Examples**:
  - Python: Current preprocessing passes (LoopUnroller, ExpressionInliner, etc.)
  - Fortran: Symbol table resolution, array shape inference

#### Pass 2: AST → Schedule Tree Conversion
- **Purpose**: Convert normalized AST to Schedule Tree IR
- **Input**: Normalized language AST
- **Output**: Schedule Tree
- **Language-specific**: Each frontend implements its own converter

#### Pass 3: Schedule Tree Transformations (Optional)
- **Purpose**: High-level optimizations on Schedule Tree
- **Input**: Schedule Tree
- **Output**: Transformed Schedule Tree
- **Shared**: Common across all frontends
- **Examples**: Loop invariant code motion, dead code elimination, constant propagation

**Optimization Strategy**:
- **Source-level optimizations** stay at Schedule Tree level (constant folding, dead code elimination)
- **Execution-level optimizations** remain at SDFG level (memory optimization, parallelization)
- **Some optimizations may be duplicated** but with different focuses:
  - Schedule Tree: Structure-preserving, source-aware
  - SDFG: Performance-focused, execution-aware

#### Pass 4: Schedule Tree → SDFG Conversion
- **Purpose**: Generate final SDFG from Schedule Tree
- **Input**: Schedule Tree
- **Output**: SDFG
- **Shared**: Single implementation for all frontends (see #1466)

### 2.3 Schedule Tree as Intermediate Representation

#### 2.3.1 Schedule Tree vs CFG-based SDFG Architecture

DaCe has recently introduced a CFG-based SDFG architecture with `ControlFlowRegion`, `LoopRegion`, and other control flow constructs. This raises the question: why introduce Schedule Tree as an intermediate representation when SDFGs now have better control flow support?

While both approaches offer similar expressiveness for representing control flow, there are important distinctions in their design goals and practical benefits:

1. Since Control Flow Regions are at parity with Schedule Trees, the latter uses the same SDFG constructs in its representation.
2. Since Schedule Trees are simpler to represent (e.g., no need to construct chained views, memlet trees) and contain
   fewer components, they are more scalable and easier to construct, especially from imperative languages.
3. One can reschedule the schedule trees in a simpler manner over existing graphs, as reconnecting the statements
   together is not necessary.
4. Debugging imperative language frontends is simpler with a statement based tree representation.

#### 2.3.2 Schedule Tree Rationale

Schedule Tree serves as the common IR because:
- **Existing Infrastructure**: Already implemented in `dace/sdfg/analysis/schedule_tree/`
- **Appropriate Abstraction**: Higher level than SDFG, lower level than language AST
- **Control Flow Representation**: Natural representation of structured control flow
- **Visitor Pattern Support**: Built-in visitor/transformer patterns for manipulation
- **Complementary to SDFG with Control Flow Regions**: Provides source-level view while SDFG provides execution view

## 3. Schedule Tree Extensions

### 3.1 Current Schedule Tree Nodes Analysis

Existing nodes in `treenodes.py`:

**Control Flow Scopes:**
- `LoopScope`
- `IfScope`, `ElifScope`, `ElseScope`
- `GBlock` (general control flow block)

**Dataflow Scopes:**
- `MapScope`, `ConsumeScope`, `PipelineScope`

**Leaf Nodes:**
- `TaskletNode`, `LibraryCall`, `CopyNode`, `ViewNode`
- `AssignNode`, `GotoNode`, `BreakNode`, `ContinueNode`
- `StateLabel`, `RefSetNode`

## 4. Pipeline Definitions

### 4.1 Python Frontend Pipeline

```python
class PythonFrontendPipeline(Pipeline):
    def __init__(self):
        ast_preprocessing_passes = FixedPointPipeline([
            GlobalResolutionPass(),
            LoopUnrollingPass(),
            ExpressionInliningPass(),
            ContextManagerInliningPass(),
            ConditionalResolutionPass(),
            DeadCodeEliminationPass(),
        ])
        passes = [
            # Pass 1: AST Preprocessing
            StructTransformationPass(),
            ModuleResolutionPass(),
            MPIResolutionPass(),
            ModuloConversionPass(),
            ast_preprocessing_passes,

            # Pass 2: AST → Schedule Tree
            PythonASTToScheduleTreePass(),

            # Pass 3: Schedule Tree Optimizations (optional)
            ScheduleTreeOptimizationPass(),

            # Pass 4: Schedule Tree → SDFG
            ScheduleTreeToSDFGPass()
        ]
        super().__init__(passes)
```

### 4.2 Fortran Frontend Pipeline

```python
class FortranFrontendPipeline(Pipeline):
    def __init__(self):
        passes = [
            # Pass 1: AST Preprocessing
            FortranSymbolTablePass(),
            FortranArrayAnalysisPass(),
            FortranTypeResolutionPass(),

            # Pass 2: AST → Schedule Tree
            FortranASTToScheduleTreePass(),

            # Pass 3: Schedule Tree Optimizations (optional)
            ScheduleTreeOptimizationPass(),

            # Pass 4: Schedule Tree → SDFG
            ScheduleTreeToSDFGPass()
        ]
        super().__init__(passes)
```

## 5. Migration Strategy

### 5.1 Phased Migration Approach

#### Phase 1: Infrastructure Setup (1-2 weeks)
1. **Extend Schedule Tree**: Implement required node extensions
2. **Create Base Pass Classes**: Implement frontend pass interfaces
3. **Implement Schedule Tree → SDFG Pass**: Complete #1466 work

#### Phase 2: Python Frontend Migration (2-3 weeks)
1. **Convert Python Preprocessing**: Migrate existing transforms to Pass framework
2. **Implement Python AST → Schedule Tree**: Create converter for Python constructs
3. **Integration Testing**: Ensure feature parity with existing Python frontend

#### Phase 3: Fortran Frontend Migration (2-3 weeks)
1. **Create Fortran Preprocessing Passes**: Structure existing Fortran preprocessing
2. **Implement Fortran AST → Schedule Tree**: Create converter for Fortran constructs
3. **Integration Testing**: Ensure feature parity with existing Fortran frontend

#### Phase 4: Optimization and Cleanup (1-2 weeks)
1. **Implement Schedule Tree Optimizations**: Add high-level optimization passes
2. **Performance Testing**: Ensure no regression in compilation performance
3. **Documentation**: Update frontend documentation

### 5.2 Backward Compatibility

- **Maintain Existing APIs**: Keep current frontend entry points unchanged
- **Feature Flags**: Allow switching between old and new architecture during transition
- **Gradual Deprecation**: Mark old APIs as deprecated after new architecture is stable

### 5.3 Migration Testing Strategy

1. **Regression Test Suite**: Ensure all existing functionality works with new architecture
2. **Performance Benchmarks**: Compare compilation times between old and new architecture
3. **Correctness Validation**: Verify SDFG output is equivalent between architectures

## 6. Testing and Verification Strategy

### 6.1 Unit Testing Framework

Pytest.

### 6.2 Test Categories

#### 6.2.1 Pass-Level Tests
- **Individual Pass Testing**: Test each pass in isolation
- **Pass Composition Testing**: Test pass sequences

#### 6.2.2 Integration Tests
- **End-to-End Testing**: Full pipeline from source to SDFG
- **Multi-Language Testing**: Ensure consistent behavior across frontends
- **Regression Testing**: Verify existing functionality preserved (will happen with current CI)

#### 6.2.3 Performance Tests
- **Compilation Time**: Measure pipeline performance
- **Memory Usage**: Track memory consumption during compilation
- **Scalability**: Test with large programs

### 6.3 Verification Strategy

#### 6.3.1 Semantic Equivalence
- **SDFG Comparison**: Compare generated SDFGs with reference implementations
- **Execution Testing**: Run generated code and compare outputs
- **Property Preservation**: Verify semantic properties maintained

#### 6.3.2 Schedule Tree Validation
- **Well-Formedness**: Ensure Schedule Trees are structurally valid
- **Type Safety**: Verify type correctness at Schedule Tree level
- **Control Flow Validation**: Check control flow is properly represented

## 7. Optimization Opportunities

### 7.1 High-Level Optimizations at Schedule Tree Level

#### 7.1.1 Loop Optimizations
- **Loop Invariant Code Motion**: Move invariant computations outside loops
- **Loop Fusion**: Merge compatible adjacent loops
- **Loop Interchange**: Reorder nested loops for better locality

#### 7.1.2 Control Flow Optimizations
- **Dead Code Elimination**: Remove unreachable code blocks
- **Constant Propagation**: Propagate constants across control flow
- **Branch Elimination**: Remove branches with constant conditions

#### 7.1.3 Data Flow Optimizations
- **Operation Reordering**: Move operations around a scope to minimize the number of generates states in SDFG

## 8. Benefits Analysis

### 8.1 Code Reuse
- **Single Schedule Tree → SDFG converter**: ~3000 lines of shared code
- **Common optimization passes**: ~1000 lines of shared optimization logic
- **Reduced maintenance burden**: Bug fixes only need to be applied once

### 8.2 Verification Improvements
- **Schedule Tree validation**: Intermediate verification point
- **Type safety**: Earlier type checking
- **Semantic analysis**: Better error reporting

### 8.3 Extensibility
- **New language frontends**: Only need AST → Schedule Tree conversion
- **Custom optimizations**: Easy to add new optimization passes
- **Domain-specific extensions**: Schedule Tree can be extended for specific domains

### 8.4 Development Productivity
- **Cleaner separation of concerns**: Frontend vs. backend clearly separated
- **Easier debugging**: Intermediate representations aid debugging
- **Better testing**: Each stage can be tested independently

## 9. Optimization Pass Strategy

### 9.1 Schedule Tree vs SDFG Optimization Levels

#### 9.1.1 **Schedule Tree Level Optimizations:**
- **Source-preserving transformations**: Maintain traceability to original source
- **High-level semantic optimizations**: Constant propagation, dead code elimination
- **Control flow simplifications**: Loop fusion, condition simplification
- **Language-agnostic optimizations**: Shared across all frontends

#### 9.1.2 **SDFG Level Optimizations:**
- **Execution-focused transformations**: Memory layout optimization, parallelization
- **Hardware-specific optimizations**: GPU kernel fusion, vectorization
- **Performance-critical optimizations**: Memory access patterns, data movement

### 9.2 Optimization Pass Exclusivity

#### 9.2.1 **Non-exclusive optimizations** (may appear at both levels):
- **Constant propagation**: Schedule Tree (source-aware) vs SDFG (execution-aware)
- **Dead code elimination**: Schedule Tree (structural) vs SDFG (performance-focused)

#### 9.2.2 **Schedule Tree exclusive optimizations**:
- **Source-level transformations**: Preserving language semantics
- **Frontend-specific optimizations**: Language-specific idiom recognition

#### 9.2.3 **SDFG exclusive optimizations**:
- **Memory management**: Buffer allocation, data layout
- **Hardware mapping**: Device-specific optimizations
- **Execution scheduling**: Task and data parallelism

### 9.3 Maintainability Considerations
- **Documentation**: Need comprehensive documentation for new architecture
- **Testing**: Extensive test suite required
- **Code organization**: Clear organization of frontend passes

## 10. Conclusion

The proposed modular frontend architecture addresses current limitations of DaCe's frontend design by:

1. **Introducing Schedule Tree as common IR**: Provides shared intermediate representation
2. **Implementing multi-pass pipeline**: Structured, configurable transformation pipeline
3. **Enabling code reuse**: Single Schedule Tree → SDFG converter for all frontends
4. **Supporting optimization**: High-level optimizations at Schedule Tree level
5. **Improving verification**: Intermediate verification and validation points

The phased migration strategy ensures minimal disruption while providing clear benefits for maintainability, extensibility, and optimization opportunities.
