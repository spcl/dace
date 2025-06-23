# Design Document: Modular Frontend Architecture

## Executive Summary

This document outlines a design for refactoring DaCe's frontend architecture to be more portable, verifiable, and modular. The proposed architecture introduces Schedule Tree as a common intermediate representation (IR) shared across language frontends, structured as a multi-pass pipeline using DaCe's existing Pass Pipeline infrastructure.

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
5. **GlobalResolver**: Resolves global variables outside the function context
6. **Multiple passes loop**:
   - **LoopUnroller**: Unrolls loops where possible
   - **ExpressionInliner**: Inlines dace.inline() expressions
   - **ContextManagerInliner**: Handles context managers
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
Language AST → [Pass 1: Preprocessing] → [Pass 2: AST→ScheduleTree] → [Pass 3: ScheduleTree Opts] → [Pass 4: ScheduleTree→SDFG] → SDFG
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

#### Pass 4: Schedule Tree → SDFG Conversion
- **Purpose**: Generate final SDFG from Schedule Tree
- **Input**: Schedule Tree
- **Output**: SDFG
- **Shared**: Single implementation for all frontends (see #1466)

### 2.3 Schedule Tree as Intermediate Representation

Schedule Tree serves as the common IR because:
- **Existing Infrastructure**: Already implemented in `dace/sdfg/analysis/schedule_tree/`
- **Appropriate Abstraction**: Higher level than SDFG, lower level than language AST
- **Control Flow Representation**: Natural representation of structured control flow
- **Visitor Pattern Support**: Built-in visitor/transformer patterns for manipulation

## 3. Schedule Tree Extensions

### 3.1 Current Schedule Tree Nodes Analysis

Existing nodes in `treenodes.py`:

**Control Flow Scopes:**
- `ForScope`, `WhileScope`, `DoWhileScope`, `GeneralLoopScope`
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

### 4.3 Pass Interface Definitions

#### 4.3.1 Base Frontend Pass
```python
class FrontendPass(Pass):
    """Base class for frontend passes."""

    def modifies(self) -> Modifies:
        return Modifies.Everything  # Conservative default

    def should_reapply(self, modified: Modifies) -> bool:
        return True  # Conservative default
```

#### 4.3.2 AST Preprocessing Pass
```python
class ASTPreprocessingPass(FrontendPass):
    """Base class for language-specific AST preprocessing."""

    def apply_pass(self, ast_data: Any, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        raise NotImplementedError
```

#### 4.3.3 AST to Schedule Tree Pass
```python
class ASTToScheduleTreePass(FrontendPass):
    """Base class for AST → Schedule Tree conversion."""

    def apply_pass(self, ast_data: Any, pipeline_results: Dict[str, Any]) -> Optional[ScheduleTreeScope]:
        raise NotImplementedError
```

#### 4.3.4 Schedule Tree Optimization Pass
```python
class ScheduleTreeOptimizationPass(FrontendPass):
    """Base class for Schedule Tree optimizations."""

    def apply_pass(self, schedule_tree: ScheduleTreeScope, pipeline_results: Dict[str, Any]) -> Optional[ScheduleTreeScope]:
        raise NotImplementedError
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

## 6. Common Pattern Examples

### 6.1 For Loop Pattern

#### Python Source:
```python
for i in range(10):
    A[i] = B[i] + C[i]
```

#### Schedule Tree Representation:
```python
ForScope(
    header=ForScopeHeader(itervar='i', init=0, condition='i < 10', update='i++'),
    children=[
        TaskletNode(
            node=...,
            in_memlets={'B_in': Memlet('B[i]'), 'C_in': Memlet('C[i]')},
            out_memlets={'A_out': Memlet('A[i]')}
        )
    ]
)
```

### 6.2 Conditional Pattern

#### Python Source:
```python
if condition:
    result = A + B
else:
    result = A - B
```

#### Schedule Tree Representation:
```python
IfScope(
    condition=CodeBlock('condition'),
    children=[
        TaskletNode(..., code='result = A + B')
    ]
),
ElseScope(
    children=[
        TaskletNode(..., code='result = A - B')
    ]
)
```

### 6.3 Function Call Pattern

#### Python Source:
```python
result = numpy.matmul(A, B)
```

#### Schedule Tree Representation:
```python
LibraryCall(
    node=MatMul(),
    in_memlets={'A', 'B'},
    out_memlets={'result'}
)
```

## 7. Testing and Verification Strategy

### 7.1 Unit Testing Framework

Pytest.

### 7.2 Test Categories

#### 7.2.1 Pass-Level Tests
- **Individual Pass Testing**: Test each pass in isolation
- **Pass Composition Testing**: Test pass sequences
- **Pass Dependency Testing**: Verify dependency resolution

#### 7.2.2 Integration Tests
- **End-to-End Testing**: Full pipeline from source to SDFG
- **Multi-Language Testing**: Ensure consistent behavior across frontends
- **Regression Testing**: Verify existing functionality preserved

#### 7.2.3 Performance Tests
- **Compilation Time**: Measure pipeline performance
- **Memory Usage**: Track memory consumption during compilation
- **Scalability**: Test with large programs

### 7.3 Verification Strategy

#### 7.3.1 Semantic Equivalence
- **SDFG Comparison**: Compare generated SDFGs with reference implementations
- **Execution Testing**: Run generated code and compare outputs
- **Property Preservation**: Verify semantic properties maintained

#### 7.3.2 Schedule Tree Validation
- **Well-Formedness**: Ensure Schedule Trees are structurally valid
- **Type Safety**: Verify type correctness at Schedule Tree level
- **Control Flow Validation**: Check control flow is properly represented

## 8. Optimization Opportunities

### 8.1 High-Level Optimizations at Schedule Tree Level

#### 8.1.1 Loop Optimizations
- **Loop Invariant Code Motion**: Move invariant computations outside loops
- **Loop Fusion**: Merge compatible adjacent loops
- **Loop Interchange**: Reorder nested loops for better locality

#### 8.1.2 Control Flow Optimizations
- **Dead Code Elimination**: Remove unreachable code blocks
- **Constant Propagation**: Propagate constants across control flow
- **Branch Elimination**: Remove branches with constant conditions

#### 8.1.3 Data Flow Optimizations
- **Copy Elimination**: Remove unnecessary data copies
- **View Optimization**: Optimize data view operations
- **Memory Layout Optimization**: Optimize data layout for access patterns

### 8.2 Optimization Pass Examples

```python
class LoopInvariantCodeMotionPass(ScheduleTreeOptimizationPass):
    """Move loop-invariant computations outside loops."""

    def apply_pass(self, schedule_tree: ScheduleTreeScope, pipeline_results: Dict[str, Any]) -> Optional[ScheduleTreeScope]:
        # Implementation details...
        pass

class ConstantPropagationPass(ScheduleTreeOptimizationPass):
    """Propagate constants through Schedule Tree."""

    def apply_pass(self, schedule_tree: ScheduleTreeScope, pipeline_results: Dict[str, Any]) -> Optional[ScheduleTreeScope]:
        # Implementation details...
        pass
```

## 9. Benefits Analysis

### 9.1 Code Reuse
- **Single Schedule Tree → SDFG converter**: ~3000 lines of shared code
- **Common optimization passes**: ~1000 lines of shared optimization logic
- **Reduced maintenance burden**: Bug fixes only need to be applied once

### 9.2 Verification Improvements
- **Schedule Tree validation**: Intermediate verification point
- **Type safety**: Earlier type checking
- **Semantic analysis**: Better error reporting

### 9.3 Extensibility
- **New language frontends**: Only need AST → Schedule Tree conversion
- **Custom optimizations**: Easy to add new optimization passes
- **Domain-specific extensions**: Schedule Tree can be extended for specific domains

### 9.4 Development Productivity
- **Cleaner separation of concerns**: Frontend vs. backend clearly separated
- **Easier debugging**: Intermediate representations aid debugging
- **Better testing**: Each stage can be tested independently

## 10. Implementation Considerations

### 10.1 Performance Considerations
- **Pass overhead**: Need to balance pass granularity with overhead
- **Memory usage**: Schedule Tree adds intermediate representation overhead
- **Compilation time**: Pipeline may increase compilation time initially

### 10.2 Compatibility Considerations
- **Existing code**: Must maintain compatibility with existing DaCe programs
- **API stability**: Frontend APIs should remain stable
- **Migration path**: Clear path for migrating existing code

### 10.3 Maintainability Considerations
- **Documentation**: Need comprehensive documentation for new architecture
- **Testing**: Extensive test suite required
- **Code organization**: Clear organization of frontend passes

## 11. Conclusion

The proposed modular frontend architecture addresses current limitations of DaCe's frontend design by:

1. **Introducing Schedule Tree as common IR**: Provides shared intermediate representation
2. **Implementing multi-pass pipeline**: Structured, configurable transformation pipeline
3. **Enabling code reuse**: Single Schedule Tree → SDFG converter for all frontends
4. **Supporting optimization**: High-level optimizations at Schedule Tree level
5. **Improving verification**: Intermediate verification and validation points

The phased migration strategy ensures minimal disruption while providing clear benefits for maintainability, extensibility, and optimization opportunities.

## 12. Next Steps

1. **Community Review**: Gather feedback on proposed architecture
2. **Prototype Implementation**: Implement core infrastructure
3. **Pilot Migration**: Start with Python frontend migration
4. **Performance Validation**: Ensure no significant performance regression
5. **Full Implementation**: Complete migration of all frontends

---

*This design document serves as a foundation for implementing the modular frontend architecture in DaCe. Implementation details may be refined based on community feedback and prototyping results.*
