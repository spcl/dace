# Modular Frontend Architecture - Executive Summary

## Overview
This document proposes a new modular architecture for DaCe frontends using Schedule Tree as an intermediate representation with a structured multi-pass pipeline.

## Current Problems
- **Code Duplication**: Each frontend reimplements similar AST-to-SDFG conversion logic
- **Maintainability**: Bug fixes must be replicated across all frontends
- **Limited Optimization**: No shared high-level optimization infrastructure
- **Verification Difficulty**: Direct AST-to-SDFG conversion is hard to verify

## Proposed Solution

### Four-Pass Pipeline Architecture
```
Language AST → [Pass 1: Preprocessing] → [Pass 2: AST→ScheduleTree] → [Pass 3: ScheduleTree Opts] → [Pass 4: ScheduleTree→SDFG] → SDFG
```

1. **Pass 1**: Language-specific AST preprocessing (existing logic)
2. **Pass 2**: Convert AST to Schedule Tree (language-specific)
3. **Pass 3**: High-level optimizations on Schedule Tree (shared)
4. **Pass 4**: Convert Schedule Tree to SDFG (shared, implements #1466)

### Key Components
- **Schedule Tree IR**: Common intermediate representation for all frontends
- **Pass Pipeline Integration**: Uses existing `dace.transformation.pass_pipeline.py`
- **Shared Backend**: Single Schedule Tree → SDFG converter for all languages

## Benefits
- **Code Reuse**: ~3000+ lines of shared Schedule Tree → SDFG conversion
- **Easier Verification**: Schedule Tree provides intermediate validation point
- **Better Optimization**: High-level optimizations at Schedule Tree level
- **Extensibility**: New frontends only need AST → Schedule Tree conversion
- **Cleaner Architecture**: Clear separation between frontend parsing and SDFG generation

## Migration Strategy
1. **Phase 1**: Extend Schedule Tree and implement base infrastructure
2. **Phase 2**: Migrate Python frontend to new architecture
3. **Phase 3**: Migrate Fortran frontend
4. **Phase 4**: Add optimizations and cleanup

## Implementation Timeline
- **Infrastructure Setup**: 1-2 weeks
- **Python Migration**: 2-3 weeks
- **Fortran Migration**: 2-3 weeks
- **Optimization & Cleanup**: 1-2 weeks

**Total**: ~6-10 weeks for complete implementation

---

For complete details, see the full [Design Document](modular-frontend-architecture.md).
