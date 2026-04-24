# `hlfir-inline-all` segfault — root cause & fix

## Summary

**Segfault is fixed.** Root cause: our bridge's `MLIRContext` had no
`DialectInlinerInterface` attached to the `fir`, `func`, or `LLVM`
dialects. When `mlir::inlineCall` dispatched
`InlinerInterface::handleArgument` into the per-dialect collection it
got `nullptr` and crashed.

## Crash site

Stack trace from `gdb python3 /tmp/gdb_repro.py` on a clean Release build:

```
Thread 1 "python3" received signal SIGSEGV, Segmentation fault.
0x00007ffff5ecca59 in mlir::InlinerInterface::handleArgument(
    mlir::OpBuilder&, mlir::Operation*, mlir::Operation*,
    mlir::Value, mlir::DictionaryAttr) const ()
  from /usr/lib/llvm-21/lib/libMLIR.so.21.1
#1 in ?? ()
#2 in ?? ()
#3 in mlir::inlineCall(InlinerInterface&, ..., CallOpInterface,
       CallableOpInterface, Region*, bool) ()
#4 in hlfir_bridge InlineAllPass::sweep()
```

The crash is inside `libMLIR.so` at `handleArgument`, reached from
`mlir::inlineCall`. `handleArgument` in `InlinerInterface` looks up
the per-dialect `DialectInlinerInterface` for the op's dialect; if
nothing is registered, it dereferences a null interface pointer.

## Root cause

Our bridge registers dialects but never attaches their inliner
extensions. Flang's canonical setup
(`/usr/lib/llvm-21/include/flang/Optimizer/Support/InitFIR.h`) makes
three additional calls after the `DialectRegistry::insert`:

```cpp
mlir::func::registerInlinerExtension(registry);   // func dialect
mlir::LLVM::registerInlinerInterface(registry);   // LLVM dialect
fir::addFIRInlinerExtension(registry);            // FIR dialect
```

Without these, `mlir::inlineCall` on any `fir.call` → `func.func` pair
has no dialect interface to dispatch to, and the base
`InlinerInterface::handleArgument` fault.

## Fix

`dace/frontend/hlfir/bridge/bridge.cpp`, `HLFIRModule::HLFIRModule()`:

```cpp
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
// flang/Optimizer/Dialect/FIRDialect.h already brings addFIRInlinerExtension.
...
mlir::func::registerInlinerExtension(registry_);
mlir::LLVM::registerInlinerInterface(registry_);
fir::addFIRInlinerExtension(registry_);
```

With those three lines added, `m.run_passes('hlfir-inline-all')` no
longer segfaults on `/tmp/intersub.hlfir`.

## Remaining follow-up — NOT fixed on this branch

With the inliner extensions attached, the pass now returns a verifier
error rather than crashing:

```
loc("/tmp/intersub.hlfir":13:3): error: 'func.func' op entry block must
have 1 arguments to match function signature
```

That is a **separate bug** in our `InlineAll.cpp` cloneCallback /
erase-call sequence — after `mlir::inlineCall` completes, the outer
function's entry block arg count no longer matches its signature.
Hypotheses to check next (out of scope for this 90-min investigation):

1. Our cloneCallback uses `src->cloneInto(inlineBlock->getParent(),
   inlineBlock->getIterator(), mapper)` — upstream MLIR's
   `InlinerPass` clones into the split-after block, not the block the
   call was in. We may be inserting the callee body into the caller's
   entry block and disturbing its args.
2. We explicitly call `call->erase()` after `inlineCall` succeeds —
   but depending on the single-block-optimisation fast path,
   `inlineCall` may already have removed the op or left it in a
   block that's been spliced.
3. Our `AggressiveInlinerInterface` returns `true` from all three
   `isLegalToInline` overloads. Maybe FIR's interface would have
   vetoed this specific inline (e.g. `fir.dummy_scope` at callee
   entry) and upstream respects that veto — we paper over it and
   produce an ill-formed result.

Recommendation: replace our hand-rolled pass with a thin wrapper over
`mlir::createInlinerPass()` after marking all non-entry functions
private (like our `set_entry_symbol` + `symbol-dce` combo does). The
stock pass handles the verifier invariants correctly.

## Reproducer (for the record)

```fortran
! /tmp/intersub.f90
subroutine inner(d)
  real(8), intent(inout) :: d(4)
  d(2) = 4.2d0
end subroutine
subroutine outer(d)
  real(8), intent(inout) :: d(4)
  d(2) = 5.5d0
  call inner(d)
end subroutine
```

```sh
flang-new-21 -fc1 -emit-hlfir /tmp/intersub.f90 -o /tmp/intersub.hlfir
python3 -c "
import sys
sys.path.insert(0, '/home/primrose/Work/d2/dace/frontend/hlfir')
from build_bridge import hb
m = hb.HLFIRModule()
m.parse_file('/tmp/intersub.hlfir')
m.run_passes('hlfir-inline-all')
"
```

Before the fix: SIGSEGV.
After the fix: verifier error (see follow-up above).

## Test status

`python3 -m pytest tests/hlfir/ -q` still at **87 passed + 3 xfailed**
on this branch. Multi-file driver (which explicitly *avoids* inline-all
via `MULTI_FILE_PIPELINE`) is unaffected.
