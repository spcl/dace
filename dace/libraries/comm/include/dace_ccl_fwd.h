// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

// The state struct of a program holds a pointer to its NCCL context. Declaring the class suffices for
// the translation units that only see the struct (the GPU code), which then need neither NCCL nor MPI.
namespace dace {

namespace comm {

class NcclContext;

}  // namespace comm

}  // namespace dace
