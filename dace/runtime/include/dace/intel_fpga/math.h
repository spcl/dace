// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
/**
    This file contains a set of preprocessor macros useful
    for simple arithmetic
*/

#pragma once

#define int_ceil(N,D) ((int)((N+D-1)/D))
#define int_floor(N,D) (1+(int)((N-1)/D))
#define Min min
#define Max max
#define Abs abs
