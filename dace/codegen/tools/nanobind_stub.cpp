// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Placeholder source for nanobind's helper-library target (named by nanobind after the
// options in ../CMakeLists.txt, and discovered there rather than assumed) when a
// machine-cached copy of its archive is linked instead of compiling nanobind's translation
// units
// (see DACE_NANOBIND_STATIC_LIB in ../CMakeLists.txt). Deliberately a file shipped with
// DaCe rather than one generated at configure time: recorded builds (command_db) replay
// into fresh build folders without running CMake, so every compiled source must exist
// at a path that does not depend on the configure step having run.
