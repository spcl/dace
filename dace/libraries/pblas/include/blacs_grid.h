#ifndef DACE_BLACS_GRID_H
#define DACE_BLACS_GRID_H

#include <vector>

// A BLACS grid built inside the dataflow, kept so the finalize code can free it.
template <typename T> struct DaceBlacsGrid {
  int fcomm;
  T prows;
  T pcols;
  T context;
};

// Look up the context of an already built grid. The key is the communicator's
// Fortran handle plus the grid shape, so a program that uses several grids
// builds each of them exactly once.
template <typename T>
inline bool dace_blacs_grid_find(const std::vector<DaceBlacsGrid<T>> &grids,
                                 int fcomm, T prows, T pcols, T *context) {
  for (const auto &grid : grids) {
    if (grid.fcomm == fcomm && grid.prows == prows && grid.pcols == pcols) {
      *context = grid.context;
      return true;
    }
  }
  return false;
}

#endif
