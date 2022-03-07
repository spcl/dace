// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_COMM_H
#define __DACE_COMM_H

namespace dace {

    namespace comm {

        int cart_rank(int grid_length, const int* grid, const int* coords) {
            int rank = coords[0];
            for (auto i = 1; i < grid_length; ++i) {
                rank *= grid[i];
                rank += coords[i];
            }
            return rank;
        }

        void cart_coords(int rank, int grid_length, const int* grid, int* coords) {
            int tmp = rank;
            for (auto i = grid_length - 1; i > 0; --i) {
                coords[i] = tmp % grid[i];
                tmp /= grid[i];
            }
            coords[0] = tmp;
        }

    } // namespace comm

} // namespace dace

#endif  // __DACE_COMM_H
