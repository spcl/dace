import numpy as np
from math import sqrt

class InterleaveHandler:
    array = None
    block_shape:tuple = None
    cluster_dims:tuple = None
    cluster_dims_dace:tuple = None
    split_scheme:tuple = None
    placement_scheme:tuple = None
    def __init__(self, array:np.array, cluster_dims:tuple, block_shape:tuple):
        self.array = array
        self.block_shape = block_shape
        self.cluster_dims_dace = cluster_dims
        (dim_x, dim_y) = cluster_dims
        num_clusters = dim_x * dim_y
        self.cluster_dims = (int(sqrt(num_clusters)), int(sqrt(num_clusters)))
        
    def print_info(self):
        print("Array shape: ", self.array.shape)
        print("Block shape: ", self.block_shape)
        print("Cluster dimensions: ", self.cluster_dims)
        print("Split scheme: ", self.split_scheme)
        print("Placement scheme: ", self.placement_scheme)
        
    def split_horizental(self):
        self.split_scheme = (self.array.shape[0] // self.block_shape[0], 1)
    
    def split_vertical(self):
        self.split_scheme = (1, self.array.shape[1] // self.block_shape[1])
        
    def split_to_blocks(self, tile_dims=None):
        if tile_dims is None:
            self.split_scheme = (self.array.shape[0] // self.block_shape[0], self.array.shape[1] // self.block_shape[1])
        else:
            self.split_scheme = (self.array.shape[0] // tile_dims[0], self.array.shape[1] // tile_dims[1])
    
    def place_to_range(self, place_range:tuple):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        if len(place_range) != 3:
            raise ValueError("Range must be a tuple of 3 elements")
        (start, end, step) = place_range
        num_channels = (end - start + 1) // step
        total_channels = int(sqrt(self.cluster_dims[0] * self.cluster_dims[1])) * 4
        num_tiles = self.split_scheme[0] * self.split_scheme[1]
        if num_tiles % num_channels != 0:
            raise ValueError(f"Number of channels must be a multiple of number of tiles, tiles={num_tiles}, num_channels={num_channels}")
        self.placement_scheme = ()
        for i in range(num_tiles):
            channel_id = (start + (i * step) % num_channels + total_channels) % total_channels
            self.placement_scheme += (channel_id,)

    def systolic_place_to_left_and_bottom(self):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        num_tiles = self.split_scheme[0] * self.split_scheme[1]
        self.placement_scheme = ()
        dim_x = self.cluster_dims[0]
        dim_y = self.cluster_dims[1]
        if dim_x % 4 != 0 or dim_y % 4 != 0:
            raise ValueError("Cluster dimensions must be multiples of 4")
        split_x = self.split_scheme[0]
        split_y = self.split_scheme[1]
        # print(split_x, split_y)
        (dim_x_dace, dim_y_dace) = self.cluster_dims_dace
        if split_x % dim_x_dace != 0 or split_y % dim_y_dace != 0:
            print(split_x, split_y, dim_x_dace, dim_y_dace)
            raise ValueError("Split scheme must be multiples of cluster dimensions")
        place_base = [
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0]
        ]

        # From the dim_x and dim_y, get the real place base
        real_place_base = [[0 for _ in range(dim_y)] for _ in range(dim_x)]
        for i in range(dim_x):
            for j in range(dim_y):
                if (i+j) % 2 == 0:
                    real_place_base[i][j] = place_base[i % 4][j % 4]
                else:
                    real_place_base[i][j] = place_base[i % 4][j % 4]
        for (i, j) in [(i, j) for i in range(split_x) for j in range(split_y)]:
            pi_dace = i % dim_x_dace
            pj_dace = j % dim_y_dace

            p_real = pi_dace * dim_y_dace + pj_dace
            pi_real = p_real // dim_y
            pj_real = p_real % dim_y

            place = real_place_base[pi_real][pj_real]
            if place == 1:
                channel_id = pj_real % dim_y
            else:
                channel_id = 2 * dim_y + dim_x + pi_real % dim_x
            self.placement_scheme += (channel_id,)
            

    def summa_place_to_left_and_bottom(self):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        num_tiles = self.split_scheme[0] * self.split_scheme[1]
        self.placement_scheme = ()
        dim_x = self.cluster_dims[0]
        dim_y = self.cluster_dims[1]
        if dim_x % 4 != 0 or dim_y % 4 != 0:
            raise ValueError("Cluster dimensions must be multiples of 4")
        split_x = self.split_scheme[0]
        split_y = self.split_scheme[1]
        # print(split_x, split_y)
        (dim_x_dace, dim_y_dace) = self.cluster_dims_dace
        if split_x % dim_x != 0 or split_y % dim_y!= 0:
            print(split_x, split_y, dim_x, dim_y)
            raise ValueError(f"Split scheme must be multiples of cluster dimensions")
        place_base = [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ]

        # From the dim_x and dim_y, get the real place base
        real_place_base = [[0 for _ in range(dim_y)] for _ in range(dim_x)]
        for i in range(dim_x):
            for j in range(dim_y):
                if (i+j) % 2 == 0:
                    real_place_base[i][j] = place_base[i % 4][j % 4]
                else:
                    real_place_base[i][j] = place_base[i % 4][j % 4]
        for (i, j) in [(i, j) for i in range(split_x) for j in range(split_y)]:
            pi_dace = i % dim_x
            pj_dace = j % dim_y

            p_real = pi_dace * dim_y + pj_dace
            pi_real = p_real // dim_y
            pj_real = p_real % dim_y

            place = real_place_base[pi_real][pj_real]
            if place == 1:
                channel_id = pj_real % dim_y
            else:
                channel_id = 2 * dim_y + dim_x + pi_real % dim_x
            self.placement_scheme += (channel_id,)
# [1, 1, 1, 0][1, 1, 1, 0],
# [0, 0, 0, 0][0, 0, 0, 0],
# [1, 1, 1, 1][1, 1, 1, 1],
# [1, 0, 0, 0][1, 0, 0, 0]
# [1, 1, 1, 0][1, 1, 1, 0],
# [0, 0, 0, 0][0, 0, 0, 0],
# [1, 1, 1, 1][1, 1, 1, 1],
# [1, 0, 0, 0][1, 0, 0, 0]