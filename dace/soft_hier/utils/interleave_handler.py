import numpy as np

class InterleaveHandler:
    array = None
    block_shape:tuple = None
    cluster_dims:tuple = None
    split_scheme:tuple = None
    placement_scheme:tuple = None
    def __init__(self, array:np.array, cluster_dims:tuple, block_shape:tuple):
        self.array = array
        self.block_shape = block_shape
        self.cluster_dims = cluster_dims
        
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
        
    def split_to_blocks(self):
        self.split_scheme = (self.array.shape[0] // self.block_shape[0], self.array.shape[1] // self.block_shape[1])
    
    def place_to_range(self, place_range:tuple):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        if len(place_range) != 3:
            raise ValueError("Range must be a tuple of 3 elements")
        (start, end, step) = place_range
        num_channels = (end - start + 1) // step
        total_channels = (self.cluster_dims[0] + self.cluster_dims[1]) * 2
        num_tiles = self.split_scheme[0] * self.split_scheme[1]
        if num_tiles % num_channels != 0:
            raise ValueError("Number of chunks must be a multiple of number of tiles")
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
        if split_x % dim_x != 0 or split_y % dim_y != 0:
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
            place = real_place_base[i % dim_x][j % dim_y]
            if place == 1:
                channel_id = j % dim_y
            else:
                channel_id = 2 * dim_y + dim_x + i % dim_x
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
        if split_x % dim_x != 0 or split_y % dim_y != 0:
            raise ValueError("Split scheme must be multiples of cluster dimensions")
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
            place = real_place_base[i % dim_x][j % dim_y]
            if place == 1:
                channel_id = j % dim_y
            else:
                channel_id = 2 * dim_y + dim_x + i % dim_x
            self.placement_scheme += (channel_id,)
# [1, 1, 1, 0][1, 1, 1, 0],
# [0, 0, 0, 0][0, 0, 0, 0],
# [1, 1, 1, 1][1, 1, 1, 1],
# [1, 0, 0, 0][1, 0, 0, 0]
# [1, 1, 1, 0][1, 1, 1, 0],
# [0, 0, 0, 0][0, 0, 0, 0],
# [1, 1, 1, 1][1, 1, 1, 1],
# [1, 0, 0, 0][1, 0, 0, 0]