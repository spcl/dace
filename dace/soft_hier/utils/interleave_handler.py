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
        
    def get_tile_num(self):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        num_tiles = self.split_scheme[0] * self.split_scheme[1]
        return num_tiles

    def print_info(self):
        print("Array shape: ", self.array.shape)
        print("Block shape: ", self.block_shape)
        print("Cluster dimensions: ", self.cluster_dims)
        print("Cluster dimensions (DACE): ", self.cluster_dims_dace)
        print("Split scheme: ", self.split_scheme)
        if self.split_scheme is not None:
            print("Tiling shape: ", (self.array.shape[0] // self.split_scheme[0], self.array.shape[1] // self.split_scheme[1]))
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
        if split_x % dim_x != 0 or split_y % dim_y != 0:
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

    def rhs_split_K_place_to_left_and_bottom(self, k_group_dims:tuple):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        split_x = self.split_scheme[0]
        split_y = self.split_scheme[1]
        dim_x = self.cluster_dims[0]
        dim_y = self.cluster_dims[1]
        dim_x_dace = self.cluster_dims_dace[0]
        dim_y_dace = self.cluster_dims_dace[1]
        if len(k_group_dims) != 2:
            raise ValueError("K group dimensions must be a tuple of 2 elements")
        (k_dim_x, k_dim_y) = k_group_dims
        # self.print_info()
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

        self.placement_scheme = ()
        for (i, j) in [(i, j) for i in range(split_x) for j in range(split_y)]:
            k_group_index = j 
            k_group_offset = i
            kg_i = k_group_index % (dim_x_dace// k_dim_x)
            kg_j = k_group_index // (dim_x_dace// k_dim_x)
            kg_oi = k_group_offset % k_dim_x
            kg_oj = k_group_offset // k_dim_x
            pi_dace = kg_oi + kg_i * k_dim_x
            pj_dace = kg_oj + kg_j * k_dim_y
            pi_index = pj_dace * dim_x_dace + pi_dace
            pi_real = pi_index % dim_x
            pj_real = pi_index // dim_x
            place = real_place_base[pi_real][pj_real]
            if place == 1:
                channel_id = pj_real % dim_y
            else:
                channel_id = 2 * dim_y + dim_x + pi_real % dim_x
            # print(f"i: {i}, j: {j}, k_group_index: {k_group_index}, k_group_offset: {k_group_offset}, kg_i: {kg_i}, kg_j: {kg_j}, kg_oi: {kg_oi}, kg_oj: {kg_oj}, pi_dace: {pi_dace}, pj_dace: {pj_dace}, pi_index: {pi_index}, pi_real: {pi_real}, pj_real: {pj_real}, place: {place}")
            self.placement_scheme += (channel_id,)
        
    def result_split_K_place_to_left_and_bottom(self, k_group_dims:tuple):
        print(f"k_group_dims: {k_group_dims}")
        (kg_m, kg_n) = k_group_dims
        kg_num = kg_m * kg_n
        
        split_x = self.split_scheme[0]
        split_y = self.split_scheme[1]
        dim_x = self.cluster_dims[0]
        dim_y = self.cluster_dims[1]
        dim_x_dace = self.cluster_dims_dace[0]
        dim_y_dace = self.cluster_dims_dace[1]
        index_diff_list = []
        self.placement_scheme = ()
        for (i, j) in [(i, j) for i in range(split_x) for j in range(split_y)]:
            kg_i = i % (dim_x_dace//kg_m)
            kg_j = j % (dim_y_dace//kg_n)
            gi = kg_i * kg_m
            gj = kg_j * kg_n
            cid_store = ((gi+gj*dim_x_dace)//dim_x)%kg_num
            kg_oi = cid_store % kg_m
            kg_oj = cid_store // kg_m
            pi_dace = kg_oi + kg_i * kg_m
            pj_dace = kg_oj + kg_j * kg_n
            pi_index = pj_dace * dim_x_dace + pi_dace
            pi_real = pi_index % dim_x
            pj_real = pi_index // dim_x

            index_diff = (pi_real - pj_real + dim_y) % dim_y
            if index_diff not in index_diff_list:
                index_diff_list.append(index_diff)
            
            if index_diff >= dim_x//2:
                channel_id = pj_real % dim_y
            else:
                channel_id = 2 * dim_y + dim_x + pi_real % dim_x
            self.placement_scheme += (channel_id,)

            # print(f"kg_i: {kg_i}, kg_j: {kg_j}, gi: {gi}, gj: {gj}, cid_store: {cid_store}, kg_oi: {kg_oi}, kg_oj: {kg_oj}, pi_dace: {pi_dace}, pj_dace: {pj_dace}, pi_index: {pi_index}, pi_real: {pi_real}, pj_real: {pj_real}, index_diff: {index_diff}")    
        # self.print_info()
        # raise NotImplementedError("Not implemented yet")
        
    def my_sys_stream_place_to_left_and_bottom(self, systolic_range=None):
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
        
        sr_m, sr_n = systolic_range # sr_m = 1, y-axis stream; sr_n = 1, x-axis stream
        if sr_m > 1 and sr_n > 1:
            raise ValueError("stream can only be 1 in one dimension")

        # From the dim_x and dim_y, get the real place base
        real_place_base = [[0 for _ in range(dim_y)] for _ in range(dim_x)]
        for i in range(dim_x):
            for j in range(dim_y):
                if ((i // (dim_x//2)) + (j // (dim_y//2))) % 2 == 0:
                    real_place_base[i][j] = 1
                else:
                    real_place_base[i][j] = 0

        for (i, j) in [(i, j) for i in range(split_x) for j in range(split_y)]:
                pi_dace = i % dim_x
                pj_dace = j % dim_y

                p_real = pj_dace * dim_x_dace + pi_dace
                pi_real = p_real % dim_x
                pj_real = p_real // dim_x

                place = real_place_base[pi_real][pj_real]
                if sr_m == 1:
                    if place == 1:
                        oi = (pi_real % (dim_x//2)) // ((dim_x//2)//sr_n)
                        channel_id = (pj_real // sr_n) * sr_n + oi
                    else:
                        channel_id = 2 * dim_y + dim_x + pi_real % dim_x
                    self.placement_scheme += (channel_id,)
                elif sr_n == 1:
                    if place == 1:
                        channel_id = pj_real % dim_y
                    else:
                        oj = (pj_real % (dim_y//2)) // ((dim_y//2)//sr_m)
                        channel_id = 2 * dim_y + dim_x + (pi_real // sr_m) * sr_m + oj
                    self.placement_scheme += (channel_id,)
                    
    def multistream_place_to_left_and_bottom(self, n_streams:int, direction='x'):
        if self.split_scheme is None:
            raise ValueError("Split scheme is not set")
        num_tiles = self.split_scheme[0] * self.split_scheme[1]
        self.placement_scheme = ()
        dim_x = self.cluster_dims[0]
        dim_y = self.cluster_dims[1]
        split_x = self.split_scheme[0]
        split_y = self.split_scheme[1]
        (dim_x_dace, dim_y_dace) = self.cluster_dims_dace
        if split_x % dim_x != 0 or split_y % dim_y!= 0:
            print(split_x, split_y, dim_x, dim_y)
            raise ValueError(f"Split scheme must be multiples of cluster dimensions")
        
        if direction == 'x':
            summa_range = (dim_x // n_streams, dim_y)
        elif direction == 'y':
            summa_range = (dim_x, dim_y // n_streams)
        
        sr_m, sr_n = summa_range 

        # From the dim_x and dim_y, get the real place base
        real_place_base = [[0 for _ in range(dim_y)] for _ in range(dim_x)]
        for i in range(dim_x):
            for j in range(dim_y):
                if direction == 'x':
                    if (i % sr_m) // (sr_m // 2) == 0:
                        real_place_base[i][j] = 1
                    else:
                        real_place_base[i][j] = 0
                elif direction == 'y':
                    if (j % sr_n) // (sr_n // 2) == 0:
                        real_place_base[i][j] = 0
                    else:
                        real_place_base[i][j] = 1
        for (i, j) in [(i, j) for i in range(split_x) for j in range(split_y)]:
            pi_dace = i % dim_x
            pj_dace = j % dim_y

            p_real = pj_dace * dim_x_dace + pi_dace
            pi_real = p_real % dim_x
            pj_real = p_real // dim_x

            place = real_place_base[pi_real][pj_real]
            if direction == 'x':
                if place == 1:
                    channel_id = pj_real % dim_y
                else:
                    oj = pi_real % (sr_m // 2) + (pj_real // (sr_m // 2)) * (sr_m // 2)
                    channel_id = 2 * dim_y + dim_x + oj
                self.placement_scheme += (channel_id,)
            elif direction == 'y':
                if place == 1:
                    oi = (pj_real % (sr_n // 2)) + (pi_real // (sr_n // 2)) * (sr_n // 2)
                    channel_id = oi
                else:
                    channel_id = 2 * dim_y + dim_x + pi_real % dim_x
                self.placement_scheme += (channel_id,)